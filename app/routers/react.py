from __future__ import annotations

import json
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.deps import get_db_connection, get_neo4j_session, get_openai_client
from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.react.agent import AgentOutcome, ReactAgent, ReactStep
from app.react.state import ReactSessionState
from app.react.tools import ToolContext
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


router = APIRouter(prefix="/react", tags=["ReAct"])


def _to_cdata(value: str) -> str:
    return f"<![CDATA[{value}]]>"


def _new_react_run_id() -> str:
    return f"react_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


class SQLCompletenessModel(BaseModel):
    is_complete: bool
    missing_info: str
    confidence_level: str


class ToolCallModel(BaseModel):
    name: str
    raw_parameters_xml: str
    parameters: Dict[str, Any]


class ReactStepModel(BaseModel):
    iteration: int
    reasoning: str
    metadata_xml: str
    partial_sql: str
    sql_completeness: SQLCompletenessModel
    tool_call: ToolCallModel
    tool_result: Optional[str] = None
    llm_output: str


class ExecutionResultModel(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
    row_count: int
    execution_time_ms: float


class ReactResponse(BaseModel):
    status: Literal["completed", "needs_user_input", "await_step_confirmation"]
    final_sql: Optional[str] = None
    validated_sql: Optional[str] = None
    execution_result: Optional[ExecutionResultModel] = None
    steps: List[ReactStepModel] = Field(default_factory=list)
    collected_metadata: str
    partial_sql: str
    remaining_tool_calls: int
    session_state: Optional[str] = None
    question_to_user: Optional[str] = None
    warnings: Optional[List[str]] = None


class ReactRequest(BaseModel):
    question: str = Field(..., description="사용자 자연어 질문")
    dbms: Optional[str] = Field(default=None, description="DBMS 타입 (기본값: 설정값)")
    max_tool_calls: int = Field(default=30, ge=1, le=100)
    execute_final_sql: bool = Field(default=True, description="최종 SQL을 실제로 실행할지 여부")
    max_iterations: Optional[int] = Field(default=None, ge=1, le=20)
    session_state: Optional[str] = Field(default=None, description="이전 세션 상태 토큰")
    user_response: Optional[str] = Field(default=None, description="ask_user 툴에 대한 사용자 응답")
    step_confirmation_mode: Optional[bool] = Field(
        default=None, description="각 스텝마다 사용자 확인을 요구하는 interrupt 모드"
    )
    step_confirmation_response: Optional[Literal["continue"]] = Field(
        default=None, description="step confirmation 상태에서 재개 응답"
    )
    max_sql_seconds: int = Field(
        default=60, ge=1, le=3600, description="SQL 실행 최대 허용 시간(초). 기본값: 60초"
    )
    prefer_language: str = Field(
        default="ko", description="사용자 선호 언어 코드(ko, en, ja, zh 등). 기본값: ko"
    )


def _step_to_model(step: ReactStep) -> ReactStepModel:
    return ReactStepModel(
        iteration=step.iteration,
        reasoning=step.reasoning,
        metadata_xml=step.metadata_xml,
        partial_sql=step.partial_sql,
        sql_completeness=SQLCompletenessModel(
            is_complete=step.sql_completeness.is_complete,
            missing_info=step.sql_completeness.missing_info,
            confidence_level=step.sql_completeness.confidence_level,
        ),
        tool_call=ToolCallModel(
            name=step.tool_call.name,
            raw_parameters_xml=step.tool_call.raw_parameters_xml,
            parameters=step.tool_call.parsed_parameters,
        ),
        tool_result=step.tool_result,
        llm_output=step.llm_output,
    )


def _ensure_state_from_request(request: ReactRequest) -> ReactSessionState:
    if request.session_state:
        state = ReactSessionState.from_token(request.session_state)
        # 세션에서 복원해도 새로운 요청의 max_sql_seconds, prefer_language를 우선 적용
        state.max_sql_seconds = request.max_sql_seconds
        state.prefer_language = request.prefer_language
    else:
        dbms = request.dbms or settings.target_db_type
        state = ReactSessionState.new(
            user_query=request.question,
            dbms=dbms,
            remaining_tool_calls=request.max_tool_calls,
            step_confirmation_mode=request.step_confirmation_mode or False,
            max_sql_seconds=request.max_sql_seconds,
            prefer_language=request.prefer_language,
        )

    if request.step_confirmation_mode is not None:
        state.step_confirmation_mode = request.step_confirmation_mode

    if state.awaiting_step_confirmation:
        if request.step_confirmation_response == "continue":
            state.awaiting_step_confirmation = False
        else:
            raise HTTPException(
                status_code=400,
                detail="Step confirmation response required to continue.",
            )
    return state


@router.post("", response_class=StreamingResponse)
async def run_react(
    request: ReactRequest,
    neo4j_session=Depends(get_neo4j_session),
    db_conn=Depends(get_db_connection),
    openai_client=Depends(get_openai_client),
) -> StreamingResponse:
    react_run_id = _new_react_run_id()
    api_started = time.perf_counter()
    state = _ensure_state_from_request(request)

    tool_context = ToolContext(
        neo4j_session=neo4j_session,
        db_conn=db_conn,
        openai_client=openai_client,
        react_run_id=react_run_id,
        max_sql_seconds=state.max_sql_seconds,
    )

    agent = ReactAgent()
    warnings: List[str] = []
    steps: List[ReactStepModel] = []

    SmartLogger.log(
        "INFO",
        "react.api.request",
        category="react.api.request",
        params=sanitize_for_log(
            {
                "react_run_id": react_run_id,
                "request": request.model_dump(),
                "state_snapshot": state.to_dict(),
            }
        ),
    )

    async def event_iterator():
        nonlocal warnings
        try:
            async for event in agent.stream(
                state=state,
                tool_context=tool_context,
                max_iterations=request.max_iterations,
                user_response=request.user_response,
                react_run_id=react_run_id,
                api_started_perf_counter=api_started,
            ):
                event_type = event["type"]

                if event_type == "phase":
                    # 중간 진행 상태 전송 (thinking, reasoning, acting, observing)
                    payload = {
                        "event": "phase",
                        "phase": event["phase"],
                        "iteration": event["iteration"],
                        "data": event["data"],
                        "state": event["state"],
                    }
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
                    continue

                if event_type == "step":
                    step_model = _step_to_model(event["step"])
                    steps.append(step_model)
                    payload = {
                        "event": "step",
                        "step": step_model.model_dump(),
                        "state": event["state"],
                    }
                    yield json.dumps(payload, ensure_ascii=False) + "\n"

                    requires_step_confirmation = (
                        state.step_confirmation_mode
                        and step_model.tool_call.name not in {"submit_sql", "ask_user"}
                    )

                    if requires_step_confirmation:
                        state_snapshot_dict = event.get("state") or state.to_dict()
                        state.awaiting_step_confirmation = True
                        state_snapshot = ReactSessionState.from_dict(state_snapshot_dict)
                        state_snapshot.awaiting_step_confirmation = True
                        session_token = state_snapshot.to_token()
                        question = (
                            f"Iteration {step_model.iteration} 결과까지 확인했습니다. "
                            "다음 스텝으로 진행하시겠습니까?"
                        )
                        response_payload = ReactResponse(
                            status="await_step_confirmation",
                            steps=steps,
                            collected_metadata=state_snapshot.metadata.to_xml(),
                            partial_sql=state_snapshot.partial_sql,
                            remaining_tool_calls=state_snapshot.remaining_tool_calls,
                            session_state=session_token,
                            question_to_user=question,
                            warnings=None,
                        )
                        SmartLogger.log(
                            "INFO",
                            "react.api.step_confirmation",
                            category="react.api.step_confirmation",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": react_run_id,
                                    "elapsed_ms": (time.perf_counter() - api_started) * 1000.0,
                                    "response": response_payload.model_dump(),
                                    "state_snapshot": state_snapshot.to_dict(),
                                }
                            ),
                        )
                        confirmation_payload = {
                            "event": "step_confirmation",
                            "response": response_payload.model_dump(),
                            "state": event["state"],
                        }
                        yield json.dumps(confirmation_payload, ensure_ascii=False) + "\n"
                        return
                    continue

                if event_type == "outcome":
                    outcome: AgentOutcome = event["outcome"]

                    if outcome.status == "ask_user":
                        question = outcome.question_to_user or ""
                        state.pending_agent_question = question
                        state.current_tool_result = (
                            "<tool_result>"
                            f"<ask_user_question>{_to_cdata(question)}</ask_user_question>"
                            "</tool_result>"
                        )
                        session_token = state.to_token()
                        response_payload = ReactResponse(
                            status="needs_user_input",
                            steps=steps,
                            collected_metadata=state.metadata.to_xml(),
                            partial_sql=state.partial_sql,
                            remaining_tool_calls=state.remaining_tool_calls,
                            session_state=session_token,
                            question_to_user=question,
                            warnings=None,
                        )
                        SmartLogger.log(
                            "INFO",
                            "react.api.needs_user_input",
                            category="react.api.needs_user_input",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": react_run_id,
                                    "elapsed_ms": (time.perf_counter() - api_started) * 1000.0,
                                    "response": response_payload.model_dump(),
                                    "state_snapshot": state.to_dict(),
                                }
                            ),
                        )
                        payload = {
                            "event": "needs_user_input",
                            "response": response_payload.model_dump(),
                            "state": event["state"],
                        }
                        yield json.dumps(payload, ensure_ascii=False) + "\n"
                        return

                    if outcome.status != "submit_sql":
                        SmartLogger.log(
                            "ERROR",
                            "react.api.error",
                            category="react.api.error",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": react_run_id,
                                    "elapsed_ms": (time.perf_counter() - api_started) * 1000.0,
                                    "message": "Agent did not complete with submit_sql.",
                                    "outcome_status": outcome.status,
                                    "state_snapshot": state.to_dict(),
                                }
                            ),
                        )
                        error_payload = {
                            "event": "error",
                            "message": "Agent did not complete with submit_sql.",
                        }
                        yield json.dumps(error_payload, ensure_ascii=False) + "\n"
                        return

                    final_sql = outcome.final_sql or ""
                    
                    # 마지막 호출 제출인 경우 경고 추가
                    if outcome.was_last_call_submission:
                        warnings.append(
                            "남은 도구 호출 횟수가 1 이하여서 SQL이 검증 없이 제출되었습니다. "
                            "SQL이 실행되지 않으며, 결과는 참고용으로만 사용하세요."
                        )
                        if outcome.sql_not_explained:
                            warnings.append(
                                "이 SQL은 explain 도구로 사전 검증되지 않았습니다. "
                                "실행 전에 수동으로 검토하시기 바랍니다."
                            )
                    
                    guard = SQLGuard()
                    try:
                        validated_sql, _ = guard.validate(final_sql)
                    except SQLValidationError as exc:
                        SmartLogger.log(
                            "ERROR",
                            "react.api.error",
                            category="react.api.error",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": react_run_id,
                                    "elapsed_ms": (time.perf_counter() - api_started) * 1000.0,
                                    "message": f"SQL validation failed: {exc}",
                                    "final_sql": final_sql,
                                    "state_snapshot": state.to_dict(),
                                    "traceback": traceback.format_exc(),
                                }
                            ),
                        )
                        error_payload = {
                            "event": "error",
                            "message": f"SQL validation failed: {exc}",
                        }
                        yield json.dumps(error_payload, ensure_ascii=False) + "\n"
                        return

                    execution_result = None
                    # 마지막 호출 제출이 아닐 때만 SQL 실행
                    if request.execute_final_sql and not outcome.was_last_call_submission:
                        executor = SQLExecutor()
                        try:
                            raw_result = await executor.execute_query(
                                db_conn, 
                                validated_sql,
                                timeout=float(request.max_sql_seconds),
                            )
                            formatted = executor.format_results_for_json(raw_result)
                            execution_result = ExecutionResultModel(**formatted)
                        except SQLExecutionError as exc:
                            warnings.append(f"SQL execution failed: {exc}")

                    response_payload = ReactResponse(
                        status="completed",
                        final_sql=final_sql,
                        validated_sql=validated_sql,
                        execution_result=execution_result,
                        steps=steps,
                        collected_metadata=state.metadata.to_xml(),
                        partial_sql=state.partial_sql,
                        remaining_tool_calls=state.remaining_tool_calls,
                        session_state=None,
                        question_to_user=None,
                        warnings=warnings or None,
                    )
                    SmartLogger.log(
                        "INFO",
                        "react.api.completed",
                        category="react.api.completed",
                        params=sanitize_for_log(
                            {
                                "react_run_id": react_run_id,
                                "elapsed_ms": (time.perf_counter() - api_started) * 1000.0,
                                "response": response_payload.model_dump(),
                                "state_snapshot": state.to_dict(),
                            }
                        ),
                    )
                    payload = {
                        "event": "completed",
                        "response": response_payload.model_dump(),
                        "state": event["state"],
                    }
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
                    return

                if event_type == "error":
                    message = str(event["error"])
                    SmartLogger.log(
                        "ERROR",
                        "react.api.error",
                        category="react.api.error",
                        params=sanitize_for_log(
                            {
                                "react_run_id": react_run_id,
                                "elapsed_ms": (time.perf_counter() - api_started) * 1000.0,
                                "message": message,
                                "state_snapshot": state.to_dict(),
                            }
                        ),
                    )
                    payload = {"event": "error", "message": message}
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
                    return
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "react.api.error",
                category="react.api.error",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "elapsed_ms": (time.perf_counter() - api_started) * 1000.0,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                        "request": request.model_dump(),
                        "state_snapshot": state.to_dict(),
                    }
                ),
            )
            payload = {"event": "error", "message": str(exc)}
            yield json.dumps(payload, ensure_ascii=False) + "\n"

    return StreamingResponse(event_iterator(), media_type="application/x-ndjson")
