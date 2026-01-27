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
from app.react.streaming_xml_sections import StreamingXmlSectionsExtractor
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
    debug_stream_xml_tokens: bool = Field(
        default=False, description="디버그용: raw XML 토큰 스트림을 그대로 전송할지 여부"
    )
    use_cache: bool = Field(
        default=True, description="동일한 질문에 대해 캐시된 결과를 사용할지 여부"
    )
    schema_filter: Optional[List[str]] = Field(
        default=None, description="검색 대상 스키마 제한 (예: ['dw']는 OLAP 데이터만 검색)"
    )
    object_type_only: bool = Field(
        default=False, description="ObjectType(Materialized View) 테이블만 검색 (도메인 레이어 모드)"
    )
    linked_object_types: Optional[List[Dict[str, Any]]] = Field(
        default=None, 
        description="연결된 ObjectType 정보. 프롬프트 확장에 사용. [{name, columns: [{name, type}], description}]"
    )
    prefer_formula: bool = Field(
        default=False, description="계리수식/공식 우선 검색 모드. True일 경우 수식 컬럼과 계산식을 우선 탐색"
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
    from app.core.query_cache import get_query_cache
    
    react_run_id = _new_react_run_id()
    api_started = time.perf_counter()
    
    # 캐시 확인 (새 세션이고 use_cache=True인 경우에만)
    if request.use_cache and not request.session_state and not request.user_response:
        cache = get_query_cache()
        cached = cache.get(request.question)
        
        if cached:
            # 캐시 히트: 빠르게 결과 반환
            SmartLogger.log(
                "INFO",
                "react.cache.hit",
                category="react.cache",
                params={"question": request.question[:50], "hit_count": cached.hit_count},
            )
            
            async def cached_event_iterator():
                # 캐시 히트 알림
                yield json.dumps({
                    "event": "cache_hit",
                    "message": "캐시된 결과를 반환합니다",
                    "hit_count": cached.hit_count
                }, ensure_ascii=False) + "\n"
                
                # 바로 완료 이벤트 전송 - "completed"로 통일, response 객체로 감싸서 프론트엔드와 호환
                # execution_result는 이미 dict로 저장되어 있으므로 그대로 사용
                response_obj = {
                    "status": "completed",
                    "final_sql": cached.final_sql,
                    "validated_sql": cached.validated_sql,
                    "execution_result": cached.execution_result,
                    "steps": [],  # 캐시에서는 스텝 생략
                    "collected_metadata": "",
                    "partial_sql": cached.final_sql,
                    "remaining_tool_calls": 0,
                    "from_cache": True,
                }
                result_payload = {
                    "event": "completed",
                    "response": response_obj,
                    "state": None,  # 캐시에서는 state 없음
                }
                yield json.dumps(result_payload, ensure_ascii=False) + "\n"
            
            return StreamingResponse(
                cached_event_iterator(),
                media_type="text/event-stream",
                headers={"X-Cache-Hit": "true"}
            )
    
    state = _ensure_state_from_request(request)

    tool_context = ToolContext(
        neo4j_session=neo4j_session,
        db_conn=db_conn,
        openai_client=openai_client,
        react_run_id=react_run_id,
        max_sql_seconds=state.max_sql_seconds,
        schema_filter=request.schema_filter,
        object_type_only=request.object_type_only,
        linked_object_types=request.linked_object_types,  # 연결된 ObjectType 정보 추가
        prefer_formula=request.prefer_formula,  # 계리수식 우선 검색 모드
    )

    agent = ReactAgent()
    warnings: List[str] = []
    steps: List[ReactStepModel] = []
    extractor = StreamingXmlSectionsExtractor(throttle_ms=50)

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

                if event_type == "token":
                    # Always feed token chunks to the extractor.
                    extractor.feed(iteration=int(event["iteration"]), token=str(event["token"]))

                    # Flush batched user-facing deltas (50ms throttled).
                    for out_event in extractor.flush_if_due(force=False):
                        yield json.dumps(out_event, ensure_ascii=False) + "\n"

                    # Optional debug: raw XML token streaming
                    if request.debug_stream_xml_tokens:
                        payload = {
                            "event": "token",
                            "iteration": event["iteration"],
                            "token": event["token"],
                        }
                        yield json.dumps(payload, ensure_ascii=False) + "\n"
                    continue

                if event_type == "thinking_token":
                    # Gemini thinking chunks are user-facing as "exploring" section.
                    # NOTE: Intentionally no throttling/sanitization per product requirement.
                    payload = {
                        "event": "section_delta",
                        "iteration": event.get("iteration"),
                        "section": "exploring",
                        "delta": event.get("token") or "",
                    }
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
                    continue

                if event_type == "format_repair":
                    # Ensure any pending deltas flush before showing repair banner
                    for out_event in extractor.flush_if_due(force=True):
                        yield json.dumps(out_event, ensure_ascii=False) + "\n"
                    payload = {
                        "event": "format_repair",
                        "iteration": event.get("iteration"),
                        "reason": event.get("reason") or "parse_retry",
                    }
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
                    continue

                if event_type == "phase":
                    # 중간 진행 상태 전송 (thinking, reasoning, acting, observing)
                    for out_event in extractor.flush_if_due(force=True):
                        yield json.dumps(out_event, ensure_ascii=False) + "\n"
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
                    for out_event in extractor.flush_if_due(force=True):
                        yield json.dumps(out_event, ensure_ascii=False) + "\n"
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
                    for out_event in extractor.flush_if_due(force=True):
                        yield json.dumps(out_event, ensure_ascii=False) + "\n"
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

                    # Neo4j 캐싱/히스토리 저장은 유저 응답 이후 백그라운드에서 수행 (강한 게이트 포함)
                    try:
                        from app.core.background_jobs import enqueue_cache_postprocess

                        metadata_dict = {
                            "identified_tables": state.metadata.identified_tables,
                            "identified_columns": state.metadata.identified_columns,
                            "identified_values": state.metadata.identified_values,
                            "identified_relationships": state.metadata.identified_relationships,
                            "identified_constraints": state.metadata.identified_constraints,
                        }
                        steps_for_save = [step.model_dump() for step in steps] if steps else []

                        enqueue_cache_postprocess(
                            {
                                "react_run_id": react_run_id,
                                "question": request.question,
                                "final_sql": final_sql,
                                "validated_sql": validated_sql,
                                "status": "completed",
                                "row_count": execution_result.row_count if execution_result else None,
                                "execution_time_ms": execution_result.execution_time_ms if execution_result else None,
                                "steps_count": len(steps),
                                "metadata_dict": metadata_dict,
                                "steps": steps_for_save,
                            }
                        )
                    except Exception as enqueue_err:
                        SmartLogger.log(
                            "WARNING",
                            "react.neo4j.enqueue_failed",
                            category="react.neo4j.enqueue_failed",
                            params={"react_run_id": react_run_id, "error": str(enqueue_err)[:200]},
                        )
                    
                    # 캐시에 결과 저장 (성공한 경우만)
                    if execution_result and request.use_cache:
                        try:
                            cache = get_query_cache()
                            cache.put(
                                question=request.question,
                                final_sql=final_sql,
                                validated_sql=validated_sql,
                                execution_result=execution_result.model_dump() if execution_result else None,
                                steps_summary=f"{len(steps)} steps completed"
                            )
                            SmartLogger.log(
                                "INFO",
                                "react.cache.saved",
                                category="react.cache",
                                params={"question": request.question[:50]},
                            )
                        except Exception as cache_err:
                            SmartLogger.log(
                                "WARNING",
                                "react.cache.save_failed",
                                category="react.cache",
                                params={"error": str(cache_err)[:200]},
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
                    for out_event in extractor.flush_if_due(force=True):
                        yield json.dumps(out_event, ensure_ascii=False) + "\n"
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
        # Normal termination: ensure last buffered deltas are emitted.
        for out_event in extractor.flush_if_due(force=True):
            yield json.dumps(out_event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_iterator(), media_type="application/x-ndjson")
