"""
Direct SQL 실행 라우터
- SQL 직접 입력 및 실행
- AI 결과 포맷팅 (선택적)
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.react.llm_factory import create_react_llm
from app.deps import get_db_connection
from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.core.sql_mindsdb_prepare import prepare_sql_for_mindsdb
from app.core.sql_autocorrect_llm import propose_sql_fix_light_llm
from app.smart_logger import SmartLogger


router = APIRouter(prefix="/direct-sql", tags=["Direct SQL"])

AI_SUMMARY_MIN_CHARS = 80
AI_SUMMARY_STREAM_CHUNK_SIZE = 180
AI_SUMMARY_SYSTEM_PROMPT = (
    "당신은 데이터 분석 결과를 사용자에게 친절하고 정확하게 설명하는 전문가입니다. "
    "반드시 완결된 문장으로 마무리하고, 미완성 제목(예: '###')이나 끊긴 문장으로 끝내지 마세요."
)


class DirectSqlRequest(BaseModel):
    sql: str = Field(..., description="실행할 SQL 쿼리")
    datasource: str = Field(..., description="MindsDB datasource (required; Phase 1 MindsDB-only)")
    max_sql_seconds: int = Field(default=60, ge=1, le=3600, description="SQL 실행 최대 허용 시간(초)")
    format_with_ai: bool = Field(default=False, description="AI로 결과 포맷팅 여부")


class CreateMaterializedViewRequest(BaseModel):
    """(Phase 1) MindsDB view 생성 요청 (기존 MV 엔드포인트를 view로 재정의, D8)"""

    datasource: str = Field(..., description="MindsDB datasource (required)")
    view_name: str = Field(..., description="생성할 뷰 이름", pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    schema_name: str = Field(default="public", description="스키마 이름")
    # Backward-compatible: some clients use sql_query field name.
    source_sql: Optional[str] = Field(default=None, description="뷰의 기반 SQL 쿼리")
    sql_query: Optional[str] = Field(default=None, description="(legacy) 뷰의 기반 SQL 쿼리")
    refresh_on_create: bool = Field(default=True, description="생성 후 즉시 row_count 확인(베스트에포트)")
    description: str = Field(default="", description="뷰 설명(베스트에포트)")


class MaterializedViewResponse(BaseModel):
    """(호환) Materialized View 생성 결과 형태 유지"""

    status: str  # success, error
    view_name: str
    full_name: str  # datasource.schema.view_name
    row_count: int = 0
    message: str = ""
    error_message: Optional[str] = None


class DirectSqlResponse(BaseModel):
    status: str  # success, error
    sql: str
    validated_sql: Optional[str] = None
    columns: List[str] = []
    rows: List[List[Any]] = []
    row_count: int = 0
    execution_time_ms: float = 0
    truncated: Optional[bool] = None
    returned_row_count: Optional[int] = None
    max_rows_cap: Optional[int] = None
    error_message: Optional[str] = None
    formatted_summary: Optional[str] = None  # AI 포맷팅 결과


@router.post("", response_model=DirectSqlResponse)
async def execute_direct_sql(
    request: DirectSqlRequest,
    db_conn=Depends(get_db_connection),
) -> DirectSqlResponse:
    """
    SQL 쿼리를 직접 실행합니다.
    
    - SQL 검증 (읽기 전용)
    - 실행 및 결과 반환
    - 선택적으로 AI가 결과를 요약/포맷팅
    """
    start_time = time.perf_counter()
    
    SmartLogger.log(
        "INFO",
        "direct_sql.request",
        category="direct_sql.request",
        params={"sql": request.sql[:200]},
    )
    
    # 1. SQL 검증
    guard = SQLGuard()
    try:
        validated_sql, _ = guard.validate(request.sql)
    except SQLValidationError as exc:
        SmartLogger.log(
            "WARNING",
            "direct_sql.validation_error",
            category="direct_sql.validation_error",
            params={"error": str(exc)},
        )
        return DirectSqlResponse(
            status="error",
            sql=request.sql,
            error_message=f"SQL 검증 실패: {exc}"
        )
    
    # 2. SQL 실행
    executor = SQLExecutor()
    try:
        db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
        # Deterministic pipeline (MindsDB-safe):
        # 1) Try original validated SQL first (keeps passthrough inner SQL intact)
        # 2) On failure (MindsDB mode), apply deterministic prepare and retry once
        attempted_sql = validated_sql
        try:
            raw_result = await executor.execute_query(
                db_conn,
                attempted_sql,
                timeout=float(request.max_sql_seconds),
            )
        except SQLExecutionError as first_exc:
            if db_type in {"mysql", "mariadb"}:
                prep = prepare_sql_for_mindsdb(validated_sql, request.datasource)
                if prep.sql.strip() != attempted_sql.strip():
                    attempted_sql = prep.sql
                    raw_result = await executor.execute_query(
                        db_conn,
                        attempted_sql,
                        timeout=float(request.max_sql_seconds),
                    )
                else:
                    raise first_exc
            else:
                raise first_exc

        # Return the SQL that was actually executed
        validated_sql = attempted_sql
        formatted = executor.format_results_for_json(raw_result)
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        columns = formatted.get("columns", [])
        rows = formatted.get("rows", [])
        row_count = formatted.get("row_count", 0)
        truncated = formatted.get("truncated")
        returned_row_count = formatted.get("returned_row_count")
        max_rows_cap = formatted.get("max_rows_cap")
        
    except SQLExecutionError as exc:
        # Fallback: light LLM one-shot fix (MindsDB mode only)
        db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
        if db_type in {"mysql", "mariadb"}:
            try:
                llm_fix = await propose_sql_fix_light_llm(
                    datasource=request.datasource,
                    original_sql=validated_sql,
                    error_text=str(exc),
                )
                if (not llm_fix.error) and (not llm_fix.ok) and llm_fix.sql:
                    guard2 = SQLGuard()
                    llm_validated, _ = guard2.validate(llm_fix.sql)
                    raw_result = await executor.execute_query(
                        db_conn,
                        llm_validated,
                        timeout=float(request.max_sql_seconds),
                    )
                    formatted = executor.format_results_for_json(raw_result)
                    execution_time_ms = (time.perf_counter() - start_time) * 1000
                    columns = formatted.get("columns", [])
                    rows = formatted.get("rows", [])
                    row_count = formatted.get("row_count", 0)

                    formatted_summary = None
                    if request.format_with_ai and rows:
                        try:
                            formatted_summary = await _format_with_ai(
                                llm_validated,
                                columns,
                                rows[:20],
                                row_count,
                            )
                        except Exception:
                            formatted_summary = None
                    return DirectSqlResponse(
                        status="success",
                        sql=request.sql,
                        validated_sql=llm_validated,
                        columns=columns,
                        rows=rows,
                        row_count=row_count,
                        execution_time_ms=execution_time_ms,
                        truncated=bool(formatted.get("truncated")) if formatted.get("truncated") is not None else None,
                        returned_row_count=int(formatted.get("returned_row_count")) if formatted.get("returned_row_count") is not None else None,
                        max_rows_cap=int(formatted.get("max_rows_cap")) if formatted.get("max_rows_cap") is not None else None,
                        formatted_summary=formatted_summary,
                    )
            except Exception:
                # fail closed to original error response below
                pass
        SmartLogger.log(
            "ERROR",
            "direct_sql.execution_error",
            category="direct_sql.execution_error",
            params={"error": str(exc)},
        )
        return DirectSqlResponse(
            status="error",
            sql=request.sql,
            validated_sql=validated_sql,
            error_message=f"SQL 실행 실패: {exc}"
        )
    
    # 3. AI 포맷팅 (선택적)
    formatted_summary = None
    if request.format_with_ai and rows:
        try:
            formatted_summary = await _format_with_ai(
                validated_sql,
                columns,
                rows[:20],  # 처음 20행만
                row_count
            )
        except Exception as e:
            SmartLogger.log(
                "WARNING",
                "direct_sql.ai_format_error",
                category="direct_sql.ai_format_error",
                params={"error": str(e)},
            )
    
    SmartLogger.log(
        "INFO",
        "direct_sql.success",
        category="direct_sql.success",
        params={
            "row_count": row_count,
            "execution_time_ms": execution_time_ms,
        },
    )
    
    return DirectSqlResponse(
        status="success",
        sql=request.sql,
        validated_sql=validated_sql,
        columns=columns,
        rows=rows,
        row_count=row_count,
        execution_time_ms=execution_time_ms,
        truncated=bool(truncated) if truncated is not None else None,
        returned_row_count=int(returned_row_count) if returned_row_count is not None else None,
        max_rows_cap=int(max_rows_cap) if max_rows_cap is not None else None,
        formatted_summary=formatted_summary,
    )


def _build_sample_data(columns: List[str], rows: List[List[Any]], limit: int = 10) -> List[Dict[str, Any]]:
    sample_data: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        row_dict = {columns[i]: row[i] for i in range(len(columns))}
        sample_data.append(row_dict)
    return sample_data


def _build_summary_prompt(sql: str, sample_data: List[Dict[str, Any]], total_row_count: int) -> str:
    return f"""다음 SQL 쿼리 결과를 사용자에게 친절하게 요약해주세요.

SQL:
```sql
{sql}
```

결과 ({total_row_count}개 행):
```json
{json.dumps(sample_data, ensure_ascii=False, indent=2, default=str)[:2000]}
```

요구사항:
1. 결과의 핵심 내용을 2-3문장으로 요약
2. 주요 인사이트나 패턴이 있다면 언급
3. 한국어로 작성
4. 숫자는 읽기 쉽게 포맷팅 (예: 1,234.56)
5. 답변은 반드시 완결된 문장으로 마무리 (단독 '###' 등 미완성 텍스트 금지)
"""


def _looks_incomplete_summary(summary: str) -> bool:
    s = (summary or "").strip()
    if not s:
        return True
    if len(s) < AI_SUMMARY_MIN_CHARS:
        return True
    trailing_markers = (
        "###",
        "##",
        "#",
        "-",
        "*",
        "•",
        ":",
        "·",
        "(",
        "[",
        "{",
        ",",
    )
    if any(s.endswith(marker) for marker in trailing_markers):
        return True
    if s.count("```") % 2 == 1:
        return True
    return False


def _split_summary_for_stream(summary: str, chunk_size: int = AI_SUMMARY_STREAM_CHUNK_SIZE) -> List[str]:
    text = summary or ""
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def _deep_extract_text(payload: Any, depth: int = 0) -> str:
    if payload is None or depth > 5:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        preferred = []
        for key in ("text", "output_text"):
            if key in payload:
                preferred.append(_deep_extract_text(payload.get(key), depth + 1))
        preferred_joined = "".join(preferred)
        if preferred_joined.strip():
            return preferred_joined
        return "".join(_deep_extract_text(v, depth + 1) for v in payload.values())
    if isinstance(payload, list):
        return "".join(_deep_extract_text(item, depth + 1) for item in payload)
    return ""


def _llm_chunk_to_text(chunk: Any) -> str:
    # 1) 표준 content 필드 우선
    content_text = _llm_content_to_text(getattr(chunk, "content", None))
    if content_text and content_text.strip():
        return content_text
    # 2) 일부 구현체는 text/additional_kwargs 쪽에만 토큰이 존재
    text_attr = getattr(chunk, "text", None)
    if isinstance(text_attr, str) and text_attr.strip():
        return text_attr
    for attr in ("additional_kwargs", "response_metadata"):
        recovered = _deep_extract_text(getattr(chunk, attr, None))
        if recovered and recovered.strip():
            return recovered
    # 3) 마지막 보강: 청크 전체를 재귀 탐색
    recovered_from_chunk = _deep_extract_text(chunk)
    return recovered_from_chunk if recovered_from_chunk.strip() else ""


def _llm_response_to_text(response: Any) -> str:
    content_text = _llm_content_to_text(getattr(response, "content", None))
    if content_text and content_text.strip():
        return content_text
    for attr in ("additional_kwargs", "response_metadata"):
        recovered = _deep_extract_text(getattr(response, attr, None))
        if recovered and recovered.strip():
            return recovered
    recovered_full = _deep_extract_text(response)
    return recovered_full if recovered_full.strip() else ""


async def _format_with_ai(
    sql: str,
    columns: List[str],
    rows: List[List[Any]],
    total_row_count: int
) -> str:
    """AI로 결과 요약/포맷팅 (완결성 보장 재시도 포함)"""
    sample_data = _build_sample_data(columns, rows, limit=10)
    prompt = _build_summary_prompt(sql, sample_data, total_row_count)

    llm = create_react_llm(
        purpose="direct_sql.format",
        thinking_level=None,
        include_thoughts=False,
        temperature=0.0,
        use_light=True
    ).llm
    resp = await llm.ainvoke(
        [
            SystemMessage(content=AI_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    summary = _llm_response_to_text(resp).strip()
    if not _looks_incomplete_summary(summary):
        return summary

    # 1차 결과가 비정상적으로 짧거나 미완성이면 한번 더 복구 시도
    retry_resp = await llm.ainvoke(
        [
            SystemMessage(content=AI_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    prompt
                    + "\n\n추가 요구사항: 답변을 3~5문장으로 완결성 있게 마무리하고,"
                    + " 끝 문장은 마침표로 종료하세요."
                )
            ),
        ]
    )
    retry_summary = _llm_response_to_text(retry_resp).strip()
    return retry_summary if retry_summary else summary


async def _format_with_ai_stream_recovery(
    sql: str,
    columns: List[str],
    rows: List[List[Any]],
    total_row_count: int
) -> str:
    """
    스트리밍 요약을 우선 시도하고, 결과가 미완성이면 non-stream 호출로 복구한다.
    """
    sample_data = _build_sample_data(columns, rows, limit=10)
    prompt = _build_summary_prompt(sql, sample_data, total_row_count)

    llm = create_react_llm(
        purpose="direct_sql.format_stream",
        thinking_level=None,
        include_thoughts=False,
        temperature=0.0,
        use_light=True
    ).llm
    streamed_parts: List[str] = []
    async for chunk in llm.astream(
        [
            SystemMessage(content=AI_SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    ):
        token = _llm_chunk_to_text(chunk)
        if token:
            streamed_parts.append(token)

    streamed_summary = "".join(streamed_parts).strip()
    if not _looks_incomplete_summary(streamed_summary):
        return streamed_summary

    SmartLogger.log(
        "WARNING",
        "direct_sql.stream_summary_incomplete",
        category="direct_sql.stream_summary_incomplete",
        params={"length": len(streamed_summary)},
    )
    try:
        recovered_summary = await _format_with_ai(sql, columns, rows, total_row_count)
        if recovered_summary and recovered_summary.strip():
            return recovered_summary.strip()
    except Exception as exc:
        SmartLogger.log(
            "WARNING",
            "direct_sql.stream_summary_recovery_failed",
            category="direct_sql.stream_summary_recovery_failed",
            params={"error": str(exc)},
        )
    return streamed_summary


@router.post("/stream")
async def execute_direct_sql_stream(
    request: DirectSqlRequest,
    db_conn=Depends(get_db_connection),
) -> StreamingResponse:
    """
    SQL 쿼리를 실행하고 결과를 스트리밍합니다.
    AI 포맷팅도 스트리밍으로 제공됩니다.
    """
    
    async def event_generator():
        start_time = time.perf_counter()
        
        # 1. SQL 검증
        yield json.dumps({"event": "validating", "message": "SQL 검증 중..."}, ensure_ascii=False) + "\n"
        
        guard = SQLGuard()
        try:
            validated_sql, _ = guard.validate(request.sql)
            yield json.dumps({"event": "validated", "validated_sql": validated_sql}, ensure_ascii=False) + "\n"
        except SQLValidationError as exc:
            yield json.dumps({"event": "error", "message": f"SQL 검증 실패: {exc}"}, ensure_ascii=False) + "\n"
            return
        
        # 2. SQL 실행
        yield json.dumps({"event": "executing", "message": "SQL 실행 중..."}, ensure_ascii=False) + "\n"
        
        executor = SQLExecutor()
        try:
            db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
            attempted_sql = validated_sql

            # Try original first (passthrough-safe)
            try:
                raw_result = await executor.execute_query(
                    db_conn,
                    attempted_sql,
                    timeout=float(request.max_sql_seconds),
                )
            except SQLExecutionError as first_exc:
                if db_type in {"mysql", "mariadb"}:
                    prep = prepare_sql_for_mindsdb(validated_sql, request.datasource)
                    if prep.sql.strip() != attempted_sql.strip():
                        attempted_sql = prep.sql
                        yield json.dumps(
                            {
                                "event": "autocorrect",
                                "mode": "deterministic",
                                "reason": prep.reason,
                                "prepared_sql": attempted_sql,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                        raw_result = await executor.execute_query(
                            db_conn,
                            attempted_sql,
                            timeout=float(request.max_sql_seconds),
                        )
                    else:
                        raise first_exc
                else:
                    raise first_exc

            validated_sql = attempted_sql
            formatted = executor.format_results_for_json(raw_result)
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            result_payload = {
                "event": "result",
                "columns": formatted.get("columns", []),
                "rows": formatted.get("rows", []),
                "row_count": formatted.get("row_count", 0),
                "execution_time_ms": execution_time_ms,
                "truncated": formatted.get("truncated"),
                "returned_row_count": formatted.get("returned_row_count"),
                "max_rows_cap": formatted.get("max_rows_cap"),
            }
            yield json.dumps(result_payload, ensure_ascii=False, default=str) + "\n"
            
        except SQLExecutionError as exc:
            # Fallback: light LLM one-shot fix (MindsDB mode only)
            fixed = False
            try:
                db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
                if db_type in {"mysql", "mariadb"}:
                    llm_fix = await propose_sql_fix_light_llm(
                        datasource=request.datasource,
                        original_sql=validated_sql,
                        error_text=str(exc),
                    )
                    if (not llm_fix.error) and (not llm_fix.ok) and llm_fix.sql:
                        guard2 = SQLGuard()
                        llm_validated, _ = guard2.validate(llm_fix.sql)
                        yield json.dumps(
                            {
                                "event": "autocorrect",
                                "mode": "llm_light",
                                "reason": "llm_one_shot",
                                "prepared_sql": llm_validated,
                                "llm_model": llm_fix.model,
                                "llm_elapsed_ms": llm_fix.elapsed_ms,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                        raw_result = await executor.execute_query(
                            db_conn,
                            llm_validated,
                            timeout=float(request.max_sql_seconds),
                        )
                        formatted = executor.format_results_for_json(raw_result)
                        execution_time_ms = (time.perf_counter() - start_time) * 1000
                        yield json.dumps(
                            {
                                "event": "result",
                                "columns": formatted.get("columns", []),
                                "rows": formatted.get("rows", []),
                                "row_count": formatted.get("row_count", 0),
                                "execution_time_ms": execution_time_ms,
                                "truncated": formatted.get("truncated"),
                                "returned_row_count": formatted.get("returned_row_count"),
                                "max_rows_cap": formatted.get("max_rows_cap"),
                            },
                            ensure_ascii=False,
                            default=str,
                        ) + "\n"
                        validated_sql = llm_validated
                        fixed = True
            except Exception:
                fixed = False

            if not fixed:
                yield json.dumps({"event": "error", "message": f"SQL 실행 실패: {exc}"}, ensure_ascii=False) + "\n"
                return
        
        # 3. AI 포맷팅 (선택적)
        if request.format_with_ai and formatted.get("rows"):
            yield json.dumps({"event": "formatting", "message": "AI가 결과를 분석 중..."}, ensure_ascii=False) + "\n"
            
            try:
                columns = formatted.get("columns", [])
                rows = formatted.get("rows", [])
                total_row_count = formatted.get("row_count", 0)
                summary_text = await _format_with_ai_stream_recovery(
                    validated_sql,
                    columns,
                    rows,
                    total_row_count,
                )
                for token in _split_summary_for_stream(summary_text):
                    yield json.dumps(
                        {"event": "format_token", "token": token},
                        ensure_ascii=False,
                    ) + "\n"
                
                yield json.dumps({"event": "format_done"}, ensure_ascii=False) + "\n"
                
            except Exception as e:
                yield json.dumps({
                    "event": "format_error",
                    "message": f"AI 포맷팅 실패: {str(e)}"
                }, ensure_ascii=False) + "\n"
        
        # 완료
        yield json.dumps({"event": "completed"}, ensure_ascii=False) + "\n"
    
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.post("/materialized-view", response_model=MaterializedViewResponse)
async def create_materialized_view(
    request: CreateMaterializedViewRequest,
    db_conn=Depends(get_db_connection),
) -> MaterializedViewResponse:
    """
    (D8) 기존 MV 엔드포인트를 MindsDB view(가상 테이블) 생성으로 재정의한다.
    - MindsDB MySQL endpoint 기준: `CREATE VIEW datasource.schema.view AS <source_sql>`
    - refresh_on_create는 row_count 확인(베스트에포트)로 처리한다.
    """
    db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
    if db_type not in {"mysql", "mariadb"}:
        return MaterializedViewResponse(
            status="error",
            view_name=request.view_name,
            full_name=f"{request.datasource}.{request.schema_name}.{request.view_name}",
            error_message="MindsDB view 엔드포인트는 target_db_type=mysql(MindsDB) 모드에서만 지원됩니다.",
        )

    executor = SQLExecutor()
    guard = SQLGuard()

    # 1) Validate & transform source SQL (SELECT-only + datasource prefix)
    source_sql = (request.source_sql or request.sql_query or "").strip()
    if not source_sql:
        return MaterializedViewResponse(
            status="error",
            view_name=request.view_name,
            full_name=f"{request.datasource}.{request.schema_name}.{request.view_name}",
            error_message="source_sql(sql_query) is required",
        )

    try:
        validated_source, _ = guard.validate(source_sql)
        validated_source = prepare_sql_for_mindsdb(validated_source, request.datasource).sql
    except Exception as exc:
        return MaterializedViewResponse(
            status="error",
            view_name=request.view_name,
            full_name=f"{request.datasource}.{request.schema_name}.{request.view_name}",
            error_message=f"소스 SQL 검증/변환 실패: {exc}",
        )

    # 2) Create view (drop + create)
    view_ident = f"`{request.datasource}`.`{request.schema_name}`.`{request.view_name}`"
    try:
        await executor.execute_ddl(
            db_conn,
            f"DROP VIEW IF EXISTS {view_ident}",
            timeout=float(settings.sql_timeout_seconds),
        )
        await executor.execute_ddl(
            db_conn,
            f"CREATE VIEW {view_ident} AS {validated_source}",
            timeout=float(settings.sql_timeout_seconds),
        )
    except Exception as exc:
        return MaterializedViewResponse(
            status="error",
            view_name=request.view_name,
            full_name=f"{request.datasource}.{request.schema_name}.{request.view_name}",
            error_message=f"View 생성 실패: {exc}",
        )

    # 3) Best-effort row count
    row_count = 0
    if request.refresh_on_create:
        try:
            res = await executor.execute_query(
                db_conn,
                f"SELECT COUNT(*) AS cnt FROM {view_ident}",
                timeout=float(settings.sql_timeout_seconds),
            )
            rows = res.get("rows") or []
            if rows and rows[0]:
                row_count = int(rows[0][0] or 0)
        except Exception:
            row_count = 0

    return MaterializedViewResponse(
        status="success",
        view_name=request.view_name,
        full_name=f"{request.datasource}.{request.schema_name}.{request.view_name}",
        row_count=row_count,
        message=f"View '{request.view_name}' 생성 완료 ({row_count}개 행)",
    )


@router.post("/materialized-view/{view_name}/refresh")
async def refresh_materialized_view(
    view_name: str,
    datasource: str,
    schema_name: str = "public",
    db_conn=Depends(get_db_connection),
) -> dict:
    """
    MindsDB view는 일반적으로 조회 시점에 계산되므로, refresh는 'row_count 확인'으로 동작한다(베스트에포트).
    """
    executor = SQLExecutor()
    view_ident = f"`{datasource}`.`{schema_name}`.`{view_name}`"
    try:
        res = await executor.execute_query(
            db_conn,
            f"SELECT COUNT(*) AS cnt FROM {view_ident}",
            timeout=float(settings.sql_timeout_seconds),
        )
        rows = res.get("rows") or []
        row_count = int(rows[0][0] or 0) if rows and rows[0] else 0
        return {
            "status": "success",
            "view_name": view_name,
            "row_count": row_count,
            "message": f"View '{view_name}' 조회 완료 ({row_count}개 행)",
        }
    except Exception as exc:
        return {"status": "error", "view_name": view_name, "error_message": f"조회 실패: {exc}"}


@router.get("/materialized-views")
async def list_materialized_views(
    datasource: str,
    schema_name: str = "public",
    db_conn=Depends(get_db_connection),
) -> dict:
    """MindsDB view 목록 조회 (INFORMATION_SCHEMA.VIEWS 베스트에포트)"""
    executor = SQLExecutor()
    try:
        ds_esc = datasource.replace("'", "''")
        sch_esc = schema_name.replace("'", "''")
        sql = (
            "SELECT TABLE_SCHEMA, TABLE_NAME, VIEW_DEFINITION "
            "FROM INFORMATION_SCHEMA.VIEWS "
            f"WHERE TABLE_SCHEMA = '{sch_esc}' "
            f"AND (TABLE_CATALOG = '{ds_esc}' OR TABLE_CATALOG IS NULL) "
            "ORDER BY TABLE_NAME"
        )
        res = await executor.execute_query(db_conn, sql, timeout=float(settings.sql_timeout_seconds))
        views = []
        for r in res.get("rows", []) or []:
            schema = str(r[0] or "")
            name = str(r[1] or "")
            definition = str(r[2] or "") if len(r) > 2 else ""
            views.append(
                {
                    "schema": schema,
                    "name": name,
                    "owner": "mindsdb",
                    "is_populated": True,
                    "size": "",
                    "description": (definition[:2000] if definition else None),
                }
            )
        return {"status": "success", "views": views, "count": len(views)}
    except Exception as exc:
        return {"status": "error", "views": [], "count": 0, "error_message": str(exc)}


def _llm_content_to_text(content: Any) -> str:
    """
    Normalize LangChain message chunk content to plain text.
    Gemini may emit dict/list parts (including 'thinking'); we only keep 'text'.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content and str(content.get("text") or "").strip():
            return str(content.get("text") or "")
        return ""
    if isinstance(content, list):
        out: List[str] = []
        for part in content:
            if isinstance(part, str) and part:
                out.append(part)
                continue
            if isinstance(part, dict) and "text" in part and str(part.get("text") or "").strip():
                out.append(str(part.get("text") or ""))
                continue
        return "".join(out)
    return str(content)

