from __future__ import annotations

import asyncio
import json
import hashlib
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.deps import get_db_connection, get_neo4j_session
from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.react.controller import (
    ControllerConfig,
    parse_validate_sql,
    run_controller,
    triage_no_acceptable_sql,
)
from app.react.state import ReactSessionState
from app.react.tool_result_metadata_merge import (
    merge_build_sql_context_tool_result_into_metadata,
    merge_build_sql_context_tool_results,
    parse_build_sql_context_tool_result,
)
from app.react.conversation_capsule import (
    append_completed_turn_to_capsule,
    build_conversation_context,
    decode_conversation_state,
    encode_conversation_state,
    new_capsule,
)
from app.react.tools import ToolContext, execute_tool
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger
from app.core.text2sql_runtime_repair_jobs import (
    maybe_trigger_text2sql_runtime_repair,
    run_text2sql_runtime_repair_blocking,
)


router = APIRouter(prefix="/react", tags=["ReAct"])


def _to_cdata(value: str) -> str:
    return f"<![CDATA[{value}]]>"


def _new_react_run_id() -> str:
    return f"react_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _has_build_sql_context(tool_result_xml: Optional[str]) -> bool:
    """
    Best-effort check for whether current_tool_result contains build_sql_context output.
    We prefer substring check (fast/robust) over strict XML parsing.
    """
    s = (tool_result_xml or "").strip()
    if not s:
        return False
    return "<build_sql_context_result" in s


def _norm_ws_key(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _extract_light_query_sqls(tool_result_xml: Optional[str], *, max_items: int = 200) -> List[str]:
    """
    Extract previously executed light_queries SQL strings from a build_sql_context tool_result XML.
    Used to avoid duplicate light_queries during context-refresh loops.
    """
    s = (tool_result_xml or "").strip()
    if not s:
        return []
    st = s.find("<light_queries>")
    en = s.find("</light_queries>")
    if st < 0 or en < 0 or en <= st:
        return []
    section = s[st : en + len("</light_queries>")]
    out: List[str] = []
    # Prefer CDATA form
    for m in re.finditer(r"<sql>\s*<!\[CDATA\[(.*?)\]\]>\s*</sql>", section, flags=re.IGNORECASE | re.DOTALL):
        sql = (m.group(1) or "").strip()
        if sql:
            out.append(sql)
            if len(out) >= max_items:
                return out
    # Fallback: plain <sql>...</sql>
    if not out:
        for m in re.finditer(r"<sql>([\s\S]*?)</sql>", section, flags=re.IGNORECASE):
            sql_raw = (m.group(1) or "").strip()
            sql = sql_raw.replace("<![CDATA[", "").replace("]]>", "").strip()
            if sql:
                out.append(sql)
                if len(out) >= max_items:
                    break
    return out


def _context_signature(tool_result_xml: str) -> Dict[str, Any]:
    """
    Best-effort signature for stop conditions.
    Prefer structured parsing; fall back to hashing key XML blocks on parse errors.
    """
    s = (tool_result_xml or "").strip()
    if not s:
        return {"hash": "", "tables": set(), "values": set(), "relationships": set(), "parse_mode": "empty"}

    parsed = parse_build_sql_context_tool_result(s, description_limit=120)
    tables = set()
    values = set()
    rels = set()
    if parsed.tables or parsed.values or parsed.relationships:
        for t in parsed.tables:
            schema = (t.get("schema") or "").strip()
            name = (t.get("name") or "").strip()
            if name:
                tables.add(f"{schema}.{name}".strip(".").lower())
        for v in parsed.values:
            schema = (v.get("schema") or "").strip()
            table = (v.get("table") or "").strip()
            col = (v.get("column") or "").strip()
            actual = (v.get("actual_value") or "").strip()
            if table and col and actual:
                values.add(f"{schema}.{table}.{col}={actual}".strip(".").lower())
        for r in parsed.relationships:
            cond = (r.get("condition") or "").strip().lower()
            if cond:
                rels.add(cond[:300])
        blob = "\n".join(sorted(list(tables)[:400]) + sorted(list(values)[:400]) + sorted(list(rels)[:400]))
        h = hashlib.sha1(blob.encode("utf-8")).hexdigest()
        return {"hash": h, "tables": tables, "values": values, "relationships": rels, "parse_mode": "structured"}

    # Fallback: hash high-signal blocks by substring (fast/robust).
    def _slice(tag: str) -> str:
        st = s.find(f"<{tag}>")
        en = s.find(f"</{tag}>")
        if st < 0 or en < 0 or en <= st:
            return ""
        return s[st : en + len(f"</{tag}>")]

    blob2 = "\n".join([_slice("schema_candidates"), _slice("column_value_hints"), _slice("resolved_values"), _slice("fk_relationships")])
    h2 = hashlib.sha1(blob2.encode("utf-8")).hexdigest()
    return {"hash": h2, "tables": set(), "values": set(), "relationships": set(), "parse_mode": "substring"}


def _enrich_question_for_context_build(
    question: str, *, conversation_context: Optional[Dict[str, Any]]
) -> str:
    """
    For under-specified follow-up questions like "방금 결과에서...",
    inject compact, high-signal hints from the conversation capsule.

    Keep this short to avoid ballooning embedding/HyDE costs.
    """
    base = (question or "").strip()
    if not base:
        return ""
    ctx = conversation_context if isinstance(conversation_context, dict) else None
    if not ctx:
        return base
    turns = ctx.get("turns")
    if not isinstance(turns, list) or not turns:
        return base
    last = turns[-1] if isinstance(turns[-1], dict) else None
    if not last:
        return base

    derived = last.get("derived_filters")
    if isinstance(derived, dict) and derived:
        pairs: List[str] = []
        for k, v in list(derived.items())[:16]:
            kk = str(k).strip()
            vv = str(v).strip()
            if kk and vv:
                pairs.append(f"{kk}={vv}")
        if pairs:
            hint = ", ".join(pairs)[:800]
            base = f"{base}\n\n[이전 결과에서 추출된 필터]\n{hint}".strip()
    return base


def _strip_trailing_order_by_and_limit(sql: str) -> str:
    """
    Best-effort removal of trailing ORDER BY / LIMIT clauses (top-level).
    Intended for simple, router-generated SQL patterns (not a general SQL parser).
    """
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return ""
    # Remove trailing LIMIT n
    s2 = re.sub(r"(?is)\blimit\s+\d+\s*$", "", s).strip().rstrip(";").strip()
    # Remove trailing ORDER BY ... (if present at very end)
    s3 = re.sub(r"(?is)\border\s+by\b[\s\S]*$", "", s2).strip().rstrip(";").strip()
    # If we removed too much (empty), fall back to LIMIT-removed version.
    return s3 or s2 or s


def _maybe_followup_transform_sql(
    *,
    followup_question: str,
    conversation_context: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Deterministic follow-up SQL transform for common "modify previous result" requests.

    Current target:
    - "최근 7일" + "전일 대비" + "증감" patterns (usecase_1.1)

    Strategy:
    - Use last turn's final_sql as a stable "daily series" base (remove ORDER BY/LIMIT)
    - IMPORTANT: Phase 1 runs through MindsDB(MySQL endpoint). MindsDB's MySQL parser frequently
      rejects top-level CTE/window-function SQL (e.g., WITH / LAG ... OVER).
    - Therefore, we build a *passthrough* query:
        SELECT * FROM <datasource> ( <inner_sql_in_external_db_dialect> )
      so that the complex SQL is parsed/executed by the external DB (e.g., PostgreSQL),
      not by MindsDB's MySQL parser.
    """
    q = (followup_question or "").strip()
    ctx = conversation_context if isinstance(conversation_context, dict) else None
    if not q or not ctx:
        return None
    turns = ctx.get("turns")
    if not isinstance(turns, list) or not turns:
        return None

    qq = q.replace(" ", "")
    wants_last7 = ("최근7일" in qq) or bool(re.search(r"최근\s*\d+\s*일", q))
    wants_dod = ("전일대비" in qq) or ("전날대비" in qq) or ("전일" in qq)
    wants_delta = ("증감" in qq) or ("증가" in qq) or ("감소" in qq)
    wants_percent = ("퍼센" in qq) or ("%" in qq) or ("percent" in qq.lower())
    if not (wants_last7 and wants_dod and wants_delta):
        return None

    last = turns[-1] if isinstance(turns[-1], dict) else None
    if not last:
        return None
    last_sql = str(last.get("final_sql") or "").strip()
    if not last_sql:
        return None

    base_daily = _strip_trailing_order_by_and_limit(last_sql)
    if not base_daily:
        return None

    def _strip_ident(ident: str) -> str:
        s = (ident or "").strip()
        if not s:
            return ""
        if (s.startswith("`") and s.endswith("`")) or (s.startswith('"') and s.endswith('"')):
            s = s[1:-1]
        return s.strip()

    def _pg_quote(ident: str) -> str:
        # Best-effort: quote as PostgreSQL identifier.
        x = (ident or "").strip()
        if not x:
            return ""
        return '"' + x.replace('"', '""') + '"'

    def _detect_datasource(sql: str) -> str:
        s = str(sql or "")
        # e.g., FROM postgresql.`RWIS`.`RDITAG_TB` ...
        m = re.search(r"(?is)\bfrom\s+([A-Za-z_][A-Za-z0-9_]*)\s*\.", s)
        if m:
            return str(m.group(1) or "").strip()
        m = re.search(r"(?is)\bjoin\s+([A-Za-z_][A-Za-z0-9_]*)\s*\.", s)
        if m:
            return str(m.group(1) or "").strip()
        return ""

    def _mindsdb_sql_to_postgres(sql: str, datasource: str) -> str:
        """
        Convert a MindsDB-addressed SQL snippet into PostgreSQL dialect:
        - Remove datasource prefix: datasource.<schema>.<table> -> <schema>.<table>
        - Convert MySQL backticks around identifiers to PostgreSQL double quotes
        - Keep single-quoted string literals intact
        """
        inner = str(sql or "")
        ds = (datasource or "").strip()
        if not inner.strip() or not ds:
            return inner.strip()

        # Protect single-quoted string literals (e.g., '청주정수장')
        placeholders: Dict[str, str] = {}
        counter = 0

        def _protect_str(m: re.Match) -> str:
            nonlocal counter
            key = f"__xstrx_{counter}__"
            counter += 1
            placeholders[key] = m.group(0)
            return key

        tmp = re.sub(r"'(?:[^']|'')*'", _protect_str, inner)

        # Remove datasource prefix at identifier boundaries: postgresql.`RWIS`... / postgresql.RWIS...
        tmp = re.sub(rf"(?i)\b{re.escape(ds)}\s*\.", "", tmp)
        # Backticks -> double quotes (identifiers)
        tmp = re.sub(r"`([^`]*)`", r'"\1"', tmp)

        for k, v in placeholders.items():
            tmp = tmp.replace(k, v)
        return tmp.strip()

    datasource = _detect_datasource(last_sql)
    if not datasource:
        # Without a datasource, we cannot build a passthrough query.
        return None

    base_daily_pg = _mindsdb_sql_to_postgres(base_daily, datasource)

    # Heuristic column names from usecase_1 output
    date_col = "`LOG_TIME`" if "`LOG_TIME`" in last_sql else "LOG_TIME"
    tagsn_col = "`TAGSN`" if "`TAGSN`" in last_sql else ""
    suj_name_col = "`SUJ_NAME`" if "`SUJ_NAME`" in last_sql else ""
    metric_col = "`AVG_FLOW_RATE`" if "`AVG_FLOW_RATE`" in last_sql else ""
    if not metric_col:
        # Try to find an alias like AS `SOMETHING`
        m = re.search(r"(?is)\bas\s+`([A-Za-z0-9_]+)`", last_sql)
        if m:
            metric_col = f"`{m.group(1)}`"
    if not metric_col:
        return None

    # Use a stable ordering column for DoD; if date_col is not found, bail out.
    if not date_col:
        return None

    date_name = _strip_ident(date_col)
    tagsn_name = _strip_ident(tagsn_col) if tagsn_col else ""
    suj_name_name = _strip_ident(suj_name_col) if suj_name_col else ""
    metric_name = _strip_ident(metric_col)
    if not date_name or not metric_name:
        return None

    date_pg = _pg_quote(date_name)
    tagsn_pg = _pg_quote(tagsn_name) if tagsn_name else ""
    suj_name_pg = _pg_quote(suj_name_name) if suj_name_name else ""
    metric_pg = _pg_quote(metric_name)

    # Build select list for final output (PostgreSQL dialect)
    id_cols_pg: List[str] = [date_pg]
    if tagsn_pg:
        id_cols_pg.append(tagsn_pg)
    if suj_name_pg:
        id_cols_pg.append(suj_name_pg)
    id_cols_pg.append(metric_pg)

    part = f"PARTITION BY {tagsn_pg} " if tagsn_pg else ""

    pct_expr = (
        f"CASE WHEN \"PREV_VALUE\" IS NULL OR \"PREV_VALUE\" = 0 THEN NULL "
        f"ELSE (({metric_pg} - \"PREV_VALUE\") / \"PREV_VALUE\") * 100 END AS \"DOD_DELTA_PCT\""
        if wants_percent
        else f"CASE WHEN \"PREV_VALUE\" IS NULL OR \"PREV_VALUE\" = 0 THEN NULL "
        f"ELSE (({metric_pg} - \"PREV_VALUE\") / \"PREV_VALUE\") END AS \"DOD_DELTA_PCT\""
    )

    inner = f"""
WITH daily AS (
  {base_daily_pg}
),
w AS (
  SELECT
    {", ".join(id_cols_pg)},
    LAG({metric_pg}) OVER ({part}ORDER BY {date_pg}) AS "PREV_VALUE"
  FROM daily
),
last7 AS (
  SELECT *
  FROM w
  ORDER BY {date_pg} DESC
  LIMIT 7
)
SELECT
  {", ".join(id_cols_pg)},
  ({metric_pg} - "PREV_VALUE") AS "DOD_DELTA_ABS",
  {pct_expr}
FROM last7
ORDER BY {date_pg} DESC
""".strip()

    # MindsDB passthrough: keep the inner SQL in external DB dialect (PostgreSQL).
    # Quote datasource for MindsDB(MySQL) compatibility: FROM `postgresql` ( ... )
    transformed = f"SELECT * FROM `{datasource}` (\n{inner}\n)"
    return transformed.strip()


def _context_delta(prev_sig: Dict[str, Any], new_sig: Dict[str, Any]) -> Dict[str, int]:
    prev_tables = prev_sig.get("tables") if isinstance(prev_sig.get("tables"), set) else set()
    prev_vals = prev_sig.get("values") if isinstance(prev_sig.get("values"), set) else set()
    prev_rels = prev_sig.get("relationships") if isinstance(prev_sig.get("relationships"), set) else set()
    new_tables = new_sig.get("tables") if isinstance(new_sig.get("tables"), set) else set()
    new_vals = new_sig.get("values") if isinstance(new_sig.get("values"), set) else set()
    new_rels = new_sig.get("relationships") if isinstance(new_sig.get("relationships"), set) else set()
    return {
        "tables_added": int(len(new_tables - prev_tables)) if (new_tables and prev_tables) else 0,
        "values_added": int(len(new_vals - prev_vals)) if (new_vals and prev_vals) else 0,
        "relationships_added": int(len(new_rels - prev_rels)) if (new_rels and prev_rels) else 0,
    }


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


class QueryQualityGateRoundModel(BaseModel):
    accept: bool
    confidence: float
    reasons: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    summary: str = ""


class QueryQualityGateModel(BaseModel):
    """
    Query 품질 게이트 요약 (저장 게이트와 동일 정책/임계값 사용).
    - ok=true 이면 "자동 검증 통과"로 간주하고 UI 피드백 버튼을 숨긴다.
    - ok=false 이면 "자동 검증 실패"로 간주하고 UI 피드백 버튼을 노출한다.
    """

    policy: str = "llm_judge_2x"
    threshold: float
    rounds: int
    ok: bool
    verified_confidence: Optional[float] = None
    verified_confidence_avg: Optional[float] = None
    results: List[QueryQualityGateRoundModel] = Field(default_factory=list)
    error: Optional[str] = None


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
    conversation_state: Optional[str] = Field(
        default=None,
        description="completed 이후 follow-up 연속질문을 위한 대화 컨텍스트 토큰(클라이언트 보관)",
    )
    question_to_user: Optional[str] = None
    warnings: Optional[List[str]] = None
    from_cache: bool = Field(default=False, description="캐시 히트로 반환된 응답인지 여부")
    quality_gate: Optional[QueryQualityGateModel] = Field(
        default=None, description="저장 게이트(LLM judge) 판정 결과"
    )
    feedback_required: bool = Field(
        default=False, description="자동 검증 실패 시 UI 피드백 버튼 노출"
    )


class ReactRequest(BaseModel):
    question: str = Field(..., description="사용자 자연어 질문")
    datasource: str = Field(..., description="MindsDB datasource (required; Phase 1 MindsDB-only)")
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
    prefetch_build_sql_context: bool = Field(
        default=True,
        description="새 세션 시작 시 build_sql_context를 선행(prefetch)할지 여부",
    )
    use_cache: bool = Field(
        default=True, description="동일한 질문에 대해 캐시된 결과를 사용할지 여부"
    )
    conversation_state: Optional[str] = Field(
        default=None,
        description="이전 completed 응답에서 받은 conversation_state 토큰(연속질문 컨텍스트)",
    )
    schema_filter: Optional[List[str]] = Field(
        default=None, description="검색 대상 스키마 제한 (예: ['dw']는 OLAP 데이터만 검색)"
    )


def _preview_for_query_quality_gate(
    execution_result: Optional[ExecutionResultModel],
    *,
    max_rows: int = 3,
    max_cols: int = 20,
) -> Optional[Dict[str, Any]]:
    if execution_result is None:
        return None
    cols = list(execution_result.columns or [])[: max(1, int(max_cols))]
    out_rows: List[List[Any]] = []
    for row in list(execution_result.rows or [])[: max(1, int(max_rows))]:
        r = list(row or [])[: len(cols)]
        out_rows.append([str(v) if v is not None else None for v in r])
    return {
        "columns": cols,
        "rows": out_rows,
        "row_count": int(execution_result.row_count),
    }


async def _run_query_quality_gate_for_ui(
    *,
    question: str,
    sql: str,
    execution_result: ExecutionResultModel,
    metadata: Dict[str, Any],
    steps: List[ReactStepModel],
) -> QueryQualityGateModel:
    """
    UI 피드백 노출 제어용 Query 품질 게이트.
    - cache_postprocess의 judge와 동일한 설정/프롬프트를 사용한다.
    - ok=true => 피드백 버튼 숨김
    - ok=false => 피드백 버튼 노출 (사용자 부담은 줄이되, 자동 valid 실패일 때만 요청)
    """

    judge_enabled = bool(getattr(settings, "cache_postprocess_query_quality_gate_enabled", True))
    judge_rounds = max(1, int(getattr(settings, "cache_postprocess_query_judge_rounds", 2) or 2))
    judge_threshold = float(getattr(settings, "cache_postprocess_query_judge_conf_threshold", 0.90) or 0.90)
    min_row_count = int(getattr(settings, "cache_postprocess_query_min_row_count", 1) or 1)

    s = (sql or "").strip()
    if not judge_enabled:
        return QueryQualityGateModel(
            threshold=float(judge_threshold),
            rounds=int(judge_rounds),
            ok=True,
            verified_confidence=None,
            results=[],
            error="judge_disabled",
        )
    if not s:
        return QueryQualityGateModel(
            threshold=float(judge_threshold),
            rounds=int(judge_rounds),
            ok=False,
            verified_confidence=None,
            results=[],
            error="empty_sql",
        )

    row_count = int(execution_result.row_count)
    if row_count < min_row_count:
        return QueryQualityGateModel(
            threshold=float(judge_threshold),
            rounds=int(judge_rounds),
            ok=False,
            verified_confidence=None,
            results=[],
            error=f"min_row_count_failed(row_count={row_count},min={min_row_count})",
        )

    # Reuse the exact steps-tail compactor from cache_postprocess,
    # and the unified query-quality judge generator (structured output).
    from app.core.cache_postprocess import _compact_steps_for_judge
    from app.react.generators.query_quality_gate_generator import get_query_quality_gate_generator

    steps_tail = _compact_steps_for_judge([st.model_dump() for st in (steps or [])], limit=6)
    preview = _preview_for_query_quality_gate(execution_result, max_rows=3, max_cols=20)

    rounds: List[QueryQualityGateRoundModel] = []
    confs: List[float] = []
    qgen = get_query_quality_gate_generator()
    for ridx in range(1, judge_rounds + 1):
        jr = await qgen.judge_round(
            question=(question or "").strip(),
            sql=s,
            row_count=row_count,
            execution_time_ms=float(execution_result.execution_time_ms),
            metadata=metadata or {},
            steps_tail=steps_tail,
            preview=preview,
            round_idx=int(ridx),
            purpose="react.query_quality_gate",
        )
        rounds.append(
            QueryQualityGateRoundModel(
                accept=bool(jr.accept),
                confidence=float(jr.confidence),
                reasons=list(jr.reasons or [])[:12],
                risk_flags=list(jr.risk_flags or [])[:12],
                summary=str(jr.summary or ""),
            )
        )
        confs.append(float(jr.confidence))

    ok = all((r.accept and float(r.confidence) >= float(judge_threshold)) for r in rounds)
    verified_conf = min(confs) if confs else None
    verified_conf_avg = (sum(confs) / len(confs)) if confs else None
    return QueryQualityGateModel(
        threshold=float(judge_threshold),
        rounds=int(judge_rounds),
        ok=bool(ok),
        verified_confidence=float(verified_conf) if verified_conf is not None else None,
        verified_confidence_avg=float(verified_conf_avg) if verified_conf_avg is not None else None,
        results=rounds,
        error=None,
    )


def _controller_attempt_to_step_model(*, iteration: int, reasoning: str, sql: str, tool_result_xml: str, metadata_xml: str) -> ReactStepModel:
    # Controller pipeline emits validate_sql attempts; model them as a step for frontend compatibility.
    return ReactStepModel(
        iteration=iteration,
        reasoning=reasoning,
        metadata_xml=metadata_xml,
        partial_sql=sql,
        sql_completeness=SQLCompletenessModel(
            is_complete=False,
            missing_info="",
            confidence_level="low",
        ),
        tool_call=ToolCallModel(
            name="validate_sql",
            raw_parameters_xml="<parameters/>",
            parameters={"sql": sql},
        ),
        tool_result=tool_result_xml or None,
        llm_output="",
    )


def _business_missing_info_for_attempt(
    *,
    verdict: str,
    score: float,
    score_threshold: float,
    preview: Optional[Dict[str, Any]] = None,
    fail_reason: str = "",
) -> str:
    """
    B-1: Provide business-oriented feedback even when validate_sql is PASS but the attempt
    is not acceptable (e.g., empty preview / low score).
    """
    v = (verdict or "").strip().upper()
    if v != "PASS":
        # Keep it non-technical by default.
        return (
            "현재 조건으로는 실행 가능한 질의를 확정하지 못했습니다. "
            "시스템이 자동으로 문맥을 보강하고 다시 시도합니다."
        )

    try:
        sc = float(score or 0.0)
    except Exception:
        sc = 0.0
    try:
        th = float(score_threshold or 0.0)
    except Exception:
        th = 0.0
    if sc >= th:
        return ""

    rc = None
    if isinstance(preview, dict):
        rc = preview.get("row_count")
    try:
        if isinstance(rc, str) and rc.strip():
            rc = int(rc.strip())
    except Exception:
        rc = None

    if isinstance(rc, int) and rc <= 0:
        return (
            "조회 결과가 비어 있어 대상(예: 정수장)이나 지표(예: 유량) 식별이 정확하지 않을 수 있습니다. "
            "시스템이 자동으로 대상 식별을 보강하고 다시 시도합니다."
        )

    # PASS but still not acceptable: likely semantic mismatch (wrong grain/metric/entity).
    return (
        "조회는 되었지만 질문 의도와 정확히 일치하는 결과인지 확신이 부족합니다. "
        "시스템이 자동으로 조건을 조정해 다시 시도합니다."
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
) -> StreamingResponse:
    from app.core.query_cache import get_query_cache
    
    react_run_id = _new_react_run_id()
    api_started = time.perf_counter()
    runtime_repair_status = "unknown"
    runtime_repair_check: Optional[Dict[str, Any]] = None
    cold_start_blocking_needed = False

    # Runtime self-heal trigger (throttled):
    # If graph was reinitialized while server is already running, newly created Table/Column nodes
    # can miss text_to_sql_* properties. Detect quickly and repair in background.
    try:
        runtime_repair = await maybe_trigger_text2sql_runtime_repair(
            neo4j_session=neo4j_session,
            source="react.run_react",
        )
        runtime_repair_status = str(runtime_repair.get("status") or "")
        if isinstance(runtime_repair.get("check"), dict):
            runtime_repair_check = dict(runtime_repair.get("check") or {})
        cold_start_blocking_needed = runtime_repair_status == "cold_start_detected"
        if runtime_repair_status in {"triggered", "already_running", "cold_start_detected"}:
            SmartLogger.log(
                "INFO",
                "react.runtime_repair.active",
                category="react.runtime_repair",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "status": runtime_repair_status,
                        "check": runtime_repair_check,
                    }
                ),
                max_inline_chars=0,
            )
    except Exception as runtime_repair_exc:
        SmartLogger.log(
            "WARNING",
            "react.runtime_repair.check_failed",
            category="react.runtime_repair",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "error": repr(runtime_repair_exc),
                }
            ),
            max_inline_chars=0,
        )
    
    # 캐시 확인 (새 세션이고 use_cache=True인 경우에만)
    # NOTE: follow-up 연속질문(conversation_state)에서는 캐시를 사용하지 않는다.
    if (
        request.use_cache
        and not request.session_state
        and not request.user_response
        and not request.conversation_state
        and not cold_start_blocking_needed
    ):
        cache = get_query_cache()
        cached = cache.get(request.question, datasource=request.datasource)
        
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
                    # 정책: 캐시 히트/히스토리에서는 피드백 버튼 노출 금지
                    "feedback_required": False,
                    "quality_gate": None,
                }
                # Cache hit에서도 conversation_state는 발급(후속질문 연속성 유지; evidence_ctx_xml은 비움)
                try:
                    cap = new_capsule(dbms=str(request.dbms or settings.target_db_type), schema_filter=request.schema_filter)
                    cap = append_completed_turn_to_capsule(
                        cap,
                        question=request.question,
                        final_sql=str(cached.final_sql or ""),
                        execution_result=cached.execution_result if isinstance(cached.execution_result, dict) else None,
                        collected_metadata_xml="",
                        build_sql_context_xml="",
                    )
                    response_obj["conversation_state"] = encode_conversation_state(cap)
                except Exception:
                    response_obj["conversation_state"] = None
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
        datasource=request.datasource,
        react_run_id=react_run_id,
        max_sql_seconds=state.max_sql_seconds,
        schema_filter=request.schema_filter,
    ).with_overrides(
        # Controller path prefers a stronger context build for stability.
        table_rerank_top_k=60,
        table_rerank_fetch_k=200,
        table_relation_limit=40,
        column_relation_limit=20,
        schema_filter=request.schema_filter,
    )
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
        async def _stream_build_sql_context_with_progress(*, parameters: Dict[str, Any]):
            """
            Execute build_sql_context while interleaving internal progress events.

            - Tool internals emit `pipeline_stage` / `pipeline_item` via ToolContext.emit()
              which routes into this queue.
            - This async generator yields those progress events in real time (best-effort),
              and finally yields a terminal marker: {"event":"__tool_result__","tool_result": "..."}.
            """
            q: asyncio.Queue = asyncio.Queue(maxsize=1000)

            def _emit(ev: Dict[str, Any]) -> None:
                try:
                    q.put_nowait(ev)
                except Exception:
                    # Drop on overflow/closed queue - best effort only.
                    return

            ctx_stream = tool_context.with_overrides(stream_emit=_emit)
            task = asyncio.create_task(
                execute_tool(
                    tool_name="build_sql_context",
                    context=ctx_stream,
                    parameters=dict(parameters),
                )
            )

            # Drain queue while the tool runs
            while True:
                if task.done():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.25)
                    if isinstance(ev, dict) and ev.get("event"):
                        yield ev
                except asyncio.TimeoutError:
                    # No progress event yet - keep waiting
                    pass

            # Task finished: drain remaining events
            try:
                while True:
                    ev = q.get_nowait()
                    if isinstance(ev, dict) and ev.get("event"):
                        yield ev
            except Exception:
                pass

            # Return tool result via terminal marker
            result = await task
            yield {"event": "__tool_result__", "tool_result": result}

        async def _stream_run_controller_with_progress(
            *,
            question: str,
            dbms: str,
            max_sql_seconds: int,
            config: ControllerConfig,
            prebuilt_context_xml: Optional[str],
            conversation_context: Optional[Dict[str, Any]],
        ):
            """
            Execute run_controller while interleaving controller progress events.

            - Controller internals emit `pipeline_stage` / `pipeline_item` via ToolContext.emit()
              which routes into this queue.
            - This async generator yields those progress events in real time (best-effort),
              and finally yields a terminal marker: {"event":"__controller_result__","result": ...}.
            - Adds a lightweight 1s heartbeat for the current running stage to avoid “stuck” UX
              when a single LLM call takes long.
            """
            q: asyncio.Queue = asyncio.Queue(maxsize=2000)

            def _emit(ev: Dict[str, Any]) -> None:
                try:
                    q.put_nowait(ev)
                except Exception:
                    return

            ctx_stream = tool_context.with_overrides(stream_emit=_emit)
            task = asyncio.create_task(
                run_controller(
                    question=question,
                    dbms=dbms,
                    tool_context=ctx_stream,
                    max_sql_seconds=max_sql_seconds,
                    config=config,
                    prebuilt_context_xml=prebuilt_context_xml,
                    conversation_context=conversation_context,
                )
            )

            current_stage_name: str = ""
            current_stage_seq: int = 0
            current_stage_started_ms: Optional[int] = None
            last_heartbeat_at = time.perf_counter()

            # Drain queue while controller runs
            while True:
                if task.done():
                    break
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.25)
                    if isinstance(ev, dict) and ev.get("event"):
                        # Track current stage (for heartbeat)
                        if (
                            ev.get("event") == "pipeline_stage"
                            and str(ev.get("pipeline") or "") == "controller"
                            and str(ev.get("status") or "") == "start"
                        ):
                            current_stage_name = str(ev.get("stage") or "")
                            try:
                                current_stage_seq = int(ev.get("seq") or 0)
                            except Exception:
                                current_stage_seq = 0
                            try:
                                current_stage_started_ms = int(ev.get("ts_ms")) if ev.get("ts_ms") is not None else None
                            except Exception:
                                current_stage_started_ms = None
                            last_heartbeat_at = time.perf_counter()
                        yield ev
                except asyncio.TimeoutError:
                    # Heartbeat: emit "start" update every ~1s for the current stage.
                    if current_stage_name and current_stage_started_ms is not None:
                        now = time.perf_counter()
                        if (now - last_heartbeat_at) >= 1.0:
                            ts_ms = int(time.time() * 1000)
                            yield {
                                "event": "pipeline_stage",
                                "pipeline": "controller",
                                "stage": current_stage_name,
                                "status": "start",
                                "seq": int(current_stage_seq),
                                "iteration": 0,
                                "ts_ms": ts_ms,
                                "elapsed_ms": float(ts_ms - int(current_stage_started_ms)),
                            }
                            last_heartbeat_at = now
                    pass

            # Task finished: drain remaining events
            try:
                while True:
                    ev = q.get_nowait()
                    if isinstance(ev, dict) and ev.get("event"):
                        yield ev
            except Exception:
                pass

            result = await task
            yield {"event": "__controller_result__", "result": result}

        try:
            # Cold-start mode:
            # - If text_to_sql vectors are missing for all tables, block this request until
            #   runtime bootstrap/repair is completed, and stream progress to user.
            if cold_start_blocking_needed and isinstance(runtime_repair_check, dict):
                cold_started = time.perf_counter()
                ts_ms = int(time.time() * 1000)
                yield json.dumps(
                    {
                        "event": "pipeline_stage",
                        "pipeline": "runtime_repair",
                        "stage": "runtime_repair_cold_start",
                        "status": "start",
                        "seq": 0,
                        "iteration": 0,
                        "ts_ms": ts_ms,
                        "notes": [
                            "cold_start_detected",
                            "Text2SQL 속성 초기화 진행 중",
                        ],
                    },
                    ensure_ascii=False,
                ) + "\n"

                blocking_result = await run_text2sql_runtime_repair_blocking(
                    source="react.run_react.cold_start_blocking",
                    check=runtime_repair_check,
                    db_conn=db_conn,
                )
                repair = (
                    blocking_result.get("repair")
                    if isinstance(blocking_result, dict) and isinstance(blocking_result.get("repair"), dict)
                    else {}
                )
                after_check = repair.get("after") if isinstance(repair.get("after"), dict) else {}
                counts = {
                    "table_missing_vector": int(after_check.get("table_missing_vector") or 0),
                    "table_missing_embedding_text": int(after_check.get("table_missing_embedding_text") or 0),
                    "column_missing_is_valid": int(after_check.get("column_missing_is_valid") or 0),
                    "column_missing_db_exists": int(after_check.get("column_missing_db_exists") or 0),
                }
                elapsed_ms = (time.perf_counter() - cold_started) * 1000.0
                repair_ok = bool(repair.get("ok"))
                if repair_ok:
                    yield json.dumps(
                        {
                            "event": "pipeline_stage",
                            "pipeline": "runtime_repair",
                            "stage": "runtime_repair_cold_start",
                            "status": "done",
                            "seq": 0,
                            "iteration": 0,
                            "ts_ms": int(time.time() * 1000),
                            "elapsed_ms": elapsed_ms,
                            "counts": counts,
                        },
                        ensure_ascii=False,
                    ) + "\n"
                else:
                    err_msg = str(repair.get("error") or "cold start runtime repair failed")
                    warnings.append(f"cold_start_runtime_repair failed: {err_msg}")
                    yield json.dumps(
                        {
                            "event": "pipeline_stage",
                            "pipeline": "runtime_repair",
                            "stage": "runtime_repair_cold_start",
                            "status": "error",
                            "seq": 0,
                            "iteration": 0,
                            "ts_ms": int(time.time() * 1000),
                            "elapsed_ms": elapsed_ms,
                            "counts": counts,
                            "error": err_msg,
                        },
                        ensure_ascii=False,
                    ) + "\n"

            # Follow-up conversation capsule (client token). Used for:
            # - candidate LLM payload: conversation_context
            # - optional evidence merge: prior_evidence_xml
            capsule = decode_conversation_state(request.conversation_state)
            if not getattr(capsule, "dbms", ""):
                try:
                    capsule.dbms = state.dbms
                except Exception:
                    pass
            if getattr(capsule, "schema_filter", None) is None and request.schema_filter:
                try:
                    capsule.schema_filter = list(request.schema_filter)
                except Exception:
                    pass

            conversation_context: Optional[Dict[str, Any]] = None
            prior_evidence_xml: str = ""
            try:
                if getattr(capsule, "turns", None):
                    conversation_context = build_conversation_context(
                        capsule,
                        followup_question=request.question,
                        recent_n=5,
                        relevant_m=3,
                        max_preview_cols=12,
                        max_preview_rows=3,
                    )
                    prior_evidence_xml = (capsule.turns[-1].evidence_ctx_xml or "").strip()  # type: ignore[attr-defined]
            except Exception:
                conversation_context = None
                prior_evidence_xml = ""

            # Follow-up continuity: seed current_tool_result with prior evidence (best-effort).
            # This prevents an expensive/ambiguous build_sql_context call on under-specified follow-up questions
            # like "방금 결과에서..." and helps avoid "infinite loading" when the tool/LLM is slow.
            try:
                # ReactSessionState.new() initializes current_tool_result with "<tool_result/>"
                # which is non-empty but carries no build_sql_context evidence. Use a structural check.
                if prior_evidence_xml and not _has_build_sql_context(state.current_tool_result):
                    state.current_tool_result = prior_evidence_xml
            except Exception:
                pass

            # Fast-path for common follow-up transforms ("최근 7일" + "전일 대비 증감").
            # This avoids expensive controller/refresh loops when we can deterministically rewrite
            # the previous turn's SQL and validate it in one tool call.
            try:
                if request.conversation_state and conversation_context:
                    fast_sql = _maybe_followup_transform_sql(
                        followup_question=request.question,
                        conversation_context=conversation_context,
                    )
                else:
                    fast_sql = None
            except Exception:
                fast_sql = None

            if fast_sql:
                if state.remaining_tool_calls <= 0:
                    warnings.append("followup_fastpath skipped: remaining_tool_calls <= 0")
                else:
                    try:
                        # validate_sql (includes preview) — single tool call
                        started_fast = time.perf_counter()
                        tool_result_xml = await execute_tool(
                            tool_name="validate_sql",
                            context=tool_context,
                            parameters={"sql": fast_sql},
                        )
                        state.remaining_tool_calls = max(0, state.remaining_tool_calls - 1)
                        state.iteration += 1

                        parsed = parse_validate_sql(tool_result_xml)
                        verdict = str(parsed.get("verdict") or "").strip().upper()
                        selected_sql = str(parsed.get("selected_sql") or "").strip() or fast_sql
                        preview = parsed.get("preview") if isinstance(parsed.get("preview"), dict) else {}

                        if verdict == "PASS":
                            # Emit a single step for transparency
                            step_model = ReactStepModel(
                                iteration=int(state.iteration),
                                reasoning="followup_fastpath: rewrite previous SQL (last 7 days + day-over-day delta)",
                                metadata_xml=state.metadata.to_xml(),
                                partial_sql=selected_sql,
                                sql_completeness=SQLCompletenessModel(
                                    is_complete=True,
                                    missing_info="",
                                    confidence_level="high",
                                ),
                                tool_call=ToolCallModel(
                                    name="validate_sql",
                                    raw_parameters_xml=f"<sql>{_to_cdata(selected_sql)}</sql>",
                                    parameters={"sql": selected_sql},
                                ),
                                tool_result=tool_result_xml,
                                llm_output="",
                            )
                            steps.append(step_model)
                            yield json.dumps(
                                {"event": "step", "step": step_model.model_dump(), "state": state.to_dict()},
                                ensure_ascii=False,
                            ) + "\n"

                            # Build execution_result from preview (no extra DB execution here)
                            execution_result = None
                            try:
                                execution_result = ExecutionResultModel(
                                    columns=list(preview.get("columns") or []),
                                    rows=list(preview.get("rows") or []),
                                    row_count=int(preview.get("row_count") or 0),
                                    execution_time_ms=float(preview.get("execution_time_ms") or 0),
                                )
                            except Exception:
                                execution_result = None

                            # Build/extend conversation capsule
                            cap = decode_conversation_state(request.conversation_state)
                            if not getattr(cap, "dbms", ""):
                                try:
                                    cap.dbms = state.dbms
                                except Exception:
                                    pass
                            cap = append_completed_turn_to_capsule(
                                cap,
                                question=request.question,
                                final_sql=selected_sql,
                                execution_result=execution_result.model_dump() if execution_result else None,
                                collected_metadata_xml=state.metadata.to_xml(),
                                build_sql_context_xml=state.current_tool_result or "",
                            )
                            out_conv_state = encode_conversation_state(cap)

                            response_obj = {
                                "status": "completed",
                                "final_sql": selected_sql,
                                "validated_sql": selected_sql,
                                "execution_result": (execution_result.model_dump() if execution_result else None),
                                "steps": [s.model_dump() for s in steps] if steps else [],
                                "collected_metadata": state.metadata.to_xml(),
                                "partial_sql": selected_sql,
                                "remaining_tool_calls": state.remaining_tool_calls,
                                "from_cache": False,
                                "feedback_required": False,
                                "quality_gate": None,
                                "conversation_state": out_conv_state,
                                "warnings": warnings or None,
                                "elapsed_ms": (time.perf_counter() - started_fast) * 1000.0,
                            }
                            yield json.dumps(
                                {"event": "completed", "response": response_obj, "state": state.to_dict()},
                                ensure_ascii=False,
                            ) + "\n"
                            return
                        else:
                            warnings.append(
                                f"followup_fastpath validate_sql FAIL: {(parsed.get('fail_reason') or '')[:160]}"
                            )
                    except Exception as exc:
                        warnings.append(f"followup_fastpath error: {exc}")

            # Prefetch build_sql_context for NEW sessions to ground the very first LLM call.
            # - Only on fresh question (no session_state, no ask_user user_response).
            # - Skip on follow-up (conversation_state provided): we already have prior evidence + conversation context.
            # - Skip if tool call budget is too small (<=1) to avoid blocking the agent loop.
            if (
                request.prefetch_build_sql_context
                and not request.session_state
                and not request.user_response
                and not request.conversation_state
                and state.iteration == 0
            ):
                if state.remaining_tool_calls <= 1:
                    warnings.append(
                        "prefetch_build_sql_context skipped: remaining_tool_calls <= 1"
                    )
                else:
                    prefetch_started = time.perf_counter()
                    try:
                        tool_result: str = ""
                        async for ev in _stream_build_sql_context_with_progress(
                            parameters={"question": state.user_query}
                        ):
                            if isinstance(ev, dict) and ev.get("event") == "__tool_result__":
                                tool_result = str(ev.get("tool_result") or "")
                                break
                            # Forward progress events to client
                            yield json.dumps(ev, ensure_ascii=False) + "\n"
                        state.current_tool_result = tool_result
                        state.remaining_tool_calls -= 1

                        # Merge prior evidence context (best-effort; no extra tool call)
                        if prior_evidence_xml:
                            try:
                                merged_ctx = merge_build_sql_context_tool_results(
                                    base_tool_result_xml=prior_evidence_xml,
                                    new_tool_result_xml=state.current_tool_result or "",
                                )
                                state.current_tool_result = merged_ctx or state.current_tool_result
                            except Exception:
                                pass

                        # tool_result를 tool-grounded memory로 승격 (중복 없이 merge)
                        try:
                            ms = merge_build_sql_context_tool_result_into_metadata(
                                metadata=state.metadata,
                                tool_result_xml=tool_result,
                                description_limit=600,
                            )
                            SmartLogger.log(
                                "INFO",
                                "react.prefetch.metadata.merge.build_sql_context",
                                category="react.metadata.merge",
                                params=sanitize_for_log(
                                    {
                                        "react_run_id": react_run_id,
                                        "added": ms.added,
                                        "updated": ms.updated,
                                        "skipped": ms.skipped,
                                        "invalid": ms.invalid,
                                    }
                                ),
                            )
                        except Exception as exc:
                            SmartLogger.log(
                                "WARNING",
                                "react.prefetch.metadata.merge.build_sql_context.failed",
                                category="react.metadata.merge",
                                params=sanitize_for_log(
                                    {
                                        "react_run_id": react_run_id,
                                        "error": repr(exc),
                                        "traceback": traceback.format_exc(),
                                    }
                                ),
                                max_inline_chars=0,
                            )
                        elapsed_ms = (time.perf_counter() - prefetch_started) * 1000.0
                        SmartLogger.log(
                            "INFO",
                            "react.prefetch.build_sql_context.done",
                            category="react.prefetch",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": react_run_id,
                                    "elapsed_ms": elapsed_ms,
                                    "remaining_tool_calls": state.remaining_tool_calls,
                                }
                            ),
                        )
                        yield json.dumps(
                            {
                                "event": "prefetch",
                                "tool_name": "build_sql_context",
                                "elapsed_ms": elapsed_ms,
                                "remaining_tool_calls": state.remaining_tool_calls,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    except Exception as exc:
                        elapsed_ms = (time.perf_counter() - prefetch_started) * 1000.0
                        warnings.append(f"prefetch_build_sql_context failed: {exc}")
                        SmartLogger.log(
                            "WARNING",
                            "react.prefetch.build_sql_context.failed",
                            category="react.prefetch",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": react_run_id,
                                    "elapsed_ms": elapsed_ms,
                                    "error": repr(exc),
                                    "traceback": traceback.format_exc(),
                                }
                            ),
                            max_inline_chars=0,
                        )
                        yield json.dumps(
                            {
                                "event": "prefetch_error",
                                "tool_name": "build_sql_context",
                                "elapsed_ms": elapsed_ms,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ) + "\n"

            # Controller pipeline: build_context -> validate/score -> submit OR ask_user.
            # Ensure build_sql_context exists even if prefetch was skipped (e.g., resumed sessions).
            if not _has_build_sql_context(state.current_tool_result):
                if state.remaining_tool_calls <= 0:
                    yield json.dumps(
                        {"event": "error", "message": "max_tool_calls too low: cannot build context"},
                        ensure_ascii=False,
                    ) + "\n"
                    return
                prefetch_started = time.perf_counter()
                tool_result: str = ""
                context_question = _enrich_question_for_context_build(
                    state.user_query, conversation_context=conversation_context
                )
                async for ev in _stream_build_sql_context_with_progress(
                    parameters={"question": context_question}
                ):
                    if isinstance(ev, dict) and ev.get("event") == "__tool_result__":
                        tool_result = str(ev.get("tool_result") or "")
                        break
                    yield json.dumps(ev, ensure_ascii=False) + "\n"
                state.current_tool_result = tool_result
                state.remaining_tool_calls = max(0, state.remaining_tool_calls - 1)
                # Merge prior evidence context (best-effort; no extra tool call)
                if prior_evidence_xml:
                    try:
                        merged_ctx = merge_build_sql_context_tool_results(
                            base_tool_result_xml=prior_evidence_xml,
                            new_tool_result_xml=state.current_tool_result or "",
                        )
                        state.current_tool_result = merged_ctx or state.current_tool_result
                    except Exception:
                        pass
                # tool_result를 tool-grounded memory로 승격 (중복 없이 merge)
                try:
                    ms = merge_build_sql_context_tool_result_into_metadata(
                        metadata=state.metadata,
                        tool_result_xml=tool_result,
                        description_limit=600,
                    )
                    SmartLogger.log(
                        "INFO",
                        "react.metadata.merge.build_sql_context",
                        category="react.metadata.merge",
                        params=sanitize_for_log(
                            {
                                "react_run_id": react_run_id,
                                "added": ms.added,
                                "updated": ms.updated,
                                "skipped": ms.skipped,
                                "invalid": ms.invalid,
                            }
                        ),
                    )
                except Exception as exc:
                    SmartLogger.log(
                        "WARNING",
                        "react.metadata.merge.build_sql_context.failed",
                        category="react.metadata.merge",
                        params=sanitize_for_log(
                            {
                                "react_run_id": react_run_id,
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            }
                        ),
                        max_inline_chars=0,
                    )
                elapsed_ms = (time.perf_counter() - prefetch_started) * 1000.0
                yield json.dumps(
                    {
                        "event": "prefetch",
                        "tool_name": "build_sql_context",
                        "elapsed_ms": elapsed_ms,
                        "remaining_tool_calls": state.remaining_tool_calls,
                    },
                    ensure_ascii=False,
                ) + "\n"

            # Optional debug mode: require confirmation before validate stage.
            if state.step_confirmation_mode and state.iteration == 0 and not request.session_state:
                state_snapshot = ReactSessionState.from_dict(state.to_dict())
                state_snapshot.awaiting_step_confirmation = True
                session_token = state_snapshot.to_token()
                question = "컨트롤러가 SQL 검증/제출 단계를 시작합니다. 다음 스텝으로 진행하시겠습니까?"
                response_payload = ReactResponse(
                    status="await_step_confirmation",
                    steps=steps,
                    collected_metadata=state_snapshot.metadata.to_xml(),
                    partial_sql=state_snapshot.partial_sql,
                    remaining_tool_calls=state_snapshot.remaining_tool_calls,
                    session_state=session_token,
                    conversation_state=request.conversation_state,
                    question_to_user=question,
                    warnings=None,
                )
                yield json.dumps(
                    {"event": "step_confirmation", "response": response_payload.model_dump(), "state": state.to_dict()},
                    ensure_ascii=False,
                ) + "\n"
                return

            effective_question = state.user_query
            if request.user_response:
                effective_question = f"{effective_question}\n\n[사용자 추가 정보]\n{request.user_response}".strip()

                # Refresh build_sql_context using enriched question, then merge into existing context.
                if state.remaining_tool_calls <= 0:
                    warnings.append(
                        "context_refresh skipped: remaining_tool_calls <= 0"
                    )
                else:
                    refresh_started = time.perf_counter()
                    base_ctx = state.current_tool_result
                    try:
                        exclude_light_sqls = _extract_light_query_sqls(base_ctx)
                        new_ctx: str = ""
                        async for ev in _stream_build_sql_context_with_progress(
                            parameters={
                                "question": effective_question,
                                "exclude_light_sqls": exclude_light_sqls,
                            }
                        ):
                            if isinstance(ev, dict) and ev.get("event") == "__tool_result__":
                                new_ctx = str(ev.get("tool_result") or "")
                                break
                            yield json.dumps(ev, ensure_ascii=False) + "\n"
                        state.remaining_tool_calls = max(0, state.remaining_tool_calls - 1)

                        # Merge new tool_result into metadata (idempotent, deduped)
                        try:
                            ms = merge_build_sql_context_tool_result_into_metadata(
                                metadata=state.metadata,
                                tool_result_xml=new_ctx,
                                description_limit=600,
                            )
                            SmartLogger.log(
                                "INFO",
                                "react.context_refresh.metadata.merge.build_sql_context",
                                category="react.metadata.merge",
                                params=sanitize_for_log(
                                    {
                                        "react_run_id": react_run_id,
                                        "added": ms.added,
                                        "updated": ms.updated,
                                        "skipped": ms.skipped,
                                        "invalid": ms.invalid,
                                    }
                                ),
                            )
                        except Exception as exc:
                            SmartLogger.log(
                                "WARNING",
                                "react.context_refresh.metadata.merge.build_sql_context.failed",
                                category="react.metadata.merge",
                                params=sanitize_for_log(
                                    {
                                        "react_run_id": react_run_id,
                                        "error": repr(exc),
                                        "traceback": traceback.format_exc(),
                                    }
                                ),
                                max_inline_chars=0,
                            )

                        # Merge tool_result XML itself for better controller context coverage
                        merged = merge_build_sql_context_tool_results(
                            base_tool_result_xml=base_ctx,
                            new_tool_result_xml=new_ctx,
                        )
                        state.current_tool_result = merged or new_ctx

                        elapsed_ms = (time.perf_counter() - refresh_started) * 1000.0
                        yield json.dumps(
                            {
                                "event": "context_refresh",
                                "tool_name": "build_sql_context",
                                "elapsed_ms": elapsed_ms,
                                "remaining_tool_calls": state.remaining_tool_calls,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                    except Exception as exc:
                        elapsed_ms = (time.perf_counter() - refresh_started) * 1000.0
                        warnings.append(f"context_refresh failed: {exc}")
                        SmartLogger.log(
                            "WARNING",
                            "react.context_refresh.build_sql_context.failed",
                            category="react.context_refresh",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": react_run_id,
                                    "elapsed_ms": elapsed_ms,
                                    "error": repr(exc),
                                    "traceback": traceback.format_exc(),
                                }
                            ),
                            max_inline_chars=0,
                        )
                        yield json.dumps(
                            {
                                "event": "context_refresh_error",
                                "tool_name": "build_sql_context",
                                "elapsed_ms": elapsed_ms,
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ) + "\n"

            # Use a few candidates to reduce brittle failures from a single bad enum/value guess.
            cfg = ControllerConfig(n_candidates=4, score_threshold=0.75, allow_one_revision=True)
            result = None
            async for ev in _stream_run_controller_with_progress(
                question=effective_question,
                dbms=state.dbms,
                max_sql_seconds=state.max_sql_seconds,
                config=cfg,
                prebuilt_context_xml=state.current_tool_result,
                conversation_context=conversation_context,
            ):
                if isinstance(ev, dict) and ev.get("event") == "__controller_result__":
                    result = ev.get("result")
                    break
                yield json.dumps(ev, ensure_ascii=False) + "\n"
            if result is None:
                raise RuntimeError("controller_result_missing")

            # Account validate_sql tool calls
            state.remaining_tool_calls = max(0, state.remaining_tool_calls - len(result.attempts or []))

            for attempt in result.attempts or []:
                state.iteration += 1
                reasoning = (
                    f"controller: validate_sql (candidate={attempt.candidate_index}, "
                    f"score={attempt.score:.2f}, verdict={attempt.verdict})"
                )
                step_model = _controller_attempt_to_step_model(
                    iteration=state.iteration,
                    reasoning=reasoning,
                    sql=attempt.sql,
                    tool_result_xml=attempt.tool_result_xml or "",
                    metadata_xml=state.metadata.to_xml(),
                )
                # Fill completeness based on controller verdict/score
                step_model.sql_completeness = SQLCompletenessModel(
                    is_complete=bool(
                        attempt.verdict == "PASS" and attempt.score >= cfg.score_threshold
                    ),
                    missing_info=_business_missing_info_for_attempt(
                        verdict=str(getattr(attempt, "verdict", "") or ""),
                        score=float(getattr(attempt, "score", 0.0) or 0.0),
                        score_threshold=float(cfg.score_threshold),
                        preview=getattr(attempt, "preview", None) if isinstance(getattr(attempt, "preview", None), dict) else {},
                        fail_reason=str(getattr(attempt, "fail_reason", "") or ""),
                    ),
                    confidence_level=(
                        "high"
                        if (attempt.verdict == "PASS" and float(getattr(attempt, "score", 0.0) or 0.0) >= float(cfg.score_threshold))
                        else ("medium" if attempt.verdict == "PASS" else "low")
                    ),
                )
                steps.append(step_model)
                yield json.dumps(
                    {"event": "step", "step": step_model.model_dump(), "state": state.to_dict()},
                    ensure_ascii=False,
                ) + "\n"

            if result.status == "ask_user":
                # New behavior: single LLM triage + stop-condition-based auto context refresh loop.
                # Goal: avoid asking the user if we can improve context automatically.
                def _is_infra_fail_reason(text: str) -> bool:
                    """
                    Detect infrastructure/connection failures that cannot be fixed by context refresh.
                    (Observed in usecase_2: validate_sql -> "(0, 'Not connected')" causing step explosion.)
                    """
                    t = (text or "").lower()
                    needles = [
                        "not connected",
                        "mcp server not connected",
                        "connection refused",
                        "connection reset",
                        "server has gone away",
                        "lost connection",
                        "timeout",
                    ]
                    return any(n in t for n in needles)

                def _best_pass_attempt(attempts_any: Any) -> Optional[Any]:
                    atts = list(attempts_any or [])
                    passes = [a for a in atts if str(getattr(a, "verdict", "") or "").upper() == "PASS"]
                    if not passes:
                        return None
                    return sorted(passes, key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)[0]

                # Fast-fail: if attempts indicate infra issues, do NOT spend more tool calls refreshing context.
                infra_hits = 0
                for a in list(result.attempts or [])[:12]:
                    fr = str(getattr(a, "fail_reason", "") or "")
                    if fr and _is_infra_fail_reason(fr):
                        infra_hits += 1
                if infra_hits > 0:
                    from app.react.controller import ControllerTriageResult

                    warnings.append("auto_context_refresh skipped: infra failure detected in validate_sql")
                    triage = ControllerTriageResult(
                        decision="ask_user",
                        why=[
                            "validate_sql preview failed due to an infrastructure/connection issue (e.g., MindsDB MySQL endpoint not connected).",
                            "Context refresh will not fix connectivity/parsing layer problems and only increases steps.",
                        ],
                        ask_user_question=(
                            "현재 MindsDB(MySQL 엔드포인트) 연결이 불안정/끊김 상태로 보입니다. "
                            "MindsDB 서비스 상태 및 TARGET_DB_HOST/PORT 접근 가능 여부를 확인한 뒤 재시도해 주세요. "
                            "가능하면 연결이 끊겨도 자동 재연결(ping/reconnect)하도록 서버 측을 보강하는 것이 좋습니다."
                        ),
                        enrichment_queries=[],
                    )
                else:
                    triage = await triage_no_acceptable_sql(
                        user_question=effective_question,
                        dbms=state.dbms,
                        remaining_tool_calls=state.remaining_tool_calls,
                        attempts=result.attempts or [],
                        build_sql_context_xml=state.current_tool_result or "",
                        allow_give_best_effort=True,
                    )

                tried_enrichment: set[str] = set()
                no_progress_streak = 0
                prev_ctx_sig = _context_signature(state.current_tool_result or "")
                prev_fail_fp = ""

                # Use only the initial triage enrichment set (one triage call per request).
                if triage.decision == "context_refresh" and state.remaining_tool_calls > 0:
                    for idx, enrich in enumerate(list(triage.enrichment_queries or [])[:8], start=1):
                        if state.remaining_tool_calls <= 0:
                            break
                        key = _norm_ws_key(enrich)
                        if not key or key in tried_enrichment:
                            continue
                        tried_enrichment.add(key)

                        # 1) Refresh build_sql_context with enrichment (but keep controller question stable).
                        refresh_started = time.perf_counter()
                        base_ctx = state.current_tool_result
                        enriched_question_for_context = (
                            f"{effective_question}\n\n[자동 문맥 보강]\n{enrich}".strip()
                        )
                        try:
                            new_ctx: str = ""
                            async for ev in _stream_build_sql_context_with_progress(
                                parameters={
                                    "question": enriched_question_for_context,
                                    "exclude_light_sqls": _extract_light_query_sqls(base_ctx),
                                }
                            ):
                                if isinstance(ev, dict) and ev.get("event") == "__tool_result__":
                                    new_ctx = str(ev.get("tool_result") or "")
                                    break
                                yield json.dumps(ev, ensure_ascii=False) + "\n"
                            state.remaining_tool_calls = max(0, state.remaining_tool_calls - 1)

                            # Merge new tool_result into metadata (idempotent, deduped)
                            try:
                                ms = merge_build_sql_context_tool_result_into_metadata(
                                    metadata=state.metadata,
                                    tool_result_xml=new_ctx,
                                    description_limit=600,
                                )
                                SmartLogger.log(
                                    "INFO",
                                    "react.auto_context_refresh.metadata.merge.build_sql_context",
                                    category="react.metadata.merge",
                                    params=sanitize_for_log(
                                        {
                                            "react_run_id": react_run_id,
                                            "added": ms.added,
                                            "updated": ms.updated,
                                            "skipped": ms.skipped,
                                            "invalid": ms.invalid,
                                        }
                                    ),
                                )
                            except Exception as exc:
                                SmartLogger.log(
                                    "WARNING",
                                    "react.auto_context_refresh.metadata.merge.build_sql_context.failed",
                                    category="react.metadata.merge",
                                    params=sanitize_for_log(
                                        {
                                            "react_run_id": react_run_id,
                                            "error": repr(exc),
                                            "traceback": traceback.format_exc(),
                                        }
                                    ),
                                    max_inline_chars=0,
                                )

                            merged = merge_build_sql_context_tool_results(
                                base_tool_result_xml=base_ctx,
                                new_tool_result_xml=new_ctx,
                            )
                            state.current_tool_result = merged or new_ctx

                            elapsed_ms = (time.perf_counter() - refresh_started) * 1000.0
                            new_sig = _context_signature(state.current_tool_result or "")
                            delta = _context_delta(prev_ctx_sig, new_sig)
                            yield json.dumps(
                                {
                                    "event": "context_refresh",
                                    "mode": "auto",
                                    "tool_name": "build_sql_context",
                                    "elapsed_ms": elapsed_ms,
                                    "remaining_tool_calls": state.remaining_tool_calls,
                                    "enrichment_index": idx,
                                    "context_parse_mode": new_sig.get("parse_mode"),
                                    "context_hash_changed": bool(new_sig.get("hash") and new_sig.get("hash") != prev_ctx_sig.get("hash")),
                                    "context_delta": delta,
                                },
                                ensure_ascii=False,
                            ) + "\n"

                            # 2) Retry controller with refreshed context
                            result = None
                            async for ev in _stream_run_controller_with_progress(
                                question=effective_question,
                                dbms=state.dbms,
                                max_sql_seconds=state.max_sql_seconds,
                                config=cfg,
                                prebuilt_context_xml=state.current_tool_result,
                                conversation_context=conversation_context,
                            ):
                                if isinstance(ev, dict) and ev.get("event") == "__controller_result__":
                                    result = ev.get("result")
                                    break
                                yield json.dumps(ev, ensure_ascii=False) + "\n"
                            if result is None:
                                raise RuntimeError("controller_result_missing")
                            state.remaining_tool_calls = max(0, state.remaining_tool_calls - len(result.attempts or []))
                            for attempt in result.attempts or []:
                                state.iteration += 1
                                reasoning = (
                                    f"controller: validate_sql (candidate={attempt.candidate_index}, "
                                    f"score={attempt.score:.2f}, verdict={attempt.verdict})"
                                )
                                step_model = _controller_attempt_to_step_model(
                                    iteration=state.iteration,
                                    reasoning=reasoning,
                                    sql=attempt.sql,
                                    tool_result_xml=attempt.tool_result_xml or "",
                                    metadata_xml=state.metadata.to_xml(),
                                )
                                step_model.sql_completeness = SQLCompletenessModel(
                                    is_complete=bool(
                                        attempt.verdict == "PASS" and attempt.score >= cfg.score_threshold
                                    ),
                                    missing_info=_business_missing_info_for_attempt(
                                        verdict=str(getattr(attempt, "verdict", "") or ""),
                                        score=float(getattr(attempt, "score", 0.0) or 0.0),
                                        score_threshold=float(cfg.score_threshold),
                                        preview=getattr(attempt, "preview", None) if isinstance(getattr(attempt, "preview", None), dict) else {},
                                        fail_reason=str(getattr(attempt, "fail_reason", "") or ""),
                                    ),
                                    confidence_level=(
                                        "high"
                                        if (attempt.verdict == "PASS" and float(getattr(attempt, "score", 0.0) or 0.0) >= float(cfg.score_threshold))
                                        else ("medium" if attempt.verdict == "PASS" else "low")
                                    ),
                                )
                                steps.append(step_model)
                                yield json.dumps(
                                    {"event": "step", "step": step_model.model_dump(), "state": state.to_dict()},
                                    ensure_ascii=False,
                                ) + "\n"

                            if result.status == "submit_sql":
                                break

                            # Stop condition: no context improvement + repeated failure fingerprint.
                            fail_fp = "|".join(
                                [
                                    (str(getattr(a, "fail_reason", "") or "").strip()[:120])
                                    for a in list(result.attempts or [])
                                    if str(getattr(a, "verdict", "") or "").upper() != "PASS"
                                ][:6]
                            ).lower()
                            no_context_change = bool(new_sig.get("hash") and new_sig.get("hash") == prev_ctx_sig.get("hash"))
                            if no_context_change and fail_fp and fail_fp == prev_fail_fp:
                                no_progress_streak += 1
                            else:
                                no_progress_streak = 0
                            prev_fail_fp = fail_fp
                            prev_ctx_sig = new_sig

                            if no_progress_streak >= 2:
                                warnings.append("auto_context_refresh stopped: no progress (context hash + fail pattern)")
                                break
                        except Exception as exc:
                            warnings.append(f"auto_context_refresh failed: {exc}")
                            SmartLogger.log(
                                "WARNING",
                                "react.auto_context_refresh.build_sql_context.failed",
                                category="react.context_refresh",
                                params=sanitize_for_log(
                                    {
                                        "react_run_id": react_run_id,
                                        "error": repr(exc),
                                        "traceback": traceback.format_exc(),
                                    }
                                ),
                                max_inline_chars=0,
                            )
                            yield json.dumps(
                                {
                                    "event": "context_refresh_error",
                                    "mode": "auto",
                                    "tool_name": "build_sql_context",
                                    "error": str(exc),
                                },
                                ensure_ascii=False,
                            ) + "\n"
                            break

                # Policy: allow best-effort submit ONLY when tool-call budget is exhausted.
                if result.status == "ask_user" and state.remaining_tool_calls <= 0:
                    best = _best_pass_attempt(result.attempts or [])
                    if best is not None:
                        warnings.append("best_effort: submitting best PASS SQL due to exhausted tool budget")
                        # Force fall-through as if submit_sql
                        result.status = "submit_sql"
                        result.final_sql = str(getattr(best, "sql", "") or "")
                        result.validated_sql = str(getattr(best, "sql", "") or "")
                        result.preview = dict(getattr(best, "preview", {}) or {})

                if result.status == "ask_user":
                    question = (triage.ask_user_question or "").strip() or (result.question_to_user or "").strip()
                    state.pending_agent_question = question
                    session_token = state.to_token()
                    response_payload = ReactResponse(
                        status="needs_user_input",
                        steps=steps,
                        collected_metadata=state.metadata.to_xml(),
                        partial_sql=state.partial_sql,
                        remaining_tool_calls=state.remaining_tool_calls,
                        session_state=session_token,
                        conversation_state=request.conversation_state,
                        question_to_user=question,
                        warnings=warnings or None,
                    )
                    payload = {"event": "needs_user_input", "response": response_payload.model_dump(), "state": state.to_dict()}
                    yield json.dumps(payload, ensure_ascii=False) + "\n"
                    return

            if result.status != "submit_sql":
                yield json.dumps(
                    {"event": "error", "message": f"Controller did not complete: {result.status}"},
                    ensure_ascii=False,
                ) + "\n"
                return

            final_sql = result.final_sql or ""
            state.partial_sql = final_sql
            state.last_validated_sql = (result.validated_sql or final_sql).strip()
            state.last_validation_preview = result.preview or {}

            validated_sql = (state.last_validated_sql or "").strip()
            execution_result = None

            if validated_sql:
                preview = state.last_validation_preview or {}
                if isinstance(preview, dict) and preview.get("columns") is not None:
                    try:
                        execution_result = ExecutionResultModel(
                            columns=list(preview.get("columns") or []),
                            rows=list(preview.get("rows") or []),
                            row_count=int(preview.get("row_count") or 0),
                            execution_time_ms=float(preview.get("execution_time_ms") or 0),
                        )
                    except Exception:
                        execution_result = None
            else:
                guard = SQLGuard()
                try:
                    validated_sql, _ = guard.validate(final_sql)
                except SQLValidationError as exc:
                    yield json.dumps(
                        {"event": "error", "message": f"SQL validation failed: {exc}"},
                        ensure_ascii=False,
                    ) + "\n"
                    return
                if request.execute_final_sql:
                    executor = SQLExecutor()
                    try:
                        db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
                        if db_type in {"mysql", "mariadb"}:
                            from app.core.sql_mindsdb_prepare import prepare_sql_for_mindsdb

                            validated_sql = prepare_sql_for_mindsdb(validated_sql, request.datasource).sql
                        raw_result = await executor.execute_query(
                            db_conn,
                            validated_sql,
                            timeout=float(request.max_sql_seconds),
                        )
                        formatted = executor.format_results_for_json(raw_result)
                        execution_result = ExecutionResultModel(**formatted)
                    except SQLExecutionError as exc:
                        warnings.append(f"SQL execution failed: {exc}")

            # Prepare small metadata + steps snapshot for gate + postprocess.
            metadata_dict = {
                "identified_tables": state.metadata.identified_tables,
                "identified_columns": state.metadata.identified_columns,
                "identified_values": state.metadata.identified_values,
                "identified_relationships": state.metadata.identified_relationships,
                "identified_constraints": state.metadata.identified_constraints,
            }
            steps_for_save = [step.model_dump() for step in steps] if steps else []

            # Query quality gate (UI feedback exposure)
            quality_gate: Optional[QueryQualityGateModel] = None
            feedback_required = False
            if execution_result is not None:
                try:
                    quality_gate = await _run_query_quality_gate_for_ui(
                        question=request.question,
                        sql=final_sql,
                        execution_result=execution_result,
                        metadata=metadata_dict,
                        steps=steps,
                    )
                    feedback_required = bool(not quality_gate.ok)
                except Exception as gate_exc:
                    # Fail-closed for UI feedback: if gate fails unexpectedly, request feedback.
                    SmartLogger.log(
                        "WARNING",
                        "react.query_quality_gate.failed",
                        category="react.query_quality_gate",
                        params=sanitize_for_log(
                            {
                                "react_run_id": react_run_id,
                                "error": repr(gate_exc),
                            }
                        ),
                    )
                    judge_rounds = max(1, int(getattr(settings, "cache_postprocess_query_judge_rounds", 2) or 2))
                    judge_threshold = float(getattr(settings, "cache_postprocess_query_judge_conf_threshold", 0.90) or 0.90)
                    quality_gate = QueryQualityGateModel(
                        threshold=float(judge_threshold),
                        rounds=int(judge_rounds),
                        ok=False,
                        verified_confidence=None,
                        verified_confidence_avg=None,
                        results=[],
                        error=f"gate_exception:{type(gate_exc).__name__}",
                    )
                    feedback_required = True

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
                conversation_state=None,
                question_to_user=None,
                warnings=warnings or None,
                quality_gate=quality_gate,
                feedback_required=bool(feedback_required),
            )

            # Build/append conversation_state for follow-up continuity (best-effort).
            try:
                base_cap = capsule
                if not getattr(base_cap, "turns", None) and not getattr(base_cap, "created_at_ms", 0):
                    base_cap = new_capsule(dbms=state.dbms, schema_filter=request.schema_filter)
                exec_dict = execution_result.model_dump() if execution_result is not None else None
                base_cap = append_completed_turn_to_capsule(
                    base_cap,
                    question=request.question,
                    final_sql=final_sql,
                    execution_result=exec_dict,
                    collected_metadata_xml=state.metadata.to_xml(),
                    build_sql_context_xml=state.current_tool_result or "",
                )
                response_payload.conversation_state = encode_conversation_state(base_cap)
            except Exception:
                response_payload.conversation_state = request.conversation_state

            # Neo4j caching/history postprocess (best-effort)
            try:
                from app.core.background_jobs import enqueue_cache_postprocess

                if settings.react_enable_query_value_mapping_generation:
                    enqueue_cache_postprocess(
                        {
                            "react_run_id": react_run_id,
                            "question": request.question,
                            # MindsDB-only mode requires datasource for SQL transform / DB-side verification.
                            "datasource": request.datasource,
                            "final_sql": final_sql,
                            "validated_sql": validated_sql,
                            "status": "completed",
                            "row_count": execution_result.row_count if execution_result else None,
                            "execution_time_ms": execution_result.execution_time_ms if execution_result else None,
                            "steps_count": len(steps),
                            "metadata_dict": metadata_dict,
                            "steps": steps_for_save,
                            # Avoid double judge calls: reuse server-computed gate results in worker.
                            "query_quality_gate": (
                                quality_gate.model_dump() if quality_gate is not None else None
                            ),
                        }
                    )
                else:
                    SmartLogger.log(
                        "INFO",
                        "react.neo4j.enqueue_skipped",
                        category="react.neo4j.enqueue_skipped",
                        params={
                            "react_run_id": react_run_id,
                            "reason": "react_enable_query_value_mapping_generation=false",
                        },
                    )
            except Exception as enqueue_err:
                SmartLogger.log(
                    "WARNING",
                    "react.neo4j.enqueue_failed",
                    category="react.neo4j.enqueue_failed",
                    params={"react_run_id": react_run_id, "error": str(enqueue_err)[:200]},
                )

            if execution_result and request.use_cache and not request.conversation_state:
                try:
                    from app.core.query_cache import get_query_cache

                    cache = get_query_cache()
                    cache.put(
                        question=request.question,
                        datasource=request.datasource,
                        final_sql=final_sql,
                        validated_sql=validated_sql,
                        execution_result=execution_result.model_dump() if execution_result else None,
                        steps_summary=f"{len(steps)} steps completed",
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
            payload = {"event": "completed", "response": response_payload.model_dump(), "state": state.to_dict()}
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
