"""
Cache post-processing pipeline (runs in background workers).

Responsibilities:
- Build minimal steps summary (LLM) for storage.
- Extract ValueMapping candidates (LLM).
- Apply strong gate: DB existence checks before persisting ValueMapping.
- Upsert Query node using db+question deterministic id and overwrite policy.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.deps import neo4j_conn
from app.core.llm_factory import create_embedding_client, create_llm
from app.core.sql_exec import SQLExecutor
from app.models.neo4j_history import Neo4jQueryRepository
from app.react.generators.query_quality_gate_generator import (
    QueryQualityJudgeResult,
    get_query_quality_gate_generator,
)
from app.smart_logger import SmartLogger
from app.react.utils.log_sanitize import sanitize_for_log


"""
ValueMapping 정책 (Option C):
- stopword 기반 차단은 사용하지 않는다.
- 필요 시 스키마/테이블/컬럼별 allow/deny 정책으로 제어한다.
  (현재 기본 정책은 "모든 컬럼 허용"이며, DB 존재성/컬럼해결 게이트가 품질을 보장한다.)
"""

# Optional: block specific columns if they create noise.
# Keep empty by default.
VALUE_MAPPING_COLUMN_NAME_DENYLIST = set()

# Optional: allowlist/denylist by FQN regex.
# If allowlist is non-empty, only matching FQNs are allowed.
VALUE_MAPPING_COLUMN_FQN_ALLOWLIST_REGEX: List[str] = []
VALUE_MAPPING_COLUMN_FQN_DENYLIST_REGEX: List[str] = []


@dataclass
class ValueMappingCandidate:
    schema: str
    table: str
    column: str
    natural_value: str
    code_value: str
    confidence: float = 0.0
    evidence: str = ""


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _extract_first_json_obj(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    st = s.find("{")
    en = s.rfind("}")
    if st < 0 or en < 0 or en <= st:
        return None
    cand = s[st : en + 1]
    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


async def _fetch_sql_preview_postgres(
    conn: asyncpg.Connection,
    *,
    sql: str,
    limit_rows: int = 3,
    timeout_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Best-effort SQL preview for PostgreSQL (used as judge evidence).
    - Never raises (caller should still wrap for safety).
    """
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return {"columns": [], "rows": [], "row_count": None, "error": "empty_sql"}
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        return {"columns": [], "rows": [], "row_count": None, "error": "unsupported_dbms"}
    lim = max(1, int(limit_rows))
    # Wrap as subquery to enforce LIMIT without attempting to rewrite the SQL.
    preview_sql = f"SELECT * FROM ({s}) AS _subq LIMIT {lim}"
    try:
        rows = await conn.fetch(preview_sql, timeout=float(timeout_s))
        if not rows:
            return {"columns": [], "rows": [], "row_count": 0}
        cols = list(rows[0].keys())
        out_rows: List[List[Any]] = []
        for r in rows[:lim]:
            out_rows.append([str(v) if v is not None else None for v in list(r.values())])
        return {"columns": cols[:50], "rows": out_rows[:lim], "row_count": None}
    except Exception as exc:
        return {"columns": [], "rows": [], "row_count": None, "error": f"{type(exc).__name__}:{str(exc)[:160]}"}


async def _fetch_sql_preview_mindsdb(
    conn: Any,
    *,
    sql: str,
    datasource: str,
    limit_rows: int = 3,
    timeout_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Best-effort SQL preview via MindsDB(MySQL protocol).
    - Expects an aiomysql connection (created in _open_mindsdb_connection()).
    - Uses the same transform used by validate_sql (datasource prefix + backticks).
    - Never raises.
    """
    try:
        ds = str(datasource or "").strip()
        s = (sql or "").strip().rstrip(";").strip()
        if not s:
            return {"columns": [], "rows": [], "row_count": None, "error": "empty_sql"}
        if not ds:
            return {"columns": [], "rows": [], "row_count": None, "error": "missing_datasource"}
        # Ensure MindsDB addressing/quoting.
        try:
            from app.core.sql_mindsdb_prepare import prepare_sql_for_mindsdb

            s = prepare_sql_for_mindsdb(s, ds).sql
        except Exception as exc:
            return {"columns": [], "rows": [], "row_count": None, "error": f"transform_failed:{type(exc).__name__}:{str(exc)[:120]}"}
        ex = SQLExecutor()
        preview = await ex.preview_query(conn, s, row_limit=max(1, int(limit_rows)), timeout=float(timeout_s))
        # Normalize keys (already consistent, but keep defensive)
        return {
            "columns": list(preview.get("columns") or []),
            "rows": list(preview.get("rows") or []),
            "row_count": int(preview.get("row_count") or 0),
            "execution_time_ms": float(preview.get("execution_time_ms") or 0),
        }
    except Exception as exc:
        return {"columns": [], "rows": [], "row_count": None, "error": f"{type(exc).__name__}:{str(exc)[:160]}"}


async def _open_mindsdb_connection() -> Any:
    """
    Open a fresh aiomysql connection to MindsDB MySQL endpoint using current settings.
    This is used by cache_postprocess to perform DB-side verification even when the
    underlying physical DB is not PostgreSQL.
    """
    try:
        import aiomysql  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("aiomysql is required for TARGET_DB_TYPE=mysql but not installed") from exc
    return await aiomysql.connect(
        host=settings.target_db_host,
        port=int(settings.target_db_port),
        user=settings.target_db_user,
        password=settings.target_db_password,
        db=settings.target_db_name,
        autocommit=True,
    )


def _safe_ident(part: str) -> str:
    # Very conservative identifier sanitizer (fail-closed).
    return re.sub(r"[^A-Za-z0-9_]", "", str(part or ""))


def _escape_sql_string_literal(value: str) -> str:
    # Escape single quotes for SQL string literal.
    return str(value or "").replace("'", "''")


async def _value_exists_in_mindsdb(
    conn: Any,
    *,
    datasource: str,
    schema: str,
    table: str,
    column: str,
    code_value: str,
    timeout_s: float = 1.5,
) -> bool:
    """
    Strong-ish gate for MindsDB: try executing a minimal existence query.
    If it returns at least 1 row, treat as exists. Any error => fail-closed (False).
    """
    ds = str(datasource or "").strip()
    if not ds:
        return False
    schema_id = _safe_ident(schema)
    table_id = _safe_ident(table)
    col_id = _safe_ident(column)
    if not table_id or not col_id:
        return False

    # Two-pass literal strategy: numeric first (for '8537'), then quoted string.
    raw = str(code_value or "").strip()
    literals: List[str] = []
    if raw and re.fullmatch(r"-?\d+", raw):
        literals.append(raw)  # int-like
    elif raw and re.fullmatch(r"-?\d+\.\d+", raw):
        literals.append(raw)  # float-like
    if raw:
        literals.append(f"'{_escape_sql_string_literal(raw)}'")

    table_ref = ".".join([p for p in [schema_id, table_id] if p])
    ex = SQLExecutor()
    for lit in literals[:2]:
        probe = f"SELECT 1 AS one FROM {table_ref} WHERE {col_id} = {lit} LIMIT 1"
        try:
            from app.core.sql_mindsdb_prepare import prepare_sql_for_mindsdb

            probe2 = prepare_sql_for_mindsdb(probe, ds).sql
        except Exception:
            continue
        try:
            prev = await ex.preview_query(conn, probe2, row_limit=1, timeout=float(timeout_s))
            if int(prev.get("row_count") or 0) > 0:
                return True
        except Exception:
            continue
    return False


async def _strict_verify_value_mapping_mindsdb(
    conn: Any,
    *,
    datasource: str,
    schema: str,
    table: str,
    code_column: str,
    code_value: str,
    natural_value: str,
    timeout_s: float = 1.5,
) -> bool:
    """
    Strict verify (MindsDB): fetch 1 row by code_column=code_value, then check whether any
    returned cell (excluding the code column itself) contains natural_value (normalized).
    Any error => fail-closed.
    """
    ds = str(datasource or "").strip()
    nat = _norm_text_key(natural_value)
    if not ds or not nat:
        return False
    schema_id = _safe_ident(schema)
    table_id = _safe_ident(table)
    col_id = _safe_ident(code_column)
    if not table_id or not col_id:
        return False

    raw = str(code_value or "").strip()
    if not raw:
        return False
    # Prefer numeric literal when obvious, else quoted.
    if re.fullmatch(r"-?\d+", raw) or re.fullmatch(r"-?\d+\.\d+", raw):
        lit = raw
    else:
        lit = f"'{_escape_sql_string_literal(raw)}'"

    table_ref = ".".join([p for p in [schema_id, table_id] if p])
    probe = f"SELECT * FROM {table_ref} WHERE {col_id} = {lit} LIMIT 1"
    try:
        from app.core.sql_mindsdb_prepare import prepare_sql_for_mindsdb

        probe2 = prepare_sql_for_mindsdb(probe, ds).sql
    except Exception:
        return False

    ex = SQLExecutor()
    try:
        prev = await ex.preview_query(conn, probe2, row_limit=1, timeout=float(timeout_s))
    except Exception:
        return False
    if int(prev.get("row_count") or 0) <= 0:
        return False

    cols = list(prev.get("columns") or [])
    rows = list(prev.get("rows") or [])
    if not rows:
        return False
    row0 = rows[0] if isinstance(rows[0], list) else []
    if not row0:
        return False

    for idx, v in enumerate(row0):
        colname = str(cols[idx] or "") if idx < len(cols) else ""
        if colname and colname.strip().lower() == col_id.lower():
            continue
        if v is None:
            continue
        if nat in _norm_text_key(str(v)):
            return True
    return False


def _extract_upstream_query_quality_gate(
    payload: Dict[str, Any],
) -> Tuple[bool, Optional[bool], Optional[float], Optional[float], str]:
    """
    If React API already computed query-quality gate (server-side), reuse it to avoid
    double judge calls (UI gate + cache_postprocess gate).

    Returns:
      (used_upstream, ok, min_conf, avg_conf, json_str)
    """
    obj = payload.get("query_quality_gate")
    if not isinstance(obj, dict):
        return (False, None, None, None, "")
    ok = obj.get("ok")
    if not isinstance(ok, bool):
        # Require explicit ok to treat as upstream gate.
        return (False, None, None, None, "")
    results = obj.get("results")
    confs: List[float] = []
    if isinstance(results, list):
        for r in results:
            if not isinstance(r, dict):
                continue
            try:
                confs.append(float(r.get("confidence") or 0.0))
            except Exception:
                continue
    min_conf = None
    avg_conf = None
    try:
        min_conf = float(min(confs)) if confs else None
    except Exception:
        min_conf = None
    try:
        avg_conf = float(sum(confs) / len(confs)) if confs else None
    except Exception:
        avg_conf = None
    # Prefer upstream-provided aggregate fields when present.
    try:
        if obj.get("verified_confidence") is not None:
            min_conf = float(obj.get("verified_confidence"))  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        if obj.get("verified_confidence_avg") is not None:
            avg_conf = float(obj.get("verified_confidence_avg"))  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = ""
    return (True, bool(ok), min_conf, avg_conf, s)


def _compact_steps_for_judge(steps: Any, *, limit: int = 6) -> List[Dict[str, Any]]:
    """
    Steps emitted from controller can be huge (tool_result_xml). Keep a tiny, stable subset.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(steps, list):
        return out
    tail = steps[-max(1, int(limit)) :]
    for s in tail:
        if not isinstance(s, dict):
            continue
        tool_call = s.get("tool_call") if isinstance(s.get("tool_call"), dict) else {}
        sql_comp = s.get("sql_completeness") if isinstance(s.get("sql_completeness"), dict) else {}
        out.append(
            {
                "iteration": s.get("iteration"),
                "tool_name": tool_call.get("name"),
                "partial_sql_preview": str(s.get("partial_sql") or "")[:260],
                "confidence_level": str(sql_comp.get("confidence_level") or ""),
                "missing_info": str(sql_comp.get("missing_info") or "")[:240],
            }
        )
    return out


def _norm_text_key(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


async def _verify_value_mapping_by_text_columns(
    conn: asyncpg.Connection,
    *,
    schema: str,
    table: str,
    code_column: str,
    code_value: str,
    natural_value: str,
    max_text_cols: int = 4,
    timeout_s: float = 1.5,
) -> bool:
    """
    Strict verify for ValueMapping:
    - Fetch one row by code_column=code_value
    - Check whether any *text-ish* columns (name/desc-like) contains natural_value.
    """
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        return False
    nat = _norm_text_key(natural_value)
    if not nat:
        return False

    q_cols = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE lower(table_schema) = lower($1)
      AND lower(table_name) = lower($2)
    ORDER BY ordinal_position
    """
    try:
        cols = await conn.fetch(q_cols, schema, table, timeout=float(timeout_s))
    except Exception:
        return False
    # Pick name/desc/title-like columns first; fall back to any text columns.
    candidates: List[str] = []
    textish: List[str] = []
    for r in cols:
        cn = str(r.get("column_name") or "").strip()
        dt = str(r.get("data_type") or "").strip().lower()
        if not cn:
            continue
        is_text = dt in {"text", "character varying", "character", "varchar", "char"}
        if not is_text:
            continue
        textish.append(cn)
        c_l = cn.lower()
        if any(tok in c_l for tok in ("name", "nm", "title", "desc", "description")):
            candidates.append(cn)
    # Ensure deterministic order + cap
    picked = candidates[: max(0, int(max_text_cols))]
    if len(picked) < max(0, int(max_text_cols)):
        for c in textish:
            if c in picked:
                continue
            picked.append(c)
            if len(picked) >= max(0, int(max_text_cols)):
                break
    picked = [p for p in picked if p]
    if not picked:
        return False

    def q_ident(x: str) -> str:
        return '"' + x.replace('"', '""') + '"'

    # Coerce bind param to avoid DataError on non-text code columns (e.g., int).
    code_value_any: Any = code_value
    try:
        meta = await conn.fetchrow(
            """
            SELECT data_type, udt_name
            FROM information_schema.columns
            WHERE lower(table_schema) = lower($1)
              AND lower(table_name) = lower($2)
              AND lower(column_name) = lower($3)
            LIMIT 1
            """,
            schema,
            table,
            code_column,
            timeout=float(timeout_s),
        )
        data_type = str(meta.get("data_type") or "").strip().lower() if meta else ""
        udt_name = str(meta.get("udt_name") or "").strip().lower() if meta else ""
        raw = (code_value or "").strip()
        if raw:
            if data_type in {"integer", "bigint", "smallint"} or udt_name in {"int2", "int4", "int8"}:
                code_value_any = int(raw)
            elif data_type in {"numeric", "decimal", "real", "double precision"} or udt_name in {"numeric", "float4", "float8"}:
                code_value_any = float(raw)
            elif data_type == "boolean" or udt_name == "bool":
                if raw.lower() in {"true", "t", "1", "y", "yes"}:
                    code_value_any = True
                elif raw.lower() in {"false", "f", "0", "n", "no"}:
                    code_value_any = False
                else:
                    return False
    except Exception:
        return False

    sel_cols = ", ".join([q_ident(c) for c in picked])
    sql_row = (
        f"SELECT {sel_cols} FROM {q_ident(schema)}.{q_ident(table)} "
        f"WHERE {q_ident(code_column)} = $1 LIMIT 1"
    )
    try:
        row = await conn.fetchrow(sql_row, code_value_any, timeout=float(timeout_s))
    except Exception:
        return False
    if not row:
        return False
    for c in picked:
        try:
            v = row.get(c)
        except Exception:
            v = None
        if v is None:
            continue
        if nat in _norm_text_key(str(v)):
            return True
    return False


def _candidate_brief(c: ValueMappingCandidate) -> Dict[str, Any]:
    return {
        "schema": c.schema,
        "table": c.table,
        "column": c.column,
        "natural_value": c.natural_value,
        "code_value": c.code_value,
        "confidence": c.confidence,
        "evidence": (c.evidence or "")[:200],
    }


def _append_sample(bucket: List[Dict[str, Any]], cand: ValueMappingCandidate, *, limit: int = 5) -> None:
    if len(bucket) >= limit:
        return
    bucket.append(_candidate_brief(cand))


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _cap_str(value: Any, *, max_len: int) -> str:
    s = str(value or "").strip()
    if max_len <= 0:
        return ""
    return s[:max_len]


def _cap_list(values: Any, *, max_items: int, item_max_len: int) -> List[str]:
    if max_items <= 0:
        return []
    out: List[str] = []
    if isinstance(values, list):
        for x in values:
            s = str(x or "").strip()
            if not s:
                continue
            out.append(_cap_str(s, max_len=item_max_len))
            if len(out) >= max_items:
                break
    elif isinstance(values, str):
        # tolerate comma-separated strings
        parts = [p.strip() for p in values.split(",")]
        for p in parts:
            if not p:
                continue
            out.append(_cap_str(p, max_len=item_max_len))
            if len(out) >= max_items:
                break
    return out


def _extract_steps_features(*, steps_summary_json: str) -> Dict[str, Any]:
    """
    Convert steps_summary(JSON string) into a stable, size-capped feature dict.
    This is intended for:
    - best_context selection/merge
    - feeding into light disambiguation query generation (low token, robust)
    """
    obj = _safe_json_loads(steps_summary_json) or {}
    # Keep only a stable subset
    features: Dict[str, Any] = {
        "intent": _cap_str(obj.get("intent"), max_len=240),
        "tables": _cap_list(obj.get("tables"), max_items=12, item_max_len=120),
        "columns": _cap_list(obj.get("columns"), max_items=24, item_max_len=140),
        "filters": _cap_list(obj.get("filters"), max_items=18, item_max_len=220),
        "aggregations": _cap_list(obj.get("aggregations"), max_items=10, item_max_len=120),
        "group_by": _cap_list(obj.get("group_by"), max_items=10, item_max_len=120),
        "order_by": _cap_list(obj.get("order_by"), max_items=10, item_max_len=120),
        "time_range": _cap_str(obj.get("time_range"), max_len=220),
        "notes": _cap_str(obj.get("notes"), max_len=500),
    }
    # Drop empty keys to keep storage small
    compact: Dict[str, Any] = {}
    for k, v in features.items():
        if isinstance(v, str) and v.strip():
            compact[k] = v
        elif isinstance(v, list) and v:
            compact[k] = v
    return compact


def _compute_context_richness_score(
    *,
    steps_features: Dict[str, Any],
    metadata_dict: Dict[str, Any],
    validated_value_mappings_count: int,
    row_count: Optional[int],
) -> float:
    """
    Deterministic score in [0,1] for selecting best_context independent of best_sql.
    Focus on reusable structural information and grounding signals.
    """
    score = 0.0

    # Structure
    if str(steps_features.get("time_range") or "").strip():
        score += 0.20
    if (steps_features.get("group_by") or []) and isinstance(steps_features.get("group_by"), list):
        score += 0.10
    if (steps_features.get("aggregations") or []) and isinstance(steps_features.get("aggregations"), list):
        score += 0.10
    # A weak heuristic: if notes mention "최근/가장 최근" or "일별/월별" it's still useful.
    notes_l = str(steps_features.get("notes") or "").lower()
    if any(tok in notes_l for tok in ("일별", "월별", "시간별", "daily", "monthly", "hourly")):
        score += 0.10
    if any(tok in notes_l for tok in ("최근", "가장 최근", "latest", "recent")):
        score += 0.05

    # Entity/filters
    filters = steps_features.get("filters")
    if isinstance(filters, list) and len(filters) > 0:
        score += 0.15

    # Reuse signals
    try:
        identified_values = (metadata_dict or {}).get("identified_values") or []
        if isinstance(identified_values, list) and len(identified_values) > 0:
            score += 0.15
    except Exception:
        pass

    if int(validated_value_mappings_count or 0) > 0:
        score += 0.10

    # Breadth (small)
    tables = steps_features.get("tables")
    cols = steps_features.get("columns")
    if isinstance(tables, list) and len(tables) >= 2:
        score += 0.05
    if isinstance(cols, list) and len(cols) >= 6:
        score += 0.05

    # Sanity: non-empty results are a mild positive (avoid over-optimizing for empty queries)
    try:
        if isinstance(row_count, int) and row_count > 0:
            score += 0.05
    except Exception:
        pass

    return max(0.0, min(1.0, float(score)))


async def process_cache_postprocess_payload(payload: Dict[str, Any]) -> None:
    """
    Main entrypoint for worker.
    payload should include:
      - react_run_id, question, final_sql/validated_sql, execution_time_ms, row_count, steps_count
      - metadata_dict (identified_*)
      - steps (raw steps list; may be large)
    """
    started = time.perf_counter()
    react_run_id = payload.get("react_run_id")
    question = payload.get("question") or ""
    sql = payload.get("validated_sql") or payload.get("final_sql") or payload.get("sql") or ""
    status = payload.get("status") or "completed"
    row_count_any = payload.get("row_count")
    row_count = int(row_count_any) if isinstance(row_count_any, int) else None
    execution_time_ms = (
        float(payload.get("execution_time_ms"))
        if payload.get("execution_time_ms") is not None and isinstance(payload.get("execution_time_ms"), (int, float))
        else None
    )

    # Feature flag: allow disabling Query/ValueMapping generation at runtime.
    if not getattr(settings, "react_enable_query_value_mapping_generation", True):
        SmartLogger.log(
            "INFO",
            "cache_postprocess.skipped",
            category="cache_postprocess.skipped",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "reason": "react_enable_query_value_mapping_generation=false",
                }
            ),
            max_inline_chars=0,
        )
        return

    # Fail-closed: do not persist empty-result queries.
    min_row = int(getattr(settings, "cache_postprocess_query_min_row_count", 1) or 1)
    if row_count is not None and row_count < min_row:
        SmartLogger.log(
            "INFO",
            "cache_postprocess.skipped_row_count_gate",
            category="cache_postprocess.skipped",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "reason": "row_count_below_min",
                    "row_count": row_count,
                    "min_row_count": min_row,
                    "question": question[:200],
                }
            ),
            max_inline_chars=0,
        )
        return

    if not str(sql or "").strip():
        SmartLogger.log(
            "INFO",
            "cache_postprocess.skipped_no_sql",
            category="cache_postprocess.skipped",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "reason": "no_sql",
                    "question": question[:200],
                }
            ),
            max_inline_chars=0,
        )
        return

    SmartLogger.log(
        "INFO",
        "cache_postprocess.start",
        category="cache_postprocess",
        params=sanitize_for_log(
            {
                "react_run_id": react_run_id,
                "question": question,
                "has_sql": bool(sql),
            }
        ),
        max_inline_chars=0,
    )

    neo4j_session = None
    db_conn: Any = None
    try:
        neo4j_session = await neo4j_conn.get_session()
        # NOTE:
        # - In MindsDB-only mode (TARGET_DB_TYPE=mysql), the "target DB" is a MySQL protocol endpoint.
        # - This cache_postprocess pipeline currently uses asyncpg (PostgreSQL) only for:
        #   - best-effort SQL preview evidence for the judge
        #   - strong ValueMapping validation (information_schema + existence checks)
        # If target_db_type is not PostgreSQL, we must NOT attempt asyncpg.connect(), otherwise
        # the worker crashes and (:Query) nodes are never created.
        db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
        if db_type in {"postgresql", "postgres"}:
            db_conn = await _open_db_connection()
        elif db_type in {"mysql", "mariadb"}:
            # MindsDB MySQL endpoint (Phase 1)
            db_conn = await _open_mindsdb_connection()
        else:
            db_conn = None

        metadata_dict = payload.get("metadata_dict") or {}
        steps_tail = _compact_steps_for_judge(payload.get("steps") or [], limit=6)

        # 1) Strong Query quality gate BEFORE generating any cache artifacts (fail-closed).
        judge_enabled = bool(getattr(settings, "cache_postprocess_query_quality_gate_enabled", True))
        judge_rounds = max(1, int(getattr(settings, "cache_postprocess_query_judge_rounds", 2) or 2))
        judge_threshold = float(getattr(settings, "cache_postprocess_query_judge_conf_threshold", 0.90) or 0.90)
        quality_gate_json_str: str = ""
        verified_confidence: Optional[float] = None
        verified_confidence_avg: Optional[float] = None
        preview: Optional[Dict[str, Any]] = None
        used_upstream, upstream_ok, upstream_min, upstream_avg, upstream_json = _extract_upstream_query_quality_gate(payload)
        if used_upstream:
            quality_gate_json_str = upstream_json or ""
            verified_confidence = upstream_min
            verified_confidence_avg = upstream_avg
            SmartLogger.log(
                "INFO",
                "cache_postprocess.query_quality_gate.used_upstream",
                category="cache_postprocess.query_quality_gate",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "ok": bool(upstream_ok),
                        "verified_confidence": verified_confidence,
                        "verified_confidence_avg": verified_confidence_avg,
                    }
                ),
                max_inline_chars=0,
            )
            if not bool(upstream_ok):
                SmartLogger.log(
                    "INFO",
                    "cache_postprocess.skipped_query_quality_gate",
                    category="cache_postprocess.skipped",
                    params=sanitize_for_log(
                        {
                            "react_run_id": react_run_id,
                            "reason": "query_quality_gate_rejected_upstream",
                            "threshold": float(judge_threshold),
                            "question": question[:200],
                        }
                    ),
                    max_inline_chars=0,
                )
                return
        elif judge_enabled:
            # Best-effort: provide evidence preview to the judge.
            datasource = str(payload.get("datasource") or "").strip()
            if db_type in {"postgresql", "postgres"} and db_conn is not None:
                try:
                    preview = await _fetch_sql_preview_postgres(db_conn, sql=sql, limit_rows=3, timeout_s=2.0)
                except Exception:
                    preview = None
            elif db_type in {"mysql", "mariadb"} and db_conn is not None:
                preview = await _fetch_sql_preview_mindsdb(db_conn, sql=sql, datasource=datasource, limit_rows=3, timeout_s=2.0)
            else:
                preview = {"columns": [], "rows": [], "row_count": None, "error": "db_preview_skipped_unsupported_dbms"}
            qgen = get_query_quality_gate_generator()
            judge_results: List[QueryQualityJudgeResult] = []
            for ridx in range(1, judge_rounds + 1):
                jr = await qgen.judge_round(
                    question=question,
                    sql=sql,
                    row_count=row_count,
                    execution_time_ms=execution_time_ms,
                    metadata=dict(metadata_dict or {}),
                    steps_tail=steps_tail,
                    preview=preview,
                    round_idx=ridx,
                    react_run_id=react_run_id,
                    purpose="cache_postprocess.query_quality_gate",
                )
                judge_results.append(jr)
            ok = all((r.accept and float(r.confidence) >= float(judge_threshold)) for r in judge_results)
            try:
                verified_confidence = min([float(r.confidence) for r in judge_results]) if judge_results else None
            except Exception:
                verified_confidence = None
            try:
                verified_confidence_avg = (
                    float(sum([float(r.confidence) for r in judge_results]) / len(judge_results))
                    if judge_results
                    else None
                )
            except Exception:
                verified_confidence_avg = None
            try:
                quality_gate_json_str = json.dumps(
                    {
                        "policy": "llm_judge_2x",
                        "threshold": float(judge_threshold),
                        "rounds": int(judge_rounds),
                        "row_count": row_count,
                        "execution_time_ms": execution_time_ms,
                        "preview": preview or {},
                        "verified_confidence": verified_confidence,
                        "verified_confidence_avg": verified_confidence_avg,
                        "results": [
                            {
                                "accept": bool(r.accept),
                                "confidence": float(r.confidence),
                                "reasons": list(r.reasons or [])[:8],
                                "risk_flags": list(r.risk_flags or [])[:8],
                                "summary": (r.summary or "")[:240],
                            }
                            for r in judge_results
                        ],
                    },
                    ensure_ascii=False,
                    default=str,
                )
            except Exception:
                quality_gate_json_str = ""
            SmartLogger.log(
                "INFO",
                "cache_postprocess.query_quality_gate.summary",
                category="cache_postprocess.query_quality_gate",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "ok": bool(ok),
                        "threshold": float(judge_threshold),
                        "rounds": int(judge_rounds),
                        "results": [
                            {
                                "accept": bool(r.accept),
                                "confidence": float(r.confidence),
                                "reasons": list(r.reasons or [])[:6],
                                "risk_flags": list(r.risk_flags or [])[:6],
                                "summary": (r.summary or "")[:200],
                            }
                            for r in judge_results
                        ],
                    }
                ),
                max_inline_chars=0,
            )
            if not ok:
                SmartLogger.log(
                    "INFO",
                    "cache_postprocess.skipped_query_quality_gate",
                    category="cache_postprocess.skipped",
                    params=sanitize_for_log(
                        {
                            "react_run_id": react_run_id,
                            "reason": "query_quality_gate_rejected",
                            "threshold": float(judge_threshold),
                            "question": question[:200],
                        }
                    ),
                    max_inline_chars=0,
                )
                return

        # 2) LLM summarize steps + extract mapping candidates (only after Query passes gate)
        steps_summary = await _llm_build_steps_summary(
            question=question,
            sql=sql,
            metadata=metadata_dict,
            steps=payload.get("steps") or [],
        )
        steps_features = _extract_steps_features(steps_summary_json=steps_summary)
        mapping_candidates = await _llm_extract_value_mappings(
            question=question,
            sql=sql,
            metadata=metadata_dict,
            steps_summary=steps_summary,
        )

        # MindsDB enhancement: derive NAME->CODE mappings deterministically (accuracy > speed).
        # This avoids relying on LLM confidence for common "엔티티명 -> 코드" pairs.
        if db_type in {"mysql", "mariadb"} and db_conn is not None:
            datasource = str(payload.get("datasource") or "").strip()
            if datasource:
                derived: List[ValueMappingCandidate] = []
                for cand in list(mapping_candidates or [])[:20]:
                    try:
                        # Only try derivation when the current candidate looks like a name-ish column.
                        col_l = str(cand.column or "").strip().lower()
                        if not col_l:
                            continue
                        if "name" not in col_l and not col_l.endswith("_nm") and not col_l.endswith("nm"):
                            continue
                        d = await _derive_code_mapping_from_name_column_mindsdb(
                            db_conn,
                            datasource=datasource,
                            schema=str(cand.schema or ""),
                            table=str(cand.table or ""),
                            name_column=str(cand.column or ""),
                            natural_value=str(cand.natural_value or ""),
                            timeout_s=1.5,
                        )
                        if d is not None:
                            derived.append(d)
                    except Exception:
                        continue
                if derived:
                    # Merge + basic dedup
                    merged_all = list(mapping_candidates or []) + derived
                    uniq: List[ValueMappingCandidate] = []
                    seen = set()
                    for c in merged_all:
                        key = (
                            str(c.schema or "").strip().lower(),
                            str(c.table or "").strip().lower(),
                            str(c.column or "").strip().lower(),
                            str(c.natural_value or "").strip(),
                            str(c.code_value or "").strip(),
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        uniq.append(c)
                    mapping_candidates = uniq

        SmartLogger.log(
            "INFO",
            "cache_postprocess.value_mapping.candidates",
            category="cache_postprocess.value_mapping",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "target_db_type": settings.target_db_type,
                    "react_caching_db_type": settings.react_caching_db_type,
                    "candidates_count": len(mapping_candidates),
                    "candidates_sample": [_candidate_brief(c) for c in mapping_candidates[:5]],
                }
            ),
            max_inline_chars=0,
        )

        # 2) Strong gate: validate candidates against DB existence
        validated_mappings: List[ValueMappingCandidate] = []
        reject_reason_counts = {
            "policy_rejected": 0,
            "below_confidence": 0,
            "not_in_question": 0,
            "resolve_column_failed": 0,
            "value_not_in_db": 0,
            "strict_verify_failed": 0,
        }
        reject_samples: Dict[str, List[Dict[str, Any]]] = {
            "policy_rejected": [],
            "below_confidence": [],
            "not_in_question": [],
            "resolve_column_failed": [],
            "value_not_in_db": [],
            "strict_verify_failed": [],
        }
        strict_verify = bool(getattr(settings, "cache_postprocess_value_mapping_strict_verify_enabled", True))
        min_vm_conf = float(getattr(settings, "cache_postprocess_value_mapping_min_confidence", 0.92) or 0.92)

        datasource = str(payload.get("datasource") or "").strip()
        if db_conn is None:
            # Keep Query node creation; fail-closed for mappings.
            reject_reason_counts["resolve_column_failed"] += len(mapping_candidates)
            for cand in mapping_candidates[:5]:
                _append_sample(reject_samples["resolve_column_failed"], cand)
        elif db_type in {"postgresql", "postgres"}:
            for cand in mapping_candidates:
                if not _passes_value_mapping_policy(cand):
                    reject_reason_counts["policy_rejected"] += 1
                    _append_sample(reject_samples["policy_rejected"], cand)
                    continue
                if float(getattr(cand, "confidence", 0.0) or 0.0) < float(min_vm_conf):
                    reject_reason_counts["below_confidence"] += 1
                    _append_sample(reject_samples["below_confidence"], cand)
                    continue
                if _norm_text_key(cand.natural_value) not in _norm_text_key(question):
                    reject_reason_counts["not_in_question"] += 1
                    _append_sample(reject_samples["not_in_question"], cand)
                    continue
                resolved = await _resolve_column_case(db_conn, cand.schema, cand.table, cand.column)
                if not resolved:
                    reject_reason_counts["resolve_column_failed"] += 1
                    _append_sample(reject_samples["resolve_column_failed"], cand)
                    continue
                schema_real, table_real, column_real = resolved
                exists = await _value_exists_in_db(db_conn, schema_real, table_real, column_real, cand.code_value)
                if not exists:
                    reject_reason_counts["value_not_in_db"] += 1
                    _append_sample(reject_samples["value_not_in_db"], cand)
                    continue
                if strict_verify:
                    ok_vm = await _verify_value_mapping_by_text_columns(
                        db_conn,
                        schema=schema_real,
                        table=table_real,
                        code_column=column_real,
                        code_value=cand.code_value,
                        natural_value=cand.natural_value,
                        max_text_cols=4,
                        timeout_s=1.5,
                    )
                    if not ok_vm:
                        reject_reason_counts["strict_verify_failed"] += 1
                        _append_sample(reject_samples["strict_verify_failed"], cand)
                        continue
                validated_mappings.append(
                    ValueMappingCandidate(
                        schema=schema_real,
                        table=table_real,
                        column=column_real,
                        natural_value=cand.natural_value,
                        code_value=cand.code_value,
                        confidence=cand.confidence,
                        evidence=cand.evidence,
                    )
                )
        elif db_type in {"mysql", "mariadb"}:
            # MindsDB mode: validate via small probe queries (no information_schema dependency).
            if not datasource:
                reject_reason_counts["resolve_column_failed"] += len(mapping_candidates)
                for cand in mapping_candidates[:5]:
                    _append_sample(reject_samples["resolve_column_failed"], cand)
            else:
                for cand in mapping_candidates:
                    if not _passes_value_mapping_policy(cand):
                        reject_reason_counts["policy_rejected"] += 1
                        _append_sample(reject_samples["policy_rejected"], cand)
                        continue
                    if float(getattr(cand, "confidence", 0.0) or 0.0) < float(min_vm_conf):
                        reject_reason_counts["below_confidence"] += 1
                        _append_sample(reject_samples["below_confidence"], cand)
                        continue
                    if _norm_text_key(cand.natural_value) not in _norm_text_key(question):
                        reject_reason_counts["not_in_question"] += 1
                        _append_sample(reject_samples["not_in_question"], cand)
                        continue
                    schema_id = _safe_ident(cand.schema)
                    table_id = _safe_ident(cand.table)
                    col_id = _safe_ident(cand.column)
                    if not table_id or not col_id:
                        reject_reason_counts["resolve_column_failed"] += 1
                        _append_sample(reject_samples["resolve_column_failed"], cand)
                        continue
                    exists = await _value_exists_in_mindsdb(
                        db_conn,
                        datasource=datasource,
                        schema=schema_id,
                        table=table_id,
                        column=col_id,
                        code_value=str(cand.code_value or ""),
                        timeout_s=1.5,
                    )
                    if not exists:
                        reject_reason_counts["value_not_in_db"] += 1
                        _append_sample(reject_samples["value_not_in_db"], cand)
                        continue
                    if strict_verify:
                        ok_vm = await _strict_verify_value_mapping_mindsdb(
                            db_conn,
                            datasource=datasource,
                            schema=schema_id,
                            table=table_id,
                            code_column=col_id,
                            code_value=str(cand.code_value or ""),
                            natural_value=str(cand.natural_value or ""),
                            timeout_s=1.5,
                        )
                        if not ok_vm:
                            reject_reason_counts["strict_verify_failed"] += 1
                            _append_sample(reject_samples["strict_verify_failed"], cand)
                            continue
                    validated_mappings.append(
                        ValueMappingCandidate(
                            schema=schema_id,
                            table=table_id,
                            column=col_id,
                            natural_value=cand.natural_value,
                            code_value=cand.code_value,
                            confidence=cand.confidence,
                            evidence=cand.evidence,
                        )
                    )
        else:
            # Unsupported DBMS for mapping validation; keep Query node only.
            reject_reason_counts["resolve_column_failed"] += len(mapping_candidates)
            for cand in mapping_candidates[:5]:
                _append_sample(reject_samples["resolve_column_failed"], cand)

        SmartLogger.log(
            "INFO",
            "cache_postprocess.value_mapping.validation_summary",
            category="cache_postprocess.value_mapping",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "candidates_count": len(mapping_candidates),
                    "validated_count": len(validated_mappings),
                    "reject_reason_counts": reject_reason_counts,
                    "reject_samples": reject_samples,
                }
            ),
            max_inline_chars=0,
        )

        # Deterministic best_context score (used for best_context merge).
        try:
            context_score = _compute_context_richness_score(
                steps_features=steps_features,
                metadata_dict=payload.get("metadata_dict") or {},
                validated_value_mappings_count=len(validated_mappings),
                row_count=payload.get("row_count") if isinstance(payload.get("row_count"), int) else None,
            )
        except Exception:
            context_score = 0.0

        # 3) Upsert Query + relationships + mappings
        repo = Neo4jQueryRepository(neo4j_session)
        await repo.setup_constraints()

        # Build compact metadata for storage in Query node.
        identified_tables = (payload.get("metadata_dict") or {}).get("identified_tables") or []
        identified_columns = (payload.get("metadata_dict") or {}).get("identified_columns") or []
        # IMPORTANT: For Neo4j cache graph (Table/Column nodes), use the configured caching DB TYPE label.
        query_db = settings.react_caching_db_type

        query_id = await repo.save_query(
            question=question,
            sql=sql,
            status=status,
            metadata={
                "identified_tables": identified_tables,
                "identified_columns": identified_columns,
                "steps_summary": steps_summary,
                "steps_features": steps_features,
                "validated_value_mappings": [cand.__dict__ for cand in validated_mappings],
                "quality_gate_json": (quality_gate_json_str or "")[:8000],
            },
            row_count=payload.get("row_count"),
            execution_time_ms=payload.get("execution_time_ms"),
            steps_count=payload.get("steps_count"),
            error_message=payload.get("error_message"),
            steps=None,  # Do not store raw steps in Neo4j.
            db=query_db,
            steps_summary=steps_summary,
            value_mappings=[cand.__dict__ for cand in validated_mappings],
            best_context_score=float(context_score),
            best_context_steps_features=dict(steps_features or {}),
            best_context_steps_summary=steps_summary,
            verified=True,
            verified_confidence=verified_confidence,
            verified_confidence_avg=verified_confidence_avg,
            verified_source="cache_postprocess",
            quality_gate_json=quality_gate_json_str,
        )
        SmartLogger.log(
            "INFO",
            "cache_postprocess.best_context.score",
            category="cache_postprocess.best_context",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "query_id": query_id,
                    "context_richness_score": float(context_score),
                    "steps_features": steps_features,
                }
            ),
            max_inline_chars=0,
        )

        # 4) Save value mappings (only validated ones)
        for cand in validated_mappings:
            column_fqn = f"{cand.schema}.{cand.table}.{cand.column}"
            await repo.save_value_mapping_by_fqn(
                natural_value=cand.natural_value,
                code_value=cand.code_value,
                column_fqn=column_fqn,
                verified=True,
                verified_confidence=float(cand.confidence or 0.0),
                verified_source="cache_postprocess",
            )

        # 5) Update Query vector for embedding search
        vec_info = await _update_query_vector(
            neo4j_session,
            query_id=query_id,
            question=question,
            steps_summary=steps_summary,
        )

        # 6) Similarity clustering (best-effort; can be disabled by settings)
        cluster_info: Dict[str, Any] = {"enabled": False}
        if bool(getattr(settings, "query_similarity_cluster_enabled", True)):
            try:
                cluster_info = await _cluster_query_by_similarity(
                    neo4j_session,
                    query_id=query_id,
                    vector_question=list(vec_info.get("vector_question") or []),
                    now_ms=int(time.time() * 1000),
                    high_threshold=float(getattr(settings, "query_similarity_high_threshold", 0.95) or 0.95),
                    mid_threshold=float(getattr(settings, "query_similarity_mid_threshold", 0.80) or 0.80),
                    link_top_k=int(getattr(settings, "query_similarity_link_top_k", 5) or 5),
                )
            except Exception as exc:
                cluster_info = {"enabled": True, "reason": f"cluster_failed:{type(exc).__name__}", "error": repr(exc)[:200]}

        SmartLogger.log(
            "INFO",
            "cache_postprocess.query_similarity_cluster.summary",
            category="cache_postprocess.query_similarity_cluster",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "query_id": query_id,
                    "cluster": cluster_info,
                }
            ),
            max_inline_chars=0,
        )

        SmartLogger.log(
            "INFO",
            "cache_postprocess.done",
            category="cache_postprocess",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "query_id": query_id,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "validated_mappings_count": len(validated_mappings),
                }
            ),
            max_inline_chars=0,
        )
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "cache_postprocess.error",
            category="cache_postprocess",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            max_inline_chars=0,
        )
        raise
    finally:
        try:
            if db_conn is not None:
                # asyncpg: await close(); aiomysql: close()+wait_closed()
                try:
                    if hasattr(db_conn, "wait_closed"):
                        db_conn.close()
                        await db_conn.wait_closed()  # type: ignore[func-returns-value]
                    else:
                        await db_conn.close()
                except Exception:
                    # Best-effort close only.
                    pass
        finally:
            if neo4j_session is not None:
                await neo4j_session.close()


async def _open_db_connection() -> asyncpg.Connection:
    # SSL mode: 'disable' -> ssl=False, other values passed as ssl parameter
    ssl_mode = settings.target_db_ssl if settings.target_db_ssl != "disable" else False
    conn = await asyncpg.connect(
        host=settings.target_db_host,
        port=settings.target_db_port,
        database=settings.target_db_name,
        user=settings.target_db_user,
        password=settings.target_db_password,
        ssl=ssl_mode,
    )
    schemas = settings.target_db_schemas.split(",")
    schemas_str = ", ".join(s.strip() for s in schemas if s.strip())
    if schemas_str:
        await conn.execute(f"SET search_path TO {schemas_str}")
    return conn


def _passes_value_mapping_policy(cand: ValueMappingCandidate) -> bool:
    natural = (cand.natural_value or "").strip()
    code = (cand.code_value or "").strip()
    if not natural or not code:
        return False
    if len(natural) < 2:
        return False
    # Mapping must be meaningful: reject identity mappings (noise, not a "mapping").
    if _norm_text_key(natural) == _norm_text_key(code):
        return False
    # Avoid gigantic strings.
    if len(code) > 128 or len(natural) > 64:
        return False

    col = (cand.column or "").strip()
    if col and col.lower() in {c.lower() for c in VALUE_MAPPING_COLUMN_NAME_DENYLIST}:
        return False

    schema = (cand.schema or "").strip()
    table = (cand.table or "").strip()
    fqn = ".".join([p for p in [schema, table, col] if p])

    # Denylist has priority.
    for pat in VALUE_MAPPING_COLUMN_FQN_DENYLIST_REGEX:
        try:
            if re.search(pat, fqn, flags=re.IGNORECASE):
                return False
        except re.error:
            # Bad regex should not break runtime; ignore.
            continue

    # If allowlist is set, require at least one match.
    if VALUE_MAPPING_COLUMN_FQN_ALLOWLIST_REGEX:
        ok = False
        for pat in VALUE_MAPPING_COLUMN_FQN_ALLOWLIST_REGEX:
            try:
                if re.search(pat, fqn, flags=re.IGNORECASE):
                    ok = True
                    break
            except re.error:
                continue
        if not ok:
            return False

    return True


async def _derive_code_mapping_from_name_column_mindsdb(
    conn: Any,
    *,
    datasource: str,
    schema: str,
    table: str,
    name_column: str,
    natural_value: str,
    timeout_s: float = 1.5,
) -> Optional[ValueMappingCandidate]:
    """
    MindsDB-only deterministic enhancement:
    If we have a name-ish column (e.g., SUJ_NAME) and a natural value, try to derive a
    corresponding code column (e.g., SUJ_CODE) by probing the DB.

    This helps avoid relying on LLM confidence for common NAME->CODE mappings.
    Returns a ValueMappingCandidate on success; otherwise None.
    """
    ds = str(datasource or "").strip()
    nat = str(natural_value or "").strip()
    if not ds or not nat:
        return None
    schema_id = _safe_ident(schema)
    table_id = _safe_ident(table)
    name_col = _safe_ident(name_column)
    if not table_id or not name_col:
        return None

    # Guess code column variants from the name column.
    # Example: SUJ_NAME -> SUJ_CODE, BPLC_NM -> BPLC_CD, NAME -> CODE.
    variants: List[str] = []
    nl = name_col.lower()
    if nl.endswith("_name"):
        variants.append(name_col[:-5] + "_CODE")
    if nl.endswith("name"):
        variants.append(name_col[:-4] + "CODE")
    if nl.endswith("_nm"):
        variants.append(name_col[:-3] + "_CD")
        variants.append(name_col[:-3] + "_CODE")
    if nl.endswith("nm"):
        variants.append(name_col[:-2] + "CD")
        variants.append(name_col[:-2] + "CODE")
    # Fallback: try a generic "<prefix>_ID" too (last resort).
    if "_" in name_col:
        base = name_col.rsplit("_", 1)[0]
        if base:
            variants.append(base + "_ID")

    # Dedup while preserving order
    seen = set()
    code_cols: List[str] = []
    for v in variants:
        vv = _safe_ident(v)
        if not vv or vv.lower() == name_col.lower():
            continue
        k = vv.lower()
        if k in seen:
            continue
        seen.add(k)
        code_cols.append(vv)
    if not code_cols:
        return None

    table_ref = ".".join([p for p in [schema_id, table_id] if p])
    nat_lit = "'" + _escape_sql_string_literal(nat) + "'"
    ex = SQLExecutor()

    for code_col in code_cols[:4]:
        probe = (
            f"SELECT DISTINCT {code_col} AS code_value "
            f"FROM {table_ref} "
            f"WHERE {name_col} = {nat_lit} AND {code_col} IS NOT NULL "
            f"LIMIT 1"
        )
        try:
            from app.core.sql_mindsdb_prepare import prepare_sql_for_mindsdb

            probe2 = prepare_sql_for_mindsdb(probe, ds).sql
        except Exception:
            continue
        try:
            prev = await ex.preview_query(conn, probe2, row_limit=1, timeout=float(timeout_s))
        except Exception:
            continue
        if int(prev.get("row_count") or 0) <= 0:
            continue
        rows = prev.get("rows") or []
        if not isinstance(rows, list) or not rows:
            continue
        row0 = rows[0] if isinstance(rows[0], list) else []
        if not row0:
            continue
        v0 = row0[0]
        if v0 is None:
            continue
        code_value = str(v0).strip()
        if not code_value:
            continue
        return ValueMappingCandidate(
            schema=schema_id,
            table=table_id,
            column=code_col,
            natural_value=nat,
            code_value=code_value,
            confidence=1.0,
            evidence=f"derived_by_db_probe: {schema_id}.{table_id}.{name_col} -> {code_col}",
        )
    return None


async def _resolve_column_case(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    column: str,
) -> Optional[Tuple[str, str, str]]:
    """
    Resolve exact identifier case via information_schema (PostgreSQL).
    Returns (schema, table, column) as stored in DB.
    """
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        # Strong gate requires DBMS-specific metadata queries. Unsupported DBMS => skip.
        return None
    q = """
    SELECT table_schema, table_name, column_name
    FROM information_schema.columns
    WHERE lower(table_schema) = lower($1)
      AND lower(table_name) = lower($2)
      AND lower(column_name) = lower($3)
    LIMIT 1
    """
    row = await conn.fetchrow(q, schema, table, column)
    if not row:
        return None
    return row["table_schema"], row["table_name"], row["column_name"]


async def _value_exists_in_db(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    column: str,
    code_value: str,
) -> bool:
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        return False
    # Resolve column type to coerce bind param (avoid asyncpg DataError on int columns).
    code_value_any: Any = code_value
    try:
        row_meta = await conn.fetchrow(
            """
            SELECT data_type, udt_name
            FROM information_schema.columns
            WHERE lower(table_schema) = lower($1)
              AND lower(table_name) = lower($2)
              AND lower(column_name) = lower($3)
            LIMIT 1
            """,
            schema,
            table,
            column,
        )
        data_type = str(row_meta.get("data_type") or "").strip().lower() if row_meta else ""
        udt_name = str(row_meta.get("udt_name") or "").strip().lower() if row_meta else ""
        raw = (code_value or "").strip()
        if raw:
            # integers
            if data_type in {"integer", "bigint", "smallint"} or udt_name in {"int2", "int4", "int8"}:
                code_value_any = int(raw)
            # numerics
            elif data_type in {"numeric", "decimal", "real", "double precision"} or udt_name in {"numeric", "float4", "float8"}:
                code_value_any = float(raw)
            # booleans
            elif data_type == "boolean" or udt_name == "bool":
                if raw.lower() in {"true", "t", "1", "y", "yes"}:
                    code_value_any = True
                elif raw.lower() in {"false", "f", "0", "n", "no"}:
                    code_value_any = False
                else:
                    return False
    except Exception:
        # If type resolution/coercion fails, do not crash the worker; fail-closed.
        return False
    # Safe-quote identifiers by doubling quotes.
    def q_ident(x: str) -> str:
        return '"' + x.replace('"', '""') + '"'

    sql = (
        f"SELECT 1 AS one FROM {q_ident(schema)}.{q_ident(table)} "
        f"WHERE {q_ident(column)} = $1 LIMIT 1"
    )
    try:
        row = await conn.fetchrow(sql, code_value_any)
        return bool(row)
    except Exception:
        # Most commonly: DataError due to bind type mismatch (e.g., '8537' into int4).
        return False


async def _llm_build_steps_summary(
    *,
    question: str,
    sql: str,
    metadata: Dict[str, Any],
    steps: List[Dict[str, Any]],
) -> str:
    """
    Returns a compact JSON string.
    This is intentionally stored as text for portability.
    """
    # Trim steps to reduce token usage.
    trimmed_steps = steps[-10:] if isinstance(steps, list) else []
    user_prompt = {
        "question": question,
        "sql": sql,
        "metadata": metadata,
        "steps_tail": trimmed_steps,
        "requirements": {
            "language": "ko",
            "output_format": "json",
            "must_include_fields": [
                "intent",
                "tables",
                "columns",
                "filters",
                "aggregations",
                "group_by",
                "order_by",
                "time_range",
                "notes",
            ],
            "max_chars": 3000,
        },
    }

    llm = create_llm(
        purpose="cache_postprocess.steps_summary",
        thinking_level="low",
        include_thoughts=False,
        temperature=0.1,
        max_output_tokens=900,
    ).llm
    resp = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "당신은 Text2SQL 실행 과정을 요약하여 캐시/재사용에 필요한 최소 정보만 추출하는 전문가입니다. "
                    "반드시 JSON만 출력하세요(설명문 금지)."
                )
            ),
            HumanMessage(content=json.dumps(user_prompt, ensure_ascii=False, default=str)),
        ]
    )
    text = _llm_content_to_text(getattr(resp, "content", None)).strip()
    text = _strip_code_fences(text)
    # Best-effort: ensure it's JSON.
    try:
        obj = json.loads(text)
    except Exception:
        obj = {"intent": "", "tables": [], "columns": [], "filters": [], "aggregations": [], "notes": text[:1000]}
    return json.dumps(obj, ensure_ascii=False)


async def _llm_extract_value_mappings(
    *,
    question: str,
    sql: str,
    metadata: Dict[str, Any],
    steps_summary: str,
) -> List[ValueMappingCandidate]:
    """
    Produce candidates for ValueMapping with schema/table/column hints.
    LLM output is validated by strong gate before saving.
    """
    prompt = {
        "question": question,
        "sql": sql,
        "metadata": metadata,
        "steps_summary_json": steps_summary,
        "task": (
            "질문에서 사용자가 말한 자연어 값(natural_value)과 SQL/메타데이터에 사용된 코드/식별자(code_value)를 찾아 "
            "ValueMapping 후보를 생성하세요."
        ),
        "output_schema": {
            "type": "array",
            "items": {
                "schema": "string",
                "table": "string",
                "column": "string",
                "natural_value": "string",
                "code_value": "string",
                "confidence": "number",
                "evidence": "string",
            },
        },
        "constraints": {
            "language": "ko",
            "no_generic_terms": True,
            "max_items": 10,
        },
    }

    llm = create_llm(
        purpose="cache_postprocess.value_mappings",
        thinking_level="low",
        include_thoughts=False,
        temperature=0.1,
        max_output_tokens=900,
    ).llm
    resp = await llm.ainvoke(
        [
            SystemMessage(
                content=(
                    "당신은 데이터베이스 ValueMapping 후보를 추출하는 시스템입니다. "
                    "반드시 JSON 배열만 출력하세요(설명문/마크다운 금지)."
                )
            ),
            HumanMessage(content=json.dumps(prompt, ensure_ascii=False, default=str)),
        ]
    )
    text = _llm_content_to_text(getattr(resp, "content", None)).strip()
    text = _strip_code_fences(text)
    try:
        arr = json.loads(text)
    except Exception:
        SmartLogger.log(
            "WARNING",
            "cache_postprocess.value_mapping.llm_parse_failed",
            category="cache_postprocess.value_mapping",
            params=sanitize_for_log(
                {
                    "llm_provider": settings.llm_provider,
                    "llm_model": settings.llm_model,
                    "text_len": len(text),
                    "text_head": text[:500],
                }
            ),
            max_inline_chars=0,
        )
        arr = []

    candidates: List[ValueMappingCandidate] = []
    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue
            candidates.append(
                ValueMappingCandidate(
                    schema=str(item.get("schema") or "").strip(),
                    table=str(item.get("table") or "").strip(),
                    column=str(item.get("column") or "").strip(),
                    natural_value=str(item.get("natural_value") or "").strip(),
                    code_value=str(item.get("code_value") or "").strip(),
                    confidence=float(item.get("confidence") or 0.0),
                    evidence=str(item.get("evidence") or "").strip(),
                )
            )

    # Basic de-dup
    seen = set()
    uniq: List[ValueMappingCandidate] = []
    for c in candidates:
        key = (c.schema.lower(), c.table.lower(), c.column.lower(), c.natural_value, c.code_value)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _llm_content_to_text(content: Any) -> str:
    """
    Normalize LangChain message content to text.
    Gemini may emit list/dict parts (including 'thinking'); we only keep 'text'.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content and str(content.get("text") or ""):
            return str(content.get("text") or "")
        return ""
    if isinstance(content, list):
        out: List[str] = []
        for part in content:
            if isinstance(part, str) and part:
                out.append(part)
                continue
            if isinstance(part, dict) and "text" in part and str(part.get("text") or ""):
                out.append(str(part.get("text") or ""))
                continue
        return "".join(out)
    return str(content)


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    # Best-effort: remove outermost markdown fence
    lines = s.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


async def _update_query_vector(
    neo4j_session,
    *,
    query_id: str,
    question: str,
    steps_summary: str,
) -> Dict[str, Any]:
    try:
        embedder = create_embedding_client()

        # Extract a stable, 1-line intent text from steps_summary JSON.
        intent_text = ""
        try:
            obj = json.loads(steps_summary or "")
            if isinstance(obj, dict):
                intent_text = str(obj.get("intent") or "").strip()
        except Exception:
            intent_text = ""
        if not intent_text:
            # Fallback: keep it strict and simple (no extra LLM here).
            intent_text = (question or "").strip()

        # Embed question + intent (batch to reduce network calls).
        q_text = (question or "").strip()[:8000]
        i_text = (intent_text or "").strip()[:8000]
        vectors = await embedder.embed_batch([q_text, i_text])
        vector_question = vectors[0] if vectors and len(vectors) > 0 else []
        vector_intent = vectors[1] if vectors and len(vectors) > 1 else []

        query = """
        MATCH (q:Query {id: $id})
        SET q.vector_question = $vector_question,
            q.vector_question_updated_at = datetime(),
            q.vector_intent = $vector_intent,
            q.vector_intent_updated_at = datetime(),
            q.intent_text = $intent_text,
            q.intent_text_updated_at = datetime()
        """
        await neo4j_session.run(
            query,
            id=query_id,
            vector_question=vector_question,
            vector_intent=vector_intent,
            intent_text=intent_text,
        )
        return {
            "intent_text": intent_text,
            "vector_question": vector_question,
            "vector_intent": vector_intent,
        }
    except Exception as exc:
        SmartLogger.log(
            "WARNING",
            "cache_postprocess.query_vector.update_failed",
            category="cache_postprocess.query_vector",
            params=sanitize_for_log({"query_id": query_id, "exception": repr(exc)}),
            max_inline_chars=0,
        )
        return {"intent_text": "", "vector_question": [], "vector_intent": []}


async def _cluster_query_by_similarity(
    neo4j_session,
    *,
    query_id: str,
    vector_question: List[float],
    now_ms: int,
    high_threshold: float,
    mid_threshold: float,
    link_top_k: int,
) -> Dict[str, Any]:
    """
    Best-effort clustering:
    - Find similar completed queries by question vector
    - Assign canonical_id based on highest-similarity match when similarity>=high_threshold
    - Create SIMILAR_TO links to top-K matches with similarity>=mid_threshold
    """
    if not vector_question:
        return {"enabled": True, "reason": "no_vector_question", "linked": 0, "canonical_id": None}

    fetch_k = max(10, int(link_top_k) * 6)
    link_top_k = max(0, int(link_top_k))
    try:
        res = await neo4j_session.run(
            """
            CALL db.index.vector.queryNodes('query_question_vec_index', $fetch_k, $embedding)
            YIELD node, score
            WITH node, score
            WHERE node:Query
              AND node.status = 'completed'
              AND node.id IS NOT NULL
              AND node.id <> $id
              AND score >= $mid_threshold
            RETURN node.id AS id,
                   COALESCE(node.canonical_id, node.id) AS canonical_id,
                   node.best_context_score AS best_context_score,
                   score AS similarity_score
            ORDER BY similarity_score DESC
            LIMIT $fetch_k
            """,
            id=query_id,
            fetch_k=int(fetch_k),
            embedding=vector_question,
            mid_threshold=float(mid_threshold),
        )
        rows = await res.data()
    except Exception as exc:
        return {"enabled": True, "reason": f"search_failed:{type(exc).__name__}", "error": str(exc)[:180]}

    top = rows[0] if rows else None
    top_score = float(top.get("similarity_score") or 0.0) if isinstance(top, dict) else 0.0
    top_canonical = str(top.get("canonical_id") or "") if isinstance(top, dict) else ""
    canonical_id = top_canonical if (top_canonical and top_score >= float(high_threshold)) else query_id

    # Assign canonical_id
    try:
        await neo4j_session.run(
            """
            MATCH (q:Query {id: $id})
            SET q.canonical_id = $canonical_id,
                q.canonical_updated_at_ms = $now_ms,
                q.canonical_similarity_score = $top_score
            """,
            id=query_id,
            canonical_id=canonical_id,
            now_ms=int(now_ms),
            top_score=float(top_score),
        )
    except Exception:
        pass

    # Create SIMILAR_TO links (best-effort)
    linked = 0
    if link_top_k > 0 and rows:
        links = [
            {"id": str(r.get("id") or ""), "score": float(r.get("similarity_score") or 0.0)}
            for r in rows[:link_top_k]
            if isinstance(r, dict) and str(r.get("id") or "").strip()
        ]
        try:
            await neo4j_session.run(
                """
                MATCH (q:Query {id: $id})
                WITH q
                UNWIND $links AS l
                MATCH (o:Query {id: l.id})
                MERGE (q)-[r:SIMILAR_TO]->(o)
                SET r.score = l.score,
                    r.updated_at_ms = $now_ms
                """,
                id=query_id,
                links=links,
                now_ms=int(now_ms),
            )
            linked = len(links)
        except Exception:
            linked = 0

    return {
        "enabled": True,
        "candidates": len(rows),
        "top_score": float(top_score),
        "canonical_id": canonical_id,
        "linked": int(linked),
        "high_threshold": float(high_threshold),
        "mid_threshold": float(mid_threshold),
    }


