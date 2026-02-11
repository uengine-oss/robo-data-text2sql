from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import re
from xml.sax.saxutils import escape as xml_escape

from app.react.generators.intent_extract_generator import get_intent_extract_generator
from app.smart_logger import SmartLogger

from .neo4j import _neo4j_find_similar_queries_and_mappings
from .text import _truncate
from .xml_utils import emit_text, to_cdata


@dataclass(frozen=True)
class SimilarFlowResult:
    similar_queries: List[Dict[str, Any]]
    value_mappings: List[Dict[str, Any]]
    intent_embedding: Optional[List[float]]


def _try_parse_json_dict(x: Any) -> Optional[Dict[str, Any]]:
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        return None
    s = (x or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _compact_steps_features(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Keep only low-token, high-signal fields for LLM context.
    (The full tables_used/columns_used remain stored on Query nodes for debugging/analytics.)
    """
    if not obj:
        return {}

    def cap_str(v: Any, n: int) -> str:
        s = str(v or "").strip()
        return s[:n]

    def cap_list(v: Any, max_items: int, item_n: int) -> List[str]:
        if not isinstance(v, list):
            return []
        out: List[str] = []
        for x in v:
            s = str(x or "").strip()
            if not s:
                continue
            out.append(cap_str(s, item_n))
            if len(out) >= max_items:
                break
        return out

    compact: Dict[str, Any] = {}
    if cap_str(obj.get("intent"), 240):
        compact["intent"] = cap_str(obj.get("intent"), 240)
    if cap_list(obj.get("tables"), 8, 140):
        compact["tables"] = cap_list(obj.get("tables"), 8, 140)
    if cap_list(obj.get("filters"), 10, 220):
        compact["filters"] = cap_list(obj.get("filters"), 10, 220)
    if cap_list(obj.get("aggregations"), 6, 140):
        compact["aggregations"] = cap_list(obj.get("aggregations"), 6, 140)
    if cap_list(obj.get("group_by"), 6, 140):
        compact["group_by"] = cap_list(obj.get("group_by"), 6, 140)
    if cap_list(obj.get("order_by"), 6, 140):
        compact["order_by"] = cap_list(obj.get("order_by"), 6, 140)
    if cap_str(obj.get("time_range"), 220):
        compact["time_range"] = cap_str(obj.get("time_range"), 220)
    # notes are often verbose; keep short only
    if cap_str(obj.get("notes"), 200):
        compact["notes"] = cap_str(obj.get("notes"), 200)
    return compact


_RE_QIDENT = re.compile(r'"([^"]+)"\."([^"]+)"')


def _parse_json_list(x: Any) -> Optional[List[Any]]:
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if not isinstance(x, str):
        return None
    s = (x or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else None
    except Exception:
        return None


def _tables_in_sql(sql: str, *, limit: int = 20) -> List[str]:
    """
    Extract "SCHEMA"."TABLE" pairs from SQL.
    Returns ["SCHEMA.TABLE", ...] with order preserved (best-effort).
    """
    s = str(sql or "")
    out: List[str] = []
    seen = set()
    for m in _RE_QIDENT.finditer(s):
        sch = (m.group(1) or "").strip()
        tb = (m.group(2) or "").strip()
        if not sch or not tb:
            continue
        key = f"{sch}.{tb}"
        key_l = key.lower()
        if key_l in seen:
            continue
        seen.add(key_l)
        out.append(key)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _compact_tables_used(*, tables_used: Any, sql: str, limit: int = 10) -> List[str]:
    """
    Keep only tables that appear in the SQL (intersection), capped.
    Falls back to tables parsed from SQL when needed.
    """
    sql_tables = _tables_in_sql(sql, limit=30)
    sql_set = {t.lower() for t in sql_tables}
    tu = _parse_json_list(tables_used) or []
    out: List[str] = []
    seen = set()
    for x in tu:
        s = str(x or "").strip()
        if not s:
            continue
        if s.lower() not in sql_set:
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    if out:
        return out
    return sql_tables[: max(1, int(limit))]


def _compact_columns_used(*, columns_used: Any, keep_tables: Sequence[str], limit: int = 18) -> List[str]:
    """
    Compact columns_used to reduce token noise:
    - Keep only columns whose table is in keep_tables
    - Keep only columns likely to matter for disambiguation (time/value/name/code/join keys)
    """
    keep_tables_l = {str(t or "").strip().lower() for t in (keep_tables or []) if str(t or "").strip()}
    cu = _parse_json_list(columns_used) or []
    out: List[str] = []
    seen = set()

    def important(col_fqn: str) -> bool:
        # Expect SCHEMA.TABLE.COL
        parts = col_fqn.lower().split(".")
        if len(parts) < 3:
            return False
        col = parts[-1]
        # high-signal patterns
        # NOTE: avoid domain-overfit tokens; keep generic names only.
        if col in ("log_time", "val", "value", "cnt"):
            return True
        if "time" in col or "date" in col or col.endswith("_dt") or col.endswith("_tm"):
            return True
        if "unit" in col:
            return True
        if "name" in col or col.endswith("_nm") or col.endswith("nm"):
            return True
        if "code" in col or col.endswith("_cd") or col.endswith("_code") or col.endswith("_id") or col.endswith("id"):
            return True
        return False

    for x in cu:
        s = str(x or "").strip()
        if not s or s.lower() in seen:
            continue
        parts = s.lower().split(".")
        if len(parts) < 3:
            continue
        table_key = ".".join(parts[:2])
        if keep_tables_l and table_key not in keep_tables_l:
            continue
        if not important(s):
            continue
        seen.add(s.lower())
        out.append(s)
        if len(out) >= max(1, int(limit)):
            break
    return out


async def build_similar_and_append_xml(
    *,
    context,
    embedder,
    question: str,
    react_run_id: str | None,
    question_embedding: List[float],
    fallback_terms: Sequence[str],
    result_parts: List[str],
    min_similarity: float = 0.3,
) -> SimilarFlowResult:
    q = (question or "").strip()

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.similar_queries_and_mappings.start",
        category="react.tool.detail.build_sql_context",
        params={
            "question": q,
            "terms": list(fallback_terms),
            "min_similarity": float(min_similarity),
        },
        max_inline_chars=0,
    )

    similar_queries: List[Dict[str, Any]] = []
    value_mappings: List[Dict[str, Any]] = []
    intent_embedding: Optional[List[float]] = None
    try:
        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.intent_extract.start",
            category="react.tool.detail.build_sql_context",
            params={"react_run_id": react_run_id, "question": _truncate(q, 200)},
            max_inline_chars=0,
        )

        intent_text, intent_mode = await get_intent_extract_generator().generate(
            question=q, react_run_id=react_run_id
        )
        if intent_text:
            intent_embedding = await embedder.embed_text(intent_text[:8000])

        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.intent_extract.done",
            category="react.tool.detail.build_sql_context",
            params={
                "react_run_id": react_run_id,
                "mode": intent_mode,
                "intent_text": _truncate(intent_text or "", 200),
            },
            max_inline_chars=0,
        )

        similar_queries, value_mappings = await _neo4j_find_similar_queries_and_mappings(
            context=context,
            question=q,
            question_embedding=question_embedding,
            intent_embedding=intent_embedding,
            terms=list(fallback_terms),
            min_similarity=float(min_similarity),
        )

        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.similar_queries_and_mappings.done",
            category="react.tool.detail.build_sql_context",
            params={
                "similar_queries_count": len(similar_queries),
                "similar_queries": similar_queries,
                "value_mappings_count": len(value_mappings),
                "value_mappings": value_mappings,
            },
            max_inline_chars=0,
        )
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "react.build_sql_context.similar_queries_and_mappings.error",
            category="react.tool.detail.build_sql_context",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )
        result_parts.append(f"<warning>similar_query_lookup_failed: {xml_escape(str(exc)[:160])}</warning>")

    # XML: similar_queries
    result_parts.append("<similar_queries>")
    # Keep this block low-token: it is often fed back into LLM context.
    # Full `tables_used/columns_used` remain on Query nodes for debugging/analytics.
    for sq in (similar_queries or [])[:3]:
        result_parts.append("<query>")
        result_parts.append(emit_text("similarity_score", str(sq.get("similarity_score", ""))))
        result_parts.append(emit_text("original_question", _truncate(str(sq.get("question") or ""), 400)))
        # Similar query SQL is useful but can be huge; cap aggressively.
        sql_text = str(sq.get("sql") or "")
        result_parts.append(f"<sql>{to_cdata(_truncate(sql_text, 3200))}</sql>")
        if sq.get("intent_text") is not None:
            result_parts.append(emit_text("intent_text", _truncate(str(sq.get("intent_text") or ""), 300)))
        if sq.get("best_context_score") is not None:
            result_parts.append(emit_text("best_context_score", str(sq.get("best_context_score") or "")))

        # Prefer steps_features (low-token, stable) over raw tables_used/columns_used.
        sf = _try_parse_json_dict(sq.get("best_context_steps_features"))
        sf_compact = _compact_steps_features(sf)
        if sf_compact:
            result_parts.append(emit_text("steps_features", json.dumps(sf_compact, ensure_ascii=False)))

        # Provide compact table/column hints (intersection with the SQL), NOT full lists.
        # This often helps the LLM recognize time-grain tables (e.g. daily tables) without flooding context.
        tables_hint = _compact_tables_used(tables_used=sq.get("tables_used"), sql=sql_text, limit=10)
        if tables_hint:
            result_parts.append(emit_text("tables_used_compact", json.dumps(tables_hint, ensure_ascii=False)))
        cols_hint = _compact_columns_used(
            columns_used=sq.get("columns_used"),
            keep_tables=tables_hint,
            limit=18,
        )
        if cols_hint:
            result_parts.append(emit_text("columns_used_compact", json.dumps(cols_hint, ensure_ascii=False)))
        result_parts.append("</query>")
    result_parts.append("</similar_queries>")

    # XML: value_mappings
    result_parts.append("<value_mappings>")
    for vm in (value_mappings or [])[:20]:
        result_parts.append("<mapping>")
        result_parts.append(emit_text("natural_value", str(vm.get("natural_value") or "")))
        result_parts.append(emit_text("code_value", str(vm.get("code_value") or "")))
        result_parts.append(emit_text("column_fqn", str(vm.get("column_fqn") or "")))
        result_parts.append(emit_text("usage_count", str(vm.get("usage_count") or "")))
        result_parts.append("</mapping>")
    result_parts.append("</value_mappings>")

    return SimilarFlowResult(
        similar_queries=list(similar_queries or []),
        value_mappings=list(value_mappings or []),
        intent_embedding=intent_embedding,
    )


__all__ = ["SimilarFlowResult", "build_similar_and_append_xml"]


