from __future__ import annotations

import json
import time
import traceback
import re
from typing import Any, Dict, List, Sequence

from app.smart_logger import SmartLogger

from .models import ColumnCandidate, TableCandidate
from .neo4j import _neo4j_fetch_table_schemas
from .xml_utils import emit_text


def _tfqn_l(schema: str, table: str) -> str:
    s = (schema or "").strip().lower()
    t = (table or "").strip().lower()
    return f"{s}.{t}" if s else t


def _parse_enum_values(raw: Any) -> List[Dict[str, Any]]:
    """
    Column.enum_values is stored as a JSON string (see cache router).
    Returns: list of {"value": str, "count": int}
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        # Already parsed upstream (rare).
        out = []
        for x in raw:
            if isinstance(x, dict) and x.get("value") is not None:
                out.append({"value": str(x.get("value")), "count": int(x.get("count") or 0)})
        return out
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            obj = json.loads(s)
        except Exception:
            return []
        if isinstance(obj, list):
            out = []
            for x in obj:
                if isinstance(x, dict) and x.get("value") is not None:
                    out.append({"value": str(x.get("value")), "count": int(x.get("count") or 0)})
            return out
    return []

_KOREAN_ADMIN_SUFFIXES = ("시", "군", "구", "도", "읍", "면", "동")
_KOREAN_TRAILING_PARTICLES = (
    "의",
    "을",
    "를",
    "은",
    "는",
    "이",
    "가",
    "과",
    "와",
    "에",
    "에서",
    "으로",
    "로",
    "부터",
    "까지",
    "및",
)


def _expand_search_terms(terms: Sequence[str]) -> List[str]:
    """
    Expand terms for fuzzy value matching (generic, language-aware).
    - Adds variants stripping common Korean administrative suffixes: 청주시 -> 청주.
    - Adds variants stripping common trailing particles: 청주정수장의 -> 청주정수장.
    - Keeps only tokens length>=2 to avoid noise.
    """
    out: List[str] = []
    seen = set()
    for t in terms or []:
        s = str(t or "").strip()
        if not s:
            continue
        cand = [s]
        # Strip common particles
        for suf in _KOREAN_TRAILING_PARTICLES:
            if s.endswith(suf) and len(s) >= 3:
                cand.append(s[: -len(suf)].strip())
        for suf in _KOREAN_ADMIN_SUFFIXES:
            if s.endswith(suf) and len(s) >= 3:
                cand.append(s[: -len(suf)])
        for x in cand:
            x = x.strip()
            if len(x) < 2:
                continue
            k = x.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(x)
    return out[:20]


def _pick_values_with_query_match(
    *,
    enum_values: List[Dict[str, Any]],
    search_terms: Sequence[str],
    max_matches: int,
) -> List[Dict[str, Any]]:
    """
    Pick enum values whose text contains any search term (case-insensitive).
    Returns up to max_matches, preserving original order (usually count-desc).
    """
    if not enum_values or not search_terms:
        return []
    terms_l = [str(t or "").strip().lower() for t in search_terms if str(t or "").strip()]
    out: List[Dict[str, Any]] = []
    seen = set()
    for ev in enum_values:
        v = str(ev.get("value") or "").strip()
        if not v:
            continue
        vl = v.lower()
        if any(t in vl for t in terms_l):
            if vl in seen:
                continue
            seen.add(vl)
            out.append(ev)
            if len(out) >= int(max_matches):
                break
    return out


async def append_column_value_hints_xml(
    *,
    context,
    selected_tables: Sequence[TableCandidate],
    per_table_columns: Dict[str, List[ColumnCandidate]],
    table_schemas: Sequence[Dict[str, Any]] | None = None,
    result_parts: List[str],
    value_limit: int = 10,
    fallback_terms: Sequence[str] | None = None,
) -> None:
    """
    Append cached enum-like value hints per selected table/column.

    Output XML:
    <column_value_hints>
      <table>
        <schema>...</schema><name>...</name>
        <columns>
          <column>
            <name>...</name><dtype>...</dtype><cardinality>...</cardinality>
            <values>
              <value><text>...</text><count>...</count></value>
            </values>
          </column>
        </columns>
      </table>
    </column_value_hints>
    """
    started = time.perf_counter()
    try:
        value_limit_i = max(1, min(50, int(value_limit)))
        tables = list(selected_tables or [])
        if not tables:
            return

        # Fetch full schema metadata (includes enum_values + cardinality).
        # Allow caller to pass pre-fetched schemas to avoid duplicate Neo4j calls.
        if table_schemas is None:
            table_schemas = await _neo4j_fetch_table_schemas(context=context, tables=tables)
        meta_by_table: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for t in table_schemas or []:
            tfqn = _tfqn_l(str(t.get("schema") or ""), str(t.get("name") or ""))
            cols = t.get("columns") or []
            by_name: Dict[str, Dict[str, Any]] = {}
            for c in cols:
                if not isinstance(c, dict):
                    continue
                nm = str(c.get("name") or "").strip()
                if not nm:
                    continue
                by_name[nm.lower()] = c
            meta_by_table[tfqn] = by_name

        emitted_tables = 0
        emitted_columns = 0
        search_terms = _expand_search_terms(list(fallback_terms or []))
        result_parts.append("<column_value_hints>")

        for t in tables:
            tfqn = _tfqn_l(t.schema, t.name)
            selected_cols = list(per_table_columns.get(tfqn, []) or [])
            if not selected_cols:
                continue

            meta_cols = meta_by_table.get(tfqn, {})

            # Only include columns that have cached enum_values.
            cols_with_values: List[Dict[str, Any]] = []
            for c in selected_cols:
                nm = (c.name or "").strip()
                if not nm:
                    continue
                meta = meta_cols.get(nm.lower()) or {}
                enum_vals = _parse_enum_values(meta.get("enum_values"))
                if not enum_vals:
                    continue
                # Ensure query-matching values are included even if not in top-N.
                matched = _pick_values_with_query_match(
                    enum_values=enum_vals,
                    search_terms=search_terms,
                    max_matches=min(5, value_limit_i),
                )
                # Merge matched + top-N (dedup), bounded by value_limit_i.
                merged: List[Dict[str, Any]] = []
                seen_vals = set()
                for ev in list(matched) + list(enum_vals):
                    txt = str(ev.get("value") or "").strip()
                    if not txt:
                        continue
                    k = txt.lower()
                    if k in seen_vals:
                        continue
                    seen_vals.add(k)
                    merged.append(ev)
                    if len(merged) >= value_limit_i:
                        break
                cols_with_values.append(
                    {
                        "name": nm,
                        "dtype": str(meta.get("dtype") or c.dtype or ""),
                        "cardinality": meta.get("cardinality"),
                        "enum_values": merged[:value_limit_i],
                    }
                )

            if not cols_with_values:
                continue

            emitted_tables += 1
            result_parts.append("<table>")
            result_parts.append(emit_text("schema", str(t.schema or "")))
            result_parts.append(emit_text("name", str(t.name or "")))
            result_parts.append("<columns>")
            for col in cols_with_values:
                emitted_columns += 1
                result_parts.append("<column>")
                result_parts.append(emit_text("name", str(col.get("name") or "")))
                result_parts.append(emit_text("dtype", str(col.get("dtype") or "")))
                if col.get("cardinality") is not None:
                    result_parts.append(emit_text("cardinality", str(col.get("cardinality") or "")))
                result_parts.append("<values>")
                for ev in list(col.get("enum_values") or [])[:value_limit_i]:
                    result_parts.append("<value>")
                    result_parts.append(emit_text("text", str(ev.get("value") or "")))
                    result_parts.append(emit_text("count", str(ev.get("count") or 0)))
                    result_parts.append("</value>")
                result_parts.append("</values>")
                result_parts.append("</column>")
            result_parts.append("</columns>")
            result_parts.append("</table>")

        result_parts.append("</column_value_hints>")

        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.column_value_hints.done",
            category="react.tool.detail.build_sql_context",
            params={
                "tables_count": len(tables),
                "tables_emitted": int(emitted_tables),
                "columns_emitted": int(emitted_columns),
                "value_limit": int(value_limit_i),
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            },
            max_inline_chars=0,
        )
    except Exception as exc:
        SmartLogger.log(
            "WARNING",
            "react.build_sql_context.column_value_hints.error",
            category="react.tool.detail.build_sql_context",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )
        result_parts.append(f"<warning>column_value_hints_failed: {str(exc)[:160]}</warning>")


__all__ = ["append_column_value_hints_xml"]


