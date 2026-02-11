from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

from app.smart_logger import SmartLogger

from .db_probe import _limited_db_probe
from .models import ColumnCandidate
from .text import _GENERIC_TERMS
from .xml_utils import emit_text


async def resolve_values_and_append_xml(
    *,
    context,
    value_mappings: Sequence[Dict[str, Any]],
    fallback_terms: Sequence[str],
    column_candidates: Sequence[ColumnCandidate],
    result_parts: List[str],
) -> List[Dict[str, str]]:
    """
    Resolve user terms into actual DB values:
    1) ValueMappings
    2) limited DB probe (strict budget)
    Appends <resolved_values> to XML and returns resolved list.
    """
    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.resolved_values.start",
        category="react.tool.detail.build_sql_context",
        params={
            "value_mappings_count": len(list(value_mappings or [])),
            "fallback_terms_for_resolution": list(fallback_terms)[:10],
        },
        max_inline_chars=0,
    )

    resolved: List[Dict[str, str]] = []
    resolved_terms_lower = set()

    # Priority 1: Value Mappings
    for vm in list(value_mappings or [])[:20]:
        natural = str(vm.get("natural_value") or "").strip()
        code = str(vm.get("code_value") or "").strip()
        col_fqn = str(vm.get("column_fqn") or "").strip()
        if not natural or not code:
            continue
        resolved.append(
            {
                "user_term": natural,
                "actual_value": code,
                "source": "value_mapping",
                "column_fqn": col_fqn,
            }
        )
        resolved_terms_lower.add(natural.lower())

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.resolved_values.value_mappings_applied",
        category="react.tool.detail.build_sql_context",
        params={"resolved_from_mappings": [r for r in resolved if r["source"] == "value_mapping"]},
        max_inline_chars=0,
    )

    # Priority 2: limited DB probe (strict)
    probe_budget = 2
    probe_timeout_s = min(2.0, float(getattr(context, "max_sql_seconds", 60)))
    value_limit = max(1, min(10, int(context.scaled(getattr(context, "value_limit", 10)))))
    col_list = list(column_candidates or [])

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.resolved_values.db_probe_start",
        category="react.tool.detail.build_sql_context",
        params={
            "probe_budget": probe_budget,
            "probe_timeout_s": probe_timeout_s,
            "value_limit": value_limit,
            "terms_to_probe": [
                t
                for t in list(fallback_terms)[:10]
                if t.lower() not in resolved_terms_lower and t not in _GENERIC_TERMS
            ],
        },
        max_inline_chars=0,
    )

    probe_attempts: List[Dict[str, Any]] = []
    for term in list(fallback_terms)[:10]:
        if probe_budget <= 0:
            break
        if term.lower() in resolved_terms_lower:
            continue
        if term in _GENERIC_TERMS:
            continue

        # choose a single best text-ish column candidate
        chosen: Optional[ColumnCandidate] = None
        for col in col_list[: min(20, len(col_list))]:
            dtype = (col.dtype or "").lower()
            if any(x in dtype for x in ("char", "text", "varchar")) or dtype == "":
                chosen = col
                break
        if not chosen:
            break

        probe_start = time.perf_counter()
        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.resolved_values.db_probe_attempt",
            category="react.tool.detail.build_sql_context",
            params={
                "term": term,
                "chosen_column": {
                    "table_schema": chosen.table_schema,
                    "table_name": chosen.table_name,
                    "name": chosen.name,
                    "dtype": chosen.dtype,
                    "column_fqn": chosen.column_fqn,
                },
                "remaining_budget": probe_budget,
            },
            max_inline_chars=0,
        )

        values = await _limited_db_probe(
            context=context,
            keyword=term,
            column=chosen,
            timeout_s=probe_timeout_s,
            value_limit=value_limit,
        )
        probe_budget -= 1
        probe_elapsed = time.perf_counter() - probe_start

        probe_attempt_result = {
            "term": term,
            "chosen_column_fqn": chosen.column_fqn,
            "values_found": values,
            "elapsed_ms": probe_elapsed * 1000.0,
        }
        probe_attempts.append(probe_attempt_result)

        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.resolved_values.db_probe_result",
            category="react.tool.detail.build_sql_context",
            params=probe_attempt_result,
            max_inline_chars=0,
        )

        if not values:
            continue
        resolved.append(
            {
                "user_term": term,
                "actual_value": values[0],
                "source": "db_probe",
                "column_fqn": chosen.column_fqn,
            }
        )
        resolved_terms_lower.add(term.lower())

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.resolved_values.db_probe_done",
        category="react.tool.detail.build_sql_context",
        params={
            "probe_attempts_count": len(probe_attempts),
            "probe_attempts": probe_attempts,
            "resolved_from_db_probe": [r for r in resolved if r["source"] == "db_probe"],
        },
        max_inline_chars=0,
    )

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.resolved_values.summary",
        category="react.tool.detail.build_sql_context",
        params={
            "total_resolved_count": len(resolved),
            "all_resolved_values": resolved,
            "breakdown_by_source": {
                "value_mapping": len([r for r in resolved if r["source"] == "value_mapping"]),
                "db_probe": len([r for r in resolved if r["source"] == "db_probe"]),
            },
        },
        max_inline_chars=0,
    )

    # XML
    result_parts.append("<resolved_values>")
    for rv in resolved[:30]:
        result_parts.append("<value>")
        result_parts.append(emit_text("user_term", rv.get("user_term", "")))
        result_parts.append(emit_text("actual_value", rv.get("actual_value", "")))
        result_parts.append(emit_text("source", rv.get("source", "")))
        if rv.get("column_fqn"):
            result_parts.append(emit_text("column_fqn", rv.get("column_fqn", "")))
        result_parts.append("</value>")
    result_parts.append("</resolved_values>")

    return resolved


__all__ = ["resolve_values_and_append_xml"]


