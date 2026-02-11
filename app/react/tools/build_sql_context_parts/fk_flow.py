from __future__ import annotations

import traceback
from typing import Any, Dict, List, Sequence
from xml.sax.saxutils import escape as xml_escape

from app.smart_logger import SmartLogger

from .models import TableCandidate
from .neo4j import _neo4j_fetch_fk_relationships
from .xml_utils import emit_text


async def fetch_fk_relationships_and_append_xml(
    *,
    context,
    selected_tables: Sequence[TableCandidate],
    result_parts: List[str],
    limit: int = 50,
    fk_relationships: Sequence[Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    try:
        if fk_relationships is None:
            fqns: List[str] = []
            for t in selected_tables:
                if t.schema and t.name:
                    fqns.append(f"{t.schema}.{t.name}")

            SmartLogger.log(
                "DEBUG",
                "react.build_sql_context.fk_search_start",
                category="react.tool.detail.build_sql_context",
                params={"table_fqns": fqns, "limit": int(limit)},
                max_inline_chars=0,
            )

            fk_relationships = await _neo4j_fetch_fk_relationships(
                context=context, table_fqns=fqns, limit=int(limit)
            )

        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.fk_search_done",
            category="react.tool.detail.build_sql_context",
            params={"fk_relationships_count": len(fk_relationships), "fk_relationships": fk_relationships},
            max_inline_chars=0,
        )
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "react.build_sql_context.fk_search_error",
            category="react.tool.detail.build_sql_context",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )
        result_parts.append(f"<warning>fk_relationship_fetch_failed: {xml_escape(str(exc)[:160])}</warning>")
        fk_relationships = []

    if fk_relationships:
        result_parts.append("<fk_relationships>")
        for rel in fk_relationships[: int(limit)]:
            result_parts.append("<fk>")
            result_parts.append(emit_text("from_schema", str(rel.get("from_schema") or "")))
            result_parts.append(emit_text("from_table", str(rel.get("from_table") or "")))
            result_parts.append(emit_text("from_column", str(rel.get("from_column") or "")))
            result_parts.append(emit_text("to_schema", str(rel.get("to_schema") or "")))
            result_parts.append(emit_text("to_table", str(rel.get("to_table") or "")))
            result_parts.append(emit_text("to_column", str(rel.get("to_column") or "")))
            if rel.get("constraint_name"):
                result_parts.append(emit_text("constraint_name", str(rel.get("constraint_name") or "")))
            result_parts.append("</fk>")
        result_parts.append("</fk_relationships>")

    return fk_relationships


__all__ = ["fetch_fk_relationships_and_append_xml"]


