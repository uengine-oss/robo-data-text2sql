from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .models import ColumnCandidate, TableCandidate
from .xml_utils import emit_text


def append_schema_candidates_xml(
    *,
    result_parts: List[str],
    table_candidates: Sequence[TableCandidate],
    selected_tables: Sequence[TableCandidate],
    per_table_columns: Dict[str, List[ColumnCandidate]],
    per_table_mode: str,
    per_table_k: int,
    rerank_top_k: int,
    table_description_overrides: Optional[Dict[str, str]] = None,
) -> None:
    """
    Append <schema_candidates> block (tables + per_table_columns) to tool result XML.
    """
    per_table_k = max(int(per_table_k), 1)
    rerank_top_k = max(int(rerank_top_k), 1)

    result_parts.append("<schema_candidates>")
    result_parts.append("<tables>")
    for t in list(table_candidates)[:rerank_top_k]:
        result_parts.append("<table>")
        result_parts.append(emit_text("schema", t.schema))
        result_parts.append(emit_text("name", t.name))
        tfqn_l = (t.fqn or "").strip().lower()
        override_desc = ""
        if table_description_overrides and tfqn_l:
            override_desc = str(table_description_overrides.get(tfqn_l, "") or "").strip()
        # Prefer text_to_sql_embedding_text (override) since Table.description/analyzed_description can be missing/weak.
        fallback_desc = (t.analyzed_description or "").strip() or (t.description or "").strip()
        result_parts.append(emit_text("description", override_desc or fallback_desc))
        result_parts.append(emit_text("score", f"{t.score:.4f}"))
        result_parts.append("</table>")
    result_parts.append("</tables>")

    result_parts.append("<per_table_columns>")
    for t in list(selected_tables)[:rerank_top_k]:
        schema_l = (t.schema or "").lower()
        name_l = (t.name or "").lower()
        tfqn_l = f"{schema_l}.{name_l}" if schema_l else name_l
        cols = per_table_columns.get(tfqn_l, [])
        result_parts.append("<table>")
        result_parts.append(emit_text("schema", t.schema))
        result_parts.append(emit_text("name", t.name))
        result_parts.append(emit_text("mode", str(per_table_mode or "")))
        result_parts.append("<columns>")
        # Respect per_table_k (avoid forced minimum output that can cause context bloat).
        for c in cols[:per_table_k]:
            result_parts.append("<column>")
            result_parts.append(emit_text("name", c.name))
            result_parts.append(emit_text("dtype", c.dtype))
            if c.description:
                result_parts.append(emit_text("description", c.description))
            result_parts.append(emit_text("score", f"{float(c.score or 0.0):.4f}"))
            result_parts.append("</column>")
        result_parts.append("</columns>")
        result_parts.append("</table>")
    result_parts.append("</per_table_columns>")
    result_parts.append("</schema_candidates>")


__all__ = ["append_schema_candidates_xml"]


