"""
Runtime diagnostics helpers for Text2SQL properties on Neo4j graph nodes.

Purpose:
- Detect missing Text2SQL runtime properties quickly in request path.
- Keep checks lightweight (single aggregated Cypher query).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.config import settings


def _schemas_from_settings() -> List[str]:
    schemas = [s.strip().lower() for s in (settings.target_db_schemas or "").split(",") if s.strip()]
    if not schemas:
        schema_one = str(settings.target_db_schema or "").strip().lower()
        schemas = [schema_one] if schema_one else ["public"]
    return schemas


async def inspect_text2sql_property_gaps(
    *,
    neo4j_session,
    schemas: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Inspect missing Text2SQL properties for Table/Column nodes.

    Returns a compact diagnostics dict:
    {
      "schemas": [...],
      "table_total": ...,
      "table_valid_true_count": ...,
      "table_missing_vector": ...,
      "table_missing_embedding_text": ...,
      "table_missing_is_valid": ...,
      "table_missing_db_exists": ...,
      "column_total": ...,
      "column_missing_is_valid": ...,
      "column_missing_db_exists": ...,
      "missing_total": ...,
      "needs_repair": bool,
    }
    """
    schemas_l = [str(s).strip().lower() for s in (schemas or _schemas_from_settings()) if str(s).strip()]

    query = """
    CALL () {
      MATCH (t:Table)
      WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
      RETURN
        count(t) AS table_total,
        count(CASE WHEN t.text_to_sql_is_valid = true THEN 1 END) AS table_valid_true_count,
        count(CASE WHEN (t.text_to_sql_vector IS NULL OR size(coalesce(t.text_to_sql_vector, [])) = 0) THEN 1 END) AS table_missing_vector,
        count(CASE WHEN trim(COALESCE(t.text_to_sql_embedding_text, '')) = '' THEN 1 END) AS table_missing_embedding_text,
        count(CASE WHEN t.text_to_sql_is_valid IS NULL THEN 1 END) AS table_missing_is_valid,
        count(CASE WHEN t.text_to_sql_db_exists IS NULL THEN 1 END) AS table_missing_db_exists
    }
    CALL () {
      MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
      WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
      RETURN
        count(c) AS column_total,
        count(CASE WHEN c.text_to_sql_is_valid IS NULL THEN 1 END) AS column_missing_is_valid,
        count(CASE WHEN c.text_to_sql_db_exists IS NULL THEN 1 END) AS column_missing_db_exists
    }
    RETURN
      table_total,
      table_valid_true_count,
      table_missing_vector,
      table_missing_embedding_text,
      table_missing_is_valid,
      table_missing_db_exists,
      column_total,
      column_missing_is_valid,
      column_missing_db_exists
    """

    async def _run_once(schemas_param: Optional[List[str]]) -> Dict[str, Any]:
        res = await neo4j_session.run(query, schemas=schemas_param)
        row = await res.single()
        return row.data() if row else {}

    data = await _run_once(schemas_l if schemas_l else None)
    fallback_used = False
    schema_filter_used: Optional[List[str]] = list(schemas_l) if schemas_l else None

    # Fallback: when configured schema filter doesn't match graph schemas, re-check globally.
    if int(data.get("table_total") or 0) == 0 and schemas_l:
        data = await _run_once(None)
        fallback_used = True
        schema_filter_used = None

    out: Dict[str, Any] = {
        "schemas_requested": schemas_l,
        "schemas_used": schema_filter_used,
        "schema_filter_fallback_used": bool(fallback_used),
        "table_total": int(data.get("table_total") or 0),
        "table_valid_true_count": int(data.get("table_valid_true_count") or 0),
        "table_missing_vector": int(data.get("table_missing_vector") or 0),
        "table_missing_embedding_text": int(data.get("table_missing_embedding_text") or 0),
        "table_missing_is_valid": int(data.get("table_missing_is_valid") or 0),
        "table_missing_db_exists": int(data.get("table_missing_db_exists") or 0),
        "column_total": int(data.get("column_total") or 0),
        "column_missing_is_valid": int(data.get("column_missing_is_valid") or 0),
        "column_missing_db_exists": int(data.get("column_missing_db_exists") or 0),
    }
    missing_total = int(
        out["table_missing_vector"]
        + out["table_missing_embedding_text"]
        + out["table_missing_is_valid"]
        + out["table_missing_db_exists"]
        + out["column_missing_is_valid"]
        + out["column_missing_db_exists"]
    )
    out["missing_total"] = missing_total
    table_total = int(out.get("table_total") or 0)
    # Cold start heuristic:
    # - At least one table exists in graph
    # - Every table misses text_to_sql_vector
    # - Every table misses text_to_sql_embedding_text
    # - OR no table has text_to_sql_is_valid=true
    # This distinguishes full cold boot from partial drift.
    out["cold_start_detected"] = bool(
        table_total > 0
        and (
            (
                int(out.get("table_missing_vector") or 0) >= table_total
                and int(out.get("table_missing_embedding_text") or 0) >= table_total
            )
            or int(out.get("table_valid_true_count") or 0) == 0
        )
    )
    out["needs_repair"] = bool(
        missing_total >= int(getattr(settings, "text2sql_runtime_repair_min_missing_to_trigger", 1) or 1)
    )
    return out


__all__ = ["inspect_text2sql_property_gaps"]

