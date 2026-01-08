import json
from typing import Any, Dict, List, Optional, Tuple

from app.core.sql_guard import SQLGuard
from app.react.tools.context import ToolContext
from app.react.tools.neo4j_utils import get_column_fk_relationships


def _split_table_identifier(name: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [part for part in name.split(".") if part]
    if not parts:
        return None, None
    if len(parts) == 1:
        return None, parts[0]
    return ".".join(parts[:-1]), parts[-1]


def _normalize_lower(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped.lower() if stripped else None


def _build_schema_hints(table_names: List[str]) -> Dict[str, str]:
    hints: Dict[str, str] = {}
    for raw_name in table_names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        schema_part, table_part = _parse_requested_table_identifier(name)
        normalized_table = _normalize_lower(table_part)
        if not normalized_table or normalized_table in hints or not schema_part:
            continue
        hints[normalized_table] = schema_part
    return hints


def _parse_requested_table_identifier(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse incoming table identifier for tool calls.

    Accepted inputs:
    - "TABLE"
    - "SCHEMA.TABLE"
    - "DB.SCHEMA.TABLE" (or longer): we treat the last two parts as schema/table.

    This is intentionally different from _split_table_identifier (which joins all but last),
    because tool inputs may contain a db/catalog prefix that should not be treated as schema.
    """
    parts = [part for part in str(name or "").split(".") if part]
    if not parts:
        return None, None
    if len(parts) == 1:
        return None, parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[-2], parts[-1]


def _sanitize_identifier(identifier: Optional[str]) -> Optional[str]:
    if not identifier:
        return None
    sanitized = SQLGuard.sanitize_identifier(str(identifier))
    return sanitized or None


def _quote_identifier(identifier: str) -> str:
    parts = [part for part in identifier.split(".") if part]
    if not parts:
        return '""'
    return ".".join(f'"{part}"' for part in parts)


def _quote_table_identifier(table: str, schema: Optional[str]) -> str:
    parts = []
    if schema:
        parts.extend(schema.split("."))
    parts.append(table)
    return ".".join(f'"{part}"' for part in parts if part)


def _serialize_values(values: List[Any]) -> str:
    serializable: List[Any] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            serializable.append(value)
        else:
            serializable.append(str(value))
    return json.dumps(serializable, ensure_ascii=False)


async def _fetch_table_columns(
    conn,
    table_name: Optional[str],
    schema_hint: Optional[str],
) -> Tuple[Optional[str], Dict[str, str]]:
    table_param = str(table_name or "").strip()
    if not table_param:
        return schema_hint, {}

    schema_param = str(schema_hint or "").strip()

    if schema_param:
        query = """
        SELECT table_schema, column_name
        FROM information_schema.columns
        WHERE lower(table_name) = lower($1)
          AND lower(table_schema) = lower($2)
        ORDER BY ordinal_position
        """
        rows = await conn.fetch(query, table_param, schema_param)
    else:
        query = """
        SELECT table_schema, column_name
        FROM information_schema.columns
        WHERE lower(table_name) = lower($1)
        ORDER BY ordinal_position
        """
        rows = await conn.fetch(query, table_param)

    if not rows:
        return schema_hint, {}

    column_map: Dict[str, str] = {}
    for row in rows:
        column_name = row["column_name"]
        normalized_column = _normalize_lower(column_name)
        if not column_name or not normalized_column:
            continue
        column_map[normalized_column] = column_name

    resolved_schema = schema_hint or rows[0]["table_schema"]
    return resolved_schema, column_map


async def _fetch_column_values(
    conn,
    table_name: str,
    column_name: str,
    schema: Optional[str],
    limit: int,
) -> List[Any]:
    sanitized_table = _sanitize_identifier(table_name)
    sanitized_column = _sanitize_identifier(column_name)
    sanitized_schema = _sanitize_identifier(schema) if schema else None

    if not sanitized_table or not sanitized_column or limit <= 0:
        return []

    table_identifier = _quote_table_identifier(sanitized_table, sanitized_schema)
    column_identifier = _quote_identifier(sanitized_column)

    query = (
        f"SELECT DISTINCT {column_identifier} AS value "
        f"FROM {table_identifier} "
        f"WHERE {column_identifier} IS NOT NULL "
        f"ORDER BY {column_identifier} "
        f"LIMIT {limit}"
    )
    rows = await conn.fetch(query)
    return [row["value"] for row in rows if "value" in row]


async def execute(
    context: ToolContext,
    table_names: List[str],
) -> str:
    """Neo4j 에 저장된 테이블 스키마 정보를 조회한다."""
    get_table_schema_table_name_limit = max(int(context.scaled(context.get_table_schema_table_name_limit)), 1)
    table_names = table_names[:get_table_schema_table_name_limit]

    column_relation_limit = context.scaled(context.column_relation_limit)
    value_limit = min(10, context.scaled(context.value_limit))
    requested: List[Dict[str, Optional[str]]] = []
    for raw in table_names:
        raw_text = str(raw or "").strip()
        if not raw_text:
            continue
        schema_part, table_part = _parse_requested_table_identifier(raw_text)
        normalized_table = _normalize_lower(table_part)
        if not normalized_table:
            continue
        requested.append(
            {
                "schema": _normalize_lower(schema_part) if schema_part else None,
                "name": normalized_table,
            }
        )

    schema_hints = _build_schema_hints(table_names)
    db_metadata_cache: Dict[
        Tuple[str, str],
        Tuple[Optional[str], Dict[str, str]],
    ] = {}

    query = """
    UNWIND $requested AS req
    MATCH (t:Table)
    WHERE (
      (t.name IS NOT NULL AND toLower(t.name) = req.name)
      OR (t.original_name IS NOT NULL AND toLower(t.original_name) = req.name)
    )
      AND (req.schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = req.schema))
    WITH DISTINCT t
    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
    WITH t, c
    ORDER BY c.name
    RETURN COALESCE(t.original_name, t.name) AS table_name,
           t.schema AS table_schema,
           t.description AS table_description,
           collect({
               name: c.name,
               dtype: c.dtype,
               nullable: c.nullable,
               description: c.description,
               is_primary_key: c.is_primary_key
           }) AS columns
    ORDER BY table_schema, table_name
    """

    result = await context.neo4j_session.run(
        query,
        requested=requested,
    )
    records = await result.data()

    result_parts: List[str] = ["<tool_result>"]

    if not records:
        if requested:
            result_parts.append("<warning>no_match</warning>")
            result_parts.append(
                "<hint>No matching tables found in the schema graph. "
                "If multiple schemas may contain the same table name, specify schema.table. "
                "Do NOT retry the exact same get_table_schema call; choose different tables or search again.</hint>"
            )
        result_parts.append("</tool_result>")
        return "\n".join(result_parts)

    for record in records:
        table_name = record["table_name"]
        table_schema = record.get("table_schema", "")
        table_description = record.get("table_description", "")
        columns = record["columns"]

        normalized_table_name = _normalize_lower(table_name)
        schema_hint = table_schema or schema_hints.get(normalized_table_name or "", "")
        schema_hint = schema_hint or None
        metadata_key: Optional[Tuple[str, str]] = None
        resolved_schema: Optional[str] = schema_hint
        db_columns: Dict[str, str] = {}

        if normalized_table_name:
            schema_key = _normalize_lower(schema_hint) or ""
            metadata_key = (schema_key, normalized_table_name)
            if metadata_key not in db_metadata_cache:
                resolved_schema, db_columns = await _fetch_table_columns(
                    context.db_conn,
                    table_name,
                    schema_hint,
                )
                db_metadata_cache[metadata_key] = (resolved_schema, db_columns)
            else:
                resolved_schema, db_columns = db_metadata_cache[metadata_key]

        result_parts.append("<table>")
        result_parts.append(f"<schema>{table_schema}</schema>")
        result_parts.append(f"<name>{table_name}</name>")
        if table_description:
            result_parts.append(f"<description>{table_description}</description>")

        result_parts.append("<columns>")

        def is_primary_key_column(column: dict) -> bool:
            raw_value = column.get("is_primary_key")
            if raw_value is None:
                return False
            if isinstance(raw_value, bool):
                return raw_value
            normalized = str(raw_value).strip().lower()
            return normalized not in {"", "none", "null", "false", "0"}

        sorted_columns = sorted(
            columns,
            key=lambda column: 0 if is_primary_key_column(column) else 1,
        )

        for col in sorted_columns:
            if not col["name"]:
                continue

            col_name = col["name"]
            col_dtype = col.get("dtype", "")
            col_nullable = str(col.get("nullable", True)).lower()
            col_description = col.get("description", "")
            is_primary_key = str(col.get("is_primary_key", False)).lower()

            if table_schema:
                fqn = f"{table_schema}.{table_name}.{col_name}"
            else:
                fqn = f"{table_name}.{col_name}"

            result_parts.append("<column>")
            result_parts.append(f"<fqn>{fqn}</fqn>")
            result_parts.append(f"<name>{col_name}</name>")
            if col_dtype:
                result_parts.append(f"<dtype>{col_dtype}</dtype>")
            result_parts.append(f"<nullable>{col_nullable}</nullable>")
            if is_primary_key and is_primary_key not in ["none", "null", "false", "0"]:
                result_parts.append(f"<is_primary_key>{is_primary_key}</is_primary_key>")
            if col_description:
                result_parts.append(f"<description>{col_description}</description>")

            fk_relationships = await get_column_fk_relationships(
                context.neo4j_session,
                table_name,
                col_name,
                column_relation_limit,
                table_schema or None
            )
            if fk_relationships:
                fk_relationships = sorted(
                    fk_relationships,
                    key=lambda fk: (
                        fk.get("referenced_table_schema") or "",
                        fk.get("referenced_table") or "",
                        fk.get("referenced_column") or "",
                    ),
                )
                result_parts.append("<foreign_keys>")
                for fk in fk_relationships:
                    result_parts.append("<foreign_key>")
                    if fk.get("referenced_table_schema"):
                        result_parts.append(
                            f"<referenced_table_schema>{fk['referenced_table_schema']}</referenced_table_schema>"
                        )
                    result_parts.append(f"<referenced_table>{fk['referenced_table']}</referenced_table>")
                    if fk.get("referenced_table_description"):
                        result_parts.append(
                            f"<referenced_table_description>{fk['referenced_table_description']}</referenced_table_description>"
                        )
                    result_parts.append(f"<referenced_column>{fk['referenced_column']}</referenced_column>")
                    if fk.get("referenced_column_description"):
                        result_parts.append(
                            f"<referenced_column_description>{fk['referenced_column_description']}</referenced_column_description>"
                        )
                    if fk.get("constraint_name"):
                        result_parts.append(f"<constraint_name>{fk['constraint_name']}</constraint_name>")
                    result_parts.append("</foreign_key>")
                result_parts.append("</foreign_keys>")

            normalized_column_name = _normalize_lower(col_name)
            column_values: List[Any] = []
            if (
                value_limit > 0
                and normalized_column_name
                and normalized_column_name in db_columns
            ):
                actual_column_name = db_columns[normalized_column_name]
                try:
                    column_values = await _fetch_column_values(
                        context.db_conn,
                        table_name,
                        actual_column_name,
                        resolved_schema,
                        value_limit,
                    )
                except Exception:
                    column_values = []

            if column_values:
                result_parts.append(f"<values>{_serialize_values(column_values)}</values>")
            result_parts.append("</column>")
        result_parts.append("</columns>")
        result_parts.append("</table>")

    result_parts.append("</tool_result>")
    return "\n".join(result_parts)

