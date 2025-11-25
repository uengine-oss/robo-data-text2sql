"""PostgreSQL-specific query builder for execution plan benchmarking."""

import json
from typing import Any, Dict, List, Tuple

from app.react.utils.db_query_builder.base import ExecutionPlanQueryBuilder
from app.react.utils.db_query_builder.type import ExecutionPlanResult, TableIndexMetadata, TableMetadata


class PostgreSQLQueryBuilder(ExecutionPlanQueryBuilder):
    """PostgreSQL implementation of execution plan query builder."""

    async def fetch_execution_plan(
        self,
        conn: Any,
        sql: str,
        *,
        analyze: bool = True,
        verbose: bool = True,
        buffers: bool = False,
        format: str = "JSON",
    ) -> ExecutionPlanResult:
        """Execute PostgreSQL EXPLAIN and normalize metrics."""
        explain_sql = self._build_explain_sql(
            sql,
            analyze=analyze,
            verbose=verbose,
            buffers=buffers,
            format=format,
        )

        try:
            plan_payload = await conn.fetchval(explain_sql)
        except Exception as exc:
            raise RuntimeError(f"EXPLAIN 실행 실패: {exc}") from exc

        if not plan_payload:
            raise RuntimeError("PostgreSQL 실행 계획 결과가 비어 있습니다.")

        plan_json = self._normalize_plan_payload(plan_payload)
        root_plan = plan_json[0] if isinstance(plan_json, list) else plan_json
        plan_node = root_plan.get("Plan", {})

        total_cost = float(plan_node.get("Total Cost", 0.0))
        execution_time_ms = float(
            root_plan.get("Execution Time", plan_node.get("Actual Total Time", 0.0))
        )
        row_count = int(
            plan_node.get("Actual Rows")
            or plan_node.get("Plan Rows")
            or root_plan.get("Plan Rows", 0)
            or 0
        )

        return ExecutionPlanResult(
            total_cost=total_cost,
            execution_time_ms=execution_time_ms,
            row_count=row_count,
            raw_plan=root_plan,
        )

    async def collect_table_metadata(
        self,
        conn: Any,
        raw_plan: Any,
    ) -> List[TableMetadata]:
        table_pairs = self._extract_tables_from_plan(raw_plan)
        if not table_pairs:
            return []

        table_keys = [f"{schema}.{table}" for schema, table in table_pairs]

        stats_rows = await conn.fetch(_TABLE_STATS_SQL, table_keys)
        row_count_map = {
            row["table_key"]: int(row["row_count"]) for row in stats_rows
        }

        index_rows = await conn.fetch(_TABLE_INDEXES_SQL, table_keys)
        index_map: Dict[str, List[TableIndexMetadata]] = {}
        for row in index_rows:
            columns = [col for col in row["columns"] or [] if col]
            index_meta = TableIndexMetadata(
                index_name=row["index_name"],
                is_unique=row["is_unique"],
                columns=columns,
                definition=row["definition"],
            )
            index_map.setdefault(row["table_key"], []).append(index_meta)

        metadata: List[TableMetadata] = []
        for schema, table in table_pairs:
            key = f"{schema}.{table}"
            metadata.append(
                TableMetadata(
                    schema_name=schema,
                    table_name=table,
                    row_count=row_count_map.get(key, 0),
                    indexes=index_map.get(key, []),
                )
            )

        return metadata

    @staticmethod
    def _build_explain_sql(
        sql: str,
        *,
        analyze: bool,
        verbose: bool,
        buffers: bool,
        format: str,
    ) -> str:
        options = [f"FORMAT {format.upper()}"]
        if analyze:
            options.append("ANALYZE")
        if verbose:
            options.append("VERBOSE")
        if buffers:
            options.append("BUFFERS")
        options_clause = ", ".join(options)
        return f"EXPLAIN ({options_clause}) {sql}"

    @staticmethod
    def _normalize_plan_payload(payload: Any) -> Any:
        if isinstance(payload, str):
            return json.loads(payload)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload
        return payload

    @staticmethod
    def _extract_tables_from_plan(raw_plan: Any) -> List[Tuple[str, str]]:
        def _normalize_plan(plan_payload: Any) -> Any:
            if isinstance(plan_payload, str):
                try:
                    return json.loads(plan_payload)
                except json.JSONDecodeError:
                    return {}
            return plan_payload

        normalized = _normalize_plan(raw_plan)
        if isinstance(normalized, dict) and "Plan" in normalized:
            normalized = normalized["Plan"]

        discovered: Dict[str, Tuple[str, str]] = {}

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                relation = node.get("Relation Name")
                schema = node.get("Schema")
                if relation and schema:
                    key = f"{schema}.{relation}"
                    discovered[key] = (schema, relation)
                for child in node.get("Plans", []):
                    _walk(child)
            elif isinstance(node, list):
                for child in node:
                    _walk(child)

        _walk(normalized)
        return sorted(discovered.values())


_TABLE_STATS_SQL = """
SELECT
    n.nspname AS schema_name,
    c.relname AS table_name,
    COALESCE(s.n_live_tup, 0)::bigint AS row_count,
    (n.nspname || '.' || c.relname) AS table_key
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
WHERE (n.nspname || '.' || c.relname) = ANY($1::text[])
"""

_TABLE_INDEXES_SQL = """
WITH expanded AS (
    SELECT
        n.nspname AS schema_name,
        c.relname AS table_name,
        i.relname AS index_name,
        ix.indisunique AS is_unique,
        pg_get_indexdef(i.oid) AS definition,
        (n.nspname || '.' || c.relname) AS table_key,
        ord.ordinality,
        CASE
            WHEN ord.attnum > 0 THEN pg_attribute.attname
            ELSE NULL
        END AS column_name
    FROM pg_index ix
    JOIN pg_class c ON c.oid = ix.indrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_class i ON i.oid = ix.indexrelid
    LEFT JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS ord(attnum, ordinality)
        ON TRUE
    LEFT JOIN pg_attribute
        ON pg_attribute.attrelid = c.oid
        AND pg_attribute.attnum = ord.attnum
    WHERE (n.nspname || '.' || c.relname) = ANY($1::text[])
)
SELECT
    schema_name,
    table_name,
    index_name,
    is_unique,
    definition,
    table_key,
    ARRAY_REMOVE(ARRAY_AGG(column_name ORDER BY ordinality), NULL) AS columns
FROM expanded
GROUP BY schema_name, table_name, index_name, is_unique, definition, table_key
ORDER BY schema_name, table_name, index_name
"""

