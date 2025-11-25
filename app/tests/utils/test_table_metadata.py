# python -m pytest app/tests/utils/test_table_metadata.py -v

"""Tests for PostgreSQL-specific table metadata utilities."""

import pytest

from app.react.utils.db_query_builder.postgresql import PostgreSQLQueryBuilder
from app.react.utils.db_query_builder import TableMetadata


def test_extract_tables_from_plan_collects_unique_pairs():
    """Ensure table extraction walks nested plan nodes."""
    raw_plan = {
        "Plan": {
            "Node Type": "Nested Loop",
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Relation Name": "orders",
                    "Schema": "public",
                },
                {
                    "Node Type": "Index Scan",
                    "Relation Name": "customers",
                    "Schema": "sales",
                    "Plans": [
                        {
                            "Node Type": "Seq Scan",
                            "Relation Name": "orders",
                            "Schema": "public",
                        }
                    ],
                },
            ],
        }
    }

    tables = PostgreSQLQueryBuilder._extract_tables_from_plan(raw_plan)

    assert tables == [("public", "orders"), ("sales", "customers")]


@pytest.mark.asyncio
async def test_collect_table_metadata_merges_stats_and_indexes():
    """collect_table_metadata should merge stats and index information."""

    class DummyConn:
        def __init__(self):
            self.calls = []

        async def fetch(self, sql, params):
            self.calls.append(sql)
            if "pg_stat_user_tables" in sql:
                return [
                    {
                        "schema_name": "public",
                        "table_name": "orders",
                        "row_count": 42,
                        "table_key": "public.orders",
                    }
                ]
            return [
                {
                    "schema_name": "public",
                    "table_name": "orders",
                    "index_name": "orders_pkey",
                    "is_unique": True,
                    "definition": "CREATE UNIQUE INDEX orders_pkey ON public.orders(id)",
                    "table_key": "public.orders",
                    "columns": ["id"],
                }
            ]

    conn = DummyConn()
    raw_plan = {"Plan": {"Relation Name": "orders", "Schema": "public"}}
    builder = PostgreSQLQueryBuilder()

    metadata = await builder.collect_table_metadata(conn, raw_plan)

    assert len(metadata) == 1
    table_meta: TableMetadata = metadata[0]
    assert table_meta.schema_name == "public"
    assert table_meta.table_name == "orders"
    assert table_meta.row_count == 42
    assert len(table_meta.indexes) == 1
    assert table_meta.indexes[0].index_name == "orders_pkey"
    assert table_meta.indexes[0].is_unique is True
    assert table_meta.indexes[0].columns == ["id"]

