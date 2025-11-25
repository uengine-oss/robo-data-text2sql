# python -m pytest app/tests/utils/test_db_query_builder.py -v

"""Tests for database query builder."""

import asyncio

import pytest

from app.react.utils.db_query_builder import (
    ExecutionPlanQueryBuilder,
    ExecutionPlanResult,
    get_query_builder,
)
from app.react.utils.db_query_builder.factory import clear_builder_cache


class TestQueryBuilderFactory:
    """Test query builder factory."""
    
    def test_get_postgresql_builder(self):
        """Test getting PostgreSQL builder."""
        builder = get_query_builder("postgresql")
        assert isinstance(builder, ExecutionPlanQueryBuilder)
    
    def test_get_postgres_alias(self):
        """Test postgres alias works."""
        builder = get_query_builder("postgres")
        assert isinstance(builder, ExecutionPlanQueryBuilder)
    
    def test_case_insensitive(self):
        """Test database type is case-insensitive."""
        builder1 = get_query_builder("PostgreSQL")
        builder2 = get_query_builder("POSTGRESQL")
        assert isinstance(builder1, ExecutionPlanQueryBuilder)
        assert isinstance(builder2, ExecutionPlanQueryBuilder)
    
    def test_unsupported_database(self):
        """Test unsupported database raises NotImplementedError."""
        with pytest.raises(NotImplementedError) as exc_info:
            get_query_builder("mysql")
        assert "mysql" in str(exc_info.value).lower()
        assert "not supported" in str(exc_info.value).lower()
    
    def test_builder_caching(self):
        """Test builder instances are cached."""
        clear_builder_cache()
        builder1 = get_query_builder("postgresql")
        builder2 = get_query_builder("postgresql")
        assert builder1 is builder2  # Same instance


class TestPostgreSQLQueryBuilder:
    """Test PostgreSQL query builder."""
    
    def test_fetch_execution_plan_parses_metrics(self):
        """Test that fetch_execution_plan normalizes metrics."""

        class DummyConn:
            def __init__(self, payload):
                self.payload = payload
                self.last_sql = None

            async def fetchval(self, sql):
                self.last_sql = sql
                return self.payload

        builder = get_query_builder("postgresql")
        sample_plan = [
            {
                "Plan": {
                    "Total Cost": 123.45,
                    "Actual Total Time": 4.56,
                    "Actual Rows": 789,
                    "Plan Rows": 1000,
                },
                "Execution Time": 4.78,
            }
        ]

        conn = DummyConn(sample_plan)

        async def run_test():
            result: ExecutionPlanResult = await builder.fetch_execution_plan(
                conn,
                "SELECT 1",
                analyze=True,
                verbose=True,
                buffers=False,
            )

            assert "EXPLAIN" in conn.last_sql
            assert result.total_cost == pytest.approx(123.45)
            # Should prefer Execution Time over Actual Total Time
            assert result.execution_time_ms == pytest.approx(4.78)
            assert result.row_count == 789
            assert result.raw_plan == sample_plan[0]

        asyncio.run(run_test())

