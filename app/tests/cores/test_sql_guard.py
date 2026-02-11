# python -m pytest app/tests/cores/test_sql_guard.py -v

import pytest

from app.config import settings
from app.core.sql_guard import SQLGuard, SQLValidationError


class TestSQLGuardSubqueryDepth:
    """SQLGuard 서브쿼리 깊이 검증 테스트"""

    def test_validate_allows_subquery_within_limit(self, monkeypatch):
        """허용 깊이 이하의 서브쿼리는 검증을 통과해야 한다"""
        monkeypatch.setattr(settings, "max_subquery_depth", 1)
        guard = SQLGuard()

        sql = """
        SELECT *
        FROM (
            SELECT 1 AS value
        ) subquery
        """

        cleaned_sql, is_valid = guard.validate(sql)

        assert is_valid
        assert "SELECT" in cleaned_sql  # validate는 SQL을 변형하지 않고 안전성만 검사한다

    def test_validate_raises_when_subquery_depth_exceeded(self, monkeypatch):
        """허용 깊이를 초과하면 SQLValidationError가 발생해야 한다"""
        monkeypatch.setattr(settings, "max_subquery_depth", 1)
        guard = SQLGuard()

        sql = """
        SELECT *
        FROM (
            SELECT *
            FROM (
                SELECT 1 AS value
            ) inner_sq
        ) outer_sq
        """

        with pytest.raises(SQLValidationError) as excinfo:
            guard.validate(sql)

        assert "Subquery depth exceeds limit" in str(excinfo.value)

