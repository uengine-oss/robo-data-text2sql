from app.react.state import ReactSessionState


def _should_raise_without_any_explain(state: ReactSessionState, sql: str) -> bool:
    """
    Mirrors the relaxed runtime gate:
    - If there was at least one successful explain in the session, allow even if SQL doesn't match.
    - Otherwise, require exact match via is_sql_explained().
    """
    is_explained = state.is_sql_explained(sql)
    has_any_explain = state.has_any_explained_sql()
    return (not is_explained) and (not has_any_explain)


def test_gate_blocks_when_no_explain_success_and_sql_not_matched() -> None:
    state = ReactSessionState.new(
        user_query="q",
        dbms="postgres",
        remaining_tool_calls=10,
    )
    assert state.has_any_explained_sql() is False
    assert _should_raise_without_any_explain(state, "SELECT 1") is True


def test_gate_allows_when_any_explain_success_even_if_sql_not_matched() -> None:
    state = ReactSessionState.new(
        user_query="q",
        dbms="postgres",
        remaining_tool_calls=10,
    )
    state.add_explained_sql("SELECT 1")
    assert state.has_any_explained_sql() is True

    # Different SQL: should be allowed by relaxed policy.
    assert state.is_sql_explained("SELECT 2") is False
    assert _should_raise_without_any_explain(state, "SELECT 2") is False


def test_gate_allows_when_sql_is_exactly_explained() -> None:
    state = ReactSessionState.new(
        user_query="q",
        dbms="postgres",
        remaining_tool_calls=10,
    )
    state.add_explained_sql("SELECT 1")
    assert state.is_sql_explained(" select   1 ") is True
    assert _should_raise_without_any_explain(state, " select   1 ") is False


