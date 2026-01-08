from app.react.agent import (
    SQLCompleteness,
    ToolCall,
    ReactStep,
    _maybe_rewrite_tool_call_for_policy,
    _tool_result_has_error,
)
from app.react.state import ReactSessionState


def _make_step(tool_name: str, sql: str) -> ReactStep:
    return ReactStep(
        iteration=1,
        reasoning="",
        metadata_xml="<collected_metadata/>",
        partial_sql="",
        sql_completeness=SQLCompleteness(
            is_complete=False,
            missing_info="",
            confidence_level="low",
        ),
        tool_call=ToolCall(
            name=tool_name,
            raw_parameters_xml=f"<parameters>{sql}</parameters>",
            parsed_parameters={"sql": sql},
        ),
    )


def test_rewrite_execute_preview_to_submit_on_last_call() -> None:
    state = ReactSessionState.new(user_query="q", dbms="postgres", remaining_tool_calls=1)
    step = _make_step("execute_sql_preview", "SELECT 1")

    info = _maybe_rewrite_tool_call_for_policy(step=step, state=state)

    assert info is not None
    assert info["from"] == "execute_sql_preview"
    assert info["to"] == "submit_sql"
    assert step.tool_call.name == "submit_sql"
    assert step.tool_call.parsed_parameters == {"sql": "SELECT 1"}
    assert "<parameters>" in step.tool_call.raw_parameters_xml


def test_rewrite_execute_preview_to_explain_when_no_explain_yet() -> None:
    state = ReactSessionState.new(user_query="q", dbms="postgres", remaining_tool_calls=10)
    step = _make_step("execute_sql_preview", "SELECT 1")

    info = _maybe_rewrite_tool_call_for_policy(step=step, state=state)

    assert info is not None
    assert info["to"] == "explain"
    assert step.tool_call.name == "explain"
    assert step.tool_call.parsed_parameters == {"sql": "SELECT 1"}


def test_rewrite_submit_to_explain_when_no_explain_yet() -> None:
    state = ReactSessionState.new(user_query="q", dbms="postgres", remaining_tool_calls=10)
    step = _make_step("submit_sql", "SELECT 1")

    info = _maybe_rewrite_tool_call_for_policy(step=step, state=state)

    assert info is not None
    assert info["from"] == "submit_sql"
    assert info["to"] == "explain"
    assert step.tool_call.name == "explain"


def test_no_rewrite_when_any_explain_succeeded_even_if_sql_not_matched() -> None:
    state = ReactSessionState.new(user_query="q", dbms="postgres", remaining_tool_calls=10)
    state.add_explained_sql("SELECT 1")  # simulate successful explain happened earlier
    step = _make_step("execute_sql_preview", "SELECT 2")

    info = _maybe_rewrite_tool_call_for_policy(step=step, state=state)

    assert info is None
    assert step.tool_call.name == "execute_sql_preview"


def test_tool_result_has_error_detection() -> None:
    assert _tool_result_has_error("<tool_result><error>bad</error></tool_result>") is True
    assert _tool_result_has_error("<tool_result><preview /></tool_result>") is False
    assert _tool_result_has_error("not xml") is True


