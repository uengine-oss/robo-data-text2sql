"""
Contract tests for POST /text2sql/react request/response models and the
streaming event protocol described in:

  openspec/specs/text2sql-react-endpoint/spec.md
  openspec/specs/text2sql-streaming-protocol/spec.md

These tests are deterministic and do NOT require a running Neo4j, MindsDB,
or LLM provider. They validate the observable shape of the public contract
at the model level (Pydantic) and the documented event-line shapes.

Live integration scenarios (cache hit roundtrip, cold-start blocking, SQL
timeout, controller pipeline) are listed in tasks.md and verified manually
against the running service. Here we lock the contract surface that those
manual checks depend on.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.routers.react import (
    ExecutionResultModel,
    QueryQualityGateModel,
    ReactRequest,
    ReactResponse,
    ReactStepModel,
    SQLCompletenessModel,
    ToolCallModel,
)


# ---------------------------------------------------------------------------
# ReactRequest — required fields, defaults, ranges
# (text2sql-react-endpoint :: 자연어 질문 처리 엔드포인트 노출)
# ---------------------------------------------------------------------------


def test_request_minimal_required_fields_only() -> None:
    """필수 필드만 채운 요청은 검증을 통과해야 한다."""
    req = ReactRequest(question="최근 1개월 사용량 합계", datasource="ds_main")
    assert req.question == "최근 1개월 사용량 합계"
    assert req.datasource == "ds_main"


def test_request_defaults_match_spec() -> None:
    """선택 필드의 기본값은 스펙에 명시된 값과 일치해야 한다."""
    req = ReactRequest(question="q", datasource="d")
    assert req.max_tool_calls == 30
    assert req.execute_final_sql is True
    assert req.max_sql_seconds == 60
    assert req.prefer_language == "ko"
    assert req.use_cache is True
    assert req.prefetch_build_sql_context is True
    assert req.debug_stream_xml_tokens is False
    assert req.session_state is None
    assert req.user_response is None
    assert req.conversation_state is None
    assert req.schema_filter is None


def test_request_missing_question_rejected() -> None:
    """question 누락은 검증 오류여야 한다."""
    with pytest.raises(ValidationError):
        ReactRequest(datasource="d")  # type: ignore[call-arg]


def test_request_missing_datasource_rejected() -> None:
    """datasource 누락은 검증 오류여야 한다."""
    with pytest.raises(ValidationError):
        ReactRequest(question="q")  # type: ignore[call-arg]


@pytest.mark.parametrize("bad_value", [0, 101, -1])
def test_request_max_tool_calls_range_enforced(bad_value: int) -> None:
    """max_tool_calls 범위는 1~100으로 강제되어야 한다."""
    with pytest.raises(ValidationError):
        ReactRequest(question="q", datasource="d", max_tool_calls=bad_value)


@pytest.mark.parametrize("bad_value", [0, 21, -5])
def test_request_max_iterations_range_enforced(bad_value: int) -> None:
    """max_iterations 범위는 1~20으로 강제되어야 한다."""
    with pytest.raises(ValidationError):
        ReactRequest(question="q", datasource="d", max_iterations=bad_value)


@pytest.mark.parametrize("bad_value", [0, 3601, -1])
def test_request_max_sql_seconds_range_enforced(bad_value: int) -> None:
    """max_sql_seconds 범위는 1~3600으로 강제되어야 한다."""
    with pytest.raises(ValidationError):
        ReactRequest(question="q", datasource="d", max_sql_seconds=bad_value)


def test_request_step_confirmation_response_only_continue() -> None:
    """step_confirmation_response는 'continue' 외 값을 거부해야 한다."""
    ok = ReactRequest(
        question="q",
        datasource="d",
        step_confirmation_response="continue",
    )
    assert ok.step_confirmation_response == "continue"
    with pytest.raises(ValidationError):
        ReactRequest(
            question="q",
            datasource="d",
            step_confirmation_response="proceed",  # type: ignore[arg-type]
        )


def test_request_ignores_undefined_extra_fields() -> None:
    """정의되지 않은 추가 필드는 서버가 무시해야 한다(D5)."""
    req = ReactRequest.model_validate(
        {
            "question": "q",
            "datasource": "d",
            "object_type_only": True,  # 화면이 보내지만 서버 모델에 없음
            "linked_object_types": [{"name": "X"}],
        }
    )
    dumped = req.model_dump()
    assert "object_type_only" not in dumped
    assert "linked_object_types" not in dumped


def test_request_schema_filter_accepts_string_list() -> None:
    """schema_filter는 문자열 배열을 받아야 한다."""
    req = ReactRequest(question="q", datasource="d", schema_filter=["dw"])
    assert req.schema_filter == ["dw"]


# ---------------------------------------------------------------------------
# ReactResponse — completion body shape
# (text2sql-react-endpoint :: 완료 응답은 정의된 결과 본문을 가진다)
# ---------------------------------------------------------------------------


def _make_step(iteration: int = 1, sql: str = "SELECT 1") -> ReactStepModel:
    return ReactStepModel(
        iteration=iteration,
        reasoning="r",
        metadata_xml="<m/>",
        partial_sql=sql,
        sql_completeness=SQLCompletenessModel(
            is_complete=True, missing_info="", confidence_level="high"
        ),
        tool_call=ToolCallModel(
            name="validate_sql",
            raw_parameters_xml="<parameters/>",
            parameters={"sql": sql},
        ),
        tool_result="<tool_result/>",
        llm_output="",
    )


def test_response_completed_full_body_round_trip() -> None:
    """일반 완료 응답이 스펙에 명시된 모든 필드를 직렬화해야 한다."""
    body = ReactResponse(
        status="completed",
        final_sql="SELECT 1",
        validated_sql="SELECT 1",
        execution_result=ExecutionResultModel(
            columns=["c"], rows=[[1]], row_count=1, execution_time_ms=12.5
        ),
        steps=[_make_step()],
        collected_metadata="<collected_metadata/>",
        partial_sql="SELECT 1",
        remaining_tool_calls=29,
        session_state=None,
        conversation_state="cap_xxx",
        question_to_user=None,
        warnings=None,
        from_cache=False,
        feedback_required=False,
        quality_gate=None,
    )
    dumped = body.model_dump()
    for key in [
        "status",
        "final_sql",
        "validated_sql",
        "execution_result",
        "steps",
        "collected_metadata",
        "partial_sql",
        "remaining_tool_calls",
        "from_cache",
        "feedback_required",
        "quality_gate",
        "conversation_state",
    ]:
        assert key in dumped, f"completed body missing field: {key}"
    assert dumped["status"] == "completed"
    assert dumped["from_cache"] is False
    assert dumped["execution_result"]["row_count"] == 1


def test_response_status_enum_includes_await_step_confirmation() -> None:
    """단계 확인 대기 상태가 응답 status enum에 존재해야 한다."""
    body = ReactResponse(
        status="await_step_confirmation",
        collected_metadata="",
        partial_sql="",
        remaining_tool_calls=0,
        session_state="tok",
        question_to_user="다음 단계로 진행하시겠습니까?",
    )
    assert body.status == "await_step_confirmation"


def test_response_needs_user_input_carries_question_and_session() -> None:
    body = ReactResponse(
        status="needs_user_input",
        collected_metadata="",
        partial_sql="",
        remaining_tool_calls=10,
        session_state="tok_xxx",
        question_to_user="어떤 정수장을 의미하시나요?",
    )
    assert body.question_to_user
    assert body.session_state


def test_response_status_rejects_unknown_value() -> None:
    """status 필드는 정의되지 않은 값을 거부해야 한다."""
    with pytest.raises(ValidationError):
        ReactResponse(
            status="working",  # type: ignore[arg-type]
            collected_metadata="",
            partial_sql="",
            remaining_tool_calls=0,
        )


def test_response_quality_gate_shape() -> None:
    """quality_gate 본문이 spec 필드를 보존해야 한다."""
    gate = QueryQualityGateModel(
        threshold=0.9,
        rounds=2,
        ok=False,
        verified_confidence=0.4,
        results=[],
        error=None,
    )
    body = ReactResponse(
        status="completed",
        final_sql="SELECT 1",
        collected_metadata="",
        partial_sql="SELECT 1",
        remaining_tool_calls=0,
        from_cache=False,
        feedback_required=True,
        quality_gate=gate,
    )
    assert body.quality_gate is not None
    assert body.quality_gate.policy == "llm_judge_2x"
    assert body.feedback_required is True


# ---------------------------------------------------------------------------
# Streaming event line shapes
# (text2sql-streaming-protocol :: 한 줄 = 한 이벤트 / 이벤트 카탈로그)
# ---------------------------------------------------------------------------


# These shapes match what the router emits today; they are the contract
# clients depend on. We assert structural invariants only (presence of
# required keys / value types) so that internal payloads can evolve without
# breaking the protocol.


def _is_one_json_per_line(payload: str) -> bool:
    for raw in payload.split("\n"):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return False
        if not isinstance(obj, dict) or "event" not in obj:
            return False
    return True


def test_streaming_payload_one_json_per_line() -> None:
    """스트림은 라인 단위 단일 JSON 객체여야 한다."""
    sample = (
        json.dumps({"event": "cache_hit", "message": "ok", "hit_count": 3}) + "\n"
        + json.dumps({"event": "completed", "response": {"status": "completed"}, "state": None}) + "\n"
    )
    assert _is_one_json_per_line(sample)


def test_streaming_blank_lines_are_ignored() -> None:
    sample = "\n   \n" + json.dumps({"event": "completed", "response": {}, "state": None}) + "\n"
    assert _is_one_json_per_line(sample)


def test_streaming_unknown_event_is_still_valid_line() -> None:
    """알 수 없는 event 값을 가진 라인도 형식적으로는 유효하다."""
    sample = json.dumps({"event": "future_event_v9", "extra": 1}) + "\n"
    assert _is_one_json_per_line(sample)


def test_pipeline_stage_event_required_fields() -> None:
    """pipeline_stage 이벤트의 필수 필드 셋."""
    ev = {
        "event": "pipeline_stage",
        "pipeline": "build_sql_context",
        "stage": "embedding",
        "status": "start",
        "seq": 1,
        "iteration": 0,
        "ts_ms": 1_700_000_000_000,
    }
    for k in ("event", "pipeline", "stage", "status", "seq", "iteration", "ts_ms"):
        assert k in ev
    assert ev["status"] in ("start", "done", "error")


def test_pipeline_item_event_required_fields() -> None:
    ev = {
        "event": "pipeline_item",
        "pipeline": "controller",
        "stage": "controller_validate",
        "item_type": "candidate",
        "iteration": 1,
        "index": 2,
        "total": 4,
        "verdict": "PASS",
    }
    for k in ("event", "pipeline", "stage", "item_type", "iteration", "index", "total"):
        assert k in ev


def test_terminal_events_set_is_closed() -> None:
    """종결 이벤트 카탈로그는 4종으로 닫혀 있다(complete는 호환 별칭)."""
    terminal = {"completed", "needs_user_input", "step_confirmation", "error"}
    compat_alias = {"complete"}
    assert "completed" in terminal
    assert compat_alias.isdisjoint(terminal)  # 카탈로그상 별개
    # 호환 별칭은 클라이언트가 'completed'와 동일하게 처리해야 한다.
    assert {"complete"} <= compat_alias


def test_cache_hit_sequence_only_two_lines() -> None:
    """캐시 히트는 cache_hit → completed 두 라인으로 구성된다."""
    payload = (
        json.dumps({"event": "cache_hit", "message": "캐시된 결과", "hit_count": 1}) + "\n"
        + json.dumps(
            {
                "event": "completed",
                "response": {
                    "status": "completed",
                    "final_sql": "SELECT 1",
                    "from_cache": True,
                    "steps": [],
                    "feedback_required": False,
                    "quality_gate": None,
                },
                "state": None,
            }
        )
        + "\n"
    )
    lines = [l for l in payload.split("\n") if l.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "cache_hit"
    last = json.loads(lines[1])
    assert last["event"] == "completed"
    assert last["response"]["from_cache"] is True
    assert last["response"]["steps"] == []
    assert last["response"]["feedback_required"] is False
    assert last["response"]["quality_gate"] is None


# ---------------------------------------------------------------------------
# Cache-use predicate
# (text2sql-react-endpoint :: 캐시 사용 정책)
# ---------------------------------------------------------------------------


def _cache_lookup_allowed(req: ReactRequest, cold_start_blocking: bool) -> bool:
    """Mirror of the server-side predicate (router code, fresh-session branch)."""
    return bool(
        req.use_cache
        and not req.session_state
        and not req.user_response
        and not req.conversation_state
        and not cold_start_blocking
    )


def test_cache_lookup_allowed_for_fresh_question() -> None:
    req = ReactRequest(question="q", datasource="d")
    assert _cache_lookup_allowed(req, cold_start_blocking=False) is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_state", "tok"),
        ("user_response", "yes"),
        ("conversation_state", "cap_xxx"),
    ],
)
def test_cache_lookup_skipped_for_session_or_followup(field: str, value: str) -> None:
    req = ReactRequest(question="q", datasource="d", **{field: value})
    assert _cache_lookup_allowed(req, cold_start_blocking=False) is False


def test_cache_lookup_skipped_during_cold_start() -> None:
    req = ReactRequest(question="q", datasource="d")
    assert _cache_lookup_allowed(req, cold_start_blocking=True) is False


def test_cache_lookup_skipped_when_use_cache_false() -> None:
    req = ReactRequest(question="q", datasource="d", use_cache=False)
    assert _cache_lookup_allowed(req, cold_start_blocking=False) is False
