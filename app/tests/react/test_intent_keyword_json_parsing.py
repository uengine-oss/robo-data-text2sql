from app.react.generators.intent_extract_generator import _try_parse_intent_json
from app.react.generators.keyword_extract_generator import _try_parse_keywords_json


def test_try_parse_intent_json_basic() -> None:
    assert _try_parse_intent_json('{"intent":"최근 1개월 매출 합계를 보여줘"}') == "최근 1개월 매출 합계를 보여줘"


def test_try_parse_intent_json_strips_code_fences() -> None:
    text = """```json
{"intent": "Show total sales last month"}
```"""
    assert _try_parse_intent_json(text) == "Show total sales last month"


def test_try_parse_intent_json_extracts_object_from_extra_text() -> None:
    text = 'OK\\n{"intent":"한줄 의도"}\\nThanks'
    assert _try_parse_intent_json(text) == "한줄 의도"


def test_try_parse_intent_json_one_line_enforced() -> None:
    # Even if the model accidentally includes a newline in the value, we keep only the first line.
    assert _try_parse_intent_json('{"intent":"line1\\nline2"}') == "line1"


def test_try_parse_keywords_json_basic() -> None:
    assert _try_parse_keywords_json('{"keywords":["매출","최근 1개월","지점"]}') == [
        "매출",
        "최근 1개월",
        "지점",
    ]


def test_try_parse_keywords_json_strips_code_fences() -> None:
    text = """```json
{"keywords": ["sales", "last month"]}
```"""
    assert _try_parse_keywords_json(text) == ["sales", "last month"]


def test_try_parse_keywords_json_extracts_object_from_extra_text() -> None:
    text = 'note:\\n{"keywords":["a","b"]}\\nend'
    assert _try_parse_keywords_json(text) == ["a", "b"]


def test_try_parse_keywords_json_rejects_wrong_shape() -> None:
    assert _try_parse_keywords_json('["a","b"]') == []
    assert _try_parse_keywords_json('{"intent":"x"}') == []

