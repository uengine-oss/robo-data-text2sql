from app.core.llm_factory import _google_thinking_kwargs


def test_gemma_allows_only_high_thinking_level() -> None:
    assert _google_thinking_kwargs(
        model="gemma-4-31b-it",
        thinking_level="high",
        include_thoughts=False,
    ) == {"thinking_level": "high", "include_thoughts": False}

    assert _google_thinking_kwargs(
        model="gemma-4-31b-it",
        thinking_level="low",
        include_thoughts=False,
    ) == {}


def test_non_gemma_keeps_requested_thinking_level() -> None:
    assert _google_thinking_kwargs(
        model="gemini-3-flash-preview",
        thinking_level="low",
        include_thoughts=True,
    ) == {"thinking_level": "low", "include_thoughts": True}
