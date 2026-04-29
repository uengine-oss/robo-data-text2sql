from app.core.llm_factory import _openrouter_extra_body


def test_openrouter_extra_body_can_pin_provider(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_PROVIDER_ONLY", "deepinfra")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "false")

    body = _openrouter_extra_body(base_url="https://openrouter.ai/api/v1")

    assert body == {"provider": {"only": ["deepinfra"], "allow_fallbacks": False}}


def test_openrouter_extra_body_can_request_reasoning(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_PROVIDER_ONLY", "deepinfra")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "false")

    body = _openrouter_extra_body(
        base_url="https://openrouter.ai/api/v1",
        thinking_level="high",
    )

    assert body == {
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        "reasoning": {"effort": "high"},
    }


def test_openrouter_extra_body_disables_reasoning_for_gemma(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_PROVIDER_ONLY", "deepinfra")
    monkeypatch.setenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "false")
    monkeypatch.setenv("OPENROUTER_REASONING_DISABLED_MODELS", "google/gemma-4-31b-it,gemma-4-31b-it")

    body = _openrouter_extra_body(
        base_url="https://openrouter.ai/api/v1",
        model="google/gemma-4-31b-it",
        thinking_level="high",
    )

    assert body == {"provider": {"only": ["deepinfra"], "allow_fallbacks": False}}


def test_openrouter_extra_body_can_globally_disable_reasoning(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_PROVIDER_ONLY", raising=False)
    monkeypatch.delenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", raising=False)
    monkeypatch.setenv("OPENROUTER_REASONING_ENABLED", "false")

    body = _openrouter_extra_body(
        base_url="https://openrouter.ai/api/v1",
        model="some/reasoning-model",
        thinking_level="high",
    )

    assert not (body or {}).get("reasoning")


def test_openrouter_extra_body_ignores_provider_options_for_non_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_PROVIDER_ONLY", "deepinfra")

    body = _openrouter_extra_body(base_url="https://api.openai.com/v1")

    assert body is None
