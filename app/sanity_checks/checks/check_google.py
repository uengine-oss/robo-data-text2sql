from __future__ import annotations

import asyncio
import traceback
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import settings
from app.react.llm_factory import create_react_llm
from app.sanity_checks.result import SanityCheckResult


def _get_react_provider() -> str:
    provider = (getattr(settings, "react_llm_provider", "") or "").strip().lower()
    if provider in {"openai", "google"}:
        return provider
    google_key = (getattr(settings, "google_api_key", "") or "").strip()
    if (not google_key) or google_key == "dummy":
        return "openai"
    return "google"


async def check_google_if_react_provider_is_google(*, timeout_seconds: float = 20.0) -> SanityCheckResult:
    """
    If ReAct provider is Google, perform a minimal real call to Gemini.
    Otherwise, return OK with a SKIPPED detail.
    """
    name = "google"

    provider = _get_react_provider()
    if provider != "google":
        return SanityCheckResult(
            name=name,
            ok=True,
            detail=f"SKIPPED (react_provider={provider})",
            data={"react_provider": provider},
            error=None,
        )

    async def _run() -> dict[str, Any]:
        handle = create_react_llm(
            purpose="startup_sanity_check",
            thinking_level="low",
            system_prompt=None,
            allow_context_cache=False,
            include_thoughts=False,
        )
        # Minimal invocation to validate credentials and model access.
        resp = await handle.llm.ainvoke([HumanMessage(content="Respond with OK only.")])
        text = getattr(resp, "content", None)
        return {
            "react_provider": provider,
            "model": settings.react_google_llm_model,
            "response": (str(text).strip()[:50] if text is not None else None),
        }

    try:
        data = await asyncio.wait_for(_run(), timeout=timeout_seconds)
        return SanityCheckResult(name=name, ok=True, detail="OK", data=data)
    except Exception as exc:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="Google Gemini sanity check failed",
            data={"react_provider": provider, "model": settings.react_google_llm_model},
            error=repr(exc) + "\n" + traceback.format_exc(),
        )


