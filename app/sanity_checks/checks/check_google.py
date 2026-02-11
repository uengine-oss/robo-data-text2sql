from __future__ import annotations

import asyncio
import traceback
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import settings
from app.core.llm_factory import create_llm
from app.sanity_checks.result import SanityCheckResult


def _norm_llm_provider(v: str) -> str:
    p = (v or "").strip().lower()
    if p in {"gemini", "google", "genai"}:
        return "google"
    return p


async def check_google_if_react_provider_is_google(*, timeout_seconds: float = 20.0) -> SanityCheckResult:
    """
    If unified llm_provider is Google, perform a minimal real call to Gemini.
    Otherwise, return OK with a SKIPPED detail.
    """
    name = "google"

    provider = _norm_llm_provider(getattr(settings, "llm_provider", "") or "")
    if provider != "google":
        return SanityCheckResult(
            name=name,
            ok=True,
            detail=f"SKIPPED (llm_provider={provider})",
            data={"llm_provider": provider},
            error=None,
        )

    async def _run() -> dict[str, Any]:
        handle = create_llm(
            purpose="startup_sanity_check",
            thinking_level="low",
            include_thoughts=False,
            temperature=0,
            max_output_tokens=10,
        )
        # Minimal invocation to validate credentials and model access.
        resp = await handle.llm.ainvoke([HumanMessage(content="Respond with OK only.")])
        text = getattr(resp, "content", None)
        return {
            "llm_provider": provider,
            "model": settings.llm_model,
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
            data={"llm_provider": provider, "model": settings.llm_model},
            error=repr(exc) + "\n" + traceback.format_exc(),
        )


