from __future__ import annotations

import asyncio
import traceback
from typing import Any

from app.config import settings
from app.deps import openai_client
from app.sanity_checks.result import SanityCheckResult


def _get_react_provider() -> str:
    """
    Match ReAct provider selection logic (same intent as app.react.llm_factory).
    - If react_llm_provider is explicitly set to openai/google, respect it.
    - Else auto-detect: if google_api_key is missing/dummy -> openai, otherwise google.
    """
    provider = (getattr(settings, "react_llm_provider", "") or "").strip().lower()
    if provider in {"openai", "google"}:
        return provider
    google_key = (getattr(settings, "google_api_key", "") or "").strip()
    if (not google_key) or google_key.lower() == "dummy":
        return "openai"
    return "google"


async def check_openai(*, timeout_seconds: float = 20.0) -> SanityCheckResult:
    """
    OpenAI connectivity check (lightweight but real calls).

    Performs:
    - embeddings.create (1 short input) and verifies embedding_dimension consistency
    - chat.completions.create (minimal tokens) to validate LLM access
    - if ReAct provider is OpenAI, also validates react_openai_llm_model via a direct OpenAI call
    """
    name = "openai"

    async def _run() -> dict[str, Any]:
        emb = await openai_client.embeddings.create(
            model=settings.openai_embedding_model,
            input="startup-sanity",
            encoding_format="float",
        )
        dim = len(emb.data[0].embedding)
        expected_dim = int(settings.embedding_dimension)
        if int(dim) != expected_dim:
            raise RuntimeError(f"Embedding dimension mismatch: expected={expected_dim} actual={dim}")

        chat = await openai_client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[{"role": "user", "content": "Respond with OK only."}],
            max_tokens=5,
            temperature=0,
        )
        text = (chat.choices[0].message.content or "").strip()

        react_provider = _get_react_provider()
        react_model = None
        react_text = None
        if react_provider == "openai":
            react_model = (getattr(settings, "react_openai_llm_model", "") or "").strip() or settings.openai_llm_model
            react_chat = await openai_client.chat.completions.create(
                model=react_model,
                messages=[{"role": "user", "content": "Respond with OK only."}],
                max_tokens=5,
                temperature=0,
            )
            react_text = (react_chat.choices[0].message.content or "").strip()

        return {
            "embedding_model": settings.openai_embedding_model,
            "embedding_dim": dim,
            "llm_model": settings.openai_llm_model,
            "chat_response": text[:50],
            "react_provider": react_provider,
            "react_llm_model": react_model,
            "react_chat_response": (react_text[:50] if react_text is not None else None),
        }

    try:
        data = await asyncio.wait_for(_run(), timeout=timeout_seconds)
        return SanityCheckResult(name=name, ok=True, detail="OK", data=data)
    except Exception as exc:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="OpenAI sanity check failed",
            data={
                "embedding_model": settings.openai_embedding_model,
                "llm_model": settings.openai_llm_model,
                "react_provider": _get_react_provider(),
                "react_llm_model": (getattr(settings, "react_openai_llm_model", None) if _get_react_provider() == "openai" else None),
            },
            error=repr(exc) + "\n" + traceback.format_exc(),
        )


