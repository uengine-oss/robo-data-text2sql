from __future__ import annotations

import asyncio
import traceback
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.core.llm_factory import create_embedding_client, create_llm
from app.sanity_checks.result import SanityCheckResult


def _norm_llm_provider(v: str) -> str:
    p = (v or "").strip().lower()
    if p in {"gemini", "google", "genai"}:
        return "google"
    if p in {"openai_compatible", "openai-compatible", "openai_compat"}:
        return "openai_compatible"
    return p


async def check_openai(*, timeout_seconds: float = 20.0) -> SanityCheckResult:
    """
    OpenAI connectivity check (fail-fast only when OpenAI is required).

    Runs when:
    - embedding_provider == "openai" (embeddings check)
    - llm_provider in {"openai", "openai_compatible"} (LLM check via core factory)

    Notes:
    - When embedding_provider=="openai", this uses OPENAI_API_KEY against the official OpenAI endpoint.
    - When llm_provider=="openai_compatible", this uses OPENAI_COMPATIBLE_API_KEY (or falls back to
      OPENAI_API_KEY) against settings.llm_provider_url.
    """
    name = "openai"

    embedding_provider = (getattr(settings, "embedding_provider", "") or "").strip().lower()
    llm_provider = _norm_llm_provider(getattr(settings, "llm_provider", "") or "")
    needs_openai = (embedding_provider == "openai") or (llm_provider in {"openai", "openai_compatible"})
    if not needs_openai:
        return SanityCheckResult(
            name=name,
            ok=True,
            detail=f"SKIPPED (embedding_provider={embedding_provider}, llm_provider={llm_provider})",
            data={"embedding_provider": embedding_provider, "llm_provider": llm_provider},
            error=None,
        )

    async def _run() -> dict[str, Any]:
        data: dict[str, Any] = {
            "embedding_provider": embedding_provider,
            "llm_provider": llm_provider,
        }

        if embedding_provider == "openai":
            try:
                key = (getattr(settings, "openai_api_key", "") or "").strip()
                if (not key) or key.lower() == "dummy":
                    raise RuntimeError("OPENAI_API_KEY is missing but embedding_provider=openai")
                embedder = create_embedding_client()
                vec = await embedder.embed_text("startup-sanity")
                dim = len(vec)
                expected_dim = int(settings.embedding_dimension)
                if int(dim) != expected_dim:
                    raise RuntimeError(f"Embedding dimension mismatch: expected={expected_dim} actual={dim}")
                data.update(
                    {
                        "embedding_model": settings.embedding_model,
                        "embedding_dim": dim,
                        "embedding_check": "ok",
                    }
                )
            except Exception as exc:
                # Make it unambiguous which step failed when we only see logs.
                raise RuntimeError(f"[stage=embeddings] {exc!r}") from exc

        if llm_provider in {"openai", "openai_compatible"}:
            try:
                if llm_provider == "openai_compatible":
                    data["llm_provider_url"] = (getattr(settings, "llm_provider_url", "") or "").strip() or None
                llm = create_llm(
                    purpose="startup_sanity_check",
                    thinking_level="low",
                    include_thoughts=False,
                    temperature=0,
                    # Some OpenAI-compatible providers may return extra metadata fields
                    # (e.g. `reasoning`). Give enough budget to always produce `content`.
                    max_output_tokens=128,
                ).llm
                resp = await llm.ainvoke(
                    [
                        SystemMessage(content="Return exactly: OK"),
                        HumanMessage(content="Say OK"),
                    ]
                )
                text = (getattr(resp, "content", None) or "")
                text_s = str(text).strip()

                # Some OpenAI-compatible providers (notably certain reasoning-style models)
                # may return an empty `content` while providing a valid response with
                # finish_reason=length/stop and token usage metadata (reasoning may be
                # provided out-of-band, e.g. message.reasoning, which LangChain may not surface).
                response_metadata = getattr(resp, "response_metadata", None) or {}
                finish_reason = response_metadata.get("finish_reason")
                token_usage = response_metadata.get("token_usage")

                # Refusal should still fail fast.
                additional_kwargs = getattr(resp, "additional_kwargs", None) or {}
                refusal = additional_kwargs.get("refusal")
                if refusal:
                    raise RuntimeError(f"LLM refusal: {str(refusal)[:200]!r}")

                # Success criteria:
                # - Prefer strict OK when present
                # - Otherwise accept a well-formed response with finish_reason in {stop,length}
                if text_s != "OK" and finish_reason not in {"stop", "length"}:
                    raise RuntimeError(
                        f"Unexpected chat content: {text_s[:50]!r} (finish_reason={finish_reason!r})"
                    )
                data.update(
                    {
                        "llm_model": settings.llm_model,
                        "chat_response": text_s[:50],
                        "chat_finish_reason": finish_reason,
                        "chat_token_usage": token_usage,
                        "llm_check": "ok",
                    }
                )
            except Exception as exc:
                raise RuntimeError(f"[stage=llm] {exc!r}") from exc

        return data

    try:
        data = await asyncio.wait_for(_run(), timeout=timeout_seconds)
        return SanityCheckResult(name=name, ok=True, detail="OK", data=data)
    except Exception as exc:
        # Try to preserve a small "stage" hint in logs without leaking secrets.
        stage: str | None = None
        msg = str(exc)
        if "[stage=embeddings]" in msg:
            stage = "embeddings"
        elif "[stage=llm]" in msg:
            stage = "llm"
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="OpenAI sanity check failed",
            data={
                "stage": stage,
                "embedding_provider": embedding_provider,
                "embedding_model": (settings.embedding_model if embedding_provider == "openai" else None),
                "llm_provider": llm_provider,
                "llm_model": (settings.llm_model if llm_provider in {"openai", "openai_compatible"} else None),
                "llm_provider_url": (
                    (getattr(settings, "llm_provider_url", "") or "").strip() or None
                    if llm_provider == "openai_compatible"
                    else None
                ),
            },
            error=repr(exc) + "\n" + traceback.format_exc(),
        )


