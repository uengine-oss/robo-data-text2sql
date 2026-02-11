"""
Core LLM factory.

Unifies all LLM construction behind:
- settings.llm_provider
- settings.llm_model

This is intentionally kept in core (옵션 A 레이어링):
- core code uses core factory directly
- react/llm_factory.py is a thin wrapper for ReAct components only

NOTE:
- Provider aliases: "gemini" -> "google"
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.config import settings
from app.core.embedding import EmbeddingClient

LLMProvider = Literal["openai", "google", "openai_compatible"]
ChatModel = Union[ChatOpenAI, ChatGoogleGenerativeAI]

#
# Compatibility shim:
# google-genai currently references aiohttp.ClientConnectorDNSError in an except clause.
# Some aiohttp versions do not expose that symbol, causing AttributeError at runtime
# even before the request is sent. Map it to ClientConnectorError when missing.
#
try:  # pragma: no cover
    import aiohttp  # type: ignore

    if not hasattr(aiohttp, "ClientConnectorDNSError") and hasattr(aiohttp, "ClientConnectorError"):
        aiohttp.ClientConnectorDNSError = aiohttp.ClientConnectorError  # type: ignore[attr-defined]
except Exception:
    pass


def _normalize_provider(value: str) -> LLMProvider:
    v = (value or "").strip().lower()
    if v in {"google", "gemini", "genai"}:
        return "google"
    if v in {"openai"}:
        return "openai"
    if v in {"openai_compatible", "openai-compatible", "openai_compat"}:
        return "openai_compatible"
    raise ValueError(
        "Unsupported llm_provider={!r}. Allowed: 'openai', 'google' (alias: 'gemini'), "
        "'openai_compatible'.".format(value)
    )


def _filter_init_kwargs(cls: type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep compatibility across LangChain versions by passing only supported kwargs.
    """
    # Pydantic-based LangChain models (notably ChatGoogleGenerativeAI) expose their
    # accepted init keys as model fields, while their __init__ signature is often
    # just (**data). In that case, signature-based filtering would incorrectly drop
    # required fields like `model`.
    try:
        model_fields = getattr(cls, "model_fields", None)  # pydantic v2
        if isinstance(model_fields, dict) and model_fields:
            allowed = set(model_fields.keys())
            return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
        legacy_fields = getattr(cls, "__fields__", None)  # pydantic v1 fallback
        if isinstance(legacy_fields, dict) and legacy_fields:
            allowed = set(legacy_fields.keys())
            return {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    except Exception:
        # Fall back to signature-based filtering below.
        pass

    try:
        sig = inspect.signature(cls.__init__)
        # If the init accepts **kwargs/**data, don't filter by signature names.
        # (Otherwise we'd only allow the var-keyword parameter name.)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return {k: v for k, v in kwargs.items() if v is not None}
        allowed = set(sig.parameters.keys())
    except Exception:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in allowed and v is not None}


def _require_api_key(*, provider: LLMProvider) -> str:
    if provider == "openai":
        key = (getattr(settings, "openai_api_key", "") or "").strip()
        if not key or key.lower() == "dummy":
            raise ValueError("OPENAI_API_KEY is missing (llm_provider=openai)")
        return key
    if provider == "openai_compatible":
        # Prefer dedicated compatible key; fall back to OPENAI_API_KEY for backward compatibility.
        key = (getattr(settings, "openai_compatible_api_key", "") or "").strip()
        if not key:
            key = (getattr(settings, "openai_api_key", "") or "").strip()
        if not key or key.lower() == "dummy":
            raise ValueError("OPENAI_COMPATIBLE_API_KEY is missing (llm_provider=openai_compatible)")
        return key
    key = (getattr(settings, "google_api_key", "") or "").strip()
    if not key or key.lower() == "dummy":
        raise ValueError("GOOGLE_API_KEY is missing (llm_provider=google)")
    return key


@lru_cache(maxsize=1)
def _get_openai_async_client() -> AsyncOpenAI:
    api_key = _require_api_key(provider="openai")
    return AsyncOpenAI(api_key=api_key)


def create_embedding_client() -> EmbeddingClient:
    provider = (getattr(settings, "embedding_provider", "") or "").strip().lower()
    if provider != "openai":
        raise NotImplementedError(
            f"embedding_provider={provider!r} is not supported yet (only 'openai' is supported)."
        )
    return EmbeddingClient(_get_openai_async_client())


@dataclass(frozen=True)
class LLMHandle:
    llm: ChatModel
    provider: LLMProvider
    model: str


def create_llm(
    *,
    purpose: str,
    thinking_level: Optional[str] = "low",
    include_thoughts: bool = False,
    temperature: float = 0.1,
    max_output_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    provider_url: Optional[str] = None,
) -> LLMHandle:
    """
    Create a LangChain chat model using unified settings.

    Args:
        purpose: for logging/diagnostics (not used for routing)
        thinking_level/include_thoughts: used for Gemini when supported by the installed LC version
        provider/model: override settings.llm_provider/settings.llm_model when provided
    """
    prov: LLMProvider = _normalize_provider(provider or settings.llm_provider)
    mdl = (model or settings.llm_model or "").strip()
    if not mdl:
        raise ValueError("llm_model is empty")

    if prov in {"openai", "openai_compatible"}:
        api_key = _require_api_key(provider=prov)
        base_url = (
            (provider_url if provider_url is not None else getattr(settings, "llm_provider_url", "") or "")
        ).strip()
        if prov == "openai_compatible" and not base_url:
            raise ValueError("llm_provider_url is required when llm_provider=openai_compatible")
        raw_kwargs: Dict[str, Any] = {
            # LangChain's ChatOpenAI has shifted parameter names across versions.
            # Keep compatibility by providing both the legacy and the newer names,
            # then filter via _filter_init_kwargs().
            "model": mdl,
            "model_name": mdl,
            "api_key": api_key,
            "openai_api_key": api_key,
            "temperature": float(temperature),
            "max_tokens": int(max_output_tokens) if max_output_tokens is not None else None,
            # Keep compatibility across langchain_openai versions:
            # - Some versions accept `base_url`
            # - Some accept `openai_api_base`
            "base_url": base_url or None,
            "openai_api_base": base_url or None,
        }
        kwargs = _filter_init_kwargs(ChatOpenAI, raw_kwargs)
        llm = ChatOpenAI(**kwargs)
        try:
            if base_url:
                print(f"[core.llm] Using OpenAI: model={mdl} base_url={base_url} purpose={purpose}")
            else:
                print(f"[core.llm] Using OpenAI: model={mdl} purpose={purpose}")
        except Exception:
            pass
        return LLMHandle(llm=llm, provider=prov, model=mdl)

    # google
    api_key = _require_api_key(provider=prov)
    raw_kwargs = {
        "model": mdl,
        "google_api_key": api_key,
        "temperature": float(temperature),
        "thinking_level": thinking_level,
        "include_thoughts": bool(include_thoughts),
        "max_output_tokens": int(max_output_tokens) if max_output_tokens is not None else None,
    }
    kwargs = _filter_init_kwargs(ChatGoogleGenerativeAI, raw_kwargs)
    llm = ChatGoogleGenerativeAI(**kwargs)
    try:
        print(f"[core.llm] Using Google: model={mdl} purpose={purpose}")
    except Exception:
        pass
    return LLMHandle(llm=llm, provider=prov, model=mdl)


