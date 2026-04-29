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
import os
import json
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


def _require_embedding_api_key(*, provider: str) -> str:
    key = (getattr(settings, "embedding_api_key", "") or "").strip()
    if provider == "openai_compatible" and not key:
        key = (os.getenv("OPENROUTER_API_KEY", "") or "").strip()
    if not key:
        key = (getattr(settings, "openai_api_key", "") or "").strip()
    if not key or key.lower() == "dummy":
        missing_name = (
            "EMBEDDING_API_KEY/OPENROUTER_API_KEY"
            if provider == "openai_compatible"
            else "OPENAI_API_KEY"
        )
        raise ValueError(f"{missing_name} is missing (embedding_provider={provider})")
    return key


def _is_google_gemma_model(model: str) -> bool:
    mdl = (model or "").strip().lower()
    return mdl.startswith("gemma-") or mdl.startswith("models/gemma-")


def _google_thinking_kwargs(
    *,
    model: str,
    thinking_level: Optional[str],
    include_thoughts: bool,
) -> Dict[str, Any]:
    level = (thinking_level or "").strip().lower()
    if not level:
        return {}
    if _is_google_gemma_model(model) and level != "high":
        # Gemma 4 31B rejects low/medium/minimal and thinking budgets.
        return {}
    return {"thinking_level": level, "include_thoughts": bool(include_thoughts)}


def _openrouter_extra_body(
    *,
    base_url: str,
    model: str = "",
    thinking_level: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if "openrouter.ai" not in (base_url or "").lower():
        return None

    provider: Dict[str, Any] = {}
    only_raw = (
        os.getenv("OPENROUTER_PROVIDER_ONLY", "")
        or getattr(settings, "openrouter_provider_only", "")
        or ""
    ).strip()
    if only_raw:
        provider["only"] = [item.strip() for item in only_raw.split(",") if item.strip()]

    order_raw = (
        os.getenv("OPENROUTER_PROVIDER_ORDER", "")
        or getattr(settings, "openrouter_provider_order", "")
        or ""
    ).strip()
    if order_raw:
        provider["order"] = [item.strip() for item in order_raw.split(",") if item.strip()]

    allow_fallbacks_raw = (
        os.getenv("OPENROUTER_PROVIDER_ALLOW_FALLBACKS", "")
        or getattr(settings, "openrouter_provider_allow_fallbacks", "")
        or ""
    ).strip().lower()
    if allow_fallbacks_raw in {"true", "1", "yes", "y", "on"}:
        provider["allow_fallbacks"] = True
    elif allow_fallbacks_raw in {"false", "0", "no", "n", "off"}:
        provider["allow_fallbacks"] = False

    require_params_raw = (
        os.getenv("OPENROUTER_PROVIDER_REQUIRE_PARAMETERS", "")
        or getattr(settings, "openrouter_provider_require_parameters", "")
        or ""
    ).strip().lower()
    if require_params_raw in {"true", "1", "yes", "y", "on"}:
        provider["require_parameters"] = True
    elif require_params_raw in {"false", "0", "no", "n", "off"}:
        provider["require_parameters"] = False

    sort_by = (
        os.getenv("OPENROUTER_PROVIDER_SORT_BY", "")
        or getattr(settings, "openrouter_provider_sort_by", "")
        or ""
    ).strip()
    sort_partition = (
        os.getenv("OPENROUTER_PROVIDER_SORT_PARTITION", "")
        or getattr(settings, "openrouter_provider_sort_partition", "")
        or ""
    ).strip()
    if sort_by:
        sort: Dict[str, Any] = {"by": sort_by}
        if sort_partition:
            sort["partition"] = sort_partition
        provider["sort"] = sort

    extra_body: Dict[str, Any] = {}
    if provider:
        extra_body["provider"] = provider

    reasoning_level = (thinking_level or "").strip().lower()
    if reasoning_level and _should_send_openrouter_reasoning(model=model):
        extra_body["reasoning"] = {"effort": reasoning_level}

    models_raw = (
        os.getenv("OPENROUTER_MODELS", "")
        or getattr(settings, "openrouter_models", "")
        or ""
    ).strip()
    if models_raw:
        try:
            models = json.loads(models_raw)
            if isinstance(models, list) and all(isinstance(item, str) for item in models):
                extra_body["models"] = models
        except json.JSONDecodeError:
            models = [item.strip() for item in models_raw.split(",") if item.strip()]
            if models:
                extra_body["models"] = models

    return extra_body or None


def _truthy_env_text(value: str) -> Optional[bool]:
    v = (value or "").strip().lower()
    if v in {"true", "1", "yes", "y", "on"}:
        return True
    if v in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _split_csv_text(value: str) -> list[str]:
    return [x.strip().lower() for x in (value or "").split(",") if x.strip()]


def _should_send_openrouter_reasoning(*, model: str) -> bool:
    enabled_raw = (
        os.getenv("OPENROUTER_REASONING_ENABLED", "")
        or getattr(settings, "openrouter_reasoning_enabled", "")
        or ""
    )
    enabled = _truthy_env_text(enabled_raw)
    if enabled is False:
        return False

    mdl = (model or "").strip().lower()
    disabled_raw = (
        os.getenv("OPENROUTER_REASONING_DISABLED_MODELS", "")
        or getattr(settings, "openrouter_reasoning_disabled_models", "")
        or ""
    )
    disabled_models = _split_csv_text(disabled_raw)
    if mdl and any(mdl == disabled or mdl.endswith("/" + disabled) for disabled in disabled_models):
        return False

    # Auto mode: reasoning is allowed for models not explicitly disabled.
    # Explicit true also cannot override a per-model disabled entry; remove the model
    # from OPENROUTER_REASONING_DISABLED_MODELS if a future provider fixes support.
    return True


def _get_embedding_base_url(*, provider: str) -> str:
    if provider == "openai":
        return ""
    base_url = (getattr(settings, "embedding_provider_url", "") or "").strip()
    if not base_url:
        raise ValueError(
            "EMBEDDING_PROVIDER_URL is required when embedding_provider=openai_compatible"
        )
    return base_url


@lru_cache(maxsize=8)
def _get_embedding_async_client(*, provider: str, api_key: str, base_url: str) -> AsyncOpenAI:
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def create_embedding_client() -> EmbeddingClient:
    provider = (getattr(settings, "embedding_provider", "") or "").strip().lower()
    if provider not in {"openai", "openai_compatible"}:
        raise NotImplementedError(
            "embedding_provider={!r} is not supported yet (allowed: 'openai', "
            "'openai_compatible').".format(provider)
        )
    api_key = _require_embedding_api_key(provider=provider)
    base_url = _get_embedding_base_url(provider=provider)
    client = _get_embedding_async_client(provider=provider, api_key=api_key, base_url=base_url)
    return EmbeddingClient(client)


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
            "extra_body": _openrouter_extra_body(base_url=base_url, model=mdl, thinking_level=thinking_level),
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
        "max_output_tokens": int(max_output_tokens) if max_output_tokens is not None else None,
    }
    raw_kwargs.update(
        _google_thinking_kwargs(
            model=mdl,
            thinking_level=thinking_level,
            include_thoughts=include_thoughts,
        )
    )
    kwargs = _filter_init_kwargs(ChatGoogleGenerativeAI, raw_kwargs)
    llm = ChatGoogleGenerativeAI(**kwargs)
    try:
        print(f"[core.llm] Using Google: model={mdl} purpose={purpose}")
    except Exception:
        pass
    return LLMHandle(llm=llm, provider=prov, model=mdl)


