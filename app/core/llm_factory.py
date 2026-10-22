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


# ---------------------------------------------------------------------------
# 임베딩 제공자 심 (spec 007 T1 / FR-020~FR-023)
# ---------------------------------------------------------------------------
# 이전 판:
#     AsyncOpenAI(api_key=api_key)        <- base_url 인자가 없다
#     provider != "openai" -> NotImplementedError
# 이 둘이 함께 있어서 "로컬 임베딩"은 **어떤 설정으로도 도달할 수 없는 상태**였다.
# 여기가 그 두 결함을 동시에 닫는 유일한 자리다 (FR-023: 벡터 생성 지점은 앱 전역 한 곳).

EMBEDDING_PROVIDERS = ("openai", "llm_endpoint", "local")


class EmbeddingProviderUnavailable(RuntimeError):
    """제공자가 선택은 됐으나 지금 쓸 수 없다 (설정 누락 또는 아직 미구현).

    `NotImplementedError` 를 쓰지 않는 이유: 호출부가 "코드가 없다"와 "설정이 없다"를
    구분해서 사용자에게 다른 안내를 해야 하기 때문이다. 이전 판은 둘을 섞어
    설정 문제까지 "지원하지 않음"으로 보고했다.
    """


def normalize_embedding_provider(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"", "openai"}:
        return "openai"
    if v in {"llm_endpoint", "llm-endpoint", "openai_compatible", "openai-compatible"}:
        # openai_compatible 은 LLM 쪽 표기다. 임베딩에서는 같은 뜻이므로 흡수한다.
        return "llm_endpoint"
    if v in {"local", "bundled"}:
        return "local"
    raise ValueError(
        "Unsupported embedding_provider={!r}. Allowed: {}".format(value, ", ".join(EMBEDDING_PROVIDERS))
    )


def resolve_embedding_endpoint() -> Dict[str, Any]:
    """제공자별로 (base_url, api_key) 를 푼다. 실제 호출은 하지 않는다.

    설정 화면(T4)과 상태 API 가 "지금 이 설정으로 임베딩이 가능한가"를 **왕복 없이**
    말할 수 있어야 해서 분리했다.
    """
    provider = normalize_embedding_provider(getattr(settings, "embedding_provider", ""))
    explicit_key = (getattr(settings, "embedding_api_key", "") or "").strip()

    if provider == "openai":
        key = explicit_key or (getattr(settings, "openai_api_key", "") or "").strip()
        if not key or key.lower() == "dummy":
            raise EmbeddingProviderUnavailable(
                "OPENAI_API_KEY (or EMBEDDING_API_KEY) is missing while embedding_provider=openai"
            )
        return {"provider": provider, "base_url": None, "api_key": key}

    if provider == "llm_endpoint":
        base_url = (getattr(settings, "embedding_provider_url", "") or "").strip()
        if not base_url:
            # 1순위는 "사용자가 이미 붙인 그 엔드포인트 재사용" 이므로 LLM URL 로 폴백한다.
            base_url = (getattr(settings, "llm_provider_url", "") or "").strip()
        if not base_url:
            raise EmbeddingProviderUnavailable(
                "EMBEDDING_PROVIDER_URL (or LLM_PROVIDER_URL) is required when embedding_provider=llm_endpoint"
            )
        key = (
            explicit_key
            or (getattr(settings, "openai_compatible_api_key", "") or "").strip()
            or (getattr(settings, "openai_api_key", "") or "").strip()
        )
        # 로컬 서버(Ollama·LM Studio·vLLM)는 대개 키를 검사하지 않는다. 여기서 키를
        # 강제하면 "키 없이 로컬 임베딩" 이라는 이 트랙의 목적 자체가 깨진다.
        # openai SDK 는 빈 문자열을 거부하므로 placeholder 를 넣는다.
        return {"provider": provider, "base_url": base_url, "api_key": key or "not-needed"}

    # provider == "local"
    raise EmbeddingProviderUnavailable(
        "embedding_provider=local (번들 모델)은 아직 준비 중입니다. "
        "분석기 서비스가 POST /robo/embed 로 모델을 호스트한 뒤에 열립니다 (spec 007 A2 / FR-100). "
        "지금은 'llm_endpoint'(로컬 OpenAI 호환 서버 재사용) 또는 'openai' 를 쓰세요."
    )


@lru_cache(maxsize=1)
def _get_embedding_async_client_cached(base_url: Optional[str], api_key: str) -> AsyncOpenAI:
    # base_url/api_key 를 캐시 키로 둔다 — 설정을 바꾸면 다른 클라이언트가 나와야 한다.
    # (이전 판은 maxsize=1 캐시가 전역 클라이언트 한 개를 영구 고정했다.)
    if base_url:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)
    return AsyncOpenAI(api_key=api_key)


def create_embedding_client() -> EmbeddingClient:
    """앱 전역에서 임베딩 벡터를 만드는 유일한 진입점 (FR-023)."""
    ep = resolve_embedding_endpoint()
    client = _get_embedding_async_client_cached(ep["base_url"], ep["api_key"])
    return EmbeddingClient(
        client,
        provider=ep["provider"],
        model=settings.embedding_model,
        base_url=ep["base_url"],
    )


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


