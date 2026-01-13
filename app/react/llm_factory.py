"""
Shared LLM factory for ReAct components.

Centralizes construction of the LLM used by:
- ReactAgent
- ExplainAnalysisGenerator

Supports both OpenAI and Google Gemini based on REACT_LLM_PROVIDER setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.config import settings
from app.react.gemini_context_cache import GeminiCachedContentManager


@dataclass(frozen=True)
class ReactLLMHandle:
    llm: Union[ChatGoogleGenerativeAI, ChatOpenAI]
    cached_content_name: Optional[str] = None

    @property
    def uses_context_cache(self) -> bool:
        return bool(self.cached_content_name)


_cache_manager: Optional[GeminiCachedContentManager] = None


def _get_cache_manager() -> Optional[GeminiCachedContentManager]:
    global _cache_manager
    if _cache_manager is not None:
        return _cache_manager
    api_key = getattr(settings, "google_api_key", "") or ""
    if not api_key.strip() or api_key.strip() == "dummy":
        return None
    _cache_manager = GeminiCachedContentManager(api_key=api_key)
    return _cache_manager


def _get_react_llm_provider() -> str:
    """
    Determine which LLM provider to use for ReAct.
    
    Priority:
    1. REACT_LLM_PROVIDER environment variable (if set)
    2. If GOOGLE_API_KEY is missing or "dummy", use "openai"
    3. Default: "google"
    """
    provider = getattr(settings, "react_llm_provider", "").strip().lower()
    if provider in ("openai", "google"):
        return provider
    
    # Auto-detect: if Google API key is missing or dummy, use OpenAI
    google_key = getattr(settings, "google_api_key", "") or ""
    if not google_key.strip() or google_key.strip() == "dummy":
        return "openai"
    
    return "google"


def create_react_llm(
    *,
    purpose: str,
    thinking_level: str,
    system_prompt: Optional[str] = None,
    allow_context_cache: bool = True,
    include_thoughts: bool = True,
) -> ReactLLMHandle:
    """
    Create the default LLM instance for the ReAct flow.

    NOTE: Centralizing this avoids drift across generators/agents.

    Supports both OpenAI and Google Gemini based on configuration.
    """
    provider = _get_react_llm_provider()
    
    if provider == "openai":
        return _create_openai_llm(purpose=purpose, system_prompt=system_prompt)
    else:
        return _create_google_llm(
            purpose=purpose,
            thinking_level=thinking_level,
            system_prompt=system_prompt,
            allow_context_cache=allow_context_cache,
            include_thoughts=include_thoughts,
        )


def _create_openai_llm(
    *,
    purpose: str,
    system_prompt: Optional[str] = None,
) -> ReactLLMHandle:
    """Create OpenAI LLM for ReAct."""
    # Use react_openai_llm_model if set, otherwise fall back to openai_llm_model
    model = getattr(settings, "react_openai_llm_model", None) or settings.openai_llm_model
    
    print(f"[react.llm] Using OpenAI: model={model} purpose={purpose}")
    
    llm = ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=0.1,  # Low temperature for more deterministic outputs
    )
    
    return ReactLLMHandle(llm=llm, cached_content_name=None)


def _create_google_llm(
    *,
    purpose: str,
    thinking_level: str,
    system_prompt: Optional[str] = None,
    allow_context_cache: bool = True,
    include_thoughts: bool = True,
) -> ReactLLMHandle:
    """Create Google Gemini LLM for ReAct."""
    cached_content_name: Optional[str] = None
    cache_info: Optional[dict] = None

    if (
        allow_context_cache
        and bool(getattr(settings, "gemini_context_cache_enabled", False))
        and (system_prompt is not None)
        and system_prompt.strip()
    ):
        mgr = _get_cache_manager()
        if mgr is not None:
            cached_content_name, cache_info = mgr.get_or_schedule(
                purpose=purpose,
                model=settings.react_google_llm_model,
                system_prompt=system_prompt,
                ttl_seconds=int(getattr(settings, "gemini_context_cache_ttl_seconds", 3600)),
                refresh_buffer_seconds=int(
                    getattr(settings, "gemini_context_cache_refresh_buffer_seconds", 120)
                ),
                retry_backoff_seconds=int(
                    getattr(settings, "gemini_context_cache_retry_backoff_seconds", 60)
                ),
            )
            if cache_info and cache_info.get("status") in {"not_ready"}:
                try:
                    print(
                        f"[gemini.context_cache.pending] purpose={purpose} model={settings.react_google_llm_model} status={cache_info.get('status')}"
                    )
                except Exception:
                    pass

    print(f"[react.llm] Using Google Gemini: model={settings.react_google_llm_model} purpose={purpose}")

    llm_kwargs = dict(
        model=settings.react_google_llm_model,
        google_api_key=settings.google_api_key,
        thinking_level=thinking_level,
        include_thoughts=include_thoughts,
    )
    if cached_content_name:
        llm_kwargs["cached_content"] = cached_content_name

    llm = ChatGoogleGenerativeAI(**llm_kwargs)
    try:
        setattr(llm, "_gemini_cached_content_name", cached_content_name)
        setattr(llm, "_gemini_cache_info", cache_info)
    except Exception:
        pass
    return ReactLLMHandle(llm=llm, cached_content_name=cached_content_name)
