"""
ReAct LLM factory (thin wrapper).

옵션 A(레이어링):
- 실제 LLM 생성/정책은 `app/core/llm_factory.py`에서 담당한다.
- ReAct 구성요소는 이 모듈을 통해서만 LLM을 생성한다.

NOTE:
- Gemini context caching 기능은 정책상 제거되었다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import settings
from app.core.llm_factory import LLMHandle, create_llm


@dataclass(frozen=True)
class ReactLLMHandle:
    llm: object
    provider: str
    model: str
    # Backward-compatible fields (cache feature removed)
    cached_content_name: Optional[str] = None

    @property
    def uses_context_cache(self) -> bool:
        # Gemini context caching is no longer supported.
        return False


def create_react_llm(
    *,
    purpose: str,
    thinking_level: Optional[str] = "low",
    system_prompt: Optional[str] = None,  # kept for backward compatibility (ignored)
    allow_context_cache: bool = True,  # kept for backward compatibility (ignored)
    include_thoughts: bool = True,
    # Default to deterministic decoding for stability (can be overridden by callers).
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
    use_light: bool = False,
) -> ReactLLMHandle:
    handle: LLMHandle = create_llm(
        purpose=purpose,
        thinking_level=thinking_level,
        include_thoughts=include_thoughts,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        provider=(settings.light_llm_provider if use_light else None),
        model=(settings.light_llm_model if use_light else None),
        provider_url=(
            ((settings.light_llm_provider_url or "").strip() or None) if use_light else None
        ),
    )
    return ReactLLMHandle(
        llm=handle.llm,
        provider=handle.provider,
        model=handle.model,
        cached_content_name=None,
    )
