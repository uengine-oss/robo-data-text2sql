"""
Shared LLM factory for ReAct components.

Centralizes construction of the LLM used by:
- ReactAgent
- ExplainAnalysisGenerator
"""

from __future__ import annotations

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


@lru_cache(maxsize=1)
def create_react_llm() -> ChatGoogleGenerativeAI:
    """
    Create the default LLM instance for the ReAct flow.

    NOTE: Centralizing this avoids drift across generators/agents.

    This factory is cached so repeated calls return the same LLM instance
    within the current process.
    """
    return ChatGoogleGenerativeAI(
        model=settings.react_google_llm_model,
        google_api_key=settings.google_api_key,
        thinking_level="low",
    )


