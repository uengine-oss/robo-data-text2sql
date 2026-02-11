from __future__ import annotations

import json
import time
import traceback
from functools import lru_cache
from typing import Any, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.llm_factory import ReactLLMHandle, create_react_llm
from app.react.prompts import get_prompt_text
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


class IntentOut(BaseModel):
    intent: str = Field(..., description="One-line intent sentence")


def _parse_one_line_intent(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    # Keep only first line
    raw = raw.splitlines()[0].strip()
    # Remove surrounding quotes/backticks if the model still added them
    if len(raw) >= 2 and ((raw[0] == raw[-1] == '"') or (raw[0] == raw[-1] == "'")):
        raw = raw[1:-1].strip()
    raw = raw.strip("`").strip()
    return raw


def _strip_code_fences(text: str) -> str:
    """
    Remove common markdown code fences while keeping inner content.
    """
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        lines = s.splitlines()
        # Drop first fence line and the last fence line if present
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            inner = "\n".join(lines[1:-1])
        else:
            inner = "\n".join(lines[1:])
        return inner.strip()
    return s


def _try_parse_intent_json(text: str) -> Optional[str]:
    """
    Parse {"intent": "..."} from a raw LLM text response.
    Returns intent string if valid, otherwise None.
    """
    s = _strip_code_fences(text)
    if not s:
        return None
    # First, try direct JSON
    try:
        obj = json.loads(s)
    except Exception:
        # Fallback: extract first {...} region
        left = s.find("{")
        right = s.rfind("}")
        if left >= 0 and right > left:
            try:
                obj = json.loads(s[left : right + 1])
            except Exception:
                return None
        else:
            return None
    if not isinstance(obj, dict):
        return None
    v = obj.get("intent")
    if not isinstance(v, str):
        return None
    out = _parse_one_line_intent(v)
    return out or None


class IntentExtractGenerator:
    """
    One-shot intent extraction generator for SQL grounding.
    Prompt is loaded from prompts/intent_extract_prompt.md. Responsibility: LLM call + parsing only.
    """

    _PROMPT_FILE = "intent_extract_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="intent-extract",
            thinking_level=None,
            allow_context_cache=False,
            include_thoughts=False,
            max_output_tokens=80,
            use_light=True,
        )

    async def generate(
        self,
        *,
        question: str,
        react_run_id: Optional[str],
    ) -> Tuple[Optional[str], str]:
        """
        Returns (intent_text, debug_note).
        On failure returns (None, 'llm_error'|'llm_empty').
        """
        q = (question or "").strip()
        if not q:
            return None, "empty_question"
        system_prompt = self.system_prompt

        try:
            started = time.perf_counter()
            mode = "json_text"
            llm = self.llm_handle.llm
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=q)]

            # Prefer structured output when available (provider=openai/google both).
            try:
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(IntentOut)  # type: ignore[attr-defined]
                    resp = await structured_llm.ainvoke(messages)
                    mode = "structured"
                else:
                    resp = await llm.ainvoke(messages)
            except Exception:
                # Fallback to plain text invoke + JSON parsing
                resp = await llm.ainvoke(messages)
                mode = "json_text"
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            intent: Optional[str] = None
            structured_obj: Optional[dict[str, Any]] = None
            if isinstance(resp, IntentOut):
                intent = _parse_one_line_intent(resp.intent)
            elif isinstance(resp, dict):
                structured_obj = resp
            else:
                # Some LC models return pydantic-like objects or messages
                if hasattr(resp, "model_dump"):
                    try:
                        structured_obj = resp.model_dump()  # type: ignore[attr-defined]
                    except Exception:
                        structured_obj = None

            if intent is None and structured_obj is not None:
                v = structured_obj.get("intent")
                if isinstance(v, str):
                    intent = _parse_one_line_intent(v)

            if intent is None:
                content = getattr(resp, "content", None)
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    # LangChain content may be list of dicts; take 'text' parts
                    buf: list[str] = []
                    for part in content:
                        if isinstance(part, str):
                            buf.append(part)
                        elif isinstance(part, dict) and part.get("text"):
                            buf.append(str(part.get("text")))
                    text = "".join(buf)
                else:
                    text = str(content or "")
                # JSON-first (preferred). If it looks like JSON but doesn't match schema,
                # don't fall back to treating the JSON literal as intent.
                json_intent = _try_parse_intent_json(text)
                if json_intent:
                    intent = json_intent
                else:
                    stripped = _strip_code_fences(text).lstrip()
                    if stripped.startswith("{") or stripped.startswith("["):
                        intent = None
                    else:
                        intent = _parse_one_line_intent(text)

            # Avoid degenerate outputs
            if not intent or len(intent) < 2:
                # Repro log even for empty/degenerate outputs
                raw_content = getattr(resp, "content", None) if "resp" in locals() else None
                log_llm_repro(
                    level="WARNING",
                    message="react.llm.repro.intent_extract.llm_empty",
                    category="react.llm.repro.intent_extract",
                    react_run_id=react_run_id,
                    generator="intent_extract_generator",
                    llm_provider=self.llm_handle.provider,
                    llm_model=self.llm_handle.model,
                    prompt=self.prompt_meta,
                    input_payload={"question": q},
                    messages_payload={"system": system_prompt, "human": q},
                    mode=mode,
                    elapsed_ms=elapsed_ms,
                    response_raw=raw_content,
                    parsed={"intent": intent, "structured_obj": structured_obj},
                )
                return None, "llm_empty"

            SmartLogger.log(
                "INFO",
                "react.build_sql_context.intent_extract.llm",
                category="react.tool.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "elapsed_ms": elapsed_ms,
                        "intent_text": intent,
                        "mode": mode,
                    }
                ),
            )
            raw_content = getattr(resp, "content", None)
            if raw_content is None and isinstance(resp, dict):
                raw_content = resp
            elif raw_content is None and hasattr(resp, "model_dump"):
                try:
                    raw_content = resp.model_dump()  # type: ignore[attr-defined]
                except Exception:
                    raw_content = str(resp)
            log_llm_repro(
                level="INFO",
                message="react.llm.repro.intent_extract.ok",
                category="react.llm.repro.intent_extract",
                react_run_id=react_run_id,
                generator="intent_extract_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload={"question": q},
                messages_payload={"system": system_prompt, "human": q},
                mode=mode,
                elapsed_ms=elapsed_ms,
                response_raw=raw_content,
                parsed={"intent": intent, "structured_obj": structured_obj},
                extra={"intent_len": len(intent or "")},
            )
            return intent, ("llm_ok_structured" if mode == "structured" else "llm_ok_json")
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.intent_extract.error",
                category="react.llm.repro.intent_extract",
                react_run_id=react_run_id,
                generator="intent_extract_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload={"question": q},
                messages_payload={"system": system_prompt, "human": q},
                mode=None,
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            SmartLogger.log(
                "WARNING",
                "react.build_sql_context.intent_extract.llm_failed",
                category="react.tool.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
            )
            return None, "llm_error"


@lru_cache(maxsize=1)
def get_intent_extract_generator() -> IntentExtractGenerator:
    """Singleton/cached generator instance."""
    return IntentExtractGenerator()

