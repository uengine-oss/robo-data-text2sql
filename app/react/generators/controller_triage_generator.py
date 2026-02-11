from __future__ import annotations

import json
import time
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.llm_factory import ReactLLMHandle, create_react_llm
from app.react.prompts import get_prompt_text
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None
    left = s.find("{")
    right = s.rfind("}")
    if left < 0 or right <= left:
        return None
    cand = s[left : right + 1]
    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class ControllerTriageGenerator:
    """
    One-shot triage LLM used when the controller cannot find acceptable SQL.
    Prompt is loaded from prompts/controller_triage_prompt.md.
    """

    _PROMPT_FILE = "controller_triage_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="controller_triage",
            thinking_level="low",
            include_thoughts=False,
            temperature=0.0,
            max_output_tokens=900,
            use_light=False,
        )

    async def generate(
        self,
        *,
        payload: Dict[str, Any],
        react_run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        human_text = json.dumps(payload or {}, ensure_ascii=False)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_text),
        ]
        started = time.perf_counter()
        try:
            resp = await self.llm_handle.llm.ainvoke(messages)
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.controller_triage.error",
                category="react.llm.repro.controller_triage",
                react_run_id=react_run_id,
                generator="controller_triage_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload or {},
                messages_payload={"system": self.system_prompt, "human": human_text},
                mode="json_text",
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            SmartLogger.log(
                "WARNING",
                "react.controller_triage.llm_failed",
                category="react.controller_triage",
                params=sanitize_for_log(
                    {"react_run_id": react_run_id, "exception": repr(exc), "traceback": traceback.format_exc()}
                ),
                max_inline_chars=0,
            )
            return None
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        text = _content_to_text(getattr(resp, "content", ""))
        obj = _extract_first_json_object(text)

        SmartLogger.log(
            "INFO",
            "react.controller_triage.llm",
            category="react.controller_triage",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "elapsed_ms": elapsed_ms,
                    "ok": bool(isinstance(obj, dict) and obj),
                }
            ),
            max_inline_chars=0,
        )
        log_llm_repro(
            level="INFO",
            message="react.llm.repro.controller_triage.ok",
            category="react.llm.repro.controller_triage",
            react_run_id=react_run_id,
            generator="controller_triage_generator",
            llm_provider=self.llm_handle.provider,
            llm_model=self.llm_handle.model,
            prompt=self.prompt_meta,
            input_payload=payload or {},
            messages_payload={"system": self.system_prompt, "human": human_text},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=text,
            parsed=obj,
        )
        return obj if isinstance(obj, dict) else None


@lru_cache(maxsize=1)
def get_controller_triage_generator() -> ControllerTriageGenerator:
    return ControllerTriageGenerator()

