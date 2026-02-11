from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.llm_factory import ReactLLMHandle, create_react_llm
from app.react.prompts import get_prompt_text
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


class LLMXmlReprintGenerator:
    """
    One-shot repair generator: ask the LLM to reprint ONLY valid XML (no extra text).
    Kept as a generator for maintainability; prompt is loaded from prompts/llm_xml_reprint_prompt.md.
    """

    _PROMPT_FILE = "llm_xml_reprint_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        # Load prompt from app/react/prompts/llm_xml_reprint_prompt.md
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="xml-reprint",
            thinking_level="low",
            allow_context_cache=False,
            include_thoughts=False,
        )

    @staticmethod
    def _extract_text_only(content: Any) -> str:
        """
        Extract only 'text' parts from Gemini/LangChain content payloads.
        Avoid polluting XML with structured 'thinking' dicts.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if "text" in content and str(content.get("text") or ""):
                return str(content.get("text") or "")
            return ""
        if isinstance(content, list):
            out: list[str] = []
            for part in content:
                if isinstance(part, str) and part:
                    out.append(part)
                    continue
                if isinstance(part, dict) and "text" in part and str(part.get("text") or ""):
                    out.append(str(part.get("text") or ""))
                    continue
            return "".join(out)
        return str(content)

    async def generate(
        self,
        raw_llm_text: str,
        *,
        react_run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> str:
        system_prompt = self.system_prompt
        human_text = (
            "Fix the following content into a valid <output> XML only:\n\n"
            f"{raw_llm_text}"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=human_text
            ),
        ]
        SmartLogger.log(
            "ERROR",
            "react.llm.reprint.request",
            category="react.llm.reprint.request",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "iteration": iteration,
                    "phase": "thinking",
                    "model": getattr(self.llm_handle.llm, "model_name", None)
                    or getattr(self.llm_handle.llm, "model", None),
                    "raw_llm_text": raw_llm_text,
                }
            ),
            max_inline_chars=0,
        )
        started = time.perf_counter()
        try:
            response = await self.llm_handle.llm.ainvoke(messages)
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.llm_xml_reprint.error",
                category="react.llm.repro.llm_xml_reprint",
                react_run_id=react_run_id,
                generator="llm_xml_reprint_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload={"raw_llm_text": raw_llm_text, "iteration": iteration},
                messages_payload={"system": system_prompt, "human": human_text},
                mode="json_text",
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        content = self._extract_text_only(getattr(response, "content", None))

        SmartLogger.log(
            "INFO",
            "react.llm.reprint.response",
            category="react.llm.reprint.response",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "iteration": iteration,
                    "phase": "thinking",
                    "elapsed_ms": elapsed_ms,
                    "assistant_response": content,
                }
            ),
            max_inline_chars=0,
        )
        log_llm_repro(
            level="INFO",
            message="react.llm.repro.llm_xml_reprint.ok",
            category="react.llm.repro.llm_xml_reprint",
            react_run_id=react_run_id,
            generator="llm_xml_reprint_generator",
            llm_provider=self.llm_handle.provider,
            llm_model=self.llm_handle.model,
            prompt=self.prompt_meta,
            input_payload={"raw_llm_text": raw_llm_text, "iteration": iteration},
            messages_payload={"system": system_prompt, "human": human_text},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=content,
            parsed={"xml": content},
        )
        return content


@lru_cache(maxsize=1)
def get_llm_xml_reprint_generator() -> LLMXmlReprintGenerator:
    """Singleton/cached generator instance."""
    return LLMXmlReprintGenerator()


