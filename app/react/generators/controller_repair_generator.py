from __future__ import annotations

import json
import os
import time
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

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


class ControllerRepairGenerator:
    """
    One-shot SQL repair generator.
    Prompt is loaded from prompts/controller_repair_prompt.md.
    """

    _PROMPT_FILE = "controller_repair_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self._handles: Dict[str, ReactLLMHandle] = {}
        self.default_temperature: float = self._read_default_temperature()
        self.llm_handle: ReactLLMHandle = self._get_llm_handle(self.default_temperature)

    @staticmethod
    def _read_default_temperature() -> float:
        """
        Default repair temperature:
        - REACT_CONTROLLER_REPAIR_TEMPERATURE (preferred)
        - REACT_CONTROLLER_SQL_REPAIR_TEMPERATURE (fallback)
        - 0.0 (default)
        """
        for key in ("REACT_CONTROLLER_REPAIR_TEMPERATURE", "REACT_CONTROLLER_SQL_REPAIR_TEMPERATURE"):
            raw = os.environ.get(key)
            if raw is None:
                continue
            try:
                return float(str(raw).strip())
            except Exception:
                continue
        return 0.0

    def _get_llm_handle(self, temperature: float) -> ReactLLMHandle:
        t = float(temperature)
        key = f"{t:.2f}"
        h = self._handles.get(key)
        if h is not None:
            return h
        h = create_react_llm(
            purpose="controller_repair",
            thinking_level="low",
            include_thoughts=False,
            temperature=t,
            max_output_tokens=900,
        )
        self._handles[key] = h
        return h

    async def generate(
        self,
        *,
        question: str,
        # Backward compatible: legacy string hints
        missing_requirements: Optional[List[str]] = None,
        # Preferred: structured rubric feedback + validate_sql hints
        failed_checks: Optional[List[Dict[str, Any]]] = None,
        passed_must_ids: Optional[List[str]] = None,
        suggested_fixes: Optional[List[str]] = None,
        auto_rewrite: Optional[Dict[str, Any]] = None,
        context_xml: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        current_sql: str,
        temperature: Optional[float] = None,
        react_run_id: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        q = (question or "").strip()
        if not q:
            return None, "empty_question"
        cur = (current_sql or "").strip()
        if not cur:
            return None, "empty_sql"
        miss = [str(x or "").strip() for x in (missing_requirements or []) if str(x or "").strip()]
        t = self.default_temperature if temperature is None else float(temperature)
        llm_handle = self._get_llm_handle(t)

        payload: Dict[str, Any] = {
            "question": q,
            "current_sql": cur,
            "context_xml": (context_xml or "").strip(),
            "failed_checks": list(failed_checks or [])[:48],
            "passed_must_ids": [str(x or "").strip() for x in (passed_must_ids or []) if str(x or "").strip()][:48],
            "suggested_fixes": [str(x or "").strip()[:400] for x in (suggested_fixes or []) if str(x or "").strip()][:12],
            "auto_rewrite": dict(auto_rewrite or {}) if isinstance(auto_rewrite, dict) else {},
            "missing_requirements_legacy": miss[:24],
            "temperature": float(t),
        }
        if isinstance(conversation_context, dict) and conversation_context:
            payload["conversation_context"] = conversation_context
        human_text = json.dumps(payload, ensure_ascii=False)
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=human_text)]
        started = time.perf_counter()
        try:
            resp = await llm_handle.llm.ainvoke(messages)
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.controller_repair.error",
                category="react.llm.repro.controller_repair",
                react_run_id=react_run_id,
                generator="controller_repair_generator",
                llm_provider=llm_handle.provider,
                llm_model=llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={"system": self.system_prompt, "human": human_text},
                mode="json_text",
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            SmartLogger.log(
                "WARNING",
                "react.controller_repair.llm_failed",
                category="react.controller_repair",
                params=sanitize_for_log(
                    {"react_run_id": react_run_id, "exception": repr(exc), "traceback": traceback.format_exc()}
                ),
                max_inline_chars=0,
            )
            return None, "llm_error"
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        text = _content_to_text(getattr(resp, "content", ""))
        obj = _extract_first_json_object(text) or {}
        sql = str(obj.get("sql") or "").strip()

        SmartLogger.log(
            "INFO",
            "react.controller_repair.llm",
            category="react.controller_repair",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "elapsed_ms": elapsed_ms,
                    "ok": bool(sql),
                    "missing_n": len(miss),
                    "failed_checks_n": len(list(failed_checks or [])),
                    "temperature": float(t),
                }
            ),
            max_inline_chars=0,
        )
        log_llm_repro(
            level=("INFO" if sql else "WARNING"),
            message=("react.llm.repro.controller_repair.ok" if sql else "react.llm.repro.controller_repair.llm_empty"),
            category="react.llm.repro.controller_repair",
            react_run_id=react_run_id,
            generator="controller_repair_generator",
            llm_provider=llm_handle.provider,
            llm_model=llm_handle.model,
            prompt=self.prompt_meta,
            input_payload=payload,
            messages_payload={"system": self.system_prompt, "human": human_text},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=text,
            parsed={"sql": sql, "parsed_obj": obj},
        )
        return (sql if sql else None), ("llm_ok" if sql else "llm_empty")


@lru_cache(maxsize=1)
def get_controller_repair_generator() -> ControllerRepairGenerator:
    return ControllerRepairGenerator()

