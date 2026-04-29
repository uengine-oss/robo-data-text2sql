from __future__ import annotations

import json
import os
import time
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.generators.passthrough_dialect_prompt import render_passthrough_dialect_prompt
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
        self.default_thinking_level: Optional[str] = self._read_default_thinking_level()
        self.default_max_output_tokens: int = self._read_default_max_output_tokens()
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

    @staticmethod
    def _read_default_thinking_level() -> Optional[str]:
        """
        Repair prompts are long and require direct JSON content. For OpenRouter-hosted
        Gemma, omitting reasoning is more stable than requesting low reasoning.
        """
        raw = os.environ.get("REACT_CONTROLLER_REPAIR_THINKING_LEVEL")
        if raw is None:
            return None
        level = str(raw).strip()
        if not level or level.lower() in {"none", "off", "false", "0"}:
            return None
        return level

    @staticmethod
    def _read_default_max_output_tokens() -> int:
        raw = os.environ.get("REACT_CONTROLLER_REPAIR_MAX_OUTPUT_TOKENS")
        if raw is not None:
            try:
                return max(700, int(str(raw).strip()))
            except Exception:
                pass
        return 1800

    def _get_llm_handle(self, temperature: float) -> ReactLLMHandle:
        t = float(temperature)
        thinking_key = str(self.default_thinking_level or "none")
        key = f"{t:.2f}:{thinking_key}:{int(self.default_max_output_tokens)}"
        h = self._handles.get(key)
        if h is not None:
            return h
        h = create_react_llm(
            purpose="controller_repair",
            thinking_level=self.default_thinking_level,
            include_thoughts=False,
            temperature=t,
            max_output_tokens=int(self.default_max_output_tokens),
        )
        self._handles[key] = h
        return h

    async def generate(
        self,
        *,
        question: str,
        generation_mode: Optional[str] = None,
        inner_dbms: Optional[str] = None,
        datasource: Optional[str] = None,
        # Backward compatible: legacy string hints
        missing_requirements: Optional[List[str]] = None,
        # Preferred: structured rubric feedback + validate_sql hints
        failed_checks: Optional[List[Dict[str, Any]]] = None,
        passed_must_ids: Optional[List[str]] = None,
        suggested_fixes: Optional[List[str]] = None,
        auto_rewrite: Optional[Dict[str, Any]] = None,
        context_xml: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        structured_generation_guidance: Optional[Dict[str, Any]] = None,
        repair_context: Optional[Dict[str, Any]] = None,
        current_sql: str,
        temperature: Optional[float] = None,
        react_run_id: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        sql, mode, _meta = await self.generate_with_plan(
            question=question,
            generation_mode=generation_mode,
            inner_dbms=inner_dbms,
            datasource=datasource,
            missing_requirements=missing_requirements,
            failed_checks=failed_checks,
            passed_must_ids=passed_must_ids,
            suggested_fixes=suggested_fixes,
            auto_rewrite=auto_rewrite,
            context_xml=context_xml,
            conversation_context=conversation_context,
            structured_generation_guidance=structured_generation_guidance,
            repair_context=repair_context,
            current_sql=current_sql,
            temperature=temperature,
            react_run_id=react_run_id,
        )
        return sql, mode

    async def generate_with_plan(
        self,
        *,
        question: str,
        generation_mode: Optional[str] = None,
        inner_dbms: Optional[str] = None,
        datasource: Optional[str] = None,
        # Backward compatible: legacy string hints
        missing_requirements: Optional[List[str]] = None,
        # Preferred: structured rubric feedback + validate_sql hints
        failed_checks: Optional[List[Dict[str, Any]]] = None,
        passed_must_ids: Optional[List[str]] = None,
        suggested_fixes: Optional[List[str]] = None,
        auto_rewrite: Optional[Dict[str, Any]] = None,
        context_xml: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        structured_generation_guidance: Optional[Dict[str, Any]] = None,
        repair_context: Optional[Dict[str, Any]] = None,
        current_sql: str,
        temperature: Optional[float] = None,
        react_run_id: Optional[str] = None,
    ) -> Tuple[Optional[str], str, Dict[str, Any]]:
        q = (question or "").strip()
        if not q:
            return None, "empty_question", {}
        cur = (current_sql or "").strip()
        if not cur:
            return None, "empty_sql", {}
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
        if generation_mode:
            payload["generation_mode"] = str(generation_mode)
        if inner_dbms:
            payload["inner_dbms"] = str(inner_dbms)
        if datasource:
            payload["datasource"] = str(datasource)
        if isinstance(conversation_context, dict) and conversation_context:
            payload["conversation_context"] = conversation_context
        if isinstance(structured_generation_guidance, dict) and structured_generation_guidance:
            payload["structured_generation_guidance"] = structured_generation_guidance
        if isinstance(repair_context, dict) and repair_context:
            payload["repair_context"] = repair_context
        human_text = json.dumps(payload, ensure_ascii=False)
        system_prompt = render_passthrough_dialect_prompt(
            self.system_prompt,
            generation_mode=generation_mode,
            inner_dbms=inner_dbms,
        )
        prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=system_prompt)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_text)]
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
                prompt=prompt_meta,
                input_payload=payload,
                messages_payload={"system": system_prompt, "human": human_text},
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
            return None, "llm_error", {}
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        text = _content_to_text(getattr(resp, "content", ""))
        response_meta: Dict[str, Any] = {}
        try:
            response_meta = {
                "response_metadata": getattr(resp, "response_metadata", None),
                "usage_metadata": getattr(resp, "usage_metadata", None),
                "additional_kwargs": getattr(resp, "additional_kwargs", None),
            }
        except Exception:
            response_meta = {}
        obj = _extract_first_json_object(text) or {}
        sql = str(obj.get("sql") or "").strip()
        plan_raw = obj.get("repair_plan")
        repair_plan: List[str] = []
        if isinstance(plan_raw, list):
            repair_plan = [str(x or "").strip()[:300] for x in plan_raw if str(x or "").strip()][:8]
        elif isinstance(plan_raw, str) and plan_raw.strip():
            repair_plan = [plan_raw.strip()[:300]]
        issue_choice = str(obj.get("issue_choice") or "").strip()[:80]
        regenerate_hint = str(obj.get("regenerate_hint") or "").strip()[:800]
        meta: Dict[str, Any] = {
            "issue_choice": issue_choice,
            "repair_plan": repair_plan,
            "regenerate_hint": regenerate_hint,
        }

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
                    "has_repair_plan": bool(repair_plan),
                    "has_regenerate_hint": bool(regenerate_hint),
                    "temperature": float(t),
                    "thinking_level": self.default_thinking_level or "",
                    "max_output_tokens": int(self.default_max_output_tokens),
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
            prompt=prompt_meta,
            input_payload=payload,
            messages_payload={"system": system_prompt, "human": human_text},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=text,
            parsed={"sql": sql, "repair_meta": meta, "parsed_obj": obj},
            extra=response_meta,
        )
        if sql:
            return sql, "llm_ok", meta
        if repair_plan or regenerate_hint:
            return None, "llm_plan_only", meta
        return None, "llm_empty", meta


@lru_cache(maxsize=1)
def get_controller_repair_generator() -> ControllerRepairGenerator:
    return ControllerRepairGenerator()

