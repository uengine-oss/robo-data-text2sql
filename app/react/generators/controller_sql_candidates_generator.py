from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
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


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            inner = "\n".join(lines[1:-1])
        else:
            inner = "\n".join(lines[1:])
        return inner.strip()
    return s


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = _strip_code_fences(str(text or "")).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        left = s.find("{")
        right = s.rfind("}")
        if left >= 0 and right > left:
            try:
                obj = json.loads(s[left : right + 1])
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
        return None


@dataclass(frozen=True)
class SqlCandidate:
    sql: str


class ControllerSqlCandidatesGenerator:
    """
    One-shot SQL candidate generator used by the controller.
    Prompt is loaded from prompts/controller_sql_candidates_prompt.md.
    """

    _PROMPT_FILE = "controller_sql_candidates_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self._handles: Dict[str, ReactLLMHandle] = {}
        self.default_temperature: float = self._read_default_temperature()
        self.llm_handle: ReactLLMHandle = self._get_llm_handle(self.default_temperature)

    @staticmethod
    def _read_default_temperature() -> float:
        """
        Default candidate temperature:
        - REACT_CONTROLLER_SQL_CANDIDATE_TEMPERATURE (preferred)
        - REACT_CONTROLLER_SQL_TEMPERATURE (fallback)
        - 0.0 (default)
        """
        for key in ("REACT_CONTROLLER_SQL_CANDIDATE_TEMPERATURE", "REACT_CONTROLLER_SQL_TEMPERATURE"):
            raw = os.environ.get(key)
            if raw is None:
                continue
            try:
                return float(str(raw).strip())
            except Exception:
                continue
        return 0.0

    def _get_llm_handle(self, temperature: float) -> ReactLLMHandle:
        # Quantize key to avoid unbounded cache.
        t = float(temperature)
        key = f"{t:.2f}"
        h = self._handles.get(key)
        if h is not None:
            return h
        h = create_react_llm(
            purpose="controller_sql",
            thinking_level="low",
            include_thoughts=False,
            temperature=t,
        )
        self._handles[key] = h
        return h

    async def generate(
        self,
        *,
        question: str,
        dbms: str,
        max_sql_seconds: int,
        context_xml: str,
        conversation_context: Optional[Dict[str, Any]] = None,
        n_candidates: int,
        temperature: Optional[float] = None,
        diversity_hints: Optional[List[str]] = None,
        seed: Optional[int] = None,
        react_run_id: Optional[str] = None,
    ) -> Tuple[List[str], str]:
        q = (question or "").strip()
        if not q:
            return [], "empty_question"
        n = max(1, int(n_candidates))
        t = self.default_temperature if temperature is None else float(temperature)
        llm_handle = self._get_llm_handle(t)

        payload = {
            "question": q,
            "dbms": str(dbms or ""),
            "max_sql_seconds": int(max_sql_seconds),
            "n_candidates": int(n),
            "context_xml": str(context_xml or ""),
            "temperature": float(t),
        }
        if isinstance(conversation_context, dict) and conversation_context:
            # Keep payload bounded (avoid sending huge data by accident)
            # The caller should already cap/trim; we add a shallow safeguard here.
            payload["conversation_context"] = conversation_context
        if isinstance(diversity_hints, list):
            payload["diversity_hints"] = [str(x or "").strip()[:400] for x in diversity_hints if str(x or "").strip()][:12]
        if seed is not None:
            try:
                payload["seed"] = int(seed)
            except Exception:
                payload["seed"] = str(seed)[:40]
        human_text = json.dumps(payload, ensure_ascii=False)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_text),
        ]

        started = time.perf_counter()
        try:
            resp = await llm_handle.llm.ainvoke(messages)
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.controller_sql_candidates.error",
                category="react.llm.repro.controller_sql_candidates",
                react_run_id=react_run_id,
                generator="controller_sql_candidates_generator",
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
                "react.controller_sql_candidates.llm_failed",
                category="react.controller_sql_candidates",
                params=sanitize_for_log(
                    {"react_run_id": react_run_id, "exception": repr(exc), "traceback": traceback.format_exc()}
                ),
                max_inline_chars=0,
            )
            return [], "llm_error"
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        content = getattr(resp, "content", None)
        text = _content_to_text(content)
        obj = _extract_first_json_object(text) or {}

        cands_raw = obj.get("candidates")
        out: List[str] = []
        if isinstance(cands_raw, list):
            for c in cands_raw:
                if not isinstance(c, dict):
                    continue
                sql = str(c.get("sql") or "").strip()
                if sql:
                    out.append(sql)

        # Dedup keep order
        seen = set()
        uniq: List[str] = []
        for x in out:
            k = x.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            uniq.append(x)
            if len(uniq) >= n:
                break

        SmartLogger.log(
            "INFO",
            "react.controller_sql_candidates.llm",
            category="react.controller_sql_candidates",
            params=sanitize_for_log(
                {"react_run_id": react_run_id, "elapsed_ms": elapsed_ms, "returned": len(uniq), "requested": int(n)}
            ),
            max_inline_chars=0,
        )
        log_llm_repro(
            level=("INFO" if uniq else "WARNING"),
            message=("react.llm.repro.controller_sql_candidates.ok" if uniq else "react.llm.repro.controller_sql_candidates.llm_empty"),
            category="react.llm.repro.controller_sql_candidates",
            react_run_id=react_run_id,
            generator="controller_sql_candidates_generator",
            llm_provider=llm_handle.provider,
            llm_model=llm_handle.model,
            prompt=self.prompt_meta,
            input_payload=payload,
            messages_payload={"system": self.system_prompt, "human": human_text},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=text,
            parsed={"candidates": uniq, "parsed_obj": obj},
            extra={"requested_n": int(n), "returned_n": int(len(uniq))},
        )
        return uniq[:n], ("llm_ok" if uniq else "llm_empty")


@lru_cache(maxsize=1)
def get_controller_sql_candidates_generator() -> ControllerSqlCandidatesGenerator:
    return ControllerSqlCandidatesGenerator()

