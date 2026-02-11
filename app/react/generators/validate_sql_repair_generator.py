from __future__ import annotations

import json
import time
import traceback
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

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
        parts = []
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


class ValidateSqlRepairGenerator:
    """
    One-shot SQL repair generator for validate_sql failures.
    Uses the configured light LLM (flash-lite) for speed/cost.
    """

    _PROMPT_FILE = "validate_sql_repair_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="validate_sql_repair",
            # Some light Gemini models reject thinking_level; pass None to omit it.
            thinking_level=None,
            include_thoughts=False,
            temperature=0.0,
            max_output_tokens=700,
            use_light=True,
        )

    async def generate(
        self,
        *,
        db_type: str,
        error_text: str,
        current_sql: str,
        react_run_id: Optional[str] = None,
    ) -> Tuple[Optional[str], str]:
        cur = (current_sql or "").strip()
        if not cur:
            return None, "empty_sql"

        user = (
            f"DB Type:\n{(db_type or '').strip()}\n\n"
            f"Error:\n{(error_text or '').strip()}\n\n"
            "Current SQL:\n"
            f"{cur}\n\n"
            'Return {"sql":"SELECT ..."}'
        )

        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user)]
        started = time.perf_counter()
        try:
            resp = await self.llm_handle.llm.ainvoke(messages)
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.validate_sql_repair.error",
                category="react.llm.repro.validate_sql_repair",
                react_run_id=react_run_id,
                generator="validate_sql_repair_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload={
                    "db_type": (db_type or "").strip(),
                    "error_text": (error_text or "").strip(),
                    "current_sql": cur,
                },
                messages_payload={"system": self.system_prompt, "human": user},
                mode="json_text",
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            SmartLogger.log(
                "WARNING",
                "react.validate_sql_repair.llm_failed",
                category="react.validate_sql_repair",
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
            "react.validate_sql_repair.llm",
            category="react.validate_sql_repair",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "elapsed_ms": elapsed_ms,
                    "ok": bool(sql),
                }
            ),
            max_inline_chars=0,
        )
        log_llm_repro(
            level=("INFO" if sql else "WARNING"),
            message=("react.llm.repro.validate_sql_repair.ok" if sql else "react.llm.repro.validate_sql_repair.llm_empty"),
            category="react.llm.repro.validate_sql_repair",
            react_run_id=react_run_id,
            generator="validate_sql_repair_generator",
            llm_provider=self.llm_handle.provider,
            llm_model=self.llm_handle.model,
            prompt=self.prompt_meta,
            input_payload={
                "db_type": (db_type or "").strip(),
                "error_text": (error_text or "").strip(),
                "current_sql": cur,
            },
            messages_payload={"system": self.system_prompt, "human": user},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=text,
            parsed={"sql": sql, "parsed_obj": obj},
        )
        return (sql if sql else None), ("llm_ok" if sql else "llm_empty")


@lru_cache(maxsize=1)
def get_validate_sql_repair_generator() -> ValidateSqlRepairGenerator:
    return ValidateSqlRepairGenerator()


__all__ = ["ValidateSqlRepairGenerator", "get_validate_sql_repair_generator"]

