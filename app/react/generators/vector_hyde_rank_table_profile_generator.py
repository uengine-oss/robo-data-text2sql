from __future__ import annotations

import json
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


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            return "\n".join(lines[1:-1]).strip()
        return "\n".join(lines[1:]).strip()
    return s


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


def _try_parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = _strip_code_fences(text)
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


class VectorHydeRankTableProfileGenerator:
    """
    Strict table profile generator for vector_hyde_rank_test.py (keeps prompt identical to legacy inline prompt).
    Prompt is loaded from prompts/vector_hyde_rank_table_profile_prompt.md.
    """

    _PROMPT_FILE = "vector_hyde_rank_table_profile_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="vector-hyde-rank.table-profile",
            thinking_level=None,
            include_thoughts=False,
            temperature=0.0,
            max_output_tokens=650,
            use_light=True,
        )

    async def generate(
        self,
        *,
        table: Dict[str, Any],
        columns: List[Dict[str, Any]],
        sample_rows: List[Dict[str, Any]],
        react_run_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        payload = {
            "table": dict(table or {}),
            "columns": list(columns or []),
            "sample_rows": list(sample_rows or []),
        }
        human_text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
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
                message="react.llm.repro.vector_hyde_rank_table_profile.error",
                category="react.llm.repro.vector_hyde_rank",
                react_run_id=react_run_id,
                generator="vector_hyde_rank_table_profile_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
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
                "react.vector_hyde_rank.table_profile.llm_failed",
                category="react.vector_hyde_rank",
                params=sanitize_for_log(
                    {"react_run_id": react_run_id, "exception": repr(exc), "traceback": traceback.format_exc()}
                ),
                max_inline_chars=0,
            )
            return None, "llm_error"
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        text = _content_to_text(getattr(resp, "content", ""))
        obj = _try_parse_json_object(text)
        if not isinstance(obj, dict):
            log_llm_repro(
                level="WARNING",
                message="react.llm.repro.vector_hyde_rank_table_profile.parse_fail",
                category="react.llm.repro.vector_hyde_rank",
                react_run_id=react_run_id,
                generator="vector_hyde_rank_table_profile_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={"system": self.system_prompt, "human": human_text},
                mode="json_text",
                elapsed_ms=elapsed_ms,
                response_raw=text,
                parsed=None,
            )
            return None, "parse_fail"
        if "embedding_text" not in obj or not isinstance(obj.get("embedding_text"), str):
            log_llm_repro(
                level="WARNING",
                message="react.llm.repro.vector_hyde_rank_table_profile.missing_embedding_text",
                category="react.llm.repro.vector_hyde_rank",
                react_run_id=react_run_id,
                generator="vector_hyde_rank_table_profile_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={"system": self.system_prompt, "human": human_text},
                mode="json_text",
                elapsed_ms=elapsed_ms,
                response_raw=text,
                parsed=obj,
            )
            return None, "missing_embedding_text"

        SmartLogger.log(
            "INFO",
            "react.vector_hyde_rank.table_profile.llm",
            category="react.vector_hyde_rank",
            params=sanitize_for_log(
                {"react_run_id": react_run_id, "elapsed_ms": elapsed_ms, "embedding_text_len": len(obj.get("embedding_text") or "")}
            ),
            max_inline_chars=0,
        )
        log_llm_repro(
            level="INFO",
            message="react.llm.repro.vector_hyde_rank_table_profile.ok",
            category="react.llm.repro.vector_hyde_rank",
            react_run_id=react_run_id,
            generator="vector_hyde_rank_table_profile_generator",
            llm_provider=self.llm_handle.provider,
            llm_model=self.llm_handle.model,
            prompt=self.prompt_meta,
            input_payload=payload,
            messages_payload={"system": self.system_prompt, "human": human_text},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=text,
            parsed=obj,
        )
        return obj, "llm_ok"


@lru_cache(maxsize=1)
def get_vector_hyde_rank_table_profile_generator() -> VectorHydeRankTableProfileGenerator:
    return VectorHydeRankTableProfileGenerator()

