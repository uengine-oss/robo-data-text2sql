from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class TableProfileInput:
    schema: str
    name: str
    description: str
    analyzed_description: str
    columns: List[Dict[str, Any]]
    sample_rows: List[Dict[str, Any]]


class TableProfileGenerator:
    """
    Generate a semantic "table profile" JSON used for embeddings (Table Embedding Text).
    Prompt is loaded from prompts/table_profile_prompt.md.
    """

    _PROMPT_FILE = "table_profile_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="text2sql-table-profile",
            thinking_level=None,
            include_thoughts=False,
            temperature=0.0,
            max_output_tokens=900,
            use_light=True,
        )

    async def generate(
        self,
        *,
        item: TableProfileInput,
        react_run_id: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        payload = {
            "table": {
                "schema": item.schema,
                "name": item.name,
                "description": item.description,
                "analyzed_description": item.analyzed_description,
            },
            "columns": [
                {
                    "name": str(c.get("name") or ""),
                    "dtype": str(c.get("dtype") or ""),
                    "description": str(c.get("description") or ""),
                }
                for c in (item.columns or [])
            ],
            "sample_rows": item.sample_rows or [],
        }
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ]

        started = time.perf_counter()
        try:
            resp = await self.llm_handle.llm.ainvoke(messages)
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.table_profile.error",
                category="react.llm.repro.table_profile",
                react_run_id=react_run_id,
                generator="table_profile_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={
                    "system": self.system_prompt,
                    "human": json.dumps(payload, ensure_ascii=False),
                },
                mode="json_text",
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            SmartLogger.log(
                "WARNING",
                "react.table_profile.llm_failed",
                category="react.table_profile",
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
            SmartLogger.log(
                "WARNING",
                "react.table_profile.parse_fail",
                category="react.table_profile",
                params=sanitize_for_log({"react_run_id": react_run_id}),
                max_inline_chars=0,
            )
            return None, "parse_fail"

        emb = obj.get("embedding_text")
        if not isinstance(emb, str) or not emb.strip():
            return None, "missing_embedding_text"

        SmartLogger.log(
            "INFO",
            "react.table_profile.llm",
            category="react.table_profile",
            params=sanitize_for_log(
                {"react_run_id": react_run_id, "elapsed_ms": elapsed_ms, "embedding_text_len": len(emb or "")}
            ),
            max_inline_chars=0,
        )
        log_llm_repro(
            level="INFO",
            message="react.llm.repro.table_profile.ok",
            category="react.llm.repro.table_profile",
            react_run_id=react_run_id,
            generator="table_profile_generator",
            llm_provider=self.llm_handle.provider,
            llm_model=self.llm_handle.model,
            prompt=self.prompt_meta,
            input_payload=payload,
            messages_payload={"system": self.system_prompt, "human": json.dumps(payload, ensure_ascii=False)},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=text,
            parsed=obj,
            extra={"embedding_text_len": len(emb or "")},
        )
        return obj, "llm_ok"


@lru_cache(maxsize=1)
def get_table_profile_generator() -> TableProfileGenerator:
    return TableProfileGenerator()

