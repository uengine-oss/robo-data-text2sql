from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.llm_factory import ReactLLMHandle, create_react_llm
from app.react.prompts import get_prompt_text
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


@dataclass(frozen=True)
class TableRerankCandidate:
    schema: str
    name: str
    description: str
    analyzed_description: str
    score: float

    @property
    def fqn(self) -> str:
        schema = (self.schema or "").strip()
        name = (self.name or "").strip()
        return f"{schema}.{name}" if schema else name


class TableRerankOut(BaseModel):
    selected: List[int] = Field(default_factory=list)


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


def _compact_ws(text: str) -> str:
    return " ".join(str(text or "").split())


def _truncate(text: str, max_len: int) -> str:
    t = str(text or "")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


class TableRerankGenerator:
    """
    One-shot table reranker.
    Prompt is loaded from prompts/table_rerank_prompt.md.
    Responsibility: LLM call + output parsing/validation only.
    """

    _PROMPT_FILE = "table_rerank_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="table-rerank",
            thinking_level="low",
            allow_context_cache=False,
            include_thoughts=False,
            temperature=0.0
        )

    async def generate(
        self,
        *,
        question: str,
        hyde_summary: str,
        candidates: Sequence[TableRerankCandidate],
        top_k: int,
        react_run_id: Optional[str],
    ) -> Tuple[Optional[List[int]], str]:
        """
        Returns (indexes, mode).
        On failure returns (None, 'llm_error'|'llm_empty'|'parse_fail'|...).
        """
        q = (question or "").strip()
        k = max(1, int(top_k))
        if not q:
            return None, "empty_question"
        if not candidates or len(candidates) < k:
            return None, "no_candidates"

        max_desc = 360
        max_analyzed = 520
        n = len(candidates)
        payload = {
            "user_question": q,
            "hyde_summary": _truncate(_compact_ws(hyde_summary or ""), 1200),
            "target_k": int(k),
            "candidates": [
                {
                    "index": i,
                    "table": c.fqn,
                    "description": _truncate(_compact_ws(c.description or ""), max_desc),
                    "analyzed_description": _truncate(
                        _compact_ws(c.analyzed_description or ""), max_analyzed
                    ),
                    "vector_score": float(c.score or 0.0),
                }
                for i, c in enumerate(candidates)
            ],
        }

        try:
            started = time.perf_counter()
            llm = self.llm_handle.llm
            human_text = json.dumps(payload, ensure_ascii=False)
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=human_text),
            ]

            mode = "json_text"
            resp: Any
            try:
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(TableRerankOut)  # type: ignore[attr-defined]
                    resp = await structured_llm.ainvoke(messages)
                    mode = "structured"
                else:
                    resp = await llm.ainvoke(messages)
            except Exception:
                resp = await llm.ainvoke(messages)
                mode = "json_text"

            elapsed_ms = (time.perf_counter() - started) * 1000.0

            out: Optional[TableRerankOut] = None
            if isinstance(resp, TableRerankOut):
                out = resp
            elif isinstance(resp, dict):
                try:
                    out = TableRerankOut.model_validate(resp)
                except Exception:
                    out = None

            if out is None:
                content = getattr(resp, "content", None)
                text = content if isinstance(content, str) else str(content or "")
                obj = _try_parse_json_object(text)
                if isinstance(obj, dict):
                    try:
                        out = TableRerankOut.model_validate(obj)
                    except Exception:
                        out = None

            if out is None or not isinstance(out.selected, list):
                # Deterministic fallback (keeps downstream from emitting "table_rerank_fallback_used").
                picked = list(range(min(k, n)))
                SmartLogger.log(
                    "WARNING",
                    "react.build_sql_context.table_rerank.parse_fail_fallback",
                    category="react.tool.build_sql_context",
                    params=sanitize_for_log(
                        {
                            "react_run_id": react_run_id,
                            "target_k": int(k),
                            "candidates_count": int(n),
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
                    level="WARNING",
                    message="react.llm.repro.table_rerank.parse_fail_fallback",
                    category="react.llm.repro.table_rerank",
                    react_run_id=react_run_id,
                    generator="table_rerank_generator",
                    llm_provider=self.llm_handle.provider,
                    llm_model=self.llm_handle.model,
                    prompt=self.prompt_meta,
                    input_payload=payload,
                    messages_payload={"system": self.system_prompt, "human": human_text},
                    mode=mode,
                    elapsed_ms=elapsed_ms,
                    response_raw=raw_content,
                    parsed={"picked": picked, "mode": mode, "reason": "parse_fail"},
                    extra={"target_k": int(k), "candidates_count": int(n)},
                )
                return picked, "deterministic_parse_fail"

            picked: List[int] = []
            seen = set()
            n = len(candidates)
            for v in out.selected:
                try:
                    idx = int(v)
                except Exception:
                    continue
                if idx < 0 or idx >= n:
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                picked.append(idx)
                if len(picked) >= k:
                    break

            if len(picked) < k:
                # Robustness: Some LLMs occasionally return fewer than target_k indexes even with strict prompts.
                # Instead of failing and forcing downstream fallback, pad deterministically using the original
                # candidate order (which is already vector-score sorted by the caller/searcher).
                for idx in range(n):
                    if idx in seen:
                        continue
                    seen.add(idx)
                    picked.append(idx)
                    if len(picked) >= k:
                        break

                if len(picked) < k:
                    # This should only happen if candidates < k (guarded above), but keep safety.
                    return None, "insufficient_indexes"

                SmartLogger.log(
                    "WARNING",
                    "react.build_sql_context.table_rerank.insufficient_indexes_padded",
                    category="react.tool.build_sql_context",
                    params=sanitize_for_log(
                        {
                            "react_run_id": react_run_id,
                            "target_k": int(k),
                            "returned_k": int(len(picked)),
                            "candidates_count": int(len(candidates)),
                        }
                    ),
                )
                log_llm_repro(
                    level="WARNING",
                    message="react.llm.repro.table_rerank.insufficient_indexes_padded",
                    category="react.llm.repro.table_rerank",
                    react_run_id=react_run_id,
                    generator="table_rerank_generator",
                    llm_provider=self.llm_handle.provider,
                    llm_model=self.llm_handle.model,
                    prompt=self.prompt_meta,
                    input_payload=payload,
                    messages_payload={"system": self.system_prompt, "human": human_text},
                    mode=mode,
                    elapsed_ms=elapsed_ms,
                    response_raw=getattr(resp, "content", None),
                    parsed={"picked": picked[:k], "returned": out.selected if out is not None else None},
                    extra={"target_k": int(k), "returned_k": int(len(picked[:k])), "candidates_count": int(n)},
                )
                return picked[:k], (
                    ("llm_ok_structured_padded" if mode == "structured" else "llm_ok_json_padded")
                )

            SmartLogger.log(
                "INFO",
                "react.build_sql_context.table_rerank.llm",
                category="react.tool.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "elapsed_ms": elapsed_ms,
                        "mode": mode,
                        "target_k": int(k),
                        "candidates_count": int(len(candidates)),
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
                message="react.llm.repro.table_rerank.ok",
                category="react.llm.repro.table_rerank",
                react_run_id=react_run_id,
                generator="table_rerank_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={"system": self.system_prompt, "human": human_text},
                mode=mode,
                elapsed_ms=elapsed_ms,
                response_raw=raw_content,
                parsed={"picked": picked[:k], "returned": out.selected if out is not None else None},
                extra={"target_k": int(k), "returned_k": int(len(picked[:k])), "candidates_count": int(n)},
            )
            return picked[:k], ("llm_ok_structured" if mode == "structured" else "llm_ok_json")
        except Exception as exc:
            SmartLogger.log(
                "WARNING",
                "react.build_sql_context.table_rerank.llm_failed",
                category="react.tool.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
            )
            # Deterministic fallback: return top-K by original candidate order.
            n = len(candidates)
            picked = list(range(min(k, n)))
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.table_rerank.error",
                category="react.llm.repro.table_rerank",
                react_run_id=react_run_id,
                generator="table_rerank_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={"system": self.system_prompt, "human": json.dumps(payload, ensure_ascii=False)},
                mode=None,
                elapsed_ms=None,
                response_raw=None,
                parsed={"picked": picked, "fallback": True},
                exception=exc,
            )
            return picked, "deterministic_llm_error"


@lru_cache(maxsize=1)
def get_table_rerank_generator() -> TableRerankGenerator:
    return TableRerankGenerator()


