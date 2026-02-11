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


class LightQuery(BaseModel):
    purpose: str = Field(..., description="Business-purpose for the query")
    sql: str = Field(..., description="SELECT-only SQL text (single statement)")


class LightQueriesOut(BaseModel):
    queries: List[LightQuery] = Field(default_factory=list)


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


def _try_parse_json_dict(text: Any) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    s = (text or "").strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _sanitize_queries(queries: Sequence[Dict[str, Any]], *, target_k: int) -> List[Dict[str, str]]:
    """
    Best-effort sanitize LLM output.
    - Keep only {purpose, sql}
    - Dedup by sql
    - Enforce target_k cap
    """
    k = max(1, int(target_k))
    out: List[Dict[str, str]] = []
    seen_sql = set()
    for q in list(queries or []):
        if len(out) >= k:
            break
        if not isinstance(q, dict):
            continue
        purpose = str(q.get("purpose") or "").strip()
        sql = str(q.get("sql") or "").strip()
        if not purpose or not sql:
            continue
        # Remove trailing semicolons defensively (policy: single statement)
        if sql.endswith(";"):
            sql = sql[:-1].rstrip()
        key = sql.strip()
        if not key:
            continue
        if key.lower() in seen_sql:
            continue
        seen_sql.add(key.lower())
        out.append({"purpose": purpose[:240], "sql": sql})
    return out[:k]


@dataclass(frozen=True)
class LightDisambiguationQuery:
    purpose: str
    sql: str


class LightDisambiguationQueriesGenerator:
    """
    One-shot generator that proposes lightweight disambiguation SQL queries.
    Prompt is loaded from prompts/light_disambiguation_queries_prompt.md.
    Responsibility: LLM call + output parsing only.
    """

    _PROMPT_FILE = "light_disambiguation_queries_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="light-disambiguation-queries",
            thinking_level="low",
            allow_context_cache=False,
            include_thoughts=False,
            temperature=0.0,
            use_light=False,
        )

    async def generate(
        self,
        *,
        user_question: str,
        target_k: int,
        schema_candidates: Dict[str, Any],
        fk_relationships: Sequence[Dict[str, Any]],
        resolved_values: Sequence[Dict[str, Any]],
        similar_queries: Sequence[Dict[str, Any]],
        react_run_id: Optional[str],
    ) -> Tuple[List[LightDisambiguationQuery], str]:
        q = (user_question or "").strip()
        k = max(1, int(target_k))
        if not q:
            return [], "empty_question"

        payload: Dict[str, Any] = {
            "user_question": q,
            "target_k": int(k),
            "schema_candidates": schema_candidates or {},
            "fk_relationships": list(fk_relationships or [])[:50],
            "resolved_values": list(resolved_values or [])[:30],
            # Similar query context:
            # - Keep SQL for structural inspiration.
            # - Also include best_context features when available (low-token, stable),
            #   so the model can generate more targeted disambiguation queries.
            "similar_queries": [
                {
                    "similarity_score": sq.get("similarity_score"),
                    "original_question": sq.get("question"),
                    "sql": _truncate(str(sq.get("sql") or ""), 1200),
                    # Optional enriched fields (may be missing on older nodes)
                    "intent_text": _truncate(str(sq.get("intent_text") or ""), 240),
                    "best_context_score": sq.get("best_context_score"),
                    "steps_features": _try_parse_json_dict(sq.get("best_context_steps_features")),
                    "steps_summary": _truncate(str(sq.get("best_context_steps_summary") or ""), 1200),
                }
                for sq in list(similar_queries or [])[:3]
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
                    structured_llm = llm.with_structured_output(LightQueriesOut)  # type: ignore[attr-defined]
                    resp = await structured_llm.ainvoke(messages)
                    mode = "structured"
                else:
                    resp = await llm.ainvoke(messages)
            except Exception:
                resp = await llm.ainvoke(messages)
                mode = "json_text"

            elapsed_ms = (time.perf_counter() - started) * 1000.0

            out: Optional[LightQueriesOut] = None
            if isinstance(resp, LightQueriesOut):
                out = resp
            elif isinstance(resp, dict):
                try:
                    out = LightQueriesOut.model_validate(resp)
                except Exception:
                    out = None

            if out is None:
                content = getattr(resp, "content", None)
                text = content if isinstance(content, str) else str(content or "")
                obj = _try_parse_json_object(text)
                if isinstance(obj, dict):
                    try:
                        out = LightQueriesOut.model_validate(obj)
                    except Exception:
                        out = None

            raw_queries: List[Dict[str, Any]] = []
            if out is not None:
                raw_queries = [q.model_dump() for q in list(out.queries or [])]

            cleaned = _sanitize_queries(raw_queries, target_k=k)
            results = [LightDisambiguationQuery(purpose=x["purpose"], sql=x["sql"]) for x in cleaned]

            SmartLogger.log(
                "INFO",
                "react.build_sql_context.light_queries.llm",
                category="react.tool.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "elapsed_ms": elapsed_ms,
                        "mode": mode,
                        "target_k": int(k),
                        "returned_k": int(len(results)),
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
                message="react.llm.repro.light_disambiguation_queries.ok",
                category="react.llm.repro.light_disambiguation_queries",
                react_run_id=react_run_id,
                generator="light_disambiguation_queries_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={"system": self.system_prompt, "human": human_text},
                mode=mode,
                elapsed_ms=elapsed_ms,
                response_raw=raw_content,
                parsed={
                    "raw_queries": raw_queries,
                    "cleaned_queries": cleaned,
                    "returned": [q.__dict__ for q in results],
                },
                extra={"target_k": int(k), "returned_k": int(len(results))},
            )
            return results, ("llm_ok_structured" if mode == "structured" else "llm_ok_json")
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.light_disambiguation_queries.error",
                category="react.llm.repro.light_disambiguation_queries",
                react_run_id=react_run_id,
                generator="light_disambiguation_queries_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={"system": self.system_prompt, "human": json.dumps(payload, ensure_ascii=False)},
                mode=None,
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            SmartLogger.log(
                "WARNING",
                "react.build_sql_context.light_queries.llm_failed",
                category="react.tool.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
            )
            return [], "llm_error"


@lru_cache(maxsize=1)
def get_light_disambiguation_queries_generator() -> LightDisambiguationQueriesGenerator:
    return LightDisambiguationQueriesGenerator()


__all__ = [
    "LightDisambiguationQuery",
    "LightDisambiguationQueriesGenerator",
    "get_light_disambiguation_queries_generator",
]


