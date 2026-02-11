from __future__ import annotations

import json
import time
import traceback
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.llm_factory import ReactLLMHandle, create_react_llm
from app.react.prompts import get_prompt_text
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


class HydeEntities(BaseModel):
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class HydeMeasurement(BaseModel):
    aggregation: str = ""
    metric_meaning: str = ""
    storage_type_hint: str = ""


class HydeJoinFilterHints(BaseModel):
    join_keys: list[str] = Field(default_factory=list)
    filter_column_meanings: list[str] = Field(default_factory=list)
    needs_time_range: Optional[bool] = None


class HydeSearchKeywords(BaseModel):
    tables: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)


class HydeSchemaOut(BaseModel):
    """
    Structured HyDE output used for schema grounding.
    NOTE: Keep this small and stable; downstream will build deterministic texts for embeddings and rerank summary.
    """

    intent: str = ""
    entities: HydeEntities = Field(default_factory=HydeEntities)
    measurement: HydeMeasurement = Field(default_factory=HydeMeasurement)
    schema_roles: list[str] = Field(default_factory=list)
    join_filter_hints: HydeJoinFilterHints = Field(default_factory=HydeJoinFilterHints)
    search_keywords: HydeSearchKeywords = Field(default_factory=HydeSearchKeywords)


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


def _try_parse_hyde_json(text: str) -> Optional[HydeSchemaOut]:
    """
    Best-effort JSON parser for HyDE responses (fallback when structured output isn't available).
    Accepts extra leading/trailing text as long as a JSON object is extractable.
    """
    s = _strip_code_fences(text)
    if not s:
        return None
    obj: Any
    try:
        obj = json.loads(s)
    except Exception:
        left = s.find("{")
        right = s.rfind("}")
        if left >= 0 and right > left:
            try:
                obj = json.loads(s[left : right + 1])
            except Exception:
                return None
        else:
            return None
    if not isinstance(obj, dict):
        return None
    try:
        return HydeSchemaOut.model_validate(obj)
    except Exception:
        return None


def _sanitize_hyde_structured(out: HydeSchemaOut) -> HydeSchemaOut:
    """
    Normalize whitespace and cap list sizes.
    """
    def _norm_list(xs: list[str], *, limit: int) -> list[str]:
        uniq: list[str] = []
        seen = set()
        for x in xs or []:
            s = str(x or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(s)
            if len(uniq) >= limit:
                break
        return uniq

    out.intent = (out.intent or "").strip()
    out.schema_roles = _norm_list(out.schema_roles, limit=10)
    out.entities.include = _norm_list(out.entities.include, limit=15)
    out.entities.exclude = _norm_list(out.entities.exclude, limit=15)
    out.join_filter_hints.join_keys = _norm_list(out.join_filter_hints.join_keys, limit=10)
    out.join_filter_hints.filter_column_meanings = _norm_list(
        out.join_filter_hints.filter_column_meanings, limit=10
    )
    out.search_keywords.tables = _norm_list(out.search_keywords.tables, limit=10)
    out.search_keywords.columns = _norm_list(out.search_keywords.columns, limit=10)

    out.measurement.aggregation = (out.measurement.aggregation or "").strip()[:40]
    out.measurement.metric_meaning = (out.measurement.metric_meaning or "").strip()[:200]
    out.measurement.storage_type_hint = (out.measurement.storage_type_hint or "").strip()[:200]

    return out


def build_hyde_embedding_text(out: HydeSchemaOut, *, max_chars: int = 8000) -> str:
    """
    Build deterministic Korean HyDE embedding text for stable vector retrieval.
    Design:
    - Keep labels minimal (reduce "section header" noise)
    - Keep high-signal keywords early
    - Skip empty parts
    - Do NOT include the original user question
    """

    def _join_list(xs: list[str], *, limit: int) -> str:
        picked: list[str] = []
        seen = set()
        for x in xs or []:
            s = str(x or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(s)
            if len(picked) >= int(limit):
                break
        return ", ".join(picked)

    def _bool_kr(v: Optional[bool]) -> str:
        if v is True:
            return "예"
        if v is False:
            return "아니오"
        return ""

    parts: list[str] = []

    # Put keywords/entities first for both retrieval and 1200-char rerank truncation friendliness.
    kw_tables = _join_list(out.search_keywords.tables, limit=10)
    kw_cols = _join_list(out.search_keywords.columns, limit=10)
    if kw_tables:
        parts.append(kw_tables)
    if kw_cols:
        parts.append(kw_cols)

    ent_incl = _join_list(out.entities.include, limit=15)
    ent_excl = _join_list(out.entities.exclude, limit=15)
    if ent_incl:
        parts.append(ent_incl)
    if ent_excl:
        parts.append(ent_excl)

    # Intent/measurement add meaning but can be shorter than keyword lists.
    intent = (out.intent or "").strip()
    if intent:
        parts.append(intent)

    agg = (out.measurement.aggregation or "").strip()
    meaning = (out.measurement.metric_meaning or "").strip()
    stype = (out.measurement.storage_type_hint or "").strip()
    if agg or meaning or stype:
        if agg:
            parts.append(agg)
        if meaning:
            parts.append(meaning)
        if stype:
            parts.append(stype)

    if out.schema_roles:
        roles = _join_list(out.schema_roles, limit=10)
        if roles:
            parts.append(roles)

    join_keys = _join_list(out.join_filter_hints.join_keys, limit=10)
    filters = _join_list(out.join_filter_hints.filter_column_meanings, limit=10)
    needs_time = _bool_kr(out.join_filter_hints.needs_time_range)
    if join_keys or filters or needs_time:
        if join_keys:
            parts.append(join_keys)
        if filters:
            parts.append(filters)
        if needs_time:
            parts.append(needs_time)

    text = "\n".join([p for p in (s.strip() for s in parts) if p]).strip()
    if not text:
        return ""
    if len(text) > int(max_chars):
        text = text[: int(max_chars)].rstrip()
    return text


def build_hyde_rerank_summary(out: HydeSchemaOut, *, max_chars: int = 8000) -> str:
    """
    Build deterministic Korean HyDE summary for table reranking & debugging.
    - Korean section headers (human-readable)
    - Skip empty sections
    - Put high-signal keywords early (1200-char truncation safety)
    - Do NOT include the original user question
    """

    def _join_list(xs: list[str], *, limit: int) -> str:
        picked: list[str] = []
        seen = set()
        for x in xs or []:
            s = str(x or "").strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(s)
            if len(picked) >= int(limit):
                break
        return ", ".join(picked)

    def _bool_kr(v: Optional[bool]) -> str:
        if v is True:
            return "예"
        if v is False:
            return "아니오"
        return ""

    lines: list[str] = []

    # 1) SEARCH KEYWORDS (most important for rerank)
    kw_tables = _join_list(out.search_keywords.tables, limit=10)
    kw_cols = _join_list(out.search_keywords.columns, limit=10)
    if kw_tables or kw_cols:
        lines.append("[검색 키워드]")
        if kw_tables:
            lines.append(f"- 테이블 검색 키워드: {kw_tables}")
        if kw_cols:
            lines.append(f"- 컬럼 검색 키워드: {kw_cols}")
        lines.append("")

    # 2) ENTITIES
    ent_incl = _join_list(out.entities.include, limit=15)
    ent_excl = _join_list(out.entities.exclude, limit=15)
    if ent_incl or ent_excl:
        lines.append("[엔터티]")
        if ent_incl:
            lines.append(f"- 포함 키워드: {ent_incl}")
        if ent_excl:
            lines.append(f"- 제외 키워드: {ent_excl}")
        lines.append("")

    # 3) INTENT
    intent = (out.intent or "").strip()
    if intent:
        lines.append("[의도]")
        lines.append(intent)
        lines.append("")

    # 4) MEASUREMENT
    agg = (out.measurement.aggregation or "").strip()
    meaning = (out.measurement.metric_meaning or "").strip()
    stype = (out.measurement.storage_type_hint or "").strip()
    if agg or meaning or stype:
        lines.append("[측정값]")
        if agg:
            lines.append(f"- 집계: {agg}")
        if meaning:
            lines.append(f"- 측정값 의미: {meaning}")
        if stype:
            lines.append(f"- 저장 형태 후보: {stype}")
        lines.append("")

    # 5) SCHEMA ROLES
    if out.schema_roles:
        lines.append("[스키마 역할]")
        for r in out.schema_roles[:10]:
            s = str(r or "").strip()
            if s:
                lines.append(f"- {s}")
        lines.append("")

    # 6) JOIN & FILTER HINTS
    join_keys = _join_list(out.join_filter_hints.join_keys, limit=10)
    filters = _join_list(out.join_filter_hints.filter_column_meanings, limit=10)
    needs_time = _bool_kr(out.join_filter_hints.needs_time_range)
    if join_keys or filters or needs_time:
        lines.append("[조인 & 필터 힌트]")
        if join_keys:
            lines.append(f"- 조인키 후보: {join_keys}")
        if filters:
            lines.append(f"- 필터 대상 컬럼 의미 후보: {filters}")
        if needs_time:
            lines.append(f"- 시간 범위 필요 여부: {needs_time}")
        lines.append("")

    while lines and not (lines[-1] or "").strip():
        lines.pop()

    text = "\n".join(lines).strip()
    if not text:
        return ""
    if len(text) > int(max_chars):
        text = text[: int(max_chars)].rstrip()
    return text


class HydeSchemaGenerator:
    """
    One-shot HyDE-style schema hint generator.
    Prompt is loaded from prompts/hyde_schema_prompt.md.
    """

    _PROMPT_FILE = "hyde_schema_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="hyde-schema",
            thinking_level=None,
            allow_context_cache=False,
            include_thoughts=False,
            max_output_tokens=600,
            use_light=True,
        )

    async def generate(
        self,
        *,
        question: str,
        react_run_id: Optional[str],
    ) -> Tuple[Optional[HydeSchemaOut], str]:
        """
        Returns (hyde_structured, debug_note).
        On failure returns (None, 'llm_error'|'llm_empty').
        """
        q = (question or "").strip()
        if not q:
            return None, "empty_question"

        try:
            started = time.perf_counter()
            llm = self.llm_handle.llm
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=q)]
            mode = "json_text"
            resp: Any
            # Prefer structured output when available.
            try:
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(HydeSchemaOut)  # type: ignore[attr-defined]
                    resp = await structured_llm.ainvoke(messages)
                    mode = "structured"
                else:
                    resp = await llm.ainvoke(messages)
            except Exception:
                resp = await llm.ainvoke(messages)
                mode = "json_text"
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            out: Optional[HydeSchemaOut] = None
            if isinstance(resp, HydeSchemaOut):
                out = resp
            elif isinstance(resp, dict):
                try:
                    out = HydeSchemaOut.model_validate(resp)
                except Exception:
                    out = None

            if out is None:
                content = getattr(resp, "content", None)
                text = content if isinstance(content, str) else str(content or "")
                out = _try_parse_hyde_json(text)

            if out is None:
                return None, "llm_empty"
            out = _sanitize_hyde_structured(out)
            embed_text = build_hyde_embedding_text(out)
            if not embed_text or len(embed_text) < 20:
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
                    message="react.llm.repro.hyde_schema.llm_empty",
                    category="react.llm.repro.hyde_schema",
                    react_run_id=react_run_id,
                    generator="hyde_schema_generator",
                    llm_provider=self.llm_handle.provider,
                    llm_model=self.llm_handle.model,
                    prompt=self.prompt_meta,
                    input_payload={"question": q},
                    messages_payload={"system": self.system_prompt, "human": q},
                    mode=mode,
                    elapsed_ms=elapsed_ms,
                    response_raw=raw_content,
                    parsed={"structured": (out.model_dump() if hasattr(out, "model_dump") else str(out)), "embed_text": embed_text},
                )
                return None, "llm_empty"

            SmartLogger.log(
                "INFO",
                "react.build_sql_context.hyde_schema.llm",
                category="react.tool.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "elapsed_ms": elapsed_ms,
                        "hyde_len": len(embed_text or ""),
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
                level="INFO",
                message="react.llm.repro.hyde_schema.ok",
                category="react.llm.repro.hyde_schema",
                react_run_id=react_run_id,
                generator="hyde_schema_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload={"question": q},
                messages_payload={"system": self.system_prompt, "human": q},
                mode=mode,
                elapsed_ms=elapsed_ms,
                response_raw=raw_content,
                parsed={
                    "structured": (out.model_dump() if hasattr(out, "model_dump") else str(out)),
                    "hyde_embedding_text": embed_text,
                    "hyde_rerank_summary": build_hyde_rerank_summary(out),
                },
                extra={"hyde_len": len(embed_text or "")},
            )
            return out, ("llm_ok_structured" if mode == "structured" else "llm_ok_json")
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.hyde_schema.error",
                category="react.llm.repro.hyde_schema",
                react_run_id=react_run_id,
                generator="hyde_schema_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload={"question": (question or "").strip()},
                messages_payload={"system": self.system_prompt, "human": (question or "").strip()},
                mode=None,
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            SmartLogger.log(
                "WARNING",
                "react.build_sql_context.hyde_schema.llm_failed",
                category="react.tool.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
            )
            return None, "llm_error"


@lru_cache(maxsize=1)
def get_hyde_schema_generator() -> HydeSchemaGenerator:
    """Singleton/cached generator instance."""
    return HydeSchemaGenerator()