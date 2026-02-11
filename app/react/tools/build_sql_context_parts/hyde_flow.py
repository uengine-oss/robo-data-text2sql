from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from app.react.generators.hyde_schema_generator import (
    build_hyde_embedding_text,
    build_hyde_rerank_summary,
    get_hyde_schema_generator,
)
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger

from .text import _dedupe_keep_order, _truncate
from .xml_utils import emit_text


@dataclass(frozen=True)
class HydeFlowResult:
    used_fallback: bool
    hyde_embed_text: str
    hyde_rerank_text: str
    schema_embedding: List[float]
    keywords_tables: List[str]
    keywords_columns: List[str]
    all_keywords: List[str]


async def build_hyde_and_append_xml(
    *,
    embedder,
    question: str,
    react_run_id: str | None,
    fallback_terms: Sequence[str],
    result_parts: List[str],
) -> HydeFlowResult:
    """
    Run HyDE-Schema variants, build schema embedding & keywords, and append <hyde> blocks into tool XML.
    Returns structured HyDE artifacts needed for downstream retrieval/rerank.
    """
    q = (question or "").strip()

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.hyde_schema.start",
        category="react.tool.detail.build_sql_context",
        params={
            "react_run_id": react_run_id,
            "question_truncated": _truncate(q, 200),
        },
        max_inline_chars=0,
    )

    out = None
    mode = "not_run"
    try:
        out, mode = await get_hyde_schema_generator().generate(question=q, react_run_id=react_run_id)
    except Exception as exc:
        out = None
        mode = f"error:{type(exc).__name__}"
        SmartLogger.log(
            "WARNING",
            "react.build_sql_context.hyde_schema.unexpected_error",
            category="react.tool.detail.build_sql_context",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )

    used_fallback = not bool(out)

    hyde_embed_text = ""
    hyde_rerank_text = ""
    if out is not None:
        try:
            hyde_embed_text = str(build_hyde_embedding_text(out) or "").strip()
        except Exception:
            hyde_embed_text = ""
        try:
            hyde_rerank_text = str(build_hyde_rerank_summary(out) or "").strip()
        except Exception:
            hyde_rerank_text = ""

    schema_text_for_embed = (hyde_embed_text or q)[:8000]
    schema_embedding = await embedder.embed_text(schema_text_for_embed)

    kw_tables_all: List[str] = []
    kw_cols_all: List[str] = []
    if out is not None:
        try:
            kw_tables_all.extend(list(getattr(getattr(out, "search_keywords", None), "tables", []) or []))
        except Exception:
            pass
        try:
            kw_cols_all.extend(list(getattr(getattr(out, "search_keywords", None), "columns", []) or []))
        except Exception:
            pass

    keywords_tables = _dedupe_keep_order(list(kw_tables_all) or list(fallback_terms), limit=10)
    keywords_columns = _dedupe_keep_order(list(kw_cols_all) or list(fallback_terms), limit=10)
    all_keywords = _dedupe_keep_order(
        list(keywords_tables) + list(keywords_columns) + list(fallback_terms)[:10],
        limit=30,
    )

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.hyde_schema.done",
        category="react.tool.detail.build_sql_context",
        params=sanitize_for_log(
            {
                "react_run_id": react_run_id,
                "hyde_mode": str(mode),
                "used_fallback": used_fallback,
                "hyde_embedding_text_len": len(hyde_embed_text or ""),
                "hyde_rerank_summary_len": len(hyde_rerank_text or ""),
                "hyde_keywords_tables": keywords_tables,
                "hyde_keywords_columns": keywords_columns,
                "schema_embedding_dim": len(schema_embedding),
                "schema_embedding_sample": schema_embedding[:10],
            }
        ),
        max_inline_chars=0,
    )

    # XML: Compact HyDE block (keywords-only)
    result_parts.append("<hyde>")
    result_parts.append(emit_text("mode", "single"))
    result_parts.append(emit_text("used_fallback", str(used_fallback)))
    result_parts.append("<search_keywords>")
    result_parts.append("<tables>")
    for kw in keywords_tables:
        result_parts.append(emit_text("keyword", kw))
    result_parts.append("</tables>")
    result_parts.append("<columns>")
    for kw in keywords_columns:
        result_parts.append(emit_text("keyword", kw))
    result_parts.append("</columns>")
    result_parts.append("</search_keywords>")
    result_parts.append("</hyde>")

    return HydeFlowResult(
        used_fallback=bool(used_fallback),
        hyde_embed_text=str(hyde_embed_text or ""),
        hyde_rerank_text=str(hyde_rerank_text or ""),
        schema_embedding=schema_embedding,
        keywords_tables=keywords_tables,
        keywords_columns=keywords_columns,
        all_keywords=all_keywords,
    )


__all__ = ["HydeFlowResult", "build_hyde_and_append_xml"]


