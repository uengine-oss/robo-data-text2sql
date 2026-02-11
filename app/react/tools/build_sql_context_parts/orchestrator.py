"""
Orchestrator for build_sql_context tool.

This module keeps ONLY the end-to-end flow (`execute`) and delegates:
- HyDE generation + HyDE XML: `hyde_flow.py`
- Similar queries + value mappings XML: `similar_flow.py`
- Table retrieval/rerank: `table_search_flow.py`
- Column retrieval: `column_search_flow.py`
- Schema XML: `schema_xml.py`
- FK XML: `fk_flow.py`
- Resolved values XML: `resolved_values_flow.py`
- Suggestions XML: `suggestions_flow.py`
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Sequence

from app.core.llm_factory import create_embedding_client
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger

from .column_search_flow import search_columns_per_table
from .fk_flow import fetch_fk_relationships_and_append_xml
from .hyde_flow import build_hyde_and_append_xml
from .resolved_values_flow import resolve_values_and_append_xml
from .schema_xml import append_schema_candidates_xml
from .similar_flow import build_similar_and_append_xml
from .table_search_flow import search_tables_and_rerank
from .text import _regex_terms, _truncate
from .suggestions_flow import append_suggestions_xml
from .neo4j import (
    _neo4j_fetch_fk_relationships,
    _neo4j_fetch_table_embedding_texts_for_tables,
    _neo4j_fetch_table_schemas,
)
from .column_value_hints_flow import append_column_value_hints_xml
from .light_queries_flow import append_light_queries_and_results_xml


def _emit_pipeline_stage(
    context,
    *,
    stage: str,
    status: str,
    seq: int,
    elapsed_ms: Optional[float] = None,
    counts: Optional[Dict[str, int]] = None,
    notes: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> None:
    """
    Best-effort: emit progress event to the router's NDJSON stream.
    Must never raise.
    """
    try:
        ts_ms = int(time.time() * 1000)
        payload: Dict[str, Any] = {
            "event": "pipeline_stage",
            "pipeline": "build_sql_context",
            "stage": str(stage or ""),
            "status": str(status or ""),
            "seq": int(seq),
            "iteration": 0,
            "ts_ms": ts_ms,
        }
        if elapsed_ms is not None:
            payload["elapsed_ms"] = float(elapsed_ms)
        if counts:
            payload["counts"] = {str(k): int(v) for k, v in dict(counts).items() if isinstance(v, int)}
        if notes:
            payload["notes"] = [str(x) for x in list(notes) if str(x).strip()][:12]
        if error:
            payload["error"] = str(error)[:500]
        getattr(context, "emit")(payload)  # type: ignore[misc]
    except Exception:
        return


async def execute(context, question: str, *, exclude_light_sqls: Optional[Sequence[str]] = None) -> str:
    """
    Build SQL context as a single tool call.
    Returns <tool_result> XML.
    """
    started = time.perf_counter()
    result_parts: List[str] = ["<tool_result>", "<build_sql_context_result>"]

    react_run_id = getattr(context, "react_run_id", None)
    q = (question or "").strip()
    if not q:
        return "<tool_result><error>question parameter is required</error></tool_result>"

    SmartLogger.log(
        "INFO",
        "react.build_sql_context.start",
        category="react.tool.build_sql_context",
        params=sanitize_for_log({"react_run_id": react_run_id, "question": _truncate(q, 200)}),
    )

    fallback_terms = _regex_terms(q, limit=20)

    # 1) embedding (single)
    stage_seq = 1
    emb_started = time.perf_counter()
    _emit_pipeline_stage(context, stage="embedding", status="start", seq=stage_seq)
    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.embedding.start",
        category="react.tool.detail.build_sql_context",
        params={"question_length": len(q), "question_truncated": q[:8000]},
        max_inline_chars=0,
    )

    embedder = create_embedding_client()
    question_embedding = await embedder.embed_text(q[:8000])
    _emit_pipeline_stage(
        context,
        stage="embedding",
        status="done",
        seq=stage_seq,
        elapsed_ms=(time.perf_counter() - emb_started) * 1000.0,
        counts={"embedding_dim": int(len(question_embedding))},
    )

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.embedding.done",
        category="react.tool.detail.build_sql_context",
        params={
            "embedding_dim": len(question_embedding),
            "embedding_sample": question_embedding[:10],
        },
        max_inline_chars=0,
    )

    # 1.5) HyDE + 3) similar queries/value mappings
    # Run in parallel BUT keep XML output order stable by collecting parts separately.
    stage_seq += 1
    enrich_started = time.perf_counter()
    _emit_pipeline_stage(context, stage="enrich", status="start", seq=stage_seq)
    hyde_parts: List[str] = []
    sim_parts: List[str] = []
    hyde, sim = await asyncio.gather(
        build_hyde_and_append_xml(
            embedder=embedder,
            question=q,
            react_run_id=react_run_id,
            fallback_terms=fallback_terms,
            result_parts=hyde_parts,
        ),
        build_similar_and_append_xml(
            context=context,
            embedder=embedder,
            question=q,
            react_run_id=react_run_id,
            question_embedding=question_embedding,
            fallback_terms=fallback_terms,
            result_parts=sim_parts,
            min_similarity=0.3,
        ),
    )
    result_parts.extend(hyde_parts)
    result_parts.extend(sim_parts)
    _emit_pipeline_stage(
        context,
        stage="enrich",
        status="done",
        seq=stage_seq,
        elapsed_ms=(time.perf_counter() - enrich_started) * 1000.0,
        counts={
            "similar_queries": int(len(sim.similar_queries or [])),
            "value_mappings": int(len(sim.value_mappings or [])),
            "keywords": int(len(hyde.all_keywords or [])),
        },
    )

    # 4) table retrieval + rerank (may append warnings)
    stage_seq += 1
    table_started = time.perf_counter()
    _emit_pipeline_stage(context, stage="table_search", status="start", seq=stage_seq)
    table_res = await search_tables_and_rerank(
        context=context,
        embedder=embedder,
        question=q,
        react_run_id=react_run_id,
        question_embedding=question_embedding,
        schema_embedding=hyde.schema_embedding,
        hyde_embed_text=hyde.hyde_embed_text,
        hyde_rerank_text=hyde.hyde_rerank_text,
        fallback_terms=fallback_terms,
        intent_embedding=sim.intent_embedding,
        similar_queries=sim.similar_queries,
        result_parts=result_parts,
    )
    _emit_pipeline_stage(
        context,
        stage="table_search",
        status="done",
        seq=stage_seq,
        elapsed_ms=(time.perf_counter() - table_started) * 1000.0,
        counts={
            "table_candidates": int(len(table_res.table_candidates or [])),
            "selected_tables": int(len(table_res.selected_tables or [])),
        },
        notes=[f"rerank_mode={getattr(table_res, 'rerank_mode', '')}"],
    )

    selected_tables_for_context = table_res.selected_tables[: table_res.rerank_top_k]

    # 4.5) FK relationships (prefetch; used for join-column forcing in column selection)
    table_fqns = [
        f"{t.schema}.{t.name}".strip(".").lower()
        for t in selected_tables_for_context
        if (t.schema or "").strip() and (t.name or "").strip()
    ]
    fk_relationships_prefetch = await _neo4j_fetch_fk_relationships(
        context=context, table_fqns=table_fqns, limit=50
    )

    # 4.55) table schemas (prefetch once; reused for column scoring + enum value hints)
    table_schemas_prefetch = await _neo4j_fetch_table_schemas(
        context=context, tables=selected_tables_for_context
    )

    # 5) per-table column retrieval
    stage_seq += 1
    col_started = time.perf_counter()
    _emit_pipeline_stage(context, stage="column_search", status="start", seq=stage_seq)
    col_res = await search_columns_per_table(
        context=context,
        schema_embedding=hyde.schema_embedding,
        selected_tables=selected_tables_for_context,
        per_table_k=table_res.per_table_k,
        all_keywords=hyde.all_keywords,
        fk_relationships=fk_relationships_prefetch,
        similar_queries=sim.similar_queries,
        table_schemas=table_schemas_prefetch,
        result_parts=result_parts,
    )
    _emit_pipeline_stage(
        context,
        stage="column_search",
        status="done",
        seq=stage_seq,
        elapsed_ms=(time.perf_counter() - col_started) * 1000.0,
        counts={
            "selected_tables": int(len(col_res.selected_tables or [])),
            "column_candidates": int(len(col_res.column_candidates or [])),
        },
    )

    # 5.5) schema_candidates XML
    table_desc_overrides = await _neo4j_fetch_table_embedding_texts_for_tables(
        context=context,
        tables=list(table_res.table_candidates)[: table_res.rerank_top_k],
    )
    append_schema_candidates_xml(
        result_parts=result_parts,
        table_candidates=table_res.table_candidates,
        selected_tables=col_res.selected_tables,
        per_table_columns=col_res.per_table_columns,
        per_table_mode=col_res.per_table_mode,
        per_table_k=table_res.per_table_k,
        rerank_top_k=table_res.rerank_top_k,
        table_description_overrides=table_desc_overrides,
    )

    # 5.55) column_value_hints XML (cached enum values; fast at runtime)
    await append_column_value_hints_xml(
        context=context,
        selected_tables=col_res.selected_tables,
        per_table_columns=col_res.per_table_columns,
        table_schemas=table_schemas_prefetch,
        result_parts=result_parts,
        value_limit=int(getattr(context, "value_limit", 10) or 10),
        fallback_terms=fallback_terms,
    )

    # 5.6) fk_relationships XML
    stage_seq += 1
    fk_started = time.perf_counter()
    _emit_pipeline_stage(context, stage="fk_relationships", status="start", seq=stage_seq)
    fk_relationships = await fetch_fk_relationships_and_append_xml(
        context=context,
        selected_tables=col_res.selected_tables,
        result_parts=result_parts,
        limit=50,
        fk_relationships=fk_relationships_prefetch,
    )
    _emit_pipeline_stage(
        context,
        stage="fk_relationships",
        status="done",
        seq=stage_seq,
        elapsed_ms=(time.perf_counter() - fk_started) * 1000.0,
        counts={"fk_relationships": int(len(fk_relationships or []))},
    )

    # 6) resolved_values XML
    stage_seq += 1
    rv_started = time.perf_counter()
    _emit_pipeline_stage(context, stage="resolved_values", status="start", seq=stage_seq)
    resolved_values = await resolve_values_and_append_xml(
        context=context,
        value_mappings=sim.value_mappings,
        fallback_terms=fallback_terms,
        column_candidates=col_res.column_candidates,
        result_parts=result_parts,
    )
    _emit_pipeline_stage(
        context,
        stage="resolved_values",
        status="done",
        seq=stage_seq,
        elapsed_ms=(time.perf_counter() - rv_started) * 1000.0,
        counts={"resolved_values": int(len(resolved_values or []))},
    )

    # 7) suggestions XML
    stage_seq += 1
    sug_started = time.perf_counter()
    _emit_pipeline_stage(context, stage="suggestions", status="start", seq=stage_seq)
    suggestions = append_suggestions_xml(
        question=q,
        resolved_values=resolved_values,
        fallback_terms=fallback_terms,
        result_parts=result_parts,
    )
    _emit_pipeline_stage(
        context,
        stage="suggestions",
        status="done",
        seq=stage_seq,
        elapsed_ms=(time.perf_counter() - sug_started) * 1000.0,
        counts={"suggestions": int(len(suggestions or []))},
    )

    # 8) light disambiguation queries (always) + preview results
    stage_seq += 1
    lq_started = time.perf_counter()
    _emit_pipeline_stage(context, stage="light_queries", status="start", seq=stage_seq)
    light_generated_count, light_pass_count = await append_light_queries_and_results_xml(
        context=context,
        question=q,
        react_run_id=react_run_id,
        table_candidates=table_res.table_candidates,
        selected_tables=col_res.selected_tables,
        per_table_columns=col_res.per_table_columns,
        per_table_k=table_res.per_table_k,
        rerank_top_k=table_res.rerank_top_k,
        fk_relationships=fk_relationships,
        resolved_values=resolved_values,
        similar_queries=sim.similar_queries,
        result_parts=result_parts,
        table_description_overrides=table_desc_overrides,
        exclude_light_sqls=exclude_light_sqls,
    )
    _emit_pipeline_stage(
        context,
        stage="light_queries",
        status="done",
        seq=stage_seq,
        elapsed_ms=(time.perf_counter() - lq_started) * 1000.0,
        counts={
            "light_queries_total": int(light_generated_count),
            "light_queries_pass": int(light_pass_count),
        },
    )

    # Close XML
    result_parts.append("</build_sql_context_result>")
    result_parts.append("</tool_result>")
    tool_result = "\n".join(result_parts)

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.final_summary",
        category="react.tool.detail.build_sql_context",
        params={
            "react_run_id": react_run_id,
            "question": q,
            "elapsed_ms": elapsed_ms,
            "hyde": {
                "used_fallback": bool(hyde.used_fallback),
                "tables_keywords_count": len(hyde.keywords_tables),
                "columns_keywords_count": len(hyde.keywords_columns),
            },
            "similar_queries": {
                "count": len(sim.similar_queries),
                "top_scores": [sq.get("similarity_score") for sq in sim.similar_queries[:3]],
            },
            "value_mappings": {
                "count": len(sim.value_mappings),
            },
            "schema_candidates": {
                "tables_count": len(table_res.table_candidates),
                "columns_count": len(col_res.column_candidates),
                "per_table_columns_mode": col_res.per_table_mode,
                "per_table_k": table_res.per_table_k,
            },
            "relationships": {
                "fk_relationships_count": len(fk_relationships),
            },
            "resolved_values": {
                "total_count": len(resolved_values),
                "by_source": {
                    "value_mapping": len([r for r in resolved_values if r.get("source") == "value_mapping"]),
                    "db_probe": len([r for r in resolved_values if r.get("source") == "db_probe"]),
                },
            },
            "suggestions_count": len(suggestions),
            "light_queries": {
                "generated_count": int(light_generated_count),
                "pass_count": int(light_pass_count),
            },
        },
        max_inline_chars=0,
    )

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.xml_output",
        category="react.tool.detail.build_sql_context",
        params={"react_run_id": react_run_id, "tool_result": tool_result},
        max_inline_chars=0,
    )

    SmartLogger.log(
        "INFO",
        "react.build_sql_context.done",
        category="react.tool.build_sql_context",
        params=sanitize_for_log(
            {
                "react_run_id": react_run_id,
                "elapsed_ms": elapsed_ms,
                "keywords_count": len(hyde.all_keywords),
                "tables_count": len(table_res.table_candidates),
                "columns_count": len(col_res.column_candidates),
                "similar_queries_count": len(sim.similar_queries),
                "value_mappings_count": len(sim.value_mappings),
                "resolved_values_count": len(resolved_values),
            }
        ),
        max_inline_chars=0,
    )

    return tool_result


__all__ = ["execute"]


