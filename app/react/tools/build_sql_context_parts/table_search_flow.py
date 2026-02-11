from __future__ import annotations

import json
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

from app.config import settings
from app.react.generators.table_rerank_generator import TableRerankCandidate, get_table_rerank_generator
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger

from .models import TableCandidate
from .neo4j import (
    _neo4j_fetch_fk_neighbors_1hop,
    _neo4j_fetch_table_embedding_texts,
    _neo4j_fetch_table_embedding_texts_for_tables,
    _neo4j_fetch_tables_by_names,
    _neo4j_search_tables_text2sql_vector,
)
from .text import _base_name_candidates, _dedupe_keep_order, _pack_table_match_for_log, _table_name_penalty


@dataclass(frozen=True)
class TableSearchResult:
    rerank_top_k: int
    per_axis_top_k: int
    per_table_k: int
    table_candidates: List[TableCandidate]  # selected-first order
    selected_tables: List[TableCandidate]   # top rerank_top_k tables (after rerank/fallback)
    rerank_mode: str


async def search_tables_and_rerank(
    *,
    context,
    embedder,
    question: str,
    react_run_id: str | None,
    question_embedding: List[float],
    schema_embedding: List[float],
    hyde_embed_text: str,
    hyde_rerank_text: str,
    fallback_terms: Sequence[str],
    intent_embedding: Optional[List[float]],
    similar_queries: Sequence[Dict[str, Any]],
    result_parts: List[str],
) -> TableSearchResult:
    q = (question or "").strip()

    rerank_top_k = max(
        int(getattr(context, "table_rerank_top_k", getattr(settings, "hyde_union_rerank_top_a", 20))),
        1,
    )
    per_axis_top_k = max(int(getattr(settings, "hyde_per_axis_top_k", 20)), 1)
    per_table_k = max(int(context.scaled(getattr(context, "column_relation_limit", 10))), 1)

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.table_search.start",
        category="react.tool.detail.build_sql_context",
        params={
            "per_axis_top_k": int(per_axis_top_k),
            "rerank_top_k": int(rerank_top_k),
            "per_table_k": int(per_table_k),
            "schema_filter": getattr(context, "schema_filter", None),
        },
        max_inline_chars=0,
    )

    table_candidates: List[TableCandidate] = []
    selected_tables: List[TableCandidate] = []
    rerank_mode: str = "not_run"

    try:
        # P0: multi-axis retrieval + weighted merge
        axis_defs: List[Tuple[str, List[float], float]] = []
        axis_names_for_log: List[str] = []

        w_q = float(getattr(settings, "table_axis_weight_question", 0.35) or 0.35)
        w_h = float(getattr(settings, "table_axis_weight_hyde", 1.0) or 1.0)
        w_r = float(getattr(settings, "table_axis_weight_regex", 0.5) or 0.5)
        w_i = float(getattr(settings, "table_axis_weight_intent", 0.7) or 0.7)

        axis_defs.append(("question", question_embedding, w_q))

        # HyDE axis: reuse already-computed schema_embedding (HyDE embed text is what produced it).
        # This avoids a redundant embed call per tool run.
        axis_defs.append(("hyde_schema_embedding", schema_embedding, w_h))

        # Regex axis
        if fallback_terms:
            try:
                regex_text = " ".join(list(fallback_terms)[:20]).strip()
                if regex_text:
                    regex_vec = await embedder.embed_text(regex_text[:8000])
                    axis_defs.append(("regex", regex_vec, w_r))
            except Exception:
                pass

        # Intent axis
        if intent_embedding:
            axis_defs.append(("intent", intent_embedding, w_i))

        axis_keep_k = max(1, int(getattr(settings, "table_axis_keep_k", 80) or 80))
        axis_top_k = max(1, int(per_axis_top_k))

        agg_scores: Dict[str, float] = defaultdict(float)
        best_meta: Dict[str, TableCandidate] = {}
        best_axis_score: Dict[str, float] = defaultdict(lambda: -1e9)
        per_key_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        modes: List[str] = []

        for axis_name, axis_vec, axis_w in axis_defs:
            axis_names_for_log.append(axis_name)
            matches, mode = await _neo4j_search_tables_text2sql_vector(
                context=context,
                embedding=axis_vec,
                k=axis_top_k,
                schema_filter=getattr(context, "schema_filter", None),
            )
            modes.append(mode)
            adjusted: List[Tuple[float, TableCandidate]] = []
            for t in matches:
                base = float(t.score or 0.0)
                adj = float(axis_w) * base - _table_name_penalty(t.name)
                key = t.fqn.lower()
                adjusted.append((adj, t))
                per_key_scores[key][axis_name] = float(adj)
            adjusted.sort(key=lambda x: float(x[0]), reverse=True)
            for adj, t in adjusted[:axis_keep_k]:
                key = t.fqn.lower()
                agg_scores[key] = float(agg_scores.get(key, 0.0)) + float(adj)
                if float(adj) > float(best_axis_score.get(key, -1e9)):
                    best_axis_score[key] = float(adj)
                    best_meta[key] = t

        # P3: similar_queries injection
        try:
            boost = float(getattr(settings, "table_similar_query_boost", 0.02) or 0.02)
            inject_names: List[str] = []
            for sq in (similar_queries or [])[:5]:
                tu = sq.get("tables_used")
                if isinstance(tu, str):
                    try:
                        tu = json.loads(tu)
                    except Exception:
                        tu = None
                if isinstance(tu, (list, tuple)):
                    for x in tu:
                        s = str(x or "").strip()
                        if not s:
                            continue
                        inject_names.append(s.split(".", 1)[-1])
            inject_names = _dedupe_keep_order(inject_names, limit=50)
            if inject_names:
                schema_single = (
                    context.schema_filter[0]
                    if getattr(context, "schema_filter", None) and len(context.schema_filter) == 1
                    else None
                )
                fetched = await _neo4j_fetch_tables_by_names(
                    context=context, names=inject_names, schema=(schema_single or None)
                )
                for t in fetched:
                    key = t.fqn.lower()
                    adj = float(boost) - _table_name_penalty(t.name)
                    agg_scores[key] = float(agg_scores.get(key, 0.0)) + float(adj)
                    per_key_scores[key]["similar_query"] = float(adj)
                    if key not in best_meta:
                        best_meta[key] = t
        except Exception:
            pass

        # P1: PRF 1x
        try:
            prf_top_a = max(1, int(getattr(settings, "table_prf_top_a", 20) or 20))
            prf_top_k = max(1, int(getattr(settings, "table_prf_top_k", 300) or 300))
            prf_w = float(getattr(settings, "table_prf_weight", 0.8) or 0.8)
            prf_max_chars = max(200, int(getattr(settings, "table_prf_max_chars", 2500) or 2500))

            cur: List[Tuple[str, float]] = sorted(
                [(k, float(v)) for k, v in agg_scores.items() if k in best_meta],
                key=lambda x: float(x[1]),
                reverse=True,
            )
            top_keys = [k for k, _ in cur[:prf_top_a]]
            top_names = [best_meta[k].name for k in top_keys if k in best_meta and best_meta[k].name]
            if top_names:
                schema_single = (
                    context.schema_filter[0]
                    if getattr(context, "schema_filter", None) and len(context.schema_filter) == 1
                    else None
                )
                text_map = await _neo4j_fetch_table_embedding_texts(
                    context=context, names=top_names, schema=(schema_single or None)
                )
                parts: List[str] = []
                for k in top_keys:
                    txt = str(text_map.get(k, "") or "").strip()
                    if not txt:
                        continue
                    parts.append(txt)
                    if sum(len(x) for x in parts) > prf_max_chars:
                        break
                fb = "\n".join(parts).strip()
                if fb:
                    prf_text = (q + "\n" + fb).strip()
                    if len(prf_text) > prf_max_chars:
                        prf_text = prf_text[:prf_max_chars].rstrip()
                    prf_vec = await embedder.embed_text(prf_text[:8000])
                    prf_matches, prf_mode = await _neo4j_search_tables_text2sql_vector(
                        context=context,
                        embedding=prf_vec,
                        k=prf_top_k,
                        schema_filter=getattr(context, "schema_filter", None),
                    )
                    modes.append(prf_mode)
                    for t in prf_matches:
                        key = t.fqn.lower()
                        adj = float(prf_w) * float(t.score or 0.0) - _table_name_penalty(t.name)
                        agg_scores[key] = float(agg_scores.get(key, 0.0)) + float(adj)
                        per_key_scores[key]["prf"] = float(adj)
                        if key not in best_meta or float(adj) > float(best_axis_score.get(key, -1e9)):
                            best_meta[key] = t
                            best_axis_score[key] = float(adj)
        except Exception:
            pass

        # P2a: sibling expansion
        try:
            sibling_boost = float(getattr(settings, "table_sibling_boost", 0.006) or 0.006)
            cur_keys = sorted(
                [(k, float(v)) for k, v in agg_scores.items() if k in best_meta],
                key=lambda x: float(x[1]),
                reverse=True,
            )
            seeds = cur_keys[: min(200, len(cur_keys))]
            want: Dict[str, float] = {}
            want_name: Dict[str, str] = {}
            for k, sc in seeds:
                nm = best_meta[k].name if k in best_meta else ""
                for bn in _base_name_candidates(nm):
                    if not bn:
                        continue
                    if bn.lower() == (nm or "").lower():
                        continue
                    key_bn = bn.lower()
                    prev = float(want.get(key_bn, -1e9))
                    if float(sc) > prev:
                        want[key_bn] = float(sc)
                        want_name[key_bn] = bn
            if want_name:
                schema_single = (
                    context.schema_filter[0]
                    if getattr(context, "schema_filter", None) and len(context.schema_filter) == 1
                    else None
                )
                fetched = await _neo4j_fetch_tables_by_names(
                    context=context, names=list(want_name.values())[:200], schema=(schema_single or None)
                )
                for t in fetched:
                    base_sc = float(want.get(t.name.lower(), 0.0))
                    cand_sc = base_sc + float(sibling_boost) - _table_name_penalty(t.name)
                    key = t.fqn.lower()
                    if key not in agg_scores or float(cand_sc) > float(agg_scores.get(key, -1e9)):
                        agg_scores[key] = float(cand_sc)
                        best_meta[key] = t
                        per_key_scores[key]["sibling"] = float(sibling_boost)
        except Exception:
            pass

        # P2b: FK expansion
        try:
            fk_boost = float(getattr(settings, "table_fk_boost", 0.01) or 0.01)
            fk_seed_k = max(1, int(getattr(settings, "table_fk_seed_k", 25) or 25))
            fk_max_neighbors = max(10, int(getattr(settings, "table_fk_max_neighbors", 800) or 800))
            fk_enable_2hop = bool(getattr(settings, "table_fk_enable_2hop", True))

            cur_keys = sorted(
                [(k, float(v)) for k, v in agg_scores.items() if k in best_meta],
                key=lambda x: float(x[1]),
                reverse=True,
            )
            seed_keys = [k for k, _ in cur_keys[:fk_seed_k]]
            seed_fqns = [best_meta[k].fqn for k in seed_keys if k in best_meta and best_meta[k].fqn]
            if seed_fqns:
                schema_single = (
                    context.schema_filter[0]
                    if getattr(context, "schema_filter", None) and len(context.schema_filter) == 1
                    else None
                )
                hop1 = await _neo4j_fetch_fk_neighbors_1hop(
                    context=context,
                    seed_fqns=seed_fqns,
                    schema=(schema_single or None),
                    limit=fk_max_neighbors,
                )
                neighbors = list(hop1)
                if fk_enable_2hop and hop1:
                    hop1_fqns = [t.fqn for t in hop1 if t.fqn][:200]
                    hop2 = await _neo4j_fetch_fk_neighbors_1hop(
                        context=context,
                        seed_fqns=list(seed_fqns)[:200] + hop1_fqns,
                        schema=(schema_single or None),
                        limit=min(fk_max_neighbors, 600),
                    )
                    neighbors.extend(hop2)

                seed_best = float(cur_keys[0][1]) if cur_keys else 0.0
                base_neighbor = max(seed_best - 0.02, 0.0)
                for t in neighbors[:fk_max_neighbors]:
                    key = t.fqn.lower()
                    cand_sc = float(base_neighbor) + float(fk_boost) - _table_name_penalty(t.name)
                    if key not in agg_scores or float(cand_sc) > float(agg_scores.get(key, -1e9)):
                        agg_scores[key] = float(cand_sc)
                        best_meta[key] = t
                        per_key_scores[key]["fk"] = float(fk_boost)
        except Exception:
            pass

        table_candidates_full: List[TableCandidate] = []
        for key, sc in agg_scores.items():
            t = best_meta.get(key)
            if not t:
                continue
            table_candidates_full.append(
                TableCandidate(
                    schema=t.schema,
                    name=t.name,
                    description=t.description,
                    analyzed_description=t.analyzed_description,
                    score=float(sc),
                )
            )
        table_candidates_full.sort(key=lambda x: float(x.score or 0.0), reverse=True)

        # Log union preview
        try:
            preview = [_pack_table_match_for_log(x) for x in table_candidates_full[:50]]
            SmartLogger.log(
                "DEBUG",
                "react.build_sql_context.table_search.text2sql_vector.union",
                category="react.tool.detail.build_sql_context",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "schema_filter": getattr(context, "schema_filter", None),
                        "hyde_axes": axis_names_for_log,
                        "per_axis_top_k": int(axis_top_k),
                        "per_axis_keep_k": int(axis_keep_k),
                        "union_count": len(table_candidates_full),
                        "modes": modes,
                        "preview": preview,
                    }
                ),
                max_inline_chars=0,
            )
        except Exception as exc:
            SmartLogger.log(
                "WARNING",
                "react.build_sql_context.table_search.text2sql_vector.union.log_failed",
                category="react.tool.detail.build_sql_context",
                params={"error": str(exc), "traceback": traceback.format_exc()},
                max_inline_chars=0,
            )

        # Rerank (LLM) - failure handled inside generator
        reranked_indexes: Optional[List[int]] = None
        rerank_mode = "not_run"
        rerank_pool: List[TableCandidate] = []
        try:
            rerank_fetch_k = max(
                int(getattr(context, "table_rerank_fetch_k", 60) or 60),
                int(getattr(settings, "table_rerank_fetch_k", 0) or 0),
            )
            rerank_fetch_k = max(rerank_fetch_k, int(rerank_top_k))
            rerank_pool = table_candidates_full[: min(len(table_candidates_full), int(rerank_fetch_k))]

            # Prefer Table.text_to_sql_embedding_text for rerank input since description/analyzed_description
            # can be missing/weak. Keep this best-effort and never fail rerank due to metadata fetch.
            embed_text_by_fqn: Dict[str, str] = {}
            try:
                embed_text_by_fqn = await _neo4j_fetch_table_embedding_texts_for_tables(
                    context=context,
                    tables=list(rerank_pool)[: int(rerank_fetch_k)],
                )
            except Exception:
                embed_text_by_fqn = {}

            def _rerank_desc(t: TableCandidate) -> tuple[str, str]:
                tfqn_l = (t.fqn or "").strip().lower()
                embed = (embed_text_by_fqn.get(tfqn_l, "") if tfqn_l else "") or ""
                embed = str(embed).strip()
                if embed:
                    # Use embedding text for both fields to maximize useful signal.
                    return embed, embed
                desc = str(t.description or "").strip()
                analyzed = str(t.analyzed_description or "").strip()
                # Keep stable non-empty best-effort values.
                return (desc or analyzed), (analyzed or desc)

            rerank_candidates: List[TableRerankCandidate] = []
            for t in rerank_pool:
                desc, analyzed = _rerank_desc(t)
                rerank_candidates.append(
                    TableRerankCandidate(
                        schema=t.schema,
                        name=t.name,
                        description=desc,
                        analyzed_description=analyzed,
                        score=t.score,
                    )
                )
            reranked_indexes, rerank_mode = await get_table_rerank_generator().generate(
                question=q,
                hyde_summary=str(hyde_rerank_text or ""),
                candidates=rerank_candidates,
                top_k=int(rerank_top_k),
                react_run_id=react_run_id,
            )
        except Exception as exc:
            reranked_indexes = None
            rerank_mode = f"error:{type(exc).__name__}"
            SmartLogger.log(
                "WARNING",
                "react.build_sql_context.table_rerank.unexpected_error",
                category="react.tool.detail.build_sql_context",
                params={"error": str(exc), "traceback": traceback.format_exc()},
                max_inline_chars=0,
            )

        fallback_used = False
        fallback_mode = ""
        selected_tables = []
        if reranked_indexes:
            selected_tables = [
                rerank_pool[i] for i in reranked_indexes[:rerank_top_k] if 0 <= int(i) < len(rerank_pool)
            ]
        else:
            fallback_used = True
            selected_tables = table_candidates_full[: min(len(table_candidates_full), rerank_top_k)]
            fallback_gap = 0.002
            fallback_cap = 50
            pool_count = 0
            if len(table_candidates_full) >= rerank_top_k:
                score_at_k = float(table_candidates_full[rerank_top_k - 1].score or 0.0)
                pool = [
                    t for t in table_candidates_full if (score_at_k - float(t.score or 0.0)) <= float(fallback_gap)
                ]
                pool_count = min(len(pool), int(fallback_cap))
            fallback_mode = f"union_vec_top{rerank_top_k}_gap_le_{fallback_gap}"
            result_parts.append(
                f"<warning>table_rerank_fallback_used: {xml_escape(fallback_mode)}; pool_count={pool_count}</warning>"
            )

        selected_keys = {t.fqn.lower() for t in selected_tables}
        table_candidates = list(selected_tables) + [t for t in table_candidates_full if t.fqn.lower() not in selected_keys]

        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.table_search.done",
            category="react.tool.detail.build_sql_context",
            params={
                "table_candidates_count": len(table_candidates_full),
                "hyde_axes": axis_names_for_log,
                "per_axis_top_k": int(axis_top_k),
                "per_axis_keep_k": int(axis_keep_k),
                "rerank_mode": rerank_mode,
                "fallback_used": bool(fallback_used),
                "fallback_mode": fallback_mode,
                "selected_tables": [t.fqn for t in selected_tables],
            },
            max_inline_chars=0,
        )

        return TableSearchResult(
            rerank_top_k=int(rerank_top_k),
            per_axis_top_k=int(per_axis_top_k),
            per_table_k=int(per_table_k),
            table_candidates=table_candidates,
            selected_tables=selected_tables,
            rerank_mode=str(rerank_mode),
        )
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "react.build_sql_context.table_search.error",
            category="react.tool.detail.build_sql_context",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )
        result_parts.append(f"<warning>table_vec_search_failed: {xml_escape(str(exc)[:160])}</warning>")
        return TableSearchResult(
            rerank_top_k=int(rerank_top_k),
            per_axis_top_k=int(per_axis_top_k),
            per_table_k=int(per_table_k),
            table_candidates=[],
            selected_tables=[],
            rerank_mode="error",
        )


__all__ = ["TableSearchResult", "search_tables_and_rerank"]


