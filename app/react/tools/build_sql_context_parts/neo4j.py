from __future__ import annotations

import asyncio
import json
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.smart_logger import SmartLogger

from .models import ColumnCandidate, TableCandidate

_QUERY_INDEX_ENSURE_LOCK = asyncio.Lock()
_QUERY_INDEX_ENSURED_AT_MS: float = 0.0

_TEXT2SQL_TABLE_INDEX = "text_to_sql_table_vec_index"


def _is_missing_vector_index_error(exc: Exception, *, index_name: str) -> bool:
    msg = str(exc or "")
    if not msg:
        return False
    # neo4j procedure error often includes:
    # "There is no such vector schema index: <name>"
    return ("no such vector schema index" in msg.lower()) and (index_name in msg)


async def _ensure_query_vector_indexes(context) -> None:
    """
    Ensure Query-related constraints and vector indexes exist.
    - Safe to call multiple times (IF NOT EXISTS).
    - Throttled to avoid doing schema ops on every tool call.
    """
    global _QUERY_INDEX_ENSURED_AT_MS
    now_ms = time.time() * 1000.0
    # throttle: once per 10 minutes per process
    if _QUERY_INDEX_ENSURED_AT_MS and (now_ms - _QUERY_INDEX_ENSURED_AT_MS) < 10 * 60 * 1000.0:
        return
    async with _QUERY_INDEX_ENSURE_LOCK:
        now_ms = time.time() * 1000.0
        if _QUERY_INDEX_ENSURED_AT_MS and (now_ms - _QUERY_INDEX_ENSURED_AT_MS) < 10 * 60 * 1000.0:
            return
        try:
            # Import lazily to avoid potential circular imports during module init.
            from app.models.neo4j_history import Neo4jQueryRepository

            repo = Neo4jQueryRepository(context.neo4j_session)
            await repo.setup_constraints()
            _QUERY_INDEX_ENSURED_AT_MS = now_ms
            SmartLogger.log(
                "DEBUG",
                "react.build_sql_context.query_index.ensure.done",
                category="react.tool.detail.build_sql_context",
                params={"ensured_at_ms": int(_QUERY_INDEX_ENSURED_AT_MS)},
                max_inline_chars=0,
            )
        except Exception as exc:
            # Don't hard-fail the tool; similar query search will gracefully warn/continue.
            SmartLogger.log(
                "WARNING",
                "react.build_sql_context.query_index.ensure.failed",
                category="react.tool.detail.build_sql_context",
                params={"error": str(exc), "traceback": traceback.format_exc()},
                max_inline_chars=0,
            )


async def _neo4j_search_tables_text2sql_vector(
    *,
    context,
    embedding: List[float],
    k: int,
    schema_filter: Optional[Sequence[str]] = None,
) -> Tuple[List[TableCandidate], str]:
    """
    Search tables using Table.text_to_sql_vector (preferred) via vector index.
    Returns (candidates, mode).
    """
    k = max(1, int(k))
    schema_filter = [s.strip() for s in (schema_filter or []) if str(s or "").strip()]
    schema_filter_lower = [s.lower() for s in schema_filter]

    # Prefer vector index
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
    YIELD node, score
    WITH node, score
    WHERE node:Table AND node.text_to_sql_vector IS NOT NULL AND size(node.text_to_sql_vector) > 0
      AND COALESCE(node.text_to_sql_is_valid, true) = true
      AND ($schemas IS NULL OR toLower(COALESCE(node.schema,'')) IN $schemas)
    RETURN
      COALESCE(node.schema,'') AS schema,
      COALESCE(node.name,'') AS name,
      COALESCE(node.description,'') AS description,
      COALESCE(node.analyzed_description,'') AS analyzed_description,
      score AS score
    ORDER BY score DESC, schema ASC, name ASC
    LIMIT $k
    """
    try:
        res = await context.neo4j_session.run(
            cypher,
            index_name=_TEXT2SQL_TABLE_INDEX,
            k=int(k),
            embedding=embedding,
            schemas=(schema_filter_lower if schema_filter_lower else None),
        )
        rows = await res.data()
        out: List[TableCandidate] = []
        for r in rows:
            out.append(
                TableCandidate(
                    schema=str(r.get("schema") or ""),
                    name=str(r.get("name") or ""),
                    description=str(r.get("description") or ""),
                    analyzed_description=str(r.get("analyzed_description") or ""),
                    score=float(r.get("score") or 0.0),
                )
            )
        return out, "text2sql_vec_index"
    except Exception as exc:
        if _is_missing_vector_index_error(exc, index_name=_TEXT2SQL_TABLE_INDEX):
            # fallback to scan below
            pass
        else:
            SmartLogger.log(
                "WARNING",
                "react.build_sql_context.table_search.text2sql_vec_index.failed",
                category="react.tool.detail.build_sql_context",
                params={"error": str(exc), "traceback": traceback.format_exc()},
                max_inline_chars=0,
            )

    # Fallback: full scan + cosine similarity (slower; debug-friendly)
    cypher_scan = """
    MATCH (t:Table)
    WHERE t.text_to_sql_vector IS NOT NULL AND size(t.text_to_sql_vector) > 0
      AND COALESCE(t.text_to_sql_is_valid, true) = true
      AND ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
    WITH t, vector.similarity.cosine(t.text_to_sql_vector, $embedding) AS score
    RETURN
      COALESCE(t.schema,'') AS schema,
      COALESCE(t.name,'') AS name,
      COALESCE(t.description,'') AS description,
      COALESCE(t.analyzed_description,'') AS analyzed_description,
      score AS score
    ORDER BY score DESC, schema ASC, name ASC
    LIMIT $k
    """
    res = await context.neo4j_session.run(
        cypher_scan,
        k=int(k),
        embedding=embedding,
        schemas=(schema_filter_lower if schema_filter_lower else None),
    )
    rows = await res.data()
    out2: List[TableCandidate] = []
    for r in rows:
        out2.append(
            TableCandidate(
                schema=str(r.get("schema") or ""),
                name=str(r.get("name") or ""),
                description=str(r.get("description") or ""),
                analyzed_description=str(r.get("analyzed_description") or ""),
                score=float(r.get("score") or 0.0),
            )
        )
    return out2, "text2sql_vec_scan_fallback"


async def _neo4j_fetch_tables_by_names(
    *,
    context,
    names: Sequence[str],
    schema: Optional[str],
) -> List[TableCandidate]:
    cleaned = [str(x).strip() for x in (names or []) if str(x or "").strip()]
    if not cleaned:
        return []
    names_l = [x.lower() for x in cleaned][:200]
    cypher = """
    UNWIND $names AS nm
    MATCH (t:Table)
    WHERE toLower(COALESCE(t.name,'')) = nm
      AND ($schema IS NULL OR toLower(COALESCE(t.schema,'')) = toLower($schema))
      AND COALESCE(t.text_to_sql_is_valid, true) = true
    RETURN
      COALESCE(t.schema,'') AS schema,
      COALESCE(t.name,'') AS name,
      COALESCE(t.description,'') AS description,
      COALESCE(t.analyzed_description,'') AS analyzed_description
    """
    res = await context.neo4j_session.run(cypher, names=names_l, schema=(schema or None))
    rows = await res.data()
    out: List[TableCandidate] = []
    for r in rows:
        out.append(
            TableCandidate(
                schema=str(r.get("schema") or ""),
                name=str(r.get("name") or ""),
                description=str(r.get("description") or ""),
                analyzed_description=str(r.get("analyzed_description") or ""),
                score=0.0,
            )
        )
    return out


async def _neo4j_fetch_table_embedding_texts(
    *,
    context,
    names: Sequence[str],
    schema: Optional[str],
) -> Dict[str, str]:
    cleaned = [str(x).strip() for x in (names or []) if str(x or "").strip()]
    if not cleaned:
        return {}
    names_l = [x.lower() for x in cleaned][:200]
    cypher = """
    UNWIND $names AS nm
    MATCH (t:Table)
    WHERE toLower(COALESCE(t.name,'')) = nm
      AND ($schema IS NULL OR toLower(COALESCE(t.schema,'')) = toLower($schema))
    RETURN
      COALESCE(t.schema,'') AS schema,
      COALESCE(t.name,'') AS name,
      COALESCE(t.text_to_sql_embedding_text,'') AS embedding_text
    """
    res = await context.neo4j_session.run(cypher, names=names_l, schema=(schema or None))
    rows = await res.data()
    out: Dict[str, str] = {}
    for r in rows:
        sch = str(r.get("schema") or "")
        nm = str(r.get("name") or "")
        fqn = f"{sch}.{nm}" if sch else nm
        out[fqn.lower()] = str(r.get("embedding_text") or "")
    return out


async def _neo4j_fetch_table_embedding_texts_for_tables(
    *,
    context,
    tables: Sequence[TableCandidate],
) -> Dict[str, str]:
    """
    Fetch Table.text_to_sql_embedding_text for the provided tables, keyed by fqn_lower ("schema.name" lower).

    This is intended for *output rendering* (e.g. build_sql_context XML) so we only fetch for a small set of
    already-selected tables to avoid pulling large embedding texts during the recall/rerank stages.
    """
    requested: List[Dict[str, Optional[str]]] = []
    for t in tables or []:
        name = (t.name or "").strip()
        schema = (t.schema or "").strip()
        if not name:
            continue
        requested.append({"schema": schema.lower() if schema else None, "name": name.lower()})
    if not requested:
        return {}

    cypher = """
    UNWIND $requested AS req
    MATCH (t:Table)
    WHERE (
      (t.name IS NOT NULL AND toLower(t.name) = req.name)
      OR (t.original_name IS NOT NULL AND toLower(t.original_name) = req.name)
    )
      AND (req.schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = req.schema))
    RETURN
      COALESCE(t.schema,'') AS schema,
      COALESCE(t.name,'') AS name,
      COALESCE(t.text_to_sql_embedding_text,'') AS embedding_text
    """
    res = await context.neo4j_session.run(cypher, requested=requested)
    rows = await res.data()
    out: Dict[str, str] = {}
    for r in rows:
        sch = str(r.get("schema") or "")
        nm = str(r.get("name") or "")
        fqn = f"{sch}.{nm}" if sch else nm
        out[fqn.lower()] = str(r.get("embedding_text") or "")
    return out


async def _neo4j_fetch_fk_neighbors_1hop(
    *,
    context,
    seed_fqns: Sequence[str],
    schema: Optional[str],
    limit: int,
) -> List[TableCandidate]:
    """
    FK neighbor expansion outside the seed set (1 hop).
    """
    seeds = [str(x).strip().lower() for x in (seed_fqns or []) if str(x or "").strip()]
    if not seeds:
        return []
    cypher = """
    UNWIND $seed_fqns AS seed
    MATCH (t1:Table)-[:HAS_COLUMN]->(c1:Column)-[:FK_TO]->(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
    WITH t2,
         (toLower(COALESCE(t1.schema,'')) + '.' + toLower(COALESCE(t1.name,''))) AS fqn1
    WHERE fqn1 = seed
      AND ($schema IS NULL OR toLower(COALESCE(t2.schema,'')) = toLower($schema))
      AND COALESCE(t2.text_to_sql_is_valid, true) = true
    RETURN DISTINCT
      COALESCE(t2.schema,'') AS schema,
      COALESCE(t2.name,'') AS name,
      COALESCE(t2.description,'') AS description,
      COALESCE(t2.analyzed_description,'') AS analyzed_description
    LIMIT $limit
    """
    res = await context.neo4j_session.run(
        cypher, seed_fqns=seeds[:200], schema=(schema or None), limit=int(max(1, limit))
    )
    rows = await res.data()
    out: List[TableCandidate] = []
    for r in rows:
        out.append(
            TableCandidate(
                schema=str(r.get("schema") or ""),
                name=str(r.get("name") or ""),
                description=str(r.get("description") or ""),
                analyzed_description=str(r.get("analyzed_description") or ""),
                score=0.0,
            )
        )
    return out


async def _neo4j_search_table_scoped_columns(
    *,
    context,
    embedding: List[float],
    tables: Sequence[TableCandidate],
    per_table_k: int,
) -> Tuple[Dict[str, List[ColumnCandidate]], str]:
    """
    Return per-table top-K columns restricted to the provided tables.
    Primary mode: compute cosine similarity within the table-scoped columns.
    Fallback mode: global vector index query + post-filter + per-table grouping (best-effort).
    Returns: (map: table_fqn_lower -> [ColumnCandidate], mode)
    """
    requested: List[Dict[str, Optional[str]]] = []
    table_fqns_lower: List[str] = []
    for t in tables:
        name = (t.name or "").strip()
        schema = (t.schema or "").strip()
        if not name:
            continue
        requested.append({"schema": schema.lower() if schema else None, "name": name.lower()})
        fqn_lower = f"{schema.lower()}.{name.lower()}" if schema else name.lower()
        table_fqns_lower.append(fqn_lower)

    per_table_k = max(int(per_table_k), 1)
    if not requested:
        return {}, "no_tables"

    cypher_cosine = """
    UNWIND $requested AS req
    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
    WHERE (
      (c.vector IS NOT NULL AND size(c.vector) > 0)
      AND COALESCE(t.text_to_sql_is_valid, true) = true
      AND COALESCE(c.text_to_sql_is_valid, true) = true
      AND (
        (t.name IS NOT NULL AND toLower(t.name) = req.name)
        OR (t.original_name IS NOT NULL AND toLower(t.original_name) = req.name)
      )
      AND (req.schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = req.schema))
    )
    WITH t, c, req,
         vector.similarity.cosine(c.vector, $embedding) AS score
    ORDER BY score DESC
    WITH req, t, collect({
      name: c.name,
      dtype: c.dtype,
      description: c.description,
      score: score
    }) AS cols
    RETURN COALESCE(t.schema,'') AS table_schema,
           COALESCE(t.original_name, t.name) AS table_name,
           cols[0..$per_table_k] AS columns
    """
    try:
        res = await context.neo4j_session.run(
            cypher_cosine,
            requested=requested,
            embedding=embedding,
            per_table_k=int(per_table_k),
        )
        rows = await res.data()
        out: Dict[str, List[ColumnCandidate]] = {}
        for r in rows:
            schema = str(r.get("table_schema") or "")
            name = str(r.get("table_name") or "")
            schema_l = schema.lower()
            name_l = name.lower()
            tfqn_l = f"{schema_l}.{name_l}" if schema_l else name_l
            cols = r.get("columns") or []
            cand_list: List[ColumnCandidate] = []
            for c in cols:
                if not isinstance(c, dict):
                    continue
                col_name = str(c.get("name") or "").strip()
                if not col_name:
                    continue
                cand_list.append(
                    ColumnCandidate(
                        table_schema=schema,
                        table_name=name,
                        name=col_name,
                        dtype=str(c.get("dtype") or ""),
                        description=str(c.get("description") or ""),
                        score=float(c.get("score") or 0.0),
                    )
                )
            out[tfqn_l] = cand_list
        return out, "table_scoped_cosine"
    except Exception as exc:
        SmartLogger.log(
            "WARNING",
            "react.build_sql_context.column_search.table_scoped_cosine.failed",
            category="react.tool.detail.build_sql_context",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )

    fetch_k = min(8000, max(500, per_table_k * len(requested) * 20))
    cypher_fallback = """
    CALL db.index.vector.queryNodes('column_vec_index', $k, $embedding)
    YIELD node, score
    MATCH (t:Table)-[:HAS_COLUMN]->(node)
    WITH t, node, score,
         (toLower(COALESCE(t.schema,'')) + '.' + toLower(COALESCE(t.original_name, t.name))) AS tfqn_lower
    WHERE tfqn_lower IN $table_fqns_lower
      AND COALESCE(t.text_to_sql_is_valid, true) = true
      AND COALESCE(node.text_to_sql_is_valid, true) = true
    RETURN tfqn_lower AS table_fqn_lower,
           COALESCE(t.schema,'') AS table_schema,
           COALESCE(t.original_name, t.name) AS table_name,
           node.name AS name,
           node.dtype AS dtype,
           node.description AS description,
           score AS score
    ORDER BY table_fqn_lower ASC, score DESC, name ASC
    """
    res = await context.neo4j_session.run(
        cypher_fallback,
        k=int(fetch_k),
        embedding=embedding,
        table_fqns_lower=table_fqns_lower,
    )
    rows = await res.data()
    by_table: Dict[str, List[ColumnCandidate]] = {tfqn: [] for tfqn in table_fqns_lower}
    for r in rows:
        tfqn_l = str(r.get("table_fqn_lower") or "").lower()
        if not tfqn_l or tfqn_l not in by_table:
            continue
        col_name = str(r.get("name") or "").strip()
        if not col_name:
            continue
        by_table[tfqn_l].append(
            ColumnCandidate(
                table_schema=str(r.get("table_schema") or ""),
                table_name=str(r.get("table_name") or ""),
                name=col_name,
                dtype=str(r.get("dtype") or ""),
                description=str(r.get("description") or ""),
                score=float(r.get("score") or 0.0),
            )
        )
    out2: Dict[str, List[ColumnCandidate]] = {}
    for tfqn, cols in by_table.items():
        seen = set()
        picked: List[ColumnCandidate] = []
        for c in cols:
            key = (c.name or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            picked.append(c)
            if len(picked) >= per_table_k:
                break
        out2[tfqn] = picked
    return out2, "global_vec_fallback"


async def _neo4j_fetch_anchor_like_columns_for_tables(
    *,
    context,
    tables: Sequence[TableCandidate],
    name_substrings_lower: Sequence[str],
    keywords_lower: Sequence[str],
    per_table_limit: int = 10,
) -> List[ColumnCandidate]:
    requested: List[Dict[str, Optional[str]]] = []
    for t in tables:
        name = (t.name or "").strip()
        schema = (t.schema or "").strip()
        if not name:
            continue
        requested.append({"schema": schema.lower() if schema else None, "name": name.lower()})
    if not requested:
        return []

    subs = [str(s or "").strip().lower() for s in name_substrings_lower if str(s or "").strip()]
    kws = [str(k or "").strip().lower() for k in keywords_lower if str(k or "").strip()]
    subs = subs[:20]
    kws = kws[:20]
    if not subs and not kws:
        return []

    cypher = """
    UNWIND $requested AS req
    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
    WHERE (
      (
        (t.name IS NOT NULL AND toLower(t.name) = req.name)
        OR (t.original_name IS NOT NULL AND toLower(t.original_name) = req.name)
      )
      AND (req.schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = req.schema))
      AND COALESCE(t.text_to_sql_is_valid, true) = true
      AND COALESCE(c.text_to_sql_is_valid, true) = true
      AND (
        any(sub IN $subs WHERE (c.name IS NOT NULL AND toLower(c.name) CONTAINS sub))
        OR any(kw IN $kws WHERE
          (c.name IS NOT NULL AND toLower(c.name) CONTAINS kw)
          OR (c.description IS NOT NULL AND toLower(c.description) CONTAINS kw)
        )
      )
    )
    WITH req, t, c
    ORDER BY c.name
    WITH req, t, collect(c)[0..$per_table_limit] AS cols
    UNWIND cols AS c
    RETURN COALESCE(t.schema,'') AS table_schema,
           COALESCE(t.original_name, t.name) AS table_name,
           c.name AS name,
           c.dtype AS dtype,
           c.description AS description
    ORDER BY table_schema, table_name, name
    """
    res = await context.neo4j_session.run(
        cypher,
        requested=requested,
        subs=subs,
        kws=kws,
        per_table_limit=int(max(1, per_table_limit)),
    )
    rows = await res.data()
    out: List[ColumnCandidate] = []
    for r in rows:
        out.append(
            ColumnCandidate(
                table_schema=str(r.get("table_schema") or ""),
                table_name=str(r.get("table_name") or ""),
                name=str(r.get("name") or ""),
                dtype=str(r.get("dtype") or ""),
                description=str(r.get("description") or ""),
                score=0.5,
            )
        )
    return out


async def _neo4j_search_columns(
    *,
    context,
    embedding: List[float],
    k: int,
) -> List[ColumnCandidate]:
    cypher = """
    CALL db.index.vector.queryNodes('column_vec_index', $k, $embedding)
    YIELD node, score
    MATCH (t:Table)-[:HAS_COLUMN]->(node)
    WHERE COALESCE(t.text_to_sql_is_valid, true) = true
      AND COALESCE(node.text_to_sql_is_valid, true) = true
    RETURN node.name AS name,
           t.schema AS table_schema,
           COALESCE(t.original_name, t.name) AS table_name,
           node.dtype AS dtype,
           node.description AS description,
           score AS score
    ORDER BY score DESC, table_schema ASC, table_name ASC, name ASC
    LIMIT $k
    """
    res = await context.neo4j_session.run(cypher, k=int(k), embedding=embedding)
    records = await res.data()
    out: List[ColumnCandidate] = []
    for r in records:
        out.append(
            ColumnCandidate(
                table_schema=str(r.get("table_schema") or ""),
                table_name=str(r.get("table_name") or ""),
                name=str(r.get("name") or ""),
                dtype=str(r.get("dtype") or ""),
                description=str(r.get("description") or ""),
                score=float(r.get("score") or 0.0),
            )
        )
    return out


async def _neo4j_find_similar_queries_and_mappings(
    *,
    context,
    question: str,
    question_embedding: List[float],
    intent_embedding: Optional[List[float]] = None,
    terms: Sequence[str],
    min_similarity: float = 0.3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    similar_queries: List[Dict[str, Any]] = []
    value_mappings: List[Dict[str, Any]] = []

    # NOTE: For safety, we can restrict cache usage to verified-only data.
    # This is a fail-closed choice to reduce contamination loops from bad cache artifacts.
    from app.config import settings

    use_verified_only = bool(getattr(settings, "cache_postprocess_use_verified_only", True))
    allow_vm_substring_fallback = bool(getattr(settings, "value_mapping_substring_fallback_enabled", False))

    verified_clause_query = "AND COALESCE(node.verified, false) = true" if use_verified_only else ""
    verified_clause_vm = "AND COALESCE(v.verified, false) = true" if use_verified_only else ""

    cypher_similar = f"""
    CALL db.index.vector.queryNodes($index_name, $k, $embedding)
    YIELD node, score
    WITH node, score
    WHERE node:Query
      AND node.status = 'completed'
      AND node.sql IS NOT NULL
      {verified_clause_query}
      AND score >= $min_score
    RETURN node.id AS id,
           node.question AS question,
           node.sql AS sql,
           node.steps_count AS steps_count,
           node.execution_time_ms AS execution_time_ms,
           node.tables_used AS tables_used,
           node.columns_used AS columns_used,
           node.best_run_at_ms AS best_run_at_ms,
           node.intent_text AS intent_text,
           node.best_context_score AS best_context_score,
           node.best_context_steps_features AS best_context_steps_features,
           node.best_context_steps_summary AS best_context_steps_summary,
           score AS similarity_score
    ORDER BY similarity_score DESC, best_run_at_ms DESC
    LIMIT $k
    """

    fetch_k = 20
    limit_k = 5
    q_index = "query_question_vec_index"
    try:
        q_res = await context.neo4j_session.run(
            cypher_similar,
            index_name=q_index,
            k=int(fetch_k),
            embedding=question_embedding,
            min_score=float(min_similarity),
        )
        q_rows = await q_res.data()
    except Exception as exc:
        if _is_missing_vector_index_error(exc, index_name=q_index):
            await _ensure_query_vector_indexes(context)
            q_res = await context.neo4j_session.run(
                cypher_similar,
                index_name=q_index,
                k=int(fetch_k),
                embedding=question_embedding,
                min_score=float(min_similarity),
            )
            q_rows = await q_res.data()
        else:
            raise

    i_rows: List[Dict[str, Any]] = []
    if intent_embedding:
        i_index = "query_intent_vec_index"
        try:
            i_res = await context.neo4j_session.run(
                cypher_similar,
                index_name=i_index,
                k=int(fetch_k),
                embedding=intent_embedding,
                min_score=float(min_similarity),
            )
            i_rows = await i_res.data()
        except Exception as exc:
            if _is_missing_vector_index_error(exc, index_name=i_index):
                await _ensure_query_vector_indexes(context)
                i_res = await context.neo4j_session.run(
                    cypher_similar,
                    index_name=i_index,
                    k=int(fetch_k),
                    embedding=intent_embedding,
                    min_score=float(min_similarity),
                )
                i_rows = await i_res.data()
            else:
                raise

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in q_rows:
        qid = str(r.get("id") or "")
        if not qid:
            continue
        base = dict(r)
        base["_question_score"] = float(r.get("similarity_score") or 0.0)
        base["_intent_score"] = 0.0
        by_id[qid] = base

    for r in i_rows:
        qid = str(r.get("id") or "")
        if not qid:
            continue
        if qid not in by_id:
            base = dict(r)
            base["_question_score"] = 0.0
            base["_intent_score"] = float(r.get("similarity_score") or 0.0)
            by_id[qid] = base
        else:
            by_id[qid]["_intent_score"] = max(
                float(by_id[qid].get("_intent_score") or 0.0),
                float(r.get("similarity_score") or 0.0),
            )

    use_intent = bool(intent_embedding)
    for _qid, r in by_id.items():
        q_score = float(r.get("_question_score") or 0.0)
        i_score = float(r.get("_intent_score") or 0.0)
        final = (0.6 * i_score + 0.4 * q_score) if use_intent else q_score
        r["similarity_score"] = float(final)
        r["question_similarity_score"] = q_score
        r["intent_similarity_score"] = i_score

    def _best_run_ms(x: Dict[str, Any]) -> int:
        try:
            return int(x.get("best_run_at_ms") or 0)
        except Exception:
            return 0

    similar_queries = sorted(
        by_id.values(),
        key=lambda x: (float(x.get("similarity_score") or 0.0), _best_run_ms(x)),
        reverse=True,
    )[:limit_k]

    if terms:
        # Prefer exact term equality lookups so Neo4j can use indexes and avoid relationship scans.
        # NOTE: `terms` already includes particle-stripped variants (see _regex_terms), so exact match
        # usually works for Korean entity strings (e.g., 청주정수장의 -> 청주정수장).
        cleaned_terms = [str(t or "").strip() for t in list(terms or [])[:50] if str(t or "").strip()]
        # 1) Exact match (index-friendly)
        # NOTE: f-string is used only to inject verified-only clause; Cypher map literals must escape braces.
        cypher_vm_exact = f"""
        UNWIND $terms AS term
        MATCH (v:ValueMapping {{natural_value: term}})-[:MAPS_TO]->(c:Column)
        WHERE true {verified_clause_vm}
        RETURN v.natural_value AS natural_value,
               v.code_value AS code_value,
               c.fqn AS column_fqn,
               c.name AS column_name,
               v.usage_count AS usage_count
        ORDER BY v.usage_count DESC
        LIMIT 20
        """
        vm_res = await context.neo4j_session.run(cypher_vm_exact, terms=cleaned_terms)
        value_mappings = await vm_res.data()

        # 2) Fallback: substring match (slower; only used when exact match yields nothing)
        if allow_vm_substring_fallback and (not value_mappings) and cleaned_terms:
            cypher_vm_fallback = f"""
            MATCH (v:ValueMapping)-[:MAPS_TO]->(c:Column)
            WHERE any(term IN $terms WHERE toLower(v.natural_value) CONTAINS toLower(term))
              {verified_clause_vm}
            RETURN v.natural_value AS natural_value,
                   v.code_value AS code_value,
                   c.fqn AS column_fqn,
                   c.name AS column_name,
                   v.usage_count AS usage_count
            ORDER BY v.usage_count DESC
            LIMIT 20
            """
            vm_res2 = await context.neo4j_session.run(cypher_vm_fallback, terms=cleaned_terms)
            value_mappings = await vm_res2.data()

    return similar_queries, value_mappings


async def _neo4j_fetch_table_schemas(
    *,
    context,
    tables: Sequence[TableCandidate],
) -> List[Dict[str, Any]]:
    requested: List[Dict[str, Optional[str]]] = []
    for t in tables:
        name = (t.name or "").strip()
        schema = (t.schema or "").strip()
        if not name:
            continue
        requested.append({"schema": schema.lower() if schema else None, "name": name.lower()})
    if not requested:
        return []

    query = """
    UNWIND $requested AS req
    MATCH (t:Table)
    WHERE (
      (t.name IS NOT NULL AND toLower(t.name) = req.name)
      OR (t.original_name IS NOT NULL AND toLower(t.original_name) = req.name)
    )
      AND (req.schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = req.schema))
      AND COALESCE(t.text_to_sql_is_valid, true) = true
    WITH DISTINCT t
    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
    WHERE COALESCE(c.text_to_sql_is_valid, true) = true
    WITH t, c
    ORDER BY c.name
    RETURN COALESCE(t.original_name, t.name) AS table_name,
           t.schema AS table_schema,
           t.description AS table_description,
           collect({
               name: c.name,
               fqn: c.fqn,
               dtype: c.dtype,
               nullable: c.nullable,
               description: c.description,
               is_primary_key: c.is_primary_key,
               enum_values: c.enum_values,
               cardinality: c.cardinality
           }) AS columns
    ORDER BY table_schema, table_name
    """
    res = await context.neo4j_session.run(query, requested=requested)
    records = await res.data()
    out: List[Dict[str, Any]] = []
    for r in records:
        out.append(
            {
                "schema": str(r.get("table_schema") or ""),
                "name": str(r.get("table_name") or ""),
                "description": str(r.get("table_description") or ""),
                "columns": r.get("columns") or [],
            }
        )
    return out


async def _neo4j_fetch_fk_relationships(
    *,
    context,
    table_fqns: Sequence[str],
    limit: int = 30,
) -> List[Dict[str, Any]]:
    if not table_fqns:
        return []
    # Normalize to lowercase for case-insensitive matching (Table.name is stored lowercase,
    # but Table.original_name may preserve original casing).
    table_fqns_l = [str(x or "").strip().lower() for x in (table_fqns or []) if str(x or "").strip()]
    cypher = """
    MATCH (t1:Table)-[:HAS_COLUMN]->(c1:Column)-[fk:FK_TO]->(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
    WITH t1, c1, fk, c2, t2,
         (toLower(COALESCE(t1.schema,'')) + '.' + toLower(COALESCE(t1.original_name, t1.name))) AS fqn1,
         (toLower(COALESCE(t2.schema,'')) + '.' + toLower(COALESCE(t2.original_name, t2.name))) AS fqn2
    WHERE fqn1 IN $table_fqns AND fqn2 IN $table_fqns
      AND COALESCE(t1.text_to_sql_is_valid, true) = true
      AND COALESCE(t2.text_to_sql_is_valid, true) = true
      AND COALESCE(c1.text_to_sql_is_valid, true) = true
      AND COALESCE(c2.text_to_sql_is_valid, true) = true
    RETURN COALESCE(t1.original_name, t1.name) AS from_table,
           t1.schema AS from_schema,
           c1.name AS from_column,
           COALESCE(t2.original_name, t2.name) AS to_table,
           t2.schema AS to_schema,
           c2.name AS to_column,
           fk.constraint AS constraint_name
    ORDER BY from_schema, from_table, to_schema, to_table, from_column, to_column
    LIMIT $limit
    """
    res = await context.neo4j_session.run(
        cypher, table_fqns=list(table_fqns_l), limit=int(limit)
    )
    return await res.data()


__all__ = [
    "_TEXT2SQL_TABLE_INDEX",
    "_is_missing_vector_index_error",
    "_ensure_query_vector_indexes",
    "_neo4j_search_tables_text2sql_vector",
    "_neo4j_fetch_tables_by_names",
    "_neo4j_fetch_table_embedding_texts",
    "_neo4j_fetch_table_embedding_texts_for_tables",
    "_neo4j_fetch_fk_neighbors_1hop",
    "_neo4j_search_table_scoped_columns",
    "_neo4j_fetch_anchor_like_columns_for_tables",
    "_neo4j_search_columns",
    "_neo4j_find_similar_queries_and_mappings",
    "_neo4j_fetch_table_schemas",
    "_neo4j_fetch_fk_relationships",
]


