"""
Startup-time vectorization for Text-to-SQL table retrieval.

Creates/updates:
- (:Table).text_to_sql_vector: embedding vector used for Text-to-SQL table retrieval
- (:Table).text_to_sql_profile: JSON string (small) for debugging
- (:Table).text_to_sql_updated_at: datetime()

Design goals:
- Deterministic & bounded: per-table DB sample is limited (LIMIT N, timeout)
- Light LLM: use settings.light_llm_* (via react.llm_factory.create_react_llm)
- Batch embeddings for cost/speed
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import asyncpg

from app.config import settings
from app.core.llm_factory import create_embedding_client
from app.core.sql_guard import SQLGuard
from app.react.generators.table_profile_generator import (
    TableProfileInput,
    get_table_profile_generator,
)
from app.smart_logger import SmartLogger


def _truncate(s: str, max_len: int) -> str:
    t = str(s or "")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _truncate_value(v: Any, *, max_len: int = 80) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = " ".join(str(v).split())
    return _truncate(s, max_len)


async def _open_db_pool(*, max_size: int) -> asyncpg.Pool:
    ssl_mode = settings.target_db_ssl if settings.target_db_ssl != "disable" else False
    schemas = (settings.target_db_schemas or "").split(",")
    schemas_str = ", ".join(s.strip() for s in schemas if s.strip())

    async def _init(conn: asyncpg.Connection) -> None:
        if schemas_str:
            await conn.execute(f"SET search_path TO {schemas_str}")

    return await asyncpg.create_pool(
        host=settings.target_db_host,
        port=settings.target_db_port,
        database=settings.target_db_name,
        user=settings.target_db_user,
        password=settings.target_db_password,
        ssl=ssl_mode,
        min_size=1,
        max_size=max(1, int(max_size)),
        init=_init,
    )


async def _resolve_table_case(
    conn: asyncpg.Connection, *, schema: str, table: str
) -> Optional[Tuple[str, str]]:
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        return None
    q = """
    SELECT table_schema, table_name
    FROM information_schema.tables
    WHERE lower(table_schema) = lower($1)
      AND lower(table_name) = lower($2)
    LIMIT 1
    """
    row = await conn.fetchrow(q, schema, table)
    if not row:
        return None
    return row["table_schema"], row["table_name"]


async def _fetch_sample_rows(
    *,
    db_pool: asyncpg.Pool,
    schema: str,
    table_name: str,
    limit: int,
    timeout_s: float,
) -> List[Dict[str, Any]]:
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        return []

    async with db_pool.acquire() as conn:
        resolved = await _resolve_table_case(conn, schema=schema, table=table_name)

        def q_ident(x: str) -> str:
            return '"' + str(x).replace('"', '""') + '"'

        if resolved:
            schema_real, table_real = resolved
            table_ident = f"{q_ident(schema_real)}.{q_ident(table_real)}"
        else:
            schema_id = SQLGuard.sanitize_identifier(schema) if schema else ""
            table_id = SQLGuard.sanitize_identifier(table_name)
            if not table_id:
                return []
            table_ident = ".".join([f'"{p}"' for p in [schema_id, table_id] if p])

        sql = f"SELECT * FROM {table_ident} T LIMIT {int(max(1, limit))}"
        try:
            rows = await asyncio.wait_for(conn.fetch(sql), timeout=timeout_s)
        except Exception:
            return []

    out: List[Dict[str, Any]] = []
    for row in rows:
        d: Dict[str, Any] = {}
        for k in row.keys():
            d[str(k)] = _truncate_value(row.get(k), max_len=80)
        out.append(d)
    return out


@dataclass(frozen=True)
class _TableItem:
    tid: str
    db: str
    schema: str
    name: str
    description: str
    analyzed_description: str
    columns: List[Dict[str, Any]]

    @property
    def fqn(self) -> str:
        s = (self.schema or "").strip()
        n = (self.name or "").strip()
        return f"{s}.{n}" if s else n


async def _fetch_tables_missing_text2sql_vector(neo4j_session) -> List[_TableItem]:
    """
    Fetch tables where text_to_sql_vector is missing/empty, including columns metadata.
    """
    # Schema filter:
    # - Historically this used settings.target_db_schemas (default "public"), which can
    #   accidentally exclude all ingested :Table nodes (e.g., schemas like "RWIS").
    # - We keep schema filtering (to bound cost) but add a safe fallback:
    #   if the filter yields 0 tables, retry without the schema filter.
    schemas_raw = (getattr(settings, "target_db_schemas", "") or "").strip()
    schema_filter = [s.strip().lower() for s in schemas_raw.split(",") if s.strip()]

    base_where = "(t.text_to_sql_vector IS NULL OR size(coalesce(t.text_to_sql_vector, [])) = 0)"

    def _cypher(where_clause: str) -> str:
        return f"""
        MATCH (t:Table)
        WHERE {where_clause}
    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
    WITH t, c
    ORDER BY c.name
    WITH t, collect({{
      name: COALESCE(c.name,''),
      dtype: COALESCE(c.dtype,''),
      description: COALESCE(c.description,''),
      nullable: c.nullable,
      fqn: COALESCE(c.fqn,'')
    }}) AS columns
    RETURN
      elementId(t) AS tid,
      COALESCE(t.db,'') AS db,
      COALESCE(t.schema,'') AS schema,
      COALESCE(t.name,'') AS name,
      COALESCE(t.description,'') AS description,
      COALESCE(t.analyzed_description,'') AS analyzed_description,
      columns AS columns
    ORDER BY schema ASC, name ASC
        """

    # First attempt: apply schema filter if configured
    rows: List[Dict[str, Any]] = []
    used_schema_filter = False
    if schema_filter:
        used_schema_filter = True
        where = base_where + " AND toLower(COALESCE(t.schema,'')) IN $schemas"
        res = await neo4j_session.run(_cypher(where), schemas=schema_filter)
        rows = [r.data() async for r in res]

    # Fallback: if schema filter excluded everything, retry without filtering
    if not rows:
        if used_schema_filter:
            SmartLogger.log(
                "WARNING",
                "text2sql.table_vectorizer.schema_filter.no_match_fallback",
                category="text2sql.table_vectorizer",
                params={"target_db_schemas": schema_filter},
            )
        res = await neo4j_session.run(_cypher(base_where))
        rows = [r.data() async for r in res]

    out: List[_TableItem] = []
    for r in rows:
        cols = r.get("columns") or []
        cleaned_cols = []
        for c in cols:
            if not isinstance(c, dict):
                continue
            if not str(c.get("name") or "").strip():
                continue
            cleaned_cols.append(c)
        out.append(
            _TableItem(
                tid=str(r.get("tid") or ""),
                db=str(r.get("db") or ""),
                schema=str(r.get("schema") or ""),
                name=str(r.get("name") or ""),
                description=str(r.get("description") or ""),
                analyzed_description=str(r.get("analyzed_description") or ""),
                columns=cleaned_cols,
            )
        )
    return out


async def _llm_profile_for_table(
    *,
    item: _TableItem,
    sample_rows: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    gen = get_table_profile_generator()
    obj, _mode = await gen.generate(
        item=TableProfileInput(
            schema=item.schema,
            name=item.name,
            description=item.description,
            analyzed_description=item.analyzed_description,
            columns=list(item.columns or []),
            sample_rows=list(sample_rows or []),
        ),
        react_run_id=None,
    )
    if not isinstance(obj, dict):
        return None
    if not isinstance(obj.get("embedding_text"), str) or not str(obj.get("embedding_text") or "").strip():
        return None
    return obj


async def _write_vectors_to_neo4j(
    neo4j_session,
    *,
    items: Sequence[Dict[str, Any]],
) -> None:
    """
    items: [{tid, vector, profile_json, embedding_text}]
    """
    cypher = """
    UNWIND $items AS item
    MATCH (t) WHERE elementId(t) = item.tid
    SET t.text_to_sql_vector = item.vector,
        t.text_to_sql_profile = item.profile_json,
        t.text_to_sql_embedding_text = item.embedding_text,
        t.text_to_sql_updated_at = datetime()
    """
    res = await neo4j_session.run(cypher, items=list(items))
    await res.consume()


async def ensure_text_to_sql_table_vectors(neo4j_session) -> None:
    """
    Blocking startup job (called from main lifespan).
    Creates Table.text_to_sql_vector when missing.
    """
    if not bool(getattr(settings, "text2sql_vectorize_on_startup", True)):
        return

    started = time.perf_counter()
    try:
        tables = await _fetch_tables_missing_text2sql_vector(neo4j_session)
    except Exception as exc:
        SmartLogger.log(
            "WARNING",
            "text2sql.table_vectorizer.fetch_tables_failed",
            category="text2sql.table_vectorizer",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )
        return

    if not tables:
        SmartLogger.log(
            "INFO",
            "text2sql.table_vectorizer.skip.no_missing",
            category="text2sql.table_vectorizer",
            params={"missing": 0},
        )
        return

    max_tables = int(getattr(settings, "text2sql_vector_max_tables", 0) or 0)
    if max_tables > 0 and len(tables) > max_tables:
        SmartLogger.log(
            "WARNING",
            "text2sql.table_vectorizer.limit_applied",
            category="text2sql.table_vectorizer",
            params={"tables_missing": len(tables), "max_tables": max_tables},
        )
        tables = list(tables)[:max_tables]

    sample_rows_n = max(1, int(getattr(settings, "text2sql_vector_sample_rows", 10)))
    timeout_s = float(getattr(settings, "text2sql_vector_db_timeout_seconds", 2.0))
    llm_conc = max(1, int(getattr(settings, "text2sql_vector_llm_concurrency", 30)))
    embed_batch = max(1, int(getattr(settings, "text2sql_vector_embed_batch_size", 128)))

    SmartLogger.log(
        "INFO",
        "text2sql.table_vectorizer.start",
        category="text2sql.table_vectorizer",
        params={"tables_missing": len(tables), "llm_concurrency": llm_conc, "embed_batch": embed_batch},
    )

    # Light LLM profile generation is handled by TableProfileGenerator (react/generators + react/prompts).
    embedder = create_embedding_client()

    # DB pool (optional):
    # - PostgreSQL: sample rows from source DB.
    # - Non-PostgreSQL (e.g., MindsDB/MySQL endpoint): skip DB sampling and rely on schema metadata.
    db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
    use_db_sampling = db_type in {"postgresql", "postgres"}
    db_pool = None
    if use_db_sampling:
        pool_size = max(2, min(32, llm_conc))
        db_pool = await _open_db_pool(max_size=pool_size)
    else:
        SmartLogger.log(
            "INFO",
            "text2sql.table_vectorizer.db_sampling.skipped_non_postgres",
            category="text2sql.table_vectorizer",
            params={"target_db_type": db_type},
        )

    sem = asyncio.Semaphore(llm_conc)
    profiles: Dict[str, Dict[str, Any]] = {}
    embedding_texts: Dict[str, str] = {}

    async def _one(t: _TableItem) -> None:
        # sample rows
        samples: List[Dict[str, Any]] = []
        if db_pool is not None:
            samples = await _fetch_sample_rows(
                db_pool=db_pool,
                schema=t.schema,
                table_name=t.name,
                limit=sample_rows_n,
                timeout_s=timeout_s,
            )
        async with sem:
            prof = await _llm_profile_for_table(item=t, sample_rows=samples)
        if not prof:
            return
        profiles[t.tid] = prof
        embedding_texts[t.tid] = str(prof.get("embedding_text") or "").strip()

    await asyncio.gather(*[_one(t) for t in tables])
    if db_pool is not None:
        await db_pool.close()

    # Embed in batches
    to_embed: List[Tuple[str, str]] = [(tid, txt) for tid, txt in embedding_texts.items() if txt]
    vectors: Dict[str, List[float]] = {}
    for i in range(0, len(to_embed), embed_batch):
        batch = to_embed[i : i + embed_batch]
        texts = [t[:8000] for _, t in batch]
        vecs = await embedder.embed_batch(texts)
        for (tid, _), vec in zip(batch, vecs):
            if vec:
                vectors[tid] = list(vec)

    # Write to Neo4j in batches
    updated = 0
    payload: List[Dict[str, Any]] = []
    for t in tables:
        tid = t.tid
        vec = vectors.get(tid)
        prof = profiles.get(tid)
        if not vec or not prof:
            continue
        prof_small = {
            "entity_type_guess": prof.get("entity_type_guess", ""),
            "one_line_summary": prof.get("one_line_summary", ""),
            "filters_users_might_use": prof.get("filters_users_might_use", [])[:10],
            "value_signatures": prof.get("value_signatures", [])[:6],
            "search_keywords_ko": prof.get("search_keywords_ko", [])[:25],
        }
        payload.append(
            {
                "tid": tid,
                "vector": vec,
                "profile_json": json.dumps(prof_small, ensure_ascii=False),
                "embedding_text": _truncate(str(prof.get("embedding_text") or ""), 1800),
            }
        )
        if len(payload) >= 200:
            await _write_vectors_to_neo4j(neo4j_session, items=payload)
            updated += len(payload)
            payload = []
    if payload:
        await _write_vectors_to_neo4j(neo4j_session, items=payload)
        updated += len(payload)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    SmartLogger.log(
        "INFO",
        "text2sql.table_vectorizer.done",
        category="text2sql.table_vectorizer",
        params={"tables_missing": len(tables), "tables_profiled": len(profiles), "tables_updated": updated, "elapsed_ms": elapsed_ms},
    )


