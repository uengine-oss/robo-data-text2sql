"""
Text2SQL validity bootstrap (PostgreSQL only).

Goal:
- Mark Neo4j (:Table)/(:Column) nodes with runtime-usable validity flags so build_sql_context
  can filter out:
  - DB-missing objects (schema/table/column drift)
  - effectively empty objects (row_count=0 tables / all-null columns)

Design:
- Fail-open for runtime: build_sql_context uses COALESCE(flag, true)=true.
- For bootstrap, we set text_to_sql_is_valid for *all* tables/columns in configured schemas,
  defaulting to True unless we have strong evidence of invalidity.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import asyncpg

from app.config import settings
from app.smart_logger import SmartLogger


def _schemas_from_settings() -> List[str]:
    schemas = [s.strip() for s in (settings.target_db_schemas or "").split(",") if s.strip()]
    if not schemas:
        schemas = [str(settings.target_db_schema or "").strip() or "public"]
    # store lowercase for comparisons
    return [s.lower() for s in schemas if s]


def _q_ident(name: str) -> str:
    n = str(name or "")
    return '"' + n.replace('"', '""') + '"'


def _now_ms() -> int:
    return int(time.time() * 1000)


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


@dataclass(frozen=True)
class _Neo4jTableKey:
    db: str
    schema_prop: str
    name_prop: str
    schema_l: str
    table_l: str


@dataclass(frozen=True)
class _Neo4jColumnKey:
    fqn: str
    schema_l: str
    table_l: str
    col_l: str


async def has_any_text2sql_validity_flags(*, neo4j_session, schemas: Optional[Sequence[str]] = None) -> bool:
    """
    Returns True if there exists at least one (:Table) node with text_to_sql_is_valid in the target schemas.
    """
    schemas_l = [str(s).strip().lower() for s in (schemas or _schemas_from_settings()) if str(s).strip()]
    # Primary: check within configured schemas.
    try:
        if schemas_l:
            q = """
            MATCH (t:Table)
            WHERE toLower(COALESCE(t.schema,'')) IN $schemas
              AND t.text_to_sql_is_valid IS NOT NULL
            RETURN count(t) AS cnt
            """
            res = await neo4j_session.run(q, schemas=schemas_l)
            row = await res.single()
            if int((row or {}).get("cnt") or 0) > 0:
                return True
    except Exception:
        pass

    # Fallback: if schema config doesn't match the graph, treat "any Table has the flag" as present
    # to avoid repeated blocking work.
    try:
        q2 = """
        MATCH (t:Table)
        WHERE t.text_to_sql_is_valid IS NOT NULL
        RETURN count(t) AS cnt
        """
        res2 = await neo4j_session.run(q2)
        row2 = await res2.single()
        return int((row2 or {}).get("cnt") or 0) > 0
    except Exception:
        return False


async def ensure_text2sql_validity_flags(
    *,
    neo4j_session,
    db_pool: Optional[asyncpg.Pool] = None,
    schemas: Optional[Sequence[str]] = None,
    concurrency: Optional[int] = None,
    query_timeout_s: Optional[float] = None,
    confirm_table_timeout_s: Optional[float] = None,
    confirm_column_timeout_s: Optional[float] = None,
    confirm_max_tables: Optional[int] = None,
    confirm_max_columns: Optional[int] = None,
    max_tables: Optional[int] = None,
    max_columns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute and store validity flags into Neo4j for tables/columns in target schemas.
    Returns a small stats dict for logging/diagnostics.
    """
    started = time.perf_counter()
    schemas_l = [str(s).strip().lower() for s in (schemas or _schemas_from_settings()) if str(s).strip()]
    schema_filter_used: Optional[List[str]] = schemas_l if schemas_l else None

    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        return {"ok": False, "reason": "unsupported_dbms"}

    concurrency_i = max(1, int(concurrency or getattr(settings, "text2sql_validity_bootstrap_concurrency", 6) or 6))
    query_timeout_s = float(query_timeout_s or getattr(settings, "text2sql_validity_bootstrap_query_timeout_s", 2.0) or 2.0)
    confirm_table_timeout_s = float(
        confirm_table_timeout_s
        or getattr(settings, "text2sql_validity_confirm_table_timeout_s", 1.0)
        or 1.0
    )
    confirm_column_timeout_s = float(
        confirm_column_timeout_s
        or getattr(settings, "text2sql_validity_confirm_column_timeout_s", 1.0)
        or 1.0
    )
    # NOTE: confirm_max_* supports 0 => "no limit" (do not use `or` which would treat 0 as falsy).
    if confirm_max_tables is None:
        _cmt = getattr(settings, "text2sql_validity_confirm_max_tables", 200)
        confirm_max_tables = 200 if _cmt is None else _cmt
    if confirm_max_columns is None:
        _cmc = getattr(settings, "text2sql_validity_confirm_max_columns", 600)
        confirm_max_columns = 600 if _cmc is None else _cmc
    confirm_max_tables = int(confirm_max_tables)
    confirm_max_columns = int(confirm_max_columns)
    max_tables = int(max_tables or getattr(settings, "text2sql_validity_bootstrap_max_tables", 0) or 0)
    max_columns = int(max_columns or getattr(settings, "text2sql_validity_bootstrap_max_columns", 0) or 0)

    created_pool = False
    if db_pool is None:
        db_pool = await _open_db_pool(max_size=concurrency_i)
        created_pool = True

    try:
        # 1) Fetch target Table / Column nodes from Neo4j.
        base_match = "MATCH (t:Table)\n"
        where_clause = "WHERE toLower(COALESCE(t.schema,'')) IN $schemas\n" if schema_filter_used else ""
        table_q = (
            base_match
            + where_clause
            + """
        RETURN
          COALESCE(t.db,'') AS db,
          COALESCE(t.schema,'') AS schema_prop,
          COALESCE(t.name,'') AS name_prop,
          COALESCE(t.original_name, t.name, '') AS display_name
        ORDER BY schema_prop, name_prop
        """
        )
        if max_tables > 0:
            table_q += "\nLIMIT $limit"

        res = await neo4j_session.run(
            table_q,
            schemas=(schema_filter_used or []),
            limit=int(max_tables) if max_tables > 0 else None,
        )
        table_rows = await res.data()
        table_keys: List[_Neo4jTableKey] = []
        for r in table_rows:
            db = str(r.get("db") or "postgres").strip().lower() or "postgres"
            schema_prop = str(r.get("schema_prop") or "").strip()
            name_prop = str(r.get("name_prop") or "").strip()
            display = str(r.get("display_name") or name_prop or "").strip()
            if not schema_prop or not name_prop:
                continue
            schema_l = schema_prop.lower()
            table_l = display.lower()
            table_keys.append(_Neo4jTableKey(db=db, schema_prop=schema_prop, name_prop=name_prop, schema_l=schema_l, table_l=table_l))

        # Fallback: if schema filter yielded 0 tables, retry without schema restriction (best-effort).
        if not table_keys and schemas_l:
            SmartLogger.log(
                "WARNING",
                "text2sql.validity_bootstrap.schema_filter.no_match_fallback",
                category="text2sql.validity_bootstrap",
                params={"schemas": schemas_l},
                max_inline_chars=0,
            )
            table_q2 = """
            MATCH (t:Table)
            RETURN
              COALESCE(t.db,'') AS db,
              COALESCE(t.schema,'') AS schema_prop,
              COALESCE(t.name,'') AS name_prop,
              COALESCE(t.original_name, t.name, '') AS display_name
            ORDER BY schema_prop, name_prop
            """
            if max_tables > 0:
                table_q2 += "\nLIMIT $limit"
            res_f = await neo4j_session.run(table_q2, limit=int(max_tables) if max_tables > 0 else None)
            table_rows_f = await res_f.data()
            for r in table_rows_f:
                db = str(r.get("db") or "postgres").strip().lower() or "postgres"
                schema_prop = str(r.get("schema_prop") or "").strip()
                name_prop = str(r.get("name_prop") or "").strip()
                display = str(r.get("display_name") or name_prop or "").strip()
                if not schema_prop or not name_prop:
                    continue
                schema_l = schema_prop.lower()
                table_l = display.lower()
                table_keys.append(
                    _Neo4jTableKey(
                        db=db,
                        schema_prop=schema_prop,
                        name_prop=name_prop,
                        schema_l=schema_l,
                        table_l=table_l,
                    )
                )
            # When we fallback to "no schema filter" for tables, do the same for columns.
            schema_filter_used = None

        col_q = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
        RETURN
          COALESCE(c.fqn,'') AS fqn,
          COALESCE(c.name,'') AS column_name,
          COALESCE(t.schema,'') AS schema_prop,
          COALESCE(t.name,'') AS name_prop,
          COALESCE(t.original_name, t.name, '') AS display_name
        ORDER BY fqn
        """
        if max_columns > 0:
            col_q += "\nLIMIT $limit"
        res2 = await neo4j_session.run(
            col_q,
            schemas=schema_filter_used,
            limit=int(max_columns) if max_columns > 0 else None,
        )
        col_rows = await res2.data()
        col_keys: List[_Neo4jColumnKey] = []
        for r in col_rows:
            fqn = str(r.get("fqn") or "").strip()
            if not fqn or fqn.count(".") < 2:
                # Fallback: derive from schema/table display + column name if possible
                schema_prop = str(r.get("schema_prop") or "").strip()
                display = str(r.get("display_name") or "").strip()
                coln = str(r.get("column_name") or "").strip()
                if schema_prop and display and coln:
                    fqn = f"{schema_prop}.{display}.{coln}".lower()
            parts = [p.strip() for p in fqn.split(".") if p.strip()]
            if len(parts) < 3:
                continue
            schema_l, table_l, col_l = parts[0].lower(), parts[1].lower(), parts[2].lower()
            col_keys.append(_Neo4jColumnKey(fqn=fqn.lower(), schema_l=schema_l, table_l=table_l, col_l=col_l))

        # De-dup columns by fqn
        seen_fqn = set()
        col_keys_dedup: List[_Neo4jColumnKey] = []
        for ck in col_keys:
            if ck.fqn in seen_fqn:
                continue
            seen_fqn.add(ck.fqn)
            col_keys_dedup.append(ck)
        col_keys = col_keys_dedup

        # Restrict DB metadata queries to the tables we actually have in Neo4j (bounded cost).
        fqn_list = sorted({f"{t.schema_l}.{t.table_l}" for t in table_keys if t.schema_l and t.table_l})

        # 2) Fetch DB metadata in batch.
        async with db_pool.acquire() as conn:
            # 2.1) information_schema tables + types
            rows_tbl = await asyncio.wait_for(
                conn.fetch(
                    """
                    SELECT table_schema, table_name, table_type
                    FROM information_schema.tables
                    WHERE (lower(table_schema) || '.' || lower(table_name)) = ANY($1::text[])
                      AND table_type IN ('BASE TABLE', 'VIEW')
                    """,
                    fqn_list if fqn_list else ["__none__.__none__"],
                ),
                timeout=query_timeout_s,
            )
            db_table_exists: Dict[Tuple[str, str], Dict[str, str]] = {}
            for r in rows_tbl:
                sch = str(r.get("table_schema") or "")
                nm = str(r.get("table_name") or "")
                typ = str(r.get("table_type") or "")
                key = (sch.lower(), nm.lower())
                # keep first
                db_table_exists.setdefault(key, {"schema": sch, "name": nm, "type": typ})

            # 2.2) information_schema columns
            rows_col = await asyncio.wait_for(
                conn.fetch(
                    """
                    SELECT table_schema, table_name, column_name
                    FROM information_schema.columns
                    WHERE (lower(table_schema) || '.' || lower(table_name)) = ANY($1::text[])
                    """,
                    fqn_list if fqn_list else ["__none__.__none__"],
                ),
                timeout=query_timeout_s,
            )
            db_col_exists: Dict[Tuple[str, str, str], Dict[str, str]] = {}
            for r in rows_col:
                sch = str(r.get("table_schema") or "")
                tb = str(r.get("table_name") or "")
                cn = str(r.get("column_name") or "")
                key = (sch.lower(), tb.lower(), cn.lower())
                db_col_exists.setdefault(key, {"schema": sch, "table": tb, "column": cn})

        # 2.3) pg_stat_user_tables row count estimates (restricted)
        row_est: Dict[Tuple[str, str], int] = {}
        if fqn_list:
            async with db_pool.acquire() as conn:
                rows_stats = await asyncio.wait_for(
                    conn.fetch(
                        """
                        SELECT
                            lower(n.nspname) AS schema_l,
                            lower(c.relname) AS table_l,
                            COALESCE(s.n_live_tup, 0)::bigint AS row_count_est
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                        WHERE (lower(n.nspname) || '.' || lower(c.relname)) = ANY($1::text[])
                        """,
                        fqn_list,
                    ),
                    timeout=query_timeout_s,
                )
                for r in rows_stats:
                    row_est[(str(r.get("schema_l") or ""), str(r.get("table_l") or ""))] = int(r.get("row_count_est") or 0)

        # 2.4) pg_stats null_frac estimates for columns (restricted)
        null_frac: Dict[Tuple[str, str, str], float] = {}
        if fqn_list:
            async with db_pool.acquire() as conn:
                rows_nf = await asyncio.wait_for(
                    conn.fetch(
                        """
                        SELECT
                            lower(schemaname) AS schema_l,
                            lower(tablename) AS table_l,
                            lower(attname) AS col_l,
                            null_frac AS null_frac
                        FROM pg_stats
                        WHERE (lower(schemaname) || '.' || lower(tablename)) = ANY($1::text[])
                        """,
                        fqn_list,
                    ),
                    timeout=query_timeout_s,
                )
                for r in rows_nf:
                    try:
                        null_frac[(str(r.get("schema_l") or ""), str(r.get("table_l") or ""), str(r.get("col_l") or ""))] = float(
                            r.get("null_frac") if r.get("null_frac") is not None else 0.0
                        )
                    except Exception:
                        continue

        # 3) Confirm phase (only suspects)
        # Tables: if exists and base table and row_count_est == 0 -> confirm with LIMIT 1.
        suspects_tables: List[Tuple[str, str]] = []
        for tk in table_keys:
            meta = db_table_exists.get((tk.schema_l, tk.table_l))
            if not meta:
                continue
            if str(meta.get("type") or "").upper() == "VIEW":
                continue
            if int(row_est.get((tk.schema_l, tk.table_l), 0)) == 0:
                suspects_tables.append((tk.schema_l, tk.table_l))
        if int(confirm_max_tables) > 0:
            suspects_tables = suspects_tables[: int(confirm_max_tables)]

        async def _confirm_table_has_rows(schema_l: str, table_l: str) -> Optional[bool]:
            meta = db_table_exists.get((schema_l, table_l))
            if not meta:
                return None
            schema_real = meta["schema"]
            table_real = meta["name"]
            sql = f"SELECT 1 AS one FROM {_q_ident(schema_real)}.{_q_ident(table_real)} LIMIT 1"
            try:
                async with db_pool.acquire() as conn:
                    row = await asyncio.wait_for(conn.fetchrow(sql), timeout=float(confirm_table_timeout_s))
                return bool(row)
            except Exception:
                return None

        table_has_rows_confirmed: Dict[Tuple[str, str], Optional[bool]] = {}
        if suspects_tables:
            # bounded concurrency
            sem = asyncio.Semaphore(concurrency_i)

            async def _run_one(st: Tuple[str, str]) -> None:
                async with sem:
                    table_has_rows_confirmed[st] = await _confirm_table_has_rows(st[0], st[1])

            await asyncio.gather(*[_run_one(st) for st in suspects_tables], return_exceptions=True)

        # Columns: if exists and null_frac_est >= ~1.0 -> confirm with IS NOT NULL LIMIT 1.
        suspects_cols: List[Tuple[str, str, str]] = []
        for ck in col_keys:
            key3 = (ck.schema_l, ck.table_l, ck.col_l)
            if key3 not in db_col_exists:
                continue
            nf = null_frac.get(key3)
            if nf is None:
                continue
            if float(nf) >= 0.999999:
                suspects_cols.append(key3)
        suspects_cols = suspects_cols[: max(0, int(confirm_max_columns))]

        async def _confirm_col_has_nonnull(schema_l: str, table_l: str, col_l: str) -> Optional[bool]:
            meta = db_col_exists.get((schema_l, table_l, col_l))
            tmeta = db_table_exists.get((schema_l, table_l))
            if not meta or not tmeta:
                return None
            schema_real = tmeta["schema"]
            table_real = tmeta["name"]
            col_real = meta["column"]
            sql = (
                f"SELECT 1 AS one FROM {_q_ident(schema_real)}.{_q_ident(table_real)} "
                f"WHERE {_q_ident(col_real)} IS NOT NULL LIMIT 1"
            )
            try:
                async with db_pool.acquire() as conn:
                    row = await asyncio.wait_for(conn.fetchrow(sql), timeout=float(confirm_column_timeout_s))
                return bool(row)
            except Exception:
                return None

        col_has_nonnull_confirmed: Dict[Tuple[str, str, str], Optional[bool]] = {}
        if suspects_cols:
            sem2 = asyncio.Semaphore(concurrency_i)

            async def _run_one_c(sc: Tuple[str, str, str]) -> None:
                async with sem2:
                    col_has_nonnull_confirmed[sc] = await _confirm_col_has_nonnull(sc[0], sc[1], sc[2])

            await asyncio.gather(*[_run_one_c(sc) for sc in suspects_cols], return_exceptions=True)

        # 4) Build updates for Neo4j.
        validated_at_ms = _now_ms()
        table_updates: List[Dict[str, Any]] = []
        by_table_valid: Dict[Tuple[str, str], bool] = {}

        for tk in table_keys:
            meta = db_table_exists.get((tk.schema_l, tk.table_l))
            if not meta:
                is_valid = False
                reason = "missing_in_db"
                table_type = ""
                rowc = 0
                db_exists = False
            else:
                db_exists = True
                table_type = str(meta.get("type") or "")
                rowc = int(row_est.get((tk.schema_l, tk.table_l), 0))
                # Views: do not invalidate by row_count_est (unreliable); keep valid if exists.
                if table_type.upper() == "VIEW":
                    is_valid = True
                    reason = ""
                else:
                    confirmed = table_has_rows_confirmed.get((tk.schema_l, tk.table_l))
                    if rowc > 0:
                        is_valid = True
                        reason = ""
                    elif confirmed is True:
                        is_valid = True
                        reason = ""
                    elif confirmed is False:
                        is_valid = False
                        reason = "empty_table_confirmed"
                    else:
                        # row_count_est == 0 and we couldn't confirm non-emptiness -> fail-closed
                        # so empty tables are not used by Text2SQL context building.
                        is_valid = False
                        reason = "empty_table_or_confirm_failed"

            by_table_valid[(tk.schema_l, tk.table_l)] = bool(is_valid)
            table_updates.append(
                {
                    "db": tk.db,
                    "schema_prop": tk.schema_prop,
                    "name_prop": tk.name_prop,
                    "is_valid": bool(is_valid),
                    "db_exists": bool(db_exists),
                    "invalid_reason": str(reason or ""),
                    "validated_at_ms": int(validated_at_ms),
                    "row_count_est": int(rowc),
                    "table_type": str(table_type or ""),
                }
            )

        col_updates: List[Dict[str, Any]] = []
        for ck in col_keys:
            meta = db_col_exists.get((ck.schema_l, ck.table_l, ck.col_l))
            if not meta:
                is_valid = False
                reason = "missing_in_db"
                db_exists = False
                nf = None
                confirmed = None
            else:
                db_exists = True
                nf = null_frac.get((ck.schema_l, ck.table_l, ck.col_l))
                confirmed = col_has_nonnull_confirmed.get((ck.schema_l, ck.table_l, ck.col_l))
                # If parent table was strongly invalid (confirmed empty), invalidate column too.
                if not by_table_valid.get((ck.schema_l, ck.table_l), True):
                    is_valid = False
                    reason = "parent_table_invalid"
                else:
                    if confirmed is True:
                        is_valid = True
                        reason = ""
                    elif confirmed is False:
                        is_valid = False
                        reason = "all_null_confirmed"
                    else:
                        # stats-only strong signal can be wrong; treat as valid unless confirmed false.
                        is_valid = True
                        reason = ""

            col_updates.append(
                {
                    "fqn": ck.fqn,
                    "is_valid": bool(is_valid),
                    "db_exists": bool(db_exists),
                    "invalid_reason": str(reason or ""),
                    "validated_at_ms": int(validated_at_ms),
                    "null_frac_est": (float(nf) if nf is not None else None),
                    "has_nonnull_confirmed": (bool(confirmed) if confirmed is not None else None),
                }
            )

        # 5) Write updates to Neo4j (batched).
        async def _write_table_batch(rows: List[Dict[str, Any]]) -> None:
            if not rows:
                return
            q = """
            UNWIND $rows AS r
            MATCH (t:Table {db: r.db, schema: r.schema_prop, name: r.name_prop})
            SET t.text_to_sql_is_valid = r.is_valid,
                t.text_to_sql_db_exists = r.db_exists,
                t.text_to_sql_invalid_reason = r.invalid_reason,
                t.text_to_sql_validated_at_ms = r.validated_at_ms,
                t.text_to_sql_row_count_est = r.row_count_est,
                t.text_to_sql_table_type = r.table_type
            """
            res = await neo4j_session.run(q, rows=rows)
            await res.consume()

        async def _write_col_batch(rows: List[Dict[str, Any]]) -> None:
            if not rows:
                return
            q = """
            UNWIND $rows AS r
            MATCH (c:Column {fqn: r.fqn})
            SET c.text_to_sql_is_valid = r.is_valid,
                c.text_to_sql_db_exists = r.db_exists,
                c.text_to_sql_invalid_reason = r.invalid_reason,
                c.text_to_sql_validated_at_ms = r.validated_at_ms,
                c.text_to_sql_null_frac_est = r.null_frac_est,
                c.text_to_sql_has_nonnull_confirmed = r.has_nonnull_confirmed
            """
            res = await neo4j_session.run(q, rows=rows)
            await res.consume()

        # Write in reasonable chunk sizes to avoid very large parameters.
        chunk = 500
        for i in range(0, len(table_updates), chunk):
            await _write_table_batch(table_updates[i : i + chunk])
        for i in range(0, len(col_updates), chunk):
            await _write_col_batch(col_updates[i : i + chunk])

        # 5.1) Final backfill for nodes that could not be addressed by key-based updates.
        # This mainly protects against malformed/legacy Column nodes (e.g., missing/invalid fqn).
        where_tables = "WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)"
        where_cols = "WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)"
        q_backfill_tables = f"""
        MATCH (t:Table)
        {where_tables}
          AND (t.text_to_sql_is_valid IS NULL OR t.text_to_sql_db_exists IS NULL)
        SET t.text_to_sql_is_valid = COALESCE(t.text_to_sql_is_valid, true),
            t.text_to_sql_db_exists = COALESCE(t.text_to_sql_db_exists, true),
            t.text_to_sql_invalid_reason = COALESCE(t.text_to_sql_invalid_reason, ''),
            t.text_to_sql_validated_at_ms = COALESCE(t.text_to_sql_validated_at_ms, $validated_at_ms)
        RETURN count(t) AS cnt
        """
        q_backfill_cols = f"""
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        {where_cols}
          AND (c.text_to_sql_is_valid IS NULL OR c.text_to_sql_db_exists IS NULL)
        SET c.text_to_sql_is_valid = COALESCE(c.text_to_sql_is_valid, COALESCE(t.text_to_sql_is_valid, true)),
            c.text_to_sql_db_exists = COALESCE(c.text_to_sql_db_exists, COALESCE(t.text_to_sql_db_exists, true)),
            c.text_to_sql_invalid_reason = COALESCE(
                c.text_to_sql_invalid_reason,
                CASE WHEN COALESCE(t.text_to_sql_is_valid, true) THEN '' ELSE 'parent_table_invalid' END
            ),
            c.text_to_sql_validated_at_ms = COALESCE(c.text_to_sql_validated_at_ms, $validated_at_ms)
        RETURN count(c) AS cnt
        """
        backfilled_tables = 0
        backfilled_columns = 0
        try:
            bf_t = await neo4j_session.run(
                q_backfill_tables,
                schemas=schema_filter_used,
                validated_at_ms=int(validated_at_ms),
            )
            row_t = await bf_t.single()
            backfilled_tables = int((row_t or {}).get("cnt") or 0)
        except Exception:
            backfilled_tables = 0
        try:
            bf_c = await neo4j_session.run(
                q_backfill_cols,
                schemas=schema_filter_used,
                validated_at_ms=int(validated_at_ms),
            )
            row_c = await bf_c.single()
            backfilled_columns = int((row_c or {}).get("cnt") or 0)
        except Exception:
            backfilled_columns = 0

        invalid_tables = len([x for x in table_updates if not x["is_valid"]])
        invalid_cols = len([x for x in col_updates if not x["is_valid"]])

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        out = {
            "ok": True,
            "schemas": schemas_l,
            "tables_total": len(table_updates),
            "columns_total": len(col_updates),
            "tables_invalid": int(invalid_tables),
            "columns_invalid": int(invalid_cols),
            "tables_confirmed": len([k for k, v in table_has_rows_confirmed.items() if v is not None]),
            "columns_confirmed": len([k for k, v in col_has_nonnull_confirmed.items() if v is not None]),
            "tables_backfilled": int(backfilled_tables),
            "columns_backfilled": int(backfilled_columns),
            "elapsed_ms": float(elapsed_ms),
        }

        SmartLogger.log(
            "INFO",
            "text2sql.validity_bootstrap.done",
            category="text2sql.validity_bootstrap",
            params=out,
            max_inline_chars=0,
        )
        return out
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "text2sql.validity_bootstrap.error",
            category="text2sql.validity_bootstrap",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )
        return {"ok": False, "reason": str(exc), "traceback": traceback.format_exc()}
    finally:
        if created_pool and db_pool is not None:
            try:
                await db_pool.close()
            except Exception:
                pass


__all__ = [
    "has_any_text2sql_validity_flags",
    "ensure_text2sql_validity_flags",
]


