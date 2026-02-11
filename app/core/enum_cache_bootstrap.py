"""
Enum cache bootstrap (best-effort).

Goal:
- Populate Column.enum_values + Column.cardinality in Neo4j for low-cardinality columns.
- Designed for cold-start: can be slow on first run, but makes runtime value hinting fast.

Notes:
- We intentionally avoid domain-hardcoding. We use generic name patterns and dtypes.
- We only cache when distinct count <= max_values.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import asyncpg

from app.config import settings
from app.core.sql_exec import SQLExecutor
from app.smart_logger import SmartLogger


_NAME_HINT_RE = re.compile(r"(name|nm|title|code|cd|id|sn)$|(^|_)(name|nm|title|code|cd|id|sn)($|_)", re.I)
_TEXT_DTYPE_RE = re.compile(r"(char|text|varchar)", re.I)


@dataclass(frozen=True)
class EnumCacheBootstrapResult:
    schema: str
    scanned_columns: int
    cached_columns: int
    skipped_high_cardinality: int
    skipped_empty: int
    skipped_missing: int
    errors: int
    elapsed_ms: float


def _norm_schema(s: str) -> str:
    return str(s or "").strip().lower()


def _norm_fqn(s: str) -> str:
    return str(s or "").strip().lower()

def _quote_ident_local(name: str) -> str:
    """
    Very small local identifier-quoter for Postgres identifiers.
    - Escapes embedded double quotes by doubling them.
    - Always returns a double-quoted identifier.
    """
    n = str(name or "")
    return '"' + n.replace('"', '""') + '"'


def _pg_sqlstate(exc: BaseException) -> str:
    # asyncpg exceptions typically expose .sqlstate
    return str(getattr(exc, "sqlstate", "") or "")


def _is_pg_missing_object_error(exc: BaseException) -> bool:
    """
    Treat missing schema/table/column as a non-fatal, expected condition.
    SQLSTATE:
    - 42P01: undefined_table
    - 42703: undefined_column
    - 3F000: invalid_schema_name
    """
    sqlstate = _pg_sqlstate(exc)
    return sqlstate in {"42P01", "42703", "3F000"}


def _is_missing_object_error(exc: BaseException) -> bool:
    if _is_pg_missing_object_error(exc):
        return True
    msg = str(exc or "").lower()
    needles = [
        "unknown table",
        "unknown column",
        "does not exist",
        "doesn't exist",
        "no such table",
        "no such column",
        "invalid schema",
        "table or view does not exist",
        "column not found",
        "object not found",
    ]
    return any(n in msg for n in needles)


def _ident_variants(schema: str, table: str, column: str) -> List[Tuple[str, str, str]]:
    """
    Generate a small set of identifier casing variants to survive:
    - Neo4j metadata stored as UPPERCASE (quoted) while Postgres objects are lowercase (unquoted)
    - or the opposite (objects created with quoted uppercase identifiers)
    """
    s = str(schema or "").strip()
    t = str(table or "").strip()
    c = str(column or "").strip()
    out: List[Tuple[str, str, str]] = []
    for ss, tt, cc in [
        (s, t, c),
        (s.lower(), t.lower(), c.lower()),
        (s.upper(), t.upper(), c.upper()),
    ]:
        if not (ss and tt and cc):
            continue
        if (ss, tt, cc) not in out:
            out.append((ss, tt, cc))
    return out


def _should_consider_column(*, column_name: str, dtype: str, name_hint_only: bool) -> bool:
    cn = str(column_name or "").strip()
    dt = str(dtype or "").strip()
    if not cn:
        return False
    if name_hint_only:
        return bool(_NAME_HINT_RE.search(cn))
    # default: (name hint) OR (text-ish dtype)
    return bool(_NAME_HINT_RE.search(cn) or _TEXT_DTYPE_RE.search(dt))


async def _fetch_candidate_columns_from_neo4j(
    *,
    neo4j_session,
    schema: str,
    limit: int,
) -> List[Dict[str, str]]:
    schema_l = _norm_schema(schema)
    q = """
    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
    WHERE toLower(COALESCE(t.schema,'')) = $schema
    RETURN
      COALESCE(t.datasource,'') AS table_datasource,
      COALESCE(t.schema,'') AS table_schema,
      COALESCE(t.name,'') AS table_name,
      COALESCE(c.name,'') AS column_name,
      COALESCE(c.dtype,'') AS dtype,
      COALESCE(c.fqn,'') AS fqn,
      COALESCE(c.enum_values,'') AS enum_values
    ORDER BY table_schema, table_name, column_name
    LIMIT $limit
    """
    res = await neo4j_session.run(q, schema=schema_l, limit=int(max(1, limit)))
    rows = await res.data()
    out: List[Dict[str, str]] = []
    for r in rows:
        out.append(
            {
                "table_datasource": str(r.get("table_datasource") or ""),
                "table_schema": str(r.get("table_schema") or ""),
                "table_name": str(r.get("table_name") or ""),
                "column_name": str(r.get("column_name") or ""),
                "dtype": str(r.get("dtype") or ""),
                "fqn": str(r.get("fqn") or ""),
                "enum_values": str(r.get("enum_values") or ""),
            }
        )
    return out


async def _cache_one_column(
    *,
    neo4j_session,
    neo4j_session_lock: Optional[asyncio.Lock],
    db_conn: Union[asyncpg.Connection, Any],
    db_conn_lock: Optional[asyncio.Lock],
    table_schema: str,
    table_name: str,
    column_name: str,
    column_fqn: str,
    max_values: int,
    query_timeout_s: float,
    datasource: Optional[str],
) -> Tuple[str, str]:
    """
    Returns: (status, detail)
    status: cached|skipped_high_cardinality|skipped_empty|skipped_missing
    """
    last_missing_exc: Optional[BaseException] = None
    last_exc: Optional[BaseException] = None
    rows = None

    datasource_s = str(datasource or "").strip()
    db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
    if not datasource_s and db_type in {"mysql", "mariadb"}:
        return "error", "missing_datasource_from_table_node"

    async def _fetch_rows(sql: str, *, limit_rows: int) -> List[Dict[str, Any]]:
        if datasource_s:
            # MindsDB passthrough mode: keep inner SQL in upstream DB dialect.
            ds_quoted = datasource_s.replace("`", "``")
            sql_exec = f"SELECT * FROM `{ds_quoted}` (\n{sql}\n)"
            executor = SQLExecutor()

            async def _run_one(conn_obj: Any) -> List[Dict[str, Any]]:
                preview = await executor.preview_query(
                    conn_obj,
                    sql_exec,
                    row_limit=max(1, int(limit_rows)),
                    timeout=float(query_timeout_s),
                )
                out_rows: List[Dict[str, Any]] = []
                for row in list(preview.get("rows") or []):
                    if not isinstance(row, list) or len(row) < 2:
                        continue
                    out_rows.append({"value": row[0], "count": row[1]})
                return out_rows

            if hasattr(db_conn, "acquire"):
                async with cast(Any, db_conn).acquire() as conn:
                    return await _run_one(conn)
            if db_conn_lock is not None:
                async with db_conn_lock:
                    return await _run_one(db_conn)
            return await _run_one(db_conn)

        # PostgreSQL direct mode (asyncpg)
        if hasattr(db_conn, "acquire"):
            async with cast(Any, db_conn).acquire() as conn:
                rows_obj = await asyncio.wait_for(conn.fetch(sql), timeout=float(query_timeout_s))
        elif db_conn_lock is not None:
            async with db_conn_lock:
                rows_obj = await asyncio.wait_for(
                    cast(asyncpg.Connection, db_conn).fetch(sql), timeout=float(query_timeout_s)
                )
        else:
            rows_obj = await asyncio.wait_for(
                cast(asyncpg.Connection, db_conn).fetch(sql), timeout=float(query_timeout_s)
            )
        out_rows2: List[Dict[str, Any]] = []
        for r in rows_obj:
            out_rows2.append({"value": r.get("value"), "count": r.get("count")})
        return out_rows2

    # Try a few identifier casing variants to reduce false negatives due to quoting+case.
    for (s, t, c) in _ident_variants(table_schema, table_name, column_name):
        schema_ident = _quote_ident_local(s)
        table_ident = _quote_ident_local(t)
        col_ident = _quote_ident_local(c)

        # Get top values (up to max_values + 1) to detect high-cardinality fast.
        # MindsDB path uses CAST(... AS CHAR) to reduce type-related GROUP BY failures.
        if datasource_s:
            sql = f"""
            SELECT CAST({col_ident} AS CHAR) AS value, COUNT(*) AS count
            FROM {schema_ident}.{table_ident}
            WHERE {col_ident} IS NOT NULL
            GROUP BY CAST({col_ident} AS CHAR)
            ORDER BY count DESC
            LIMIT {int(max_values) + 1}
            """
        else:
            sql = f"""
            SELECT {col_ident} AS value, COUNT(*) AS count
            FROM {schema_ident}.{table_ident}
            WHERE {col_ident} IS NOT NULL
            GROUP BY {col_ident}
            ORDER BY count DESC
            LIMIT {int(max_values) + 1}
            """
        try:
            rows = await _fetch_rows(sql, limit_rows=int(max_values) + 1)
            last_exc = None
            last_missing_exc = None
            break
        except Exception as exc:
            last_exc = exc
            if _is_missing_object_error(exc):
                last_missing_exc = exc
                # try next variant
                continue
            return "error", f"db_query_failed: {repr(exc)}"

    if rows is None:
        if last_missing_exc is not None:
            return "skipped_missing", f"missing_schema_table_or_column: {repr(last_missing_exc)}"
        if last_exc is not None:
            return "error", f"db_query_failed: {repr(last_exc)}"
        return "error", "db_query_failed: unknown"

    if not rows:
        return "skipped_empty", "no_values"

    if len(rows) > int(max_values):
        return "skipped_high_cardinality", f"distinct_gt_{max_values}"

    enum_values: List[Dict[str, Any]] = []
    for row in rows[: int(max_values)]:
        v = row.get("value")
        if v is None:
            continue
        enum_values.append({"value": str(v), "count": int(row.get("count") or 0)})

    cardinality = len(enum_values)
    if cardinality <= 0:
        return "skipped_empty", "no_non_null_values"

    # Store into Neo4j (JSON string, consistent with cache router).
    cypher = """
    MATCH (c:Column {fqn: $fqn})
    SET c.enum_values = $enum_values,
        c.cardinality = $cardinality,
        c.enum_cached_at = datetime()
    RETURN c.fqn AS fqn
    """
    try:
        # Neo4j AsyncSession is not safe for concurrent run() calls.
        # Serialize writes to avoid: "read() called while another coroutine is already waiting..."
        if neo4j_session_lock is not None:
            async with neo4j_session_lock:
                res = await neo4j_session.run(
                    cypher,
                    fqn=_norm_fqn(column_fqn),
                    enum_values=json.dumps(enum_values, ensure_ascii=False),
                    cardinality=int(cardinality),
                )
                try:
                    await res.consume()
                except Exception:
                    pass
        else:
            res = await neo4j_session.run(
                cypher,
                fqn=_norm_fqn(column_fqn),
                enum_values=json.dumps(enum_values, ensure_ascii=False),
                cardinality=int(cardinality),
            )
            try:
                await res.consume()
            except Exception:
                pass
    except Exception as exc:
        return "error", f"neo4j_update_failed: {repr(exc)}"

    return "cached", f"cardinality={cardinality}"


async def ensure_enum_cache_for_schema(
    *,
    neo4j_session,
    db_conn: Union[asyncpg.Connection, Any],
    schema: str,
    max_columns: int = 5000,
    max_values: int = 200,
    query_timeout_s: float = 2.0,
    concurrency: int = 6,
    name_hint_only: bool = False,
    include_fqns: Optional[Sequence[str]] = None,
    skip_if_cached: bool = True,
    datasource: Optional[str] = None,
) -> EnumCacheBootstrapResult:
    """
    Populate enum cache for low-cardinality columns in the given schema.

    - If include_fqns is provided, only those columns are considered.
    - If skip_if_cached is True, columns with existing enum_values are skipped.
    """
    started = time.perf_counter()
    schema_l = _norm_schema(schema)
    include_fqns_l = {_norm_fqn(x) for x in (include_fqns or []) if _norm_fqn(x)}

    SmartLogger.log(
        "INFO",
        "enum_cache.bootstrap.start",
        category="enum_cache.bootstrap",
        params={
            "schema": schema,
            "max_columns": int(max_columns),
            "max_values": int(max_values),
            "query_timeout_s": float(query_timeout_s),
            "concurrency": int(concurrency),
            "name_hint_only": bool(name_hint_only),
            "include_fqns_count": len(include_fqns_l),
            "skip_if_cached": bool(skip_if_cached),
            "datasource": (str(datasource or "").strip() or None),
        },
        max_inline_chars=0,
    )

    candidates = await _fetch_candidate_columns_from_neo4j(
        neo4j_session=neo4j_session,
        schema=schema_l,
        limit=int(max_columns),
    )

    # Filter candidates (generic, avoid domain hardcoding).
    filtered: List[Dict[str, str]] = []
    for c in candidates:
        fqn = _norm_fqn(c.get("fqn") or "")
        if include_fqns_l and fqn not in include_fqns_l:
            continue
        if skip_if_cached and str(c.get("enum_values") or "").strip():
            continue
        if not _should_consider_column(
            column_name=str(c.get("column_name") or ""),
            dtype=str(c.get("dtype") or ""),
            name_hint_only=bool(name_hint_only),
        ):
            continue
        filtered.append(c)

    sem = asyncio.Semaphore(max(1, int(concurrency)))
    # If caller passed a single asyncpg.Connection and requests concurrency>1,
    # we must serialize DB fetches to avoid asyncpg.InterfaceError:
    # "cannot perform operation: another operation is in progress".
    db_conn_lock: Optional[asyncio.Lock] = None
    if not hasattr(db_conn, "acquire") and int(concurrency) > 1:
        db_conn_lock = asyncio.Lock()
    # Neo4j AsyncSession also must not be used concurrently.
    neo4j_session_lock: Optional[asyncio.Lock] = None
    if int(concurrency) > 1:
        neo4j_session_lock = asyncio.Lock()

    cached = 0
    skipped_hi = 0
    skipped_empty = 0
    skipped_missing = 0
    errors = 0

    async def _run_one(c: Dict[str, str]) -> None:
        nonlocal cached, skipped_hi, skipped_empty, skipped_missing, errors
        async with sem:
            candidate_ds = str(c.get("table_datasource") or "").strip()
            datasource_effective = candidate_ds or (str(datasource or "").strip() or None)
            status, detail = await _cache_one_column(
                neo4j_session=neo4j_session,
                neo4j_session_lock=neo4j_session_lock,
                db_conn=db_conn,
                db_conn_lock=db_conn_lock,
                table_schema=str(c.get("table_schema") or ""),
                table_name=str(c.get("table_name") or ""),
                column_name=str(c.get("column_name") or ""),
                column_fqn=str(c.get("fqn") or ""),
                max_values=int(max_values),
                query_timeout_s=float(query_timeout_s),
                datasource=datasource_effective,
            )
            if status == "cached":
                cached += 1
            elif status == "skipped_high_cardinality":
                skipped_hi += 1
            elif status == "skipped_empty":
                skipped_empty += 1
            elif status == "skipped_missing":
                skipped_missing += 1
            else:
                errors += 1
            # Keep logs compact (sample only).
            if errors <= 3 and status == "error":
                SmartLogger.log(
                    "WARNING",
                    "enum_cache.bootstrap.column.error",
                    category="enum_cache.bootstrap",
                    params={
                        "schema": schema_l,
                        "fqn": str(c.get("fqn") or ""),
                        "table": f"{c.get('table_schema','')}.{c.get('table_name','')}",
                        "column": c.get("column_name"),
                        "detail": detail[:200],
                    },
                    max_inline_chars=0,
                )
            if skipped_missing <= 3 and status == "skipped_missing":
                SmartLogger.log(
                    "INFO",
                    "enum_cache.bootstrap.column.skip_missing",
                    category="enum_cache.bootstrap",
                    params={
                        "schema": schema_l,
                        "fqn": str(c.get("fqn") or ""),
                        "table": f"{c.get('table_schema','')}.{c.get('table_name','')}",
                        "column": c.get("column_name"),
                        "detail": detail[:200],
                    },
                    max_inline_chars=0,
                )

    await asyncio.gather(*[_run_one(c) for c in filtered], return_exceptions=False)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    out = EnumCacheBootstrapResult(
        schema=schema_l,
        scanned_columns=len(filtered),
        cached_columns=int(cached),
        skipped_high_cardinality=int(skipped_hi),
        skipped_empty=int(skipped_empty),
        skipped_missing=int(skipped_missing),
        errors=int(errors),
        elapsed_ms=float(elapsed_ms),
    )

    SmartLogger.log(
        "INFO",
        "enum_cache.bootstrap.done",
        category="enum_cache.bootstrap",
        params=out.__dict__,
        max_inline_chars=0,
    )
    return out


__all__ = ["EnumCacheBootstrapResult", "ensure_enum_cache_for_schema"]


