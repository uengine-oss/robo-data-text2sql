"""
Text2SQL validity bootstrap for MindsDB(MySQL endpoint) mode.

Purpose:
- Validate Table/Column nodes against the real datasource by executing lightweight passthrough probes.
- Persist text_to_sql_* validity flags in Neo4j with the same shape used by PostgreSQL bootstrap.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import settings
from app.core.sql_exec import SQLExecutor
from app.smart_logger import SmartLogger


def _schemas_from_settings() -> List[str]:
    schemas = [s.strip() for s in (settings.target_db_schemas or "").split(",") if s.strip()]
    if not schemas:
        schemas = [str(settings.target_db_schema or "").strip() or "public"]
    return [s.lower() for s in schemas if s]


def _q_ident(name: str) -> str:
    n = str(name or "")
    return '"' + n.replace('"', '""') + '"'


def _now_ms() -> int:
    return int(time.time() * 1000)


def _is_missing_object_error(exc: BaseException) -> bool:
    msg = str(exc or "").lower()
    needles = [
        "undefined_table",
        "undefined_column",
        "invalid_schema_name",
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


def _wrap_passthrough(inner_sql: str, datasource: str) -> str:
    ds = str(datasource or "").strip().replace("`", "``")
    return f"SELECT * FROM `{ds}` (\n{inner_sql}\n)"


def _pick_datasource(*, hint: str, table_prop: str) -> str:
    return str(hint or "").strip() or str(table_prop or "").strip()


def _ident_variants(schema: str, table: str, column: Optional[str] = None) -> List[Tuple[str, str, Optional[str]]]:
    s = str(schema or "").strip()
    t = str(table or "").strip()
    c = str(column or "").strip() if column is not None else None
    out: List[Tuple[str, str, Optional[str]]] = []
    candidates: List[Tuple[str, str, Optional[str]]] = [
        (s, t, c),
        (s.lower(), t.lower(), c.lower() if c is not None else None),
        (s.upper(), t.upper(), c.upper() if c is not None else None),
    ]
    for item in candidates:
        ss, tt, cc = item
        if not ss or not tt:
            continue
        if column is not None and not cc:
            continue
        if item not in out:
            out.append(item)
    return out


async def _preview_query(
    *,
    db_conn: Any,
    sql: str,
    timeout_s: float,
    row_limit: int = 1,
) -> Dict[str, Any]:
    executor = SQLExecutor()
    if hasattr(db_conn, "acquire"):
        async with db_conn.acquire() as conn:
            return await executor.preview_query(
                conn,
                sql,
                row_limit=max(1, int(row_limit)),
                timeout=float(timeout_s),
            )
    return await executor.preview_query(
        db_conn,
        sql,
        row_limit=max(1, int(row_limit)),
        timeout=float(timeout_s),
    )


@dataclass(frozen=True)
class _TableTarget:
    tid: str
    datasource_prop: str
    schema_prop: str
    table_name: str
    schema_l: str
    table_l: str


@dataclass(frozen=True)
class _ColumnTarget:
    cid: str
    fqn: str
    datasource_prop: str
    schema_prop: str
    table_name: str
    col_name: str
    schema_l: str
    table_l: str
    col_l: str


async def _open_mindsdb_pool(*, max_size: int):
    try:
        import aiomysql  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "aiomysql is required for target_db_type=mysql (MindsDB endpoint)."
        ) from exc
    return await aiomysql.create_pool(
        host=settings.target_db_host,
        port=int(settings.target_db_port),
        user=settings.target_db_user,
        password=settings.target_db_password,
        db=settings.target_db_name,
        minsize=1,
        maxsize=max(1, int(max_size)),
        autocommit=True,
    )


async def ensure_text2sql_validity_flags_mindsdb(
    *,
    neo4j_session,
    datasource: Optional[str] = None,
    db_conn: Optional[Any] = None,
    schemas: Optional[Sequence[str]] = None,
    concurrency: Optional[int] = None,
    query_timeout_s: Optional[float] = None,
    max_tables: Optional[int] = None,
    max_columns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    MindsDB path:
    - Table validity: SELECT 1 FROM schema.table LIMIT 1 (passthrough).
    - Column validity: SELECT 1 FROM schema.table WHERE col IS NOT NULL LIMIT 1 (passthrough).
    """
    started = time.perf_counter()
    datasource_hint = str(datasource or "").strip()

    schemas_l = [str(s).strip().lower() for s in (schemas or _schemas_from_settings()) if str(s).strip()]
    schema_filter_used: Optional[List[str]] = schemas_l if schemas_l else None

    concurrency_i = max(
        1,
        int(concurrency or getattr(settings, "text2sql_validity_bootstrap_concurrency", 6) or 6),
    )
    timeout_s = float(
        query_timeout_s
        or getattr(settings, "text2sql_validity_bootstrap_query_timeout_s", 2.0)
        or 2.0
    )
    max_tables_i = int(max_tables or getattr(settings, "text2sql_validity_bootstrap_max_tables", 0) or 0)
    max_columns_i = int(max_columns or getattr(settings, "text2sql_validity_bootstrap_max_columns", 0) or 0)

    created_pool = None
    if db_conn is None:
        db_conn = await _open_mindsdb_pool(max_size=concurrency_i)
        created_pool = db_conn

    # Single DB connection cannot run concurrent queries.
    db_lock: Optional[asyncio.Lock] = None
    if not hasattr(db_conn, "acquire"):
        db_lock = asyncio.Lock()
        concurrency_i = 1

    try:
        table_q = """
        MATCH (t:Table)
        WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
        RETURN
          elementId(t) AS tid,
          COALESCE(t.datasource,'') AS datasource_prop,
          COALESCE(t.schema,'') AS schema_prop,
          COALESCE(t.name,'') AS name_prop,
          COALESCE(t.original_name, t.name, '') AS display_name
        ORDER BY schema_prop, name_prop
        """
        if max_tables_i > 0:
            table_q += "\nLIMIT $limit"

        table_res = await neo4j_session.run(
            table_q,
            schemas=(schema_filter_used or []),
            limit=max_tables_i if max_tables_i > 0 else None,
        )
        table_rows = await table_res.data()

        table_targets: List[_TableTarget] = []
        for r in table_rows:
            tid = str(r.get("tid") or "").strip()
            datasource_prop = str(r.get("datasource_prop") or "").strip()
            schema_prop = str(r.get("schema_prop") or "").strip()
            name_prop = str(r.get("name_prop") or "").strip()
            display_name = str(r.get("display_name") or name_prop or "").strip()
            if not tid or not schema_prop or not display_name:
                continue
            table_targets.append(
                _TableTarget(
                    tid=tid,
                    datasource_prop=datasource_prop,
                    schema_prop=schema_prop,
                    table_name=display_name,
                    schema_l=schema_prop.lower(),
                    table_l=display_name.lower(),
                )
            )

        if not table_targets and schemas_l:
            SmartLogger.log(
                "WARNING",
                "text2sql.validity_mindsdb.schema_filter.no_match_fallback",
                category="text2sql.validity_mindsdb",
                params={"schemas": schemas_l},
                max_inline_chars=0,
            )
            table_q2 = """
            MATCH (t:Table)
            RETURN
              elementId(t) AS tid,
              COALESCE(t.datasource,'') AS datasource_prop,
              COALESCE(t.schema,'') AS schema_prop,
              COALESCE(t.name,'') AS name_prop,
              COALESCE(t.original_name, t.name, '') AS display_name
            ORDER BY schema_prop, name_prop
            """
            if max_tables_i > 0:
                table_q2 += "\nLIMIT $limit"
            table_res2 = await neo4j_session.run(
                table_q2,
                limit=max_tables_i if max_tables_i > 0 else None,
            )
            table_rows2 = await table_res2.data()
            table_targets = []
            for r in table_rows2:
                tid = str(r.get("tid") or "").strip()
                datasource_prop = str(r.get("datasource_prop") or "").strip()
                schema_prop = str(r.get("schema_prop") or "").strip()
                name_prop = str(r.get("name_prop") or "").strip()
                display_name = str(r.get("display_name") or name_prop or "").strip()
                if not tid or not schema_prop or not display_name:
                    continue
                table_targets.append(
                    _TableTarget(
                        tid=tid,
                        datasource_prop=datasource_prop,
                        schema_prop=schema_prop,
                        table_name=display_name,
                        schema_l=schema_prop.lower(),
                        table_l=display_name.lower(),
                    )
                )
            schema_filter_used = None

        col_q = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
        RETURN
          elementId(c) AS cid,
          COALESCE(t.datasource,'') AS datasource_prop,
          COALESCE(c.fqn,'') AS fqn,
          COALESCE(c.name,'') AS column_name,
          COALESCE(t.schema,'') AS schema_prop,
          COALESCE(t.original_name, t.name, '') AS display_name
        ORDER BY fqn
        """
        if max_columns_i > 0:
            col_q += "\nLIMIT $limit"

        col_res = await neo4j_session.run(
            col_q,
            schemas=schema_filter_used,
            limit=max_columns_i if max_columns_i > 0 else None,
        )
        col_rows = await col_res.data()
        col_targets: List[_ColumnTarget] = []
        for r in col_rows:
            cid = str(r.get("cid") or "").strip()
            datasource_prop = str(r.get("datasource_prop") or "").strip()
            fqn = str(r.get("fqn") or "").strip()
            col_name = str(r.get("column_name") or "").strip()
            schema_prop = str(r.get("schema_prop") or "").strip()
            display_name = str(r.get("display_name") or "").strip()
            if not cid:
                continue
            if not fqn or fqn.count(".") < 2:
                if schema_prop and display_name and col_name:
                    fqn = f"{schema_prop}.{display_name}.{col_name}".lower()
            parts = [p.strip() for p in fqn.split(".") if p.strip()]
            if len(parts) < 3:
                continue
            col_name_src = col_name or parts[2]
            table_name_src = display_name or parts[1]
            schema_src = schema_prop or parts[0]
            col_targets.append(
                _ColumnTarget(
                    cid=cid,
                    fqn=fqn.lower(),
                    datasource_prop=datasource_prop,
                    schema_prop=schema_src,
                    table_name=table_name_src,
                    col_name=col_name_src,
                    schema_l=parts[0].lower(),
                    table_l=parts[1].lower(),
                    col_l=parts[2].lower(),
                )
            )

        validated_at_ms = _now_ms()
        sem = asyncio.Semaphore(concurrency_i)

        table_probe: Dict[str, Dict[str, Any]] = {}
        table_valid_by_key: Dict[Tuple[str, str, str], bool] = {}

        async def _probe_one_table(t: _TableTarget) -> None:
            datasource_for_table = _pick_datasource(
                hint=datasource_hint,
                table_prop=t.datasource_prop,
            )
            ds_l_effective = datasource_for_table.lower()
            if not datasource_for_table:
                table_probe[t.tid] = {
                    "db_exists": False,
                    "is_valid": False,
                    "invalid_reason": "missing_datasource",
                    "row_count_est": 0,
                }
                table_valid_by_key[(ds_l_effective, t.schema_l, t.table_l)] = False
                return

            last_exc: Optional[Exception] = None
            for (schema_v, table_v, _) in _ident_variants(t.schema_prop, t.table_name, None):
                inner = f"SELECT 1 AS one FROM {_q_ident(schema_v)}.{_q_ident(table_v)} LIMIT 1"
                sql = _wrap_passthrough(inner, datasource_for_table)
                try:
                    async with sem:
                        if db_lock is not None:
                            async with db_lock:
                                preview = await _preview_query(
                                    db_conn=db_conn,
                                    sql=sql,
                                    timeout_s=timeout_s,
                                    row_limit=1,
                                )
                        else:
                            preview = await _preview_query(
                                db_conn=db_conn,
                                sql=sql,
                                timeout_s=timeout_s,
                                row_limit=1,
                            )
                    has_rows = int(preview.get("row_count") or 0) > 0
                    table_probe[t.tid] = {
                        "db_exists": True,
                        "is_valid": bool(has_rows),
                        "invalid_reason": "" if has_rows else "empty_table_confirmed",
                        "row_count_est": 1 if has_rows else 0,
                    }
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if _is_missing_object_error(exc):
                        continue
                    table_probe[t.tid] = {
                        "db_exists": True,
                        "is_valid": False,
                        "invalid_reason": "probe_failed",
                        "row_count_est": 0,
                    }
                    break

            if t.tid not in table_probe:
                table_probe[t.tid] = {
                    "db_exists": False,
                    "is_valid": False,
                    "invalid_reason": "missing_in_db" if _is_missing_object_error(last_exc or Exception()) else "probe_failed",
                    "row_count_est": 0,
                }
            table_valid_by_key[(ds_l_effective, t.schema_l, t.table_l)] = bool(
                table_probe.get(t.tid, {}).get("is_valid")
            )

        await asyncio.gather(*[_probe_one_table(t) for t in table_targets], return_exceptions=True)

        col_probe: Dict[str, Dict[str, Any]] = {}

        async def _probe_one_col(c: _ColumnTarget) -> None:
            ds_l = (_pick_datasource(hint=datasource_hint, table_prop=c.datasource_prop)).lower()
            parent_valid = bool(table_valid_by_key.get((ds_l, c.schema_l, c.table_l), True))
            if not parent_valid:
                col_probe[c.cid] = {
                    "db_exists": bool(
                        any(
                            (_pick_datasource(hint=datasource_hint, table_prop=t.datasource_prop).lower() == ds_l)
                            and
                            t.schema_l == c.schema_l
                            and t.table_l == c.table_l
                            and bool(table_probe.get(t.tid, {}).get("db_exists"))
                            for t in table_targets
                        )
                    ),
                    "is_valid": False,
                    "invalid_reason": "parent_table_invalid",
                    "has_nonnull_confirmed": False,
                }
                return

            table_match = next(
                (
                    t
                    for t in table_targets
                    if (_pick_datasource(hint=datasource_hint, table_prop=t.datasource_prop).lower() == ds_l)
                    and t.schema_l == c.schema_l
                    and t.table_l == c.table_l
                ),
                None,
            )
            if table_match is None:
                col_probe[c.cid] = {
                    "db_exists": False,
                    "is_valid": False,
                    "invalid_reason": "missing_in_db",
                    "has_nonnull_confirmed": None,
                }
                return

            datasource_for_col = _pick_datasource(
                hint=datasource_hint,
                table_prop=(c.datasource_prop or table_match.datasource_prop),
            )
            if not datasource_for_col:
                col_probe[c.cid] = {
                    "db_exists": False,
                    "is_valid": False,
                    "invalid_reason": "missing_datasource",
                    "has_nonnull_confirmed": None,
                }
                return

            last_exc: Optional[Exception] = None
            for (schema_v, table_v, col_v) in _ident_variants(
                c.schema_prop or table_match.schema_prop,
                c.table_name or table_match.table_name,
                c.col_name,
            ):
                if col_v is None:
                    continue
                inner = (
                    f"SELECT 1 AS one FROM {_q_ident(schema_v)}.{_q_ident(table_v)} "
                    f"WHERE {_q_ident(col_v)} IS NOT NULL LIMIT 1"
                )
                sql = _wrap_passthrough(inner, datasource_for_col)
                try:
                    async with sem:
                        if db_lock is not None:
                            async with db_lock:
                                preview = await _preview_query(
                                    db_conn=db_conn,
                                    sql=sql,
                                    timeout_s=timeout_s,
                                    row_limit=1,
                                )
                        else:
                            preview = await _preview_query(
                                db_conn=db_conn,
                                sql=sql,
                                timeout_s=timeout_s,
                                row_limit=1,
                            )
                    has_nonnull = int(preview.get("row_count") or 0) > 0
                    col_probe[c.cid] = {
                        "db_exists": True,
                        "is_valid": bool(has_nonnull),
                        "invalid_reason": "" if has_nonnull else "all_null_confirmed",
                        "has_nonnull_confirmed": bool(has_nonnull),
                    }
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    if _is_missing_object_error(exc):
                        continue
                    col_probe[c.cid] = {
                        "db_exists": True,
                        "is_valid": False,
                        "invalid_reason": "probe_failed",
                        "has_nonnull_confirmed": None,
                    }
                    break

            if c.cid not in col_probe:
                col_probe[c.cid] = {
                    "db_exists": False,
                    "is_valid": False,
                    "invalid_reason": "missing_in_db" if _is_missing_object_error(last_exc or Exception()) else "probe_failed",
                    "has_nonnull_confirmed": None,
                }

        await asyncio.gather(*[_probe_one_col(c) for c in col_targets], return_exceptions=True)

        table_updates = []
        for t in table_targets:
            p = table_probe.get(t.tid) or {}
            table_updates.append(
                {
                    "tid": t.tid,
                    "is_valid": bool(p.get("is_valid")),
                    "db_exists": bool(p.get("db_exists")),
                    "invalid_reason": str(p.get("invalid_reason") or ""),
                    "validated_at_ms": int(validated_at_ms),
                    "row_count_est": int(p.get("row_count_est") or 0),
                }
            )

        col_updates = []
        for c in col_targets:
            p = col_probe.get(c.cid) or {}
            col_updates.append(
                {
                    "cid": c.cid,
                    "is_valid": bool(p.get("is_valid")),
                    "db_exists": bool(p.get("db_exists")),
                    "invalid_reason": str(p.get("invalid_reason") or ""),
                    "validated_at_ms": int(validated_at_ms),
                    "has_nonnull_confirmed": (
                        bool(p.get("has_nonnull_confirmed"))
                        if p.get("has_nonnull_confirmed") is not None
                        else None
                    ),
                }
            )

        async def _write_table_batch(rows: List[Dict[str, Any]]) -> None:
            if not rows:
                return
            q = """
            UNWIND $rows AS r
            MATCH (t) WHERE elementId(t) = r.tid
            SET t.text_to_sql_is_valid = r.is_valid,
                t.text_to_sql_db_exists = r.db_exists,
                t.text_to_sql_invalid_reason = r.invalid_reason,
                t.text_to_sql_validated_at_ms = r.validated_at_ms,
                t.text_to_sql_row_count_est = r.row_count_est,
                t.text_to_sql_table_type = 'BASE TABLE'
            """
            res = await neo4j_session.run(q, rows=rows)
            await res.consume()

        async def _write_col_batch(rows: List[Dict[str, Any]]) -> None:
            if not rows:
                return
            q = """
            UNWIND $rows AS r
            MATCH (c) WHERE elementId(c) = r.cid
            SET c.text_to_sql_is_valid = r.is_valid,
                c.text_to_sql_db_exists = r.db_exists,
                c.text_to_sql_invalid_reason = r.invalid_reason,
                c.text_to_sql_validated_at_ms = r.validated_at_ms,
                c.text_to_sql_null_frac_est = NULL,
                c.text_to_sql_has_nonnull_confirmed = r.has_nonnull_confirmed
            """
            res = await neo4j_session.run(q, rows=rows)
            await res.consume()

        chunk = 500
        for i in range(0, len(table_updates), chunk):
            await _write_table_batch(table_updates[i : i + chunk])
        for i in range(0, len(col_updates), chunk):
            await _write_col_batch(col_updates[i : i + chunk])

        q_backfill_tables = """
        MATCH (t:Table)
        WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
          AND (t.text_to_sql_is_valid IS NULL OR t.text_to_sql_db_exists IS NULL)
        SET t.text_to_sql_is_valid = COALESCE(t.text_to_sql_is_valid, true),
            t.text_to_sql_db_exists = COALESCE(t.text_to_sql_db_exists, true),
            t.text_to_sql_invalid_reason = COALESCE(t.text_to_sql_invalid_reason, ''),
            t.text_to_sql_validated_at_ms = COALESCE(t.text_to_sql_validated_at_ms, $validated_at_ms)
        RETURN count(t) AS cnt
        """
        q_backfill_cols = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
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
        datasources_used = sorted(
            {
                str(_pick_datasource(hint=datasource_hint, table_prop=t.datasource_prop) or "").strip()
                for t in table_targets
                if str(_pick_datasource(hint=datasource_hint, table_prop=t.datasource_prop) or "").strip()
            }
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        out = {
            "ok": True,
            "datasource_hint": datasource_hint or None,
            "datasources_used": datasources_used,
            "schemas": schemas_l,
            "tables_total": len(table_updates),
            "columns_total": len(col_updates),
            "tables_invalid": int(invalid_tables),
            "columns_invalid": int(invalid_cols),
            "tables_backfilled": int(backfilled_tables),
            "columns_backfilled": int(backfilled_columns),
            "elapsed_ms": float(elapsed_ms),
        }
        SmartLogger.log(
            "INFO",
            "text2sql.validity_mindsdb.done",
            category="text2sql.validity_mindsdb",
            params=out,
            max_inline_chars=0,
        )
        return out
    except Exception as exc:
        out_err = {"ok": False, "reason": str(exc), "traceback": traceback.format_exc()}
        SmartLogger.log(
            "ERROR",
            "text2sql.validity_mindsdb.error",
            category="text2sql.validity_mindsdb",
            params=out_err,
            max_inline_chars=0,
        )
        return out_err
    finally:
        if created_pool is not None:
            try:
                created_pool.close()
                await created_pool.wait_closed()
            except Exception:
                pass


__all__ = ["ensure_text2sql_validity_flags_mindsdb"]

