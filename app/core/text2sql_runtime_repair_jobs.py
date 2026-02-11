"""
Runtime background repair jobs for Text2SQL properties.

Strategy:
- Request path performs a lightweight, throttled drift check.
- If missing properties are detected, trigger a single-flight background repair.
- Repair pipeline is best-effort and never blocks the user response path.
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Dict, Optional, Sequence

from app.config import settings
from app.core.enum_cache_bootstrap import ensure_enum_cache_for_schema
from app.core.neo4j_bootstrap import ensure_neo4j_schema
from app.core.text2sql_runtime_repair import inspect_text2sql_property_gaps
from app.core.text2sql_table_vectorizer import ensure_text_to_sql_table_vectors
from app.core.text2sql_validity_bootstrap import ensure_text2sql_validity_flags
from app.core.text2sql_validity_mindsdb import ensure_text2sql_validity_flags_mindsdb
from app.deps import neo4j_conn
from app.smart_logger import SmartLogger


_task: Optional[asyncio.Task] = None
_check_lock = asyncio.Lock()
_last_check_at_ms: float = 0.0
_last_check_result: Optional[Dict[str, Any]] = None
_last_repair_result: Optional[Dict[str, Any]] = None


def is_text2sql_runtime_repair_running() -> bool:
    return _task is not None and not _task.done()


def _now_ms() -> float:
    return time.time() * 1000.0


async def _backfill_minimum_validity_flags(*, neo4j_session, schemas: Optional[Sequence[str]]) -> Dict[str, int]:
    """
    DB-independent fallback:
    - Fill missing text_to_sql_is_valid / text_to_sql_db_exists for Table/Column.
    - Does not touch vectors.
    """
    q_table = """
    MATCH (t:Table)
    WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
      AND (t.text_to_sql_is_valid IS NULL OR t.text_to_sql_db_exists IS NULL)
    SET t.text_to_sql_is_valid = COALESCE(t.text_to_sql_is_valid, true),
        t.text_to_sql_db_exists = COALESCE(t.text_to_sql_db_exists, true),
        t.text_to_sql_invalid_reason = COALESCE(t.text_to_sql_invalid_reason, ''),
        t.text_to_sql_validated_at_ms = COALESCE(t.text_to_sql_validated_at_ms, timestamp())
    RETURN count(t) AS cnt
    """
    q_col = """
    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
    WHERE ($schemas IS NULL OR toLower(COALESCE(t.schema,'')) IN $schemas)
      AND (c.text_to_sql_is_valid IS NULL OR c.text_to_sql_db_exists IS NULL)
    SET c.text_to_sql_is_valid = COALESCE(c.text_to_sql_is_valid, COALESCE(t.text_to_sql_is_valid, true)),
        c.text_to_sql_db_exists = COALESCE(c.text_to_sql_db_exists, COALESCE(t.text_to_sql_db_exists, true)),
        c.text_to_sql_invalid_reason = COALESCE(c.text_to_sql_invalid_reason, ''),
        c.text_to_sql_validated_at_ms = COALESCE(c.text_to_sql_validated_at_ms, timestamp())
    RETURN count(c) AS cnt
    """
    table_cnt = 0
    col_cnt = 0
    try:
        rt = await neo4j_session.run(q_table, schemas=(list(schemas) if schemas else None))
        row_t = await rt.single()
        table_cnt = int((row_t or {}).get("cnt") or 0)
    except Exception:
        table_cnt = 0
    try:
        rc = await neo4j_session.run(q_col, schemas=(list(schemas) if schemas else None))
        row_c = await rc.single()
        col_cnt = int((row_c or {}).get("cnt") or 0)
    except Exception:
        col_cnt = 0
    return {"tables_backfilled": int(table_cnt), "columns_backfilled": int(col_cnt)}


async def _run_repair_pipeline(
    *,
    source: str,
    before: Dict[str, Any],
    datasource: Optional[str] = None,
    db_conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Heavy repair pipeline (background):
    1) Ensure Neo4j schema/indexes.
    2) Ensure Table text_to_sql_vector / text_to_sql_embedding_text.
    3) Ensure Table/Column text_to_sql_is_valid / text_to_sql_db_exists.
    4) Re-check gaps.
    """
    session = None
    started = _now_ms()
    try:
        session = await neo4j_conn.get_session()
        steps: Dict[str, Any] = {}
        schemas_used = before.get("schemas_used")
        if not isinstance(schemas_used, list):
            schemas_used = None
        db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
        datasource_used = str(datasource or "").strip()

        try:
            await ensure_neo4j_schema(session)
            steps["ensure_neo4j_schema"] = {"ok": True}
        except Exception as exc:
            steps["ensure_neo4j_schema"] = {"ok": False, "error": str(exc)}

        try:
            await ensure_text_to_sql_table_vectors(session)
            steps["ensure_text_to_sql_table_vectors"] = {"ok": True}
        except Exception as exc:
            steps["ensure_text_to_sql_table_vectors"] = {"ok": False, "error": str(exc)}

        try:
            if db_type in {"mysql", "mariadb"}:
                validity_out = await ensure_text2sql_validity_flags_mindsdb(
                    neo4j_session=session,
                    db_conn=db_conn,
                    datasource=(datasource_used or None),
                    schemas=schemas_used,
                )
            else:
                validity_out = await ensure_text2sql_validity_flags(
                    neo4j_session=session,
                    schemas=schemas_used,
                )
            steps["ensure_text2sql_validity_flags"] = {"ok": bool(validity_out.get("ok")), "result": validity_out}
        except Exception as exc:
            steps["ensure_text2sql_validity_flags"] = {"ok": False, "error": str(exc)}

        # Keep enum cache refresh in runtime repair so MindsDB mode can self-heal
        # when startup bootstrap could not run (e.g., datasource unknown at startup).
        if bool(getattr(settings, "enum_cache_bootstrap_on_startup", True)):
            enum_schema = (
                str((schemas_used or [getattr(settings, "target_db_schema", "public")])[0] or "").strip()
                or str(getattr(settings, "target_db_schema", "public") or "public").strip()
            )
            max_values = int(getattr(settings, "enum_cache_bootstrap_max_values", 200) or 200)
            max_columns = int(getattr(settings, "enum_cache_bootstrap_max_columns", 5000) or 5000)
            concurrency = int(getattr(settings, "enum_cache_bootstrap_concurrency", 6) or 6)
            query_timeout_s = float(getattr(settings, "enum_cache_bootstrap_query_timeout_s", 2.0) or 2.0)
            enum_conn_created = None
            enum_conn = db_conn
            try:
                if db_type in {"mysql", "mariadb"}:
                    if enum_conn is None:
                        import aiomysql  # type: ignore

                        enum_conn = await aiomysql.create_pool(
                            host=settings.target_db_host,
                            port=int(settings.target_db_port),
                            user=settings.target_db_user,
                            password=settings.target_db_password,
                            db=settings.target_db_name,
                            minsize=1,
                            maxsize=max(1, int(concurrency)),
                            autocommit=True,
                        )
                        enum_conn_created = enum_conn
                    enum_out = await ensure_enum_cache_for_schema(
                        neo4j_session=session,
                        db_conn=enum_conn,
                        schema=enum_schema,
                        max_columns=max_columns,
                        max_values=max_values,
                        concurrency=concurrency,
                        query_timeout_s=query_timeout_s,
                        name_hint_only=False,
                        include_fqns=None,
                        skip_if_cached=True,
                        datasource=(datasource_used or None),
                    )
                else:
                    import asyncpg

                    if enum_conn is None:
                        ssl_mode = settings.target_db_ssl if settings.target_db_ssl != "disable" else False

                        async def _init_conn(conn: asyncpg.Connection) -> None:
                            schemas = settings.target_db_schemas.split(",")
                            schemas_str = ", ".join(s.strip() for s in schemas if s.strip())
                            if schemas_str:
                                await conn.execute(f"SET search_path TO {schemas_str}")

                        enum_conn = await asyncpg.create_pool(
                            host=settings.target_db_host,
                            port=settings.target_db_port,
                            database=settings.target_db_name,
                            user=settings.target_db_user,
                            password=settings.target_db_password,
                            ssl=ssl_mode,
                            min_size=1,
                            max_size=max(1, int(concurrency)),
                            init=_init_conn,
                        )
                        enum_conn_created = enum_conn
                    enum_out = await ensure_enum_cache_for_schema(
                        neo4j_session=session,
                        db_conn=enum_conn,
                        schema=enum_schema,
                        max_columns=max_columns,
                        max_values=max_values,
                        concurrency=concurrency,
                        query_timeout_s=query_timeout_s,
                        name_hint_only=False,
                        include_fqns=None,
                        skip_if_cached=True,
                        datasource=None,
                    )
                steps["ensure_enum_cache_for_schema"] = {"ok": True, "result": dict(enum_out.__dict__)}
            except Exception as exc:
                steps["ensure_enum_cache_for_schema"] = {"ok": False, "error": str(exc)}
            finally:
                if enum_conn_created is not None:
                    try:
                        if hasattr(enum_conn_created, "wait_closed"):
                            enum_conn_created.close()
                            await enum_conn_created.wait_closed()
                        else:
                            await enum_conn_created.close()
                    except Exception:
                        pass

        try:
            backfill_out = await _backfill_minimum_validity_flags(
                neo4j_session=session,
                schemas=schemas_used,
            )
            steps["backfill_minimum_validity_flags"] = {"ok": True, "result": backfill_out}
        except Exception as exc:
            steps["backfill_minimum_validity_flags"] = {"ok": False, "error": str(exc)}

        after = await inspect_text2sql_property_gaps(neo4j_session=session)
        out = {
            "ok": bool(int(after.get("missing_total") or 0) == 0),
            "source": source,
            "before": before,
            "after": after,
            "steps": steps,
            "elapsed_ms": float(_now_ms() - started),
        }
        SmartLogger.log(
            "INFO",
            "text2sql.runtime_repair.done",
            category="text2sql.runtime_repair",
            params=out,
            max_inline_chars=0,
        )
        return out
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        out_err = {
            "ok": False,
            "source": source,
            "before": before,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_ms": float(_now_ms() - started),
        }
        SmartLogger.log(
            "ERROR",
            "text2sql.runtime_repair.error",
            category="text2sql.runtime_repair",
            params=out_err,
            max_inline_chars=0,
        )
        return out_err
    finally:
        try:
            if session is not None:
                await session.close()
        except Exception:
            pass


async def run_text2sql_runtime_repair_blocking(
    *,
    source: str,
    check: Dict[str, Any],
    datasource: Optional[str] = None,
    db_conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Ensure runtime repair pipeline is completed before returning.
    Used for cold-start blocking mode in request path.
    """
    global _task

    if not bool(getattr(settings, "text2sql_runtime_repair_enabled", True)):
        return {"status": "disabled", "check": check}
    if not bool(check.get("needs_repair")):
        return {"status": "clean", "check": check}

    wait_task: Optional[asyncio.Task] = None
    async with _check_lock:
        if is_text2sql_runtime_repair_running():
            wait_task = _task
        else:
            _task = asyncio.create_task(
                _run_repair_pipeline(
                    source=source,
                    before=check,
                    datasource=datasource,
                    db_conn=db_conn,
                ),
                name="text2sql_runtime_repair",
            )
            _register_repair_task(_task)
            wait_task = _task

    repair_result: Dict[str, Any]
    try:
        repair_result = await wait_task  # type: ignore[assignment]
    except Exception as exc:
        repair_result = {"ok": False, "error": str(exc)}
    return {"status": "done", "check": check, "repair": repair_result}


def _register_repair_task(task: asyncio.Task) -> None:
    global _last_repair_result

    def _on_done(done_task: asyncio.Task) -> None:
        global _last_repair_result
        try:
            _last_repair_result = done_task.result()
        except Exception as exc:
            _last_repair_result = {"ok": False, "error": str(exc)}

    task.add_done_callback(_on_done)


def get_last_text2sql_runtime_repair_result() -> Optional[Dict[str, Any]]:
    return dict(_last_repair_result) if isinstance(_last_repair_result, dict) else None


async def maybe_trigger_text2sql_runtime_repair(
    *,
    neo4j_session,
    source: str = "react",
    schemas: Optional[Sequence[str]] = None,
    force_check: bool = False,
    datasource: Optional[str] = None,
    db_conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Lightweight entrypoint called from request path.

    Returns:
    - {"status":"disabled"}
    - {"status":"throttled", "check": ...}
    - {"status":"clean", "check": ...}
    - {"status":"cold_start_detected", "check": ...}
    - {"status":"already_running", "check": ...}
    - {"status":"triggered", "check": ...}
    """
    global _task, _last_check_at_ms, _last_check_result

    if not bool(getattr(settings, "text2sql_runtime_repair_enabled", True)):
        return {"status": "disabled"}

    check_interval_s = float(getattr(settings, "text2sql_runtime_repair_check_interval_s", 30.0) or 30.0)
    now = _now_ms()

    async with _check_lock:
        if not force_check and _last_check_result is not None and (now - _last_check_at_ms) < (check_interval_s * 1000.0):
            check = dict(_last_check_result)
            if bool(check.get("cold_start_detected")):
                return {"status": "cold_start_detected", "check": check}
            if is_text2sql_runtime_repair_running():
                return {"status": "already_running", "check": check}
            if bool(check.get("needs_repair")):
                _task = asyncio.create_task(
                    _run_repair_pipeline(
                        source=source,
                        before=check,
                        datasource=datasource,
                        db_conn=db_conn,
                    ),
                    name="text2sql_runtime_repair",
                )
                _register_repair_task(_task)
                SmartLogger.log(
                    "INFO",
                    "text2sql.runtime_repair.triggered.cached_check",
                    category="text2sql.runtime_repair",
                    params={"source": source, "check": check},
                    max_inline_chars=0,
                )
                return {"status": "triggered", "check": check}
            return {"status": "throttled", "check": check}

        check = await inspect_text2sql_property_gaps(neo4j_session=neo4j_session, schemas=schemas)
        _last_check_at_ms = _now_ms()
        _last_check_result = dict(check)

        if not bool(check.get("needs_repair")):
            return {"status": "clean", "check": check}
        if bool(check.get("cold_start_detected")):
            return {"status": "cold_start_detected", "check": check}

        if is_text2sql_runtime_repair_running():
            return {"status": "already_running", "check": check}

        _task = asyncio.create_task(
            _run_repair_pipeline(
                source=source,
                before=check,
                datasource=datasource,
                db_conn=db_conn,
            ),
            name="text2sql_runtime_repair",
        )
        _register_repair_task(_task)
        SmartLogger.log(
            "INFO",
            "text2sql.runtime_repair.triggered",
            category="text2sql.runtime_repair",
            params={"source": source, "check": check},
            max_inline_chars=0,
        )
        return {"status": "triggered", "check": check}


__all__ = [
    "get_last_text2sql_runtime_repair_result",
    "is_text2sql_runtime_repair_running",
    "maybe_trigger_text2sql_runtime_repair",
    "run_text2sql_runtime_repair_blocking",
]

