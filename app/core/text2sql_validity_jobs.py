"""
Background runner for Text2SQL validity bootstrap.

Why separate module:
- FastAPI lifespan should decide blocking vs background.
- Background run must not depend on a session that may be closed; it opens its own sessions.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Any, Dict, Optional, Sequence

from app.config import settings
from app.deps import neo4j_conn
from app.smart_logger import SmartLogger

from .text2sql_validity_bootstrap import ensure_text2sql_validity_flags


_task: Optional[asyncio.Task] = None


def is_text2sql_validity_task_running() -> bool:
    return _task is not None and not _task.done()


async def start_text2sql_validity_refresh_task(
    *,
    schemas: Optional[Sequence[str]] = None,
) -> None:
    """
    Start a one-shot background refresh task (idempotent).
    """
    global _task
    if not bool(getattr(settings, "text2sql_validity_bootstrap_enabled", True)):
        return
    db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
    if db_type in {"mysql", "mariadb"}:
        # MindsDB mode needs datasource-aware probing and is handled in request-time runtime repair.
        return
    if is_text2sql_validity_task_running():
        return

    async def _runner() -> Dict[str, Any]:
        session = None
        try:
            session = await neo4j_conn.get_session()
            return await ensure_text2sql_validity_flags(neo4j_session=session, schemas=schemas)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "text2sql.validity_job.error",
                category="text2sql.validity_job",
                params={"error": str(exc), "traceback": traceback.format_exc()},
                max_inline_chars=0,
            )
            return {"ok": False, "reason": str(exc)}
        finally:
            try:
                if session is not None:
                    await session.close()
            except Exception:
                pass

    _task = asyncio.create_task(_runner(), name="text2sql_validity_refresh")
    SmartLogger.log(
        "INFO",
        "text2sql.validity_job.started",
        category="text2sql.validity_job",
        params={"schemas": list(schemas) if schemas else None},
        max_inline_chars=0,
    )


async def stop_text2sql_validity_refresh_task() -> None:
    """Stop background refresh task (best-effort)."""
    global _task
    if _task is None:
        return
    if not _task.done():
        _task.cancel()
    try:
        await _task
    except Exception:
        pass
    _task = None
    SmartLogger.log(
        "INFO",
        "text2sql.validity_job.stopped",
        category="text2sql.validity_job",
        params=None,
        max_inline_chars=0,
    )


__all__ = [
    "is_text2sql_validity_task_running",
    "start_text2sql_validity_refresh_task",
    "stop_text2sql_validity_refresh_task",
]


