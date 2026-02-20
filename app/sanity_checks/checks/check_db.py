from __future__ import annotations

import asyncio
import traceback
from typing import Any

from app.config import settings
from app.deps import get_db_connection
from app.sanity_checks.result import SanityCheckResult


async def check_target_db(*, timeout_seconds: float = 30.0) -> SanityCheckResult:
    """
    MindsDB(MySQL endpoint) connectivity sanity check.

    We only verify:
    - socket/auth connection is successful
    - a minimal round-trip query works (SELECT 1)

    We intentionally DO NOT validate datasource/schema existence because
    startup may run before any datasource is attached in MindsDB.
    """
    name = "target_db"

    db_type = (settings.target_db_type or "").strip().lower()
    if db_type not in {"mysql", "mariadb"}:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="MindsDB sanity check supports mysql/mariadb endpoint mode only.",
            data={"target_db_type": settings.target_db_type},
            error="target_db_type_mismatch",
        )

    async def _run() -> dict[str, Any]:
        async for conn in get_db_connection():
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                row0 = await cur.fetchone()
                ping = int(row0[0]) if row0 and row0[0] is not None else 0
                if ping != 1:
                    raise RuntimeError(f"Unexpected ping result from MindsDB endpoint: {row0}")

                await cur.execute("SELECT VERSION()")
                row = await cur.fetchone()
                version = row[0] if row else None

                await cur.execute("SELECT DATABASE()")
                row2 = await cur.fetchone()
                current_db = row2[0] if row2 else None

            return {
                "db_type": settings.target_db_type,
                "host": f"{settings.target_db_host}:{settings.target_db_port}",
                "database": settings.target_db_name,
                "ping": ping,
                "current_db": current_db,
                "version": str(version or ""),
            }

        raise RuntimeError("DB connection generator yielded no connection")

    try:
        data = await asyncio.wait_for(_run(), timeout=timeout_seconds)
        return SanityCheckResult(name=name, ok=True, detail="OK", data=data)
    except Exception as exc:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="MindsDB endpoint sanity check failed",
            data={
                "db_type": settings.target_db_type,
                "host": f"{settings.target_db_host}:{settings.target_db_port}",
                "database": settings.target_db_name,
            },
            error=repr(exc) + "\n" + traceback.format_exc(),
        )


