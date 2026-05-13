from __future__ import annotations

import asyncio
import traceback
from typing import Any

from app.config import settings
from app.deps import get_db_connection
from app.sanity_checks.result import SanityCheckResult


async def check_target_db(*, timeout_seconds: float = 30.0) -> SanityCheckResult:
    """
    Target database sanity check:

    - mysql/mariadb: MindsDB MySQL protocol ping (SELECT 1)
    - postgres/postgresql: asyncpg ping (SELECT 1)
    """
    name = "target_db"

    db_type = (settings.target_db_type or "").strip().lower()

    if db_type in {"postgresql", "postgres"}:

        async def _run_pg() -> dict[str, Any]:
            async for conn in get_db_connection():
                one = await conn.fetchval("SELECT 1")
                ping = int(one) if one is not None else 0
                if ping != 1:
                    raise RuntimeError(f"Unexpected PostgreSQL ping result: {one!r}")
                ver = await conn.fetchval("SELECT version()")
                return {
                    "db_type": settings.target_db_type,
                    "host": f"{settings.target_db_host}:{settings.target_db_port}",
                    "database": settings.target_db_name,
                    "ping": ping,
                    "version": str(ver or "")[:160],
                }
            raise RuntimeError("DB connection generator yielded no connection")

        try:
            data = await asyncio.wait_for(_run_pg(), timeout=timeout_seconds)
            return SanityCheckResult(name=name, ok=True, detail="OK", data=data)
        except Exception as exc:
            return SanityCheckResult(
                name=name,
                ok=False,
                detail="PostgreSQL connectivity sanity check failed",
                data={
                    "target_db_type": settings.target_db_type,
                    "host": f"{settings.target_db_host}:{settings.target_db_port}",
                    "database": settings.target_db_name,
                },
                error=repr(exc) + "\n" + traceback.format_exc(),
            )

    if db_type not in {"mysql", "mariadb"}:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="Target DB sanity check supports mysql/mariadb or postgres endpoints only.",
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
