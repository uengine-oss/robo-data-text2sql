from __future__ import annotations

import asyncio
import traceback
from typing import Any

from app.config import settings
from app.deps import get_db_connection
from app.sanity_checks.result import SanityCheckResult


async def check_target_db(*, timeout_seconds: float = 10.0) -> SanityCheckResult:
    """
    Target DB connection + basic metadata queries.

    Fail-fast conditions:
    - target_db_type is not PostgreSQL-compatible (current code uses asyncpg).
    - configured schemas are missing.
    """
    name = "target_db"

    db_type = (settings.target_db_type or "").strip().lower()
    if db_type not in {"postgresql", "postgres"}:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="Unsupported target_db_type for current implementation (asyncpg/PostgreSQL only).",
            data={"target_db_type": settings.target_db_type},
            error="target_db_type_mismatch",
        )

    schemas = [s.strip() for s in (settings.target_db_schemas or "").split(",") if s.strip()]
    if not schemas:
        schemas = [settings.target_db_schema]

    async def _run() -> dict[str, Any]:
        async for conn in get_db_connection():
            version = await conn.fetchval("SELECT version()")
            current_db = await conn.fetchval("SELECT current_database()")
            current_schema = await conn.fetchval("SELECT current_schema()")
            search_path = await conn.fetchval("SHOW search_path")

            rows = await conn.fetch(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name = ANY($1::text[])",
                schemas,
            )
            existing = {r["schema_name"] for r in rows}
            missing = sorted(set(schemas) - existing)
            if missing:
                raise RuntimeError(f"Missing schemas in target DB: {missing}")

            table_count = await conn.fetchval(
                """
                SELECT count(*)
                FROM information_schema.tables
                WHERE table_schema = ANY($1::text[])
                  AND table_type = 'BASE TABLE'
                """,
                schemas,
            )

            return {
                "db_type": settings.target_db_type,
                "host": f"{settings.target_db_host}:{settings.target_db_port}",
                "database": settings.target_db_name,
                "schemas": schemas,
                "current_db": current_db,
                "current_schema": current_schema,
                "search_path": search_path,
                "table_count": int(table_count or 0),
                "version": (version.split(",")[0] if isinstance(version, str) else str(version)),
            }

        # should never reach here
        raise RuntimeError("DB connection generator yielded no connection")

    try:
        data = await asyncio.wait_for(_run(), timeout=timeout_seconds)
        return SanityCheckResult(name=name, ok=True, detail="OK", data=data)
    except Exception as exc:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="Target DB sanity check failed",
            data={
                "db_type": settings.target_db_type,
                "host": f"{settings.target_db_host}:{settings.target_db_port}",
                "database": settings.target_db_name,
                "schemas": schemas,
            },
            error=repr(exc) + "\n" + traceback.format_exc(),
        )


