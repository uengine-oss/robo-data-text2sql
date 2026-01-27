from __future__ import annotations

import asyncio
import traceback
from typing import Any

from app.config import settings
from app.deps import get_db_connection, MYSQL_AVAILABLE
from app.sanity_checks.result import SanityCheckResult


async def check_target_db(*, timeout_seconds: float = 10.0) -> SanityCheckResult:
    """
    Target DB connection + basic metadata queries.

    Fail-fast conditions:
    - target_db_type is not supported (postgresql or mysql).
    - configured schemas are missing.
    """
    name = "target_db"

    db_type = (settings.target_db_type or "").strip().lower()
    supported_types = {"postgresql", "postgres", "mysql"}
    
    if db_type not in supported_types:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail=f"Unsupported target_db_type. Supported: {supported_types}",
            data={"target_db_type": settings.target_db_type},
            error="target_db_type_mismatch",
        )
    
    if db_type == "mysql" and not MYSQL_AVAILABLE:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="MySQL support requires aiomysql. Run: pip install aiomysql",
            data={"target_db_type": settings.target_db_type},
            error="aiomysql_not_installed",
        )

    schemas = [s.strip() for s in (settings.target_db_schemas or "").split(",") if s.strip()]
    if not schemas:
        schemas = [settings.target_db_schema]

    async def _run_mysql() -> dict[str, Any]:
        """MySQL/MindsDB specific sanity check"""
        async for conn in get_db_connection():
            async with conn.cursor() as cursor:
                # Get version
                await cursor.execute("SELECT VERSION()")
                version_row = await cursor.fetchone()
                version = version_row[0] if version_row else "unknown"
                
                # Get current database
                await cursor.execute("SELECT DATABASE()")
                db_row = await cursor.fetchone()
                current_db = db_row[0] if db_row else settings.target_db_name
                
                # List databases (MindsDB specific)
                await cursor.execute("SHOW DATABASES")
                db_rows = await cursor.fetchall()
                databases = [r[0] for r in db_rows]
                
                # List tables in current database
                await cursor.execute("SHOW TABLES")
                table_rows = await cursor.fetchall()
                table_count = len(table_rows)
                
                return {
                    "db_type": settings.target_db_type,
                    "host": f"{settings.target_db_host}:{settings.target_db_port}",
                    "database": settings.target_db_name,
                    "schemas": schemas,
                    "current_db": current_db,
                    "available_databases": databases,
                    "table_count": table_count,
                    "version": str(version),
                }
        raise RuntimeError("DB connection generator yielded no connection")

    async def _run_postgres() -> dict[str, Any]:
        """PostgreSQL specific sanity check"""
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

        raise RuntimeError("DB connection generator yielded no connection")

    try:
        if db_type == "mysql":
            data = await asyncio.wait_for(_run_mysql(), timeout=timeout_seconds)
        else:
            data = await asyncio.wait_for(_run_postgres(), timeout=timeout_seconds)
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


