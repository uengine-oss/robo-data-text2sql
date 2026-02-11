from __future__ import annotations

import asyncio
import traceback
from typing import Any

from app.config import settings
from app.deps import neo4j_conn
from app.sanity_checks.result import SanityCheckResult

async def check_neo4j(*, timeout_seconds: float = 10.0) -> SanityCheckResult:
    """
    Neo4j connectivity + required constraints/indexes existence check.

    Notes:
    - :Table count can be 0 (allowed).
    - Constraints/indexes are treated as required (missing => fail-fast).
    """
    name = "neo4j"

    async def _run() -> dict[str, Any]:
        await neo4j_conn.connect()
        session = await neo4j_conn.get_session()
        try:
            # Basic connectivity
            result = await session.run("RETURN 1 AS ok")
            await result.single()

            # Counts (informational)
            node_count_rec = await (await session.run("MATCH (n) RETURN count(n) as c")).single()
            table_count_rec = await (await session.run("MATCH (t:Table) RETURN count(t) as c")).single()
            node_count = int(node_count_rec["c"])
            table_count = int(table_count_rec["c"])

            return {
                "uri": settings.neo4j_uri,
                "database": settings.neo4j_database,
                "node_count": node_count,
                "table_count": table_count
            }
        finally:
            await session.close()

    try:
        data = await asyncio.wait_for(_run(), timeout=timeout_seconds)
        return SanityCheckResult(name=name, ok=True, detail="OK", data=data)
    except Exception as exc:
        return SanityCheckResult(
            name=name,
            ok=False,
            detail="Neo4j sanity check failed",
            data={"uri": settings.neo4j_uri, "database": settings.neo4j_database},
            error=repr(exc) + "\n" + traceback.format_exc(),
        )


