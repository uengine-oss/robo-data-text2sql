from __future__ import annotations

import asyncio
import traceback
from typing import Any

from app.config import settings
from app.deps import neo4j_conn
from app.sanity_checks.result import SanityCheckResult


_REQUIRED_CONSTRAINT_NAMES = {"table_key", "column_fqn"}
_REQUIRED_INDEX_NAMES = {"table_vec_index", "column_vec_index"}


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

            # Required constraints
            cons_res = await session.run("SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names")
            cons_rec = await cons_res.single()
            constraint_names = set(cons_rec["names"] or [])

            missing_constraints = sorted(_REQUIRED_CONSTRAINT_NAMES - constraint_names)
            if missing_constraints:
                raise RuntimeError(f"Missing Neo4j constraints: {missing_constraints}")

            # Required indexes (including VECTOR indexes)
            idx_res = await session.run("SHOW INDEXES YIELD name RETURN collect(name) AS names")
            idx_rec = await idx_res.single()
            index_names = set(idx_rec["names"] or [])

            missing_indexes = sorted(_REQUIRED_INDEX_NAMES - index_names)
            if missing_indexes:
                raise RuntimeError(f"Missing Neo4j indexes: {missing_indexes}")

            return {
                "uri": settings.neo4j_uri,
                "database": settings.neo4j_database,
                "node_count": node_count,
                "table_count": table_count,
                "constraints_checked": sorted(_REQUIRED_CONSTRAINT_NAMES),
                "indexes_checked": sorted(_REQUIRED_INDEX_NAMES),
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


