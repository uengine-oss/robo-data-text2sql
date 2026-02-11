"""
Neo4j bootstrap utilities executed at server startup.

Goal:
- Ensure required constraints/indexes (including vector indexes) exist before serving requests.
- Use IF NOT EXISTS statements so this is safe to run repeatedly.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.models.neo4j_history import Neo4jQueryRepository
from app.smart_logger import SmartLogger


def _is_constraint_conflict_error(exc: Exception) -> bool:
    """
    Heuristic check for Neo4j constraint conflicts across driver/Neo4j versions.
    """
    code = getattr(exc, "code", None) or getattr(exc, "neo4j_code", None)
    msg = str(exc)
    if code and "ConstraintAlreadyExists" in str(code):
        return True
    return ("ConstraintAlreadyExists" in msg) or ("Conflicting constraint already exists" in msg)


async def _find_table_identity_constraint(session) -> Optional[Dict[str, Any]]:
    """
    Detect an existing constraint that already guarantees identity for (:Table {db, schema, name}).

    Why:
    - Creating a NODE KEY constraint conflicts with an existing UNIQUENESS constraint on the same
      property set (Neo.ClientError.Schema.ConstraintAlreadyExists / conflicting constraint).
    - On startup we want idempotency; if a compatible constraint already exists, we skip creation.
    """
    # Prefer Neo4j 5+ syntax
    try:
        res = await session.run(
            """
            SHOW CONSTRAINTS
            YIELD name, type, entityType, labelsOrTypes, properties
            WHERE entityType = 'NODE'
              AND any(l IN labelsOrTypes WHERE l = 'Table')
            RETURN name, type, properties
            """
        )
        rows = await res.data()
        wanted_props = {"db", "schema", "name"}
        for r in rows:
            props = r.get("properties") or []
            ctype = (r.get("type") or "").upper()
            if set(props) == wanted_props and ("NODE_KEY" in ctype or "UNIQUENESS" in ctype or "UNIQUE" in ctype):
                return {"name": r.get("name"), "type": r.get("type"), "properties": props}
    except Exception:
        # Fall back to Neo4j 4.x / older procedures
        pass

    try:
        res = await session.run("CALL db.constraints()")
        rows = await res.data()
        wanted_tokens = [":Table", "db", "schema", "name"]
        for r in rows:
            desc = str(r.get("description") or "")
            desc_u = desc.upper()
            if all(t.upper() in desc_u for t in wanted_tokens) and ("IS UNIQUE" in desc_u or "IS NODE KEY" in desc_u):
                return {"name": r.get("name"), "type": "UNKNOWN", "description": desc}
    except Exception:
        pass

    return None


async def ensure_neo4j_schema(session) -> None:
    """
    Best-effort schema/index bootstrap.
    - Does NOT raise on failure (server can still start); errors are logged.
    """
    # 1) Table/Column graph constraints + vector indexes (used by graph_search + build_sql_context)
    table_column_queries: List[Tuple[str, bool]] = [
        (
            """
            CREATE CONSTRAINT table_key IF NOT EXISTS
            FOR (t:Table) REQUIRE (t.db, t.schema, t.name) IS NODE KEY
            """,
            False,
        ),
        (
            """
            CREATE CONSTRAINT column_fqn IF NOT EXISTS
            FOR (c:Column) REQUIRE c.fqn IS UNIQUE
            """,
            False,
        ),
        (
            """
            CREATE INDEX table_name_idx IF NOT EXISTS
            FOR (t:Table) ON (t.name)
            """,
            False,
        ),
        (
            """
            CREATE INDEX column_name_idx IF NOT EXISTS
            FOR (c:Column) ON (c.name)
            """,
            False,
        ),
        (
            """
            CREATE VECTOR INDEX table_vec_index IF NOT EXISTS
            FOR (t:Table) ON (t.vector)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            True,
        ),
        (
            """
            CREATE VECTOR INDEX text_to_sql_table_vec_index IF NOT EXISTS
            FOR (t:Table) ON (t.text_to_sql_vector)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            True,
        ),
        (
            """
            CREATE VECTOR INDEX column_vec_index IF NOT EXISTS
            FOR (c:Column) ON (c.vector)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            True,
        ),
    ]

    ok = 0
    fail = 0
    # Avoid noisy/confusing WARNING logs when a compatible identity constraint already exists.
    table_identity = await _find_table_identity_constraint(session)
    for cypher, needs_dim in table_column_queries:
        if "CREATE CONSTRAINT table_key" in cypher and table_identity is not None:
            SmartLogger.log(
                "INFO",
                "neo4j.bootstrap.table_key.skip.existing_identity_constraint",
                category="neo4j.bootstrap",
                params={"existing_constraint": table_identity},
                max_inline_chars=0,
            )
            ok += 1
            continue
        try:
            if needs_dim:
                res = await session.run(cypher, dimensions=settings.embedding_dimension)
            else:
                res = await session.run(cypher)
            # Make sure the statement is fully executed on the server.
            try:
                await res.consume()
            except Exception:
                # Some driver versions may not require consumption for schema ops.
                pass
            ok += 1
        except Exception as exc:
            # If Table identity is already enforced via another constraint, creating a NODE KEY will conflict.
            # Treat this as "already satisfied" to keep startup idempotent and avoid noisy warnings.
            if "CREATE CONSTRAINT table_key" in cypher and _is_constraint_conflict_error(exc):
                ok += 1
                SmartLogger.log(
                    "INFO",
                    "neo4j.bootstrap.table_key.skip.conflicting_constraint_already_exists",
                    category="neo4j.bootstrap",
                    params={"error": str(exc)},
                    max_inline_chars=0,
                )
                continue

            fail += 1
            SmartLogger.log(
                "WARNING",
                "neo4j.bootstrap.table_column_schema.warning",
                category="neo4j.bootstrap",
                params={"error": str(exc), "cypher": cypher.strip(), "traceback": traceback.format_exc()},
                max_inline_chars=0,
            )

    # 2) Query history constraints + vector indexes (used by similar query search)
    try:
        repo = Neo4jQueryRepository(session)
        await repo.setup_constraints()
        SmartLogger.log(
            "INFO",
            "neo4j.bootstrap.query_history_schema.done",
            category="neo4j.bootstrap",
            params={"embedding_dimension": int(settings.embedding_dimension)},
            max_inline_chars=0,
        )
    except Exception as exc:
        SmartLogger.log(
            "WARNING",
            "neo4j.bootstrap.query_history_schema.warning",
            category="neo4j.bootstrap",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )

    SmartLogger.log(
        "INFO",
        "neo4j.bootstrap.summary",
        category="neo4j.bootstrap",
        params={"table_column_ok": ok, "table_column_fail": fail},
        max_inline_chars=0,
    )


