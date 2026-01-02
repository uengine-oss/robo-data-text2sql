"""Vectorization endpoint - fill only vector embeddings into existing Neo4j graph"""
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.deps import get_neo4j_session, get_openai_client
from app.core.embedding import EmbeddingClient
from app.ingest.to_neo4j import Neo4jSchemaLoader


router = APIRouter(prefix="/vectorize", tags=["Vectorization"])


class VectorizeRequest(BaseModel):
    """Request to vectorize existing graph metadata"""
    db_name: Optional[str] = "postgres"
    schema: Optional[str] = None
    include_tables: bool = True
    include_columns: bool = True
    reembed_existing: bool = False
    batch_size: int = 100


class VectorizeResponse(BaseModel):
    """Response from vectorization"""
    message: str
    status: str
    tables_vectorized: int
    columns_vectorized: int


def _parse_fqn(fqn: str) -> Dict[str, str]:
    """Parse 'schema.table.column' -> dict"""
    parts = fqn.split(".")
    if len(parts) < 3:
        return {"schema": "", "table": "", "column": fqn}
    return {"schema": parts[0], "table": parts[1], "column": parts[2]}


@router.post("", response_model=VectorizeResponse)
async def vectorize_graph(
    request: VectorizeRequest,
    neo4j_session=Depends(get_neo4j_session),
    openai_client=Depends(get_openai_client),
):
    """
    Analyze existing Neo4j graph and fill vector embeddings only.
    - Does NOT extract from RDB or modify graph topology
    - Optionally re-embeds even if vector already exists
    """
    try:
        embedding_client = EmbeddingClient(openai_client)
        loader = Neo4jSchemaLoader(neo4j_session, embedding_client)
        # Ensure vector indexes have correct dimensions (safe if already exist)
        await loader.setup_constraints_and_indexes()

        total_tables = 0
        total_columns = 0

        db_filter = request.db_name
        schema_filter = request.schema
        reembed = request.reembed_existing

        if request.include_tables:
            query_tables = """
            MATCH (t:Table)
            WHERE ($db IS NULL OR (t.db IS NOT NULL AND toLower(t.db) = toLower($db)))
              AND ($schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = toLower($schema)))
              AND ($reembed = true OR t.vector IS NULL OR size(t.vector) = 0)
            RETURN elementId(t) AS tid, t.db AS db, t.schema AS schema, t.name AS name, 
                   coalesce(t.description, '') AS description
            ORDER BY schema, name
            """
            result = await neo4j_session.run(
                query_tables,
                db=db_filter,
                schema=schema_filter,
                reembed=reembed,
            )
            tables: List[Dict[str, Any]] = [record.data() async for record in result]

            for item in tables:
                # Use description for table embedding
                description = item.get("description", "") or ""
                text = embedding_client.format_table_text(
                    table_name=item.get("name", ""),
                    description=description
                )
                vector = await embedding_client.embed_text(text)
                set_q = """
                MATCH (t)
                WHERE elementId(t) = $tid
                SET t.vector = $vector,
                    t.updated_at = datetime()
                """
                r2 = await neo4j_session.run(
                    set_q,
                    tid=item["tid"],
                    vector=vector,
                )
                await r2.consume()
            total_tables = len(tables)

        if request.include_columns:
            query_columns = """
            MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
            WHERE ($db IS NULL OR (t.db IS NOT NULL AND toLower(t.db) = toLower($db)))
              AND ($schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = toLower($schema)))
              AND ($reembed = true OR c.vector IS NULL OR size(c.vector) = 0)
            RETURN elementId(c) AS cid,
                   c.fqn AS fqn,
                   t.schema AS schema,
                   t.name AS table_name,
                   c.name AS column_name,
                   coalesce(c.description, '') AS description
            ORDER BY schema, table_name, fqn
            """
            result_c = await neo4j_session.run(
                query_columns,
                db=db_filter,
                schema=schema_filter,
                reembed=reembed,
            )
            cols: List[Dict[str, Any]] = [record.data() async for record in result_c]

            # Build texts using description ONLY for columns
            texts: List[str] = [(item.get("description") or "") for item in cols]

            batch_size = max(1, int(request.batch_size))
            embeddings: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                vecs = await embedding_client.embed_batch(batch)
                embeddings.extend(vecs)

            for item, vector in zip(cols, embeddings):
                set_c = """
                MATCH (c)
                WHERE elementId(c) = $cid
                SET c.vector = $vector,
                    c.updated_at = datetime()
                """
                rc = await neo4j_session.run(set_c, cid=item["cid"], vector=vector)
                await rc.consume()
            total_columns = len(cols)

        return VectorizeResponse(
            message="Vectorization completed",
            status="success",
            tables_vectorized=total_tables,
            columns_vectorized=total_columns,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")


