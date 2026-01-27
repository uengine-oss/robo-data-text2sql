"""Metadata endpoints for schema exploration"""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional

from app.deps import get_neo4j_session


router = APIRouter(prefix="/meta", tags=["Metadata"])


class TableInfo(BaseModel):
    """Table metadata"""
    name: str
    schema: str
    description: str
    column_count: int
    datasource: Optional[str] = None  # 데이터 소스 이름 (예: mysql_jjy)


class ColumnInfo(BaseModel):
    """Column metadata"""
    name: str
    table_name: str
    dtype: str
    nullable: bool
    description: str


@router.get("/tables", response_model=List[TableInfo])
async def list_tables(
    search: Optional[str] = Query(None, description="Search term for table names/descriptions"),
    schema: Optional[str] = Query(None, description="Filter by schema"),
    limit: int = Query(50, ge=1, le=500),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    List available tables with optional search and filtering.
    """
    if search:
        # Text search using CONTAINS
        query = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE ($schema IS NULL OR t.schema = $schema)
          AND (toLower(t.name) CONTAINS toLower($search) 
               OR toLower(t.description) CONTAINS toLower($search))
        WITH t, count(c) AS col_count
        RETURN t.name AS name,
               t.schema AS schema,
               coalesce(t.datasource, '') AS datasource,
               t.description AS description,
               col_count AS column_count
        ORDER BY name
        LIMIT $limit
        """
        params = {"search": search, "schema": schema, "limit": limit}
    else:
        # List all
        query = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE $schema IS NULL OR t.schema = $schema
        WITH t, count(c) AS col_count
        RETURN t.name AS name,
               t.schema AS schema,
               coalesce(t.datasource, '') AS datasource,
               t.description AS description,
               col_count AS column_count
        ORDER BY name
        LIMIT $limit
        """
        params = {"schema": schema, "limit": limit}
    
    result = await neo4j_session.run(query, **params)
    records = await result.data()
    
    # Some records may have description=None; coalesce to empty string
    return [
        TableInfo(
            name=r["name"],
            schema=r["schema"],
            datasource=r.get("datasource") or None,
            description=(r.get("description") or ""),
            column_count=r["column_count"]
        )
        for r in records
    ]


@router.get("/tables/{table_name}/columns", response_model=List[ColumnInfo])
async def list_table_columns(
    table_name: str,
    schema: str = Query("public", description="Table schema"),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    Get all columns for a specific table.
    """
    query = """
    MATCH (t:Table {name: $table_name, schema: $schema})-[:HAS_COLUMN]->(c:Column)
    RETURN c.name AS name,
           t.name AS table_name,
           c.dtype AS dtype,
           c.nullable AS nullable,
           c.description AS description
    ORDER BY c.name
    """
    
    result = await neo4j_session.run(query, table_name=table_name, schema=schema)
    records = await result.data()
    
    return [
        ColumnInfo(
            name=r["name"],
            table_name=r["table_name"],
            dtype=r.get("dtype") or "unknown",
            nullable=r.get("nullable") if r.get("nullable") is not None else True,
            description=r.get("description") or ""
        )
        for r in records
    ]


@router.get("/columns", response_model=List[ColumnInfo])
async def search_columns(
    search: str = Query(..., description="Search term for column names/descriptions"),
    limit: int = Query(50, ge=1, le=500),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    Search for columns across all tables.
    """
    query = """
    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
    WHERE toLower(c.name) CONTAINS toLower($search)
       OR toLower(c.description) CONTAINS toLower($search)
    RETURN c.name AS name,
           t.name AS table_name,
           c.dtype AS dtype,
           c.nullable AS nullable,
           c.description AS description
    ORDER BY c.name
    LIMIT $limit
    """
    
    result = await neo4j_session.run(query, search=search, limit=limit)
    records = await result.data()
    
    return [
        ColumnInfo(
            name=r["name"],
            table_name=r["table_name"],
            dtype=r.get("dtype") or "unknown",
            nullable=r.get("nullable") if r.get("nullable") is not None else True,
            description=r.get("description") or ""
        )
        for r in records
    ]

