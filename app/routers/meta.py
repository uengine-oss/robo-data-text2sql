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
    datasource: Optional[str] = None
    description: str
    column_count: int
    materialized_view: Optional[str] = None


class ColumnInfo(BaseModel):
    """Column metadata"""
    name: str
    table_name: str
    dtype: str
    nullable: bool
    description: str


@router.get("/tables", response_model=List[TableInfo])
async def list_tables(
    datasource: str = Query(..., description="MindsDB datasource (required; Phase 1)"),
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
          AND toLower(COALESCE(t.datasource, t.db, '')) = toLower($datasource)
          AND (toLower(t.name) CONTAINS toLower($search) 
               OR toLower(t.description) CONTAINS toLower($search))
        WITH t, count(c) AS col_count
        RETURN t.name AS name,
               t.schema AS schema,
               COALESCE(t.datasource, t.db, '') AS datasource,
               t.description AS description,
               col_count AS column_count
        ORDER BY name
        LIMIT $limit
        """
        params = {"search": search, "schema": schema, "limit": limit, "datasource": datasource}
    else:
        # List all
        query = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE ($schema IS NULL OR t.schema = $schema)
          AND toLower(COALESCE(t.datasource, t.db, '')) = toLower($datasource)
        WITH t, count(c) AS col_count
        RETURN t.name AS name,
               t.schema AS schema,
               COALESCE(t.datasource, t.db, '') AS datasource,
               t.description AS description,
               col_count AS column_count
        ORDER BY name
        LIMIT $limit
        """
        params = {"schema": schema, "limit": limit, "datasource": datasource}
    
    result = await neo4j_session.run(query, **params)
    records = await result.data()
    
    # Some records may have description=None; coalesce to empty string
    return [
        TableInfo(
            name=r["name"],
            schema=r["schema"],
            datasource=(r.get("datasource") or ""),
            description=(r.get("description") or ""),
            column_count=r["column_count"]
        )
        for r in records
    ]


@router.get("/tables/{table_name}/columns", response_model=List[ColumnInfo])
async def list_table_columns(
    table_name: str,
    datasource: str = Query(..., description="MindsDB datasource (required; Phase 1)"),
    schema: str = Query("public", description="Table schema"),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    Get all columns for a specific table.
    """
    query = """
    MATCH (t:Table {name: $table_name, schema: $schema})-[:HAS_COLUMN]->(c:Column)
    WHERE toLower(COALESCE(t.datasource, t.db, '')) = toLower($datasource)
    RETURN c.name AS name,
           t.name AS table_name,
           c.dtype AS dtype,
           c.nullable AS nullable,
           c.description AS description
    ORDER BY c.name
    """
    
    result = await neo4j_session.run(query, table_name=table_name, schema=schema, datasource=datasource)
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
    datasource: str = Query(..., description="MindsDB datasource (required; Phase 1)"),
    search: str = Query(..., description="Search term for column names/descriptions"),
    limit: int = Query(50, ge=1, le=500),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    Search for columns across all tables.
    """
    query = """
    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
    WHERE toLower(COALESCE(t.datasource, t.db, '')) = toLower($datasource)
      AND (
        toLower(c.name) CONTAINS toLower($search)
       OR toLower(c.description) CONTAINS toLower($search)
      )
    RETURN c.name AS name,
           t.name AS table_name,
           c.dtype AS dtype,
           c.nullable AS nullable,
           c.description AS description
    ORDER BY c.name
    LIMIT $limit
    """
    
    result = await neo4j_session.run(query, datasource=datasource, search=search, limit=limit)
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


@router.get("/datasources", response_model=List[str])
async def list_datasources(
    neo4j_session=Depends(get_neo4j_session),
):
    """List available datasources (derived from Neo4j Table.db / Table.datasource)."""
    query = """
    MATCH (t:Table)
    WITH DISTINCT COALESCE(t.datasource, t.db, '') AS ds
    WHERE ds IS NOT NULL AND ds <> ''
    RETURN ds AS datasource
    ORDER BY datasource
    """
    result = await neo4j_session.run(query)
    records = await result.data()
    return [str(r.get("datasource") or "") for r in records if str(r.get("datasource") or "").strip()]


@router.get("/datasources/{datasource}/schemas", response_model=List[str])
async def list_schemas_by_datasource(
    datasource: str,
    neo4j_session=Depends(get_neo4j_session),
):
    """List schemas for a datasource."""
    query = """
    MATCH (t:Table)
    WHERE toLower(COALESCE(t.datasource, t.db, '')) = toLower($datasource)
    WITH DISTINCT COALESCE(t.schema, '') AS sch
    WHERE sch IS NOT NULL AND sch <> ''
    RETURN sch AS schema
    ORDER BY schema
    """
    result = await neo4j_session.run(query, datasource=datasource)
    records = await result.data()
    return [str(r.get("schema") or "") for r in records if str(r.get("schema") or "").strip()]


@router.get("/datasources/{datasource}/schemas/{schema}/tables", response_model=List[TableInfo])
async def list_tables_by_datasource_and_schema(
    datasource: str,
    schema: str,
    limit: int = Query(500, ge=1, le=1000),
    neo4j_session=Depends(get_neo4j_session),
):
    """List tables for a datasource + schema."""
    query = """
    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
    WHERE toLower(COALESCE(t.datasource, t.db, '')) = toLower($datasource)
      AND toLower(COALESCE(t.schema,'')) = toLower($schema)
    WITH t, count(c) AS col_count
    RETURN t.name AS name,
           t.schema AS schema,
           COALESCE(t.datasource, t.db, '') AS datasource,
           COALESCE(t.description, '') AS description,
           col_count AS column_count
    ORDER BY name
    LIMIT $limit
    """
    result = await neo4j_session.run(query, datasource=datasource, schema=schema, limit=int(limit))
    records = await result.data()
    return [
        TableInfo(
            name=str(r.get("name") or ""),
            schema=str(r.get("schema") or ""),
            datasource=str(r.get("datasource") or ""),
            description=str(r.get("description") or ""),
            column_count=int(r.get("column_count") or 0),
        )
        for r in records
    ]


@router.get("/objecttypes", response_model=List[TableInfo])
async def list_objecttypes(
    datasource: str = Query(..., description="MindsDB datasource (required; Phase 1)"),
    limit: int = Query(500, ge=1, le=1000),
    neo4j_session=Depends(get_neo4j_session),
):
    """
    List ObjectType nodes (domain-layer tables).  
    NOTE: ObjectType datasource binding is data-dependent; we keep datasource as a required API contract (D9),
    but filtering may be best-effort depending on how ObjectType nodes are stored in Neo4j.
    """
    query = """
    MATCH (t:ObjectType)-[:HAS_COLUMN]->(c:Column)
    WITH t, count(c) AS col_count
    RETURN COALESCE(t.name,'') AS name,
           COALESCE(t.schema,'dw') AS schema,
           COALESCE(t.materializedView,'') AS materialized_view,
           COALESCE(t.description,'') AS description,
           col_count AS column_count
    ORDER BY name
    LIMIT $limit
    """
    result = await neo4j_session.run(query, limit=int(limit))
    records = await result.data()
    # Best-effort: expose datasource as request-provided value for client-side consistency.
    return [
        TableInfo(
            name=str(r.get("name") or ""),
            schema=str(r.get("schema") or ""),
            datasource=str(datasource or ""),
            description=str(r.get("description") or ""),
            column_count=int(r.get("column_count") or 0),
            materialized_view=(str(r.get("materialized_view") or "") or None),
        )
        for r in records
    ]

