"""Schema editing endpoints for user customization"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging

from app.deps import get_neo4j_session
from app.core.llm_factory import create_embedding_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schema-edit", tags=["Schema Editing"])


class TableUpdateRequest(BaseModel):
    """Request to update table metadata"""
    name: str
    schema: str = "public"
    description: Optional[str] = None


class ColumnUpdateRequest(BaseModel):
    """Request to update column metadata"""
    table_name: str
    table_schema: str = "public"
    column_name: str
    description: Optional[str] = None


class RelationshipRequest(BaseModel):
    """Request to add/update FK_TO_TABLE relationship
    
    Attributes:
        from_table: 소스 테이블명
        from_schema: 소스 스키마 (기본값: public)
        sourceColumn: FK 컬럼명 (소스 테이블의 컬럼)
        to_table: 타겟 테이블명
        to_schema: 타겟 스키마 (기본값: public)
        targetColumn: 참조 컬럼명 (타겟 테이블의 컬럼)
        type: 관계 유형 (many_to_one, one_to_one 등)
        description: 관계 설명 (선택)
    """
    from_table: str
    from_schema: str = "public"
    sourceColumn: str
    to_table: str
    to_schema: str = "public"
    targetColumn: str
    type: str = "many_to_one"  # many_to_one, one_to_one
    description: Optional[str] = None


class RelationshipResponse(BaseModel):
    """Response for relationship operations"""
    from_table: str
    sourceColumn: str
    to_table: str
    targetColumn: str
    type: str
    description: Optional[str] = None
    created: bool


@router.put("/tables/{table_name}/description")
async def update_table_description(
    table_name: str,
    request: TableUpdateRequest,
    neo4j_session=Depends(get_neo4j_session),
):
    """Update table description and regenerate embedding vector"""
    
    # 1. 테이블 정보 조회 (컬럼 목록 포함)
    info_query = """
    MATCH (t:Table {name: $table_name, schema: $schema})
    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
    RETURN t.name AS name, t.schema AS schema, collect(c.name) AS columns
    """
    
    info_result = await neo4j_session.run(
        info_query,
        table_name=table_name,
        schema=request.schema
    )
    info_records = await info_result.data()
    
    if not info_records:
        raise HTTPException(status_code=404, detail="Table not found")
    
    table_info = info_records[0]
    columns = table_info.get("columns", [])
    
    # 2. 임베딩 생성 (설명 + 컬럼 목록 포함)
    embedding = None
    if request.description:
        try:
            embedding_client = create_embedding_client()
            embed_text = embedding_client.format_table_text(
                table_name=table_name,
                description=request.description or "",
                columns=columns
            )
            embedding = await embedding_client.embed_text(embed_text)
            logger.info(f"[SchemaEdit] 테이블 '{table_name}' 임베딩 재생성 완료 (dim={len(embedding)})")
        except Exception as e:
            logger.warning(f"[SchemaEdit] 테이블 '{table_name}' 임베딩 생성 실패: {e}")
    
    # 3. 설명 및 벡터 업데이트 (vector 프로퍼티 사용 - 벡터 인덱스와 일치)
    if embedding:
        update_query = """
        MATCH (t:Table {name: $table_name, schema: $schema})
        SET t.description = $description, t.vector = $vector, t.updated_at = datetime()
        RETURN t.name AS name, t.description AS description
        """
        result = await neo4j_session.run(
            update_query, 
            table_name=table_name,
            schema=request.schema,
            description=request.description,
            vector=embedding
        )
    else:
        update_query = """
        MATCH (t:Table {name: $table_name, schema: $schema})
        SET t.description = $description
        RETURN t.name AS name, t.description AS description
        """
        result = await neo4j_session.run(
            update_query, 
            table_name=table_name,
            schema=request.schema,
            description=request.description
        )
    
    records = await result.data()
    if not records:
        raise HTTPException(status_code=404, detail="Table not found")
    
    return {
        "message": "Table description updated" + (" with embedding" if embedding else ""),
        "data": records[0],
        "embedding_updated": embedding is not None
    }


@router.put("/tables/{table_name}/columns/{column_name}/description")
async def update_column_description(
    table_name: str,
    column_name: str,
    request: ColumnUpdateRequest,
    neo4j_session=Depends(get_neo4j_session),
):
    """Update column description and regenerate embedding vector"""
    
    # 1. 컬럼 정보 조회
    info_query = """
    MATCH (t:Table {name: $table_name, schema: $schema})-[:HAS_COLUMN]->(c:Column {name: $column_name})
    RETURN c.name AS name, c.dtype AS dtype
    """
    
    info_result = await neo4j_session.run(
        info_query,
        table_name=table_name,
        schema=request.table_schema,
        column_name=column_name
    )
    info_records = await info_result.data()
    
    if not info_records:
        raise HTTPException(status_code=404, detail="Column not found")
    
    col_info = info_records[0]
    dtype = col_info.get("dtype", "unknown")
    
    # 2. 임베딩 생성
    embedding = None
    if request.description:
        try:
            embedding_client = create_embedding_client()
            embed_text = embedding_client.format_column_text(
                column_name=column_name,
                table_name=table_name,
                dtype=dtype,
                description=request.description or ""
            )
            embedding = await embedding_client.embed_text(embed_text)
            logger.info(f"[SchemaEdit] 컬럼 '{table_name}.{column_name}' 임베딩 재생성 완료 (dim={len(embedding)})")
        except Exception as e:
            logger.warning(f"[SchemaEdit] 컬럼 '{table_name}.{column_name}' 임베딩 생성 실패: {e}")
    
    # 3. 설명 및 벡터 업데이트 (vector 프로퍼티 사용 - 벡터 인덱스와 일치)
    if embedding:
        update_query = """
        MATCH (t:Table {name: $table_name, schema: $schema})-[:HAS_COLUMN]->(c:Column {name: $column_name})
        SET c.description = $description, c.vector = $vector, c.updated_at = datetime()
        RETURN c.name AS name, c.description AS description
        """
        result = await neo4j_session.run(
            update_query,
            table_name=table_name,
            schema=request.table_schema,
            column_name=column_name,
            description=request.description,
            vector=embedding
        )
    else:
        update_query = """
        MATCH (t:Table {name: $table_name, schema: $schema})-[:HAS_COLUMN]->(c:Column {name: $column_name})
        SET c.description = $description
        RETURN c.name AS name, c.description AS description
        """
        result = await neo4j_session.run(
            update_query,
            table_name=table_name,
            schema=request.table_schema,
            column_name=column_name,
            description=request.description
        )
    
    records = await result.data()
    if not records:
        raise HTTPException(status_code=404, detail="Column not found")
    
    return {
        "message": "Column description updated" + (" with embedding" if embedding else ""),
        "data": records[0],
        "embedding_updated": embedding is not None
    }


@router.post("/relationships", response_model=RelationshipResponse)
async def add_relationship(
    request: RelationshipRequest,
    neo4j_session=Depends(get_neo4j_session)
):
    """Add a new FK_TO_TABLE relationship between tables
    
    각 FK 매핑마다 별도의 FK_TO_TABLE 관계가 생성됩니다.
    속성: sourceColumn, targetColumn, type
    """
    
    # Check if both tables exist
    check_query = """
    MATCH (t1:Table {name: $from_table, schema: $from_schema})
    MATCH (t2:Table {name: $to_table, schema: $to_schema})
    RETURN t1.name AS from_table, t2.name AS to_table
    """
    
    check_result = await neo4j_session.run(
        check_query,
        from_table=request.from_table,
        from_schema=request.from_schema,
        to_table=request.to_table,
        to_schema=request.to_schema
    )
    
    check_records = await check_result.data()
    if not check_records:
        raise HTTPException(status_code=404, detail="One or both tables not found")
    
    # Check if relationship already exists (by sourceColumn and targetColumn)
    existing_query = """
    MATCH (t1:Table {name: $from_table, schema: $from_schema})-[r:FK_TO_TABLE]->(t2:Table {name: $to_table, schema: $to_schema})
    WHERE r.sourceColumn = $sourceColumn AND r.targetColumn = $targetColumn
    RETURN r
    """
    
    existing_result = await neo4j_session.run(
        existing_query,
        from_table=request.from_table,
        from_schema=request.from_schema,
        to_table=request.to_table,
        to_schema=request.to_schema,
        sourceColumn=request.sourceColumn,
        targetColumn=request.targetColumn
    )
    
    existing_records = await existing_result.data()
    if existing_records:
        return RelationshipResponse(
            from_table=request.from_table,
            sourceColumn=request.sourceColumn,
            to_table=request.to_table,
            targetColumn=request.targetColumn,
            type=request.type,
            description=request.description,
            created=False
        )
    
    # Create the relationship with unified attribute names
    # source='user': 사용자가 수동으로 추가 (실선 표시)
    create_query = """
    MATCH (t1:Table {name: $from_table, schema: $from_schema})
    MATCH (t2:Table {name: $to_table, schema: $to_schema})
    CREATE (t1)-[r:FK_TO_TABLE {
        sourceColumn: $sourceColumn,
        targetColumn: $targetColumn,
        type: $type,
        description: $description,
        source: 'user'
    }]->(t2)
    RETURN r
    """
    
    create_result = await neo4j_session.run(
        create_query,
        from_table=request.from_table,
        from_schema=request.from_schema,
        to_table=request.to_table,
        to_schema=request.to_schema,
        sourceColumn=request.sourceColumn,
        targetColumn=request.targetColumn,
        type=request.type,
        description=request.description
    )
    
    await create_result.data()
    
    return RelationshipResponse(
        from_table=request.from_table,
        sourceColumn=request.sourceColumn,
        to_table=request.to_table,
        targetColumn=request.targetColumn,
        type=request.type,
        description=request.description,
        created=True
    )


@router.delete("/relationships")
async def remove_relationship(
    from_table: str,
    from_schema: str = "public",
    sourceColumn: str = None,
    to_table: str = None,
    to_schema: str = "public",
    targetColumn: str = None,
    neo4j_session=Depends(get_neo4j_session)
):
    """Remove a FK_TO_TABLE relationship (only user-added ones, source='user')"""
    
    if sourceColumn and to_table and targetColumn:
        # Remove specific relationship
        query = """
        MATCH (t1:Table {name: $from_table, schema: $from_schema})-[r:FK_TO_TABLE]->(t2:Table {name: $to_table, schema: $to_schema})
        WHERE r.sourceColumn = $sourceColumn AND r.targetColumn = $targetColumn AND r.source = 'user'
        DELETE r
        RETURN count(r) AS deleted_count
        """
        
        params = {
            "from_table": from_table,
            "from_schema": from_schema,
            "sourceColumn": sourceColumn,
            "to_table": to_table,
            "to_schema": to_schema,
            "targetColumn": targetColumn
        }
    else:
        # Remove all user-added relationships for the table
        query = """
        MATCH (t1:Table {name: $from_table, schema: $from_schema})-[r:FK_TO_TABLE]->(t2:Table)
        WHERE r.source = 'user'
        DELETE r
        RETURN count(r) AS deleted_count
        """
        
        params = {
            "from_table": from_table,
            "from_schema": from_schema
        }
    
    result = await neo4j_session.run(query, **params)
    records = await result.data()
    
    deleted_count = records[0]["deleted_count"] if records else 0
    
    return {"message": f"Removed {deleted_count} relationship(s)"}


@router.get("/relationships/user-added")
async def list_user_added_relationships(
    neo4j_session=Depends(get_neo4j_session)
):
    """List all user-added FK_TO_TABLE relationships (source='user')"""
    query = """
    MATCH (t1:Table)-[r:FK_TO_TABLE]->(t2:Table)
    WHERE r.source = 'user'
    RETURN t1.name AS from_table,
           t1.schema AS from_schema,
           r.sourceColumn AS sourceColumn,
           t2.name AS to_table,
           t2.schema AS to_schema,
           r.targetColumn AS targetColumn,
           r.type AS type,
           r.source AS source,
           r.description AS description
    ORDER BY t1.name, r.sourceColumn
    """
    
    result = await neo4j_session.run(query)
    records = await result.data()
    
    return {
        "relationships": [
            {
                "from_table": r["from_table"],
                "from_schema": r["from_schema"],
                "sourceColumn": r["sourceColumn"],
                "to_table": r["to_table"],
                "to_schema": r["to_schema"],
                "targetColumn": r["targetColumn"],
                "type": r.get("type", "many_to_one"),
                "description": r.get("description")
            }
            for r in records
        ]
    }


@router.get("/relationships/all")
async def list_all_relationships(
    schema: str = None,
    neo4j_session=Depends(get_neo4j_session)
):
    """List all FK_TO_TABLE relationships (including DDL/procedure-derived ones)
    
    source 속성:
    - 'ddl': DDL에서 추출 (실선)
    - 'user': 사용자가 수동 추가 (실선)
    - 'procedure': 스토어드 프로시저 분석에서 추출 (점선)
    """
    if schema:
        query = """
        MATCH (t1:Table {schema: $schema})-[r:FK_TO_TABLE]->(t2:Table)
        RETURN t1.name AS from_table,
               t1.schema AS from_schema,
               r.sourceColumn AS sourceColumn,
               t2.name AS to_table,
               t2.schema AS to_schema,
               r.targetColumn AS targetColumn,
               r.type AS type,
               r.source AS source,
               r.description AS description
        ORDER BY t1.name, r.sourceColumn
        """
        result = await neo4j_session.run(query, schema=schema)
    else:
        query = """
        MATCH (t1:Table)-[r:FK_TO_TABLE]->(t2:Table)
        RETURN t1.name AS from_table,
               t1.schema AS from_schema,
               r.sourceColumn AS sourceColumn,
               t2.name AS to_table,
               t2.schema AS to_schema,
               r.targetColumn AS targetColumn,
               r.type AS type,
               r.source AS source,
               r.description AS description
        ORDER BY t1.name, r.sourceColumn
        """
        result = await neo4j_session.run(query)
    
    records = await result.data()
    
    return {
        "relationships": [
            {
                "from_table": r["from_table"],
                "from_schema": r["from_schema"],
                "sourceColumn": r["sourceColumn"],
                "to_table": r["to_table"],
                "to_schema": r["to_schema"],
                "targetColumn": r["targetColumn"],
                "type": r.get("type", "many_to_one"),
                "source": r.get("source", "ddl"),  # 기본값: ddl
                "description": r.get("description")
            }
            for r in records
        ]
    }
