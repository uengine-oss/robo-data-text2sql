"""Query history API router - Neo4j 기반"""
from typing import Optional, List, Any, Dict
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import json

from app.deps import get_neo4j_session
from app.models.neo4j_history import Neo4jQueryRepository


router = APIRouter(prefix="/history", tags=["history"])


# ============== Pydantic Models ==============

class QueryStepModel(BaseModel):
    """쿼리 실행 단계 모델"""
    iteration: Optional[int] = None
    reasoning: Optional[str] = None
    tool_name: Optional[str] = None
    tool_result_preview: Optional[str] = None


class ExecutionResultModel(BaseModel):
    """쿼리 실행 결과 모델"""
    columns: List[str] = []
    rows: List[List[Any]] = []
    row_count: int = 0
    execution_time_ms: float = 0


class QueryHistoryItem(BaseModel):
    """히스토리 항목 모델"""
    id: str  # Neo4j에서는 해시 ID 사용
    question: str
    final_sql: Optional[str] = None
    validated_sql: Optional[str] = None
    execution_result: Optional[ExecutionResultModel] = None
    row_count: Optional[int] = None
    status: str = "completed"
    error_message: Optional[str] = None
    steps_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    steps: Optional[List[QueryStepModel]] = None
    tables_used: List[str] = []


class QueryHistoryResponse(BaseModel):
    """히스토리 목록 응답"""
    items: List[QueryHistoryItem]
    total: int
    page: int
    page_size: int


class QueryHistoryCreate(BaseModel):
    """히스토리 생성 요청"""
    question: str
    final_sql: Optional[str] = None
    validated_sql: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    row_count: Optional[int] = None
    status: str = "completed"
    error_message: Optional[str] = None
    steps_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    steps: Optional[List[Dict[str, Any]]] = None


# ============== API Endpoints ==============

@router.get("", response_model=QueryHistoryResponse)
async def list_history(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search in question or SQL"),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    List query history from Neo4j with pagination and optional filters.
    """
    repo = Neo4jQueryRepository(neo4j_session)
    
    skip = (page - 1) * page_size
    
    # 검색어가 있으면 검색 쿼리 사용
    if search:
        query = """
        MATCH (q:Query)
        WHERE ($status IS NULL OR q.status = $status)
          AND (toLower(q.question) CONTAINS toLower($search) 
               OR toLower(COALESCE(q.sql, '')) CONTAINS toLower($search))
        
        OPTIONAL MATCH (q)-[:USES_TABLE]->(t:Table)
        WITH q, COLLECT(DISTINCT t.name) AS tables
        
        RETURN q.id AS id,
               q.question AS question,
               q.sql AS final_sql,
               q.sql AS validated_sql,
               q.status AS status,
               q.row_count AS row_count,
               q.execution_time_ms AS execution_time_ms,
               q.steps_count AS steps_count,
               q.error_message AS error_message,
               q.steps AS steps_json,
               toString(q.created_at) AS created_at,
               toString(q.updated_at) AS updated_at,
               tables AS tables_used
        ORDER BY q.created_at DESC
        SKIP $skip
        LIMIT $limit
        """
    else:
        query = """
        MATCH (q:Query)
        WHERE $status IS NULL OR q.status = $status
        
        OPTIONAL MATCH (q)-[:USES_TABLE]->(t:Table)
        WITH q, COLLECT(DISTINCT t.name) AS tables
        
        RETURN q.id AS id,
               q.question AS question,
               q.sql AS final_sql,
               q.sql AS validated_sql,
               q.status AS status,
               q.row_count AS row_count,
               q.execution_time_ms AS execution_time_ms,
               q.steps_count AS steps_count,
               q.error_message AS error_message,
               q.steps AS steps_json,
               toString(q.created_at) AS created_at,
               toString(q.updated_at) AS updated_at,
               tables AS tables_used
        ORDER BY q.created_at DESC
        SKIP $skip
        LIMIT $limit
        """
    
    result = await neo4j_session.run(
        query,
        status=status,
        search=search,
        skip=skip,
        limit=page_size
    )
    records = await result.data()
    
    # 총 개수 조회
    if search:
        count_query = """
        MATCH (q:Query)
        WHERE ($status IS NULL OR q.status = $status)
          AND (toLower(q.question) CONTAINS toLower($search) 
               OR toLower(COALESCE(q.sql, '')) CONTAINS toLower($search))
        RETURN COUNT(q) AS total
        """
    else:
        count_query = """
        MATCH (q:Query)
        WHERE $status IS NULL OR q.status = $status
        RETURN COUNT(q) AS total
        """
    
    count_result = await neo4j_session.run(count_query, status=status, search=search)
    count_record = await count_result.single()
    total = count_record['total'] if count_record else 0
    
    # 응답 변환
    items = []
    for record in records:
        # steps JSON 파싱
        steps = None
        if record.get('steps_json'):
            try:
                steps_data = json.loads(record['steps_json'])
                steps = [QueryStepModel(**s) if isinstance(s, dict) else s for s in steps_data]
            except (json.JSONDecodeError, TypeError):
                steps = None
        
        items.append(QueryHistoryItem(
            id=record['id'],
            question=record['question'],
            final_sql=record.get('final_sql'),
            validated_sql=record.get('validated_sql'),
            execution_result=None,  # 별도 조회 필요시 확장
            row_count=record.get('row_count'),
            status=record.get('status', 'completed'),
            error_message=record.get('error_message'),
            steps_count=record.get('steps_count'),
            execution_time_ms=record.get('execution_time_ms'),
            created_at=record.get('created_at'),
            updated_at=record.get('updated_at'),
            metadata=None,
            steps=steps,
            tables_used=record.get('tables_used', [])
        ))
    
    return QueryHistoryResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{query_id}", response_model=QueryHistoryItem)
async def get_history(
    query_id: str,
    neo4j_session=Depends(get_neo4j_session)
):
    """Get a specific history entry by ID."""
    
    query = """
    MATCH (q:Query {id: $query_id})
    
    OPTIONAL MATCH (q)-[:USES_TABLE]->(t:Table)
    WITH q, COLLECT(DISTINCT t.name) AS tables
    
    RETURN q.id AS id,
           q.question AS question,
           q.sql AS final_sql,
           q.sql AS validated_sql,
           q.status AS status,
           q.row_count AS row_count,
           q.execution_time_ms AS execution_time_ms,
           q.steps_count AS steps_count,
           q.error_message AS error_message,
           q.steps AS steps_json,
           toString(q.created_at) AS created_at,
           toString(q.updated_at) AS updated_at,
           tables AS tables_used
    """
    
    result = await neo4j_session.run(query, query_id=query_id)
    record = await result.single()
    
    if not record:
        raise HTTPException(status_code=404, detail="History entry not found")
    
    # steps JSON 파싱
    steps = None
    if record.get('steps_json'):
        try:
            steps_data = json.loads(record['steps_json'])
            steps = [QueryStepModel(**s) if isinstance(s, dict) else s for s in steps_data]
        except (json.JSONDecodeError, TypeError):
            steps = None
    
    return QueryHistoryItem(
        id=record['id'],
        question=record['question'],
        final_sql=record.get('final_sql'),
        validated_sql=record.get('validated_sql'),
        execution_result=None,
        row_count=record.get('row_count'),
        status=record.get('status', 'completed'),
        error_message=record.get('error_message'),
        steps_count=record.get('steps_count'),
        execution_time_ms=record.get('execution_time_ms'),
        created_at=record.get('created_at'),
        updated_at=record.get('updated_at'),
        metadata=None,
        steps=steps,
        tables_used=record.get('tables_used', [])
    )


@router.post("", response_model=QueryHistoryItem)
async def create_history(
    entry: QueryHistoryCreate,
    neo4j_session=Depends(get_neo4j_session)
):
    """
    Create a new history entry in Neo4j.
    """
    repo = Neo4jQueryRepository(neo4j_session)
    
    query_id = await repo.save_query(
        question=entry.question,
        sql=entry.final_sql or entry.validated_sql,
        status=entry.status,
        metadata=entry.metadata,
        row_count=entry.row_count,
        execution_time_ms=entry.execution_time_ms,
        steps_count=entry.steps_count,
        error_message=entry.error_message,
        steps=entry.steps,
        # History is not a verification source. Keep it unverified to prevent cache poisoning.
        verified=False,
        verified_source="history_api",
    )
    
    # 저장된 항목 조회하여 반환
    return await get_history(query_id, neo4j_session)


@router.delete("/{query_id}")
async def delete_history(
    query_id: str,
    neo4j_session=Depends(get_neo4j_session)
):
    """Delete a specific history entry."""
    repo = Neo4jQueryRepository(neo4j_session)
    
    deleted = await repo.delete_query(query_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History entry not found")
    
    return {"message": "History entry deleted", "id": query_id}


@router.delete("")
async def delete_all_history(
    neo4j_session=Depends(get_neo4j_session)
):
    """Delete all history entries."""
    
    query = """
    MATCH (q:Query)
    WITH q, COUNT(q) AS count
    DETACH DELETE q
    RETURN count
    """
    
    # 먼저 개수 조회
    count_result = await neo4j_session.run("MATCH (q:Query) RETURN COUNT(q) AS count")
    count_record = await count_result.single()
    count = count_record['count'] if count_record else 0
    
    # 삭제 실행
    await neo4j_session.run("MATCH (q:Query) DETACH DELETE q")
    
    return {"message": f"Deleted {count} history entries", "count": count}


# ============== 통계 및 분석 엔드포인트 ==============

@router.get("/stats/tables")
async def get_table_usage_stats(
    neo4j_session=Depends(get_neo4j_session)
):
    """테이블 사용 통계 조회"""
    repo = Neo4jQueryRepository(neo4j_session)
    return await repo.get_table_usage_stats()


@router.get("/stats/columns")
async def get_column_usage_stats(
    neo4j_session=Depends(get_neo4j_session)
):
    """컬럼 사용 통계 조회"""
    repo = Neo4jQueryRepository(neo4j_session)
    return await repo.get_column_usage_stats()
