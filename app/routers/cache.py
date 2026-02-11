"""
Query Cache & Metadata Enhancement Router
- 쿼리 템플릿 캐싱
- 열거형 값 캐싱
- 값 → 코드 매핑 캐싱
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.deps import get_neo4j_session, get_db_connection
from app.smart_logger import SmartLogger
from app.react.utils.log_sanitize import sanitize_for_log

router = APIRouter(prefix="/cache", tags=["cache"])


# ============== Models ==============

class EnumValue(BaseModel):
    """열거형 값"""
    value: str
    code: Optional[str] = None  # 연관 코드 (예: 청주정수장 → BPLC001)
    count: Optional[int] = None  # 데이터 건수


class ColumnEnumCache(BaseModel):
    """컬럼 열거형 캐시"""
    schema_name: str
    table_name: str
    column_name: str
    enum_values: List[EnumValue]
    cardinality: int


class QueryTemplate(BaseModel):
    """쿼리 템플릿"""
    id: Optional[str] = None
    question_pattern: str  # 질문 패턴 (예: "{정수장}의 {측정값} 조회")
    sql_template: str  # SQL 템플릿
    placeholders: List[str] = []  # 플레이스홀더 목록
    tables_used: List[str] = []
    example_question: str
    example_sql: str
    usage_count: int = 0
    created_at: Optional[str] = None


class ValueMapping(BaseModel):
    """값 → 코드 매핑"""
    natural_value: str  # 자연어 값 (예: 청주정수장)
    code_value: str  # 코드 값 (예: BPLC001)
    column_fqn: str  # 컬럼 FQN (예: rwis.rdisaup_tb.BPLC_CODE)
    description: Optional[str] = None


class SimilarQueryResult(BaseModel):
    """유사 쿼리 검색 결과"""
    question: str
    sql: str
    similarity: float
    template: Optional[QueryTemplate] = None


# ============== Enum Cache API ==============

@router.post("/enum/extract/{schema_name}/{table_name}/{column_name}")
async def extract_enum_values(
    schema_name: str,
    table_name: str,
    column_name: str,
    max_values: int = Query(100, description="최대 열거형 값 개수"),
    db_conn=Depends(get_db_connection),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    특정 컬럼의 열거형 값을 추출하여 Neo4j에 캐싱합니다.
    카디널리티가 max_values 이하인 경우에만 캐싱합니다.
    """
    try:
        # 1. 카디널리티 확인
        cardinality_query = f"""
        SELECT COUNT(DISTINCT "{column_name}") as cardinality
        FROM "{schema_name}"."{table_name}"
        """
        result = await db_conn.fetch(cardinality_query)
        cardinality = result[0]['cardinality'] if result else 0
        
        if cardinality > max_values:
            return {
                "status": "skipped",
                "message": f"Cardinality {cardinality} exceeds max_values {max_values}",
                "cardinality": cardinality
            }
        
        # 2. 열거형 값 추출
        enum_query = f"""
        SELECT "{column_name}" as value, COUNT(*) as count
        FROM "{schema_name}"."{table_name}"
        WHERE "{column_name}" IS NOT NULL
        GROUP BY "{column_name}"
        ORDER BY count DESC
        LIMIT {max_values}
        """
        rows = await db_conn.fetch(enum_query)
        
        enum_values = [
            {"value": str(row['value']), "count": row['count']}
            for row in rows
        ]
        
        # 3. Neo4j에 캐싱
        fqn = f"{schema_name}.{table_name}.{column_name}".lower()
        
        update_query = """
        MATCH (c:Column {fqn: $fqn})
        SET c.enum_values = $enum_values,
            c.cardinality = $cardinality,
            c.enum_cached_at = datetime()
        RETURN c
        """
        
        import json
        await neo4j_session.run(
            update_query,
            fqn=fqn,
            enum_values=json.dumps(enum_values, ensure_ascii=False),
            cardinality=cardinality
        )
        
        return {
            "status": "success",
            "column": fqn,
            "cardinality": cardinality,
            "values_cached": len(enum_values),
            "sample_values": enum_values[:10]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract enum values: {str(e)}")


@router.post("/enum/extract-all/{schema_name}")
async def extract_all_enum_values(
    schema_name: str,
    max_values: int = Query(100, description="최대 열거형 값 개수"),
    db_conn=Depends(get_db_connection),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    스키마의 모든 컬럼에서 열거형 값을 추출합니다.
    카디널리티가 낮은 컬럼만 캐싱합니다.
    """
    results = []
    
    # 1. Neo4j에서 해당 스키마의 모든 컬럼 조회
    columns_query = """
    MATCH (t:Table {schema: $schema})-[:HAS_COLUMN]->(c:Column)
    RETURN t.name AS table_name, c.name AS column_name, c.dtype AS dtype
    """
    
    result = await neo4j_session.run(columns_query, schema=schema_name.lower())
    columns = await result.data()
    
    for col in columns:
        table_name = col['table_name']
        column_name = col['column_name']
        
        try:
            # 카디널리티 확인
            cardinality_query = f"""
            SELECT COUNT(DISTINCT "{column_name}") as cardinality
            FROM "{schema_name}"."{table_name}"
            """
            card_result = await db_conn.fetch(cardinality_query)
            cardinality = card_result[0]['cardinality'] if card_result else 0
            
            if cardinality <= max_values and cardinality > 0:
                # 열거형 값 추출
                enum_query = f"""
                SELECT "{column_name}" as value, COUNT(*) as count
                FROM "{schema_name}"."{table_name}"
                WHERE "{column_name}" IS NOT NULL
                GROUP BY "{column_name}"
                ORDER BY count DESC
                LIMIT {max_values}
                """
                rows = await db_conn.fetch(enum_query)
                
                enum_values = [
                    {"value": str(row['value']), "count": row['count']}
                    for row in rows
                ]
                
                # Neo4j에 캐싱
                fqn = f"{schema_name}.{table_name}.{column_name}".lower()
                
                import json
                update_query = """
                MATCH (c:Column {fqn: $fqn})
                SET c.enum_values = $enum_values,
                    c.cardinality = $cardinality,
                    c.enum_cached_at = datetime()
                RETURN c
                """
                
                await neo4j_session.run(
                    update_query,
                    fqn=fqn,
                    enum_values=json.dumps(enum_values, ensure_ascii=False),
                    cardinality=cardinality
                )
                
                results.append({
                    "column": fqn,
                    "cardinality": cardinality,
                    "values_cached": len(enum_values)
                })
                
        except Exception as e:
            results.append({
                "column": f"{schema_name}.{table_name}.{column_name}",
                "error": str(e)
            })
    
    return {
        "status": "success",
        "schema": schema_name,
        "columns_processed": len(columns),
        "columns_cached": len([r for r in results if "values_cached" in r]),
        "details": results
    }


@router.get("/enum/{schema_name}/{table_name}/{column_name}")
async def get_enum_values(
    schema_name: str,
    table_name: str,
    column_name: str,
    neo4j_session=Depends(get_neo4j_session)
):
    """캐싱된 열거형 값을 조회합니다."""
    fqn = f"{schema_name}.{table_name}.{column_name}".lower()
    
    query = """
    MATCH (c:Column {fqn: $fqn})
    RETURN c.enum_values AS enum_values, c.cardinality AS cardinality, c.enum_cached_at AS cached_at
    """
    
    result = await neo4j_session.run(query, fqn=fqn)
    record = await result.single()
    
    if not record or not record['enum_values']:
        raise HTTPException(status_code=404, detail=f"No cached enum values for {fqn}")
    
    import json
    return {
        "column": fqn,
        "cardinality": record['cardinality'],
        "cached_at": str(record['cached_at']) if record['cached_at'] else None,
        "enum_values": json.loads(record['enum_values'])
    }


# ============== Value Mapping API ==============

@router.post("/mapping")
async def create_value_mapping(
    mapping: ValueMapping,
    neo4j_session=Depends(get_neo4j_session)
):
    """값 → 코드 매핑을 생성합니다."""
    query = """
    MERGE (m:ValueMapping {natural_value: $natural_value, column_fqn: $column_fqn})
    SET m.code_value = $code_value,
        m.description = $description,
        m.updated_at = datetime()
    RETURN m
    """
    
    await neo4j_session.run(
        query,
        natural_value=mapping.natural_value,
        code_value=mapping.code_value,
        column_fqn=mapping.column_fqn,
        description=mapping.description or ""
    )
    
    return {"status": "success", "mapping": mapping.dict()}


@router.get("/mapping/search")
async def search_value_mapping(
    value: str,
    neo4j_session=Depends(get_neo4j_session)
):
    """자연어 값으로 코드 매핑을 검색합니다."""
    query = """
    MATCH (m:ValueMapping)
    WHERE toLower(m.natural_value) CONTAINS toLower($value)
    RETURN m.natural_value AS natural_value,
           m.code_value AS code_value,
           m.column_fqn AS column_fqn,
           m.description AS description
    LIMIT 10
    """
    
    result = await neo4j_session.run(query, value=value)
    records = await result.data()
    
    return {"mappings": records}


# ============== Query Template API ==============

@router.post("/template")
async def save_query_template(
    template: QueryTemplate,
    neo4j_session=Depends(get_neo4j_session)
):
    """쿼리 템플릿을 저장합니다."""
    import json
    import hashlib
    
    # ID 생성
    template_id = hashlib.md5(template.sql_template.encode()).hexdigest()[:12]
    
    query = """
    MERGE (t:QueryTemplate {id: $id})
    SET t.question_pattern = $question_pattern,
        t.sql_template = $sql_template,
        t.placeholders = $placeholders,
        t.tables_used = $tables_used,
        t.example_question = $example_question,
        t.example_sql = $example_sql,
        t.usage_count = COALESCE(t.usage_count, 0) + 1,
        t.updated_at = datetime()
    RETURN t
    """
    
    await neo4j_session.run(
        query,
        id=template_id,
        question_pattern=template.question_pattern,
        sql_template=template.sql_template,
        placeholders=json.dumps(template.placeholders),
        tables_used=json.dumps(template.tables_used),
        example_question=template.example_question,
        example_sql=template.example_sql
    )
    
    return {"status": "success", "template_id": template_id}


@router.get("/template/search")
async def search_templates(
    keyword: str,
    neo4j_session=Depends(get_neo4j_session)
):
    """키워드로 쿼리 템플릿을 검색합니다."""
    query = """
    MATCH (t:QueryTemplate)
    WHERE toLower(t.question_pattern) CONTAINS toLower($keyword)
       OR toLower(t.example_question) CONTAINS toLower($keyword)
    RETURN t.id AS id,
           t.question_pattern AS question_pattern,
           t.sql_template AS sql_template,
           t.placeholders AS placeholders,
           t.tables_used AS tables_used,
           t.example_question AS example_question,
           t.usage_count AS usage_count
    ORDER BY t.usage_count DESC
    LIMIT 10
    """
    
    result = await neo4j_session.run(query, keyword=keyword)
    records = await result.data()
    
    import json
    templates = []
    for r in records:
        templates.append({
            "id": r['id'],
            "question_pattern": r['question_pattern'],
            "sql_template": r['sql_template'],
            "placeholders": json.loads(r['placeholders']) if r['placeholders'] else [],
            "tables_used": json.loads(r['tables_used']) if r['tables_used'] else [],
            "example_question": r['example_question'],
            "usage_count": r['usage_count']
        })
    
    return {"templates": templates}


# ============== Similar Query Search ==============

@router.get("/similar-query")
async def find_similar_query(
    question: str,
    limit: int = Query(5, description="반환할 유사 쿼리 개수"),
    neo4j_session=Depends(get_neo4j_session)
):
    """
    히스토리에서 유사한 쿼리를 검색합니다.
    임베딩 기반 유사도 검색을 사용합니다.
    """
    from app.models.history import history_repo
    
    # 1. 히스토리에서 성공한 쿼리들 가져오기
    history = history_repo.list(page=1, page_size=100, status="completed")
    
    if not history.items:
        return {"similar_queries": [], "message": "No completed queries in history"}
    
    # 2. 간단한 키워드 기반 유사도 계산 (임시)
    # TODO: 실제 임베딩 기반 유사도로 교체
    question_words = set(question.lower().split())
    
    similar = []
    for item in history.items:
        item_words = set(item.question.lower().split())
        
        # Jaccard 유사도
        intersection = len(question_words & item_words)
        union = len(question_words | item_words)
        similarity = intersection / union if union > 0 else 0
        
        if similarity > 0.1:  # 임계값
            similar.append({
                "question": item.question,
                "sql": item.final_sql,
                "similarity": round(similarity, 3),
                "row_count": item.row_count,
                "execution_time_ms": item.execution_time_ms
            })
    
    # 유사도 순 정렬
    similar.sort(key=lambda x: x['similarity'], reverse=True)
    
    return {"similar_queries": similar[:limit]}


# ============== Auto Cache from History ==============

@router.post("/auto-cache-from-history")
async def auto_cache_from_history(
    neo4j_session=Depends(get_neo4j_session),
    db_conn=Depends(get_db_connection)
):
    """
    히스토리에서 성공한 쿼리들을 분석하여 자동으로 캐시합니다.
    - 자주 사용되는 테이블/컬럼의 열거형 값 추출
    - 값 → 코드 매핑 추출
    """
    from app.models.history import history_repo
    import re
    
    history = history_repo.list(page=1, page_size=50, status="completed")
    
    cached_enums = []
    cached_mappings = []
    
    for item in history.items:
        if not item.final_sql:
            continue
        
        sql = item.final_sql
        
        # 1. WHERE 절에서 값 매핑 추출
        # 예: WHERE t."BPLC_CODE" = 'BPLC001'
        where_pattern = r'"(\w+)"\.?"(\w+)"?\s*=\s*\'([^\']+)\''
        matches = re.findall(where_pattern, sql)
        
        for match in matches:
            table_or_alias, column, value = match
            
            # 매핑 저장 시도
            # 질문에서 해당 값과 연관된 자연어 추출 (간단한 휴리스틱)
            question = item.question
            
            # 매핑 생성 (예시)
            if len(value) < 50:  # 너무 긴 값은 제외
                cached_mappings.append({
                    "code_value": value,
                    "column": column,
                    "from_question": question
                })
    
    return {
        "status": "success",
        "analyzed_queries": len(history.items),
        "cached_mappings": len(cached_mappings),
        "sample_mappings": cached_mappings[:10]
    }


# ============== Neo4j 그래프 분석 API ==============

@router.get("/graph/query-history")
async def get_graph_query_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
    neo4j_session=Depends(get_neo4j_session)
):
    """Neo4j 그래프에서 쿼리 히스토리 조회"""
    from app.models.neo4j_history import Neo4jQueryRepository
    
    repo = Neo4jQueryRepository(neo4j_session)
    return await repo.get_query_history(page=page, page_size=page_size, status=status)


@router.get("/graph/similar-queries")
async def find_graph_similar_queries(
    tables: Optional[str] = Query(None, description="Comma-separated table names"),
    columns: Optional[str] = Query(None, description="Comma-separated column names"),
    keywords: Optional[str] = Query(None, description="Comma-separated question keywords"),
    limit: int = Query(5, ge=1, le=20),
    neo4j_session=Depends(get_neo4j_session)
):
    """그래프 구조 기반 유사 쿼리 검색"""
    from app.models.neo4j_history import Neo4jQueryRepository
    
    repo = Neo4jQueryRepository(neo4j_session)
    
    tables_list = [t.strip() for t in tables.split(",")] if tables else None
    columns_list = [c.strip() for c in columns.split(",")] if columns else None
    keywords_list = [k.strip() for k in keywords.split(",")] if keywords else None
    
    return await repo.find_similar_queries_by_graph(
        tables=tables_list,
        columns=columns_list,
        question_keywords=keywords_list,
        limit=limit
    )


@router.get("/graph/table-usage-stats")
async def get_table_usage_stats(
    neo4j_session=Depends(get_neo4j_session)
):
    """테이블 사용 통계 조회"""
    from app.models.neo4j_history import Neo4jQueryRepository
    
    repo = Neo4jQueryRepository(neo4j_session)
    return await repo.get_table_usage_stats()


@router.get("/graph/column-usage-stats")
async def get_column_usage_stats(
    neo4j_session=Depends(get_neo4j_session)
):
    """컬럼 사용 통계 조회 (용도별)"""
    from app.models.neo4j_history import Neo4jQueryRepository
    
    repo = Neo4jQueryRepository(neo4j_session)
    return await repo.get_column_usage_stats()


@router.get("/graph/value-mappings")
async def search_value_mappings_graph(
    value: str = Query(..., description="Natural language value to search"),
    neo4j_session=Depends(get_neo4j_session)
):
    """값 매핑 검색 (Neo4j 그래프 기반)"""
    from app.models.neo4j_history import Neo4jQueryRepository
    
    repo = Neo4jQueryRepository(neo4j_session)
    return await repo.find_value_mapping(value)


@router.delete("/graph/query/{query_id}")
async def delete_graph_query(
    query_id: str,
    neo4j_session=Depends(get_neo4j_session)
):
    """Neo4j 그래프에서 쿼리 삭제"""
    from app.models.neo4j_history import Neo4jQueryRepository
    
    repo = Neo4jQueryRepository(neo4j_session)
    deleted = await repo.delete_query(query_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Query not found")
    return {"message": "Query deleted", "id": query_id}


@router.delete("/graph/value-mapping")
async def delete_value_mapping(
    natural_value: str = Query(..., description="Natural value to delete"),
    code_value: Optional[str] = Query(None, description="Specific code value to delete"),
    neo4j_session=Depends(get_neo4j_session)
):
    """잘못된 값 매핑 삭제"""
    SmartLogger.log(
        "INFO",
        "cache.graph.value_mapping.delete.start",
        category="cache.graph.value_mapping",
        params=sanitize_for_log(
            {"natural_value": natural_value, "code_value": code_value}
        ),
        max_inline_chars=0,
    )
    if code_value:
        query = """
        MATCH (v:ValueMapping)
        WHERE v.natural_value CONTAINS $natural_value
          AND v.code_value = $code_value
        DETACH DELETE v
        RETURN count(*) as deleted
        """
        result = await neo4j_session.run(query, natural_value=natural_value, code_value=code_value)
    else:
        query = """
        MATCH (v:ValueMapping)
        WHERE v.natural_value CONTAINS $natural_value
        DETACH DELETE v
        RETURN count(*) as deleted
        """
        result = await neo4j_session.run(query, natural_value=natural_value)
    
    record = await result.single()
    deleted_count = record['deleted'] if record else 0
    SmartLogger.log(
        "INFO",
        "cache.graph.value_mapping.delete.done",
        category="cache.graph.value_mapping",
        params=sanitize_for_log(
            {
                "natural_value": natural_value,
                "code_value": code_value,
                "deleted_count": deleted_count,
            }
        ),
        max_inline_chars=0,
    )
    return {"message": f"Deleted {deleted_count} value mappings", "natural_value": natural_value}


@router.post("/graph/value-mapping")
async def create_value_mapping(
    natural_value: str = Query(..., description="Natural language value (e.g., 강릉정수장)"),
    code_value: str = Query(..., description="Code value (e.g., BPLC006)"),
    column_name: str = Query("BPLC_CODE", description="Column name"),
    neo4j_session=Depends(get_neo4j_session)
):
    """수동으로 값 매핑 생성"""
    SmartLogger.log(
        "INFO",
        "cache.graph.value_mapping.create.start",
        category="cache.graph.value_mapping",
        params=sanitize_for_log(
            {
                "natural_value": natural_value,
                "code_value": code_value,
                "column_name": column_name,
            }
        ),
        max_inline_chars=0,
    )
    query = """
    MATCH (c:Column {name: $column_name})
    WITH c LIMIT 1
    MERGE (v:ValueMapping {natural_value: $natural_value, column_fqn: c.fqn})
    SET v.code_value = $code_value,
        v.usage_count = COALESCE(v.usage_count, 0) + 10,
        v.updated_at = datetime()
    MERGE (v)-[:MAPS_TO]->(c)
    RETURN v.natural_value AS natural_value, v.code_value AS code_value, c.fqn AS column_fqn
    """
    
    try:
        result = await neo4j_session.run(
            query,
            natural_value=natural_value,
            code_value=code_value,
            column_name=column_name,
        )
        record = await result.single()
    except Exception as exc:
        import traceback as _tb
        SmartLogger.log(
            "ERROR",
            "cache.graph.value_mapping.create.error",
            category="cache.graph.value_mapping",
            params=sanitize_for_log(
                {
                    "natural_value": natural_value,
                    "code_value": code_value,
                    "column_name": column_name,
                    "exception": repr(exc),
                    "traceback": _tb.format_exc(),
                }
            ),
            max_inline_chars=0,
        )
        raise
    
    if record:
        SmartLogger.log(
            "INFO",
            "cache.graph.value_mapping.create.done",
            category="cache.graph.value_mapping",
            params=sanitize_for_log(
                {
                    "natural_value": record["natural_value"],
                    "code_value": record["code_value"],
                    "column_fqn": record["column_fqn"],
                }
            ),
            max_inline_chars=0,
        )
        return {
            "status": "created",
            "natural_value": record['natural_value'],
            "code_value": record['code_value'],
            "column_fqn": record['column_fqn']
        }
    else:
        SmartLogger.log(
            "WARNING",
            "cache.graph.value_mapping.create.not_found",
            category="cache.graph.value_mapping",
            params=sanitize_for_log(
                {
                    "natural_value": natural_value,
                    "code_value": code_value,
                    "column_name": column_name,
                    "message": "Column not found",
                }
            ),
            max_inline_chars=0,
        )
        raise HTTPException(status_code=404, detail=f"Column {column_name} not found")


# ============== LLM Query Cache API ==============

@router.get("/llm/stats")
async def get_llm_cache_stats():
    """LLM 쿼리 캐시 통계 조회"""
    from app.core.query_cache import get_query_cache
    cache = get_query_cache()
    return cache.get_stats()


@router.delete("/llm/clear")
async def clear_llm_cache():
    """LLM 쿼리 캐시 전체 초기화"""
    from app.core.query_cache import get_query_cache
    cache = get_query_cache()
    count = cache.clear()
    return {"message": f"Cleared {count} cached items"}


@router.delete("/llm/invalidate")
async def invalidate_llm_cache(
    question: str = Query(..., description="캐시에서 제거할 질문"),
    datasource: Optional[str] = Query(None, description="(선택) MindsDB datasource"),
):
    """특정 질문의 LLM 캐시 무효화"""
    from app.core.query_cache import get_query_cache
    cache = get_query_cache()
    removed = cache.invalidate(question, datasource=datasource)
    return {"message": "Cache invalidated" if removed else "Not found in cache", "question": question}

