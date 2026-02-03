"""Main /ask endpoint for natural language to SQL"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time

from app.deps import get_neo4j_session, get_db_connection, get_openai_client
from app.core.embedding import EmbeddingClient
from app.core.graph_search import GraphSearcher, format_subschema_for_prompt
from app.core.prompt import SQLChain
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.viz import VizRecommender


router = APIRouter(prefix="/ask", tags=["Query"])


class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    question: str = Field(..., description="Natural language question")
    db_key: str = Field(default="default", description="Database identifier")
    visual_pref: Optional[List[str]] = Field(default=None, description="Preferred chart types")
    limit: Optional[int] = Field(default=1000, description="Row limit")
    include_explain: bool = Field(default=False, description="Include query execution plan")


class ProvenanceInfo(BaseModel):
    """Provenance information about how the query was generated"""
    tables: List[str]
    columns: List[str]
    neo4j_paths: List[str]
    vector_matches: List[Dict[str, Any]]
    prompt_snapshot_id: str


class PerformanceMetrics(BaseModel):
    """Performance metrics for the request"""
    embedding_ms: float
    graph_search_ms: float
    llm_ms: float
    sql_ms: float
    total_ms: float


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    sql: str
    table: Dict[str, Any]
    charts: List[Dict[str, Any]]
    provenance: ProvenanceInfo
    perf: PerformanceMetrics
    warnings: Optional[List[str]] = None


@router.post("", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    neo4j_session=Depends(get_neo4j_session),
    db_conn=Depends(get_db_connection),
    openai_client=Depends(get_openai_client)
):
    """
    Convert natural language question to SQL, execute it, and return results with visualizations.
    """
    warnings = []
    total_start = time.time()
    
    generated_sql: Optional[str] = None
    validated_sql: Optional[str] = None

    try:
        # 1. Generate query embedding
        embed_start = time.time()
        embedding_client = EmbeddingClient(openai_client)
        query_embedding = await embedding_client.embed_text(request.question)
        embedding_ms = (time.time() - embed_start) * 1000
        
        # 2. Search Neo4j graph for relevant schema
        graph_start = time.time()
        searcher = GraphSearcher(neo4j_session)
        subschema = await searcher.build_subschema(query_embedding)
        graph_search_ms = (time.time() - graph_start) * 1000
        
        if not subschema.tables:
            raise HTTPException(
                status_code=404,
                detail="No relevant tables found for your question. Please rephrase or check schema ingestion."
            )
        
        # 3. Generate SQL using LLM
        llm_start = time.time()
        schema_text = format_subschema_for_prompt(subschema)
        join_hints = "\n".join(subschema.join_hints) if subschema.join_hints else "No specific join hints."
        
        sql_chain = SQLChain()
        generated_sql = await sql_chain.generate_sql(
            question=request.question,
            schema_text=schema_text,
            join_hints=join_hints
        )
        llm_ms = (time.time() - llm_start) * 1000
        
        # 4. Validate SQL
        sql_guard = SQLGuard()
        allowed_tables = [t.name for t in subschema.tables]
        
        try:
            validated_sql, _ = sql_guard.validate(generated_sql, allowed_tables=allowed_tables)
        except SQLValidationError as e:
            # Include generated SQL in error response
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"SQL validation failed: {str(e)}",
                    "sql": generated_sql or ""
                }
            )
        
        # 4.5. MindsDB용 datasource 프리픽스 추가 (schema.table → datasource.schema.table)
        from app.react.tools.utils import add_datasource_prefix, quote_uppercase_identifiers
        original_sql = validated_sql
        validated_sql = add_datasource_prefix(validated_sql)
        if original_sql != validated_sql:
            print(f"[Text2SQL] Datasource prefix 추가: {original_sql[:100]}... → {validated_sql[:100]}...")
        
        # 4.6. MindsDB용 식별자 백틱 추가 (대문자/한글 식별자 처리)
        validated_sql = quote_uppercase_identifiers(validated_sql)
        
        # 5. Execute SQL
        sql_start = time.time()
        executor = SQLExecutor()
        
        try:
            results = await executor.execute_query(db_conn, validated_sql)  # type: ignore[arg-type]
        except SQLExecutionError as e:
            # Return the attempted SQL even on failure
            raise HTTPException(
                status_code=500,
                detail={
                    "message": f"SQL execution failed: {str(e)}",
                    "sql": validated_sql or generated_sql or ""
                }
            )
        
        sql_ms = (time.time() - sql_start) * 1000
        
        # 6. Generate visualizations
        viz_recommender = VizRecommender()
        charts = viz_recommender.recommend_charts(
            columns=results["columns"],
            rows=results["rows"]
        )
        
        # Apply visual preferences if provided
        if request.visual_pref:
            charts = [c for c in charts if c.get("type") in request.visual_pref] or charts
        
        # 7. Build provenance
        prompt_snapshot_id = f"ps_{int(time.time())}_{hash(request.question) % 10000}"
        
        # 테이블명 생성: datasource가 있으면 datasource.schema.table, 없으면 schema.table
        table_names = []
        for t in subschema.tables:
            if t.datasource:
                table_names.append(f"{t.datasource}.{t.schema}.{t.name}")
            else:
                table_names.append(f"{t.schema}.{t.name}")
        
        provenance = ProvenanceInfo(
            tables=table_names,
            columns=[f"{c.table_name}.{c.name}" for c in subschema.columns[:10]],
            neo4j_paths=[f"{fk['from_table']} -> {fk['to_table']}" for fk in subschema.fk_relationships],
            vector_matches=[
                {"node": f"Table:{t.name}", "score": round(t.score, 3)}
                for t in subschema.tables[:5]
            ],
            prompt_snapshot_id=prompt_snapshot_id
        )
        
        # 8. Build performance metrics
        total_ms = (time.time() - total_start) * 1000
        perf = PerformanceMetrics(
            embedding_ms=round(embedding_ms, 2),
            graph_search_ms=round(graph_search_ms, 2),
            llm_ms=round(llm_ms, 2),
            sql_ms=round(sql_ms, 2),
            total_ms=round(total_ms, 2)
        )
        
        # Format results for JSON
        table_data = executor.format_results_for_json(results)
        
        return AskResponse(
            sql=validated_sql,
            table=table_data,
            charts=charts,
            provenance=provenance,
            perf=perf,
            warnings=warnings if warnings else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # On unexpected errors, still include the best-known SQL
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Unexpected error: {str(e)}",
                "sql": validated_sql or generated_sql or ""
            }
        )

