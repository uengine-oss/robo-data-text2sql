"""
Finding 노드 - ReAct 에이전트의 성공적인 발견을 저장하고 재사용

그래프 스키마:
  (:Query)
      └──[:HAS_FINDING]──► (:Finding {intent, finding_type, description, vector})
                               ├──[:FROM_TABLE]──────► (:Table)
                               ├──[:SELECT_COLUMN]───► (:Column)
                               ├──[:FILTER_COLUMN]───► (:Column)
                               └──[:JOIN_PATH]───────► (:Column)

Finding Types:
  - table_discovery: 의미 있는 테이블을 발견함
  - column_discovery: 유용한 컬럼을 발견함
  - value_mapping: 자연어→코드 매핑을 발견함
  - join_path: 테이블 간 조인 경로를 발견함
  - data_insight: 실제 데이터에서 의미 있는 인사이트를 발견함

사용 흐름:
  1. ReAct 에이전트가 툴 실행 중 의미 있는 발견을 하면 Finding 생성
  2. 새 쿼리가 들어오면 intent 기반 벡터 검색으로 관련 Finding 조회
  3. 발견된 Finding의 테이블/컬럼 관계를 활용하여 빠르게 탐색
"""

import hashlib
import time
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.smart_logger import SmartLogger
from app.react.utils.log_sanitize import sanitize_for_log
from app.config import settings


class FindingNode(BaseModel):
    """Finding 노드 모델"""
    id: Optional[str] = None
    intent: str                              # 어떤 의도로 이것을 찾았는지
    finding_type: str                        # table_discovery, column_discovery, value_mapping, join_path, data_insight
    description: str                         # 무엇을 발견했는지
    reasoning: Optional[str] = None          # 왜 이것이 유용한지
    confidence: float = 1.0                  # 신뢰도 (0.0-1.0)
    usage_count: int = 1
    
    # 관련 엔티티들 (저장 시 릴레이션으로 변환됨)
    tables: List[str] = []                   # schema.table 형태
    columns: List[str] = []                  # schema.table.column 형태 (FQN)
    column_purposes: Dict[str, str] = {}     # column_fqn -> purpose (select, filter, join, group)
    
    # 값 매핑 (finding_type이 value_mapping인 경우)
    natural_value: Optional[str] = None
    code_value: Optional[str] = None


class FindingRepository:
    """Finding 노드를 Neo4j에 저장하고 검색하는 저장소"""
    
    def __init__(self, session, embedding_client=None):
        self.session = session
        self.embedding_client = embedding_client
    
    async def setup_constraints(self):
        """Finding 관련 제약조건 및 인덱스 설정"""
        constraints = [
            """
            CREATE CONSTRAINT finding_id IF NOT EXISTS
            FOR (f:Finding) REQUIRE f.id IS UNIQUE
            """,
            """
            CREATE INDEX finding_type_idx IF NOT EXISTS
            FOR (f:Finding) ON (f.finding_type)
            """,
            """
            CREATE INDEX finding_intent_idx IF NOT EXISTS
            FOR (f:Finding) ON (f.intent)
            """,
            """
            CREATE VECTOR INDEX finding_vec_index IF NOT EXISTS
            FOR (f:Finding) ON (f.vector)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
        ]
        
        for query in constraints:
            try:
                if "$dimensions" in query:
                    await self.session.run(query, dimensions=settings.embedding_dimension)
                else:
                    await self.session.run(query)
            except Exception as e:
                SmartLogger.log(
                    "WARNING",
                    "finding.setup_constraints.warning",
                    category="neo4j.finding",
                    params=sanitize_for_log({"cypher": query, "exception": repr(e)}),
                    max_inline_chars=0,
                )
    
    def _generate_finding_id(self, finding_type: str, intent: str, key_info: str = "") -> str:
        """Finding ID 생성"""
        content = f"{finding_type}:{intent}:{key_info}"
        return "f_" + hashlib.md5(content.encode()).hexdigest()[:12]
    
    async def save_finding(
        self,
        finding: FindingNode,
        query_id: Optional[str] = None,
    ) -> str:
        """Finding을 Neo4j에 저장하고 관련 엔티티와 연결"""
        started = time.perf_counter()
        
        # ID 생성
        key_info = ",".join(finding.tables[:3]) or ",".join(finding.columns[:3]) or ""
        finding_id = finding.id or self._generate_finding_id(
            finding.finding_type, finding.intent, key_info
        )
        
        SmartLogger.log(
            "INFO",
            "finding.save.start",
            category="neo4j.finding",
            params=sanitize_for_log({
                "finding_id": finding_id,
                "finding_type": finding.finding_type,
                "intent": finding.intent,
                "query_id": query_id,
            }),
            max_inline_chars=0,
        )
        
        try:
            # 1. 임베딩 생성
            vector = None
            if self.embedding_client:
                try:
                    vector = await self.embedding_client.embed_text(finding.intent[:2000])
                except Exception as emb_err:
                    SmartLogger.log(
                        "WARNING",
                        "finding.save.embedding_failed",
                        category="neo4j.finding",
                        params={"error": str(emb_err)},
                        max_inline_chars=0,
                    )
            
            # 2. Finding 노드 MERGE
            cypher = """
            MERGE (f:Finding {id: $id})
            ON CREATE SET 
                f.created_at = datetime(),
                f.usage_count = 1
            ON MATCH SET 
                f.usage_count = f.usage_count + 1,
                f.last_used_at = datetime()
            SET f.intent = $intent,
                f.finding_type = $finding_type,
                f.description = $description,
                f.reasoning = $reasoning,
                f.confidence = $confidence,
                f.natural_value = $natural_value,
                f.code_value = $code_value,
                f.updated_at = datetime()
            WITH f
            FOREACH (_ IN CASE WHEN $vector IS NOT NULL THEN [1] ELSE [] END |
                SET f.vector = $vector
            )
            RETURN f.id AS id
            """
            
            await self.session.run(
                cypher,
                id=finding_id,
                intent=finding.intent,
                finding_type=finding.finding_type,
                description=finding.description,
                reasoning=finding.reasoning,
                confidence=finding.confidence,
                natural_value=finding.natural_value,
                code_value=finding.code_value,
                vector=vector,
            )
            
            # 3. Query와 연결 (있는 경우)
            if query_id:
                await self.session.run(
                    """
                    MATCH (q:Query {id: $query_id})
                    MATCH (f:Finding {id: $finding_id})
                    MERGE (q)-[:HAS_FINDING]->(f)
                    """,
                    query_id=query_id,
                    finding_id=finding_id,
                )
            
            # 4. 테이블 연결
            for table_ref in finding.tables:
                parts = table_ref.split(".")
                if len(parts) >= 2:
                    schema, table_name = parts[0], parts[1]
                    await self.session.run(
                        """
                        MATCH (f:Finding {id: $finding_id})
                        MATCH (t:Table)
                        WHERE toLower(t.schema) = toLower($schema)
                          AND (toLower(t.name) = toLower($table_name) 
                               OR toLower(t.original_name) = toLower($table_name))
                        WITH f, t LIMIT 1
                        MERGE (f)-[:FROM_TABLE]->(t)
                        """,
                        finding_id=finding_id,
                        schema=schema,
                        table_name=table_name,
                    )
            
            # 5. 컬럼 연결 (용도별)
            for col_fqn in finding.columns:
                purpose = finding.column_purposes.get(col_fqn, "select").upper()
                rel_type = {
                    "SELECT": "SELECT_COLUMN",
                    "FILTER": "FILTER_COLUMN", 
                    "WHERE": "FILTER_COLUMN",
                    "JOIN": "JOIN_PATH",
                    "GROUP": "GROUP_COLUMN",
                }.get(purpose, "SELECT_COLUMN")
                
                await self.session.run(
                    f"""
                    MATCH (f:Finding {{id: $finding_id}})
                    MATCH (c:Column)
                    WHERE c.fqn IS NOT NULL AND toLower(c.fqn) = toLower($fqn)
                    WITH f, c LIMIT 1
                    MERGE (f)-[:{rel_type}]->(c)
                    """,
                    finding_id=finding_id,
                    fqn=col_fqn,
                )
            
            SmartLogger.log(
                "INFO",
                "finding.save.done",
                category="neo4j.finding",
                params=sanitize_for_log({
                    "finding_id": finding_id,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "tables_linked": len(finding.tables),
                    "columns_linked": len(finding.columns),
                }),
                max_inline_chars=0,
            )
            
            return finding_id
            
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "finding.save.error",
                category="neo4j.finding",
                params=sanitize_for_log({
                    "finding_id": finding_id,
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                }),
                max_inline_chars=0,
            )
            raise
    
    async def search_findings_by_intent(
        self,
        intent: str,
        finding_types: Optional[List[str]] = None,
        min_similarity: float = 0.5,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """의도(intent) 기반 벡터 검색으로 관련 Finding 조회"""
        started = time.perf_counter()
        
        if not self.embedding_client:
            SmartLogger.log(
                "WARNING",
                "finding.search.no_embedding_client",
                category="neo4j.finding",
                params={},
                max_inline_chars=0,
            )
            return []
        
        try:
            # 임베딩 생성
            embedding = await self.embedding_client.embed_text(intent[:2000])
            
            # 벡터 검색 + 타입 필터
            type_filter = ""
            if finding_types:
                type_filter = "AND f.finding_type IN $finding_types"
            
            cypher = f"""
            CALL db.index.vector.queryNodes('finding_vec_index', $k, $embedding)
            YIELD node, score
            WITH node AS f, score
            WHERE f:Finding
              AND score >= $min_score
              {type_filter}
            
            // 관련 테이블 수집
            OPTIONAL MATCH (f)-[:FROM_TABLE]->(t:Table)
            WITH f, score, COLLECT(DISTINCT t.schema + '.' + t.name) AS tables
            
            // 관련 컬럼 수집
            OPTIONAL MATCH (f)-[:SELECT_COLUMN|FILTER_COLUMN|JOIN_PATH|GROUP_COLUMN]->(c:Column)
            WITH f, score, tables, COLLECT(DISTINCT c.fqn) AS columns
            
            RETURN f.id AS id,
                   f.intent AS intent,
                   f.finding_type AS finding_type,
                   f.description AS description,
                   f.reasoning AS reasoning,
                   f.confidence AS confidence,
                   f.usage_count AS usage_count,
                   f.natural_value AS natural_value,
                   f.code_value AS code_value,
                   score AS similarity_score,
                   tables,
                   columns
            ORDER BY score DESC, f.usage_count DESC
            LIMIT $limit
            """
            
            params = {
                "k": limit * 2,  # 필터링 후 줄어들 수 있으므로 더 많이
                "embedding": embedding,
                "min_score": min_similarity,
                "limit": limit,
            }
            if finding_types:
                params["finding_types"] = finding_types
            
            result = await self.session.run(cypher, **params)
            findings = await result.data()
            
            SmartLogger.log(
                "INFO",
                "finding.search.done",
                category="neo4j.finding",
                params=sanitize_for_log({
                    "intent": intent[:100],
                    "finding_types": finding_types,
                    "found_count": len(findings),
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                }),
                max_inline_chars=0,
            )
            
            return findings
            
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "finding.search.error",
                category="neo4j.finding",
                params=sanitize_for_log({
                    "intent": intent[:100],
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                }),
                max_inline_chars=0,
            )
            return []
    
    async def get_findings_for_query(self, query_id: str) -> List[Dict[str, Any]]:
        """특정 쿼리와 연결된 Finding들 조회"""
        cypher = """
        MATCH (q:Query {id: $query_id})-[:HAS_FINDING]->(f:Finding)
        
        OPTIONAL MATCH (f)-[:FROM_TABLE]->(t:Table)
        WITH f, COLLECT(DISTINCT t.schema + '.' + t.name) AS tables
        
        OPTIONAL MATCH (f)-[:SELECT_COLUMN|FILTER_COLUMN|JOIN_PATH|GROUP_COLUMN]->(c:Column)
        WITH f, tables, COLLECT(DISTINCT c.fqn) AS columns
        
        RETURN f.id AS id,
               f.intent AS intent,
               f.finding_type AS finding_type,
               f.description AS description,
               f.reasoning AS reasoning,
               f.confidence AS confidence,
               f.usage_count AS usage_count,
               tables,
               columns
        ORDER BY f.confidence DESC
        """
        
        result = await self.session.run(cypher, query_id=query_id)
        return await result.data()
    
    async def get_findings_by_table(self, schema: str, table_name: str) -> List[Dict[str, Any]]:
        """특정 테이블과 관련된 Finding들 조회"""
        cypher = """
        MATCH (f:Finding)-[:FROM_TABLE]->(t:Table)
        WHERE toLower(t.schema) = toLower($schema)
          AND (toLower(t.name) = toLower($table_name) 
               OR toLower(t.original_name) = toLower($table_name))
        
        RETURN f.id AS id,
               f.intent AS intent,
               f.finding_type AS finding_type,
               f.description AS description,
               f.confidence AS confidence,
               f.usage_count AS usage_count
        ORDER BY f.usage_count DESC
        LIMIT 20
        """
        
        result = await self.session.run(cypher, schema=schema, table_name=table_name)
        return await result.data()


# 싱글톤 인스턴스
_finding_repo: Optional[FindingRepository] = None


def get_finding_repo(session, embedding_client=None) -> FindingRepository:
    """Finding 저장소 인스턴스 반환"""
    global _finding_repo
    if _finding_repo is None:
        _finding_repo = FindingRepository(session, embedding_client)
    return _finding_repo
