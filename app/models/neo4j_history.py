"""
Neo4j 기반 쿼리 히스토리 온톨로지 저장소

쿼리와 테이블/컬럼 간의 관계를 그래프로 모델링하여:
- 유사 쿼리 검색 정확도 향상
- 쿼리 패턴 분석
- 값 매핑 자동 활용

그래프 스키마:
  (:Query {question, sql, status, ...})
      ├──[:USES_TABLE]────────► (:Table {name, schema})
      ├──[:SELECTS]───────────► (:Column {fqn})
      ├──[:FILTERS {op, value}]► (:Column)
      ├──[:AGGREGATES {fn}]───► (:Column)
      ├──[:JOINS_ON]──────────► (:Column)
      └──[:GROUPS_BY]─────────► (:Column)
  
  (:ValueMapping {natural_value, code_value})
      └──[:MAPS_TO]───────────► (:Column)
  
  (:QueryPattern {pattern, template_sql})
      └──[:APPLIES_TO]────────► (:Table)
"""

import re
import hashlib
import time
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field
from app.smart_logger import SmartLogger
from app.react.utils.log_sanitize import sanitize_for_log
from app.config import settings


class QueryNode(BaseModel):
    """Neo4j Query 노드 모델"""
    id: Optional[str] = None
    question: str
    sql: Optional[str] = None
    status: str = "completed"  # completed, error
    row_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    steps_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    
    # 추출된 메타데이터
    tables_used: List[str] = []
    columns_used: List[Dict[str, str]] = []
    value_mappings: List[Dict[str, str]] = []


class ValueMappingNode(BaseModel):
    """값 → 코드 매핑 노드"""
    natural_value: str  # 자연어 값 (예: 청주정수장)
    code_value: str     # 코드 값 (예: BPLC001)
    column_fqn: str     # 컬럼 FQN
    usage_count: int = 1


class QueryPatternNode(BaseModel):
    """쿼리 패턴 템플릿 노드"""
    id: Optional[str] = None
    pattern: str        # 질문 패턴 (예: "{정수장}의 {측정값} 조회")
    template_sql: str   # SQL 템플릿
    placeholders: List[str] = []
    tables_used: List[str] = []
    usage_count: int = 1


class Neo4jQueryRepository:
    """Neo4j 기반 쿼리 히스토리 저장소"""
    
    def __init__(self, session):
        self.session = session
    
    async def setup_constraints(self):
        """Neo4j 제약조건 및 인덱스 설정"""
        constraints = [
            """
            CREATE CONSTRAINT query_id IF NOT EXISTS
            FOR (q:Query) REQUIRE q.id IS UNIQUE
            """,
            """
            CREATE CONSTRAINT value_mapping_key IF NOT EXISTS
            FOR (v:ValueMapping) REQUIRE (v.natural_value, v.column_fqn) IS NODE KEY
            """,
            """
            CREATE INDEX query_question_idx IF NOT EXISTS
            FOR (q:Query) ON (q.question)
            """,
            """
            CREATE INDEX query_created_idx IF NOT EXISTS
            FOR (q:Query) ON (q.created_at)
            """,
            """
            CREATE INDEX value_mapping_natural_idx IF NOT EXISTS
            FOR (v:ValueMapping) ON (v.natural_value)
            """
            ,
            """
            CREATE VECTOR INDEX query_vec_index IF NOT EXISTS
            FOR (q:Query) ON (q.vector)
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
                # 제약조건이 이미 존재할 수 있음
                SmartLogger.log(
                    "WARNING",
                    "neo4j_history.setup_constraints.warning",
                    category="neo4j.history.setup_constraints",
                    params=sanitize_for_log({"cypher": query, "exception": repr(e)}),
                    max_inline_chars=0,
                )
    
    @staticmethod
    def _normalize_question_for_id(question: str) -> str:
        # Keep semantics strict (string-equality policy) but avoid trivial whitespace drift.
        return (question or "").strip()

    def _generate_query_id(self, db: str, question: str) -> str:
        """쿼리 ID 생성: db + question(문자열 동일) 기준 단일 Query 노드"""
        qn = self._normalize_question_for_id(question)
        content = f"{db}:{qn}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    @staticmethod
    def _status_rank(status: Optional[str]) -> int:
        s = (status or "").lower().strip()
        if s == "completed":
            return 0
        if s == "error":
            return 2
        return 1

    @classmethod
    def _candidate_rank(
        cls,
        *,
        status: Optional[str],
        steps_count: Optional[int],
        execution_time_ms: Optional[float],
        best_run_at_ms: Optional[int],
    ) -> Tuple[int, int, float, int]:
        # Lower tuple is better; tie-breaker prefers newer run => use negative timestamp.
        sc = int(steps_count) if steps_count is not None else 10**9
        et = float(execution_time_ms) if execution_time_ms is not None else 1e18
        ts = int(best_run_at_ms) if best_run_at_ms is not None else 0
        return (cls._status_rank(status), sc, et, -ts)
    
    def _extract_sql_components(self, sql: str) -> Dict[str, Any]:
        """SQL에서 컴포넌트 추출 (테이블, 컬럼, 조건 등)"""
        if not sql:
            return {}
        
        components = {
            'select_columns': [],
            'filter_conditions': [],
            'aggregate_functions': [],
            'join_conditions': [],
            'group_by_columns': [],
            'tables': []
        }
        
        sql_upper = sql.upper()
        
        # 테이블 추출 (FROM, JOIN 절)
        table_pattern = r'(?:FROM|JOIN)\s+"?(\w+)"?\."?(\w+)"?'
        for match in re.finditer(table_pattern, sql, re.IGNORECASE):
            schema, table = match.groups()
            components['tables'].append({
                'schema': schema.lower(),
                'name': table.lower()
            })
        
        # 집계 함수 추출
        agg_pattern = r'(AVG|SUM|COUNT|MAX|MIN)\s*\(\s*"?(\w+)"?\."?"?(\w+)"?\s*\)'
        for match in re.finditer(agg_pattern, sql, re.IGNORECASE):
            fn, alias_or_col, col = match.groups()
            components['aggregate_functions'].append({
                'function': fn.upper(),
                'column': col.lower()
            })
        
        # WHERE 조건 추출
        where_pattern = r'"?(\w+)"?\."?"?(\w+)"?\s*(=|LIKE|>|<|>=|<=|IN)\s*[\'"]?([^\'"\s,\)]+)[\'"]?'
        for match in re.finditer(where_pattern, sql, re.IGNORECASE):
            alias_or_col, col, op, value = match.groups()
            if col.upper() not in ['SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'JOIN']:
                components['filter_conditions'].append({
                    'column': col.lower(),
                    'operator': op.upper(),
                    'value': value
                })
        
        # GROUP BY 추출
        group_pattern = r'GROUP\s+BY\s+(.+?)(?:ORDER|LIMIT|HAVING|$)'
        group_match = re.search(group_pattern, sql, re.IGNORECASE | re.DOTALL)
        if group_match:
            group_cols = group_match.group(1)
            col_pattern = r'"?(\w+)"?\."?"?(\w+)"?'
            for match in re.finditer(col_pattern, group_cols):
                alias, col = match.groups()
                components['group_by_columns'].append(col.lower())
        
        return components
    
    def _extract_value_mappings(self, question: str, sql: str, metadata: Dict) -> List[Dict]:
        """질문과 SQL에서 값 매핑 추출"""
        mappings = []
        
        if not sql:
            return mappings
        
        # SQL에서 조건절 값 추출
        condition_pattern = r'"?(\w+)"?\."?"?(\w+)"?\s*=\s*\'([^\']+)\''
        for match in re.finditer(condition_pattern, sql):
            table_or_alias, column, value = match.groups()
            
            # 질문에서 관련 자연어 값 찾기
            # 예: "청주" in question and "BPLC001" in value
            question_words = question.split()
            for word in question_words:
                if len(word) >= 2 and word not in ['의', '을', '를', '에서', '으로']:
                    # 값이 코드 형태인지 확인 (영문+숫자)
                    if re.match(r'^[A-Z]+\d+$', value, re.IGNORECASE):
                        mappings.append({
                            'natural_value': word,
                            'code_value': value,
                            'column': column.lower()
                        })
        
        # 메타데이터에서 identified_values 활용
        if metadata and 'identified_values' in metadata:
            for val in metadata.get('identified_values', []):
                if val.get('actual_value') and val.get('user_term'):
                    mappings.append({
                        'natural_value': val['user_term'],
                        'code_value': val['actual_value'],
                        'column': val.get('column', '').lower()
                    })
        
        return mappings
    
    async def save_query(
        self,
        question: str,
        sql: Optional[str],
        status: str,
        metadata: Optional[Dict] = None,
        row_count: Optional[int] = None,
        execution_time_ms: Optional[float] = None,
        steps_count: Optional[int] = None,
        error_message: Optional[str] = None,
        steps: Optional[List[Dict]] = None,  # legacy: raw steps (will be minimized)
        *,
        db: Optional[str] = None,
        steps_summary: Optional[str] = None,
        value_mappings: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """쿼리와 관련 메타데이터를 Neo4j에 저장"""
        import json as json_module

        started = time.perf_counter()
        # IMPORTANT: Neo4j schema graph uses `Table.db` as a DB TYPE label (oracle/postgresql/mysql),
        # not the physical database name. Cache graph logic must follow `react_caching_db_type`.
        db_name = (db or settings.react_caching_db_type or "").strip()
        query_id = self._generate_query_id(db_name, question)
        now_ms = int(time.time() * 1000)

        # Overwrite policy: completed > steps_count(min) > execution_time_ms(min) > best_run_at_ms(latest)
        existing_result = await self.session.run(
            """
            MATCH (q:Query {id: $id})
            RETURN q.status AS status,
                   q.steps_count AS steps_count,
                   q.execution_time_ms AS execution_time_ms,
                   q.best_run_at_ms AS best_run_at_ms
            """,
            id=query_id,
        )
        existing_record = await existing_result.single()
        existing_rank: Optional[Tuple[int, int, float, int]] = None
        if existing_record:
            existing_rank = self._candidate_rank(
                status=existing_record.get("status"),
                steps_count=existing_record.get("steps_count"),
                execution_time_ms=existing_record.get("execution_time_ms"),
                best_run_at_ms=existing_record.get("best_run_at_ms"),
            )
        incoming_rank = self._candidate_rank(
            status=status,
            steps_count=steps_count,
            execution_time_ms=execution_time_ms,
            best_run_at_ms=now_ms,
        )
        should_overwrite = existing_rank is None or incoming_rank < existing_rank

        if steps_summary is None:
            steps_summary = self._minimize_steps_summary(steps, json_module=json_module)

        SmartLogger.log(
            "INFO",
            "neo4j_history.save_query.start",
            category="neo4j.history.save_query",
            params=sanitize_for_log(
                {
                    "query_id": query_id,
                    "db": db_name,
                    "should_overwrite": should_overwrite,
                    "question": question,
                    "sql": sql,
                    "status": status,
                    "row_count": row_count,
                    "execution_time_ms": execution_time_ms,
                    "steps_count": steps_count,
                    "error_message": error_message,
                }
            ),
            max_inline_chars=0,
        )

        query_upsert = """
        MERGE (q:Query {id: $id})
        ON CREATE SET q.created_at = datetime(),
                      q.created_at_ms = $now_ms
        SET q.question = $question,
            q.question_norm = $question_norm,
            q.last_seen_at = datetime(),
            q.last_seen_at_ms = $now_ms,
            q.seen_count = COALESCE(q.seen_count, 0) + 1
        WITH q
        FOREACH (_ IN CASE WHEN $overwrite THEN [1] ELSE [] END |
            SET q.sql = $sql,
                q.status = $status,
                q.row_count = $row_count,
                q.execution_time_ms = $execution_time_ms,
                q.steps_count = $steps_count,
                q.error_message = $error_message,
                q.steps_summary = $steps_summary,
                q.updated_at = datetime(),
                q.updated_at_ms = $now_ms,
                q.best_run_at_ms = $now_ms,
                q.value_mappings_count = $value_mappings_count,
                q.value_mapping_terms = $value_mapping_terms
        )
        RETURN q.id AS id
        """

        try:
            await self.session.run(
                query_upsert,
                id=query_id,
                question=question,
                question_norm=self._normalize_question_for_id(question),
                sql=sql,
                status=status,
                row_count=row_count,
                execution_time_ms=execution_time_ms,
                steps_count=steps_count,
                error_message=error_message,
                steps_summary=steps_summary,
                overwrite=should_overwrite,
                now_ms=now_ms,
                value_mappings_count=len(value_mappings or []),
                value_mapping_terms=[(m.get("natural_value") or "") for m in (value_mappings or []) if isinstance(m, dict)][:20],
            )

            # Only refresh graph relations when overwriting the best entry.
            if should_overwrite:
                await self.session.run(
                    """
                    MATCH (q:Query {id: $query_id})-[r]->()
                    WHERE type(r) IN ['USES_TABLE', 'SELECTS', 'FILTERS', 'AGGREGATES', 'GROUPS_BY', 'JOINS_ON']
                    DELETE r
                    """,
                    query_id=query_id,
                )

                tables_used: List[str] = []
                columns_used: List[str] = []

                if metadata:
                    for table in (metadata.get("identified_tables") or []):
                        schema = (table.get("schema") or "").strip()
                        name = (table.get("name") or "").strip()
                        if not schema or not name:
                            continue
                        tables_used.append(f"{schema}.{name}")
                        await self.session.run(
                            """
                            MATCH (q:Query {id: $query_id})
                            MATCH (t:Table)
                            WHERE toLower(t.db) = toLower($db)
                              AND toLower(t.schema) = toLower($schema)
                              AND (
                                  (t.name IS NOT NULL AND toLower(t.name) = toLower($table_name))
                                  OR (t.original_name IS NOT NULL AND toLower(t.original_name) = toLower($table_name))
                              )
                            WITH q, t LIMIT 1
                            MERGE (q)-[:USES_TABLE]->(t)
                            """,
                            query_id=query_id,
                            db=db_name,
                            schema=schema,
                            table_name=name,
                        )

                    for col in (metadata.get("identified_columns") or []):
                        purpose = (col.get("purpose") or "SELECT").upper()
                        if "FILTER" in purpose or "WHERE" in purpose:
                            rel_type = "FILTERS"
                        elif "GROUP" in purpose:
                            rel_type = "GROUPS_BY"
                        elif any(fn in purpose for fn in ["AVG", "SUM", "COUNT", "MAX", "MIN"]):
                            rel_type = "AGGREGATES"
                        elif "JOIN" in purpose:
                            rel_type = "JOINS_ON"
                        else:
                            rel_type = "SELECTS"

                        schema = (col.get("schema") or "public").strip()
                        table = (col.get("table") or "").strip()
                        name = (col.get("name") or "").strip()
                        if not table or not name:
                            continue
                        fqn = f"{schema}.{table}.{name}"
                        columns_used.append(fqn)
                        await self.session.run(
                            f"""
                            MATCH (q:Query {{id: $query_id}})
                            MATCH (c:Column)
                            WHERE c.fqn IS NOT NULL AND toLower(c.fqn) = toLower($fqn)
                            WITH q, c LIMIT 1
                            MERGE (q)-[:{rel_type}]->(c)
                            """,
                            query_id=query_id,
                            fqn=fqn,
                        )

                await self.session.run(
                    """
                    MATCH (q:Query {id: $query_id})
                    SET q.tables_used = $tables_used,
                        q.columns_used = $columns_used
                    """,
                    query_id=query_id,
                    tables_used=tables_used,
                    columns_used=columns_used,
                )

            SmartLogger.log(
                "INFO",
                "neo4j_history.save_query.done",
                category="neo4j.history.save_query",
                params=sanitize_for_log(
                    {
                        "query_id": query_id,
                        "db": db_name,
                        "should_overwrite": should_overwrite,
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    }
                ),
                max_inline_chars=0,
            )

            return query_id
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "neo4j_history.save_query.error",
                category="neo4j.history.save_query",
                params=sanitize_for_log(
                    {
                        "query_id": query_id,
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                        "question": question,
                        "sql": sql,
                        "status": status,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                max_inline_chars=0,
            )
            raise

    @staticmethod
    def _minimize_steps_summary(
        steps: Optional[List[Dict[str, Any]]],
        *,
        json_module,
    ) -> str:
        """
        Legacy fallback when caller didn't provide steps_summary.
        Store only a small, stable subset (no tool_result payloads).
        """
        if not steps:
            return ""
        minimized: List[Dict[str, Any]] = []
        for s in steps[-10:]:
            if not isinstance(s, dict):
                continue
            minimized.append(
                {
                    "iteration": s.get("iteration"),
                    "tool_name": s.get("tool_name"),
                    "reasoning": (s.get("reasoning") or "")[:400],
                }
            )
        try:
            return json_module.dumps(minimized, ensure_ascii=False, default=str)
        except Exception:
            return ""

    async def save_value_mapping_by_fqn(
        self,
        *,
        natural_value: str,
        code_value: str,
        column_fqn: str,
    ) -> None:
        """값 매핑을 Neo4j에 저장 (Column.fqn 기반, 품질 우선)."""
        started = time.perf_counter()
        cypher = """
            MATCH (c:Column)
            WHERE c.fqn IS NOT NULL AND toLower(c.fqn) = toLower($column_fqn)
            WITH c LIMIT 1
            MERGE (v:ValueMapping {natural_value: $natural_value, column_fqn: c.fqn})
            SET v.code_value = $code_value,
                v.usage_count = COALESCE(v.usage_count, 0) + 1,
                v.updated_at = datetime()
            MERGE (v)-[:MAPS_TO]->(c)
        """
        SmartLogger.log(
            "INFO",
            "neo4j_history.save_value_mapping_by_fqn.start",
            category="neo4j.history.save_value_mapping",
            params=sanitize_for_log(
                {
                    "natural_value": natural_value,
                    "code_value": code_value,
                    "column_fqn": column_fqn,
                }
            ),
            max_inline_chars=0,
        )
        try:
            result = await self.session.run(
                cypher,
                natural_value=natural_value,
                code_value=code_value,
                column_fqn=column_fqn,
            )
            summary = await result.consume()
            counters = summary.counters
            SmartLogger.log(
                "INFO",
                "neo4j_history.save_value_mapping_by_fqn.done",
                category="neo4j.history.save_value_mapping",
                params=sanitize_for_log(
                    {
                        "natural_value": natural_value,
                        "code_value": code_value,
                        "column_fqn": column_fqn,
                        "neo4j_contains_updates": bool(counters.contains_updates),
                        "neo4j_counters": {
                            "nodes_created": counters.nodes_created,
                            "nodes_deleted": counters.nodes_deleted,
                            "relationships_created": counters.relationships_created,
                            "relationships_deleted": counters.relationships_deleted,
                            "properties_set": counters.properties_set,
                        },
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    }
                ),
                max_inline_chars=0,
            )
            if not counters.contains_updates:
                SmartLogger.log(
                    "WARNING",
                    "neo4j_history.save_value_mapping_by_fqn.no_update",
                    category="neo4j.history.save_value_mapping",
                    params=sanitize_for_log(
                        {
                            "natural_value": natural_value,
                            "code_value": code_value,
                            "column_fqn": column_fqn,
                            "reason_hint": "MATCH (c:Column {fqn}) returned no rows, so MERGE didn't run.",
                        }
                    ),
                    max_inline_chars=0,
                )
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "neo4j_history.save_value_mapping_by_fqn.error",
                category="neo4j.history.save_value_mapping",
                params=sanitize_for_log(
                    {
                        "natural_value": natural_value,
                        "code_value": code_value,
                        "column_fqn": column_fqn,
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                max_inline_chars=0,
            )
            raise
    
    async def save_value_mapping(
        self,
        natural_value: str,
        code_value: str,
        column_name: str
    ):
        """값 매핑을 Neo4j에 저장"""

        started = time.perf_counter()
        cypher = """
            MATCH (c:Column) WHERE toLower(c.name) = $column_name
            WITH c LIMIT 1
            MERGE (v:ValueMapping {natural_value: $natural_value, column_fqn: c.fqn})
            SET v.code_value = $code_value,
                v.usage_count = COALESCE(v.usage_count, 0) + 1,
                v.updated_at = datetime()
            MERGE (v)-[:MAPS_TO]->(c)
        """

        SmartLogger.log(
            "INFO",
            "neo4j_history.save_value_mapping.start",
            category="neo4j.history.save_value_mapping",
            params=sanitize_for_log(
                {
                    "natural_value": natural_value,
                    "code_value": code_value,
                    "column_name": column_name,
                    "cypher": cypher,
                }
            ),
            max_inline_chars=0,
        )

        try:
            # 컬럼 찾기 및 매핑 저장
            result = await self.session.run(
                cypher,
                natural_value=natural_value,
                code_value=code_value,
                column_name=column_name.lower(),
            )
            summary = await result.consume()
            counters = summary.counters
            SmartLogger.log(
                "INFO",
                "neo4j_history.save_value_mapping.done",
                category="neo4j.history.save_value_mapping",
                params=sanitize_for_log(
                    {
                        "natural_value": natural_value,
                        "code_value": code_value,
                        "column_name": column_name,
                        "neo4j_contains_updates": bool(counters.contains_updates),
                        "neo4j_counters": {
                            "nodes_created": counters.nodes_created,
                            "nodes_deleted": counters.nodes_deleted,
                            "relationships_created": counters.relationships_created,
                            "relationships_deleted": counters.relationships_deleted,
                            "properties_set": counters.properties_set,
                        },
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    }
                ),
                max_inline_chars=0,
            )
            if not counters.contains_updates:
                SmartLogger.log(
                    "WARNING",
                    "neo4j_history.save_value_mapping.no_update",
                    category="neo4j.history.save_value_mapping",
                    params=sanitize_for_log(
                        {
                            "natural_value": natural_value,
                            "code_value": code_value,
                            "column_name": column_name,
                            "reason_hint": "MATCH (c:Column {name}) returned no rows, so MERGE didn't run.",
                        }
                    ),
                    max_inline_chars=0,
                )
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "neo4j_history.save_value_mapping.error",
                category="neo4j.history.save_value_mapping",
                params=sanitize_for_log(
                    {
                        "natural_value": natural_value,
                        "code_value": code_value,
                        "column_name": column_name,
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                max_inline_chars=0,
            )
            raise
    
    async def find_similar_queries_by_graph(
        self,
        tables: List[str] = None,
        columns: List[str] = None,
        question_keywords: List[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """그래프 구조 기반 유사 쿼리 검색"""
        
        # 테이블과 컬럼 기반 검색
        if tables or columns:
            query = """
            WITH $tables AS tables, $columns AS columns
            
            // 동일한 테이블을 사용하는 쿼리
            OPTIONAL MATCH (q:Query)-[:USES_TABLE]->(t:Table)
            WHERE t.name IN tables
            WITH q, COUNT(DISTINCT t) AS table_matches
            
            // 동일한 컬럼을 사용하는 쿼리
            OPTIONAL MATCH (q)-[:SELECTS|FILTERS|AGGREGATES|GROUPS_BY]->(c:Column)
            WHERE c.name IN columns
            WITH q, table_matches, COUNT(DISTINCT c) AS column_matches
            
            WHERE q IS NOT NULL AND q.status = 'completed'
            
            RETURN q.id AS id,
                   q.question AS question,
                   q.sql AS sql,
                   q.row_count AS row_count,
                   q.execution_time_ms AS execution_time_ms,
                   (table_matches * 2 + column_matches) AS similarity_score
            ORDER BY similarity_score DESC, q.created_at DESC
            LIMIT $limit
            """
            
            result = await self.session.run(
                query,
                tables=[t.lower() for t in (tables or [])],
                columns=[c.lower() for c in (columns or [])],
                limit=limit
            )
        
        # 질문 키워드 기반 검색
        elif question_keywords:
            query = """
            MATCH (q:Query)
            WHERE q.status = 'completed'
            WITH q, 
                 REDUCE(score = 0, keyword IN $keywords |
                     CASE WHEN toLower(q.question) CONTAINS toLower(keyword)
                          THEN score + 1 ELSE score END
                 ) AS keyword_score
            WHERE keyword_score > 0
            
            RETURN q.id AS id,
                   q.question AS question,
                   q.sql AS sql,
                   q.row_count AS row_count,
                   q.execution_time_ms AS execution_time_ms,
                   keyword_score AS similarity_score
            ORDER BY keyword_score DESC, q.created_at DESC
            LIMIT $limit
            """
            
            result = await self.session.run(
                query,
                keywords=question_keywords,
                limit=limit
            )
        else:
            # 최근 쿼리 반환
            query = """
            MATCH (q:Query)
            WHERE q.status = 'completed'
            RETURN q.id AS id,
                   q.question AS question,
                   q.sql AS sql,
                   q.row_count AS row_count,
                   q.execution_time_ms AS execution_time_ms,
                   0 AS similarity_score
            ORDER BY q.created_at DESC
            LIMIT $limit
            """
            
            result = await self.session.run(query, limit=limit)
        
        records = await result.data()
        return records
    
    async def find_value_mapping(self, natural_value: str) -> List[Dict]:
        """자연어 값에 대한 코드 매핑 검색"""
        
        query = """
        MATCH (v:ValueMapping)-[:MAPS_TO]->(c:Column)
        WHERE toLower(v.natural_value) CONTAINS toLower($natural_value)
        RETURN v.natural_value AS natural_value,
               v.code_value AS code_value,
               c.fqn AS column_fqn,
               c.name AS column_name,
               v.usage_count AS usage_count
        ORDER BY v.usage_count DESC
        LIMIT 10
        """
        
        result = await self.session.run(query, natural_value=natural_value)
        return await result.data()
    
    async def get_query_history(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None
    ) -> Dict:
        """쿼리 히스토리 조회"""
        
        skip = (page - 1) * page_size
        
        # 쿼리 목록 조회
        query = """
        MATCH (q:Query)
        WHERE $status IS NULL OR q.status = $status
        
        // 관련 테이블 수집
        OPTIONAL MATCH (q)-[:USES_TABLE]->(t:Table)
        WITH q, COLLECT(DISTINCT t.name) AS tables
        
        RETURN q.id AS id,
               q.question AS question,
               q.sql AS sql,
               q.status AS status,
               q.row_count AS row_count,
               q.execution_time_ms AS execution_time_ms,
               q.steps_count AS steps_count,
               q.created_at AS created_at,
               tables
        ORDER BY q.created_at DESC
        SKIP $skip
        LIMIT $limit
        """
        
        result = await self.session.run(
            query,
            status=status,
            skip=skip,
            limit=page_size
        )
        items = await result.data()
        
        # 총 개수 조회
        count_query = """
        MATCH (q:Query)
        WHERE $status IS NULL OR q.status = $status
        RETURN COUNT(q) AS total
        """
        count_result = await self.session.run(count_query, status=status)
        count_record = await count_result.single()
        total = count_record['total'] if count_record else 0
        
        return {
            'items': items,
            'total': total,
            'page': page,
            'page_size': page_size
        }
    
    async def get_table_usage_stats(self) -> List[Dict]:
        """테이블 사용 통계"""
        
        query = """
        MATCH (q:Query)-[:USES_TABLE]->(t:Table)
        WHERE q.status = 'completed'
        RETURN t.schema AS schema,
               t.name AS table_name,
               COUNT(q) AS usage_count,
               COLLECT(DISTINCT q.question)[0..3] AS sample_questions
        ORDER BY usage_count DESC
        LIMIT 20
        """
        
        result = await self.session.run(query)
        return await result.data()
    
    async def get_column_usage_stats(self) -> List[Dict]:
        """컬럼 사용 통계 (용도별)"""
        
        query = """
        MATCH (q:Query)-[r]->(c:Column)
        WHERE q.status = 'completed' AND type(r) IN ['SELECTS', 'FILTERS', 'AGGREGATES', 'GROUPS_BY', 'JOINS_ON']
        RETURN c.fqn AS column_fqn,
               c.name AS column_name,
               type(r) AS usage_type,
               COUNT(q) AS usage_count
        ORDER BY usage_count DESC
        LIMIT 30
        """
        
        result = await self.session.run(query)
        return await result.data()
    
    async def delete_query(self, query_id: str) -> bool:
        """쿼리 삭제"""
        
        query = """
        MATCH (q:Query {id: $query_id})
        DETACH DELETE q
        RETURN COUNT(*) > 0 AS deleted
        """
        
        result = await self.session.run(query, query_id=query_id)
        record = await result.single()
        return record['deleted'] if record else False


# 싱글톤 인스턴스 (세션 주입 필요)
_neo4j_query_repo: Optional[Neo4jQueryRepository] = None


def get_neo4j_query_repo(session) -> Neo4jQueryRepository:
    """Neo4j 쿼리 저장소 인스턴스 반환"""
    global _neo4j_query_repo
    if _neo4j_query_repo is None:
        _neo4j_query_repo = Neo4jQueryRepository(session)
    return _neo4j_query_repo

