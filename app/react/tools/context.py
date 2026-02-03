from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import asyncpg
from neo4j import AsyncSession
from openai import AsyncOpenAI

from app.config import settings

# MySQL support
try:
    import aiomysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


async def execute_fetch(
    conn: Union[asyncpg.Connection, "aiomysql.Connection"],
    query: str,
    *args
) -> List[Dict[str, Any]]:
    """
    PostgreSQL(asyncpg)과 MySQL(aiomysql) 모두 지원하는 fetch 헬퍼.
    
    Returns:
        List of dictionaries (column_name -> value)
    """
    if settings.target_db_type.lower() == "mysql":
        # MySQL (aiomysql) - cursor 기반
        async with conn.cursor(aiomysql.DictCursor) as cursor:
            await cursor.execute(query, args if args else None)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows] if rows else []
    else:
        # PostgreSQL (asyncpg) - fetch 메서드 사용
        if args:
            rows = await conn.fetch(query, *args)
        else:
            rows = await conn.fetch(query)
        return [dict(row) for row in rows]


@dataclass(slots=True)
class ToolContext:
    """
    공용 툴 실행 컨텍스트.

    비동기 Neo4j 세션, 대상 DB 커넥션, OpenAI 클라이언트를 한 번에 전달한다.
    """

    neo4j_session: AsyncSession
    db_conn: asyncpg.Connection
    openai_client: AsyncOpenAI

    # ReAct 요청 단위 상관관계(correlation)용. (라우터/에이전트가 설정)
    react_run_id: Optional[str] = None

    table_top_k: int = 20
    table_relation_limit: int = 20
    column_relation_limit: int = 10
    value_limit: int = 10
    preview_row_limit: int = 30
    tool_power_level: float = 1.0

    search_table_keyword_limit: int = 10
    get_table_schema_table_name_limit: int = 10
    search_column_values_search_keywords_limit: int = 10

    max_sql_seconds: int = 60
    
    # Schema filter for limiting search to specific schemas (e.g., ["dw"] for OLAP only)
    schema_filter: Optional[List[str]] = None
    
    # ObjectType only mode (for domain layer - only search Materialized View tables)
    object_type_only: bool = False
    
    # 연결된 ObjectType 정보 (프롬프트 확장용)
    # [{name: str, columns: [{name: str, type: str}], description?: str}]
    linked_object_types: Optional[List[Dict[str, Any]]] = None
    
    # 계리수식/공식 우선 검색 모드
    # True일 경우 수식 컬럼(formula, calculation, expression 등)과 계산식을 우선 탐색
    prefer_formula: bool = False

    def scaled(self, value: int) -> int:
        """TOOL_POWER_LEVEL 을 적용한 정수 값을 반환한다."""
        scaled = int(value * self.tool_power_level)
        return scaled if scaled > 0 else 1

    def with_overrides(
        self,
        *,
        table_top_k: Optional[int] = None,
        table_relation_limit: Optional[int] = None,
        column_relation_limit: Optional[int] = None,
        value_limit: Optional[int] = None,
        preview_row_limit: Optional[int] = None,
        search_table_keyword_limit: Optional[int] = None,
        get_table_schema_table_name_limit: Optional[int] = None,
        search_column_values_search_keywords_limit: Optional[int] = None,
        max_sql_seconds: Optional[int] = None,
        schema_filter: Optional[List[str]] = None,
        object_type_only: Optional[bool] = None,
        linked_object_types: Optional[List[Dict[str, Any]]] = None,
        prefer_formula: Optional[bool] = None,
    ) -> "ToolContext":
        """필요시 일부 파라미터를 오버라이드한 새로운 컨텍스트를 생성한다."""
        return ToolContext(
            neo4j_session=self.neo4j_session,
            db_conn=self.db_conn,
            openai_client=self.openai_client,
            react_run_id=self.react_run_id,
            table_top_k=table_top_k or self.table_top_k,
            table_relation_limit=table_relation_limit or self.table_relation_limit,
            column_relation_limit=column_relation_limit or self.column_relation_limit,
            value_limit=value_limit or self.value_limit,
            preview_row_limit=preview_row_limit or self.preview_row_limit,
            tool_power_level=self.tool_power_level,
            search_table_keyword_limit=search_table_keyword_limit or self.search_table_keyword_limit,
            get_table_schema_table_name_limit=get_table_schema_table_name_limit or self.get_table_schema_table_name_limit,
            search_column_values_search_keywords_limit=search_column_values_search_keywords_limit or self.search_column_values_search_keywords_limit,
            max_sql_seconds=max_sql_seconds or self.max_sql_seconds,
            schema_filter=schema_filter if schema_filter is not None else self.schema_filter,
            object_type_only=object_type_only if object_type_only is not None else self.object_type_only,
            linked_object_types=linked_object_types if linked_object_types is not None else self.linked_object_types,
            prefer_formula=prefer_formula if prefer_formula is not None else self.prefer_formula,
        )

