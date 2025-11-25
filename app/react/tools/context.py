from dataclasses import dataclass
from typing import Optional

import asyncpg
from neo4j import AsyncSession
from openai import AsyncOpenAI


@dataclass(slots=True)
class ToolContext:
    """
    공용 툴 실행 컨텍스트.

    비동기 Neo4j 세션, 대상 DB 커넥션, OpenAI 클라이언트를 한 번에 전달한다.
    """

    neo4j_session: AsyncSession
    db_conn: asyncpg.Connection
    openai_client: AsyncOpenAI

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
    ) -> "ToolContext":
        """필요시 일부 파라미터를 오버라이드한 새로운 컨텍스트를 생성한다."""
        return ToolContext(
            neo4j_session=self.neo4j_session,
            db_conn=self.db_conn,
            openai_client=self.openai_client,
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
        )

