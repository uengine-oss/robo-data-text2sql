from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from neo4j import AsyncSession


@dataclass(slots=True)
class ToolContext:
    """
    공용 툴 실행 컨텍스트.

    비동기 Neo4j 세션, 대상 DB 커넥션을 한 번에 전달한다.
    """

    neo4j_session: AsyncSession
    # Target DB connection (asyncpg for Postgres, aiomysql for MindsDB/MySQL, etc.)
    db_conn: Any

    # MindsDB-only (Phase 1): mandatory datasource for SQL transformation (D9).
    datasource: Optional[str] = None

    # ReAct 요청 단위 상관관계(correlation)용. (라우터/에이전트가 설정)
    react_run_id: Optional[str] = None

    # Optional: streaming hook for progress events (NDJSON).
    # - Intended for long-running tool internals (e.g., build_sql_context).
    # - Must be best-effort and never fail the main flow.
    stream_emit: Optional[Callable[[Dict[str, Any]], None]] = None

    table_rerank_fetch_k: int = 60
    table_rerank_top_k: int = 20
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

    def scaled(self, value: int) -> int:
        """TOOL_POWER_LEVEL 을 적용한 정수 값을 반환한다."""
        scaled = int(value * self.tool_power_level)
        return scaled if scaled > 0 else 1

    def emit(self, event: Dict[str, Any]) -> None:
        """
        Best-effort progress event emission.
        The caller is responsible for keeping `event` JSON-serializable.
        """
        fn = self.stream_emit
        if not fn:
            return
        try:
            fn(event)
        except Exception:
            # Never break main execution due to progress streaming failures.
            return

    def with_overrides(
        self,
        *,
        table_relation_limit: Optional[int] = None,
        column_relation_limit: Optional[int] = None,
        value_limit: Optional[int] = None,
        preview_row_limit: Optional[int] = None,
        search_table_keyword_limit: Optional[int] = None,
        get_table_schema_table_name_limit: Optional[int] = None,
        search_column_values_search_keywords_limit: Optional[int] = None,
        max_sql_seconds: Optional[int] = None,
        schema_filter: Optional[List[str]] = None,
        table_rerank_fetch_k: Optional[int] = None,
        table_rerank_top_k: Optional[int] = None,
        stream_emit: Optional[Callable[[Dict[str, Any]], None]] = None,
        datasource: Optional[str] = None,
    ) -> "ToolContext":
        """필요시 일부 파라미터를 오버라이드한 새로운 컨텍스트를 생성한다."""
        return ToolContext(
            neo4j_session=self.neo4j_session,
            db_conn=self.db_conn,
            datasource=datasource if datasource is not None else self.datasource,
            react_run_id=self.react_run_id,
            stream_emit=stream_emit if stream_emit is not None else self.stream_emit,
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
            table_rerank_fetch_k=table_rerank_fetch_k or self.table_rerank_fetch_k,
            table_rerank_top_k=table_rerank_top_k or self.table_rerank_top_k,
        )

