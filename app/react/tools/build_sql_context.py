"""
build_sql_context - 통합 SQL 컨텍스트 구축 도구 (리팩토링 버전)

목표:
- find_similar_query + search_tables + get_table_schema + search_column_values 기능을 1회 호출로 통합한다.
- 사용자 질문에서 SQL 생성을 위한 문맥(유사쿼리 패턴, ValueMapping, 스키마 후보, enum cache, 제한적 값 probe)을 제공한다.

리팩토링 원칙:
- 이 파일에는 **핵심적인 로직 흐름/공개 API**만 남긴다.
- 실제 구현은 `app.react.tools.build_sql_context_parts` 아래 모듈로 분리한다.
- p_temp_scripts 등에서 사용하던 내부 심볼들은 호환성을 위해 여기서 re-export 한다.
"""

from __future__ import annotations

from typing import Optional, Sequence

from app.react.tools.build_sql_context_parts import (  # re-export
    ColumnCandidate,
    TableCandidate,
    _limited_db_probe,
    _neo4j_fetch_fk_relationships,
    _neo4j_fetch_table_schemas,
    _neo4j_search_tables_text2sql_vector,
    _regex_terms,
    execute as _execute,
)


async def execute(context, question: str, exclude_light_sqls: Optional[Sequence[str]] = None) -> str:
    """
    build_sql_context tool entrypoint.

    Extended behavior:
    - exclude_light_sqls: when provided, prevents re-generating/re-executing duplicate light_queries
      across context-refresh loops.
    """
    return await _execute(context, question, exclude_light_sqls=exclude_light_sqls)


__all__ = [
    "execute",
    # Backward-compatible re-exports for p_temp_scripts
    "TableCandidate",
    "ColumnCandidate",
    "_neo4j_search_tables_text2sql_vector",
    "_neo4j_fetch_table_schemas",
    "_neo4j_fetch_fk_relationships",
    "_regex_terms",
    "_limited_db_probe",
]


