import re
from typing import List, Optional, Set, Tuple

from app.config import settings
from app.core.sql_guard import SQLGuard
from app.react.tools.context import ToolContext, execute_fetch


_HINT_IDENTIFIER_PATTERN = re.compile(r'"([^"]+)"')


def _normalize_identifier(identifier: str) -> str:
    return identifier.replace('"', "").lower()


def _split_identifier(identifier: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [part for part in identifier.split(".") if part]
    if not parts:
        return None, None
    if len(parts) == 1:
        return None, parts[0]
    return ".".join(parts[:-1]), parts[-1]


def _table_matches(hint_table: str, target_table: str) -> bool:
    if not target_table:
        return True
    if not hint_table:
        return True

    normalized_hint = _normalize_identifier(hint_table)
    normalized_target = _normalize_identifier(target_table)

    if normalized_hint == normalized_target:
        return True

    return normalized_hint.split(".")[-1] == normalized_target.split(".")[-1]


def _extract_hint_column(exc: Exception, target_table: str) -> Optional[str]:
    """PostgreSQL 힌트 메시지에서 칼럼명을 추출한다."""
    hint_sources: List[str] = []
    hint_attr = getattr(exc, "hint", None)
    if hint_attr:
        hint_sources.append(str(hint_attr))
    exc_str = str(exc)
    if exc_str:
        hint_sources.append(exc_str)

    for source in hint_sources:
        for identifier in _HINT_IDENTIFIER_PATTERN.findall(source):
            table_part, column_part = _split_identifier(identifier)
            if not column_part:
                continue
            if table_part and not _table_matches(table_part, target_table):
                continue
            sanitized_column = SQLGuard.sanitize_identifier(column_part)
            if sanitized_column:
                return sanitized_column
    return None


def _get_quote_char() -> str:
    """DB 타입에 따른 식별자 인용 문자 반환."""
    return '`' if settings.target_db_type.lower() == "mysql" else '"'


def _quote_identifier(identifier: str) -> str:
    """컬럼명 등의 식별자를 적절히 이스케이프한다.
    
    MySQL: 대문자 포함 시에만 백틱으로 감싼다.
    PostgreSQL: 항상 따옴표로 감싼다.
    """
    q = _get_quote_char()
    is_mysql = settings.target_db_type.lower() == "mysql"
    parts = [part for part in identifier.split(".") if part]
    if not parts:
        return f'{q}{q}'
    
    def quote_part(part: str) -> str:
        if is_mysql:
            # MySQL: 대문자 포함 시에만 백틱
            if any(c.isupper() for c in part):
                return f'{q}{part}{q}'
            return part
        else:
            # PostgreSQL: 항상 따옴표
            return f'{q}{part}{q}'
    
    return ".".join(quote_part(part) for part in parts)


def _quote_table_identifier(table: str, schema: Optional[str]) -> str:
    """스키마와 테이블을 결합하여 완전한 식별자를 만든다.
    
    MindsDB의 경우 datasource.schema.table 형식으로 반환.
    대문자 식별자는 백틱으로 감싼다.
    """
    q = _get_quote_char()
    is_mysql = settings.target_db_type.lower() == "mysql"
    
    parts = []
    
    # MindsDB datasource prefix 추가
    if is_mysql and settings.mindsdb_datasource_prefix:
        parts.append(settings.mindsdb_datasource_prefix)
    
    # 스키마 추가
    schema_parts = [part for part in (schema or "").split(".") if part]
    parts.extend(schema_parts)
    
    # 테이블 추가
    table_parts = [part for part in table.split(".") if part]
    parts.extend(table_parts)
    
    if not parts:
        return f'{q}{q}'
    
    # 대문자 식별자만 백틱으로 감싸기 (MySQL), 전체 감싸기 (PostgreSQL)
    def quote_part(part: str) -> str:
        if is_mysql:
            # MySQL: 대문자 포함 시에만 백틱
            if any(c.isupper() for c in part):
                return f'{q}{part}{q}'
            return part
        else:
            # PostgreSQL: 항상 따옴표
            return f'{q}{part}{q}'
    
    return ".".join(quote_part(part) for part in parts)


def _row_sort_key(row) -> Tuple:
    """행 정렬을 위한 키 생성. 캐싱 일관성을 위해 결과 순서를 보장한다."""
    return tuple(
        (str(v) if v is not None else "") for v in row.values()
    )


async def _fetch_rows_with_progressive_keyword(
    context: ToolContext, query_sql: str, keyword: str
) -> Tuple[str, List]:
    """키워드를 뒤에서 제거하며 시도하고 실패 시 앞에서 제거하며 재시도한다."""
    base_keyword = keyword or ""
    tried_keywords: Set[str] = set()

    suffix_keyword = base_keyword
    while True:
        if suffix_keyword not in tried_keywords:
            rows = await execute_fetch(context.db_conn, query_sql, f"%{suffix_keyword}%")
            tried_keywords.add(suffix_keyword)
            if rows:
                return suffix_keyword, rows
        if not suffix_keyword:
            break
        suffix_keyword = suffix_keyword[:-1]

    prefix_keyword = base_keyword
    while prefix_keyword:
        prefix_keyword = prefix_keyword[1:]
        if prefix_keyword in tried_keywords:
            continue
        rows = await execute_fetch(context.db_conn, query_sql, f"%{prefix_keyword}%")
        tried_keywords.add(prefix_keyword)
        if rows:
            return prefix_keyword, rows
        if not prefix_keyword:
            break

    return "", []


async def execute(
    context: ToolContext,
    table_name: str,
    column_name: str,
    search_keywords: List[str],
    schema: Optional[str] = None,
) -> str:
    """특정 스키마 내 컬럼에서 키워드와 매칭되는 값을 조회한다."""
    search_column_values_search_keywords_limit = max(int(context.scaled(context.search_column_values_search_keywords_limit)), 1)
    search_keywords = search_keywords[:search_column_values_search_keywords_limit]

    value_limit = context.scaled(context.value_limit)
    sanitized_schema = SQLGuard.sanitize_identifier(schema) if schema else None
    sanitized_table = SQLGuard.sanitize_identifier(table_name)
    sanitized_column = SQLGuard.sanitize_identifier(column_name)

    result_parts: List[str] = [
        "<tool_result>",
        f"<target_schema>{sanitized_schema or ""}</target_schema>",
        f"<target_table>{sanitized_table}</target_table>",
        f"<target_column>{sanitized_column}</target_column>",
    ]

    seen_rows: Set[Tuple] = set()

    qualified_table = _quote_table_identifier(sanitized_table, sanitized_schema)

    def build_value_query(column: str) -> str:
        qualified_column = _quote_identifier(column)
        is_mysql = settings.target_db_type.lower() == "mysql"
        if is_mysql:
            # MySQL: %s 플레이스홀더, LIKE (대소문자 구분 없음 by default)
            return (
                f"SELECT * FROM {qualified_table} "
                f"WHERE {qualified_column} LIKE %s LIMIT {value_limit}"
            )
        else:
            # PostgreSQL: $1 플레이스홀더, ILIKE, ::text 캐스팅
            return (
                f"SELECT * FROM {qualified_table} "
                f"WHERE {qualified_column}::text ILIKE $1 LIMIT {value_limit}"
            )

    def open_rows_tag(
        *,
        row_type: str,
        used_keyword: str,
        effective_keyword: Optional[str] = None,
        resolved_column: Optional[str] = None,
        fallback_from: Optional[str] = None,
    ) -> str:
        attrs = [f'type="{row_type}"', f'used_keyword="{used_keyword}"']
        if effective_keyword is not None:
            attrs.append(f'effective_keyword="{effective_keyword}"')
        if resolved_column:
            attrs.append(f'resolved_column="{resolved_column}"')
        if fallback_from and fallback_from != resolved_column:
            attrs.append(f'fallback_from="{fallback_from}"')
        return f"<rows {' '.join(attrs)}>"

    async def run_keyword_query(
        column: str, normalized_keyword: str, fallback_from: Optional[str] = None
    ) -> None:
        query_sql = build_value_query(column)
        effective_keyword, query_rows = await _fetch_rows_with_progressive_keyword(
            context, query_sql, normalized_keyword
        )
        query_rows = sorted(query_rows, key=_row_sort_key)
        result_parts.append(
            open_rows_tag(
                row_type="query",
                used_keyword=normalized_keyword,
                effective_keyword=effective_keyword,
                resolved_column=column,
                fallback_from=fallback_from,
            )
        )
        for row in query_rows:
            row_tuple = tuple(row.values())
            if row_tuple in seen_rows:
                continue
            seen_rows.add(row_tuple)
            result_parts.append("<row>")
            for col_name, col_value in row.items():
                if col_value is None:
                    continue
                value_str = str(col_value).strip()
                if not value_str:
                    continue
                result_parts.append(f"<{col_name}>{value_str}</{col_name}>")
            result_parts.append("</row>")
        result_parts.append("</rows>")

    try:
        default_sql = f"SELECT * FROM {qualified_table} LIMIT {value_limit}"
        default_rows = await execute_fetch(context.db_conn, default_sql)
        default_rows = sorted(default_rows, key=_row_sort_key)

        result_parts.append('<rows type="default">')
        for row in default_rows:
            row_tuple = tuple(row.values())
            if row_tuple in seen_rows:
                continue
            seen_rows.add(row_tuple)
            result_parts.append("<row>")
            for col_name, col_value in row.items():
                if col_value is None:
                    continue
                value_str = str(col_value).strip()
                if not value_str:
                    continue
                result_parts.append(f"<{col_name}>{value_str}</{col_name}>")
            result_parts.append("</row>")
        result_parts.append("</rows>")
    except Exception as exc:
        result_parts.append(f'<rows type="default"><error>{str(exc)}</error></rows>')

    active_column = sanitized_column

    for keyword in search_keywords:
        normalized_keyword = "" if keyword is None else str(keyword)
        try:
            await run_keyword_query(active_column, normalized_keyword)
            continue
        except Exception as exc:
            hint_column = _extract_hint_column(exc, sanitized_table)
            fallback_column = None
            fallback_error_message: Optional[str] = None

            if hint_column and hint_column != active_column:
                fallback_column = hint_column
                try:
                    await run_keyword_query(
                        fallback_column,
                        normalized_keyword,
                        fallback_from=active_column if active_column else None,
                    )
                except Exception as fallback_exc:
                    fallback_error_message = str(fallback_exc)
                else:
                    active_column = fallback_column
                    continue

            error_lines = [str(exc)]
            if fallback_column:
                error_lines.append(f"[hint_column]={fallback_column}")
            if fallback_error_message:
                error_lines.append(f"[fallback_error]={fallback_error_message}")

            result_parts.append(
                open_rows_tag(
                    row_type="query",
                    used_keyword=normalized_keyword,
                    resolved_column=active_column or sanitized_column or None,
                )
            )
            result_parts.append(f"<error>{'\n'.join(error_lines)}</error>")
            result_parts.append("</rows>")

    result_parts.append("</tool_result>")
    return "\n".join(result_parts)

