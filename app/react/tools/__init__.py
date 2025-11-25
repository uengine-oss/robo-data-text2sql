from typing import Any, Dict, List

from .context import ToolContext
from . import (
    search_tables as search_tables_tool,
    get_table_schema as get_table_schema_tool,
    search_column_values as search_column_values_tool,
    execute_sql_preview as execute_sql_preview_tool,
    explain as explain_tool,
)


class ToolExecutionError(Exception):
    """툴 실행 중 발생한 예외를 감싼다."""


TOOL_HANDLERS = {
    "search_tables": search_tables_tool.execute,
    "get_table_schema": get_table_schema_tool.execute,
    "search_column_values": search_column_values_tool.execute,
    "execute_sql_preview": execute_sql_preview_tool.execute,
    "explain": explain_tool.execute,
}


async def execute_tool(
    tool_name: str,
    context: ToolContext,
    parameters: Dict[str, Any],
) -> str:
    """
    지정한 툴을 실행하고 XML 문자열 결과를 반환한다.
    parameters 는 툴 별 기대 포맷을 따른다.
    """
    if tool_name not in TOOL_HANDLERS:
        raise ToolExecutionError(f"Unsupported tool: {tool_name}")

    handler = TOOL_HANDLERS[tool_name]

    if tool_name == "search_tables":
        keywords: List[str] = parameters.get("keywords", [])
        return await handler(context, keywords)

    if tool_name == "get_table_schema":
        table_names: List[str] = parameters.get("table_names", [])
        return await handler(context, table_names)

    if tool_name == "search_column_values":
        table_name = parameters.get("table")
        column_name = parameters.get("column")
        schema_name = parameters.get("schema")
        search_keywords: List[str] = parameters.get("search_keywords", [])
        if not table_name or not column_name:
            raise ToolExecutionError("table and column parameters are required")
        return await handler(
            context,
            table_name,
            column_name,
            search_keywords,
            schema_name,
        )

    if tool_name == "execute_sql_preview":
        sql_text = parameters.get("sql")
        if not sql_text:
            raise ToolExecutionError("sql parameter is required")
        return await handler(context, sql_text)
    
    if tool_name == "explain":
        sql_text = parameters.get("sql")
        if not sql_text:
            raise ToolExecutionError("sql parameter is required")
        return await handler(context, sql_text)

    raise ToolExecutionError(f"No handler implemented for tool: {tool_name}")


__all__ = [
    "ToolContext",
    "ToolExecutionError",
    "execute_tool",
    "TOOL_HANDLERS",
]

