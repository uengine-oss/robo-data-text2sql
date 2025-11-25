from typing import List

from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.react.tools.context import ToolContext


async def execute(
    context: ToolContext,
    sql: str,
) -> str:
    """
    SQL 예비 실행 툴.
    안전 검증 후 최대 preview_row_limit 개 행만 반환한다.
    max_sql_seconds 내에 실행되어야 하며, 초과 시 타임아웃 에러를 반환한다.
    """
    guard = SQLGuard()
    executor = SQLExecutor()
    preview_limit = context.scaled(context.preview_row_limit)

    result_parts: List[str] = ["<tool_result>"]

    try:
        validated_sql, _ = guard.validate(sql)
        results = await executor.execute_query(
            context.db_conn, 
            validated_sql,
            timeout=float(context.max_sql_seconds),
        )

        columns = results.get("columns", [])
        rows = results.get("rows", [])
        trimmed_rows = rows[:preview_limit]

        result_parts.append("<preview>")
        result_parts.append(f"<row_count>{len(rows)}</row_count>")
        result_parts.append(f"<columns>{','.join(columns)}</columns>")
        result_parts.append("<rows>")

        for idx, row in enumerate(trimmed_rows, start=1):
            result_parts.append(f'<row index="{idx}">')
            for col_name, value in zip(columns, row):
                if value is None:
                    result_parts.append(f'<cell column="{col_name}" />')
                else:
                    result_parts.append(f'<cell column="{col_name}">{str(value)}</cell>')
            result_parts.append("</row>")

        result_parts.append("</rows>")
        result_parts.append("</preview>")
    except (SQLValidationError, SQLExecutionError) as exc:
        result_parts.append(f"<error>{str(exc)}</error>")
    except Exception as exc:
        result_parts.append(f"<error>Unexpected error: {str(exc)}</error>")

    result_parts.append("</tool_result>")
    return "\n".join(result_parts)

