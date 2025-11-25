from typing import List

from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.react.tools.context import ToolContext
from app.react.generators.explain_analysis_generator import ExplainAnalysisGenerator, ExplainAnalysisResult


async def execute(
    context: ToolContext,
    sql: str,
) -> str:
    """
    SQL Explain 툴.
    """
    result_parts: List[str] = ["<tool_result>"]

    try:
        explain_analysis_generator = ExplainAnalysisGenerator()
        explain_analysis_result: ExplainAnalysisResult = await explain_analysis_generator.generate(
            sql=sql,
            db_conn=context.db_conn
        ) 
        result_parts.append(explain_analysis_result.to_xml_str())
    except (SQLValidationError, SQLExecutionError) as exc:
        result_parts.append(f"<error>{str(exc)}</error>")
    except Exception as exc:
        result_parts.append(f"<error>Unexpected error: {str(exc)}</error>")

    result_parts.append("</tool_result>")
    return "\n".join(result_parts)

