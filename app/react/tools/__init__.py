from typing import Any, Dict, List
import time
import traceback

from .context import ToolContext
from . import (
    build_sql_context as build_sql_context_tool,
    validate_sql as validate_sql_tool,
)
from app.smart_logger import SmartLogger
from app.react.utils.log_sanitize import sanitize_for_log


class ToolExecutionError(Exception):
    """툴 실행 중 발생한 예외를 감싼다."""


TOOL_HANDLERS = {
    "build_sql_context": build_sql_context_tool.execute,
    "validate_sql": validate_sql_tool.execute,
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

    started = time.perf_counter()
    SmartLogger.log(
        "INFO",
        "react.tool.call",
        category="react.tool.call",
        params=sanitize_for_log(
            {
                "react_run_id": context.react_run_id,
                "tool_name": tool_name,
                "parameters": parameters,
            }
        ),
        # Store raw parameters for reproducibility in detail logs (when file_output enabled)
        max_inline_chars=0,
    )

    try:
        if tool_name == "build_sql_context":
            question = parameters.get("question") or parameters.get("sql") or parameters.get("text")
            if not question:
                raise ToolExecutionError("question parameter is required")
            exclude_light_sqls = parameters.get("exclude_light_sqls")
            result = await handler(context, question, exclude_light_sqls=exclude_light_sqls)
        elif tool_name == "validate_sql":
            sql_text = parameters.get("sql")
            if not sql_text:
                raise ToolExecutionError("sql parameter is required")
            result = await handler(context, sql_text)
        else:
            raise ToolExecutionError(f"No handler implemented for tool: {tool_name}")

        SmartLogger.log(
            "INFO",
            "react.tool.result",
            category="react.tool.result",
            params=sanitize_for_log(
                {
                    "react_run_id": context.react_run_id,
                    "tool_name": tool_name,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    # Keep raw tool output for reproducibility (saved to detail file when enabled).
                    "tool_result": result,
                }
            ),
            max_inline_chars=0,
        )
        return result
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "react.tool.error",
            category="react.tool.error",
            params=sanitize_for_log(
                {
                    "react_run_id": context.react_run_id,
                    "tool_name": tool_name,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "parameters": parameters,
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            max_inline_chars=0,
        )
        raise

__all__ = [
    "ToolContext",
    "ToolExecutionError",
    "execute_tool",
    "TOOL_HANDLERS",
]

