from __future__ import annotations

from typing import List, Sequence

from app.smart_logger import SmartLogger

from .text import _guess_unbounded_scope
from .xml_utils import emit_text


def append_suggestions_xml(
    *,
    question: str,
    resolved_values: Sequence[dict],
    fallback_terms: Sequence[str],
    result_parts: List[str],
) -> List[str]:
    q = (question or "").strip()

    suggestions: List[str] = []
    if _guess_unbounded_scope(q):
        suggestions.append(
            "조회 범위(기간)가 넓을 수 있습니다. 기간(예: 최근 1주/1개월/3개월 또는 특정 기간)을 지정해 주세요."
        )
    if not list(resolved_values) and list(fallback_terms):
        suggestions.append(
            "대상(예: 특정 지점/지역/제품/조직 등)이 모호할 수 있습니다. 어떤 대상을 기준으로 조회할지 알려주세요."
        )

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.ask_user_suggestions",
        category="react.tool.detail.build_sql_context",
        params={
            "suggestions": suggestions,
            "unbounded_scope_detected": _guess_unbounded_scope(q),
            "has_resolved_values": len(list(resolved_values)) > 0,
        },
        max_inline_chars=0,
    )

    result_parts.append("<ask_user_suggestions>")
    for suggestion in suggestions:
        result_parts.append(emit_text("suggestion", suggestion))
    result_parts.append("</ask_user_suggestions>")

    return suggestions


__all__ = ["append_suggestions_xml"]


