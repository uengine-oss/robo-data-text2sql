from __future__ import annotations

import re
from xml.etree import ElementTree as ET


_DROP_TAGS = (
    "elapsed_ms",
    "preview_execution_time_ms",
    "target_k",
    "mode",
    "preview_row_limit",
    "timeout_s",
    "score",
    "similarity_score",
    "best_context_score",
    "usage_count",
    "cardinality",
)

_DROP_TAG_RE = re.compile(
    r"\s*<(?P<tag>"
    + "|".join(re.escape(tag) for tag in _DROP_TAGS)
    + r")>(?:<!\[CDATA\[[\s\S]*?\]\]>|[\s\S]*?)</(?P=tag)>",
    flags=re.IGNORECASE,
)

_BETWEEN_TAG_WS_RE = re.compile(r">\s+<")


def compact_build_sql_context_for_prompt(tool_result_xml: str) -> str:
    """
    Prompt-only compaction that keeps XML shape and evidence text.

    Internal parsers use the original XML. This function removes only low-value
    operational metadata and inter-tag formatting whitespace from the LLM prompt copy.
    """
    raw = (tool_result_xml or "").strip()
    if not raw:
        return ""
    if "<build_sql_context_result" not in raw:
        return raw

    compact = _DROP_TAG_RE.sub("", raw)
    compact = _BETWEEN_TAG_WS_RE.sub("><", compact)

    # Keep the prompt copy XML-compatible; if our conservative regex ever breaks
    # structure, fall back to the original tool result.
    try:
        ET.fromstring(compact)
    except ET.ParseError:
        return raw
    return compact.strip()


__all__ = ["compact_build_sql_context_for_prompt"]
