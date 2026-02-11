"""
Light LLM one-shot SQL autocorrect for MindsDB execution.

Policy:
- Use ONLY as fallback after deterministic preparation fails.
- Ask the model to return strict JSON and parse robustly.
- Do NOT change semantics; only rewrite for compatibility (quoting/prefix/passthrough form).
"""

from __future__ import annotations

import json
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.react.llm_factory import create_react_llm


@dataclass(frozen=True)
class LLMAutocorrectResult:
    ok: bool
    sql: str
    explanation: str
    provider: str = ""
    model: str = ""
    elapsed_ms: float = 0.0
    raw: str = ""
    error: str = ""


def _strip_json_fences(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$", "", s, flags=re.IGNORECASE).strip()
    return s.strip()


async def propose_sql_fix_light_llm(
    *,
    datasource: str,
    original_sql: str,
    error_text: str,
    max_output_tokens: int = 420,
) -> LLMAutocorrectResult:
    """
    One-shot fix proposal. Never raises; returns error field on failure.
    """
    ds = (datasource or "").strip()
    orig = (original_sql or "").strip()
    err = (error_text or "").strip()

    if not ds or not orig:
        return LLMAutocorrectResult(
            ok=False,
            sql="",
            explanation="",
            error="missing_datasource_or_sql",
        )

    try:
        # NOTE: some flash-lite Gemini models reject thinking_level.
        handle = create_react_llm(
            purpose="sql.autocorrect.light",
            thinking_level=None,
            include_thoughts=False,
            temperature=0.0,
            max_output_tokens=int(max_output_tokens),
            use_light=True,
        )
        llm = handle.llm

        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore

        sys_msg = SystemMessage(
            content=(
                "You are a SQL compatibility fixer for MindsDB's MySQL protocol endpoint.\n"
                "Return ONLY a JSON object with keys: ok(boolean), sql(string), explanation(string).\n"
                "- ok=true means: no change needed.\n"
                "- ok=false means: provide corrected SQL in 'sql'.\n"
                "Constraints:\n"
                "- Do NOT change semantics (only quoting/prefix/passthrough form).\n"
                "- If the query is a passthrough like: FROM <datasource> ( <inner_sql> ),\n"
                "  preserve inner SQL in PostgreSQL dialect (double quotes for identifiers; NO backticks; NO datasource. prefix inside).\n"
                "- If the query is NOT passthrough, ensure table refs are compatible with MindsDB MySQL endpoint:\n"
                "  prefer <datasource>.<schema>.<table> with backticks for identifiers when needed.\n"
            )
        )
        user_msg = HumanMessage(
            content=textwrap.dedent(
                f"""
                Datasource: {ds}

                Original SQL:
                {orig}

                Error:
                {err}

                Respond with JSON only.
                """
            ).strip()
        )

        started = time.perf_counter()
        resp = await llm.ainvoke([sys_msg, user_msg])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        content = getattr(resp, "content", None)
        raw = content if isinstance(content, str) else str(content or "")

        jtxt = _strip_json_fences(raw)
        if not jtxt:
            return LLMAutocorrectResult(
                ok=False,
                sql="",
                explanation="",
                provider=str(handle.provider or ""),
                model=str(handle.model or ""),
                elapsed_ms=float(elapsed_ms),
                raw=raw,
                error="empty_llm_output",
            )

        try:
            data = json.loads(jtxt)
        except Exception as je:
            return LLMAutocorrectResult(
                ok=False,
                sql="",
                explanation="",
                provider=str(handle.provider or ""),
                model=str(handle.model or ""),
                elapsed_ms=float(elapsed_ms),
                raw=raw,
                error=f"json_parse_failed:{type(je).__name__}",
            )

        if not isinstance(data, dict):
            return LLMAutocorrectResult(
                ok=False,
                sql="",
                explanation="",
                provider=str(handle.provider or ""),
                model=str(handle.model or ""),
                elapsed_ms=float(elapsed_ms),
                raw=raw,
                error="llm_output_not_object",
            )

        ok = bool(data.get("ok"))
        sql = str(data.get("sql") or "").strip()
        explanation = str(data.get("explanation") or "").strip()
        return LLMAutocorrectResult(
            ok=ok,
            sql=sql,
            explanation=explanation,
            provider=str(handle.provider or ""),
            model=str(handle.model or ""),
            elapsed_ms=float(elapsed_ms),
            raw=raw,
            error="",
        )
    except Exception as exc:
        return LLMAutocorrectResult(
            ok=False,
            sql="",
            explanation="",
            error=f"llm_failed:{type(exc).__name__}:{exc}",
        )


__all__ = ["LLMAutocorrectResult", "propose_sql_fix_light_llm"]

