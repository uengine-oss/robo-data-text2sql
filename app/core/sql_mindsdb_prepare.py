"""
MindsDB(MySQL endpoint) SQL preparation / deterministic auto-correct.

Why this exists
---------------
In Phase 1 this service talks to MindsDB via the MySQL protocol. To query external datasources,
we use the MindsDB "passthrough" form:

  SELECT * FROM `datasource` ( <inner_sql_in_external_db_dialect> );

In that form, the *inner SQL* is sent directly to the external DB (e.g. PostgreSQL)
without any MindsDB parsing. This supports ALL schemas and all SQL features natively.

This module provides deterministic preparation:
- Detect if SQL is already in passthrough form and leave it unchanged (or sanitize inner if corrupted).
- Otherwise, wrap the SQL in passthrough form via `transform_sql_for_mindsdb()`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


from app.core.sql_transform import transform_sql_for_mindsdb


@dataclass(frozen=True)
class MindsDBPrepareResult:
    sql: str
    changed: bool
    reason: str
    details: Dict[str, Any]


def is_passthrough_query(sql: str, datasource: str) -> bool:
    s = (sql or "").strip()
    ds = (datasource or "").strip()
    if not s or not ds:
        return False
    # FROM postgresql ( ... )
    # FROM `postgresql` ( ... )
    pat = rf"(?is)\bfrom\s+`?{re.escape(ds)}`?\s*\("
    return re.search(pat, s) is not None


def _find_matching_paren_span(text: str, open_idx: int) -> Optional[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) for the parenthesized span including parentheses.
    Best-effort:
    - tracks depth
    - ignores parentheses inside single-quoted strings
    """
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "(":
        return None
    depth = 0
    i = open_idx
    in_squote = False
    while i < len(text):
        ch = text[i]
        if in_squote:
            if ch == "'":
                # handle escaped '' inside string
                if i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                in_squote = False
            i += 1
            continue
        if ch == "'":
            in_squote = True
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return (open_idx, i)
        i += 1
    return None


def _extract_passthrough_inner(sql: str, datasource: str) -> Optional[Dict[str, Any]]:
    s = str(sql or "")
    ds = (datasource or "").strip()
    if not ds:
        return None
    m = re.search(rf"(?is)\bfrom\s+`?{re.escape(ds)}`?\s*\(", s)
    if not m:
        return None
    open_paren_idx = m.end() - 1  # points to '('
    span = _find_matching_paren_span(s, open_paren_idx)
    if not span:
        return None
    a, b = span
    inner = s[a + 1 : b].strip()
    outer_before = s[: a + 1]  # include '('
    outer_after = s[b:]  # include ')...'
    return {
        "outer_before": outer_before,
        "inner": inner,
        "outer_after": outer_after,
        "inner_span": (a, b),
    }


def _passthrough_sanitize_inner_for_postgres(inner_sql: str, datasource: str) -> str:
    """
    Best-effort sanitize for Postgres-executed inner SQL:
    - Remove accidental datasource prefix: datasource.<schema>.<table> -> <schema>.<table>
    - Convert MySQL-style backticks around identifiers to PostgreSQL double quotes.
      (Does NOT touch single-quoted string literals.)
    """
    inner = str(inner_sql or "")
    ds = (datasource or "").strip()
    if not inner.strip() or not ds:
        return inner.strip()

    # Protect single-quoted string literals
    placeholders: Dict[str, str] = {}
    counter = 0

    def _protect_str(m: re.Match) -> str:
        nonlocal counter
        key = f"__xstrx_{counter}__"
        counter += 1
        placeholders[key] = m.group(0)
        return key

    tmp = re.sub(r"'(?:[^']|'')*'", _protect_str, inner)

    # Remove datasource prefix at identifier boundaries: postgresql.`RWIS`... / postgresql."RWIS"... / postgresql.RWIS...
    tmp = re.sub(rf"(?i)\b{re.escape(ds)}\s*\.", "", tmp)

    # Backticks -> double quotes (identifiers)
    tmp = re.sub(r"`([^`]*)`", r'"\1"', tmp)

    # Restore strings
    for k, v in placeholders.items():
        tmp = tmp.replace(k, v)

    return tmp.strip()


def fix_passthrough_query_if_needed(sql: str, datasource: str) -> Optional[MindsDBPrepareResult]:
    """
    If sql is passthrough and inner looks corrupted, return a fixed SQL.
    Otherwise return None.
    """
    s = str(sql or "")
    ds = (datasource or "").strip()
    info = _extract_passthrough_inner(s, ds)
    if not info:
        return None
    inner = str(info.get("inner") or "")
    fixed_inner = _passthrough_sanitize_inner_for_postgres(inner, ds)
    if fixed_inner.strip() == inner.strip():
        return None
    rebuilt = f"{info['outer_before']}\n{fixed_inner}\n{info['outer_after']}"
    return MindsDBPrepareResult(
        sql=rebuilt,
        changed=True,
        reason="passthrough_sanitize_inner",
        details={"datasource": ds},
    )


def prepare_sql_for_mindsdb(sql: str, datasource: str) -> MindsDBPrepareResult:
    """
    Deterministically prepare SQL for execution via MindsDB MySQL endpoint.
    - If already passthrough: leave unchanged (optionally sanitize inner).
    - Otherwise: wrap in passthrough form via transform_sql_for_mindsdb().
    """
    s = (sql or "").strip()
    ds = (datasource or "").strip()
    if not s:
        return MindsDBPrepareResult(sql=s, changed=False, reason="empty_sql", details={})
    if not ds:
        return MindsDBPrepareResult(sql=s, changed=False, reason="missing_datasource", details={})

    if is_passthrough_query(s, ds):
        fixed = fix_passthrough_query_if_needed(s, ds)
        if fixed is not None:
            return fixed
        return MindsDBPrepareResult(
            sql=s,
            changed=False,
            reason="passthrough_no_change",
            details={"datasource": ds},
        )

    transformed = transform_sql_for_mindsdb(s, ds)
    changed = transformed.strip() != s.strip()
    return MindsDBPrepareResult(
        sql=transformed,
        changed=bool(changed),
        reason="transform_sql_for_mindsdb" if changed else "no_change",
        details={"datasource": ds},
    )


__all__ = [
    "MindsDBPrepareResult",
    "is_passthrough_query",
    "fix_passthrough_query_if_needed",
    "prepare_sql_for_mindsdb",
]
