"""
MindsDB(MySQL endpoint) SQL preparation / deterministic auto-correct.

Why this exists
---------------
In Phase 1 this service talks to MindsDB via the MySQL protocol. To query external datasources,
we use the MindsDB "passthrough" form:

  SELECT * FROM `datasource` ( <inner_sql_in_external_db_dialect> );

In that form, the *inner SQL* is sent directly to the external DB (e.g. PostgreSQL)
without any MindsDB parsing. This supports ALL schemas and all SQL features natively.

IMPORTANT — Why we ALWAYS use passthrough form:
MindsDB's "datasource.schema.table" addressing mode strips MySQL backticks when proxying
to PostgreSQL, which causes PostgreSQL to lowercase UPPERCASE identifiers (e.g.
`RWIS`.`RDF01HH_TB` → rwis.rdf01hh_tb → "does not exist").
Passthrough form bypasses MindsDB's parser entirely and sends the inner SQL directly
to PostgreSQL with proper double-quote quoting, preserving case.

This module provides deterministic preparation:
- Detect if SQL is already in passthrough form and leave it unchanged (or sanitize inner if corrupted).
- Otherwise, convert MySQL-style SQL to PostgreSQL-dialect and wrap in passthrough form.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


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


def infer_passthrough_inner_dialect(datasource: str) -> str:
    """
    Best-effort datasource dialect inference for MindsDB passthrough inner SQL.

    This keeps the first implementation conservative: only well-known datasource
    names opt in. Unknown datasources keep the legacy behavior.
    """
    ds = (datasource or "").strip().lower()
    if not ds:
        return ""
    if "postgres" in ds or ds in {"pg"}:
        return "postgresql"
    if "mysql" in ds or "mariadb" in ds:
        return "mysql"
    return ""


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


def _mysql_to_postgres_sql(sql: str, datasource: str) -> str:
    """
    Convert MySQL-style SQL to PostgreSQL-compatible SQL:
    - Remove datasource prefix: datasource.<schema>.<table> -> <schema>.<table>
    - Backtick-quoted function calls (`FUNC`(...)) -> unquoted (PostgreSQL is case-insensitive for functions)
    - Remaining backtick-quoted identifiers -> PostgreSQL double-quoted identifiers
    - Single-quoted string literals are PROTECTED from modification.
    """
    inner = str(sql or "")
    ds = (datasource or "").strip()
    if not inner.strip():
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

    # Remove datasource prefix at identifier boundaries:
    # postgresql.`RWIS`... / postgresql."RWIS"... / postgresql.RWIS...
    if ds:
        tmp = re.sub(rf"(?i)\b{re.escape(ds)}\s*\.", "", tmp)

    # Step 1: Backtick-quoted function names (followed by '(') -> remove backticks only.
    # e.g. `SUBSTR`(...) -> SUBSTR(...)
    # PostgreSQL is case-insensitive for function names when unquoted.
    tmp = re.sub(r"`([^`]*)`(?=\s*\()", lambda m: m.group(1), tmp)

    # Step 2: Remaining backtick-quoted identifiers -> PostgreSQL double quotes.
    # e.g. `RWIS`.`RDF01HH_TB` -> "RWIS"."RDF01HH_TB"
    # This preserves case for UPPERCASE table/schema/column names in PostgreSQL.
    tmp = re.sub(r"`([^`]*)`", r'"\1"', tmp)

    # Restore string literals
    for k, v in placeholders.items():
        tmp = tmp.replace(k, v)

    return tmp.strip()


def _passthrough_sanitize_inner_for_postgres(inner_sql: str, datasource: str) -> str:
    """
    Best-effort sanitize for Postgres-executed inner SQL.
    Delegates to the shared _mysql_to_postgres_sql converter.
    """
    return _mysql_to_postgres_sql(inner_sql, datasource)


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


def _wrap_as_passthrough(inner_sql: str, datasource: str) -> str:
    """
    Wrap SQL in MindsDB passthrough form:
      SELECT * FROM `datasource` ( <inner_sql> )

    The inner SQL is sent directly to the external DB (PostgreSQL) without
    MindsDB parsing, preserving all SQL features and identifier casing.
    """
    inner = (inner_sql or "").strip().rstrip(";").strip()
    ds = str(datasource or "").strip()
    return f"SELECT * FROM `{ds}` (\n{inner}\n)"


def wrap_inner_sql_as_passthrough(inner_sql: str, datasource: str) -> str:
    return _wrap_as_passthrough(inner_sql, datasource)


def extract_passthrough_inner_sql(sql: str, datasource: str) -> Optional[str]:
    info = _extract_passthrough_inner(sql, datasource)
    if not info:
        return None
    inner = str(info.get("inner") or "").strip()
    return inner or None


def prepare_sql_for_mindsdb(sql: str, datasource: str) -> MindsDBPrepareResult:
    """
    Deterministically prepare SQL for execution via MindsDB MySQL endpoint.

    ALWAYS uses passthrough form to avoid MindsDB stripping backticks and
    causing PostgreSQL to lowercase UPPERCASE identifiers.

    - If already passthrough: leave unchanged (optionally sanitize inner).
    - Otherwise: convert MySQL-style backticks to PostgreSQL double-quotes
      and wrap in passthrough form.
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

    # Convert MySQL-style SQL to PostgreSQL-compatible and wrap in passthrough form.
    # This ensures:
    # 1. Backtick-quoted identifiers become double-quoted (case preserved in PostgreSQL)
    # 2. Backtick-quoted function names are unquoted (PostgreSQL is case-insensitive)
    # 3. Datasource prefix is removed (passthrough sends directly to the datasource)
    pg_sql = _mysql_to_postgres_sql(s, ds)
    wrapped = _wrap_as_passthrough(pg_sql, ds)
    return MindsDBPrepareResult(
        sql=wrapped,
        changed=True,
        reason="wrapped_as_passthrough",
        details={"datasource": ds},
    )


__all__ = [
    "MindsDBPrepareResult",
    "extract_passthrough_inner_sql",
    "infer_passthrough_inner_dialect",
    "is_passthrough_query",
    "fix_passthrough_query_if_needed",
    "prepare_sql_for_mindsdb",
    "wrap_inner_sql_as_passthrough",
]
