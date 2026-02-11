"""
Domain-agnostic, lightweight SQL auto-repair utilities.

Purpose:
- Provide best-effort rewrites for common DB planning/runtime errors that are *not* user ambiguity,
  especially in LLM-generated SQL.
- Keep this module dependency-free (stdlib only) so it can be imported from quick diagnostics
  and p_temp_scripts without requiring DB drivers/LLM deps.
"""

from __future__ import annotations

import re
from typing import List, Tuple


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _norm_db_type(db_type: str | None) -> str:
    return str(db_type or "").strip().lower()


def _is_postgres_db_type(db_type: str | None) -> bool:
    dt = _norm_db_type(db_type)
    return dt in {"postgresql", "postgres", "pg"}


def _split_by_single_quotes(sql: str) -> List[Tuple[bool, str]]:
    """
    Split SQL into segments by single-quoted literals.
    Returns list of (in_single_quote, segment_text).
    Preserves quotes inside returned segments (quotes belong to in-quote segments).
    """
    s = sql or ""
    out: List[Tuple[bool, str]] = []
    buf: List[str] = []
    in_sq = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "'":
            # SQL single-quote escape: '' inside a string literal
            if in_sq and i + 1 < len(s) and s[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            # flush buffer before toggling state
            if buf:
                out.append((in_sq, "".join(buf)))
                buf = []
            in_sq = not in_sq
            buf.append("'")
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        out.append((in_sq, "".join(buf)))
    return out


_RE_HINT_IDENT_QUOTED = re.compile(
    r"(?:\bHINT\b\s*:|힌트\s*:|아마)\s*[\s\S]{0,250}?\"(?P<ident>[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\"",
    flags=re.IGNORECASE,
)
_RE_HINT_IDENT_PLAIN = re.compile(
    r"(?:\bHINT\b\s*:|힌트\s*:|아마)\s*[\s\S]{0,250}?(?P<ident>[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)",
    flags=re.IGNORECASE,
)
_RE_MISSING_IDENT_EN = re.compile(
    r"\bcolumn\s+(?P<ident>\"?[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\"?)\s+does\s+not\s+exist\b",
    flags=re.IGNORECASE,
)
_RE_MISSING_IDENT_KO = re.compile(
    r"(?P<ident>[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)\s*(?:칼럼|컬럼)\s*(?:없|존재하지)",
    flags=re.IGNORECASE,
)


def _extract_undefined_column_hint_mapping(error_text: str) -> Tuple[str, str]:
    """
    Returns (missing_ident, suggested_ident) where each is like 't2.SUJ_NAME'.
    Empty strings mean not found.
    """
    raw = str(error_text or "")
    missing = ""
    suggested = ""

    m = _RE_MISSING_IDENT_EN.search(raw)
    if m:
        missing = (m.group("ident") or "").replace('"', "").strip()
    if not missing:
        m2 = _RE_MISSING_IDENT_KO.search(raw)
        if m2:
            missing = (m2.group("ident") or "").replace('"', "").strip()

    h = _RE_HINT_IDENT_QUOTED.search(raw)
    if h:
        suggested = (h.group("ident") or "").replace('"', "").strip()
    if not suggested:
        h2 = _RE_HINT_IDENT_PLAIN.search(raw)
        if h2:
            suggested = (h2.group("ident") or "").replace('"', "").strip()

    return missing, suggested


def _rewrite_undefined_column_using_hint(sql: str, *, missing_ident: str, suggested_ident: str) -> Tuple[str, List[str]]:
    """
    Best-effort rewrite for Postgres undefined_column with HINT like:
    - missing: t2.SUJ_NAME
    - hint:    t3.SUJ_NAME

    We only apply when both are alias.column and the column name matches (case-insensitive).
    We avoid rewriting inside single-quoted literals.
    """
    reasons: List[str] = []
    s = (sql or "").strip()
    if not s:
        return s, reasons
    miss = (missing_ident or "").strip()
    sug = (suggested_ident or "").strip()
    if "." not in miss or "." not in sug:
        return s, reasons

    miss_alias, miss_col = [p.strip() for p in miss.split(".", 1)]
    sug_alias, sug_col = [p.strip() for p in sug.split(".", 1)]
    if not (miss_alias and miss_col and sug_alias and sug_col):
        return s, reasons
    if miss_alias.lower() == sug_alias.lower():
        return s, reasons
    if miss_col.lower() != sug_col.lower():
        return s, reasons

    # Replace only alias-qualified column references: miss_alias.<col> -> sug_alias.<col>
    pat = re.compile(
        rf"\b{re.escape(miss_alias)}\s*\.\s*(?:\"(?P<qcol>[^\"]+)\"|(?P<col>[A-Za-z_][A-Za-z0-9_]*))",
        flags=re.IGNORECASE,
    )

    def _rewrite_segment(seg: str) -> str:
        def _repl(m: re.Match) -> str:
            col_txt = m.group("qcol") or m.group("col") or ""
            if col_txt.strip().lower() != miss_col.lower():
                return m.group(0)
            if m.group("qcol") is not None:
                return f'{sug_alias}."{col_txt}"'
            return f"{sug_alias}.{col_txt}"

        return pat.sub(_repl, seg)

    parts = _split_by_single_quotes(s)
    changed = False
    out: List[str] = []
    for in_sq, seg in parts:
        if in_sq:
            out.append(seg)
            continue
        seg2 = _rewrite_segment(seg)
        if seg2 != seg:
            changed = True
        out.append(seg2)

    if not changed:
        return s, reasons

    reasons.append(f"rewrite: undefined_column hint mapping {miss_alias}.{miss_col} -> {sug_alias}.{sug_col}")
    return "".join(out), reasons


def _find_matching_paren(sql: str, open_paren_idx: int) -> int:
    """
    Find matching ')' for '(' at open_paren_idx, best-effort.
    Handles single-quoted strings with doubled-quote escape ('').
    Returns -1 on failure.
    """
    if open_paren_idx < 0 or open_paren_idx >= len(sql) or sql[open_paren_idx] != "(":
        return -1
    depth = 1
    in_sq = False
    i = open_paren_idx + 1
    while i < len(sql):
        ch = sql[i]
        if ch == "'":
            # SQL single-quote escape: '' inside a string literal
            if in_sq and i + 1 < len(sql) and sql[i + 1] == "'":
                i += 2
                continue
            in_sq = not in_sq
            i += 1
            continue
        if not in_sq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _split_top_level_args(text: str) -> List[str]:
    """
    Split a function arg string by top-level commas (ignoring nested parentheses and single quotes).
    """
    args: List[str] = []
    buf: List[str] = []
    depth = 0
    in_sq = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            if in_sq and i + 1 < len(text) and text[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_sq = not in_sq
            buf.append(ch)
            i += 1
            continue
        if not in_sq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                args.append("".join(buf).strip())
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    if buf:
        args.append("".join(buf).strip())
    return args


_TEXT_CAST_HINT_RE = re.compile(
    r"::\s*(text|varchar|character\s+varying|char|character)\b|cast\s*\([\s\S]*?\s+as\s+(text|varchar|character\s+varying|char|character)\b",
    flags=re.IGNORECASE,
)


def _looks_like_texty(expr_sql: str) -> bool:
    return bool(_TEXT_CAST_HINT_RE.search(expr_sql or ""))


def _rewrite_trim_like_calls_cast_text(sql: str) -> str:
    """
    Rewrite TRIM(x) / BTRIM(x) -> TRIM(CAST(x AS TEXT)) when x is not already texty.
    Best-effort; skips complex TRIM syntaxes (TRIM(... FROM ...)).
    """
    s = sql or ""
    out: List[str] = []
    i = 0
    changed = False
    lower = s.lower()
    targets = ("trim", "btrim")

    while i < len(s):
        hit = None
        for fn in targets:
            if lower.startswith(fn, i):
                # word boundary before/after
                before_ok = (i == 0) or (not (lower[i - 1].isalnum() or lower[i - 1] == "_"))
                after_idx = i + len(fn)
                after_ok = (after_idx >= len(lower)) or (not (lower[after_idx].isalnum() or lower[after_idx] == "_"))
                if before_ok and after_ok:
                    hit = fn
                    break
        if not hit:
            out.append(s[i])
            i += 1
            continue

        j = i + len(hit)
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "(":
            out.append(s[i])
            i += 1
            continue

        close = _find_matching_paren(s, j)
        if close < 0:
            out.append(s[i])
            i += 1
            continue

        inner = s[j + 1 : close].strip()
        # Skip TRIM(... FROM ...) form
        if " from " in inner.lower():
            out.append(s[i : close + 1])
            i = close + 1
            continue
        # Skip multi-arg form
        if len(_split_top_level_args(inner)) != 1:
            out.append(s[i : close + 1])
            i = close + 1
            continue
        if _looks_like_texty(inner):
            out.append(s[i : close + 1])
            i = close + 1
            continue

        changed = True
        out.append(f"{s[i:i+len(hit)]}(")
        out.append(f"CAST({inner} AS TEXT)")
        out.append(")")
        i = close + 1

    return "".join(out) if changed else (sql or "")


def _rewrite_nullif_empty_second_arg_cast_text(sql: str) -> str:
    """
    Rewrite NULLIF(x, '') -> NULLIF(CAST(x AS TEXT), '') when x is not already texty.
    This targets a common Postgres planning-time error: NULLIF(numeric, '').
    """
    s = sql or ""
    out: List[str] = []
    i = 0
    changed = False
    lower = s.lower()

    while i < len(s):
        if not lower.startswith("nullif", i):
            out.append(s[i])
            i += 1
            continue
        # word boundary checks
        before_ok = (i == 0) or (not (lower[i - 1].isalnum() or lower[i - 1] == "_"))
        after_idx = i + len("nullif")
        after_ok = (after_idx >= len(lower)) or (not (lower[after_idx].isalnum() or lower[after_idx] == "_"))
        if not (before_ok and after_ok):
            out.append(s[i])
            i += 1
            continue

        j = i + len("nullif")
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "(":
            out.append(s[i])
            i += 1
            continue
        close = _find_matching_paren(s, j)
        if close < 0:
            out.append(s[i])
            i += 1
            continue
        inner = s[j + 1 : close]
        args = _split_top_level_args(inner)
        if len(args) != 2:
            out.append(s[i : close + 1])
            i = close + 1
            continue
        a0 = (args[0] or "").strip()
        a1 = (args[1] or "").strip()
        a1_norm = _normalize_text(a1)
        # Accept '' or ''::text etc.
        is_empty_literal = a1_norm.startswith("''")
        if not is_empty_literal:
            out.append(s[i : close + 1])
            i = close + 1
            continue
        if _looks_like_texty(a0):
            out.append(s[i : close + 1])
            i = close + 1
            continue

        changed = True
        out.append("NULLIF(")
        out.append(f"CAST({a0} AS TEXT), {a1.strip()}")
        out.append(")")
        i = close + 1

    return "".join(out) if changed else (sql or "")


def auto_repair_sql_for_postgres_error(sql: str, error_text: str, *, db_type: str | None = None) -> Tuple[str, List[str]]:
    """
    Best-effort SQL rewrite for common Postgres planning/runtime type errors.
    Returns (sql_out, reasons). If no rewrite applied, returns (sql_in, []).
    """
    reasons: List[str] = []
    s = (sql or "").strip()
    if not s:
        return s, reasons

    err = _normalize_text(error_text)

    # Pattern 0: Postgres undefined_column + HINT (alias.column mapping)
    if _is_postgres_db_type(db_type) and ("42703" in err or "undefined_column" in err or "does not exist" in err or "칼럼" in err or "컬럼" in err):
        missing_ident, suggested_ident = _extract_undefined_column_hint_mapping(error_text or "")
        if missing_ident and suggested_ident:
            s2, r2 = _rewrite_undefined_column_using_hint(s, missing_ident=missing_ident, suggested_ident=suggested_ident)
            if s2 != s:
                reasons.extend(r2)
                s = s2

    # Pattern A: TRIM/BTRIM on numeric
    if ("btrim(numeric)" in err) or ("trim(numeric)" in err) or ("pg_catalog.btrim(numeric)" in err):
        s2 = _rewrite_trim_like_calls_cast_text(s)
        if s2 != s:
            reasons.append("rewrite: cast TRIM/BTRIM arg to TEXT (avoid btrim(numeric))")
            s = s2

    # Pattern B: NULLIF(numeric, '') planning-time cast failure
    raw = error_text or ""
    if ("invalid input syntax for type numeric" in err and "\"\"" in raw) or (
        "numeric 자료형" in err and "잘못된 입력" in err
    ):
        s2 = _rewrite_nullif_empty_second_arg_cast_text(s)
        if s2 != s:
            reasons.append("rewrite: cast NULLIF first arg to TEXT when comparing to ''")
            s = s2

    return s, reasons

