"""
MindsDB SQL transformation utilities (Phase 1: MindsDB-only).

Goals (merge_plan decisions):
- Do NOT inject/modify LIMIT in user SQL (D6).
- Ensure MindsDB addressing works: `datasource.schema.table` (datasource is request parameter, D9).
- Improve compatibility for uppercase/Korean identifiers by adding MySQL backticks.
- Convert PostgreSQL-style double-quoted identifiers to MySQL-style backtick-quoted identifiers.

NOTE:
- This module is intentionally *pure string transform* (no DB access).
- Execution row limiting is enforced by SQLExecutor via fetchmany/streaming caps (D11).
- MindsDB uses MySQL protocol and ONLY supports backtick quoting for identifiers.
  Double-quoted identifiers are NOT supported by MindsDB (they get stripped or mishandled).
- Data type names (DECIMAL, NUMERIC, etc.) must NEVER be backtick-quoted.
"""

from __future__ import annotations

import re
from typing import Set


# SQL keywords AND data type names (exclude from identifier quoting).
# CRITICAL: Data type names MUST be included here because backtick-quoting them
# causes MindsDB to treat them as identifiers instead of types, leading to errors
# like: [postgres/postgresql]: '`DECIMAL`'
SQL_KEYWORDS: Set[str] = {
    # --- Standard SQL keywords ---
    "SELECT",
    "FROM",
    "WHERE",
    "AND",
    "OR",
    "NOT",
    "IN",
    "IS",
    "NULL",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "FULL",
    "CROSS",
    "ON",
    "ORDER",
    "BY",
    "ASC",
    "DESC",
    "GROUP",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "AS",
    "DISTINCT",
    "ALL",
    "UNION",
    "INTERSECT",
    "EXCEPT",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "CREATE",
    "ALTER",
    "DROP",
    "TABLE",
    "INDEX",
    "VIEW",
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "REFERENCES",
    "CONSTRAINT",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "COALESCE",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "LIKE",
    "BETWEEN",
    "EXISTS",
    "TRUE",
    "FALSE",
    "CAST",
    "CONVERT",
    "DATE",
    "TIME",
    "TIMESTAMP",
    "INTERVAL",
    "WITH",
    "RECURSIVE",
    "CTE",
    "OVER",
    "PARTITION",
    "ROWS",
    "RANGE",
    "PRECEDING",
    "FOLLOWING",
    "UNBOUNDED",
    "CURRENT",
    "FETCH",
    "FIRST",
    "NEXT",
    "ONLY",
    "ROW",
    "FILTER",
    "WITHIN",
    "LATERAL",
    "NATURAL",
    "USING",
    "ILIKE",
    "SIMILAR",
    "TO",
    "ANY",
    "SOME",
    "ARRAY",
    "NULLIF",
    "GREATEST",
    "LEAST",
    "IF",
    "IFNULL",
    # --- SQL data types (MUST NOT be backtick-quoted) ---
    "NUMERIC",
    "DECIMAL",
    "DEC",
    "INTEGER",
    "INT",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "MEDIUMINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
    "PRECISION",
    "VARCHAR",
    "CHAR",
    "CHARACTER",
    "NCHAR",
    "NVARCHAR",
    "TEXT",
    "CLOB",
    "VARYING",
    "NATIONAL",
    "BOOLEAN",
    "BOOL",
    "SERIAL",
    "BIGSERIAL",
    "SMALLSERIAL",
    "MONEY",
    "UUID",
    "JSON",
    "JSONB",
    "XML",
    "BYTEA",
    "BLOB",
    "BINARY",
    "VARBINARY",
    "BIT",
    "UNSIGNED",
    "SIGNED",
    "ZONE",
    # --- Common SQL function names ---
    "TRIM",
    "LOWER",
    "UPPER",
    "SUBSTRING",
    "REPLACE",
    "CONCAT",
    "LENGTH",
    "ROUND",
    "FLOOR",
    "CEILING",
    "CEIL",
    "ABS",
    "MOD",
    "POWER",
    "SQRT",
    "LOG",
    "EXP",
    "EXTRACT",
    "EPOCH",
    "YEAR",
    "MONTH",
    "DAY",
    "HOUR",
    "MINUTE",
    "SECOND",
    "LAG",
    "LEAD",
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
    "NTILE",
    "FIRST_VALUE",
    "LAST_VALUE",
    "STRING_AGG",
    "GROUP_CONCAT",
}


def _is_korean(ch: str) -> bool:
    return "\uac00" <= ch <= "\ud7a3"


def _has_korean(s: str) -> bool:
    return any(_is_korean(c) for c in str(s or ""))


def _has_upper(s: str) -> bool:
    return any(c.isupper() for c in str(s or ""))


def add_datasource_prefix(sql: str, datasource: str) -> str:
    """
    Convert `schema.table` references to `datasource.schema.table` in FROM/JOIN lists.

    - If the SQL already references `datasource.schema.table`, it is left unchanged.
    - Only rewrites table references that appear after FROM/JOIN or comma-separated FROM items.
    - Does NOT modify user-provided LIMIT clauses (D6).
    """
    s = (sql or "").strip()
    ds = str(datasource or "").strip()
    if not s or not ds:
        return s

    # 1) Quoted schema/table: FROM "schema"."table" / FROM `schema`.`table`
    def _replace_table_ref_quoted(m: re.Match) -> str:
        prefix = m.group(1)  # FROM/JOIN/, with whitespace
        quote = m.group(2)
        schema = m.group(3)
        table = m.group(4)
        # If schema is already datasource (i.e., already ds.schema.table), skip.
        if schema.lower() == ds.lower():
            return m.group(0)
        return f"{prefix}{ds}.{quote}{schema}{quote}.{quote}{table}{quote}"

    # Avoid corrupting already 3-part refs: "ds"."schema"."table"
    pattern_quoted = (
        r'(\b(?:FROM|JOIN|,)\s+)(["`])([A-Za-z_][A-Za-z0-9_]*)\2\.\2([A-Za-z_][A-Za-z0-9_]*)\2'
        r'(?!\.\2[A-Za-z_][A-Za-z0-9_]*\2)'
    )
    s = re.sub(pattern_quoted, _replace_table_ref_quoted, s, flags=re.IGNORECASE)

    # 2) Unquoted schema.table
    def _replace_table_ref_unquoted(m: re.Match) -> str:
        prefix = m.group(1)
        schema = m.group(2)
        table = m.group(3)
        if schema.lower() == ds.lower():
            return m.group(0)
        return f"{prefix}{ds}.{schema}.{table}"

    # Avoid corrupting already 3-part refs: ds.schema.table
    # NOTE: include a word-boundary after table to prevent partial backtracking matches
    # e.g. "other.public.users" must NOT match as "other.publi" + "c..."
    pattern_unquoted = r"(\b(?:FROM|JOIN|,)\s+)([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b(?!\.)"
    s = re.sub(pattern_unquoted, _replace_table_ref_unquoted, s, flags=re.IGNORECASE)

    return s


def _dquote_to_backtick(sql: str) -> str:
    """
    Convert PostgreSQL-style double-quoted identifiers to MySQL-style backtick-quoted identifiers.

    "identifier" -> `identifier`

    In standard SQL (and PostgreSQL), double quotes are ALWAYS identifiers (single quotes are strings).
    MindsDB uses MySQL protocol which requires backtick quoting for identifiers.
    MindsDB does NOT support double-quoted identifiers - it strips or mishandles them.

    NOTE: This must run AFTER add_datasource_prefix() so that datasource prefix
    is added correctly (add_datasource_prefix handles both " and ` quoted refs).
    """
    s = str(sql or "")
    if not s.strip():
        return s
    # Replace all "identifier" with `identifier`.
    # Safe because in PostgreSQL SQL, double quotes are always identifiers, never strings.
    return re.sub(r'"([^"]*)"', r'`\1`', s)


def quote_uppercase_identifiers(sql: str) -> str:
    """
    Add MySQL backticks around UNQUOTED identifiers that contain:
    - Korean characters, or
    - Uppercase characters

    This improves MindsDB(MySQL) compatibility when upstream metadata uses uppercase names.

    IMPORTANT:
    - Single-quoted string literals are PROTECTED from modification.
    - Identifiers already in backticks or double quotes are SKIPPED.
    - SQL keywords AND data type names (in SQL_KEYWORDS) are NEVER quoted.
    - This function should run AFTER _dquote_to_backtick() so that previously
      double-quoted identifiers are already converted to backtick-quoted.
    """
    s = str(sql or "")
    if not s.strip():
        return s

    # Step 0: Protect single-quoted string literals from modification.
    # Korean characters inside string literals like '청주정수장' must NOT be backtick-quoted.
    _str_placeholders = {}
    _str_counter = [0]

    def _protect_string_literal(m: re.Match) -> str:
        key = f"__xstrx_{_str_counter[0]}__"
        _str_counter[0] += 1
        _str_placeholders[key] = m.group(0)
        return key

    # Match single-quoted strings (including escaped quotes '' inside)
    s = re.sub(r"'(?:[^']|'')*'", _protect_string_literal, s)

    # 1) Quote aliases after AS (but NOT data types in CAST(... AS type))
    def _replace_alias(m: re.Match) -> str:
        as_kw = m.group(1)
        alias = m.group(2)
        if alias.startswith("`"):
            return m.group(0)
        # CRITICAL: Do NOT quote SQL keywords / data type names that appear after AS
        # e.g., CAST(x AS DECIMAL(16,4)) - DECIMAL is a type, not an alias.
        if alias.upper() in SQL_KEYWORDS:
            return m.group(0)
        if _has_korean(alias) or _has_upper(alias):
            return f"{as_kw}`{alias}`"
        return m.group(0)

    alias_pattern = r"(?i)\b(AS\s+)([^\s,`\'\"()]+)"
    s = re.sub(alias_pattern, _replace_alias, s)

    # 2) Quote other identifiers (columns, schema/table parts, etc.)
    def _replace_identifier(m: re.Match) -> str:
        ident = m.group(0)
        if ident.upper() in SQL_KEYWORDS:
            return ident
        if ident.startswith("`"):
            return ident
        if _has_korean(ident) or _has_upper(ident):
            return f"`{ident}`"
        return ident

    # Only match unquoted identifiers starting with [A-Z or Korean]
    identifier_pattern = r"(?<!`)\b([A-Z가-힣][A-Z0-9_가-힣]*)\b(?!`)"
    s = re.sub(identifier_pattern, _replace_identifier, s)

    # Step 3: Restore single-quoted string literals.
    for key, value in _str_placeholders.items():
        s = s.replace(key, value)

    return s


def transform_sql_for_mindsdb(sql: str, datasource: str) -> str:
    """
    Single point of MindsDB SQL transformation (D4):
    1) Ensure datasource prefix: schema.table -> datasource.schema.table
    2) Convert double-quoted identifiers to backtick-quoted (MindsDB MySQL compatibility)
    3) Quote remaining unquoted uppercase/Korean identifiers with backticks

    MindsDB compatibility rules (verified by testing):
    - MindsDB ONLY supports backtick quoting for identifiers.
    - Double-quoted identifiers are NOT supported (MindsDB strips quotes, sends unquoted to PostgreSQL).
    - Data type names (DECIMAL, NUMERIC, etc.) must NEVER be backtick-quoted.
    """
    s = (sql or "").strip()
    ds = str(datasource or "").strip()
    if not ds:
        raise ValueError("datasource is required for MindsDB SQL transformation")
    s = add_datasource_prefix(s, ds)
    s = _dquote_to_backtick(s)
    s = quote_uppercase_identifiers(s)
    return s
