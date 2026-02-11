from __future__ import annotations

import asyncio
import re
from typing import List

from app.core.sql_guard import SQLGuard
from app.config import settings
from app.core.sql_exec import SQLExecutor
from app.core.sql_mindsdb_prepare import prepare_sql_for_mindsdb

from .models import ColumnCandidate


async def _limited_db_probe(
    *,
    context,
    keyword: str,
    column: ColumnCandidate,
    timeout_s: float,
    value_limit: int,
) -> List[str]:
    kw = (keyword or "").strip()
    if not kw:
        return []

    schema = SQLGuard.sanitize_identifier(column.table_schema) if column.table_schema else ""
    table = SQLGuard.sanitize_identifier(column.table_name)
    col = SQLGuard.sanitize_identifier(column.name)
    if not table or not col:
        return []

    # IMPORTANT:
    # - do NOT append LIMIT to the user's final SQL (D6).
    # - this probe is a separate small query, so a LIMIT here is acceptable.
    db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()

    def _safe_ident(part: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]", "", str(part or ""))

    schema_id = _safe_ident(schema)
    table_id = _safe_ident(table)
    col_id = _safe_ident(col)
    if not table_id or not col_id:
        return []

    # Escape keyword literal (single-quote)
    kw_esc = kw.replace("'", "''")

    if db_type in {"mysql", "mariadb"}:
        # MindsDB: requires datasource.schema.table. datasource is provided by request (D9).
        datasource = str(getattr(context, "datasource", "") or "").strip()
        if not datasource:
            return []

        # Build as schema.table then transform into datasource.schema.table with backticks.
        table_ref = ".".join([p for p in [schema_id, table_id] if p])
        col_ref = col_id
        probe_sql = (
            f"SELECT DISTINCT {col_ref} AS value "
            f"FROM {table_ref} "
            f"WHERE {col_ref} IS NOT NULL AND CAST({col_ref} AS CHAR) LIKE '%{kw_esc}%' "
            f"LIMIT {int(value_limit)}"
        )
        try:
            probe_sql = prepare_sql_for_mindsdb(probe_sql, datasource).sql
        except Exception:
            return []
    else:
        # PostgreSQL default path
        table_ident = ".".join([f'"{p}"' for p in [schema_id, table_id] if p])
        col_ident = f'"{col_id}"'
        probe_sql = (
            f"SELECT DISTINCT {col_ident} AS value "
            f"FROM {table_ident} "
            f"WHERE {col_ident} IS NOT NULL AND {col_ident}::text ILIKE '%{kw_esc}%' "
            f"LIMIT {int(value_limit)}"
        )

    executor = SQLExecutor()
    try:
        preview = await asyncio.wait_for(
            executor.preview_query(
                context.db_conn,
                probe_sql,
                row_limit=int(value_limit),
                timeout=float(timeout_s),
            ),
            timeout=timeout_s,
        )
    except Exception:
        return []

    out: List[str] = []
    for row in list(preview.get("rows", []) or [])[: int(value_limit)]:
        if not row:
            continue
        v = row[0] if isinstance(row, list) else None
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out[: int(value_limit)]


__all__ = ["_limited_db_probe"]


