from __future__ import annotations

import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape as xml_escape

from app.config import settings
from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.core.sql_mindsdb_prepare import prepare_sql_for_mindsdb
from app.react.generators.light_disambiguation_queries_generator import (
    LightDisambiguationQuery,
    get_light_disambiguation_queries_generator,
)
from app.smart_logger import SmartLogger

from .models import ColumnCandidate, TableCandidate
from .xml_utils import emit_text, to_cdata


def _tfqn_l(schema: str, table: str) -> str:
    s = (schema or "").strip().lower()
    t = (table or "").strip().lower()
    return f"{s}.{t}" if s else t


def _norm_sql_key(sql: str) -> str:
    # Normalize for cross-run dedup: trim + compact whitespace + lower.
    return " ".join(str(sql or "").split()).strip().lower()


def _build_schema_candidates_payload(
    *,
    table_candidates: Sequence[TableCandidate],
    selected_tables: Sequence[TableCandidate],
    per_table_columns: Dict[str, List[ColumnCandidate]],
    per_table_k: int,
    rerank_top_k: int,
    table_description_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Build a compact JSON payload describing candidate tables and selected columns.
    This is used as input to the light LLM generator.
    """
    per_table_k_i = max(1, int(per_table_k))
    rerank_top_k_i = max(1, int(rerank_top_k))

    def _qt(s: str) -> str:
        return f'"{str(s or "").strip()}"'

    tables_out: List[Dict[str, Any]] = []
    for t in list(table_candidates or [])[:rerank_top_k_i]:
        schema = str(t.schema or "")
        name = str(t.name or "")
        table_quoted = ".".join([_qt(p) for p in [schema, name] if str(p or "").strip()])
        tfqn_l = _tfqn_l(schema, name)
        override_desc = ""
        if table_description_overrides and tfqn_l:
            override_desc = str(table_description_overrides.get(tfqn_l, "") or "").strip()
        desc = override_desc or (str(t.analyzed_description or "").strip() or str(t.description or "").strip())
        tables_out.append(
            {
                "schema": schema,
                "name": name,
                "table_quoted": table_quoted,
                "from_sql": f"{table_quoted} t" if table_quoted else "",
                # Prefer text_to_sql_embedding_text (override) since Table.description/analyzed_description can be missing/weak.
                "description": desc,
                "score": float(t.score or 0.0),
            }
        )

    per_table_out: List[Dict[str, Any]] = []
    for t in list(selected_tables or [])[:rerank_top_k_i]:
        schema = str(t.schema or "")
        name = str(t.name or "")
        tfqn = _tfqn_l(schema, name)
        cols = list(per_table_columns.get(tfqn, []) or [])[:per_table_k_i]
        per_table_out.append(
            {
                "schema": schema,
                "name": name,
                "table_quoted": ".".join([_qt(p) for p in [schema, name] if str(p or "").strip()]),
                "alias": "t",
                "columns": [
                    {
                        "name": str(c.name or ""),
                        "dtype": str(c.dtype or ""),
                        "description": str(c.description or ""),
                        "score": float(c.score or 0.0),
                        "col_quoted": _qt(str(c.name or "")) if str(c.name or "").strip() else "",
                        "col_ref": f't.{_qt(str(c.name or ""))}' if str(c.name or "").strip() else "",
                    }
                    for c in cols
                    if (c.name or "").strip()
                ],
            }
        )

    return {
        "tables": tables_out,
        "per_table_columns": per_table_out,
        "per_table_k": int(per_table_k_i),
        "rerank_top_k": int(rerank_top_k_i),
    }


def _preview_to_xml(*, preview: Dict[str, Any]) -> List[str]:
    parts: List[str] = []
    cols = list(preview.get("columns") or [])
    rows = list(preview.get("rows") or [])
    parts.append("<preview>")
    parts.append(emit_text("row_count", str(preview.get("row_count", 0))))
    parts.append("<columns>")
    for c in cols:
        parts.append(emit_text("column", str(c)))
    parts.append("</columns>")
    parts.append("<rows>")
    for idx, row in enumerate(rows, start=1):
        parts.append(f'<row index="{idx}">')
        for col_name, value in zip(cols, row):
            col_attr = xml_escape(str(col_name), {'"': "&quot;"})
            if value is None:
                parts.append(f'<cell column="{col_attr}" />')
            else:
                parts.append(f'<cell column="{col_attr}">{to_cdata(str(value))}</cell>')
        parts.append("</row>")
    parts.append("</rows>")
    parts.append(emit_text("preview_execution_time_ms", str(preview.get("execution_time_ms", 0))))
    parts.append("</preview>")
    return parts


def _fallback_queries(
    *,
    selected_tables: Sequence[TableCandidate],
    per_table_columns: Dict[str, List[ColumnCandidate]],
    target_k: int,
) -> List[LightDisambiguationQuery]:
    """
    Deterministic fallback to keep downstream behavior stable if LLM returns empty/invalid output.
    Tries to propose simple sample queries that are cheap under cursor preview.
    """
    k = max(1, int(target_k))
    out: List[LightDisambiguationQuery] = []
    for t in list(selected_tables or [])[: max(1, k * 2)]:
        if len(out) >= k:
            break
        schema = (t.schema or "").strip()
        name = (t.name or "").strip()
        if not name:
            continue
        tfqn = _tfqn_l(schema, name)
        cols = list(per_table_columns.get(tfqn, []) or [])
        col_names = [c.name for c in cols if (c.name or "").strip()][:3]
        if col_names:
            sel = ", ".join([f'"{c}"' for c in col_names])
        else:
            sel = "*"
        table_ident = ".".join([f'"{p}"' for p in [schema, name] if p])
        sql = f"SELECT {sel} FROM {table_ident}"
        purpose = f"{name} 테이블 샘플을 확인하여 어떤 값/형태가 있는지 빠르게 파악"
        out.append(LightDisambiguationQuery(purpose=purpose, sql=sql))

    # Pad with a harmless query if still short (should be rare).
    while len(out) < k:
        out.append(
            LightDisambiguationQuery(
                purpose="시스템 연결/쿼리 실행 가능 여부를 간단히 확인",
                sql="SELECT 1 AS ok",
            )
        )
    return out[:k]


async def append_light_queries_and_results_xml(
    *,
    context,
    question: str,
    react_run_id: Optional[str],
    table_candidates: Sequence[TableCandidate],
    selected_tables: Sequence[TableCandidate],
    per_table_columns: Dict[str, List[ColumnCandidate]],
    per_table_k: int,
    rerank_top_k: int,
    fk_relationships: Sequence[Dict[str, Any]],
    resolved_values: Sequence[Dict[str, Any]],
    similar_queries: Sequence[Dict[str, Any]],
    result_parts: List[str],
    table_description_overrides: Optional[Dict[str, str]] = None,
    exclude_light_sqls: Optional[Sequence[str]] = None,
) -> Tuple[int, int]:
    """
    1) Generate N lightweight disambiguation SELECT queries via light LLM (always).
    2) For each query: validate with SQLGuard (SELECT-only) then preview (cursor fetch) without SQL mutation.
    3) Append <light_queries> block to build_sql_context tool result XML.

    Returns (generated_count, pass_count).
    """
    started = time.perf_counter()
    q = (question or "").strip()
    if not q:
        result_parts.append("<light_queries />")
        return 0, 0

    target_k = max(1, int(getattr(settings, "build_sql_context_light_query_count", 3) or 3))
    schema_candidates_payload = _build_schema_candidates_payload(
        table_candidates=table_candidates,
        selected_tables=selected_tables,
        per_table_columns=per_table_columns,
        per_table_k=per_table_k,
        rerank_top_k=rerank_top_k,
        table_description_overrides=table_description_overrides,
    )

    generator = get_light_disambiguation_queries_generator()
    queries: List[LightDisambiguationQuery]
    mode: str
    try:
        queries, mode = await generator.generate(
            user_question=q,
            target_k=target_k,
            schema_candidates=schema_candidates_payload,
            fk_relationships=fk_relationships,
            resolved_values=resolved_values,
            similar_queries=similar_queries,
            react_run_id=react_run_id,
        )
    except Exception as exc:
        SmartLogger.log(
            "WARNING",
            "react.build_sql_context.light_queries.generate_error",
            category="react.tool.detail.build_sql_context",
            params={"react_run_id": react_run_id, "error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )
        queries, mode = [], "exception"

    if len(queries) < target_k:
        # Ensure stable output size even if the LLM under-produces.
        queries = list(queries) + list(
            _fallback_queries(
                selected_tables=selected_tables,
                per_table_columns=per_table_columns,
                target_k=(target_k - len(queries)),
            )
        )
        queries = queries[:target_k]

    # Cross-run dedup: remove queries already executed in previous build_sql_context runs
    # (e.g., during auto context-refresh loops).
    exclude_set = set()
    for s in list(exclude_light_sqls or [])[:200]:
        k = _norm_sql_key(s)
        if k:
            exclude_set.add(k)

    filtered: List[LightDisambiguationQuery] = []
    seen_local = set()
    pool = list(queries or [])
    if len(pool) < target_k * 3:
        pool = pool + list(
            _fallback_queries(
                selected_tables=selected_tables,
                per_table_columns=per_table_columns,
                target_k=max(1, target_k * 3 - len(pool)),
            )
        )
    for item in pool:
        sql_key = _norm_sql_key(getattr(item, "sql", "") or "")
        if not sql_key:
            continue
        if sql_key in exclude_set:
            continue
        if sql_key in seen_local:
            continue
        seen_local.add(sql_key)
        filtered.append(item)
        if len(filtered) >= target_k:
            break

    # If still short (rare), pad with unique harmless selects.
    pad_i = 0
    while len(filtered) < target_k:
        pad_i += 1
        sql = f"SELECT {pad_i} AS ok"
        key = _norm_sql_key(sql)
        if key in exclude_set or key in seen_local:
            continue
        seen_local.add(key)
        filtered.append(
            LightDisambiguationQuery(
                purpose="시스템 연결/쿼리 실행 가능 여부를 간단히 확인",
                sql=sql,
            )
        )

    queries = filtered[:target_k]

    guard = SQLGuard()
    executor = SQLExecutor()
    preview_limit = max(1, min(10, int(context.scaled(getattr(context, "preview_row_limit", 10) or 10))))
    timeout_s = float(min(5.0, float(getattr(context, "max_sql_seconds", 60) or 60)))

    pass_count = 0
    result_parts.append("<light_queries>")
    result_parts.append(emit_text("target_k", str(target_k)))
    result_parts.append(emit_text("mode", str(mode)))
    result_parts.append(emit_text("preview_row_limit", str(preview_limit)))
    result_parts.append(emit_text("timeout_s", str(timeout_s)))

    for i, item in enumerate(list(queries)[:target_k], start=1):
        per_started = time.perf_counter()
        sql_text = (item.sql or "").strip()
        purpose = (item.purpose or "").strip()
        result_parts.append(f'<query index="{i}">')
        result_parts.append(emit_text("purpose", purpose))
        result_parts.append(f"<sql>{to_cdata(sql_text)}</sql>")

        verdict = "FAIL"
        row_count: Optional[int] = None
        try:
            validated_sql, _ = guard.validate(sql_text)
            # MindsDB (Phase 1): apply SQL transform (datasource prefix + identifier quoting).
            db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
            if db_type in {"mysql", "mariadb"}:
                try:
                    datasource = str(getattr(context, "datasource", "") or "").strip()
                    if datasource:
                        validated_sql = prepare_sql_for_mindsdb(validated_sql, datasource).sql
                except Exception:
                    pass  # Best-effort; fall through to execute original SQL
            preview = await executor.preview_query(
                context.db_conn,
                validated_sql,
                row_limit=preview_limit,
                timeout=timeout_s,
            )
            pass_count += 1
            verdict = "PASS"
            try:
                rc = preview.get("row_count") if isinstance(preview, dict) else None
                row_count = int(rc) if rc is not None else None
            except Exception:
                row_count = None
            result_parts.append(emit_text("verdict", verdict))
            result_parts.extend(_preview_to_xml(preview=preview))
        except SQLValidationError as exc:
            verdict = "FAIL"
            result_parts.append(emit_text("verdict", verdict))
            result_parts.append(emit_text("fail_reason", f"Validation failed: {exc}"))
        except (SQLExecutionError, Exception) as exc:
            verdict = "FAIL"
            result_parts.append(emit_text("verdict", verdict))
            result_parts.append(emit_text("fail_reason", f"Preview failed: {exc}"))

        # Stream per-item progress (best-effort)
        try:
            context.emit(
                {
                    "event": "pipeline_item",
                    "pipeline": "build_sql_context",
                    "stage": "light_queries",
                    "item_type": "query_preview",
                    "iteration": 0,
                    "index": int(i),
                    "total": int(target_k),
                    "verdict": verdict,
                    "row_count": row_count,
                    "elapsed_ms": float((time.perf_counter() - per_started) * 1000.0),
                    "ts_ms": int(time.time() * 1000),
                }
            )
        except Exception:
            pass

        result_parts.append("</query>")

    result_parts.append(emit_text("elapsed_ms", f"{(time.perf_counter() - started) * 1000.0:.2f}"))
    result_parts.append("</light_queries>")

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.light_queries.done",
        category="react.tool.detail.build_sql_context",
        params={
            "react_run_id": react_run_id,
            "target_k": int(target_k),
            "generated_count": int(len(queries)),
            "pass_count": int(pass_count),
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        },
        max_inline_chars=0,
    )

    return int(len(queries)), int(pass_count)


__all__ = ["append_light_queries_and_results_xml"]


