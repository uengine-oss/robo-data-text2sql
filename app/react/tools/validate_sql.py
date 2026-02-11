"""
validate_sql - 통합 SQL 검증/샘플 실행 도구

역할:
- SQLGuard 기반 안전 검증
- explain 기반 성능 판단(PASS/FAIL)
- PASS 시 샘플 실행(행 제한은 SQL 변형 없이 cursor fetch 로 수행)

주의:
- 기본 경로에서 복잡한 서브쿼리 검증을 수행하지 않는다.
  (ExplainAnalysisGenerator의 validation queries 실행은 비활성화)
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any, List, Optional
from xml.sax.saxutils import escape as xml_escape

from app.config import settings
from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.core.sql_mindsdb_prepare import is_passthrough_query, prepare_sql_for_mindsdb
from app.react.generators.explain_analysis_generator import ExplainAnalysisGenerator
from app.react.generators.validate_sql_repair_generator import get_validate_sql_repair_generator
from app.react.tools.context import ToolContext
from app.react.utils.sql_autorepair import auto_repair_sql_for_postgres_error


def _to_cdata(value: str) -> str:
    return f"<![CDATA[{value}]]>"


def _emit_text(tag: str, text: str) -> str:
    return f"<{tag}>{xml_escape(text or '')}</{tag}>"


def _emit_error(message: str) -> str:
    return f"<tool_result><error>{xml_escape(message or '')}</error></tool_result>"


def _suggest_fixes(*, risk_summary: str, max_sql_seconds: int) -> List[str]:
    """
    간단한 규칙 기반 개선 힌트. (FAIL 시 ReAct가 다음 루프를 수행할 수 있도록)
    """
    summary = (risk_summary or "").lower()
    fixes: List[str] = []
    # Common SQL authoring error: missing column/alias/table reference
    if (
        "column" in summary and "does not exist" in summary
    ) or ("칼럼" in summary and "없" in summary) or ("column" in summary and "not found" in summary):
        fixes.append("존재하지 않는 컬럼/별칭(alias)을 참조하고 있습니다. SELECT/JOIN/WHERE의 컬럼명과 별칭을 실제 스키마에 맞게 수정해 보세요.")
        fixes.append("가능하면 build_sql_context를 다시 호출해, 실제 존재하는 컬럼명을 기준으로 SQL을 재작성해 보세요.")
    if "seq scan" in summary or "full table scan" in summary:
        fixes.append("조회 범위를 줄이기 위해 기간 필터(예: 최근 1주/1개월/3개월)를 추가해 보세요.")
        fixes.append("가능하면 더 선택적인 조건(예: 특정 지역/지점/상태)을 추가해 보세요.")
    if "nested loop" in summary:
        fixes.append("조인 대상 테이블 수를 줄이거나, 더 선택적인 필터를 먼저 적용해 보세요.")
    if not fixes:
        fixes.append(
            f"제한 시간({max_sql_seconds}초) 내 처리가 어려울 수 있습니다. 기간/범위를 축소하거나 집계 수준을 올려 보세요."
        )
    # Dedup + cap
    uniq: List[str] = []
    seen = set()
    for f in fixes:
        key = f.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(key)
    return uniq[:5]


async def execute(context: ToolContext, sql: str) -> str:
    """
    validate_sql tool entrypoint.
    Returns <tool_result> XML.
    """
    sql_text = (sql or "").strip()
    if not sql_text:
        return _emit_error("sql parameter is required")

    started = time.perf_counter()
    max_total_s = float(getattr(context, "max_sql_seconds", 60) or 60)

    def _remaining_s() -> float:
        # Keep a small buffer so downstream DB calls don't start with near-zero budget.
        elapsed = time.perf_counter() - started
        return max(0.0, max_total_s - elapsed - 0.2)

    llm_repair_used = False
    auto_rewrite_reasons: List[str] = []
    auto_rewrite_source: str = ""

    def _record_rewrite(source: str, reasons: List[str]) -> None:
        nonlocal auto_rewrite_source
        rr = [str(x or "").strip() for x in (reasons or []) if str(x or "").strip()]
        if not rr:
            return
        auto_rewrite_reasons.extend(rr[:10])
        s0 = (source or "").strip().lower()
        if not auto_rewrite_source:
            auto_rewrite_source = s0 or "rule"
        elif auto_rewrite_source != (s0 or auto_rewrite_source):
            auto_rewrite_source = "mixed"

    async def _try_llm_repair(*, current_sql: str, error_text: str, stage: str) -> Optional[str]:
        nonlocal llm_repair_used
        if llm_repair_used:
            return None
        rem = _remaining_s()
        if rem <= 0.8:
            return None
        gen = get_validate_sql_repair_generator()
        try:
            # LLM timeout: bounded by remaining tool budget; also cap to keep p95 stable.
            timeout_s = min(rem, 12.0)
            repaired, mode = await asyncio.wait_for(
                gen.generate(
                    db_type=str(settings.target_db_type or ""),
                    error_text=str(error_text or ""),
                    current_sql=str(current_sql or ""),
                    react_run_id=context.react_run_id,
                ),
                timeout=timeout_s,
            )
        except Exception:
            return None
        repaired_sql = (repaired or "").strip()
        if not repaired_sql:
            return None
        if repaired_sql.strip() == (current_sql or "").strip():
            return None
        llm_repair_used = True
        _record_rewrite("llm", [f"rewrite: llm repair on {stage} ({mode})"])
        return repaired_sql

    result_parts: List[str] = ["<tool_result>", "<validate_sql_result>"]

    # 1) Safety validation (do NOT mutate SQL for LIMIT)
    guard = SQLGuard()
    try:
        validated_sql, _ = guard.validate(sql_text)
    except SQLValidationError as exc:
        # One-shot LLM repair on SQLGuard failures (e.g., comments / parse errors).
        repaired_sql = await _try_llm_repair(current_sql=sql_text, error_text=str(exc), stage="sqlguard")
        if repaired_sql:
            try:
                validated_sql, _ = guard.validate(repaired_sql)
            except Exception:
                validated_sql = ""
        else:
            validated_sql = ""
        if not validated_sql:
            result_parts.append(_emit_text("verdict", "FAIL"))
            result_parts.append(_emit_text("fail_reason", str(exc)))
            result_parts.append("</validate_sql_result>")
            result_parts.append("</tool_result>")
            return "\n".join(result_parts)
    except Exception as exc:
        result_parts.append(_emit_text("verdict", "FAIL"))
        result_parts.append(_emit_text("fail_reason", f"Unexpected validation error: {exc}"))
        result_parts.append("</validate_sql_result>")
        result_parts.append("</tool_result>")
        return "\n".join(result_parts)

    # MindsDB-only (Phase 1): skip EXPLAIN entirely (D5) and rely on preview + timeout (D11).
    db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
    if db_type in {"mysql", "mariadb"}:
        # Deterministic pipeline:
        # - Try original SQL first (passthrough-safe)
        # - If preview fails, apply deterministic prepare (transform OR passthrough inner sanitize) and retry once
        datasource = str(getattr(context, "datasource", "") or "").strip()
        used_sql = validated_sql

        def _wrap_as_passthrough(*, inner_sql: str, ds: str) -> str:
            """
            MindsDB passthrough form:
              SELECT * FROM `datasource` ( <inner_sql_in_external_db_dialect> )

            This is a pragmatic escape hatch for complex SQL (CTE/window) that MindsDB's MySQL
            parser rejects in "datasource.schema.table" addressing mode.
            """
            inner = (inner_sql or "").strip().rstrip(";").strip()
            return f"SELECT * FROM `{ds}` (\n{inner}\n)"

        # PASS criteria (redefined): preview must succeed within max_sql_seconds.
        preview_limit = max(1, int(context.scaled(context.preview_row_limit)))
        timeout_s = float(context.max_sql_seconds)
        executor = SQLExecutor()
        try:
            try:
                preview = await executor.preview_query(
                    context.db_conn,
                    used_sql,
                    row_limit=preview_limit,
                    timeout=timeout_s,
                )
            except Exception as first_exc:
                # Fallback #1: try passthrough wrap for complex SQL (CTE/window).
                # This keeps the inner SQL in external DB dialect and avoids MindsDB parser limitations.
                passthrough_ok = False
                if datasource and not is_passthrough_query(used_sql, datasource):
                    msg = str(first_exc or "").lower()
                    looks_like_parse_error = any(
                        k in msg
                        for k in (
                            "cannot be parsed",
                            "syntax error",
                            "unknown input",
                            "parse",
                        )
                    )
                    looks_like_complex_sql = bool(
                        re.search(r"(?is)\b(with|over|lag|lead|row_number|rank|dense_rank)\b", used_sql)
                    )
                    if looks_like_parse_error or looks_like_complex_sql:
                        passthrough_sql = _wrap_as_passthrough(inner_sql=validated_sql, ds=datasource)
                        if passthrough_sql.strip() and passthrough_sql.strip() != used_sql.strip():
                            try:
                                preview = await executor.preview_query(
                                    context.db_conn,
                                    passthrough_sql,
                                    row_limit=preview_limit,
                                    timeout=timeout_s,
                                )
                                used_sql = passthrough_sql
                                _record_rewrite("mindsdb_prepare", ["rewrite: wrap_as_passthrough_query"])
                                passthrough_ok = True
                            except Exception:
                                passthrough_ok = False

                # Fallback #2 (existing): deterministic MindsDB transform and retry once.
                if not passthrough_ok:
                    prep = prepare_sql_for_mindsdb(validated_sql, datasource)
                    if prep.sql.strip() != used_sql.strip():
                        used_sql = prep.sql
                        _record_rewrite("mindsdb_prepare", [f"rewrite: {prep.reason}"])
                        preview = await executor.preview_query(
                            context.db_conn,
                            used_sql,
                            row_limit=preview_limit,
                            timeout=timeout_s,
                        )
                    else:
                        raise first_exc
            result_parts.append(_emit_text("execution_time_ms", str(preview.get("execution_time_ms", 0))))
            result_parts.append(_emit_text("verdict", "PASS"))
            result_parts.append(f"<selected_sql>{_to_cdata(used_sql)}</selected_sql>")
            if auto_rewrite_reasons:
                result_parts.append("<auto_rewrite>")
                result_parts.append(_emit_text("applied", "true"))
                if auto_rewrite_source:
                    result_parts.append(_emit_text("source", auto_rewrite_source))
                for r in auto_rewrite_reasons[:8]:
                    result_parts.append(_emit_text("reason", r))
                result_parts.append("</auto_rewrite>")
            result_parts.append("<preview>")
            result_parts.append(_emit_text("row_count", str(preview.get("row_count", 0))))
            cols = preview.get("columns", []) or []
            result_parts.append("<columns>")
            for c in cols:
                result_parts.append(_emit_text("column", str(c)))
            result_parts.append("</columns>")
            result_parts.append("<rows>")
            for idx, row in enumerate(preview.get("rows", []) or [], start=1):
                result_parts.append(f'<row index="{idx}">')
                for col_name, value in zip(cols, row):
                    if value is None:
                        result_parts.append(f'<cell column="{xml_escape(str(col_name), {"\"": "&quot;"})}" />')
                    else:
                        result_parts.append(
                            f'<cell column="{xml_escape(str(col_name), {"\"": "&quot;"})}">{_to_cdata(str(value))}</cell>'
                        )
                result_parts.append("</row>")
            result_parts.append("</rows>")
            result_parts.append(_emit_text("preview_execution_time_ms", str(preview.get("execution_time_ms", 0))))
            result_parts.append("</preview>")
            result_parts.append("</validate_sql_result>")
            result_parts.append("</tool_result>")
            return "\n".join(result_parts)
        except Exception as exc:
            # Preview 실패는 FAIL로 내려 ReAct가 개선/축소를 시도하도록 한다.
            result_parts.append(_emit_text("execution_time_ms", "0"))
            result_parts.append(_emit_text("verdict", "FAIL"))
            result_parts.append(_emit_text("fail_reason", f"Preview failed: {exc}"))
            result_parts.append("<suggested_fixes>")
            for fix in _suggest_fixes(
                risk_summary=str(exc),
                max_sql_seconds=int(context.max_sql_seconds),
            ):
                result_parts.append(_emit_text("fix", fix))
            result_parts.append("</suggested_fixes>")
            result_parts.append(f"<selected_sql>{_to_cdata(used_sql)}</selected_sql>")
            result_parts.append("</validate_sql_result>")
            result_parts.append("</tool_result>")
            return "\n".join(result_parts)

    # 2) Explain analysis (PostgreSQL path only; MindsDB skips this block)
    generator = ExplainAnalysisGenerator()
    explain_result = None
    exec_ms = 0.0
    last_exc: Optional[Exception] = None
    for attempt in range(1, 3):  # at most 2 tries: original + 1 rewrite
        try:
            explain_result = await generator.generate(
                sql=validated_sql,
                db_conn=context.db_conn,
                react_run_id=context.react_run_id,
                max_sql_seconds=int(context.max_sql_seconds),
            )
            exec_ms = explain_result.execution_plan.execution_time_ms or 0
            break
        except Exception as exc:
            last_exc = exc
            # Try deterministic repair first (Postgres-only hint + type fixes).
            if attempt == 1:
                repaired, reasons = auto_repair_sql_for_postgres_error(
                    validated_sql,
                    str(exc),
                    db_type=str(settings.target_db_type or ""),
                )
                if repaired.strip() and repaired.strip() != validated_sql.strip():
                    try:
                        validated_sql, _ = guard.validate(repaired.strip())
                        _record_rewrite("rule", reasons or ["rewrite: auto-repair on EXPLAIN error"])
                        continue
                    except Exception:
                        # If rewrite breaks SQLGuard, fall back to original error.
                        pass

                # Fallback: one-shot LLM repair (at most once per validate_sql call).
                llm_sql = await _try_llm_repair(current_sql=validated_sql, error_text=str(exc), stage="explain")
                if llm_sql:
                    try:
                        validated_sql, _ = guard.validate(llm_sql.strip())
                        continue
                    except Exception:
                        pass
            break

    if explain_result is None:
        # IMPORTANT: Never let EXPLAIN/metadata exceptions escape this tool.
        # If we raise here, execute_tool/agent will terminate the whole ReAct run.
        result_parts.append(_emit_text("execution_time_ms", "0"))
        result_parts.append(_emit_text("verdict", "FAIL"))
        result_parts.append(_emit_text("fail_reason", f"EXPLAIN failed: {last_exc}"))
        if auto_rewrite_reasons:
            result_parts.append("<auto_rewrite>")
            result_parts.append(_emit_text("applied", "true"))
            if auto_rewrite_source:
                result_parts.append(_emit_text("source", auto_rewrite_source))
            for r in auto_rewrite_reasons[:8]:
                result_parts.append(_emit_text("reason", r))
            result_parts.append("</auto_rewrite>")
        result_parts.append("<suggested_fixes>")
        for fix in _suggest_fixes(
            risk_summary=str(last_exc),
            max_sql_seconds=int(context.max_sql_seconds),
        ):
            result_parts.append(_emit_text("fix", fix))
        result_parts.append("</suggested_fixes>")
        result_parts.append(f"<selected_sql>{_to_cdata(validated_sql)}</selected_sql>")
        result_parts.append("</validate_sql_result>")
        result_parts.append("</tool_result>")
        return "\n".join(result_parts)

    result_parts.append(_emit_text("execution_time_ms", str(exec_ms)))
    if auto_rewrite_reasons:
        result_parts.append("<auto_rewrite>")
        result_parts.append(_emit_text("applied", "true"))
        if auto_rewrite_source:
            result_parts.append(_emit_text("source", auto_rewrite_source))
        for r in auto_rewrite_reasons[:8]:
            result_parts.append(_emit_text("reason", r))
        result_parts.append("</auto_rewrite>")

    limit_ms = int(max(1, int(context.max_sql_seconds)) * 1000)
    if exec_ms and exec_ms > limit_ms:
        result_parts.append(_emit_text("verdict", "FAIL"))
        # FAIL reason: prefer LLM reason when available (policy: reasons only on FAIL)
        llm_reason = (explain_result.fail_reason or "").strip()
        fail_reason = llm_reason or f"Explain estimated time {int(exec_ms)}ms exceeds limit {limit_ms}ms"
        result_parts.append(_emit_text("fail_reason", fail_reason))
        result_parts.append("<suggested_fixes>")
        llm_fixes = getattr(explain_result, "suggested_fixes", None) or []
        used_fixes = list(llm_fixes) if llm_fixes else _suggest_fixes(
            risk_summary=explain_result.fail_reason or "",
            max_sql_seconds=int(context.max_sql_seconds),
        )
        for fix in used_fixes:
            result_parts.append(_emit_text("fix", fix))
        result_parts.append("</suggested_fixes>")
        result_parts.append(f"<selected_sql>{_to_cdata(validated_sql)}</selected_sql>")
        result_parts.append("</validate_sql_result>")
        result_parts.append("</tool_result>")
        return "\n".join(result_parts)

    # 3) PASS -> preview sample rows without SQL mutation
    preview_limit = max(1, int(context.scaled(context.preview_row_limit)))
    timeout_s = float(context.max_sql_seconds)
    try:
        executor = SQLExecutor()
        preview = await executor.preview_query(
            context.db_conn,
            validated_sql,
            row_limit=preview_limit,
            timeout=timeout_s,
        )
        result_parts.append(_emit_text("verdict", "PASS"))
        result_parts.append(f"<selected_sql>{_to_cdata(validated_sql)}</selected_sql>")
        if auto_rewrite_reasons:
            result_parts.append("<auto_rewrite>")
            result_parts.append(_emit_text("applied", "true"))
            if auto_rewrite_source:
                result_parts.append(_emit_text("source", auto_rewrite_source))
            for r in auto_rewrite_reasons[:8]:
                result_parts.append(_emit_text("reason", r))
            result_parts.append("</auto_rewrite>")
        result_parts.append("<preview>")
        result_parts.append(_emit_text("row_count", str(preview.get("row_count", 0))))
        cols = preview.get("columns", []) or []
        result_parts.append("<columns>")
        for c in cols:
            result_parts.append(_emit_text("column", str(c)))
        result_parts.append("</columns>")
        result_parts.append("<rows>")
        for idx, row in enumerate(preview.get("rows", []) or [], start=1):
            result_parts.append(f'<row index="{idx}">')
            for col_name, value in zip(cols, row):
                if value is None:
                    result_parts.append(f'<cell column="{xml_escape(str(col_name), {"\"": "&quot;"})}" />')
                else:
                    result_parts.append(
                        f'<cell column="{xml_escape(str(col_name), {"\"": "&quot;"})}">{_to_cdata(str(value))}</cell>'
                    )
            result_parts.append("</row>")
        result_parts.append("</rows>")
        result_parts.append(_emit_text("preview_execution_time_ms", str(preview.get("execution_time_ms", 0))))
        result_parts.append("</preview>")
    except (SQLExecutionError, Exception) as exc:
        # Preview 실패는 FAIL로 내려 ReAct가 개선/축소를 시도하도록 한다.
        # 단, 흔한 타입 오류는 1회 자동 수리 후 preview를 재시도한다.
        executor = SQLExecutor()
        repaired_validated = ""
        preview2 = None

        # 1) Deterministic repair
        repaired, reasons = auto_repair_sql_for_postgres_error(
            validated_sql,
            str(exc),
            db_type=str(settings.target_db_type or ""),
        )
        if repaired.strip() and repaired.strip() != validated_sql.strip():
            try:
                repaired_validated, _ = guard.validate(repaired.strip())
                preview2 = await executor.preview_query(
                    context.db_conn,
                    repaired_validated,
                    row_limit=preview_limit,
                    timeout=timeout_s,
                )
                _record_rewrite("rule", reasons or ["rewrite: auto-repair on Preview error"])
            except Exception:
                repaired_validated = ""
                preview2 = None

        # 2) LLM repair (only if deterministic failed)
        if preview2 is None and not llm_repair_used:
            llm_sql = await _try_llm_repair(current_sql=validated_sql, error_text=str(exc), stage="preview")
            if llm_sql:
                try:
                    repaired_validated, _ = guard.validate(llm_sql.strip())
                    preview2 = await executor.preview_query(
                        context.db_conn,
                        repaired_validated,
                        row_limit=preview_limit,
                        timeout=timeout_s,
                    )
                except Exception:
                    repaired_validated = ""
                    preview2 = None

        if preview2 is not None and repaired_validated:
            # Success: emit PASS using repaired SQL and return early.
            result_parts.append(_emit_text("verdict", "PASS"))
            result_parts.append(f"<selected_sql>{_to_cdata(repaired_validated)}</selected_sql>")
            if auto_rewrite_reasons:
                result_parts.append("<auto_rewrite>")
                result_parts.append(_emit_text("applied", "true"))
                if auto_rewrite_source:
                    result_parts.append(_emit_text("source", auto_rewrite_source))
                for r in auto_rewrite_reasons[:8]:
                    result_parts.append(_emit_text("reason", r))
                result_parts.append("</auto_rewrite>")
            result_parts.append("<preview>")
            result_parts.append(_emit_text("row_count", str(preview2.get("row_count", 0))))
            cols = preview2.get("columns", []) or []
            result_parts.append("<columns>")
            for c in cols:
                result_parts.append(_emit_text("column", str(c)))
            result_parts.append("</columns>")
            result_parts.append("<rows>")
            for idx, row in enumerate(preview2.get("rows", []) or [], start=1):
                result_parts.append(f'<row index="{idx}">')
                for col_name, value in zip(cols, row):
                    if value is None:
                        result_parts.append(f'<cell column="{xml_escape(str(col_name), {"\"": "&quot;"})}" />')
                    else:
                        result_parts.append(
                            f'<cell column="{xml_escape(str(col_name), {"\"": "&quot;"})}">{_to_cdata(str(value))}</cell>'
                        )
                result_parts.append("</row>")
            result_parts.append("</rows>")
            result_parts.append(_emit_text("preview_execution_time_ms", str(preview2.get("execution_time_ms", 0))))
            result_parts.append("</preview>")
            result_parts.append("</validate_sql_result>")
            result_parts.append("</tool_result>")
            return "\n".join(result_parts)

        result_parts.append(_emit_text("verdict", "FAIL"))
        result_parts.append(_emit_text("fail_reason", f"Preview failed: {exc}"))
        result_parts.append("<suggested_fixes>")
        # ExplainAnalysisResult no longer exposes legacy risk_analysis_summary.
        # Provide best-effort summary for rule-based fallback suggestions.
        risk_summary = (
            getattr(explain_result, "risk_analysis_summary", None)
            or getattr(explain_result, "fail_reason", None)
            or getattr(explain_result, "llm_raw_response", None)
            or ""
        )
        for fix in _suggest_fixes(risk_summary=str(risk_summary), max_sql_seconds=int(context.max_sql_seconds)):
            result_parts.append(_emit_text("fix", fix))
        result_parts.append("</suggested_fixes>")
        result_parts.append(f"<selected_sql>{_to_cdata(validated_sql)}</selected_sql>")

    result_parts.append("</validate_sql_result>")
    result_parts.append("</tool_result>")
    return "\n".join(result_parts)


