"""
SQL Explain 툴 - 캐시 기반 최적화 포함.

검증된 쿼리 패턴은 explain을 건너뛰어 성능을 향상시킵니다.
"""

import logging
from typing import List, Optional

from app.config import settings
from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.deps import neo4j_conn
from app.react.tools.context import ToolContext
from app.react.generators.explain_analysis_generator import ExplainAnalysisGenerator, ExplainAnalysisResult
from app.models.explain_cache import (
    ExplainCacheEntry,
    ExplainCacheRepository,
    QueryPatternExtractor,
)
from app.smart_logger import SmartLogger

logger = logging.getLogger(__name__)

# 캐시 사용 여부 설정
EXPLAIN_CACHE_ENABLED = settings.explain_cache_enabled
EXPLAIN_CACHE_COST_THRESHOLD = settings.explain_cache_cost_threshold


async def execute(
    context: ToolContext,
    sql: str,
    skip_cache_check: bool = False,
) -> str:
    """
    SQL Explain 툴.
    
    캐시된 패턴이 있고 안전하다고 판단되면 explain을 건너뜁니다.
    
    Args:
        context: 툴 컨텍스트
        sql: 분석할 SQL 쿼리
        skip_cache_check: True면 캐시 확인을 건너뜀 (강제 재검증)
    """
    result_parts: List[str] = ["<tool_result>"]
    
    # MySQL/MindsDB는 EXPLAIN을 지원하지 않으므로 건너뜀
    if settings.target_db_type == "mysql":
        result_parts.append("<explain_skip reason=\"mysql_not_supported\">")
        result_parts.append("<note>MySQL/MindsDB 연결에서는 EXPLAIN 분석이 지원되지 않습니다. 쿼리는 유효한 것으로 간주됩니다.</note>")
        result_parts.append(f"<sql><![CDATA[{sql}]]></sql>")
        result_parts.append("</explain_skip>")
        result_parts.append("</tool_result>")
        return "\n".join(result_parts)

    try:
        # 1. 캐시 확인 (활성화된 경우)
        cache_result = await _check_explain_cache(
            sql=sql,
            skip_cache=skip_cache_check,
            react_run_id=context.react_run_id,
        )
        
        if cache_result["skip"]:
            # 캐시 히트 - explain 건너뛰기
            cached_entry = cache_result["cached_entry"]
            result_parts.append(_build_cached_result_xml(cached_entry, cache_result["reason"]))
            result_parts.append("</tool_result>")
            return "\n".join(result_parts)
        
        # 2. 실제 explain 실행
        explain_analysis_generator = ExplainAnalysisGenerator()
        explain_analysis_result: ExplainAnalysisResult = await explain_analysis_generator.generate(
            sql=sql,
            db_conn=context.db_conn,
            react_run_id=context.react_run_id,
        )
        
        # 3. 캐시에 저장
        await _save_to_explain_cache(
            sql=sql,
            result=explain_analysis_result,
            react_run_id=context.react_run_id,
        )
        
        result_parts.append(explain_analysis_result.to_xml_str())
        
    except (SQLValidationError, SQLExecutionError) as exc:
        # 오류 발생 시 해당 패턴을 unsafe로 표시
        await _mark_pattern_unsafe(sql, str(exc), context.react_run_id)
        result_parts.append(f"<error>{str(exc)}</error>")
    except Exception as exc:
        result_parts.append(f"<error>Unexpected error: {str(exc)}</error>")

    result_parts.append("</tool_result>")
    return "\n".join(result_parts)


async def _check_explain_cache(
    sql: str,
    skip_cache: bool = False,
    react_run_id: Optional[str] = None,
) -> dict:
    """
    캐시 확인 및 건너뛰기 여부 결정.
    
    Returns:
        {"skip": bool, "cached_entry": Optional[ExplainCacheEntry], "reason": str}
    """
    if not EXPLAIN_CACHE_ENABLED or skip_cache:
        return {"skip": False, "cached_entry": None, "reason": "cache_disabled"}
    
    try:
        # Neo4j 드라이버 가져오기
        await neo4j_conn.connect()
        if not neo4j_conn.driver:
            return {"skip": False, "cached_entry": None, "reason": "no_driver"}
        
        # 패턴 추출
        pattern = QueryPatternExtractor.extract_pattern(sql)
        
        # 캐시 저장소
        repo = ExplainCacheRepository(neo4j_conn.driver)
        
        # 건너뛰기 가능 여부 확인
        skip, cached_entry, reason = await repo.should_skip_explain(
            pattern=pattern,
            max_cost_threshold=EXPLAIN_CACHE_COST_THRESHOLD,
        )
        
        if skip and cached_entry:
            SmartLogger.log(
                "INFO",
                "react.explain.cache_hit",
                category="react.explain.cache_hit",
                params={
                    "react_run_id": react_run_id,
                    "pattern_hash": pattern.pattern_hash,
                    "tables": pattern.tables,
                    "reason": reason,
                    "cached_cost": cached_entry.estimated_cost,
                    "validation_count": cached_entry.validation_count,
                },
            )
        else:
            SmartLogger.log(
                "DEBUG",
                "react.explain.cache_miss",
                category="react.explain.cache_miss",
                params={
                    "react_run_id": react_run_id,
                    "pattern_hash": pattern.pattern_hash,
                    "tables": pattern.tables,
                    "reason": reason,
                },
            )
        
        return {"skip": skip, "cached_entry": cached_entry, "reason": reason}
        
    except Exception as exc:
        logger.warning("Explain cache check failed: %s", exc, exc_info=True)
        return {"skip": False, "cached_entry": None, "reason": f"error: {exc}"}


async def _save_to_explain_cache(
    sql: str,
    result: ExplainAnalysisResult,
    react_run_id: Optional[str] = None,
) -> None:
    """성공적인 explain 결과를 캐시에 저장."""
    if not EXPLAIN_CACHE_ENABLED:
        return
    
    try:
        await neo4j_conn.connect()
        if not neo4j_conn.driver:
            return
        
        # 패턴 추출
        pattern = QueryPatternExtractor.extract_pattern(sql)
        
        # 비용 기반 안전성 판단
        is_safe = True
        if result.execution_plan.total_cost and result.execution_plan.total_cost > 5000:
            is_safe = False
        
        # 캐시 항목 생성
        entry = ExplainCacheEntry(
            pattern_hash=pattern.pattern_hash,
            tables=pattern.tables,
            join_columns=pattern.join_columns,
            filter_columns=pattern.filter_columns,
            estimated_cost=float(result.execution_plan.total_cost or 0),
            estimated_time_ms=float(result.execution_plan.execution_time_ms or 0),
            estimated_rows=int(result.execution_plan.row_count or 0),
            is_safe=is_safe,
            sample_sql=sql[:500],  # 처음 500자만 저장
        )
        
        # 저장
        repo = ExplainCacheRepository(neo4j_conn.driver)
        await repo.save(entry)
        
        SmartLogger.log(
            "INFO",
            "react.explain.cache_saved",
            category="react.explain.cache_saved",
            params={
                "react_run_id": react_run_id,
                "pattern_hash": pattern.pattern_hash,
                "tables": pattern.tables,
                "estimated_cost": entry.estimated_cost,
                "is_safe": is_safe,
            },
        )
        
    except Exception as exc:
        logger.warning("Failed to save explain cache: %s", exc, exc_info=True)


async def _mark_pattern_unsafe(
    sql: str,
    error_message: str,
    react_run_id: Optional[str] = None,
) -> None:
    """오류가 발생한 패턴을 안전하지 않음으로 표시."""
    if not EXPLAIN_CACHE_ENABLED:
        return
    
    try:
        await neo4j_conn.connect()
        if not neo4j_conn.driver:
            return
        
        pattern = QueryPatternExtractor.extract_pattern(sql)
        repo = ExplainCacheRepository(neo4j_conn.driver)
        await repo.mark_unsafe(pattern.pattern_hash)
        
        SmartLogger.log(
            "WARNING",
            "react.explain.pattern_marked_unsafe",
            category="react.explain.pattern_marked_unsafe",
            params={
                "react_run_id": react_run_id,
                "pattern_hash": pattern.pattern_hash,
                "error_message": error_message[:200],
            },
        )
        
    except Exception as exc:
        logger.warning("Failed to mark pattern unsafe: %s", exc, exc_info=True)


def _build_cached_result_xml(entry: ExplainCacheEntry, skip_reason: str) -> str:
    """캐시된 결과를 XML 형식으로 반환."""
    parts: List[str] = ["<explain_analysis_result cached=\"true\">"]
    parts.append(f"<cache_info>")
    parts.append(f"<skip_reason>{skip_reason}</skip_reason>")
    parts.append(f"<pattern_hash>{entry.pattern_hash}</pattern_hash>")
    parts.append(f"<validation_count>{entry.validation_count}</validation_count>")
    parts.append(f"<cached_at>{entry.validated_at.isoformat() if entry.validated_at else ''}</cached_at>")
    parts.append(f"</cache_info>")
    
    parts.append("<execution_plan>")
    parts.append(f"<total_cost>{entry.estimated_cost}</total_cost>")
    parts.append(f"<execution_time_ms>{entry.estimated_time_ms}</execution_time_ms>")
    parts.append(f"<row_count>{entry.estimated_rows}</row_count>")
    parts.append("<note>이 결과는 캐시된 정보입니다. 이전 검증에서 이 쿼리 패턴이 안전하다고 확인되었습니다.</note>")
    parts.append("</execution_plan>")
    
    parts.append("<tables>")
    for table in entry.tables:
        parts.append(f"<table>{table}</table>")
    parts.append("</tables>")
    
    parts.append("<risk_analysis>")
    parts.append("<summary>이전 검증 결과에 따르면 이 쿼리 패턴은 성능 문제가 없습니다.</summary>")
    parts.append("</risk_analysis>")
    
    parts.append("</explain_analysis_result>")
    return "\n".join(parts)
