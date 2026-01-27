"""
search_findings - 이전 성공적인 발견(Finding)을 검색하는 도구

Finding은 이전 ReAct 추론 과정에서 의미 있게 발견된 정보:
- table_discovery: 어떤 의도에 유용했던 테이블
- column_discovery: 유용했던 컬럼
- value_mapping: 자연어→코드 매핑
- join_path: 테이블 간 조인 경로
- data_insight: 데이터에서 발견된 인사이트

사용 시점:
- find_similar_query로 유사 쿼리를 찾지 못했을 때
- 새로운 의도에 맞는 테이블/컬럼 힌트가 필요할 때
"""

import time
from typing import Any, List, Optional

from app.config import settings
from app.core.embedding import EmbeddingClient
from app.models.finding import FindingRepository
from app.react.tools.context import ToolContext
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


async def execute(
    context: ToolContext,
    intent: str,
    finding_types: Optional[List[str]] = None,
    min_similarity: float = 0.5,
) -> str:
    """의도 기반으로 관련 Finding 검색"""
    started = time.perf_counter()
    result_parts: List[str] = ["<tool_result>"]
    
    SmartLogger.log(
        "INFO",
        "search_findings.start",
        category="react.tool.search_findings",
        params=sanitize_for_log({
            "react_run_id": context.react_run_id,
            "intent": intent[:200],
            "finding_types": finding_types,
            "min_similarity": min_similarity,
        }),
        max_inline_chars=0,
    )
    
    try:
        # EmbeddingClient 생성
        embedder = EmbeddingClient(context.openai_client)
        
        # FindingRepository 생성
        finding_repo = FindingRepository(context.neo4j_session, embedder)
        
        # Finding 검색
        findings = await finding_repo.search_findings_by_intent(
            intent=intent,
            finding_types=finding_types,
            min_similarity=min_similarity,
            limit=10,
        )
        
        if not findings:
            result_parts.append("<message>No relevant findings for this intent</message>")
            result_parts.append("<hint>Use search_tables to explore schema from scratch</hint>")
            result_parts.append("</tool_result>")
            return "\n".join(result_parts)
        
        result_parts.append(f"<found_count>{len(findings)}</found_count>")
        result_parts.append("<findings>")
        result_parts.append("<instruction>")
        result_parts.append("아래는 과거 유사한 의도에서 성공적으로 발견된 정보입니다.")
        result_parts.append("이 Finding들의 테이블/컬럼을 우선적으로 활용하세요.")
        result_parts.append("</instruction>")
        
        for f in findings:
            result_parts.append("<finding>")
            result_parts.append(f"<type>{f.get('finding_type', 'unknown')}</type>")
            result_parts.append(f"<similarity>{int(f.get('similarity_score', 0) * 100)}%</similarity>")
            result_parts.append(f"<intent>{f.get('intent', '')}</intent>")
            result_parts.append(f"<description>{f.get('description', '')}</description>")
            
            if f.get('reasoning'):
                result_parts.append(f"<reasoning>{f.get('reasoning')}</reasoning>")
            
            # 테이블 정보
            tables = f.get('tables') or []
            if tables:
                result_parts.append(f"<tables>{', '.join(tables)}</tables>")
                result_parts.append("<action>이 테이블을 get_table_schema로 조회하세요</action>")
            
            # 컬럼 정보
            columns = f.get('columns') or []
            if columns:
                result_parts.append(f"<columns>{', '.join(columns[:10])}</columns>")
            
            # 값 매핑 (있는 경우)
            if f.get('natural_value') and f.get('code_value'):
                result_parts.append("<value_mapping>")
                result_parts.append(f"<natural>{f.get('natural_value')}</natural>")
                result_parts.append(f"<code>{f.get('code_value')}</code>")
                result_parts.append("</value_mapping>")
            
            result_parts.append(f"<usage_count>{f.get('usage_count', 1)}</usage_count>")
            result_parts.append("</finding>")
        
        result_parts.append("</findings>")
        
        # 최고 유사도가 높으면 강력 권장
        top_score = findings[0].get('similarity_score', 0) if findings else 0
        if top_score >= 0.8:
            result_parts.append("<action_required>USE_FINDING_IMMEDIATELY</action_required>")
            result_parts.append("<instruction>")
            result_parts.append("매우 유사한 Finding이 있습니다. 위 테이블/컬럼을 바로 활용하세요.")
            result_parts.append("</instruction>")
        elif top_score >= 0.65:
            result_parts.append("<action_required>CONSIDER_FINDINGS</action_required>")
            result_parts.append("<instruction>")
            result_parts.append("관련 Finding이 있습니다. 위 정보를 참고하여 탐색 범위를 좁히세요.")
            result_parts.append("</instruction>")
        
        result_parts.append("</tool_result>")
        
    except Exception as exc:
        result_parts.append(f"<error>search_findings failed: {str(exc)[:200]}</error>")
        result_parts.append("</tool_result>")
        SmartLogger.log(
            "ERROR",
            "search_findings.error",
            category="react.tool.search_findings",
            params=sanitize_for_log({
                "react_run_id": context.react_run_id,
                "intent": intent[:200],
                "exception": repr(exc),
            }),
            max_inline_chars=0,
        )
    
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    SmartLogger.log(
        "INFO",
        "search_findings.done",
        category="react.tool.search_findings",
        params=sanitize_for_log({
            "react_run_id": context.react_run_id,
            "elapsed_ms": elapsed_ms,
            "findings_count": len(findings) if 'findings' in dir() else 0,
        }),
        max_inline_chars=0,
    )
    
    return "\n".join(result_parts)


# Tool metadata
NAME = "search_findings"
DESCRIPTION = """Search for previously successful discoveries (Findings) based on intent.

Findings are valuable insights from past ReAct reasoning:
- table_discovery: Tables that were useful for similar intents
- column_discovery: Columns that provided needed data
- value_mapping: Natural language → code mappings
- join_path: Useful join paths between tables

Use this when:
1. find_similar_query found no direct match
2. You need hints about which tables/columns to explore
3. Looking for value mappings (e.g., "청주정수장" → "BPLC001")

Returns tables and columns that were successfully used for similar intents."""

PARAMETERS = {
    "intent": {
        "type": "string",
        "description": "The intent or goal you're trying to achieve (e.g., '정수장의 탁도 데이터 조회')"
    },
    "finding_types": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Filter by finding types: table_discovery, column_discovery, value_mapping, join_path, data_insight"
    },
    "min_similarity": {
        "type": "number",
        "description": "Minimum similarity threshold (0.0-1.0, default 0.5)"
    }
}

REQUIRED = ["intent"]
