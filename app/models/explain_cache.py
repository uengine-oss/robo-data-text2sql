"""
ExplainCache: Neo4j 기반 Explain 결과 캐싱 시스템.

검증된 쿼리 패턴(테이블+조인 구조)을 저장하여
동일 패턴 재사용 시 explain 검증을 건너뛸 수 있도록 함.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import AsyncDriver

logger = logging.getLogger(__name__)


@dataclass
class QueryPattern:
    """SQL 쿼리에서 추출한 패턴 정보."""
    tables: List[str]  # 정렬된 테이블 목록 (schema.table 형식)
    join_columns: List[Tuple[str, str]]  # 조인 컬럼 쌍 [(col1, col2), ...]
    filter_columns: List[str]  # WHERE 절에 사용된 컬럼들
    pattern_hash: str = ""  # 패턴의 고유 해시

    def __post_init__(self):
        if not self.pattern_hash:
            self.pattern_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """패턴을 고유하게 식별하는 해시 생성."""
        # 테이블 정렬
        sorted_tables = sorted(self.tables)
        # 조인 컬럼 정렬 (각 쌍 내에서도 정렬)
        sorted_joins = sorted(tuple(sorted(pair)) for pair in self.join_columns)
        # 필터 컬럼 정렬
        sorted_filters = sorted(self.filter_columns)
        
        pattern_str = f"tables:{','.join(sorted_tables)}|joins:{sorted_joins}|filters:{sorted_filters}"
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:32]


@dataclass
class ExplainCacheEntry:
    """Neo4j에 저장되는 Explain 캐시 항목."""
    pattern_hash: str
    tables: List[str]
    join_columns: List[Tuple[str, str]]
    filter_columns: List[str]
    estimated_cost: float
    estimated_time_ms: float
    estimated_rows: int
    actual_execution_time_ms: Optional[float] = None
    validated_at: datetime = field(default_factory=datetime.now)
    validation_count: int = 1
    max_cost_threshold: float = 1000.0  # 이 비용 이하면 캐시 신뢰
    is_safe: bool = True  # 쿼리가 안전한지 (성능 문제 없음)
    sample_sql: str = ""  # 대표 SQL 예시

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_hash": self.pattern_hash,
            "tables": self.tables,
            "join_columns": [list(pair) for pair in self.join_columns],
            "filter_columns": self.filter_columns,
            "estimated_cost": self.estimated_cost,
            "estimated_time_ms": self.estimated_time_ms,
            "estimated_rows": self.estimated_rows,
            "actual_execution_time_ms": self.actual_execution_time_ms,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "validation_count": self.validation_count,
            "max_cost_threshold": self.max_cost_threshold,
            "is_safe": self.is_safe,
            "sample_sql": self.sample_sql,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExplainCacheEntry":
        validated_at = data.get("validated_at")
        if isinstance(validated_at, str):
            validated_at = datetime.fromisoformat(validated_at)
        elif validated_at is None:
            validated_at = datetime.now()
            
        return cls(
            pattern_hash=data.get("pattern_hash", ""),
            tables=data.get("tables", []),
            join_columns=[tuple(pair) for pair in data.get("join_columns", [])],
            filter_columns=data.get("filter_columns", []),
            estimated_cost=float(data.get("estimated_cost", 0)),
            estimated_time_ms=float(data.get("estimated_time_ms", 0)),
            estimated_rows=int(data.get("estimated_rows", 0)),
            actual_execution_time_ms=data.get("actual_execution_time_ms"),
            validated_at=validated_at,
            validation_count=int(data.get("validation_count", 1)),
            max_cost_threshold=float(data.get("max_cost_threshold", 1000.0)),
            is_safe=data.get("is_safe", True),
            sample_sql=data.get("sample_sql", ""),
        )


class QueryPatternExtractor:
    """SQL에서 쿼리 패턴을 추출하는 유틸리티."""

    # 테이블 패턴: FROM, JOIN 뒤의 테이블명
    TABLE_PATTERN = re.compile(
        r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)?)',
        re.IGNORECASE
    )
    
    # 조인 조건 패턴: ON 절
    JOIN_CONDITION_PATTERN = re.compile(
        r'\bON\s+([a-zA-Z_][a-zA-Z0-9_.]+)\s*=\s*([a-zA-Z_][a-zA-Z0-9_.]+)',
        re.IGNORECASE
    )
    
    # WHERE 절 컬럼 패턴
    WHERE_COLUMN_PATTERN = re.compile(
        r'\bWHERE\b(.+?)(?:\bGROUP\s+BY\b|\bORDER\s+BY\b|\bLIMIT\b|\bHAVING\b|$)',
        re.IGNORECASE | re.DOTALL
    )
    
    COLUMN_IN_WHERE_PATTERN = re.compile(
        r'([a-zA-Z_][a-zA-Z0-9_.]+)\s*(?:=|!=|<>|>=|<=|>|<|LIKE|IN|BETWEEN|IS)',
        re.IGNORECASE
    )

    @classmethod
    def extract_pattern(cls, sql: str) -> QueryPattern:
        """SQL에서 쿼리 패턴 추출."""
        normalized_sql = " ".join(sql.split())  # 공백 정규화
        
        # 1. 테이블 추출
        tables = cls._extract_tables(normalized_sql)
        
        # 2. 조인 조건 추출
        join_columns = cls._extract_join_columns(normalized_sql)
        
        # 3. WHERE 절 컬럼 추출
        filter_columns = cls._extract_filter_columns(normalized_sql)
        
        return QueryPattern(
            tables=tables,
            join_columns=join_columns,
            filter_columns=filter_columns,
        )

    @classmethod
    def _extract_tables(cls, sql: str) -> List[str]:
        """SQL에서 테이블 목록 추출."""
        tables: Set[str] = set()
        for match in cls.TABLE_PATTERN.finditer(sql):
            table_name = match.group(1).lower()
            tables.add(table_name)
        return sorted(tables)

    @classmethod
    def _extract_join_columns(cls, sql: str) -> List[Tuple[str, str]]:
        """조인 조건에서 컬럼 쌍 추출."""
        joins: List[Tuple[str, str]] = []
        for match in cls.JOIN_CONDITION_PATTERN.finditer(sql):
            col1 = match.group(1).lower()
            col2 = match.group(2).lower()
            # 테이블.컬럼에서 컬럼명만 추출
            col1_name = col1.split('.')[-1] if '.' in col1 else col1
            col2_name = col2.split('.')[-1] if '.' in col2 else col2
            joins.append((col1_name, col2_name))
        return joins

    @classmethod
    def _extract_filter_columns(cls, sql: str) -> List[str]:
        """WHERE 절에서 필터링에 사용된 컬럼 추출."""
        filter_columns: Set[str] = set()
        
        where_match = cls.WHERE_COLUMN_PATTERN.search(sql)
        if where_match:
            where_clause = where_match.group(1)
            for match in cls.COLUMN_IN_WHERE_PATTERN.finditer(where_clause):
                col = match.group(1).lower()
                # 테이블.컬럼에서 컬럼명만 추출
                col_name = col.split('.')[-1] if '.' in col else col
                filter_columns.add(col_name)
        
        return sorted(filter_columns)


class ExplainCacheRepository:
    """Neo4j 기반 Explain 캐시 저장소."""

    COST_THRESHOLD_SAFE = 500.0  # 이 비용 이하면 '안전'으로 간주
    COST_THRESHOLD_RISKY = 5000.0  # 이 비용 이상이면 항상 재검증

    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    async def ensure_indexes(self) -> None:
        """필요한 인덱스 생성."""
        async with self.driver.session() as session:
            await session.run(
                "CREATE INDEX explain_cache_hash IF NOT EXISTS FOR (n:ExplainCache) ON (n.pattern_hash)"
            )
            await session.run(
                "CREATE INDEX explain_cache_validated IF NOT EXISTS FOR (n:ExplainCache) ON (n.validated_at)"
            )

    async def find_by_pattern(self, pattern: QueryPattern) -> Optional[ExplainCacheEntry]:
        """패턴 해시로 캐시 항목 조회."""
        query = """
        MATCH (n:ExplainCache {pattern_hash: $pattern_hash})
        RETURN n
        """
        async with self.driver.session() as session:
            result = await session.run(query, pattern_hash=pattern.pattern_hash)
            record = await result.single()
            if record:
                node = record["n"]
                return ExplainCacheEntry.from_dict(dict(node))
        return None

    async def find_by_tables(self, tables: List[str]) -> List[ExplainCacheEntry]:
        """테이블 목록으로 유사한 캐시 항목 조회."""
        query = """
        MATCH (n:ExplainCache)
        WHERE all(t IN $tables WHERE t IN n.tables)
        RETURN n
        ORDER BY n.validation_count DESC
        LIMIT 10
        """
        async with self.driver.session() as session:
            result = await session.run(query, tables=sorted(tables))
            entries = []
            async for record in result:
                node = record["n"]
                entries.append(ExplainCacheEntry.from_dict(dict(node)))
            return entries

    async def save(self, entry: ExplainCacheEntry) -> None:
        """캐시 항목 저장 또는 업데이트."""
        query = """
        MERGE (n:ExplainCache {pattern_hash: $pattern_hash})
        ON CREATE SET
            n.tables = $tables,
            n.join_columns = $join_columns,
            n.filter_columns = $filter_columns,
            n.estimated_cost = $estimated_cost,
            n.estimated_time_ms = $estimated_time_ms,
            n.estimated_rows = $estimated_rows,
            n.actual_execution_time_ms = $actual_execution_time_ms,
            n.validated_at = datetime(),
            n.validation_count = 1,
            n.max_cost_threshold = $max_cost_threshold,
            n.is_safe = $is_safe,
            n.sample_sql = $sample_sql
        ON MATCH SET
            n.estimated_cost = CASE 
                WHEN $estimated_cost < n.estimated_cost THEN $estimated_cost 
                ELSE n.estimated_cost 
            END,
            n.estimated_time_ms = CASE 
                WHEN $estimated_time_ms < n.estimated_time_ms THEN $estimated_time_ms 
                ELSE n.estimated_time_ms 
            END,
            n.actual_execution_time_ms = CASE 
                WHEN $actual_execution_time_ms IS NOT NULL THEN $actual_execution_time_ms 
                ELSE n.actual_execution_time_ms 
            END,
            n.validated_at = datetime(),
            n.validation_count = n.validation_count + 1,
            n.is_safe = $is_safe
        """
        async with self.driver.session() as session:
            await session.run(
                query,
                pattern_hash=entry.pattern_hash,
                tables=entry.tables,
                join_columns=[list(pair) for pair in entry.join_columns],
                filter_columns=entry.filter_columns,
                estimated_cost=entry.estimated_cost,
                estimated_time_ms=entry.estimated_time_ms,
                estimated_rows=entry.estimated_rows,
                actual_execution_time_ms=entry.actual_execution_time_ms,
                max_cost_threshold=entry.max_cost_threshold,
                is_safe=entry.is_safe,
                sample_sql=entry.sample_sql,
            )
        logger.info(
            "ExplainCache saved: pattern_hash=%s, tables=%s, cost=%.1f, is_safe=%s",
            entry.pattern_hash,
            entry.tables,
            entry.estimated_cost,
            entry.is_safe,
        )

    async def should_skip_explain(
        self,
        pattern: QueryPattern,
        max_cost_threshold: float = 500.0,
    ) -> Tuple[bool, Optional[ExplainCacheEntry], str]:
        """
        이 패턴에 대해 explain을 건너뛸 수 있는지 확인.
        
        Returns:
            (skip: bool, cached_entry: Optional[ExplainCacheEntry], reason: str)
        """
        cached = await self.find_by_pattern(pattern)
        
        if cached is None:
            return False, None, "no_cache"
        
        # 1. 안전하지 않은 것으로 표시된 경우 항상 재검증
        if not cached.is_safe:
            return False, cached, "marked_unsafe"
        
        # 2. 비용이 임계값을 초과하면 재검증
        if cached.estimated_cost > max_cost_threshold:
            return False, cached, f"cost_exceeds_threshold ({cached.estimated_cost:.0f} > {max_cost_threshold:.0f})"
        
        # 3. 검증 횟수가 충분하면 신뢰도 높음
        if cached.validation_count >= 3:
            return True, cached, f"validated_{cached.validation_count}_times"
        
        # 4. 비용이 매우 낮으면 건너뛰기
        if cached.estimated_cost < 100.0:
            return True, cached, f"very_low_cost ({cached.estimated_cost:.0f})"
        
        # 5. 그 외는 검증 횟수에 따라 결정
        if cached.validation_count >= 2 and cached.estimated_cost < max_cost_threshold:
            return True, cached, f"sufficient_validation (count={cached.validation_count}, cost={cached.estimated_cost:.0f})"
        
        return False, cached, f"needs_more_validation (count={cached.validation_count})"

    async def mark_unsafe(self, pattern_hash: str) -> None:
        """패턴을 안전하지 않음으로 표시."""
        query = """
        MATCH (n:ExplainCache {pattern_hash: $pattern_hash})
        SET n.is_safe = false
        """
        async with self.driver.session() as session:
            await session.run(query, pattern_hash=pattern_hash)

    async def delete_old_entries(self, days: int = 30) -> int:
        """오래된 캐시 항목 삭제."""
        query = """
        MATCH (n:ExplainCache)
        WHERE n.validated_at < datetime() - duration({days: $days})
        DELETE n
        RETURN count(*) as deleted
        """
        async with self.driver.session() as session:
            result = await session.run(query, days=days)
            record = await result.single()
            return record["deleted"] if record else 0

    async def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회."""
        query = """
        MATCH (n:ExplainCache)
        RETURN 
            count(*) as total_entries,
            sum(n.validation_count) as total_validations,
            avg(n.estimated_cost) as avg_cost,
            count(CASE WHEN n.is_safe THEN 1 END) as safe_count,
            count(CASE WHEN NOT n.is_safe THEN 1 END) as unsafe_count
        """
        async with self.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            if record:
                return {
                    "total_entries": record["total_entries"],
                    "total_validations": record["total_validations"],
                    "avg_cost": record["avg_cost"],
                    "safe_count": record["safe_count"],
                    "unsafe_count": record["unsafe_count"],
                }
            return {}
