"""
LLM Query Result Cache
- 동일한 질문에 대해 캐시된 SQL 결과를 반환하여 속도 향상
- TTL 기반 만료 지원
- 해시 기반 키 생성
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import OrderedDict


@dataclass
class CachedResult:
    """캐시된 쿼리 결과"""
    question: str
    final_sql: str
    validated_sql: Optional[str]
    execution_result: Optional[Dict[str, Any]]
    steps_summary: str  # 스텝 요약 (전체 스텝 저장 시 메모리 과다 사용 방지)
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """TTL 기준 만료 여부 확인"""
        return time.time() - self.created_at > ttl_seconds


class QueryCache:
    """LRU 기반 쿼리 결과 캐시"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Args:
            max_size: 최대 캐시 항목 수
            ttl_seconds: 캐시 TTL (초), 기본값 1시간
        """
        self._cache: OrderedDict[str, CachedResult] = OrderedDict()
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._total_hits = 0
        self._total_misses = 0
    
    def _generate_key(self, question: str, *, datasource: Optional[str] = None) -> str:
        """
        캐시 키 생성.

        MindsDB-only (Phase 1)에서는 datasource가 필수 계약이므로,
        datasource를 키에 포함해 서로 다른 데이터소스 간 캐시 충돌을 방지한다.
        """
        q = (question or "").strip().lower()
        ds = (datasource or "").strip().lower()
        normalized = f"ds={ds}||q={q}" if ds else f"q={q}"
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()
    
    def get(self, question: str, *, datasource: Optional[str] = None) -> Optional[CachedResult]:
        """캐시에서 결과 조회"""
        key = self._generate_key(question, datasource=datasource)
        
        if key not in self._cache:
            self._total_misses += 1
            return None
        
        result = self._cache[key]
        
        # TTL 확인
        if result.is_expired(self._ttl_seconds):
            del self._cache[key]
            self._total_misses += 1
            return None
        
        # LRU: 최근 사용 항목을 끝으로 이동
        self._cache.move_to_end(key)
        result.hit_count += 1
        self._total_hits += 1
        
        return result
    
    def put(
        self,
        question: str,
        *,
        datasource: Optional[str] = None,
        final_sql: str,
        validated_sql: Optional[str] = None,
        execution_result: Optional[Dict[str, Any]] = None,
        steps_summary: str = ""
    ) -> str:
        """결과를 캐시에 저장"""
        key = self._generate_key(question, datasource=datasource)
        
        # 용량 초과 시 가장 오래된 항목 제거
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        
        self._cache[key] = CachedResult(
            question=question,
            final_sql=final_sql,
            validated_sql=validated_sql,
            execution_result=execution_result,
            steps_summary=steps_summary
        )
        
        return key
    
    def invalidate(self, question: str, *, datasource: Optional[str] = None) -> bool:
        """특정 질문의 캐시 무효화"""
        key = self._generate_key(question, datasource=datasource)
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self) -> int:
        """전체 캐시 초기화"""
        count = len(self._cache)
        self._cache.clear()
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        hit_rate = (
            self._total_hits / (self._total_hits + self._total_misses)
            if (self._total_hits + self._total_misses) > 0
            else 0
        )
        
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": round(hit_rate, 4),
            "items": [
                {
                    "question": result.question[:50] + "..." if len(result.question) > 50 else result.question,
                    "hit_count": result.hit_count,
                    "age_seconds": int(time.time() - result.created_at)
                }
                for result in list(self._cache.values())[-10:]  # 최근 10개만
            ]
        }


# 전역 캐시 인스턴스
_query_cache: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    """전역 쿼리 캐시 인스턴스 반환"""
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache(max_size=200, ttl_seconds=3600)  # 1시간 TTL
    return _query_cache
