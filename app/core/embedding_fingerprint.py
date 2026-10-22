"""벡터 지문 — 무엇으로 만든 벡터인지 노드마다 기록하고 대조한다.

spec 007 T2 (FR-030, FR-050, FR-051) / AD-5b 권고 1.

**왜 필요한가.**
지금까지 어떤 벡터 옆에도 *어떤 모델·어떤 차원으로 만들었는지*가 없었다
(`text2sql_table_vectorizer.py` 가 쓰는 것은 profile·embedding_text·updated_at 뿐).
`EMBEDDING_DIMENSION` 은 **설정값**이지 데이터에 붙은 사실이 아니다. 그래서:

- 모델을 바꾸면 기존 벡터가 무효가 되는데 그 사실을 감지할 근거가 그래프에 없다.
- 차원이 같고 모델만 다르면 **인덱스 차원 검사로도 안 잡힌다.** 코사인이 조용히
  무의미한 값을 낸다.
- 재색인이 도중에 죽으면 옛 벡터와 새 벡터를 구분할 수 없다.

**왜 노드별인가.** 전역에만 두면 재색인이 중간에 죽었을 때 "전부 다시" 밖에 못 하고,
수천 테이블에서 그것은 사실상 복구 불가다 (AD-5b 권고 1).

**이 모듈은 아무것도 지우지 않는다.** 삭제는 T5(재색인)의 일이다. 지문이 없는 기존
벡터는 `stale` 로 **간주만** 한다 — 모르는 것을 안다고 하지 않는다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import settings

# 노드에 붙는 속성 이름. 쓰는 쪽(색인)과 읽는 쪽(상태 판정·검색 필터)이 이 상수를
# 공유한다 — constitution 원칙 III 이 경고하는 표기 divergence 를 새로 만들지 않기 위해
# 문자열을 두 번 쓰지 않는다.
PROP_PROVIDER = "embedding_provider"
PROP_MODEL = "embedding_model"
PROP_DIMENSION = "embedding_dimension"
PROP_INDEXED_AT = "embedding_indexed_at"

FINGERPRINT_PROPS = (PROP_PROVIDER, PROP_MODEL, PROP_DIMENSION, PROP_INDEXED_AT)

# text2sql 이 실제로 검색에 쓰는 벡터 인덱스와 그 차원 원천.
# (:Query) 인덱스는 질의 후처리가 만드는 것이라 데이터소스 색인 상태와 무관하므로 뺐다.
CATALOG_VECTOR_INDEXES = (
    "text_to_sql_table_vec_index",
    "table_vec_index",
    "column_vec_index",
)


def current_fingerprint() -> Dict[str, Any]:
    """지금 설정이 만들어 낼 벡터의 지문.

    provider 정규화는 llm_factory 를 재사용한다 — 두 곳에서 각자 정규화하면
    `openai_compatible` 과 `llm_endpoint` 가 다른 지문이 되어 멀쩡한 벡터가
    `stale` 로 오판된다.
    """
    from app.core.llm_factory import normalize_embedding_provider

    try:
        provider = normalize_embedding_provider(getattr(settings, "embedding_provider", ""))
    except ValueError:
        provider = (getattr(settings, "embedding_provider", "") or "").strip().lower()
    return {
        PROP_PROVIDER: provider,
        PROP_MODEL: (getattr(settings, "embedding_model", "") or "").strip(),
        PROP_DIMENSION: int(getattr(settings, "embedding_dimension", 0) or 0),
    }


def embedding_ready() -> Dict[str, Any]:
    """임베딩을 지금 만들 수 있는가 — 왕복 없이 설정만으로 판정한다.

    상태 API 가 매 호출마다 임베딩 서버를 두드리면 UI 가 느려지고, 서버가 죽어 있을 때
    상태 조회 자체가 막힌다. 그래서 여기서는 **설정 해석까지만** 한다.
    """
    from app.core.llm_factory import EmbeddingProviderUnavailable, resolve_embedding_endpoint

    try:
        ep = resolve_embedding_endpoint()
    except EmbeddingProviderUnavailable as exc:
        return {"ready": False, "reason": str(exc), "provider": (getattr(settings, "embedding_provider", "") or "")}
    except ValueError as exc:
        return {"ready": False, "reason": str(exc), "provider": (getattr(settings, "embedding_provider", "") or "")}
    return {"ready": True, "reason": None, "provider": ep["provider"], "base_url": ep["base_url"]}


def fingerprint_set_fragment(alias: str) -> str:
    """색인이 벡터를 쓸 때 함께 붙일 SET 조각.

    호출부가 손으로 속성명을 적지 않게 한다 (쓰는 쪽/읽는 쪽 divergence 방지).
    """
    return (
        f"{alias}.{PROP_PROVIDER} = $fp_provider, "
        f"{alias}.{PROP_MODEL} = $fp_model, "
        f"{alias}.{PROP_DIMENSION} = $fp_dimension, "
        f"{alias}.{PROP_INDEXED_AT} = datetime()"
    )


def fingerprint_params(fp: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    f = fp or current_fingerprint()
    return {
        "fp_provider": f[PROP_PROVIDER],
        "fp_model": f[PROP_MODEL],
        "fp_dimension": int(f[PROP_DIMENSION]),
    }


def fingerprint_match_predicate(alias: str) -> str:
    """검색·상태 판정에서 "지금 설정으로 만든 벡터인가"를 묻는 술어.

    지문이 **없는** 노드는 여기서 탈락한다 — 그것이 의도다
    (spec Edge Case: 지문 없는 기존 벡터는 `stale` 로 간주).

    **`coalesce(..., false)` 가 반드시 필요하다.** Cypher 는 3값 논리라서
    `t.embedding_model = $x` 는 속성이 없으면 `false` 가 아니라 **NULL** 이다.
    그러면 `NOT (벡터있음 AND 지문일치)` 가 `NOT NULL` = NULL 이 되어 WHERE 절에서
    **탈락한다** — 즉 "지문 없는 옛 벡터"가 재색인 대상에서 조용히 빠진다.
    실제로 이 버그를 실행 중에 만났다(테이블 13개가 대상 0개로 잡혔다).
    """
    return (
        f"coalesce({alias}.{PROP_PROVIDER} = $fp_provider AND "
        f"{alias}.{PROP_MODEL} = $fp_model AND "
        f"{alias}.{PROP_DIMENSION} = $fp_dimension, false)"
    )


async def check_index_dimensions(neo4j_session) -> Dict[str, Any]:
    """인덱스 차원과 설정 차원을 대조한다 (FR-051).

    추측해서 진행하지 않는다. 차원이 다르면 벡터를 써도 Neo4j 가 거부하거나
    (더 나쁘게) 인덱스에 안 올라가 조용히 검색에서 빠진다.
    """
    want = int(getattr(settings, "embedding_dimension", 0) or 0)
    rows: List[Dict[str, Any]] = []
    try:
        res = await neo4j_session.run(
            "SHOW INDEXES YIELD name, type, options "
            "WHERE type = 'VECTOR' RETURN name AS name, options AS options"
        )
        rows = [r.data() async for r in res]
    except Exception as exc:  # pragma: no cover - Neo4j 버전차 방어
        return {"ok": True, "checked": False, "reason": f"SHOW INDEXES unavailable: {exc}", "indexes": []}

    out: List[Dict[str, Any]] = []
    mismatched: List[str] = []
    for r in rows:
        name = str(r.get("name") or "")
        if name not in CATALOG_VECTOR_INDEXES:
            continue
        cfg = ((r.get("options") or {}).get("indexConfig") or {})
        dim = cfg.get("vector.dimensions")
        dim_i = int(dim) if dim is not None else None
        out.append({"name": name, "dimension": dim_i})
        if dim_i is not None and want and dim_i != want:
            mismatched.append(name)

    reason = None
    if mismatched:
        detail = ", ".join(
            f"{i['name']}={i['dimension']}" for i in out if i["name"] in mismatched
        )
        reason = (
            f"벡터 인덱스 차원({detail})이 설정 차원({want})과 다릅니다. "
            "추측해서 진행하지 않습니다 — 인덱스를 다시 만들어야 합니다 (재색인 = spec 007 T5)."
        )

    return {
        "ok": not mismatched,
        "checked": True,
        "expected_dimension": want,
        "indexes": out,
        "mismatched": mismatched,
        "reason": reason,
    }
