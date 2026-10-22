"""spec 007 T1~T4 안전망.

**왜 이 파일이 있는가.**
`controller.py` 에 테스트가 0개였다는 사실 자체가, 이 저장소에서 "테스트 통과" 가
아무것도 보증하지 못했다는 증거다(spec CR-12). T1~T4 는 컨트롤러를 건드리지 않지만,
**바꾼 것이 무엇을 깨뜨렸는지 판정할 기준**은 있어야 한다. 여기서 고정하는 것은 넷이다.

1. 제공자 심이 실제로 `base_url` 을 받고 openai 전용 제약이 사라졌는가 (T1)
2. 지문 대조가 3값 논리에 걸리지 않는가 (T2) — 이것으로 실제 버그를 하나 잡았다
3. 상태 판정이 AD-4 표대로 갈리는가 (T2·T4)
4. 색인 대상 선정이 **기본값에서 파괴적이지 않은가** (T3 / FR-052 vs FR-053)

실행: `PYTHONPATH=. uv run pytest app/tests/test_spec007_embedding_indexing.py -v`
(외부 Neo4j·LLM·임베딩 서버가 필요 없다 — 전부 순수 함수 수준에서 고정한다.)
"""

import os

import pytest

os.environ.setdefault("TARGET_DB_PASSWORD", "x")

from app.config import settings  # noqa: E402
from app.core import embedding_fingerprint as fp  # noqa: E402
from app.core import index_status as istat  # noqa: E402
from app.core import datasource_indexer as di  # noqa: E402
from app.core.embedding import EmbeddingClient  # noqa: E402
from app.core.llm_factory import (  # noqa: E402
    EmbeddingProviderUnavailable,
    normalize_embedding_provider,
    resolve_embedding_endpoint,
)


@pytest.fixture
def restore_settings():
    keep = {
        k: getattr(settings, k)
        for k in (
            "embedding_provider",
            "embedding_model",
            "embedding_provider_url",
            "embedding_api_key",
            "embedding_dimension",
            "openai_api_key",
            "openai_compatible_api_key",
            "llm_provider_url",
        )
    }
    yield
    for k, v in keep.items():
        setattr(settings, k, v)


# ---------------------------------------------------------------------------
# T1 — 제공자 심 (FR-020 ~ FR-022, SC-006h)
# ---------------------------------------------------------------------------

def test_provider_aliases_normalize():
    # openai_compatible 은 LLM 쪽 표기다. 두 표기가 다른 지문이 되면 멀쩡한 벡터가
    # stale 로 오판된다 — 그래서 흡수한다.
    assert normalize_embedding_provider("") == "openai"
    assert normalize_embedding_provider("OpenAI") == "openai"
    assert normalize_embedding_provider("llm_endpoint") == "llm_endpoint"
    assert normalize_embedding_provider("openai_compatible") == "llm_endpoint"
    assert normalize_embedding_provider("local") == "local"
    with pytest.raises(ValueError):
        normalize_embedding_provider("nonsense")


def test_llm_endpoint_resolves_base_url_without_key(restore_settings):
    """로컬 서버는 키를 검사하지 않는다. 키를 강제하면 이 트랙의 목적이 깨진다."""
    settings.embedding_provider = "llm_endpoint"
    settings.embedding_provider_url = "http://127.0.0.1:11434/v1"
    settings.embedding_api_key = ""
    settings.openai_api_key = ""
    settings.openai_compatible_api_key = ""
    ep = resolve_embedding_endpoint()
    assert ep["base_url"] == "http://127.0.0.1:11434/v1"
    assert ep["api_key"]  # openai SDK 가 빈 문자열을 거부하므로 placeholder 가 들어간다


def test_llm_endpoint_falls_back_to_llm_provider_url(restore_settings):
    """1순위는 '사용자가 이미 붙인 그 엔드포인트 재사용' 이다 (AD-5)."""
    settings.embedding_provider = "llm_endpoint"
    settings.embedding_provider_url = ""
    settings.llm_provider_url = "http://localhost:1234/v1"
    assert resolve_embedding_endpoint()["base_url"] == "http://localhost:1234/v1"


def test_llm_endpoint_without_any_url_is_reported_as_config_problem(restore_settings):
    settings.embedding_provider = "llm_endpoint"
    settings.embedding_provider_url = ""
    settings.llm_provider_url = ""
    with pytest.raises(EmbeddingProviderUnavailable):
        resolve_embedding_endpoint()


def test_openai_still_requires_key(restore_settings):
    """되돌리기 확인: 기본 제공자의 동작은 이전과 같아야 한다."""
    settings.embedding_provider = "openai"
    settings.embedding_api_key = ""
    settings.openai_api_key = ""
    with pytest.raises(EmbeddingProviderUnavailable):
        resolve_embedding_endpoint()


def test_local_is_explicitly_not_ready_not_silently_broken(restore_settings):
    settings.embedding_provider = "local"
    with pytest.raises(EmbeddingProviderUnavailable) as e:
        resolve_embedding_endpoint()
    assert "준비 중" in str(e.value)


def test_embedding_client_no_longer_raises_for_non_openai(restore_settings):
    """SC-006h: 이 두 줄이 '로컬 모델' 선택지 전체를 막고 있었다."""
    settings.embedding_provider = "llm_endpoint"
    c = EmbeddingClient(object(), provider="llm_endpoint", model="qwen3-embedding")
    assert c.provider == "llm_endpoint"
    assert c.model == "qwen3-embedding"


# ---------------------------------------------------------------------------
# T2 — 지문 (FR-050, FR-051)
# ---------------------------------------------------------------------------

def test_fingerprint_predicate_is_null_safe():
    """3값 논리 회귀 방지.

    `coalesce(..., false)` 가 빠지면 `NOT (벡터있음 AND 지문일치)` 가 `NOT NULL` = NULL 이
    되어 **지문 없는 옛 벡터가 재색인 대상에서 조용히 빠진다.** 실제로 이 버그로
    테이블 13개가 대상 0개로 잡혔다.
    """
    pred = fp.fingerprint_match_predicate("t")
    assert pred.startswith("coalesce(")
    assert pred.rstrip().endswith(", false)")


def test_fingerprint_props_are_shared_constants():
    frag = fp.fingerprint_set_fragment("t")
    for prop in (fp.PROP_PROVIDER, fp.PROP_MODEL, fp.PROP_DIMENSION, fp.PROP_INDEXED_AT):
        assert f"t.{prop}" in frag
    params = fp.fingerprint_params({fp.PROP_PROVIDER: "p", fp.PROP_MODEL: "m", fp.PROP_DIMENSION: 7})
    assert params == {"fp_provider": "p", "fp_model": "m", "fp_dimension": 7}


def test_client_fingerprint_prefers_observed_dimension(restore_settings):
    """설정값은 설정이지 데이터에 붙은 사실이 아니다 (CR-5c)."""
    settings.embedding_dimension = 1536
    c = EmbeddingClient(object(), provider="llm_endpoint", model="m")
    assert c.fingerprint(dimension=4096)[fp.PROP_DIMENSION] == 4096
    assert c.fingerprint()[fp.PROP_DIMENSION] == 1536


# ---------------------------------------------------------------------------
# T2·T4 — 상태 판정 (AD-4 표 / FR-030 ~ FR-032)
# ---------------------------------------------------------------------------

READY = {"ready": True, "reason": None}
DIM_OK = {"ok": True}
COUNTS_NONE = {"tables_total": 0, "tables_with_vector": 0, "tables_fresh": 0}
COUNTS_ALL = {"tables_total": 5, "tables_with_vector": 5, "tables_fresh": 5}
COUNTS_STALE = {"tables_total": 5, "tables_with_vector": 5, "tables_fresh": 0}
COUNTS_PARTIAL = {"tables_total": 5, "tables_with_vector": 5, "tables_fresh": 2}


def test_dimension_mismatch_wins_over_everything():
    """벡터가 아무리 많아도 인덱스가 답을 못 주면 indexed 라고 말하면 안 된다."""
    status, reason, nxt = istat._decide(COUNTS_ALL, READY, {"ok": False, "reason": "차원"}, None)
    assert status == istat.STALE


def test_embedding_unconfigured_is_unindexed():
    status, reason, nxt = istat._decide(
        COUNTS_ALL, {"ready": False, "reason": "키 없음"}, DIM_OK, None
    )
    assert status == istat.UNINDEXED
    assert "온톨로지 설계" in nxt  # FR-032: 무엇이 여전히 되는지 함께 말한다


def test_status_table():
    assert istat._decide(COUNTS_NONE, READY, DIM_OK, None)[0] == istat.UNINDEXED
    assert istat._decide(COUNTS_STALE, READY, DIM_OK, None)[0] == istat.STALE
    assert istat._decide(COUNTS_PARTIAL, READY, DIM_OK, None)[0] == istat.INDEXING
    assert istat._decide(COUNTS_ALL, READY, DIM_OK, None)[0] == istat.INDEXED
    # 진행 중인 작업이 있으면 카운트와 무관하게 indexing 이다.
    running = {"done": 1, "total": 5, "failed": 0}
    assert istat._decide(COUNTS_ALL, READY, DIM_OK, running)[0] == istat.INDEXING


def test_rejection_payload_carries_reason_and_next_action():
    """나쁜 답 대신 명시적 거부 — 다만 '왜'와 '무엇을'이 없으면 거부도 나쁘다."""
    status = {
        "status": istat.UNINDEXED,
        "datasource": "sales",
        "reason": "테이블 벡터가 하나도 없습니다.",
        "next_action": "「메타데이터 증강」을 실행하세요.",
        "counts": COUNTS_NONE,
    }
    p = istat.rejection_payload(status)
    assert p["error"] == "datasource_query_not_indexed"
    assert "테이블 벡터가 하나도 없습니다." in p["message"]
    assert "「메타데이터 증강」을 실행하세요." in p["message"]


# ---------------------------------------------------------------------------
# T3 — 색인 (FR-052 vs FR-053, FR-055, 미결정 12번)
# ---------------------------------------------------------------------------

def test_default_target_selection_is_not_destructive():
    """기본값은 '벡터가 아예 없는 것' 만 대상으로 한다.

    지문이 다른 벡터를 덮어쓰는 것은 되돌릴 수 없으므로 확인이 필요하고(FR-053),
    그 확인 절차는 T5 다. 이 스펙 범위에서 자동으로 켜지면 안 된다.
    """
    default_q = di._fetch_tables_cypher(only_missing=True)
    assert "AND t.text_to_sql_vector IS NULL" in default_q
    # 지문 대조 술어가 **없어야** 한다 = 지문이 다른 기존 벡터를 대상으로 잡지 않는다.
    # (RETURN 절의 coalesce 는 무관하므로 술어 자체를 찾는다.)
    assert fp.fingerprint_match_predicate("t") not in default_q

    stale_q = di._fetch_tables_cypher(only_missing=True, include_stale=True)
    assert fp.fingerprint_match_predicate("t") in stale_q


def test_datasource_scope_always_uses_both_keys():
    """constitution 원칙 II — 한쪽만 보면 남의 데이터소스를 건드리거나 통째로 놓친다.

    실측 근거: manufacturing 은 datasource='manufacturing' 인데 db='meetingroom' 이다.
    """
    q = di._fetch_tables_cypher(only_missing=True)
    assert "t.datasource = $ds OR t.db = $ds" in q
    assert "c.datasource = $ds" in di._fetch_columns_cypher(only_missing=True)


def test_indexer_never_deletes():
    for q in (di._WRITE_TABLE_CYPHER, di._WRITE_COLUMN_CYPHER):
        upper = q.upper()
        assert "DELETE" not in upper
        assert "REMOVE" not in upper


def test_embedding_text_is_built_without_llm():
    """미결정 12번의 답 — 프로필(LLM) 없이 카탈로그만으로 임베딩 텍스트를 만든다."""
    text = di.build_table_embedding_text(
        {
            "schema": "manufacturing",
            "name": "mfg_daily_equipment",
            "description": "설비 일일 실적",
            "analyzed_description": "",
            "columns": [{"name": "utilization_rate", "type": "numeric", "description": "가동률"}],
        }
    )
    assert "manufacturing.mfg_daily_equipment" in text
    assert "설비 일일 실적" in text
    assert "utilization_rate(numeric)" in text

    col = di.build_column_embedding_text(
        {"schema": "sales", "table": "orders", "name": "amount", "dtype": "decimal", "description": "금액"}
    )
    assert "sales.orders.amount" in col
    assert "decimal" in col


def test_table_write_also_fills_legacy_index_property():
    """레거시 `table_vec_index` 는 '사람이 설명을 손으로 고칠 때만' 채워져 영구 공백이었다."""
    assert "t.vector = item.vector" in di._WRITE_TABLE_CYPHER
    assert "t.text_to_sql_vector = item.vector" in di._WRITE_TABLE_CYPHER
