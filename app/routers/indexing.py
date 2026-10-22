"""색인 API — 추출 파이프라인이 부르는 색인 단계와, UI 가 읽는 상태.

spec 007 T3·T4 (FR-025, FR-027, FR-028, FR-030, FR-033).

**왜 text2sql 이 이 엔드포인트를 갖는가.**
FR-025 는 "추출이 색인 단계를 가져야 한다"고 하고, FR-023 은 "벡터를 만드는 지점은
앱 전역에서 한 곳"이라고 한다. 둘을 동시에 만족시키는 배치는 하나뿐이다 —
**색인 능력은 벡터 인덱스를 소유한 곳(text2sql)에 두고, 추출 파이프라인(data-fabric)이
그것을 자기 단계로 호출한다.** data-fabric 안에 임베딩 코드를 새로 심으면 모델이 두 벌이
되고, 그것이 AD-5 가 구조적으로 막으려던 바로 그 불일치다.

`POST /text2sql/direct-sql` 과 `GET /text2sql/meta/*` 는 건드리지 않는다 (AD-7).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core import index_status as istat
from app.core.datasource_indexer import index_datasource
from app.deps import get_neo4j_session

router = APIRouter(prefix="/index", tags=["indexing"])


class IndexDatasourceRequest(BaseModel):
    datasource: Optional[str] = Field(
        default=None,
        description="색인 범위. 비우면 전체(운영에서는 권장하지 않는다 — 원칙 II).",
    )
    # 기본이 True 인 이유: 지울 것이 없을 때만 자동으로 도는 것이 규칙이다 (FR-052).
    only_missing: bool = True
    # 지문이 다른 기존 벡터를 **덮어쓴다.** 되돌릴 수 없으므로 기본은 꺼짐 —
    # 확인 절차(대상 수·시간·비용·되돌릴 수 없음)는 T5 가 붙인다 (FR-053).
    # 추출 파이프라인은 이 값을 켜지 않는다.
    include_stale: bool = False
    include_columns: bool = True
    # 기본이 False 인 이유가 이 스펙의 핵심이다 — 색인이 LLM 키에 묶여 있던 것이
    # "키 없이 설치"를 색인 단계에서 다시 깨뜨리던 결함이다 (미결정 12번).
    with_profile: bool = False
    max_tables: int = 0
    max_columns: int = 0


@router.post("/datasource")
async def index_datasource_endpoint(
    request: IndexDatasourceRequest,
    session=Depends(get_neo4j_session),
):
    """데이터소스의 테이블·컬럼 벡터를 만든다. 추출 파이프라인이 부르는 자리다."""
    out = await index_datasource(
        session,
        datasource=request.datasource,
        only_missing=request.only_missing,
        include_stale=request.include_stale,
        include_columns=request.include_columns,
        with_profile=request.with_profile,
        max_tables=request.max_tables,
        max_columns=request.max_columns,
    )
    status = await istat.get_index_status(session, request.datasource)
    out["index_status"] = status
    return out


@router.get("/status")
async def index_status_endpoint(
    datasource: Optional[str] = None,
    session=Depends(get_neo4j_session),
):
    """색인 상태 (FR-033).

    UI 가 "왜 안 되는지"를 표시할 수 있어야 한다 — 로그에만 남기면 사용자는 이유를
    모른 채 "안 된다"만 본다.
    """
    return await istat.get_index_status(session, datasource)


@router.get("/embedding-config")
async def embedding_config_endpoint():
    """지금 임베딩 설정이 무엇이고 쓸 수 있는 상태인지 (T4 설정 화면이 읽는다).

    자격증명은 **반환하지 않는다** (constitution 원칙 V). 설정됨/안 됨만 말한다.
    """
    from app.core import embedding_fingerprint as fp

    ready = fp.embedding_ready()
    fingerprint = fp.current_fingerprint()
    return {
        "provider": fingerprint[fp.PROP_PROVIDER],
        "model": fingerprint[fp.PROP_MODEL],
        "dimension": fingerprint[fp.PROP_DIMENSION],
        "base_url": ready.get("base_url"),
        "ready": ready.get("ready"),
        "reason": ready.get("reason"),
    }
