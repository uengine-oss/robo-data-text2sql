"""데이터소스별 색인 상태 판정 — `indexed` / `indexing` / `stale` / `unindexed`.

spec 007 T2·T4 (FR-030~FR-034, FR-051).

**왜 이 모듈이 있는가.**
"임베딩이 없으면 조금 나쁜 답" 이 아니라 **"답을 만들지 않는다"** 가 사업 결정이다
(AD-4). 그러려면 거부할지 말지를 판정할 한 곳이 필요하고, 그 판정이 UI 에도 그대로
노출되어야 한다(FR-033) — 로그에만 남기면 사용자는 이유를 모른 채 "안 된다"만 본다.

**왜 컨트롤러 밖인가.** T1~T4 는 `app/react/controller.py` 를 한 줄도 건드리지 않는다.
거부는 라우터 진입부에서 일어나고, 컨트롤러는 자기가 거부됐다는 사실조차 모른다.

**데이터소스 경계.** constitution 원칙 II 대로 `(t.datasource = $ds OR t.db = $ds)` 를
항상 함께 쓴다. 한쪽만 쓰면 같은 `schema.table` 이름을 가진 다른 데이터소스의 노드가
섞여 든다 (실측: `manufacturing` 은 `datasource='manufacturing'` 인데 `db='meetingroom'`
이다 — 두 값이 다르다는 것이 이 OR 이 필요한 이유의 증거다).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.core import embedding_fingerprint as fp

# 상태값 — 이 네 개 말고 다른 값을 내지 않는다 (AD-4 표).
INDEXED = "indexed"
INDEXING = "indexing"
STALE = "stale"
UNINDEXED = "unindexed"

# 진행 중인 색인 작업 등록부. 프로세스 로컬이다 —
# 여러 워커로 뜨면 정확하지 않다는 것을 알고 쓴다(api_workers 기본 1).
# 이것이 없으면 "부분 색인"과 "색인 진행 중"을 구분할 수 없다.
_ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}


def mark_job_started(datasource: str, *, total: int = 0) -> None:
    _ACTIVE_JOBS[datasource] = {"total": int(total), "done": 0, "failed": 0, "running": True}


def mark_job_progress(datasource: str, *, done: int = 0, failed: int = 0, total: Optional[int] = None) -> None:
    job = _ACTIVE_JOBS.get(datasource)
    if not job:
        return
    job["done"] = int(done)
    job["failed"] = int(failed)
    if total is not None:
        job["total"] = int(total)


def mark_job_finished(datasource: str) -> None:
    job = _ACTIVE_JOBS.get(datasource)
    if job:
        job["running"] = False


def active_job(datasource: str) -> Optional[Dict[str, Any]]:
    job = _ACTIVE_JOBS.get(datasource)
    if job and job.get("running"):
        return dict(job)
    return None


_COUNT_CYPHER = f"""
MATCH (t:Table)
WHERE ($ds IS NULL OR t.datasource = $ds OR t.db = $ds)
WITH count(t) AS tables_total,
     sum(CASE WHEN t.text_to_sql_vector IS NULL THEN 0 ELSE 1 END) AS tables_with_vector,
     sum(CASE WHEN t.text_to_sql_vector IS NOT NULL AND {fp.fingerprint_match_predicate('t')} THEN 1 ELSE 0 END) AS tables_fresh
OPTIONAL MATCH (c:Column)
WHERE ($ds IS NULL OR c.datasource = $ds)
WITH tables_total, tables_with_vector, tables_fresh,
     count(c) AS columns_total,
     sum(CASE WHEN c.vector IS NULL THEN 0 ELSE 1 END) AS columns_with_vector,
     sum(CASE WHEN c.vector IS NOT NULL AND {fp.fingerprint_match_predicate('c')} THEN 1 ELSE 0 END) AS columns_fresh
RETURN tables_total, tables_with_vector, tables_fresh,
       columns_total, columns_with_vector, columns_fresh
"""


async def get_index_status(neo4j_session, datasource: Optional[str] = None) -> Dict[str, Any]:
    """한 데이터소스(또는 전체)의 색인 상태.

    반환은 UI 가 그대로 보여 줄 수 있는 형태다 — 상태만 주고 이유를 안 주면
    사용자는 무엇을 해야 하는지 모른다 (FR-032).
    """
    ds = (datasource or "").strip() or None
    fingerprint = fp.current_fingerprint()
    ready = fp.embedding_ready()

    params = {"ds": ds}
    params.update(fp.fingerprint_params(fingerprint))
    counts: Dict[str, Any] = {
        "tables_total": 0,
        "tables_with_vector": 0,
        "tables_fresh": 0,
        "columns_total": 0,
        "columns_with_vector": 0,
        "columns_fresh": 0,
    }
    try:
        res = await neo4j_session.run(_COUNT_CYPHER, **params)
        row = await res.single()
        if row:
            counts = {k: int(row[k] or 0) for k in counts}
    except Exception as exc:
        return {
            "datasource": ds,
            "status": UNINDEXED,
            "reason": f"색인 상태를 조회하지 못했습니다: {exc}",
            "next_action": "Neo4j 연결을 확인하세요.",
            "counts": counts,
            "fingerprint": fingerprint,
            "embedding": ready,
            "query_allowed": False,
        }

    dim_check = await fp.check_index_dimensions(neo4j_session)
    job = active_job(ds) if ds else None

    status, reason, next_action = _decide(counts, ready, dim_check, job)

    return {
        "datasource": ds,
        "status": status,
        "reason": reason,
        "next_action": next_action,
        "counts": counts,
        "progress": job,
        "fingerprint": fingerprint,
        "embedding": ready,
        "index_dimensions": dim_check,
        # 데이터소스 자연어 질의를 허용해도 되는가. 거부의 단일 판정 지점이다.
        "query_allowed": status == INDEXED,
        # `indexing` 중 다축 폴백은 **일시적 열화로 표시**한다 (FR-034).
        # `unindexed` 를 가리는 데 쓰지 않으므로 indexing 에서만 참이다.
        "degraded": status == INDEXING,
    }


_ONTOLOGY_STILL_WORKS = "온톨로지 설계와 그래프 조회는 계속 사용할 수 있습니다."


def _decide(counts, ready, dim_check, job):
    """상태 판정. 순서가 규칙이다 — 앞의 조건이 이긴다.

    차원 불일치를 가장 앞에 두는 이유: 그 상태에서는 벡터가 아무리 많아도 인덱스가
    답을 못 준다. "벡터가 있으니 indexed" 라고 말하면 조용히 틀린 답으로 이어진다.
    """
    if not dim_check.get("ok", True):
        return (
            STALE,
            dim_check.get("reason") or "벡터 인덱스 차원이 설정과 다릅니다.",
            "임베딩 설정을 되돌리거나 재색인(인덱스 재생성)을 실행하세요. " + _ONTOLOGY_STILL_WORKS,
        )

    if not ready.get("ready"):
        return (
            UNINDEXED,
            f"임베딩이 설정되지 않았습니다: {ready.get('reason')}",
            "「메타데이터 증강」을 실행해 임베딩 방식을 고르세요. " + _ONTOLOGY_STILL_WORKS,
        )

    if job:
        return (
            INDEXING,
            f"색인 진행 중 ({job.get('done', 0)}/{job.get('total', 0)}, 실패 {job.get('failed', 0)})",
            "색인이 끝나면 데이터소스 질의가 열립니다.",
        )

    if int(counts["tables_total"]) == 0:
        return (
            UNINDEXED,
            "이 데이터소스의 테이블 카탈로그가 그래프에 없습니다.",
            "데이터소스를 등록하고 메타데이터 추출을 먼저 실행하세요. " + _ONTOLOGY_STILL_WORKS,
        )

    if int(counts["tables_with_vector"]) == 0:
        return (
            UNINDEXED,
            "테이블 벡터가 하나도 없습니다.",
            "「메타데이터 증강」을 실행하면 추출과 함께 색인됩니다. " + _ONTOLOGY_STILL_WORKS,
        )

    if int(counts["tables_fresh"]) == 0:
        # 지문이 아예 없는 옛 벡터도 여기로 온다 — 무엇으로 만들었는지 모르므로
        # 신뢰할 근거가 없다. 모르는 것을 안다고 하지 않는다 (spec Edge Case).
        return (
            STALE,
            "기존 벡터가 현재 임베딩 설정과 일치하지 않습니다 "
            "(지문 불일치이거나, 지문이 없는 이전 버전 벡터입니다).",
            "재색인이 필요합니다. 낡은 벡터로 답하지 않습니다. " + _ONTOLOGY_STILL_WORKS,
        )

    if int(counts["tables_fresh"]) < int(counts["tables_total"]):
        return (
            INDEXING,
            f"일부만 색인되어 있습니다 ({counts['tables_fresh']}/{counts['tables_total']} 테이블).",
            "「메타데이터 증강」을 다시 실행하면 없는 것만 채웁니다.",
        )

    return (INDEXED, None, None)


def rejection_payload(status: Dict[str, Any]) -> Dict[str, Any]:
    """거부를 사용자에게 전달할 형태 (FR-031·FR-032).

    나쁜 답은 사용자가 검증할 수 없다. 거부는 검증할 필요가 없다 — 대신
    **무엇이 안 되고 무엇을 하면 되는지**를 반드시 함께 준다.
    """
    return {
        "error": "datasource_query_not_indexed",
        "index_status": status.get("status"),
        "datasource": status.get("datasource"),
        "reason": status.get("reason"),
        "next_action": status.get("next_action"),
        "counts": status.get("counts"),
        "message": (
            f"[{status.get('status')}] 이 데이터소스는 아직 질의할 수 없습니다.\n"
            f"이유: {status.get('reason')}\n"
            f"다음: {status.get('next_action')}"
        ),
    }
