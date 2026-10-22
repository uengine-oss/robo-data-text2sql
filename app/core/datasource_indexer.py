"""데이터소스 색인 — 테이블·컬럼 벡터를 지문과 함께 만든다.

spec 007 T3 (FR-025~FR-028, FR-050, FR-052, FR-055).

**무엇이 달라지는가.**
지금까지 벡터는 **text2sql 기동의 부수효과**였다(`app/main.py:72` →
`ensure_text_to_sql_table_vectors`). 그 배치가 네 증상을 동시에 낳았다 —
키 없이 기동 실패 / 긴 콜드스타트 / 새 데이터소스에 벡터 없음 / 레거시 인덱스 영구 공백.
이 모듈은 같은 일을 **요청으로** 할 수 있게 만든다. 그래야 추출 파이프라인이 색인을
자기 단계로 가질 수 있다.

**미결정 12번에 대한 답 — 프로필 없이도 색인한다.**
기존 벡터화는 테이블마다 `table_profile` **LLM 호출**을 먼저 하고 그 결과(`embedding_text`)를
임베딩했다(`text2sql_table_vectorizer.py:240-256`). 그러면 "키 없이"가 색인 단계에서 다시
깨진다. 그래서 이 모듈은 임베딩 텍스트를 **카탈로그 메타데이터만으로** 결정론적으로 만든다
(`EmbeddingClient.format_table_text` / `format_column_text` 재사용 — 이미 있던 함수인데
아무도 안 쓰고 있었다). LLM 프로필은 `with_profile=True` 로 **덧붙일 수 있는 선택지**로
강등한다. 품질 비교는 하지 않았다 — 그 사실을 숨기지 않는다.

**기본 경로는 파괴적이지 않다.** 기본값은 **벡터가 아예 없는 노드**에만 쓴다
(지울 것이 없으므로 확인이 필요 없다 — FR-052). 지문이 **다른** 벡터를 덮어쓰는 것은
되돌릴 수 없으므로 `include_stale=True` 를 명시해야만 한다. 그 확인 절차는 T5 의 일이고
(FR-053), 이 스펙 범위에서는 **자동으로 켜지지 않는다**는 것만 보장한다.
어떤 경로로도 노드를 삭제하지 않고 `REMOVE` 도 하지 않는다 (constitution 원칙 I·IV).
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

from app.config import settings
from app.core import embedding_fingerprint as fpmod
from app.core import index_status as istat
from app.core.embedding import EmbeddingClient
from app.core.llm_factory import create_embedding_client
from app.smart_logger import SmartLogger

# 데이터소스 경계는 항상 이 술어로 좁힌다 (constitution 원칙 II / FR-055).
# 실측 근거: `manufacturing` 은 datasource='manufacturing' 인데 db='meetingroom' 이다.
# 한쪽만 보면 테이블을 통째로 놓치거나 남의 데이터소스를 건드린다.
_TABLE_SCOPE = "($ds IS NULL OR t.datasource = $ds OR t.db = $ds)"
_COLUMN_SCOPE = "($ds IS NULL OR c.datasource = $ds)"


def _fetch_tables_cypher(*, only_missing: bool, include_stale: bool = False) -> str:
    # only_missing (기본): **벡터가 아예 없는** 노드만. 지울 것이 없으므로 파괴적이지 않다 (FR-052).
    # include_stale: 지문이 다른 노드까지 **덮어쓴다.** 옛 벡터는 복구 경로가 없으므로
    #                이것은 확인을 받아야 하는 파괴적 작업이다 (FR-053 = T5). 기본 꺼짐.
    if only_missing and include_stale:
        freshness = (
            f"AND NOT (t.text_to_sql_vector IS NOT NULL AND {fpmod.fingerprint_match_predicate('t')})"
        )
    elif only_missing:
        freshness = "AND t.text_to_sql_vector IS NULL"
    else:
        freshness = ""
    # `text_to_sql_is_valid` 로 거르지 **않는다.**
    #
    # 그 플래그는 MindsDB 프로브 결과이고, 이 환경 실측에서 `insurance`·`sales` 의
    # 전 테이블·컬럼이 false 였다. 그것을 색인 대상에서 빼면 그 데이터소스는
    # 추출을 몇 번 돌려도 영원히 `unindexed` 에 머물고, 상태 안내는 "메타데이터 증강을
    # 실행하세요" 를 반복한다 — 사용자가 빠져나올 수 없는 고리가 된다.
    # 카탈로그에 있는 것이 존재하는 것이다(constitution 원칙 I). 검색 시점의
    # `COALESCE(flag,true)=true` 필터는 그대로 살아 있으므로 잘못된 테이블이
    # 답에 섞이지도 않는다.
    return f"""
    MATCH (t:Table)
    WHERE {_TABLE_SCOPE}
      {freshness}
    OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
    WITH t, collect({{
        name: c.name,
        type: coalesce(c.type, c.dtype),
        description: c.description
    }}) AS cols
    RETURN elementId(t) AS tid,
           coalesce(t.datasource, '') AS datasource,
           coalesce(t.db, '') AS db,
           coalesce(t.schema, '') AS schema,
           coalesce(t.name, '') AS name,
           coalesce(t.description, '') AS description,
           coalesce(t.analyzed_description, '') AS analyzed_description,
           cols AS columns
    ORDER BY name
    """


def _fetch_columns_cypher(*, only_missing: bool, include_stale: bool = False) -> str:
    if only_missing and include_stale:
        freshness = f"AND NOT (c.vector IS NOT NULL AND {fpmod.fingerprint_match_predicate('c')})"
    elif only_missing:
        freshness = "AND c.vector IS NULL"
    else:
        freshness = ""
    return f"""
    MATCH (c:Column)
    WHERE {_COLUMN_SCOPE}
      {freshness}
    RETURN elementId(c) AS cid,
           coalesce(c.name, '') AS name,
           coalesce(c.table, '') AS table,
           coalesce(c.schema, '') AS schema,
           coalesce(c.type, c.dtype, '') AS dtype,
           coalesce(c.description, '') AS description
    ORDER BY table, name
    """


def build_table_embedding_text(row: Dict[str, Any]) -> str:
    """LLM 없이 테이블 임베딩 텍스트를 만든다 (미결정 12번의 답).

    이름·설명·컬럼명만 쓴다. 값 샘플도 프로필도 쓰지 않는다 — 그 둘은 각각 DB 접속과
    LLM 키를 요구하고, 색인이 그 둘에 묶여 있던 것이 지금 고치는 결함이기 때문이다.
    """
    schema = str(row.get("schema") or "")
    name = str(row.get("name") or "")
    full = f"{schema}.{name}" if schema else name
    desc_parts = [
        str(row.get("description") or "").strip(),
        str(row.get("analyzed_description") or "").strip(),
    ]
    description = " ".join([p for p in desc_parts if p])
    columns: List[str] = []
    for c in row.get("columns") or []:
        if not isinstance(c, dict):
            continue
        cname = str(c.get("name") or "").strip()
        if not cname:
            continue
        ctype = str(c.get("type") or "").strip()
        cdesc = str(c.get("description") or "").strip()
        piece = f"{cname}({ctype})" if ctype else cname
        if cdesc:
            piece = f"{piece} - {cdesc}"
        columns.append(piece)
    return EmbeddingClient.format_table_text(full, description, columns[:200])


def build_column_embedding_text(row: Dict[str, Any]) -> str:
    schema = str(row.get("schema") or "")
    table = str(row.get("table") or "")
    full_table = f"{schema}.{table}" if schema else table
    return EmbeddingClient.format_column_text(
        str(row.get("name") or ""),
        full_table,
        str(row.get("dtype") or ""),
        str(row.get("description") or ""),
    )


_WRITE_TABLE_CYPHER = f"""
UNWIND $items AS item
MATCH (t) WHERE elementId(t) = item.tid
SET t.text_to_sql_vector = item.vector,
    t.text_to_sql_embedding_text = item.text,
    t.text_to_sql_updated_at = datetime(),
    t.vector = item.vector,
    {fpmod.fingerprint_set_fragment('t')}
"""
# `t.vector` 도 함께 쓴다: 레거시 `table_vec_index` 가 "사람이 설명을 손으로 고칠 때만"
# 채워져 사실상 영구 공백이었다(spec CR-5 표). 같은 지문·같은 모델이므로 두 인덱스가
# 어긋날 수 없다. 미결정 14번(레거시 인덱스 폐기 여부)은 그대로 열어 둔다 —
# 여기서 채우는 것이 버리는 결정을 막지 않는다.

_WRITE_COLUMN_CYPHER = f"""
UNWIND $items AS item
MATCH (c) WHERE elementId(c) = item.cid
SET c.vector = item.vector,
    c.text_to_sql_embedding_text = item.text,
    {fpmod.fingerprint_set_fragment('c')}
"""


async def _rows(neo4j_session, cypher: str, **params) -> List[Dict[str, Any]]:
    res = await neo4j_session.run(cypher, **params)
    return [r.data() async for r in res]


async def index_datasource(
    neo4j_session,
    *,
    datasource: Optional[str],
    only_missing: bool = True,
    include_stale: bool = False,
    include_columns: bool = True,
    with_profile: bool = False,
    max_tables: int = 0,
    max_columns: int = 0,
) -> Dict[str, Any]:
    """한 데이터소스의 테이블·컬럼을 색인한다.

    기본값(`only_missing=True`, `include_stale=False`)은 **벡터가 아예 없는 노드만**
    채운다. 지울 것이 없으므로 파괴적이지 않다 (AD-5b 권고 2 = FR-052) — 그래서
    추출 파이프라인이 확인 없이 부를 수 있다.

    지문이 **다른** 벡터를 덮어쓰는 것은 되돌릴 수 없는 작업이라 `include_stale=True`
    를 명시해야 한다. 그 확인 절차(대상 수·시간·비용·되돌릴 수 없음)는 T5 의 몫이고,
    이 스펙 범위(T1~T4)에서는 **자동으로 켜지지 않는다**는 것만 보장한다 (FR-053).
    """
    started = time.perf_counter()
    ds = (datasource or "").strip() or None
    fingerprint = fpmod.current_fingerprint()

    # 차원이 인덱스와 다르면 시작하지 않는다. 진행해 봤자 인덱스에 안 올라가거나
    # Neo4j 가 거부한다 — 반쯤 쓴 벡터를 남기는 것이 가장 나쁘다 (FR-051).
    dim_check = await fpmod.check_index_dimensions(neo4j_session)
    if not dim_check.get("ok", True):
        return {
            "ok": False,
            "datasource": ds,
            "error": "index_dimension_mismatch",
            "reason": dim_check.get("reason"),
            "index_dimensions": dim_check,
        }

    try:
        embedder = create_embedding_client()
    except Exception as exc:
        return {
            "ok": False,
            "datasource": ds,
            "error": "embedding_unavailable",
            "reason": str(exc),
        }

    params = {"ds": ds}
    params.update(fpmod.fingerprint_params(fingerprint))

    tables = await _rows(
        neo4j_session,
        _fetch_tables_cypher(only_missing=only_missing, include_stale=include_stale),
        **params,
    )
    if max_tables > 0:
        tables = tables[:max_tables]
    columns: List[Dict[str, Any]] = []
    if include_columns:
        columns = await _rows(
            neo4j_session,
            _fetch_columns_cypher(only_missing=only_missing, include_stale=include_stale),
            **params,
        )
        if max_columns > 0:
            columns = columns[:max_columns]

    total = len(tables) + len(columns)
    if ds:
        istat.mark_job_started(ds, total=total)

    batch_size = max(1, int(getattr(settings, "text2sql_vector_embed_batch_size", 128) or 128))
    done = 0
    failed = 0
    dim_seen: Optional[int] = None

    async def _embed_and_write(rows: Sequence[Dict[str, Any]], *, id_key: str, text_fn, cypher: str) -> None:
        nonlocal done, failed, dim_seen
        payload_texts = [(r[id_key], text_fn(r)) for r in rows]
        payload_texts = [(i, t) for i, t in payload_texts if t.strip()]
        for i in range(0, len(payload_texts), batch_size):
            chunk = payload_texts[i : i + batch_size]
            try:
                vecs = await embedder.embed_batch([t[:8000] for _, t in chunk])
            except Exception as exc:
                failed += len(chunk)
                SmartLogger.log(
                    "WARNING",
                    "text2sql.datasource_indexer.embed_failed",
                    category="text2sql.datasource_indexer",
                    params={"datasource": ds, "n": len(chunk), "error": str(exc)},
                    max_inline_chars=0,
                )
                if ds:
                    istat.mark_job_progress(ds, done=done, failed=failed)
                continue
            items = []
            for (node_id, text), vec in zip(chunk, vecs):
                if not vec:
                    failed += 1
                    continue
                if dim_seen is None:
                    dim_seen = len(vec)
                items.append({id_key: node_id, "vector": list(vec), "text": text[:1800]})
            if not items:
                continue
            write_params = {"items": items}
            # 지문은 **실측 차원**으로 쓴다. 설정값을 쓰면 설정과 실제가 어긋났을 때
            # 그 거짓말이 그대로 데이터에 남는다 (CR-5c 가 경고하는 바로 그 상황).
            write_params.update(
                fpmod.fingerprint_params(embedder.fingerprint(dimension=dim_seen or fingerprint[fpmod.PROP_DIMENSION]))
            )
            res = await neo4j_session.run(cypher, **write_params)
            await res.consume()
            done += len(items)
            if ds:
                istat.mark_job_progress(ds, done=done, failed=failed)

    try:
        if with_profile:
            # LLM 프로필은 **선택**이다. 실패해도 결정론 텍스트로 색인이 끝나야 한다.
            await _attach_llm_profiles(neo4j_session, tables)
        await _embed_and_write(
            tables, id_key="tid", text_fn=_table_text_fn, cypher=_WRITE_TABLE_CYPHER
        )
        if include_columns:
            await _embed_and_write(
                columns, id_key="cid", text_fn=build_column_embedding_text, cypher=_WRITE_COLUMN_CYPHER
            )
    finally:
        if ds:
            istat.mark_job_finished(ds)

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    out = {
        "ok": failed == 0,
        "datasource": ds,
        "tables_targeted": len(tables),
        "columns_targeted": len(columns),
        "vectors_written": done,
        "failed": failed,
        "dimension_observed": dim_seen,
        "fingerprint": embedder.fingerprint(dimension=dim_seen) if dim_seen else fingerprint,
        "elapsed_ms": round(elapsed_ms, 1),
        "with_profile": bool(with_profile),
        "include_stale": bool(include_stale),
    }
    SmartLogger.log(
        "INFO" if out["ok"] else "WARNING",
        "text2sql.datasource_indexer.done",
        category="text2sql.datasource_indexer",
        params=out,
        max_inline_chars=0,
    )
    return out


def _table_text_fn(row: Dict[str, Any]) -> str:
    # LLM 프로필이 붙어 있으면 그것을 쓰고, 없으면 결정론 텍스트로 간다.
    prof_text = str(row.get("_profile_embedding_text") or "").strip()
    return prof_text or build_table_embedding_text(row)


async def _attach_llm_profiles(neo4j_session, tables: List[Dict[str, Any]]) -> None:
    """선택적 LLM 프로필 증강. 실패는 색인을 멈추지 않는다.

    이것이 옛 색인이 LLM 키에 묶여 있던 지점이다. 이제 **덧붙이는 것**이라
    키가 없으면 그냥 안 붙고 색인은 끝난다.
    """
    try:
        from app.react.generators.table_profile_generator import (
            TableProfileInput,
            get_table_profile_generator,
        )
    except Exception as exc:  # pragma: no cover
        SmartLogger.log(
            "WARNING",
            "text2sql.datasource_indexer.profile_import_failed",
            category="text2sql.datasource_indexer",
            params={"error": str(exc)},
        )
        return

    gen = get_table_profile_generator()
    conc = max(1, int(getattr(settings, "text2sql_vector_llm_concurrency", 30) or 30))
    sem = asyncio.Semaphore(conc)

    async def _one(row: Dict[str, Any]) -> None:
        try:
            async with sem:
                obj, _mode = await gen.generate(
                    item=TableProfileInput(
                        schema=str(row.get("schema") or ""),
                        name=str(row.get("name") or ""),
                        description=str(row.get("description") or ""),
                        analyzed_description=str(row.get("analyzed_description") or ""),
                        columns=list(row.get("columns") or []),
                        sample_rows=[],
                    ),
                    react_run_id=None,
                )
            if isinstance(obj, dict):
                text = str(obj.get("embedding_text") or "").strip()
                if text:
                    row["_profile_embedding_text"] = text
        except Exception:
            # 한 테이블의 프로필 실패가 전체 색인을 멈추지 않는다.
            SmartLogger.log(
                "WARNING",
                "text2sql.datasource_indexer.profile_failed",
                category="text2sql.datasource_indexer",
                params={"table": row.get("name"), "traceback": traceback.format_exc()[:2000]},
                max_inline_chars=0,
            )

    await asyncio.gather(*[_one(r) for r in tables])
