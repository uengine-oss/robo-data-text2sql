"""
Cache post-processing pipeline (runs in background workers).

Responsibilities:
- Build minimal steps summary (LLM) for storage.
- Extract ValueMapping candidates (LLM).
- Apply strong gate: DB existence checks before persisting ValueMapping.
- Upsert Query node using db+question deterministic id and overwrite policy.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import asyncpg

from app.config import settings
from app.deps import neo4j_conn, openai_client
from app.core.embedding import EmbeddingClient
from app.models.neo4j_history import Neo4jQueryRepository
from app.smart_logger import SmartLogger
from app.react.utils.log_sanitize import sanitize_for_log


"""
ValueMapping 정책 (Option C):
- stopword 기반 차단은 사용하지 않는다.
- 필요 시 스키마/테이블/컬럼별 allow/deny 정책으로 제어한다.
  (현재 기본 정책은 "모든 컬럼 허용"이며, DB 존재성/컬럼해결 게이트가 품질을 보장한다.)
"""

# Optional: block specific columns if they create noise.
# Keep empty by default.
VALUE_MAPPING_COLUMN_NAME_DENYLIST = set()

# Optional: allowlist/denylist by FQN regex.
# If allowlist is non-empty, only matching FQNs are allowed.
VALUE_MAPPING_COLUMN_FQN_ALLOWLIST_REGEX: List[str] = []
VALUE_MAPPING_COLUMN_FQN_DENYLIST_REGEX: List[str] = []


@dataclass
class ValueMappingCandidate:
    schema: str
    table: str
    column: str
    natural_value: str
    code_value: str
    confidence: float = 0.0
    evidence: str = ""


def _candidate_brief(c: ValueMappingCandidate) -> Dict[str, Any]:
    return {
        "schema": c.schema,
        "table": c.table,
        "column": c.column,
        "natural_value": c.natural_value,
        "code_value": c.code_value,
        "confidence": c.confidence,
        "evidence": (c.evidence or "")[:200],
    }


def _append_sample(bucket: List[Dict[str, Any]], cand: ValueMappingCandidate, *, limit: int = 5) -> None:
    if len(bucket) >= limit:
        return
    bucket.append(_candidate_brief(cand))


async def process_cache_postprocess_payload(payload: Dict[str, Any]) -> None:
    """
    Main entrypoint for worker.
    payload should include:
      - react_run_id, question, final_sql/validated_sql, execution_time_ms, row_count, steps_count
      - metadata_dict (identified_*)
      - steps (raw steps list; may be large)
    """
    started = time.perf_counter()
    react_run_id = payload.get("react_run_id")
    question = payload.get("question") or ""
    sql = payload.get("validated_sql") or payload.get("final_sql") or payload.get("sql") or ""
    status = payload.get("status") or "completed"

    SmartLogger.log(
        "INFO",
        "cache_postprocess.start",
        category="cache_postprocess",
        params=sanitize_for_log(
            {
                "react_run_id": react_run_id,
                "question": question,
                "has_sql": bool(sql),
            }
        ),
        max_inline_chars=0,
    )

    neo4j_session = None
    db_conn = None
    try:
        neo4j_session = await neo4j_conn.get_session()
        db_conn = await _open_db_connection()

        # 1) LLM summarize steps + extract mapping candidates
        steps_summary = await _llm_build_steps_summary(
            question=question,
            sql=sql,
            metadata=payload.get("metadata_dict") or {},
            steps=payload.get("steps") or [],
        )
        mapping_candidates = await _llm_extract_value_mappings(
            question=question,
            sql=sql,
            metadata=payload.get("metadata_dict") or {},
            steps_summary=steps_summary,
        )

        SmartLogger.log(
            "INFO",
            "cache_postprocess.value_mapping.candidates",
            category="cache_postprocess.value_mapping",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "target_db_type": settings.target_db_type,
                    "react_caching_db_type": settings.react_caching_db_type,
                    "candidates_count": len(mapping_candidates),
                    "candidates_sample": [_candidate_brief(c) for c in mapping_candidates[:5]],
                }
            ),
            max_inline_chars=0,
        )

        # 2) Strong gate: validate candidates against DB existence
        validated_mappings: List[ValueMappingCandidate] = []
        reject_reason_counts = {
            "policy_rejected": 0,
            "resolve_column_failed": 0,
            "value_not_in_db": 0,
        }
        reject_samples: Dict[str, List[Dict[str, Any]]] = {
            "policy_rejected": [],
            "resolve_column_failed": [],
            "value_not_in_db": [],
        }
        for cand in mapping_candidates:
            if not _passes_value_mapping_policy(cand):
                reject_reason_counts["policy_rejected"] += 1
                _append_sample(reject_samples["policy_rejected"], cand)
                continue
            resolved = await _resolve_column_case(db_conn, cand.schema, cand.table, cand.column)
            if not resolved:
                reject_reason_counts["resolve_column_failed"] += 1
                _append_sample(reject_samples["resolve_column_failed"], cand)
                continue
            schema_real, table_real, column_real = resolved
            exists = await _value_exists_in_db(db_conn, schema_real, table_real, column_real, cand.code_value)
            if not exists:
                reject_reason_counts["value_not_in_db"] += 1
                _append_sample(reject_samples["value_not_in_db"], cand)
                continue
            validated_mappings.append(
                ValueMappingCandidate(
                    schema=schema_real,
                    table=table_real,
                    column=column_real,
                    natural_value=cand.natural_value,
                    code_value=cand.code_value,
                    confidence=cand.confidence,
                    evidence=cand.evidence,
                )
            )

        SmartLogger.log(
            "INFO",
            "cache_postprocess.value_mapping.validation_summary",
            category="cache_postprocess.value_mapping",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "candidates_count": len(mapping_candidates),
                    "validated_count": len(validated_mappings),
                    "reject_reason_counts": reject_reason_counts,
                    "reject_samples": reject_samples,
                }
            ),
            max_inline_chars=0,
        )

        # 3) Upsert Query + relationships + mappings
        repo = Neo4jQueryRepository(neo4j_session)
        await repo.setup_constraints()

        # Build compact metadata for storage in Query node.
        identified_tables = (payload.get("metadata_dict") or {}).get("identified_tables") or []
        identified_columns = (payload.get("metadata_dict") or {}).get("identified_columns") or []
        # IMPORTANT: For Neo4j cache graph (Table/Column nodes), use the configured caching DB TYPE label.
        query_db = settings.react_caching_db_type

        query_id = await repo.save_query(
            question=question,
            sql=sql,
            status=status,
            metadata={
                "identified_tables": identified_tables,
                "identified_columns": identified_columns,
                "steps_summary": steps_summary,
                "validated_value_mappings": [cand.__dict__ for cand in validated_mappings],
            },
            row_count=payload.get("row_count"),
            execution_time_ms=payload.get("execution_time_ms"),
            steps_count=payload.get("steps_count"),
            error_message=payload.get("error_message"),
            steps=None,  # Do not store raw steps in Neo4j.
            db=query_db,
            steps_summary=steps_summary,
            value_mappings=[cand.__dict__ for cand in validated_mappings],
        )

        # 4) Save value mappings (only validated ones)
        for cand in validated_mappings:
            column_fqn = f"{cand.schema}.{cand.table}.{cand.column}"
            await repo.save_value_mapping_by_fqn(
                natural_value=cand.natural_value,
                code_value=cand.code_value,
                column_fqn=column_fqn,
            )

        # 5) Update Query vector for embedding search
        await _update_query_vector(
            neo4j_session,
            query_id=query_id,
            question=question,
            steps_summary=steps_summary,
        )

        SmartLogger.log(
            "INFO",
            "cache_postprocess.done",
            category="cache_postprocess",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "query_id": query_id,
                    "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                    "validated_mappings_count": len(validated_mappings),
                }
            ),
            max_inline_chars=0,
        )
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "cache_postprocess.error",
            category="cache_postprocess",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "exception": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            ),
            max_inline_chars=0,
        )
        raise
    finally:
        try:
            if db_conn is not None:
                await db_conn.close()
        finally:
            if neo4j_session is not None:
                await neo4j_session.close()


async def _open_db_connection() -> asyncpg.Connection:
    # SSL mode: 'disable' -> ssl=False, other values passed as ssl parameter
    ssl_mode = settings.target_db_ssl if settings.target_db_ssl != "disable" else False
    conn = await asyncpg.connect(
        host=settings.target_db_host,
        port=settings.target_db_port,
        database=settings.target_db_name,
        user=settings.target_db_user,
        password=settings.target_db_password,
        ssl=ssl_mode,
    )
    schemas = settings.target_db_schemas.split(",")
    schemas_str = ", ".join(s.strip() for s in schemas if s.strip())
    if schemas_str:
        await conn.execute(f"SET search_path TO {schemas_str}")
    return conn


def _passes_value_mapping_policy(cand: ValueMappingCandidate) -> bool:
    natural = (cand.natural_value or "").strip()
    code = (cand.code_value or "").strip()
    if not natural or not code:
        return False
    if len(natural) < 2:
        return False
    # Avoid gigantic strings.
    if len(code) > 128 or len(natural) > 64:
        return False

    col = (cand.column or "").strip()
    if col and col.lower() in {c.lower() for c in VALUE_MAPPING_COLUMN_NAME_DENYLIST}:
        return False

    schema = (cand.schema or "").strip()
    table = (cand.table or "").strip()
    fqn = ".".join([p for p in [schema, table, col] if p])

    # Denylist has priority.
    for pat in VALUE_MAPPING_COLUMN_FQN_DENYLIST_REGEX:
        try:
            if re.search(pat, fqn, flags=re.IGNORECASE):
                return False
        except re.error:
            # Bad regex should not break runtime; ignore.
            continue

    # If allowlist is set, require at least one match.
    if VALUE_MAPPING_COLUMN_FQN_ALLOWLIST_REGEX:
        ok = False
        for pat in VALUE_MAPPING_COLUMN_FQN_ALLOWLIST_REGEX:
            try:
                if re.search(pat, fqn, flags=re.IGNORECASE):
                    ok = True
                    break
            except re.error:
                continue
        if not ok:
            return False

    return True


async def _resolve_column_case(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    column: str,
) -> Optional[Tuple[str, str, str]]:
    """
    Resolve exact identifier case via information_schema (PostgreSQL).
    Returns (schema, table, column) as stored in DB.
    """
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        # Strong gate requires DBMS-specific metadata queries. Unsupported DBMS => skip.
        return None
    q = """
    SELECT table_schema, table_name, column_name
    FROM information_schema.columns
    WHERE lower(table_schema) = lower($1)
      AND lower(table_name) = lower($2)
      AND lower(column_name) = lower($3)
    LIMIT 1
    """
    row = await conn.fetchrow(q, schema, table, column)
    if not row:
        return None
    return row["table_schema"], row["table_name"], row["column_name"]


async def _value_exists_in_db(
    conn: asyncpg.Connection,
    schema: str,
    table: str,
    column: str,
    code_value: str,
) -> bool:
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        return False
    # Safe-quote identifiers by doubling quotes.
    def q_ident(x: str) -> str:
        return '"' + x.replace('"', '""') + '"'

    sql = (
        f"SELECT 1 AS one FROM {q_ident(schema)}.{q_ident(table)} "
        f"WHERE {q_ident(column)} = $1 LIMIT 1"
    )
    row = await conn.fetchrow(sql, code_value)
    return bool(row)


async def _llm_build_steps_summary(
    *,
    question: str,
    sql: str,
    metadata: Dict[str, Any],
    steps: List[Dict[str, Any]],
) -> str:
    """
    Returns a compact JSON string.
    This is intentionally stored as text for portability.
    """
    # Trim steps to reduce token usage.
    trimmed_steps = steps[-10:] if isinstance(steps, list) else []
    user_prompt = {
        "question": question,
        "sql": sql,
        "metadata": metadata,
        "steps_tail": trimmed_steps,
        "requirements": {
            "language": "ko",
            "output_format": "json",
            "must_include_fields": [
                "intent",
                "tables",
                "columns",
                "filters",
                "aggregations",
                "group_by",
                "order_by",
                "time_range",
                "notes",
            ],
            "max_chars": 3000,
        },
    }

    response = await openai_client.chat.completions.create(
        model=settings.openai_llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 Text2SQL 실행 과정을 요약하여 캐시/재사용에 필요한 최소 정보만 추출하는 전문가입니다. "
                    "반드시 JSON만 출력하세요(설명문 금지)."
                ),
            },
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False, default=str)},
        ],
        temperature=0.1,
        max_tokens=900,
    )
    text = (response.choices[0].message.content or "").strip()
    # Best-effort: ensure it's JSON.
    try:
        obj = json.loads(text)
    except Exception:
        obj = {"intent": "", "tables": [], "columns": [], "filters": [], "aggregations": [], "notes": text[:1000]}
    return json.dumps(obj, ensure_ascii=False)


async def _llm_extract_value_mappings(
    *,
    question: str,
    sql: str,
    metadata: Dict[str, Any],
    steps_summary: str,
) -> List[ValueMappingCandidate]:
    """
    Produce candidates for ValueMapping with schema/table/column hints.
    LLM output is validated by strong gate before saving.
    """
    prompt = {
        "question": question,
        "sql": sql,
        "metadata": metadata,
        "steps_summary_json": steps_summary,
        "task": (
            "질문에서 사용자가 말한 자연어 값(natural_value)과 SQL/메타데이터에 사용된 코드/식별자(code_value)를 찾아 "
            "ValueMapping 후보를 생성하세요."
        ),
        "output_schema": {
            "type": "array",
            "items": {
                "schema": "string",
                "table": "string",
                "column": "string",
                "natural_value": "string",
                "code_value": "string",
                "confidence": "number",
                "evidence": "string",
            },
        },
        "constraints": {
            "language": "ko",
            "no_generic_terms": True,
            "max_items": 10,
        },
    }

    response = await openai_client.chat.completions.create(
        model=settings.openai_llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 데이터베이스 ValueMapping 후보를 추출하는 시스템입니다. "
                    "반드시 JSON 배열만 출력하세요(설명문/마크다운 금지)."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, default=str)},
        ],
        temperature=0.1,
        max_tokens=900,
    )
    text = (response.choices[0].message.content or "").strip()
    try:
        arr = json.loads(text)
    except Exception:
        SmartLogger.log(
            "WARNING",
            "cache_postprocess.value_mapping.llm_parse_failed",
            category="cache_postprocess.value_mapping",
            params=sanitize_for_log(
                {
                    "model": settings.openai_llm_model,
                    "text_len": len(text),
                    "text_head": text[:500],
                }
            ),
            max_inline_chars=0,
        )
        arr = []

    candidates: List[ValueMappingCandidate] = []
    if isinstance(arr, list):
        for item in arr:
            if not isinstance(item, dict):
                continue
            candidates.append(
                ValueMappingCandidate(
                    schema=str(item.get("schema") or "").strip(),
                    table=str(item.get("table") or "").strip(),
                    column=str(item.get("column") or "").strip(),
                    natural_value=str(item.get("natural_value") or "").strip(),
                    code_value=str(item.get("code_value") or "").strip(),
                    confidence=float(item.get("confidence") or 0.0),
                    evidence=str(item.get("evidence") or "").strip(),
                )
            )

    # Basic de-dup
    seen = set()
    uniq: List[ValueMappingCandidate] = []
    for c in candidates:
        key = (c.schema.lower(), c.table.lower(), c.column.lower(), c.natural_value, c.code_value)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


async def _update_query_vector(
    neo4j_session,
    *,
    query_id: str,
    question: str,
    steps_summary: str,
) -> None:
    try:
        embedder = EmbeddingClient(openai_client)
        text = f"Question: {question}\nSummary: {steps_summary}"
        vector = await embedder.embed_text(text[:8000])
        query = """
        MATCH (q:Query {id: $id})
        SET q.vector = $vector,
            q.vector_updated_at = datetime()
        """
        await neo4j_session.run(query, id=query_id, vector=vector)
    except Exception as exc:
        SmartLogger.log(
            "WARNING",
            "cache_postprocess.query_vector.update_failed",
            category="cache_postprocess.query_vector",
            params=sanitize_for_log({"query_id": query_id, "exception": repr(exc)}),
            max_inline_chars=0,
        )


