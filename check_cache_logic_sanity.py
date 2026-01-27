
# cd neo4j-text2sql; uv run python check_cache_logic_sanity.py
"""
Cache logic sanity checker (standalone).

This script performs preflight validation for Text2SQL caching flow:
- Settings sanity (required env values present)
- LLM connectivity + JSON output shape validation for cache postprocess prompts
- Neo4j connectivity + (optional) constraints/index provisioning check
- Target DB connectivity + DB existence gate query sanity (PostgreSQL only)
- Optional end-to-end run of cache postprocess worker (writes to Neo4j)

Design principles:
- By default, avoid writing to Neo4j. Use --apply-neo4j-constraints or --e2e to write.
- Provide actionable output and a non-zero exit code on failure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


def _mask_secret(value: str, *, head: int = 6, tail: int = 4) -> str:
    if not value:
        return "***"
    v = str(value)
    if len(v) <= head + tail + 3:
        return "***"
    return f"{v[:head]}...{v[-tail:]}"


def _print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _parse_fqn(fqn: str) -> Tuple[str, str, str]:
    parts = [p.strip() for p in (fqn or "").split(".") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--gate-fqn must be 'schema.table.column' (got: {fqn!r})")
    return parts[0], parts[1], parts[2]


async def _check_settings_sanity(settings) -> bool:
    _print_section("1) Settings sanity")

    ok = True
    required_fields = ["openai_api_key", "target_db_name", "target_db_user", "target_db_password"]
    for k in required_fields:
        v = getattr(settings, k, None)
        if not v:
            ok = False
            print(f"[FAIL] Missing required setting: {k}")
    if not ok:
        print("[HINT] Ensure your `.env` has required keys (OPENAI_API_KEY, TARGET_DB_*, etc.).")
        return False

    print(f"[OK] OpenAI API key: {_mask_secret(settings.openai_api_key)}")
    print(f"[OK] OpenAI embedding model: {settings.openai_embedding_model}")
    print(f"[OK] OpenAI LLM model: {settings.openai_llm_model}")
    print(f"[OK] LLM cache enabled: {bool(settings.is_use_llm_cache)}")
    print(f"[OK] LLM cache path: {settings.llm_cache_path}")

    print(f"[OK] Neo4j URI: {settings.neo4j_uri}")
    print(f"[OK] Neo4j database: {settings.neo4j_database}")
    print(f"[OK] react_caching_db_type (Neo4j label): {settings.react_caching_db_type}")
    if not (settings.react_caching_db_type or "").strip():
        ok = False
        print("[FAIL] react_caching_db_type is empty. Cache graph relations may not work as expected.")

    print(
        f"[OK] Target DB: type={settings.target_db_type} host={settings.target_db_host}:{settings.target_db_port} "
        f"db={settings.target_db_name} schemas={settings.target_db_schemas} ssl={settings.target_db_ssl}"
    )
    return ok


async def _check_llm(openai_client, settings, *, question: str, sql: str, skip_embedding: bool) -> bool:
    _print_section("2) LLM sanity (OpenAI)")

    ok = True
    try:
        if not skip_embedding:
            print("[INFO] Embedding API test...")
            emb = await openai_client.embeddings.create(
                input="cache sanity check",
                model=settings.openai_embedding_model,
                encoding_format="float",
            )
            dim = len(emb.data[0].embedding)
            print(f"[OK] Embedding dim: {dim}")
            if getattr(settings, "embedding_dimension", None) and int(settings.embedding_dimension) != int(dim):
                ok = False
                print(
                    f"[FAIL] embedding_dimension mismatch: settings={settings.embedding_dimension} actual={dim}. "
                    "Neo4j vector index dims must match."
                )

        print("[INFO] Chat completion API test...")
        completion = await openai_client.chat.completions.create(
            model=settings.openai_llm_model,
            messages=[{"role": "user", "content": "Hello, respond with 'OK' only."}],
            max_tokens=10,
        )
        text = (completion.choices[0].message.content or "").strip()
        print(f"[OK] Chat response: {text!r}")

    except Exception as exc:
        ok = False
        print(f"[FAIL] OpenAI API call failed: {exc!r}")
        return False

    # Cache postprocess prompts: JSON-only output shape
    try:
        from app.core.cache_postprocess import _llm_build_steps_summary, _llm_extract_value_mappings
    except Exception as exc:
        print(f"[FAIL] Failed to import cache_postprocess helpers: {exc!r}")
        return False

    try:
        print("[INFO] cache_postprocess: building steps_summary JSON...")
        steps_summary = await _llm_build_steps_summary(
            question=question,
            sql=sql,
            metadata={},
            steps=[
                {"iteration": 1, "tool_name": "db_schema_search", "reasoning": "find tables"},
                {"iteration": 2, "tool_name": "sql_generate", "reasoning": "generate SQL"},
            ],
        )
        summary_obj = json.loads(steps_summary)
        must_have = [
            "intent",
            "tables",
            "columns",
            "filters",
            "aggregations",
            "group_by",
            "order_by",
            "time_range",
            "notes",
        ]
        missing = [k for k in must_have if k not in summary_obj]
        if missing:
            ok = False
            print(f"[FAIL] steps_summary JSON missing keys: {missing}")
            print(f"[DEBUG] steps_summary: {steps_summary[:500]}")
        else:
            print(f"[OK] steps_summary JSON looks valid (keys={len(summary_obj.keys())})")

        print("[INFO] cache_postprocess: extracting value mapping candidates (JSON array)...")
        candidates = await _llm_extract_value_mappings(
            question=question,
            sql=sql,
            metadata={},
            steps_summary=steps_summary,
        )
        print(f"[OK] value mapping candidates extracted: {len(candidates)}")
        if candidates:
            # Show a tiny sample (no secrets here; still keep short)
            c0 = candidates[0]
            print(
                "[SAMPLE] "
                + json.dumps(
                    {
                        "schema": c0.schema,
                        "table": c0.table,
                        "column": c0.column,
                        "natural_value": c0.natural_value,
                        "code_value": c0.code_value,
                        "confidence": c0.confidence,
                    },
                    ensure_ascii=False,
                )
            )
    except Exception as exc:
        ok = False
        print(f"[FAIL] cache_postprocess LLM JSON sanity failed: {exc!r}")
        print(traceback.format_exc())

    return ok


async def _check_neo4j(neo4j_conn, settings, *, apply_constraints: bool) -> bool:
    _print_section("3) Neo4j sanity")

    ok = True
    session = None
    try:
        await neo4j_conn.connect()
        session = await neo4j_conn.get_session()

        r = await session.run("RETURN 1 AS one, datetime() AS ts")
        rec = await r.single()
        print(f"[OK] Neo4j query ok: one={rec['one']} ts={rec['ts']}")

        # Read-only inventory
        try:
            res = await session.run("SHOW CONSTRAINTS YIELD name, type RETURN name, type ORDER BY name")
            constraints = await res.data()
            print(f"[OK] Constraints: {len(constraints)}")
        except Exception as exc:
            constraints = []
            print(f"[WARN] Could not SHOW CONSTRAINTS (permission/edition?): {exc!r}")

        try:
            res = await session.run("SHOW INDEXES YIELD name, type RETURN name, type ORDER BY name")
            indexes = await res.data()
            print(f"[OK] Indexes: {len(indexes)}")
        except Exception as exc:
            indexes = []
            print(f"[WARN] Could not SHOW INDEXES (permission/edition?): {exc!r}")

        required_constraint_names = {"query_id", "value_mapping_key"}
        existing_constraint_names = {c.get("name") for c in constraints if isinstance(c, dict)}
        missing = sorted([n for n in required_constraint_names if n not in existing_constraint_names])
        if missing:
            print(f"[WARN] Missing expected constraints: {missing}")
            if apply_constraints:
                print("[INFO] Applying Neo4j constraints/indexes via Neo4jQueryRepository.setup_constraints()...")
                from app.models.neo4j_history import Neo4jQueryRepository

                repo = Neo4jQueryRepository(session)
                await repo.setup_constraints()
                print("[OK] setup_constraints() executed (errors are logged as WARNING inside).")
            else:
                print("[HINT] Run with --apply-neo4j-constraints to create missing constraints/indexes.")
        else:
            print("[OK] Expected constraints appear present (by name).")

    except Exception as exc:
        ok = False
        print(f"[FAIL] Neo4j check failed: {exc!r}")
        print(traceback.format_exc())
    finally:
        try:
            if session is not None:
                await session.close()
        finally:
            try:
                await neo4j_conn.close()
            except Exception:
                pass

    return ok


async def _pick_text_gate_target(conn, *, schemas: List[str]) -> Optional[Tuple[str, str, str, str]]:
    """
    Pick (schema, table, column, sample_value) for a text-like column that has at least one non-null value.
    This enables a deterministic "value exists" gate test.
    """
    q = """
    SELECT table_schema, table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = ANY($1)
      AND data_type IN ('character varying', 'character', 'text')
    ORDER BY table_schema, table_name, ordinal_position
    LIMIT 50
    """
    rows = await conn.fetch(q, schemas)
    for row in rows:
        schema, table, column = row["table_schema"], row["table_name"], row["column_name"]
        # Try to fetch one sample value (as text).
        try:
            def q_ident(x: str) -> str:
                return '"' + x.replace('"', '""') + '"'

            sample_sql = (
                f"SELECT {q_ident(column)}::text AS v "
                f"FROM {q_ident(schema)}.{q_ident(table)} "
                f"WHERE {q_ident(column)} IS NOT NULL "
                f"LIMIT 1"
            )
            v = await conn.fetchval(sample_sql)
            if v is None:
                continue
            v = str(v)
            if not v:
                continue
            return schema, table, column, v
        except Exception:
            # Try next candidate; permissions or types may fail.
            continue
    return None


async def _check_target_db_and_gate(
    get_db_connection,
    settings,
    *,
    gate_fqn: Optional[str],
    gate_value: Optional[str],
) -> bool:
    _print_section("4) Target DB + DB existence gate sanity")

    ok = True
    if (settings.target_db_type or "").lower() not in {"postgresql", "postgres"}:
        print(
            f"[WARN] target_db_type={settings.target_db_type!r}. "
            "DB existence gate in cache_postprocess currently supports PostgreSQL only."
        )
        # Still validate basic connection.

    try:
        from app.core.cache_postprocess import _resolve_column_case, _value_exists_in_db
    except Exception as exc:
        print(f"[FAIL] Failed to import DB gate helpers: {exc!r}")
        return False

    async for conn in get_db_connection():
        try:
            version = await conn.fetchval("SELECT version()")
            print(f"[OK] Connected. Version: {str(version).split(',')[0]}")
            print(f"[OK] current_database(): {await conn.fetchval('SELECT current_database()')}")
            print(f"[OK] current_schema(): {await conn.fetchval('SELECT current_schema()')}")

            # Determine gate test target
            schema: str
            table: str
            column: str
            value: str

            if gate_fqn:
                schema, table, column = _parse_fqn(gate_fqn)
                if gate_value is None:
                    ok = False
                    print("[FAIL] --gate-fqn provided but --gate-value is missing.")
                    return False
                value = str(gate_value)
            else:
                schemas = [s.strip() for s in (settings.target_db_schemas or "").split(",") if s.strip()]
                target = await _pick_text_gate_target(conn, schemas=schemas or [settings.target_db_schema])
                if not target:
                    print(
                        "[WARN] Could not auto-pick a text column with a sample value for gate test. "
                        "Provide --gate-fqn and --gate-value to force a deterministic test."
                    )
                    return ok
                schema, table, column, value = target

            # Resolve exact case using lower() matches (we intentionally scramble case)
            resolved = await _resolve_column_case(conn, schema.upper(), table.upper(), column.upper())
            if not resolved:
                ok = False
                print(f"[FAIL] resolve_column_case failed for {schema}.{table}.{column}")
                return False
            schema_real, table_real, column_real = resolved
            print(f"[OK] resolve_column_case -> {schema_real}.{table_real}.{column_real}")

            # Existence check
            exists = await _value_exists_in_db(conn, schema_real, table_real, column_real, value)
            if exists:
                print(
                    f"[OK] value_exists_in_db passed for {schema_real}.{table_real}.{column_real} "
                    f"(value sample: {value[:80]!r})"
                )
            else:
                ok = False
                print(
                    f"[FAIL] value_exists_in_db returned False for {schema_real}.{table_real}.{column_real}. "
                    "This may indicate quoting/casting issues, permissions, or a bad test value."
                )
                print("[HINT] Try passing an exact existing value with --gate-fqn/--gate-value.")

            return ok
        except Exception as exc:
            ok = False
            print(f"[FAIL] Target DB check failed: {exc!r}")
            print(traceback.format_exc())
            return False

    ok = False
    print("[FAIL] Could not acquire DB connection from get_db_connection().")
    return ok


async def _run_e2e_cache_postprocess(*, question: str, sql: str) -> bool:
    _print_section("5) E2E cache_postprocess (WRITES to Neo4j)")
    print("[WARN] This step writes Query/ValueMapping nodes to Neo4j (MERGE/SET).")

    try:
        from app.core.cache_postprocess import process_cache_postprocess_payload
    except Exception as exc:
        print(f"[FAIL] Failed to import process_cache_postprocess_payload: {exc!r}")
        return False

    payload: Dict[str, Any] = {
        "react_run_id": "sanity_check",
        "question": question,
        "validated_sql": sql,
        "status": "completed",
        "execution_time_ms": 0.0,
        "row_count": None,
        "steps_count": 2,
        "metadata_dict": {"identified_tables": [], "identified_columns": []},
        "steps": [
            {"iteration": 1, "tool_name": "sanity", "reasoning": "cache_postprocess sanity check"},
            {"iteration": 2, "tool_name": "sanity", "reasoning": "cache_postprocess sanity check"},
        ],
    }
    try:
        await process_cache_postprocess_payload(payload)
        print("[OK] E2E cache_postprocess completed.")
        return True
    except Exception as exc:
        print(f"[FAIL] E2E cache_postprocess failed: {exc!r}")
        print(traceback.format_exc())
        return False


async def _main_async() -> int:
    parser = argparse.ArgumentParser(description="Text2SQL cache logic sanity checker")
    parser.add_argument("--skip-llm", action="store_true", help="Skip OpenAI / LLM checks")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding call (still tests chat + JSON prompts)")
    parser.add_argument("--skip-neo4j", action="store_true", help="Skip Neo4j checks")
    parser.add_argument("--skip-db", action="store_true", help="Skip Target DB + gate checks")
    parser.add_argument(
        "--apply-neo4j-constraints",
        action="store_true",
        help="Apply Neo4j constraints/indexes via Neo4jQueryRepository.setup_constraints() (WRITES)",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run end-to-end cache_postprocess payload (WRITES to Neo4j and calls LLM/DB)",
    )
    parser.add_argument(
        "--question",
        default="샘플 질문: 특정 조건으로 데이터를 조회해줘",
        help="Question text used for LLM/cache_postprocess prompt sanity",
    )
    parser.add_argument(
        "--sql",
        default='SELECT 1 AS one',
        help="SQL text used for LLM/cache_postprocess prompt sanity",
    )
    parser.add_argument(
        "--gate-fqn",
        default=None,
        help="Deterministic DB gate test target in form 'schema.table.column' (PostgreSQL only)",
    )
    parser.add_argument(
        "--gate-value",
        default=None,
        help="DB gate test value (used with --gate-fqn). Must exist in the specified column.",
    )
    args = parser.parse_args()

    # Import lazily to provide nicer error when env is missing.
    try:
        from app.config import settings
        from app.deps import openai_client, neo4j_conn, get_db_connection
    except Exception as exc:
        _print_section("IMPORT/ENV ERROR")
        print(f"[FAIL] Failed to import app settings/deps: {exc!r}")
        print("[HINT] Run from 'neo4j-text2sql' directory and ensure `.env` has required variables.")
        print(traceback.format_exc())
        return 2

    all_ok = True

    all_ok = (await _check_settings_sanity(settings)) and all_ok

    if not args.skip_llm:
        all_ok = (await _check_llm(openai_client, settings, question=args.question, sql=args.sql, skip_embedding=args.skip_embedding)) and all_ok
    else:
        _print_section("2) LLM sanity (skipped)")
        print("[INFO] Skipped by --skip-llm")

    if not args.skip_neo4j:
        all_ok = (await _check_neo4j(neo4j_conn, settings, apply_constraints=args.apply_neo4j_constraints)) and all_ok
    else:
        _print_section("3) Neo4j sanity (skipped)")
        print("[INFO] Skipped by --skip-neo4j")

    if not args.skip_db:
        all_ok = (
            await _check_target_db_and_gate(
                get_db_connection,
                settings,
                gate_fqn=args.gate_fqn,
                gate_value=args.gate_value,
            )
        ) and all_ok
    else:
        _print_section("4) Target DB + gate sanity (skipped)")
        print("[INFO] Skipped by --skip-db")

    if args.e2e:
        all_ok = (await _run_e2e_cache_postprocess(question=args.question, sql=args.sql)) and all_ok

    _print_section("RESULT")
    if all_ok:
        print("[PASS] Cache logic sanity checks passed.")
        return 0
    print("[FAIL] One or more checks failed.")
    return 1


def main() -> None:
    try:
        code = asyncio.run(_main_async())
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()


