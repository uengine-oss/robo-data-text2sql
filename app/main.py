"""FastAPI main application"""
from dotenv import load_dotenv
load_dotenv()

import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.deps import neo4j_conn
from app.routers import ask, meta, feedback, react, history, cache, direct_sql
from app.smart_logger import SmartLogger
from app.core.background_jobs import start_cache_postprocess_workers, stop_cache_postprocess_workers
from app.core.neo4j_bootstrap import ensure_neo4j_schema
from app.core.enum_cache_bootstrap import ensure_enum_cache_for_schema
from app.core.text2sql_table_vectorizer import ensure_text_to_sql_table_vectors
from app.core.text2sql_validity_bootstrap import has_any_text2sql_validity_flags, ensure_text2sql_validity_flags
from app.core.text2sql_validity_mindsdb import ensure_text2sql_validity_flags_mindsdb
from app.core.text2sql_validity_jobs import start_text2sql_validity_refresh_task, stop_text2sql_validity_refresh_task
from app.sanity_checks.runner import run_startup_sanity_checks_or_raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    # NOTE: Avoid non-ASCII characters in stdout on Windows cp949 consoles.
    print("Starting Neo4j Text2SQL API...")
    # Fail-fast sanity checks (external dependencies)
    await run_startup_sanity_checks_or_raise()
    await neo4j_conn.connect()
    print(f"Connected to Neo4j at {settings.neo4j_uri}")
    print(
        f"Target database: {settings.target_db_type}://{settings.target_db_host}:{settings.target_db_port}/{settings.target_db_name}"
    )
    llm_url_suffix = (
        f" (base_url={settings.llm_provider_url})"
        if (settings.llm_provider in {"openai", "openai_compatible"} and getattr(settings, "llm_provider_url", ""))
        else ""
    )
    light_llm_url_suffix = (
        f" (base_url={settings.light_llm_provider_url})"
        if (
            settings.light_llm_provider in {"openai", "openai_compatible"}
            and getattr(settings, "light_llm_provider_url", "")
        )
        else ""
    )
    print(f"Using LLM: {settings.llm_provider}:{settings.llm_model}{llm_url_suffix}")
    print(
        f"Using Light LLM: {settings.light_llm_provider}:{settings.light_llm_model}{light_llm_url_suffix}"
    )
    print(f"Using Embedding Model: {settings.embedding_provider}:{settings.embedding_model}")
    
    # Ensure Neo4j schema/indexes exist (best-effort, idempotent).
    try:
        session = await neo4j_conn.get_session()
        try:
            try:
                await ensure_neo4j_schema(session)
                print("Neo4j schema/index bootstrap completed")
            except Exception as e:
                print(f"Neo4j schema/index bootstrap warning: {e}")
                SmartLogger.log(
                    "WARNING",
                    "main.lifespan.neo4j_schema.warning",
                    category="main.lifespan.start",
                    params={"error": str(e)},
                    max_inline_chars=0,
                )

            # ----------------------------------------------------------------
            # spec 007 T3 / FR-026 — 기동은 색인과 무관해야 한다.
            #
            # 이전 판은 여기서 `ensure_text_to_sql_table_vectors` 를 **블로킹**으로
            # 돌렸다. 그 잡은 테이블마다 LLM 프로필 + 임베딩을 요구하므로 키가 없으면
            # 기동 자체가 늦어지거나 실패했고, 그것이 데스크톱 크래시 루프의 한 축이었다.
            # 색인은 이제 메타데이터 추출의 단계다 (`POST /text2sql/index/datasource`).
            #
            # 되돌리기: TEXT2SQL_VECTORIZE_ON_STARTUP=true 로 기동 잡을 다시 켜고,
            #           TEXT2SQL_VECTORIZE_ON_STARTUP_BLOCKING=true 로 옛 블로킹 동작까지
            #           복원한다. 기본값만 바뀌었을 뿐 코드 경로는 그대로 남아 있다.
            # ----------------------------------------------------------------
            if bool(getattr(settings, "text2sql_vectorize_on_startup", False)):
                blocking = bool(getattr(settings, "text2sql_vectorize_on_startup_blocking", False))
                if blocking:
                    try:
                        await ensure_text_to_sql_table_vectors(session)
                        print("Text-to-SQL table vectors ensured (blocking, legacy mode)")
                    except Exception as e:
                        print(f"Text-to-SQL table vectors warning: {e}")
                        SmartLogger.log(
                            "WARNING",
                            "main.lifespan.text2sql_vectors.warning",
                            category="main.lifespan.start",
                            params={"error": str(e)},
                            max_inline_chars=0,
                        )
                else:
                    # 비블로킹: 기동을 막지 않는다. 실패해도 서비스는 뜬다.
                    async def _bg_vectorize() -> None:
                        bg_session = await neo4j_conn.get_session()
                        try:
                            await ensure_text_to_sql_table_vectors(bg_session)
                        except Exception as e:
                            SmartLogger.log(
                                "WARNING",
                                "main.lifespan.text2sql_vectors.background_warning",
                                category="main.lifespan.start",
                                params={"error": str(e)},
                                max_inline_chars=0,
                            )
                        finally:
                            await bg_session.close()

                    asyncio.create_task(_bg_vectorize())
                    print("Text-to-SQL table vectors: background job started (non-blocking)")
            else:
                print(
                    "Text-to-SQL table vectors: startup job disabled "
                    "(색인은 메타데이터 추출 단계에서 수행됩니다 — POST /text2sql/index/datasource)"
                )

            db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()

            # Text2SQL validity flags:
            # - If flags are missing entirely, compute once in blocking mode (one-time cold-start cost).
            # - Otherwise, refresh in background (non-blocking).
            if bool(getattr(settings, "text2sql_validity_bootstrap_enabled", True)):
                try:
                    try:
                        has_any = await has_any_text2sql_validity_flags(neo4j_session=session)
                    except Exception:
                        has_any = False
                    if not has_any:
                        print("Text2SQL validity flags missing: computing (blocking, one-time)...")
                        if db_type in {"mysql", "mariadb"}:
                            out = await ensure_text2sql_validity_flags_mindsdb(
                                neo4j_session=session,
                                datasource=None,
                                db_conn=None,
                            )
                        else:
                            out = await ensure_text2sql_validity_flags(neo4j_session=session)
                        if bool(out.get("ok")):
                            print("Text2SQL validity flags ensured")
                        else:
                            print(f"Text2SQL validity bootstrap warning: {out}")
                    else:
                        # Background refresh (best-effort, PostgreSQL direct mode only).
                        # MindsDB mode requires datasource-aware probe and is refreshed in request-time repair.
                        if db_type not in {"mysql", "mariadb"}:
                            await start_text2sql_validity_refresh_task()
                except Exception as e:
                    print(f"Text2SQL validity bootstrap warning: {e}")
                    SmartLogger.log(
                        "WARNING",
                        "main.lifespan.text2sql_validity.warning",
                        category="main.lifespan.start",
                        params={"error": str(e)},
                        max_inline_chars=0,
                    )

            if settings.enum_cache_bootstrap_on_startup:
                schema = settings.target_db_schema
                max_values = settings.enum_cache_bootstrap_max_values
                max_columns = settings.enum_cache_bootstrap_max_columns
                concurrency = settings.enum_cache_bootstrap_concurrency
                query_timeout_s = settings.enum_cache_bootstrap_query_timeout_s

                if db_type in {"mysql", "mariadb"}:
                    import aiomysql  # type: ignore

                    db_pool = await aiomysql.create_pool(
                        host=settings.target_db_host,
                        port=int(settings.target_db_port),
                        user=settings.target_db_user,
                        password=settings.target_db_password,
                        db=settings.target_db_name,
                        minsize=1,
                        maxsize=max(1, int(concurrency)),
                        autocommit=True,
                    )
                    try:
                        await ensure_enum_cache_for_schema(
                            neo4j_session=session,
                            db_conn=db_pool,
                            schema=schema,
                            max_columns=max_columns,
                            max_values=max_values,
                            concurrency=concurrency,
                            query_timeout_s=query_timeout_s,
                            name_hint_only=False,
                            include_fqns=None,
                            skip_if_cached=True,
                            datasource=None,
                        )
                    finally:
                        db_pool.close()
                        await db_pool.wait_closed()
                else:
                    import asyncpg

                    # Use a dedicated DB pool (do not reuse request-scoped deps).
                    # asyncpg.Connection does NOT allow concurrent queries, so we need a Pool
                    # when concurrency>1 to avoid "another operation is in progress".
                    ssl_mode = settings.target_db_ssl if settings.target_db_ssl != "disable" else False

                    async def _init_conn(conn: asyncpg.Connection) -> None:
                        # Keep consistent with request-scoped dependency: include all schemas in search_path.
                        schemas = settings.target_db_schemas.split(",")
                        schemas_str = ", ".join(s.strip() for s in schemas if s.strip())
                        if schemas_str:
                            await conn.execute(f"SET search_path TO {schemas_str}")

                    db_pool = await asyncpg.create_pool(
                        host=settings.target_db_host,
                        port=settings.target_db_port,
                        database=settings.target_db_name,
                        user=settings.target_db_user,
                        password=settings.target_db_password,
                        ssl=ssl_mode,
                        min_size=1,
                        max_size=max(1, int(concurrency)),
                        init=_init_conn,
                    )
                    try:
                        await ensure_enum_cache_for_schema(
                            neo4j_session=session,
                            db_conn=db_pool,
                            schema=schema,
                            max_columns=max_columns,
                            max_values=max_values,
                            concurrency=concurrency,
                            query_timeout_s=query_timeout_s,
                            # generic default: consider name-hint or text-ish dtype
                            name_hint_only=False,
                            include_fqns=None,
                            skip_if_cached=True,
                            datasource=None,
                        )
                    finally:
                        await db_pool.close()
        finally:
            await session.close()
    except Exception as e:
        # Best-effort: service can still start; tools may fallback/recover later.
        print(f"Neo4j schema/index bootstrap warning: {e}")
        SmartLogger.log(
            "WARNING",
            "main.lifespan.neo4j_bootstrap.warning",
            category="main.lifespan.start",
            params={"error": str(e)},
            max_inline_chars=0,
        )
    SmartLogger.log(
        "INFO",
        "Starting Neo4j Text2SQL API...",
        category="main.lifespan.start"
    )

    # Background workers (best-effort)
    await start_cache_postprocess_workers()

    yield
    
    # Shutdown
    print("Shutting down...")
    await stop_text2sql_validity_refresh_task()
    await stop_cache_postprocess_workers()
    await neo4j_conn.close()
    print("Neo4j connection closed")


app = FastAPI(
    title="Neo4j Text2SQL API",
    description="""
    Natural Language to SQL converter with Neo4j-powered RAG.
    
    ## Features
    - 🧠 Natural language to SQL conversion
    - 📊 Automatic data visualization recommendations
    - 🔍 Schema-aware query generation using Neo4j graph
    - 🔒 SQL safety guards (SELECT-only, validation)
    - 📈 Performance tracking and provenance
    - 💾 User feedback learning system
    
    ## Workflow
    1. Ask questions: `POST /ask`
    2. Explore metadata: `GET /meta/tables`
    3. Provide feedback: `POST /feedback`
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with /text2sql prefix
app.include_router(ask.router, prefix="/text2sql")
app.include_router(meta.router, prefix="/text2sql")
app.include_router(feedback.router, prefix="/text2sql")
app.include_router(react.router, prefix="/text2sql")

# Import and include schema editing router
from app.routers import schema_edit
app.include_router(schema_edit.router, prefix="/text2sql")

# Include history router
app.include_router(history.router, prefix="/text2sql")

# Include cache router
app.include_router(cache.router, prefix="/text2sql")

# Include direct SQL router
app.include_router(direct_sql.router, prefix="/text2sql")

# 색인 API (spec 007 T3·T4). 추출 파이프라인이 부르고 UI 가 상태를 읽는다.
from app.routers import indexing
app.include_router(indexing.router, prefix="/text2sql")

# Include events router for event detection and actions
from app.routers import events
app.include_router(events.router, prefix="/text2sql")

# Include event templates router
from app.routers import event_templates
app.include_router(event_templates.router, prefix="/text2sql")

# Include watch agent router
from app.routers import watch_agent
app.include_router(watch_agent.router, prefix="/text2sql")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Neo4j Text2SQL API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Neo4j connection
        session = await neo4j_conn.get_session()
        result = await session.run("RETURN 1 AS health")
        await result.single()
        await session.close()
        
        return {
            "status": "healthy",
            "neo4j": "connected",
            "config": {
                "llm_provider": settings.llm_provider,
                "llm_model": settings.llm_model,
                "embedding_provider": settings.embedding_provider,
                "embedding_model": settings.embedding_model,
                "target_db": settings.target_db_type
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

