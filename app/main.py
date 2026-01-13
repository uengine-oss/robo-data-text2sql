"""FastAPI main application"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.deps import neo4j_conn
from app.routers import ask, meta, feedback, ingest, react, vectorize, history, cache, direct_sql
from app.smart_logger import SmartLogger
from app.core.background_jobs import start_cache_postprocess_workers, stop_cache_postprocess_workers
from app.sanity_checks.runner import run_startup_sanity_checks_or_raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("🚀 Starting Neo4j Text2SQL API...")
    # Fail-fast sanity checks (external dependencies)
    await run_startup_sanity_checks_or_raise()
    await neo4j_conn.connect()
    print(f"✓ Connected to Neo4j at {settings.neo4j_uri}")
    print(f"✓ Target database: {settings.target_db_type}://{settings.target_db_host}:{settings.target_db_port}/{settings.target_db_name}")
    print(f"✓ Using LLM for Ingest: {settings.openai_llm_model}")
    print(f"✓ Using Embedding Model for Embedding Search: {settings.openai_embedding_model}")
    print(f"✓ Using LLM for ReAct Agent: {settings.react_google_llm_model}")
    SmartLogger.log(
        "INFO",
        "🚀 Starting Neo4j Text2SQL API...",
        category="main.lifespan.start"
    )

    # Background workers (best-effort)
    await start_cache_postprocess_workers()

    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    await stop_cache_postprocess_workers()
    await neo4j_conn.close()
    print("✓ Neo4j connection closed")


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
    1. Ingest your database schema: `POST /ingest`
    2. Ask questions: `POST /ask`
    3. Explore metadata: `GET /meta/tables`
    4. Provide feedback: `POST /feedback`
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
app.include_router(ingest.router, prefix="/text2sql")
app.include_router(react.router, prefix="/text2sql")
app.include_router(vectorize.router, prefix="/text2sql")

# Import and include schema editing router
from app.routers import schema_edit
app.include_router(schema_edit.router, prefix="/text2sql")

# Include history router
app.include_router(history.router, prefix="/text2sql")

# Include cache router
app.include_router(cache.router, prefix="/text2sql")

# Include direct SQL router
app.include_router(direct_sql.router, prefix="/text2sql")

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
                "llm_model": settings.openai_llm_model,
                "embedding_model": settings.openai_embedding_model,
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

