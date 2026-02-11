"""Dependency injection for FastAPI"""
import os
from typing import Any, AsyncGenerator

from neo4j import AsyncGraphDatabase, AsyncDriver
import asyncpg
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache


from app.config import settings


def init_cache():
    if not os.path.exists(settings.llm_cache_path):
        os.makedirs(os.path.dirname(settings.llm_cache_path), exist_ok=True)

    set_llm_cache(SQLiteCache(database_path=settings.llm_cache_path))

if settings.is_use_llm_cache:
    init_cache()


class Neo4jConnection:
    """Neo4j connection manager"""
    
    def __init__(self):
        self.driver: AsyncDriver | None = None
    
    async def connect(self):
        """Initialize Neo4j driver"""
        if not self.driver:
            self.driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
    
    async def close(self):
        """Close Neo4j driver"""
        if self.driver:
            await self.driver.close()
            self.driver = None
    
    async def get_session(self):
        """Get Neo4j session"""
        if not self.driver:
            await self.connect()
        return self.driver.session(database=settings.neo4j_database)


# Global instances
neo4j_conn = Neo4jConnection()


async def get_neo4j_session():
    """FastAPI dependency for Neo4j session"""
    session = await neo4j_conn.get_session()
    try:
        yield session
    finally:
        await session.close()


async def get_db_connection() -> AsyncGenerator[Any, None]:
    """FastAPI dependency for target database connection"""
    db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()

    # MindsDB-only (Phase 1): connect via MySQL protocol endpoint
    if db_type in {"mysql", "mariadb"}:
        try:
            import aiomysql  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "aiomysql is required for target_db_type=mysql (MindsDB MySQL endpoint). "
                "Please add aiomysql to dependencies."
            ) from exc

        conn: Any = await aiomysql.connect(
            host=settings.target_db_host,
            port=int(settings.target_db_port),
            user=settings.target_db_user,
            password=settings.target_db_password,
            db=settings.target_db_name,
            autocommit=True,
        )
        try:
            yield conn
        finally:
            try:
                conn.close()
                await conn.wait_closed()
            except Exception:
                pass
        return

    # PostgreSQL (legacy / non-MindsDB mode)
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
    try:
        # Set search_path to include all configured schemas (public, dw, etc.)
        schemas = (settings.target_db_schemas or "").split(",")
        schemas_str = ", ".join(s.strip() for s in schemas if s.strip())
        if schemas_str:
            await conn.execute(f"SET search_path TO {schemas_str}")
        yield conn
    finally:
        await conn.close()


