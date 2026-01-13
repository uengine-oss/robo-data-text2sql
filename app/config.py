"""Application configuration"""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # Ingest + Embedding Search
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_llm_model: str = "gpt-4.1-2025-04-14"
    is_use_llm_cache: bool = False
    llm_cache_path: str = ".cache/llm_cache.db"

    # ReAct Agent
    # LLM Provider: "openai" or "google" (auto-detected if not set)
    react_llm_provider: str = ""  # Empty = auto-detect based on API keys
    # OpenAI settings for ReAct (used when react_llm_provider="openai")
    react_openai_llm_model: str = "gpt-4.1"  # or gpt-4.1-mini, o4-mini, etc.
    # Google settings for ReAct (used when react_llm_provider="google")
    google_api_key: str = ""  # Optional, can be empty if using OpenAI
    react_google_llm_model: str = "gemini-3-flash-preview"
    is_add_delay_after_react_generator: bool = False
    delay_after_react_generator_seconds: int = 5
    previous_reasoning_limit_steps: int = 15
    is_add_mocked_db_caution: bool = False
    explain_analysis_timeout_seconds: int = 10
    react_caching_db_type: str = "oracle"
    
    # Target Database
    target_db_type: Literal["postgresql", "mysql", "oracle"] = "postgresql"
    target_db_host: str = "localhost"
    target_db_port: int = 5432
    target_db_name: str
    target_db_user: str
    target_db_password: str
    target_db_schema: str = "public"
    target_db_schemas: str = "public"  # Comma-separated list of schemas to access
    target_db_ssl: str = "disable"  # SSL mode: disable, require, verify-ca, verify-full
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # Security & Limits
    sql_timeout_seconds: int = 30
    sql_row_limit: int = 1000
    sql_max_rows: int = 100000
    max_join_depth: int = 10
    max_subquery_depth: int = 10
    
    # RAG
    vector_top_k: int = 10
    max_fk_hops: int = 3
    embedding_dimension: int = 1536

    # Cache postprocess background jobs (in-process)
    cache_postprocess_worker_count: int = 1
    cache_postprocess_queue_maxsize: int = 200
    
    # Logging
    log_level: str = "INFO"

    # LangSmith
    langsmith_tracing: bool = False
    langsmith_project: str = ""
    langsmith_api_key: str = ""

    # SmartLogger (SMART_LOGGER_*)
    smart_logger_main_log_path: str = "logs/app_flow.jsonl"
    smart_logger_detail_log_dir: str = "logs/details"
    smart_logger_min_level: str = "ERROR"
    smart_logger_include_all_min_level: str = "ERROR"
    smart_logger_console_output: bool = True
    smart_logger_file_output: bool = False
    smart_logger_remove_log_on_create: bool = False
    smart_logger_blacklist_messages: str = "[]"

    # Gemini Context Caching
    gemini_context_cache_enabled: bool = True
    gemini_context_cache_ttl_seconds: int = 3600
    gemini_context_cache_refresh_buffer_seconds: int = 120
    gemini_context_cache_retry_backoff_seconds: int = 60

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

