"""Application configuration"""
from pydantic_settings import BaseSettings
from typing import Literal, Dict, Any


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    neo4j_database: str = "neo4j"

    # Target Database endpoint (MindsDB MySQL protocol)
    target_db_type: Literal["mysql", "mariadb"] = "mysql"
    target_db_host: str = "localhost"
    target_db_port: int = 47335
    target_db_name: str = "mindsdb"
    target_db_user: str = "mindsdb"
    target_db_password: str
    # Optional schema hints used by some background jobs after datasource is connected.
    target_db_schema: str = "public"
    target_db_schemas: str = "public"
    target_db_ssl: str = "disable"  # SSL mode: disable, require, verify-ca, verify-full

    # Unified LLM configuration
    # Provider:
    # - "openai": OpenAI official endpoint
    # - "google": Gemini (alias: "gemini")
    # - "openai_compatible": OpenAI-compatible endpoint (requires *_provider_url)
    llm_provider: Literal["openai", "google", "openai_compatible"] = "google"
    llm_model: str = "gemini-3-flash-preview"

    light_llm_provider: Literal["openai", "google", "openai_compatible"] = "google"
    light_llm_model: str = "gemini-2.5-flash-lite-preview-09-2025"

    # OpenAI-compatible base URLs (used when provider is openai/openai_compatible)
    # Example: http://localhost:11434/v1 (Ollama), https://<gateway>/v1, etc.
    llm_provider_url: str = ""
    light_llm_provider_url: str = ""

    is_use_llm_cache: bool = False
    llm_cache_path: str = ".cache/llm_cache.db"

    # Unified Embedding configuration
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

    # API keys (optional at settings-load time; validated at runtime by factories)
    openai_api_key: str = ""
    # Used when llm_provider/light_llm_provider == "openai_compatible"
    # (Allows separating LLM gateway credentials from OpenAI embeddings credentials.)
    openai_compatible_api_key: str = ""
    google_api_key: str = ""

    # ReAct Agent Configuration
    is_add_delay_after_react_generator: bool = False
    delay_after_react_generator_seconds: int = 5
    previous_reasoning_limit_steps: int = 15
    is_add_mocked_db_caution: bool = False
    explain_analysis_timeout_seconds: int = 10
    react_caching_db_type: str = "oracle"
    # Optional semantic gate: treat some validate_sql PASS as semantic FAIL (can increase iterations).
    # Default OFF because the primary objective is stable low-step completion.
    react_semantic_gate_enabled: bool = False

    # ReAct -> Neo4j cache/history side-effects
    # When True, ReAct completion enqueues background postprocess that upserts (:Query) and persists (:ValueMapping).
    # When False, Query/ValueMapping generation is skipped (no enqueue; workers also short-circuit).
    react_enable_query_value_mapping_generation: bool = True

    # ----------------------------------------------------------------------------
    # Neo4j cache postprocess quality gates (fail-closed defaults)
    # ----------------------------------------------------------------------------
    # If enabled, cache_postprocess will run strong quality checks BEFORE saving (:Query)/(:ValueMapping).
    cache_postprocess_query_quality_gate_enabled: bool = True
    # Number of judge rounds; policy is "ALL rounds must accept".
    cache_postprocess_query_judge_rounds: int = 2
    # Conservative default; tune in production via env var.
    cache_postprocess_query_judge_conf_threshold: float = 0.90
    # Fail-closed: row_count==0 => do not save Query/ValueMapping.
    cache_postprocess_query_min_row_count: int = 1

    # ValueMapping candidate gate
    cache_postprocess_value_mapping_min_confidence: float = 0.92
    cache_postprocess_value_mapping_strict_verify_enabled: bool = True

    # ----------------------------------------------------------------------------
    # Neo4j cache consumption gates (build_sql_context)
    # ----------------------------------------------------------------------------
    # If True, build_sql_context will use only verified Query/ValueMapping nodes.
    cache_postprocess_use_verified_only: bool = True
    # Risk mitigation: substring fallback for ValueMapping lookup is disabled by default.
    value_mapping_substring_fallback_enabled: bool = False
    
    cache_postprocess_worker_count: int = 1
    cache_postprocess_queue_maxsize: int = 200

    # Query similarity clustering (Neo4j cache graph)
    # - If enabled, cache_postprocess will link queries by vector similarity and assign canonical_id.
    # - Use conservative defaults; tune in production with logs.
    query_similarity_cluster_enabled: bool = True
    query_similarity_high_threshold: float = 0.95
    query_similarity_mid_threshold: float = 0.80
    query_similarity_link_top_k: int = 5
    
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

    # Text-to-SQL Table vectorization (startup-time, stored in Neo4j as Table.text_to_sql_vector)
    text2sql_vectorize_on_startup: bool = True
    text2sql_vector_sample_rows: int = 10
    text2sql_vector_db_timeout_seconds: float = 2.0
    text2sql_vector_llm_concurrency: int = 30
    text2sql_vector_embed_batch_size: int = 128
    text2sql_vector_max_tables: int = 0  # 0 => no limit; useful for safe/quick smoke tests

    # Enum cache bootstrap (optional cold-start; can be heavy)
    enum_cache_bootstrap_on_startup: bool = True
    enum_cache_bootstrap_max_values: int = 200
    enum_cache_bootstrap_max_columns: int = 5000
    enum_cache_bootstrap_concurrency: int = 6
    enum_cache_bootstrap_query_timeout_s: float = 2.0

    # Text2SQL validity bootstrap (PostgreSQL only)
    # - Adds (:Table|:Column).text_to_sql_is_valid + diagnostics in Neo4j.
    # - build_sql_context will filter using COALESCE(flag,true)=true (fail-open).
    text2sql_validity_bootstrap_enabled: bool = True
    text2sql_validity_bootstrap_concurrency: int = 6
    text2sql_validity_bootstrap_query_timeout_s: float = 2.0
    # Confirm phase (only for suspects: row_count_est==0 / null_frac==1)
    text2sql_validity_confirm_table_timeout_s: float = 1.0
    text2sql_validity_confirm_column_timeout_s: float = 1.0
    # 0 => no limit (confirm all row_count_est==0 base tables)
    text2sql_validity_confirm_max_tables: int = 0
    text2sql_validity_confirm_max_columns: int = 600
    # Safety caps for very large graphs (0 => no limit)
    text2sql_validity_bootstrap_max_tables: int = 0
    text2sql_validity_bootstrap_max_columns: int = 0

    # Text2SQL runtime repair (request-time detection + background self-heal)
    # - Lightweight detection runs in request path (throttled).
    # - Heavy repair runs in background with single-flight guard.
    text2sql_runtime_repair_enabled: bool = True
    text2sql_runtime_repair_check_interval_s: float = 30.0
    text2sql_runtime_repair_min_missing_to_trigger: int = 1

    # HyDE retrieval (build_sql_context) - single HyDE only
    # NOTE: Used for ALL retrieval axes (HyDE/question/regex/intent/PRF) candidate prefetch.
    hyde_per_axis_top_k: int = 120
    hyde_union_rerank_top_a: int = 20

    # Table retrieval improvements (build_sql_context)
    table_axis_keep_k: int = 80  # keep top-N per axis after penalty/weight before union
    table_rerank_fetch_k: int = 150  # cap LLM rerank pool (ToolContext default=60; this is used if higher)

    # Axis weights (generic, no domain hardcoding)
    table_axis_weight_question: float = 0.35
    table_axis_weight_hyde: float = 1.0
    table_axis_weight_regex: float = 0.50
    table_axis_weight_intent: float = 0.70

    # PRF (Pseudo-Relevance Feedback)
    table_prf_top_a: int = 20
    table_prf_top_k: int = 300
    table_prf_weight: float = 0.80
    table_prf_max_chars: int = 2500

    # Generic name penalties (avoid overfitting)
    table_penalty_tmp: float = 0.03
    table_penalty_view: float = 0.015
    table_penalty_date: float = 0.01

    # Generic expansions
    table_sibling_boost: float = 0.006
    table_fk_boost: float = 0.010
    table_fk_seed_k: int = 25
    table_fk_max_neighbors: int = 800
    table_fk_enable_2hop: bool = True

    # Similar query injection (if available)
    table_similar_query_boost: float = 0.020

    # build_sql_context - light disambiguation query generation
    # Number of lightweight SQL queries to generate (and preview) to help resolve ambiguities.
    build_sql_context_light_query_count: int = 3

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

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

