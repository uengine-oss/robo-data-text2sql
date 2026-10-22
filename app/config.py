"""Application configuration"""
from pydantic_settings import BaseSettings
from typing import Literal, Dict, Any


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
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
    # Provider (spec 007 T1 / FR-020):
    # - "openai"       : OpenAI 공식 엔드포인트 (기존 동작. 기본값을 바꾸지 않는 이유는
    #                    되돌리기가 환경변수 한 줄이어야 하기 때문이다 — AD-8)
    # - "llm_endpoint" : 사용자가 이미 돌리는 OpenAI 호환 서버의 /v1/embeddings 재사용
    #                    (Ollama·LM Studio·vLLM). 우리 스택 메모리 증가 0 — AD-5 1순위
    # - "local"        : 번들 로컬 모델. A2(분석기 /robo/embed) 이후에 열린다 — 지금은 "준비 중"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    # OpenAI 호환 임베딩 엔드포인트. 비어 있으면 llm_provider_url 로 폴백한다.
    # (사용자가 LLM 을 붙인 그 엔드포인트를 그대로 쓰는 것이 1순위이므로 폴백이 기본이다.)
    embedding_provider_url: str = ""
    # 임베딩 전용 자격증명. 비어 있으면 provider 별로 openai_compatible_api_key →
    # openai_api_key 순으로 폴백한다. 로컬 서버는 키를 검사하지 않는 경우가 많아
    # llm_endpoint 에서는 키가 없어도 진행한다(placeholder 를 넣는다).
    embedding_api_key: str = ""
    # `dimensions` 파라미터를 임베딩 요청에 실을지. MRL 모델(qwen3-embedding 등)과
    # OpenAI text-embedding-3-* 는 이것으로 차원을 맞출 수 있어, **인덱스를 다시 만들지
    # 않고** 제공자를 갈아탈 수 있다. 지원 안 하는 서버는 400 을 내므로 기본은 끔.
    embedding_send_dimensions: bool = False

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

    # Text-to-SQL Table vectorization
    # spec 007 T3 / FR-026 — 기동은 색인과 무관해야 한다.
    # 기본값을 True -> False 로 바꿨다. 색인은 이제 메타데이터 추출의 단계다
    # (`POST /text2sql/index/datasource`). 되돌리려면 이 값을 true 로,
    # 옛 **블로킹** 동작까지 되살리려면 아래 _blocking 도 true 로 둔다.
    text2sql_vectorize_on_startup: bool = False
    text2sql_vectorize_on_startup_blocking: bool = False
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

    # 색인 게이트 (spec 007 T4 / FR-031)
    # `unindexed`/`stale` 에서 데이터소스 자연어 질의를 거부한다.
    # 되돌리기: false 로 두면 색인 상태와 무관하게 이전처럼 진행한다.
    text2sql_index_gate_enabled: bool = True

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

    # ----------------------------------------------------------------------------
    # 기동 sanity check 정책 (spec 007 T3 / SC-001)
    # ----------------------------------------------------------------------------
    # 이전 판은 네 검사 중 **하나라도** 실패하면 RuntimeError 로 서버를 안 띄웠고,
    # `embedding_provider` 기본값이 openai 라서 LLM 을 google 로 쓰는 사용자도
    # OPENAI_API_KEY 가 없으면 기동이 실패했다 (CR-5b 결과 3 — 데스크톱 크래시 루프).
    #
    # "뜨는 것"과 "질의가 되는 것"은 다른 문제다 (AD-4 규칙 3). 그래서 인프라 검사
    # (neo4j·target_db)만 치명으로 남기고, LLM/임베딩 검사는 **경고**로 강등한다.
    # 질의가 되는지는 색인 상태(FR-030)가 판정하고 사용자에게 이유와 함께 알린다.
    #
    # 되돌리기: STARTUP_SANITY_STRICT=true 로 두면 이전과 똑같이 전부 치명이 된다.
    startup_sanity_strict: bool = False

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

