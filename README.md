# Neo4j Text2SQL API

자연어 질문을 받아 Neo4j 기반 RAG(Retrieval-Augmented Generation)로 SQL을 생성하고, 대상 데이터베이스에서 실행한 뒤 표/차트로 결과를 반환하는 백엔드 시스템입니다.

## 주요 특징

- **자연어 → SQL 변환**: 단순 질문은 `/ask`로 한 번에, 복잡한 질문은 `/react` ReAct 에이전트가 다단계 추론으로 해결
- **Multi-Provider LLM**: OpenAI, Google Gemini, OpenAI-compatible 게이트웨이를 통합 지원
- **Neo4j RAG 파이프라인**: HyDE 검색, 멀티 축(question/hyde/regex/intent/PRF) 벡터 검색, FK 그래프 경로 탐색으로 관련 스키마만 추출
- **SQL 안전장치**: SELECT-only, 금지 키워드 차단, LIMIT 강제, 조인/서브쿼리 깊이 제한
- **자동 시각화**: Vega-Lite 기반 차트 자동 추천 (Bar, Line, Pie, Scatter, Area)
- **스키마 유효성 검증**: 기동 시 Text2SQL 유효성 플래그를 연산하여 빈 테이블/null-only 컬럼 필터링
- **캐시 후처리**: 백그라운드 워커가 쿼리 품질 게이트, 유사 쿼리 클러스터링, 값 매핑 추출을 자동 수행
- **이벤트 감시**: SQL 기반 이벤트 룰, CEP 연동, Watch Agent를 통한 대화형 모니터링 설정
- **피드백 학습**: 사용자가 수정한 SQL을 저장하여 지속적 품질 개선

## 아키텍처

```
┌───────────────┐     ┌──────────────────┐     ┌──────────────┐
│    Client     │────>│   FastAPI API    │────>│    Neo4j     │
│  (Vue3 SPA)   │     │   /text2sql/*    │     │ Schema Graph │
└───────────────┘     └──────────────────┘     │ + Vectors    │
                              │                └──────────────┘
                     ┌────────┴────────┐
                     │                 │
              ┌──────▼──────┐  ┌───────▼───────┐
              │  LLM        │  │  Target DB    │
              │  (Gemini /  │  │  (PostgreSQL  │
              │   OpenAI /  │  │   MySQL /     │
              │   Custom)   │  │   Oracle)     │
              └─────────────┘  └───────────────┘
```

**처리 흐름** (ReAct 기준):

1. 사용자 질문 수신
2. `build_sql_context` — 멀티 축 벡터 검색 + FK 그래프 탐색으로 서브스키마 구성
3. ReAct 루프 — SQL 후보 생성 → 검증 → 자동 수정 → 품질 게이트 통과
4. SQL 실행 및 결과 반환 (표 + 차트)
5. 백그라운드 후처리 — 쿼리 캐싱, 유사도 클러스터링, 값 매핑 추출

## 빠른 시작

### 필수 조건

- **Python** 3.13+
- **Docker & Docker Compose** (Neo4j 실행용)
- **API 키**: LLM 제공자에 따라 OpenAI / Google / OpenAI-compatible 키
- **대상 데이터베이스**: PostgreSQL, MySQL, 또는 Oracle (읽기 전용 계정 권장)

### 설치 및 실행

```bash
# 1. 의존성 설치
make install        # uv sync

# 2. Neo4j + 테스트 PostgreSQL 시작
make neo4j          # docker-compose up -d

# 3. 환경 변수 설정
cp env.test.example .env
# .env 파일을 편집하여 API 키 및 DB 접속 정보 입력

# 4. API 서버 실행
make start          # uv run python main.py
```

또는 단계별로 직접 실행:

```bash
uv sync
docker-compose up -d
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**확인 URL**:

| 서비스 | URL |
|--------|-----|
| API 문서 (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/health |
| Neo4j Browser | http://localhost:7474 |

### 테스트 환경 빠른 설정

```bash
make test-setup
# env.test.example → .env 복사, Neo4j/PostgreSQL 기동, 스키마 초기화까지 한 번에 수행
```

## 환경 변수 (.env)

### 필수 설정

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j

# 대상 데이터베이스
TARGET_DB_TYPE=postgresql          # postgresql | mysql | oracle
TARGET_DB_HOST=localhost
TARGET_DB_PORT=5432
TARGET_DB_NAME=your_database
TARGET_DB_USER=readonly_user
TARGET_DB_PASSWORD=readonly_password
TARGET_DB_SCHEMA=public
TARGET_DB_SCHEMAS=public           # 쉼표 구분으로 다중 스키마 지정 가능
TARGET_DB_SSL=disable              # disable | require | verify-ca | verify-full
```

### LLM 설정

```bash
# LLM 제공자 (google | openai | openai_compatible)
LLM_PROVIDER=google
LLM_MODEL=gemini-3-flash-preview

# 경량 LLM (비용 최적화)
LIGHT_LLM_PROVIDER=google
LIGHT_LLM_MODEL=gemini-2.5-flash-lite-preview-09-2025

# OpenAI-compatible 게이트웨이 URL (Ollama 등)
LLM_PROVIDER_URL=
LIGHT_LLM_PROVIDER_URL=

# 임베딩
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small

# API 키
OPENAI_API_KEY=your-openai-key
OPENAI_COMPATIBLE_API_KEY=your-gateway-key
GOOGLE_API_KEY=your-google-key
```

### 주요 튜닝 옵션

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `SQL_TIMEOUT_SECONDS` | 30 | SQL 실행 타임아웃 (초) |
| `SQL_ROW_LIMIT` | 1000 | SELECT 시 기본 LIMIT |
| `SQL_MAX_ROWS` | 100000 | 최대 반환 행수 |
| `MAX_JOIN_DEPTH` | 10 | SQL 조인 깊이 제한 |
| `MAX_SUBQUERY_DEPTH` | 10 | 서브쿼리 깊이 제한 |
| `VECTOR_TOP_K` | 10 | 벡터 검색 Top-K |
| `MAX_FK_HOPS` | 3 | FK 경로 탐색 최대 홉 |
| `REACT_CACHING_DB_TYPE` | oracle | ReAct 캐싱 시 사용할 DB 타입 레이블 |
| `CACHE_POSTPROCESS_WORKER_COUNT` | 1 | 백그라운드 후처리 워커 수 |
| `CACHE_POSTPROCESS_QUERY_JUDGE_ROUNDS` | 2 | 쿼리 품질 심사 라운드 수 |
| `CACHE_POSTPROCESS_QUERY_JUDGE_CONF_THRESHOLD` | 0.90 | 품질 심사 합격 신뢰도 |
| `QUERY_SIMILARITY_CLUSTER_ENABLED` | true | 유사 쿼리 클러스터링 활성화 |
| `TEXT2SQL_VECTORIZE_ON_STARTUP` | true | 기동 시 테이블 벡터 자동 생성 |
| `TEXT2SQL_VALIDITY_BOOTSTRAP_ENABLED` | true | 기동 시 유효성 플래그 연산 |
| `ENUM_CACHE_BOOTSTRAP_ON_STARTUP` | true | 기동 시 Enum 캐시 사전 로딩 |
| `IS_USE_LLM_CACHE` | false | LLM 응답 캐시 활성화 |

> 전체 설정은 `app/config.py`의 `Settings` 클래스를 참조하세요.

## API 엔드포인트

모든 엔드포인트는 `/text2sql` 프리픽스 하에 등록됩니다.

### 질의 (Query)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/text2sql/ask` | 자연어 → SQL 생성 및 실행 (단순 질의) |
| POST | `/text2sql/react` | ReAct 에이전트 스트리밍 (복잡 질의, NDJSON) |

### 메타데이터 (Metadata)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/text2sql/meta/tables` | 테이블 목록 (검색/스키마 필터) |
| GET | `/text2sql/meta/tables/{name}/columns` | 특정 테이블의 컬럼 목록 |
| GET | `/text2sql/meta/columns` | 컬럼 검색 |
| GET | `/text2sql/meta/datasources` | 데이터소스 목록 |
| GET | `/text2sql/meta/datasources/{ds}/schemas` | 스키마 목록 |
| GET | `/text2sql/meta/datasources/{ds}/schemas/{schema}/tables` | 데이터소스/스키마별 테이블 |
| GET | `/text2sql/meta/objecttypes` | 도메인 레이어 ObjectType 목록 |

### 스키마 편집 (Schema Editing)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| PUT | `/text2sql/schema-edit/tables/{name}/description` | 테이블 설명 수정 + 임베딩 재생성 |
| PUT | `/text2sql/schema-edit/tables/{name}/columns/{col}/description` | 컬럼 설명 수정 + 임베딩 재생성 |
| POST | `/text2sql/schema-edit/relationships` | FK 관계 추가 |
| DELETE | `/text2sql/schema-edit/relationships` | FK 관계 삭제 (사용자 추가분만) |
| GET | `/text2sql/schema-edit/relationships/user-added` | 사용자 추가 관계 목록 |
| GET | `/text2sql/schema-edit/relationships/all` | 전체 관계 목록 |

### 피드백 (Feedback)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/text2sql/feedback` | 생성된 SQL에 대한 피드백 제출 |
| GET | `/text2sql/feedback/stats` | 피드백 통계 |

### 이력 (History)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/text2sql/history` | 쿼리 이력 목록 (페이지네이션, 상태 필터, 검색) |
| GET | `/text2sql/history/{query_id}` | 특정 이력 조회 |
| POST | `/text2sql/history` | 이력 생성 |
| DELETE | `/text2sql/history/{query_id}` | 이력 삭제 |
| DELETE | `/text2sql/history` | 전체 이력 삭제 |
| GET | `/text2sql/history/stats/tables` | 테이블 사용 통계 |
| GET | `/text2sql/history/stats/columns` | 컬럼 사용 통계 |

### 캐시 (Cache)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/text2sql/cache/enum/extract/{schema}/{table}/{column}` | Enum 값 추출 |
| POST | `/text2sql/cache/enum/extract-all/{schema}` | 스키마 전체 Enum 추출 |
| GET | `/text2sql/cache/enum/{schema}/{table}/{column}` | Enum 캐시 조회 |
| POST | `/text2sql/cache/mapping` | 값 매핑 생성 |
| GET | `/text2sql/cache/mapping/search` | 값 매핑 검색 |
| POST | `/text2sql/cache/template` | 쿼리 템플릿 저장 |
| GET | `/text2sql/cache/template/search` | 쿼리 템플릿 검색 |
| GET | `/text2sql/cache/similar-query` | 유사 쿼리 검색 |
| GET | `/text2sql/cache/graph/query-history` | 그래프 기반 쿼리 이력 |
| GET | `/text2sql/cache/graph/similar-queries` | 그래프 기반 유사 쿼리 |
| GET | `/text2sql/cache/llm/stats` | LLM 캐시 통계 |
| DELETE | `/text2sql/cache/llm/clear` | LLM 캐시 전체 삭제 |
| DELETE | `/text2sql/cache/llm/invalidate` | 특정 질문 캐시 무효화 |

### Direct SQL

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/text2sql/direct-sql` | SQL 직접 실행 |
| POST | `/text2sql/direct-sql/stream` | SQL 스트리밍 실행 |
| POST | `/text2sql/direct-sql/materialized-view` | Materialized View 생성 |
| POST | `/text2sql/direct-sql/materialized-view/{name}/refresh` | View 새로고침 |
| GET | `/text2sql/direct-sql/materialized-views` | View 목록 |

### 이벤트 (Events)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET/POST/PUT/DELETE | `/text2sql/events/rules/*` | 이벤트 룰 CRUD |
| POST | `/text2sql/events/rules/{id}/run` | 이벤트 룰 수동 실행 |
| POST | `/text2sql/events/rules/{id}/toggle` | 활성/비활성 토글 |
| GET | `/text2sql/events/notifications` | 알림 목록 |
| POST | `/text2sql/events/scheduler/start` | 스케줄러 시작 |
| POST | `/text2sql/events/scheduler/stop` | 스케줄러 중지 |
| POST | `/text2sql/events/chat` | 대화형 이벤트 설정 |
| POST | `/text2sql/events/simple-cep/*` | SimpleCEP 연동 |
| GET | `/text2sql/events/stream/alarms` | SSE 알람 스트림 |

### 이벤트 템플릿 (Event Templates)

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/text2sql/events/templates` | 템플릿 목록 (카테고리 필터) |
| GET | `/text2sql/events/templates/categories` | 카테고리 목록 |
| GET | `/text2sql/events/templates/{id}` | 템플릿 상세 |
| POST | `/text2sql/events/templates/{id}/create-rule` | 템플릿으로 룰 생성 |

### Watch Agent

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/text2sql/watch-agent/chat` | 대화형 모니터링 에이전트 |
| POST | `/text2sql/watch-agent/generate-sql` | 모니터링 SQL 생성 |
| POST | `/text2sql/watch-agent/analyze-availability` | 데이터 가용성 분석 |

### 공통

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/` | 루트 (버전 정보) |
| GET | `/health` | 헬스체크 (Neo4j 연결 + 설정 확인) |

## 핵심 컴포넌트

### Neo4j 스키마 그래프

```cypher
// 노드
(:Table {name, schema, db, description, vector, text_to_sql_vector, text_to_sql_is_valid})
(:Column {fqn, name, dtype, nullable, description, vector, text_to_sql_is_valid})
(:Query {question, sql, summary, vector, verified, canonical_id})
(:ValueMapping {natural_value, db_value, column_fqn, confidence, verified})

// 관계
(Table)-[:HAS_COLUMN]->(Column)
(Column)-[:FK_TO]->(Column)
(Table)-[:FK_TO_TABLE]->(Table)
(Query)-[:USES_TABLE]->(Table)
(Query)-[:SIMILAR_TO {score}]->(Query)
```

### RAG 파이프라인 (`build_sql_context`)

1. **질문 임베딩**: 자연어 질문을 벡터로 변환
2. **HyDE 검색**: 가상 SQL을 생성하여 추가 검색 축 확보
3. **멀티 축 검색**: question / hyde / regex / intent / PRF 5개 축으로 후보 테이블 수집
4. **가중 합산 & 리랭킹**: 축별 가중치 적용 후 LLM 리랭커로 최종 Top-K 선정
5. **FK 그래프 탐색**: FK 관계를 따라 조인 가능한 이웃 테이블 확장
6. **유효성 필터**: `text_to_sql_is_valid` 플래그로 빈 테이블/컬럼 제거
7. **서브스키마 구성**: 관련 테이블+컬럼만 추출하여 프롬프트 생성

### ReAct 에이전트

`/react` 엔드포인트는 복잡한 질문에 대해 다단계 추론을 수행합니다:

- **도구**: `build_sql_context` (스키마 검색), `validate_sql` (SQL 검증/실행)
- **생성기**: SQL 후보 생성, 트리아지, 품질 게이트, 분석 설명 등
- **대화 캡슐**: 이전 대화 컨텍스트를 유지하여 후속 질문 처리
- **스트리밍**: NDJSON 형식으로 단계별 진행 상황 실시간 전송

### SQL 안전장치

- SELECT 문만 허용 (INSERT/UPDATE/DELETE/DDL 차단)
- LIMIT 자동 부여 (기본 1,000행)
- 다중 문장(세미콜론) 차단
- 조인 깊이 제한 (기본 10단계)
- 서브쿼리 깊이 제한 (기본 10단계)
- SQLGlot 파서 기반 구조적 검증

### 캐시 후처리 파이프라인

백그라운드 워커가 ReAct 완료 후 자동으로 수행:

1. **쿼리 품질 게이트**: N회 LLM 심사 (기본 2회, 신뢰도 0.90 이상 모두 통과 필요)
2. **쿼리 벡터 업데이트**: 질문 임베딩을 Neo4j `:Query` 노드에 저장
3. **유사 쿼리 클러스터링**: 벡터 유사도 기반 `:SIMILAR_TO` 관계 생성 및 canonical_id 할당
4. **값 매핑 추출**: 질문 내 자연어 값과 DB 코드 값 매핑을 자동 추출/검증

### 기동 시 부트스트랩

서버 시작 시 다음이 자동으로 실행됩니다:

1. **Sanity Checks**: Neo4j, 대상 DB, LLM API 연결 상태 검증 (실패 시 기동 중단)
2. **Neo4j 스키마/인덱스**: 필요한 인덱스 및 제약조건 자동 생성
3. **테이블 벡터라이즈**: `Table.text_to_sql_vector` 벡터 생성 (LLM 프로파일링 + DB 샘플 조합)
4. **유효성 플래그**: 빈 테이블/null-only 컬럼 탐지 후 `text_to_sql_is_valid` 플래그 설정
5. **Enum 캐시**: 대상 DB에서 Enum 성격 컬럼의 고유값을 사전 로딩

## 프로젝트 구조

```
neo4j-text2sql/
├── app/
│   ├── main.py                     # FastAPI 앱 및 Lifespan 관리
│   ├── config.py                   # Settings (환경 변수 → Pydantic)
│   ├── deps.py                     # 의존성 주입 (Neo4j, DB 커넥션)
│   ├── smart_logger.py             # 구조화된 JSONL 로깅
│   ├── core/
│   │   ├── embedding.py            # 임베딩 클라이언트
│   │   ├── llm_factory.py          # LLM/임베딩 팩토리 (OpenAI, Gemini, Compatible)
│   │   ├── graph_search.py         # Neo4j 벡터/그래프 검색
│   │   ├── prompt.py               # SQL 생성 프롬프트 (LangChain)
│   │   ├── sql_guard.py            # SQL 검증 (안전장치)
│   │   ├── sql_exec.py             # SQL 실행 (타임아웃, 행수 제한)
│   │   ├── sql_transform.py        # SQL 변환 유틸
│   │   ├── viz.py                  # Vega-Lite 시각화 추천
│   │   ├── query_cache.py          # LLM 응답 캐시
│   │   ├── background_jobs.py      # 비동기 백그라운드 워커 큐
│   │   ├── cache_postprocess.py    # 캐시 후처리 (품질 게이트, 클러스터링)
│   │   ├── neo4j_bootstrap.py      # Neo4j 스키마/인덱스 초기화
│   │   ├── enum_cache_bootstrap.py # Enum 캐시 사전 로딩
│   │   ├── text2sql_table_vectorizer.py    # 테이블 벡터 생성
│   │   ├── text2sql_validity_bootstrap.py  # 유효성 플래그 연산
│   │   ├── text2sql_validity_jobs.py       # 유효성 백그라운드 갱신
│   │   ├── simple_cep.py           # Simple CEP 엔진
│   │   ├── cep_client.py           # CEP 클라이언트
│   │   ├── event_poller.py         # 이벤트 폴러
│   │   └── mcp_client.py           # MCP 클라이언트
│   ├── react/
│   │   ├── controller.py           # ReAct 루프 컨트롤러
│   │   ├── state.py                # 세션 상태 관리
│   │   ├── conversation_capsule.py # 대화 컨텍스트 캡슐
│   │   ├── tools/                  # ReAct 도구 (build_sql_context, validate_sql)
│   │   ├── generators/             # LLM 생성기 (SQL 후보, 트리아지, 품질 게이트 등)
│   │   ├── prompts/                # 프롬프트 템플릿
│   │   └── utils/                  # 유틸 (SQL 자동수정, XML 파싱 등)
│   ├── routers/
│   │   ├── ask.py                  # /ask - 단순 질의
│   │   ├── react.py                # /react - ReAct 스트리밍
│   │   ├── meta.py                 # /meta/* - 메타데이터 조회
│   │   ├── schema_edit.py          # /schema-edit/* - 스키마 편집
│   │   ├── feedback.py             # /feedback - 피드백
│   │   ├── history.py              # /history - 이력 관리
│   │   ├── cache.py                # /cache/* - 캐시 관리
│   │   ├── direct_sql.py           # /direct-sql - SQL 직접 실행
│   │   ├── events.py               # /events/* - 이벤트 룰/스케줄러
│   │   ├── event_templates.py      # /events/templates/* - 이벤트 템플릿
│   │   └── watch_agent.py          # /watch-agent/* - 감시 에이전트
│   ├── models/                     # Pydantic 모델
│   ├── sanity_checks/              # 기동 시 외부 의존성 검증
│   └── tests/                      # 테스트
├── scripts/
│   ├── init_schema.py              # Neo4j 스키마 초기화 스크립트
│   ├── init_db.sql                 # 테스트 DB 초기화 SQL
│   └── sample_data.sql             # 테스트 샘플 데이터
├── docker-compose.yml              # Neo4j + PostgreSQL 컨테이너
├── Dockerfile                      # 컨테이너 빌드
├── Makefile                        # 빌드/실행 명령어
├── pyproject.toml                  # UV 의존성 관리
├── main.py                         # 엔트리포인트
├── start-all.sh / stop-all.sh      # 전체 서비스 시작/중지
└── env.test.example                # 환경 변수 템플릿
```

## Docker Compose

`docker-compose.yml`은 두 개의 서비스를 제공합니다:

| 서비스 | 이미지 | 포트 | 용도 |
|--------|--------|------|------|
| neo4j | `neo4j:5.23-community` | 7474 (HTTP), 7687 (Bolt) | 스키마 그래프 + 벡터 인덱스 |
| postgres | `postgres:16-alpine` | 5432 | 테스트 대상 데이터베이스 |

Neo4j는 APOC 플러그인이 활성화되어 있으며, 벡터 인덱스를 지원합니다.

## Makefile 명령어

```bash
make install       # uv sync - 의존성 설치
make neo4j         # Docker Compose로 Neo4j 시작
make init          # Neo4j 스키마 초기화
make start         # API 서버 시작
make stop          # Docker 서비스 중지
make clean         # 컨테이너 + 볼륨 삭제
make test          # pytest 실행
make lint          # ruff 린트
make format        # ruff 포맷
make health        # 헬스체크 호출
make setup         # install + neo4j + init (첫 설치 시)
make test-setup    # 테스트 환경 전체 설정
```

## 개발

### 테스트

```bash
make test
# 또는
uv run pytest app/tests/ -v
```

### 로깅

SmartLogger를 통해 구조화된 JSONL 로그를 생성합니다:

```bash
# 관련 환경 변수
SMART_LOGGER_MIN_LEVEL=ERROR         # 최소 로그 레벨
SMART_LOGGER_CONSOLE_OUTPUT=true     # 콘솔 출력
SMART_LOGGER_FILE_OUTPUT=false       # 파일 출력
SMART_LOGGER_MAIN_LOG_PATH=logs/app_flow.jsonl
SMART_LOGGER_DETAIL_LOG_DIR=logs/details
```

### LangSmith 연동

```bash
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=your-project-name
LANGSMITH_API_KEY=your-langsmith-key
```

## 보안 고려사항

1. **읽기 전용 DB 계정**: 대상 데이터베이스는 반드시 SELECT 권한만 가진 계정 사용
2. **SQL 안전장치**: SQLGlot 파서 기반 구조적 검증으로 DML/DDL 차단
3. **API 인증**: 프로덕션에서는 JWT/OAuth 인증 미들웨어 추가 권장
4. **CORS**: 현재 `allow_origins=["*"]` — 프로덕션 배포 시 도메인 제한 필요
5. **API 키 관리**: `.env` 파일은 절대 버전 관리에 포함하지 않을 것

## 기술 스택

| 분류 | 기술 |
|------|------|
| 웹 프레임워크 | FastAPI, Uvicorn |
| LLM 통합 | LangChain, LangChain-OpenAI, LangChain-Google-GenAI |
| 그래프 DB | Neo4j 5.x (벡터 인덱스 + APOC) |
| 대상 DB 드라이버 | asyncpg (PostgreSQL), aiomysql (MySQL), psycopg (PostgreSQL) |
| SQL 파싱 | SQLGlot |
| 데이터 모델 | Pydantic v2 |
| 패키지 관리 | UV |
| 컨테이너 | Docker, Docker Compose |
| 스트리밍 | SSE-Starlette |

---

**Neo4j Text2SQL API** — FastAPI + Neo4j + LangChain 기반 자연어 SQL 변환 시스템
