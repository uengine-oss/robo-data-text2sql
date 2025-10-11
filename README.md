# Graph-based Metadata-Augmented Text2SQL

자연어 질의를 받아 Neo4j 기반 RAG로 안전한 SQL을 생성·실행하고, 표와 차트로 결과를 반환하는 시스템입니다.

## 📺 소개 영상

[![프로젝트 소개 영상](https://img.youtube.com/vi/rWdEnJyfCGQ/0.jpg)](https://youtu.be/rWdEnJyfCGQ)

👉 [YouTube에서 보기](https://youtu.be/rWdEnJyfCGQ)

## 🎯 주요 기능

- **🧠 자연어 → SQL 변환**: LLM 기반 SQL 자동 생성
- **📊 자동 시각화**: Vega-Lite 차트 추천 (Line, Bar, Pie, Scatter)
- **🔍 Neo4j RAG**: 벡터 검색 + 그래프 경로 탐색으로 관련 스키마만 추출
- **🔒 SQL 안전장치**: SELECT-only, 금지 키워드 차단, LIMIT 강제
- **📈 소스 추적**: 어떤 테이블/컬럼이 선택됐는지 provenance 정보 제공
- **💾 피드백 학습**: 사용자 수정 사항을 저장하여 지속적 개선

## 🏗️ 아키텍처

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   사용자    │ ───> │  FastAPI    │ ───> │   Neo4j     │
│  (NL Query) │      │   (API)     │      │  (Schema    │
└─────────────┘      └─────────────┘      │   Graph)    │
                            │              └─────────────┘
                            ↓
                     ┌─────────────┐
                     │   OpenAI    │
                     │ (Embedding  │
                     │  + LLM)     │
                     └─────────────┘
                            │
                            ↓
                     ┌─────────────┐
                     │  PostgreSQL │
                     │  (Target DB)│
                     └─────────────┘
```

## 🚀 빠른 시작

### 1. 필수 조건

- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key
- PostgreSQL 데이터베이스 (읽기 전용 계정)

### 2. 설치

```bash
# 프로젝트 클론 후 디렉토리로 이동
cd neo4j_text2sql

# UV로 의존성 설치
uv sync

# Neo4j 시작
docker-compose up -d

# 환경 변수 설정
cp .env.template .env
# .env 파일을 편집하여 필요한 설정 입력
```

### 3. 환경 변수 설정 (.env)

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# OpenAI
OPENAI_API_KEY=your-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_LLM_MODEL=gpt-4o-mini

# Target Database (PostgreSQL)
TARGET_DB_TYPE=postgresql
TARGET_DB_HOST=localhost
TARGET_DB_PORT=5432
TARGET_DB_NAME=your_database
TARGET_DB_USER=readonly_user
TARGET_DB_PASSWORD=readonly_password
TARGET_DB_SCHEMA=public
```

### 4. 실행

```bash
# API 서버 시작
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API 문서: http://localhost:8000/docs

Neo4j Browser: http://localhost:7474 (neo4j/password123)

## 📖 사용법

### Step 1: 스키마 인제스천

먼저 대상 데이터베이스의 스키마를 Neo4j에 적재합니다:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "db_name": "postgres",
    "schema": "public",
    "clear_existing": true
  }'
```

이 과정에서:
- 테이블/컬럼 메타데이터 추출
- 임베딩 생성 (OpenAI)
- Neo4j 그래프 생성 (노드: Table, Column / 관계: HAS_COLUMN, FK_TO)

### Step 2: 질문하기

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "지난달 카테고리별 매출 Top 10",
    "limit": 100
  }'
```

**응답 예시:**

```json
{
  "sql": "SELECT category, SUM(amount) AS revenue FROM sales.orders WHERE order_date >= ...",
  "table": {
    "columns": ["category", "revenue"],
    "rows": [["Electronics", 125000], ["Clothing", 98000], ...],
    "row_count": 10
  },
  "charts": [
    {
      "title": "매출 by 카테고리",
      "type": "bar",
      "vega_lite": { ... }
    }
  ],
  "provenance": {
    "tables": ["sales.orders", "sales.categories"],
    "columns": ["orders.amount", "categories.category"],
    "vector_matches": [
      {"node": "Table:orders", "score": 0.82}
    ]
  },
  "perf": {
    "embedding_ms": 45,
    "graph_search_ms": 120,
    "llm_ms": 850,
    "sql_ms": 230,
    "total_ms": 1245
  }
}
```

### Step 3: 메타데이터 탐색

```bash
# 테이블 목록
curl "http://localhost:8000/meta/tables?search=order"

# 특정 테이블의 컬럼
curl "http://localhost:8000/meta/tables/orders/columns?schema=public"

# 컬럼 검색
curl "http://localhost:8000/meta/columns?search=email"
```

### Step 4: 피드백 제공

```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt_snapshot_id": "ps_1234_5678",
    "original_sql": "SELECT ...",
    "corrected_sql": "SELECT ... (수정본)",
    "rating": 4,
    "notes": "JOIN 조건이 누락되었음",
    "approved": true
  }'
```

## 🔧 주요 컴포넌트

### 1. 스키마 그래프 (Neo4j)

```cypher
// 테이블 노드
(:Table {name, schema, db, description, vector})

// 컬럼 노드
(:Column {fqn, name, dtype, nullable, description, vector})

// 관계
(Table)-[:HAS_COLUMN]->(Column)
(Column)-[:FK_TO]->(Column)
(Table)-[:FK_TO_TABLE]->(Table)
```

### 2. RAG 파이프라인

1. **쿼리 임베딩**: 자연어 질문을 벡터로 변환
2. **벡터 검색**: Neo4j 벡터 인덱스로 Top-K 테이블/컬럼 검색
3. **경로 탐색**: FK 관계를 따라 조인 가능한 테이블 발견
4. **서브스키마 구성**: 관련 테이블만 추출
5. **프롬프트 생성**: 서브스키마를 텍스트로 변환
6. **SQL 생성**: LLM으로 SQL 생성
7. **검증 & 실행**: 안전장치 통과 후 실행

### 3. SQL 안전장치

- ✅ SELECT만 허용
- ❌ INSERT/UPDATE/DELETE/DDL 차단
- ✅ LIMIT 자동 부여 (기본 1000행)
- ❌ 다중 문장(세미콜론) 차단
- ✅ 조인 깊이 제한 (기본 3단계)
- ✅ 서브쿼리 깊이 제한 (기본 3단계)
- ✅ 허용된 테이블만 사용

## 📂 프로젝트 구조

```
neo4j_text2sql/
├── app/
│   ├── main.py              # FastAPI 앱
│   ├── config.py            # 설정 (환경변수)
│   ├── deps.py              # 의존성 주입
│   ├── core/
│   │   ├── embedding.py     # 임베딩 클라이언트
│   │   ├── graph_search.py  # Neo4j 그래프 검색
│   │   ├── prompt.py        # LangChain SQL 생성
│   │   ├── sql_guard.py     # SQL 검증
│   │   ├── sql_exec.py      # SQL 실행
│   │   └── viz.py           # 시각화 추천
│   ├── ingest/
│   │   ├── ddl_extract.py   # DDL 추출
│   │   └── to_neo4j.py      # Neo4j 적재
│   └── routers/
│       ├── ask.py           # /ask 엔드포인트
│       ├── meta.py          # /meta/* 엔드포인트
│       ├── feedback.py      # /feedback 엔드포인트
│       └── ingest.py        # /ingest 엔드포인트
├── docker-compose.yml       # Neo4j 컨테이너
├── pyproject.toml           # UV 의존성
└── README.md
```

## 🔍 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| POST | `/ask` | 자연어 질의 → SQL 생성 및 실행 |
| POST | `/ingest` | 스키마 인제스천 |
| GET | `/meta/tables` | 테이블 목록 조회 |
| GET | `/meta/tables/{name}/columns` | 테이블 컬럼 조회 |
| GET | `/meta/columns` | 컬럼 검색 |
| POST | `/feedback` | 피드백 제출 |
| GET | `/feedback/stats` | 피드백 통계 |
| GET | `/health` | 헬스체크 |

## ⚙️ 설정 옵션

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `SQL_TIMEOUT_SECONDS` | 30 | SQL 실행 타임아웃 |
| `SQL_ROW_LIMIT` | 1000 | 기본 LIMIT 값 |
| `SQL_MAX_ROWS` | 100000 | 최대 결과 행수 |
| `MAX_JOIN_DEPTH` | 3 | 최대 조인 깊이 |
| `MAX_SUBQUERY_DEPTH` | 3 | 최대 서브쿼리 깊이 |
| `VECTOR_TOP_K` | 10 | 벡터 검색 Top-K |
| `MAX_FK_HOPS` | 3 | FK 경로 탐색 최대 홉 |

## 🧪 개발

### 테스트 (추후 추가)

```bash
uv run pytest tests/
```

### 로깅

```bash
# 로그 레벨 설정
export LOG_LEVEL=DEBUG

# 실행
uv run uvicorn app.main:app --log-level debug
```

## 🛡️ 보안 고려사항

1. **읽기 전용 DB 계정**: 대상 DB는 반드시 읽기 전용 계정 사용
2. **PII 마스킹**: 민감 컬럼은 별도 마스킹 규칙 적용 (추후 구현)
3. **API 인증**: 프로덕션에서는 JWT/OAuth 인증 추가 권장
4. **Rate Limiting**: API 호출 제한 (추후 구현)
5. **SQL Injection**: SQLGlot 파서로 검증 + 파라미터 바인딩

## 📊 성능

- **P95 응답 시간**: 3-6초 (임베딩 + 검색 + LLM + 쿼리 실행)
- **Neo4j 벡터 검색**: ~100ms (10M 테이블/컬럼 기준)
- **LLM 호출**: ~800-1200ms (gpt-4o-mini)
- **SQL 실행**: 쿼리 복잡도에 따라 가변

## 🔮 로드맵

- [ ] Multi-database 지원 (MySQL, Oracle, MS SQL)
- [ ] 실시간 대시보드 스트리밍
- [ ] 자동 인덱스 추천
- [ ] PII 자동 탐지 및 마스킹
- [ ] A/B 테스트 프레임워크
- [ ] 피드백 기반 자동 프롬프트 개선
- [ ] Slack/Teams 봇 통합
- [ ] 멀티테넌시 및 ACL

## 🤝 기여

이슈 및 PR 환영합니다!

## 📝 라이선스

MIT License

## 📧 문의

프로젝트 관련 문의사항은 이슈로 등록해주세요.

---

**Built with** ❤️ **using FastAPI, Neo4j, LangChain, and OpenAI**

