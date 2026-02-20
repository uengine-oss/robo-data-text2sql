# 전체 설정 가이드 (Setup Guide)

이 문서는 robo-data-text2sql 프로젝트의 전체 설정 과정을 단계별로 설명합니다.

## 목차

1. [Docker Compose 시작](#1-docker-compose-시작)
2. [PostgreSQL 데이터베이스 설정](#2-postgresql-데이터베이스-설정)
3. [SQL 샘플 데이터 설정](#3-sql-샘플-데이터-설정)
4. [MindsDB 데이터소스 등록](#4-mindsdb-데이터소스-등록)
5. [애플리케이션 실행](#5-애플리케이션-실행)
6. [문제 해결](#6-문제-해결)

---

## 1. Docker Compose 시작

### 1.1 서비스 시작

```bash
docker-compose up -d
```

### 1.2 서비스 상태 확인

```bash
docker-compose ps
```

다음 서비스들이 실행 중이어야 합니다:
- `neo4j_text2sql` (포트: 7474, 7687)
- `postgres_text2sql` (포트: 5432)
- `mindsdb_text2sql` (포트: 47334, 47335)

### 1.3 서비스 로그 확인

```bash
# 모든 서비스 로그
docker-compose logs -f

# 특정 서비스 로그
docker-compose logs -f mindsdb
docker-compose logs -f postgres
docker-compose logs -f neo4j
```

### 1.4 서비스가 정상적으로 시작될 때까지 대기

MindsDB는 시작하는데 시간이 걸릴 수 있습니다 (보통 1-2분).

```bash
# MindsDB 상태 확인
curl http://localhost:47334/api/status

# 정상 응답 예시:
# {"mindsdb_version":"25.14.1","environment":"local",...}
```

---

## 2. PostgreSQL 데이터베이스 설정

### 2.1 PostgreSQL 연결 확인

```bash
docker exec -it postgres_text2sql psql -U testuser -d testdb
```

또는 Windows에서:

```bash
docker exec postgres_text2sql psql -U testuser -d testdb
```

### 2.2 기본 스키마 확인

```bash
docker exec postgres_text2sql psql -U testuser -d testdb -c "\dn"
```

---

## 3. SQL 샘플 데이터 설정

### 3.1 SQL 샘플 파일 위치

SQL 샘플 파일들은 `scripts/sql_samples_checked_20260113/` 디렉토리에 있습니다:

```
scripts/sql_samples_checked_20260113/
├── README.md
├── real-scheme/
│   ├── RWIS_postgres_ddl_UPPER.sql          # RWIS 스키마 DDL
│   └── sp/                                   # Stored Procedures
│       └── RWIS/
│           ├── FN_CALC_RDF01HH_TRNS_TB.sql
│           ├── PRC_INSERT_RDD01DD_TB2.sql
│           └── ...
├── rwis_sql/                                 # RWIS 데이터 INSERT
│   ├── RDD01DD_TB_INSERT.sql
│   ├── RDF01HH_TB_INSERT.sql
│   ├── RDR01MI_HQJNDB_INSERT.sql
│   └── ...
└── rditag_sql/                               # RDITAG 데이터 INSERT
    └── RDITAG_TB_INSERT_202510211124.sql
```

### 3.2 RWIS 스키마 초기화

**1단계: DDL 실행 (스키마 및 테이블 생성)**

```bash
docker exec -i postgres_text2sql psql -U testuser -d testdb < scripts/sql_samples_checked_20260113/real-scheme/RWIS_postgres_ddl_UPPER.sql
```

**2단계: Stored Procedures 생성 (선택사항)**

```bash
# 각 SP 파일을 순차적으로 실행
docker exec -i postgres_text2sql psql -U testuser -d testdb < scripts/sql_samples_checked_20260113/real-scheme/sp/RWIS/FN_CALC_RDF01HH_TRNS_TB.sql
docker exec -i postgres_text2sql psql -U testuser -d testdb < scripts/sql_samples_checked_20260113/real-scheme/sp/RWIS/PRC_INSERT_RDD01DD_TB2.sql
# ... 나머지 SP 파일들
```

**3단계: 데이터 INSERT (rwis_sql 폴더의 파일들)**

큰 파일들은 시간이 오래 걸릴 수 있습니다:

```bash
# 작은 테이블부터 시작
docker exec -i postgres_text2sql psql -U testuser -d testdb < scripts/sql_samples_checked_20260113/rwis_sql/RDIDBINFO_TB_INSERT.sql
docker exec -i postgres_text2sql psql -U testuser -d testdb < scripts/sql_samples_checked_20260113/rwis_sql/RDICOMSTAT_TB_INSERT.sql

# 큰 파일들 (시간이 오래 걸림)
docker exec -i postgres_text2sql psql -U testuser -d testdb < scripts/sql_samples_checked_20260113/rwis_sql/RDR01MI_HQJNDB_INSERT.sql
```

**4단계: RDITAG 데이터 INSERT**

```bash
docker exec -i postgres_text2sql psql -U testuser -d testdb < scripts/sql_samples_checked_20260113/rditag_sql/RDITAG_TB_INSERT_202510211124.sql
```

### 3.3 데이터 확인

```bash
docker exec postgres_text2sql psql -U testuser -d testdb -c "SELECT COUNT(*) FROM rwis.RDD01DD_TB;"
docker exec postgres_text2sql psql -U testuser -d testdb -c "\dt rwis.*"
```

---

## 4. MindsDB 데이터소스 등록

### 4.1 MindsDB 상태 확인

```bash
curl http://localhost:47334/api/status
```

### 4.2 PostgreSQL 데이터소스 등록

```bash
curl -X POST http://localhost:47334/api/sql/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "CREATE DATABASE postgres_datasource WITH ENGINE = '\''postgres'\'', PARAMETERS = { '\''host'\'': '\''postgres'\'', '\''port'\'': 5432, '\''user'\'': '\''testuser'\'', '\''password'\'': '\''testpass123'\'', '\''database'\'': '\''testdb'\'' };"
  }'
```

**참고:** Docker 네트워크 내에서 `postgres`는 컨테이너 이름으로 접근 가능합니다.

### 4.3 등록된 데이터소스 확인

```bash
curl -X POST http://localhost:47334/api/sql/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SHOW DATABASES;"}'
```

응답 예시:
```json
{
  "type": "table",
  "data": [
    ["information_schema"],
    ["log"],
    ["mindsdb"],
    ["files"],
    ["postgres_datasource"]
  ],
  "column_names": ["Database"]
}
```

### 4.4 테이블 목록 확인

```bash
curl -X POST http://localhost:47334/api/sql/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SHOW TABLES FROM postgres_datasource;"}'
```

### 4.5 스키마별 테이블 확인

```bash
curl -X POST http://localhost:47334/api/sql/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM `postgres_datasource` (SELECT table_schema, table_name FROM information_schema.tables WHERE table_schema = '\''rwis'\'' ORDER BY table_name);"}'
```

### 4.6 MindsDB를 통한 쿼리 테스트

```bash
curl -X POST http://localhost:47334/api/sql/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "SELECT * FROM `postgres_datasource` (SELECT COUNT(*) as count FROM rwis.RDD01DD_TB);"
  }'
```

---

## 5. 애플리케이션 실행

### 5.1 환경 변수 설정

`.env` 파일 생성 또는 환경 변수 설정:

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
NEO4J_DATABASE=neo4j

# MindsDB (MySQL protocol endpoint)
TARGET_DB_TYPE=mysql
TARGET_DB_HOST=localhost
TARGET_DB_PORT=47335
TARGET_DB_NAME=mindsdb
TARGET_DB_USER=mindsdb
TARGET_DB_PASSWORD=

# LLM 설정
LLM_PROVIDER=google
LLM_MODEL=gemini-3-flash-preview
```

### 5.2 애플리케이션 시작

```bash
uv run python main.py
```

또는:

```bash
uvicorn app.main:app --reload --port 8000
```

### 5.3 API 테스트

```bash
# Health check
curl http://localhost:8000/health

# Direct SQL 실행
curl -X POST http://localhost:8000/api/direct-sql \
  -H "Content-Type: application/json" \
  -d '{
    "sql": "SELECT COUNT(*) FROM rwis.RDD01DD_TB",
    "datasource": "postgres_datasource"
  }'
```

---

## 6. 문제 해결

### 6.1 MindsDB 연결 풀 오류 (QueuePool)

#### 증상

```
[ERROR] Execution failed: (1149, 'QueuePool limit of size 30 overflow 200 reached, connection timed out, timeout 30.00')
```

또는 애플리케이션 시작 시:

```
TimeoutError: MindsDB endpoint sanity check failed
```

#### 원인

- **이전 QueuePool 에러**: MindsDB 내부의 SQLAlchemy 연결 풀이 가득 참 (`size 30, overflow 200`)
- **TimeoutError**: MindsDB 내부 풀이 고갈되어 새 연결을 받을 수 없는 상태
- 즉, **MindsDB 자체가 과부하 상태**

#### 해결 방법

**1단계: MindsDB 컨테이너 재시작**

```bash
docker restart mindsdb_text2sql
```

또는:

```bash
docker-compose restart mindsdb
```

**2단계: MindsDB가 완전히 시작될 때까지 대기**

```bash
# MindsDB 상태 확인 (최대 30번 시도, 5초 간격)
for i in {1..30}; do
  sleep 5
  status=$(curl -s http://localhost:47334/api/status 2>/dev/null)
  if echo "$status" | grep -q "mindsdb_version"; then
    echo "MindsDB is ready! (attempt $i)"
    echo "$status"
    break
  fi
  echo "Attempt $i - not ready yet..."
done
```

**3단계: MySQL 프로토콜 연결 테스트**

```bash
# Python으로 테스트
uv run python -c "
import asyncio
async def test():
    import aiomysql
    conn = await aiomysql.connect(
        host='localhost', port=47335, 
        user='mindsdb', password='', 
        db='mindsdb', autocommit=True
    )
    async with conn.cursor() as cur:
        await cur.execute('SELECT 1')
        row = await cur.fetchone()
        print(f'MySQL protocol OK: {row}')
    conn.close()
asyncio.run(test())
"
```

**4단계: 애플리케이션 재시작**

```bash
uv run python main.py
```

#### 예방 방법

1. **동시 요청 수 제한**: 너무 많은 동시 요청을 보내지 않도록 주의
2. **연결 풀 모니터링**: MindsDB 로그를 주기적으로 확인
3. **자동 재시작 설정**: `docker-compose.yml`에서 `restart: unless-stopped` 설정 확인

### 6.2 MindsDB 연결 실패

```bash
# MindsDB 컨테이너 재시작
docker-compose restart mindsdb

# 로그 확인
docker-compose logs mindsdb

# 컨테이너 상태 확인
docker ps | grep mindsdb
```

### 6.3 PostgreSQL 연결 실패

```bash
# PostgreSQL 컨테이너 재시작
docker-compose restart postgres

# 연결 테스트
docker exec postgres_text2sql psql -U testuser -d testdb -c "SELECT 1;"
```

### 6.4 데이터소스 재등록

```bash
# 기존 데이터소스 삭제
curl -X POST http://localhost:47334/api/sql/query \
  -H "Content-Type: application/json" \
  -d '{"query": "DROP DATABASE postgres_datasource;"}'

# 다시 등록 (4.2 단계 참조)
```

### 6.5 Neo4j 연결 실패

```bash
# Neo4j 컨테이너 재시작
docker-compose restart neo4j

# 연결 테스트
curl http://localhost:7474
```

### 6.6 포트 충돌

다른 서비스가 이미 포트를 사용 중인 경우:

```bash
# 포트 사용 확인 (Windows)
netstat -ano | findstr :47334
netstat -ano | findstr :47335
netstat -ano | findstr :5432

# 포트 사용 확인 (Linux/Mac)
lsof -i :47334
lsof -i :47335
lsof -i :5432
```

`docker-compose.yml`에서 포트를 변경하거나, 충돌하는 서비스를 중지하세요.

---

## 7. 유용한 명령어 모음

### 7.1 컨테이너 관리

```bash
# 모든 컨테이너 상태 확인
docker ps

# 특정 컨테이너 쉘 접속
docker exec -it postgres_text2sql bash
docker exec -it mindsdb_text2sql bash
docker exec -it neo4j_text2sql bash

# 컨테이너 로그 실시간 확인
docker-compose logs -f mindsdb
```

### 7.2 데이터베이스 관리

```bash
# PostgreSQL 백업
docker exec postgres_text2sql pg_dump -U testuser testdb > backup.sql

# PostgreSQL 복원
docker exec -i postgres_text2sql psql -U testuser testdb < backup.sql

# MindsDB MySQL 프로토콜로 직접 연결
mysql -h localhost -P 47335 -u mindsdb
```

### 7.3 서비스 중지 및 재시작

```bash
# 서비스 중지
docker-compose down

# 볼륨 포함 완전 삭제 (데이터 초기화)
docker-compose down -v

# 재시작
docker-compose up -d
```

### 7.4 데이터 확인

```bash
# PostgreSQL 스키마 목록
docker exec postgres_text2sql psql -U testuser -d testdb -c "\dn"

# PostgreSQL 테이블 목록
docker exec postgres_text2sql psql -U testuser -d testdb -c "\dt rwis.*"

# MindsDB 데이터소스 목록
curl -X POST http://localhost:47334/api/sql/query \
  -H "Content-Type: application/json" \
  -d '{"query": "SHOW DATABASES;"}'
```

---

## 8. 체크리스트

설정 완료 후 다음 항목들을 확인하세요:

- [ ] Docker Compose 서비스 3개 모두 실행 중
- [ ] MindsDB HTTP API 응답 정상 (`http://localhost:47334/api/status`)
- [ ] MindsDB MySQL 프로토콜 연결 가능 (포트 47335)
- [ ] PostgreSQL 연결 가능 (포트 5432)
- [ ] MindsDB에 `postgres_datasource` 등록됨
- [ ] MindsDB에서 PostgreSQL 테이블 조회 가능
- [ ] 애플리케이션 시작 시 sanity check 통과
- [ ] API 엔드포인트 정상 동작 (`/health`, `/api/direct-sql`)

---

## 9. 참고 자료

- [MindsDB 공식 문서](https://docs.mindsdb.com/)
- [PostgreSQL 공식 문서](https://www.postgresql.org/docs/)
- [Neo4j 공식 문서](https://neo4j.com/docs/)

---

**마지막 업데이트:** 2025-01-13

