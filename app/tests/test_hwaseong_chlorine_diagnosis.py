"""
Diagnostic test script: 화성정수장 잔류염소 쿼리 문제 진단
==========================================================

문제: "화성정수장의 잔류염소 추이. 최근 1년간 일별." 쿼리가 제대로 처리되지 않고
반복(loop)되는 현상.

이 스크립트는 기존 코드를 수정하지 않고, 문제의 원인을 단계별로 진단합니다.
"""

import asyncio
import json
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from neo4j import AsyncGraphDatabase


# ── Configuration ──────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345analyzer")
DATASOURCE_FROM_UI = "postgres"  # what the frontend sends


async def run_diagnosis():
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    print("=" * 80)
    print("진단 1: Neo4j Table.db 값 vs Frontend datasource 값 불일치 확인")
    print("=" * 80)
    
    async with driver.session() as session:
        # Check all distinct db values
        result = await session.run("""
            MATCH (t:Table)
            RETURN DISTINCT t.db AS db, t.datasource AS datasource, count(t) AS cnt
            ORDER BY cnt DESC
        """)
        records = await result.data()
        
        print(f"\nFrontend가 보내는 datasource 값: '{DATASOURCE_FROM_UI}'")
        print(f"Neo4j에 저장된 값들:")
        
        mismatch_found = False
        for r in records:
            db_val = r.get('db', '')
            ds_val = r.get('datasource', '')
            cnt = r.get('cnt', 0)
            
            match_db = (db_val or '').lower() == DATASOURCE_FROM_UI.lower()
            match_ds = (ds_val or '').lower() == DATASOURCE_FROM_UI.lower()
            
            status = "✅ 일치" if (match_db or match_ds) else "❌ 불일치"
            if not (match_db or match_ds):
                mismatch_found = True
            
            print(f"  db='{db_val}', datasource='{ds_val}' -> {cnt}개 테이블 {status}")
        
        if mismatch_found:
            print(f"\n⚠️  문제 발견!")
            print(f"   Neo4j의 Table.db = 'postgresql' 이지만")
            print(f"   Frontend는 datasource = 'postgres'를 보냅니다.")
            print(f"   /ask 엔드포인트의 graph_search.py에서 datasource 필터링 시")
            print(f"   'm.db.lower() == datasource.lower()' 비교가 실패합니다!")
        
        print()
        print("=" * 80)
        print("진단 2: /ask 엔드포인트의 GraphSearcher datasource 필터링 로직 분석")
        print("=" * 80)
        
        print("""
/ask 엔드포인트 (app/routers/ask.py:86-89):
    subschema = await searcher.build_subschema(
        query_embedding,
        datasource=request.datasource,  # "postgres"
    )

GraphSearcher.search_tables (app/core/graph_search.py:104-110):
    ds = (datasource or "").strip()    # "postgres"
    if ds:
        ds_l = ds.lower()             # "postgres"
        matches = [m for m in matches if (m.db or "").strip().lower() == ds_l]
                                       # m.db = "postgresql" → "postgresql" == "postgres" → False!
                                       # 모든 테이블이 필터링되어 빈 리스트 반환!

결과: "No relevant tables found for your question."
""")

        print("=" * 80)
        print("진단 3: /meta 엔드포인트 vs /ask 엔드포인트 불일치")
        print("=" * 80)
        
        print("""
/meta 엔드포인트 (app/routers/meta.py:47):
    WHERE toLower(COALESCE(t.datasource, t.db, '')) = toLower($datasource)
    → t.datasource가 없으면 t.db('postgresql') 사용
    → 'postgresql'.lower() ≠ 'postgres'.lower() → 역시 불일치!
    
    그런데 /meta/datasources가 'postgres'를 반환하는 이유:
    → t.datasource 속성이 별도로 'postgres'로 설정된 경우 가능
    → 또는 다른 데이터 경로를 통해 설정됨
""")

        # Check if t.datasource is set separately
        result = await session.run("""
            MATCH (t:Table)
            WHERE toLower(t.schema) = 'hwaseong'
            RETURN t.name AS name, t.db AS db, t.datasource AS datasource
        """)
        records = await result.data()
        
        for r in records:
            print(f"  Table: {r['name']}")
            print(f"    t.db = '{r.get('db', 'N/A')}'")
            print(f"    t.datasource = '{r.get('datasource', 'N/A')}'")
        
        print()
        print("=" * 80)
        print("진단 4: 벡터 검색에서의 영향 분석 (/react 엔드포인트)")
        print("=" * 80)
        
        print("""
/react 엔드포인트의 build_sql_context 테이블 검색:
(app/react/tools/build_sql_context_parts/neo4j.py)

_neo4j_search_tables_text2sql_vector() 함수:
    → schema_filter만 적용하고 datasource 필터링은 하지 않음!
    → 따라서 /react에서는 hwaseong 테이블을 찾을 수 있음

그러나 문제의 다른 원인들:
""")

        print("=" * 80)
        print("진단 5: 잔류염소 데이터를 올바르게 조회하기 위한 요건")
        print("=" * 80)
        
        # Check the actual 잔류염소 tag data
        result = await session.run("""
            MATCH (t:Table {name: 'tag_master', schema: 'hwaseong'})-[:HAS_COLUMN]->(c:Column)
            WHERE c.name = 'tag_desc'
            RETURN c.name AS name, c.dtype AS dtype, c.description AS description,
                   COALESCE(c.enum_values, 'NONE') AS enum_values
        """)
        records = await result.data()
        
        print(f"\ntag_master.tag_desc 컬럼 정보:")
        for r in records:
            print(f"  name: {r['name']}")
            print(f"  dtype: {r['dtype']}")
            print(f"  description: {r['description']}")
            print(f"  enum_values: {str(r['enum_values'])[:200]}")

        # Check FK relationship
        result = await session.run("""
            MATCH (t1:Table {schema: 'hwaseong'})-[:HAS_COLUMN]->(c1:Column)-[fk:FK_TO_COLUMN]->(c2:Column)<-[:HAS_COLUMN]-(t2:Table {schema: 'hwaseong'})
            RETURN t1.name AS from_table, c1.name AS from_col,
                   t2.name AS to_table, c2.name AS to_col,
                   fk.constraint AS constraint
        """)
        fk_records = await result.data()
        
        print(f"\nhwaseong 스키마 FK 관계:")
        if fk_records:
            for r in fk_records:
                print(f"  {r['from_table']}.{r['from_col']} → {r['to_table']}.{r['to_col']}")
        else:
            print("  ❌ FK 관계가 Neo4j에 없음!")
            print("  ⚠️  ahswgs.tagsn → tag_master.tagsn FK가 등록되지 않았을 수 있습니다.")
        
        print(f"""
올바른 SQL 패턴:
    SELECT 
        SUBSTRING(a.log_time, 1, 8) AS date,
        AVG(CAST(a.val AS NUMERIC)) AS avg_chlorine
    FROM "hwaseong"."ahswgs" a
    JOIN "hwaseong"."tag_master" t ON a.tagsn = t.tagsn
    WHERE t.tag_desc ILIKE '%잔류염소%'
      AND a.log_time >= TO_CHAR(CURRENT_DATE - INTERVAL '1 year', 'YYYYMMDDHHmm')
    GROUP BY SUBSTRING(a.log_time, 1, 8)
    ORDER BY date
    LIMIT 365

핵심 요건:
1. tag_master를 JOIN하여 tag_desc에서 '잔류염소' 검색
2. log_time은 VARCHAR(12) 'YYYYMMDDHHmm' 형식 → SUBSTRING으로 날짜 추출
3. PostgreSQL 문법 사용 (CURRENT_DATE, INTERVAL '1 year')
4. tagsn을 통한 JOIN 필요
""")

        print("=" * 80)
        print("진단 6: tag_master 컬럼의 text_to_sql_is_valid 상태")
        print("=" * 80)
        
        result = await session.run("""
            MATCH (t:Table {schema: 'hwaseong'})-[:HAS_COLUMN]->(c:Column)
            WHERE c.name IN ['tag_desc', 'tagsn', 'log_time', 'val']
            RETURN t.name AS table_name, c.name AS col_name, 
                   COALESCE(c.text_to_sql_is_valid, 'null') AS is_valid,
                   c.description AS description,
                   CASE WHEN c.vector IS NOT NULL AND size(c.vector) > 0 THEN 'YES' ELSE 'NO' END AS has_vec
            ORDER BY t.name, c.name
        """)
        records = await result.data()
        
        for r in records:
            valid_icon = "✅" if r['is_valid'] == True else ("❌" if r['is_valid'] == False else "⚠️")
            print(f"  {valid_icon} {r['table_name']}.{r['col_name']}: valid={r['is_valid']}, vec={r['has_vec']}, desc={r['description'][:60]}")

    await driver.close()
    
    print()
    print("=" * 80)
    print("종합 진단 결과 및 개선 방안")
    print("=" * 80)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ 문제 1: datasource 이름 불일치 (심각도: 높음)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ Neo4j에 저장된 Table.db = 'postgresql'                             │
│ Frontend가 보내는 datasource = 'postgres'                          │
│                                                                     │
│ /ask 엔드포인트의 GraphSearcher.search_tables()와                  │
│ GraphSearcher.search_columns()에서 datasource 필터링 시            │
│ 정확 일치(==) 비교를 하므로 모든 테이블이 필터링됨                 │
│                                                                     │
│ → 결과: "No relevant tables found"                                 │
│                                                                     │
│ 개선 방안:                                                         │
│ 1. graph_search.py의 datasource 비교를 유연하게 수정               │
│    - 'postgres' ≈ 'postgresql' 매핑 추가                           │
│    - 또는 COALESCE(t.datasource, t.db) 사용                       │
│ 2. Neo4j 데이터의 t.db 값을 'postgres'로 통일                     │
│ 3. 또는 t.datasource 속성을 별도로 설정                           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 문제 2: tag 기반 데이터 모델 이해 부족 (심각도: 높음)              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ 화성정수장 데이터는 EAV(Entity-Attribute-Value) 모델:              │
│ - tag_master: 태그 정의 (tag_desc에 '잔류염소' 등)                │
│ - ahswgs: 태그별 시계열 데이터 (tagsn으로 연결)                   │
│                                                                     │
│ LLM이 '잔류염소'를 검색하려면:                                     │
│ 1. tag_master.tag_desc에서 LIKE '%잔류염소%' 검색                  │
│ 2. 해당 tagsn으로 ahswgs를 JOIN                                   │
│ 3. val 컬럼에서 값 추출                                            │
│                                                                     │
│ 하지만 현재 LLM은:                                                │
│ - ahswgs 테이블만 직접 조회하려 함                                │
│ - tag_desc를 통한 간접 검색 패턴을 알지 못함                      │
│                                                                     │
│ 개선 방안:                                                         │
│ 1. tag_master의 analyzed_description/embedding_text에              │
│    EAV 패턴 설명을 포함시킴                                       │
│    예: "잔류염소, 탁도 등의 측정항목을 tag_desc로 검색하고        │
│    tagsn으로 ahswgs와 JOIN하여 시계열 값 조회"                    │
│ 2. Neo4j에 FK 관계 (ahswgs.tagsn → tag_master.tagsn) 등록        │
│ 3. ValueMapping 노드 추가:                                        │
│    '잔류염소' → tag_master.tag_desc 매핑                          │
│ 4. 프롬프트에 EAV 패턴 가이드라인 추가                            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 문제 3: log_time 형식 인식 부족 (심각도: 중간)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ log_time은 VARCHAR(12) 'YYYYMMDDHHmm' 형식                        │
│ - DATE() 함수 직접 적용 불가                                      │
│ - SUBSTRING(log_time, 1, 8)로 일별 그룹핑 필요                    │
│ - 날짜 필터: log_time >= '202502...' 형식 비교                     │
│                                                                     │
│ 개선 방안:                                                         │
│ 1. 컬럼 설명에 형식 정보 명시                                     │
│    "로그 기록 시간 (YYYYMMDDHHmm 형식, VARCHAR)"                  │
│ 2. text_to_sql_embedding_text에 사용 패턴 포함                    │
│    예: "일별 그룹핑 시 SUBSTRING(log_time, 1, 8) 사용"            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ 문제 4: SQL 방언 혼동 (심각도: 중간)                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│ MindsDB MySQL 프로토콜을 통해 PostgreSQL을 쿼리하는 구조에서       │
│ LLM이 MySQL 함수 (CURDATE, INTERVAL without quotes)를 사용         │
│ PostgreSQL 문법 (CURRENT_DATE, INTERVAL '1 year')을 써야 함       │
│                                                                     │
│ 개선 방안:                                                         │
│ - 프롬프트에 "PostgreSQL 방언 사용" 명시 (이미 있으나 강화 필요)  │
│ - sql_autocorrect에서 MySQL→PostgreSQL 변환 로직 강화             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    asyncio.run(run_diagnosis())
