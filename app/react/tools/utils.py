import json
import re
from typing import List, Set

from app.config import settings

# SQL 키워드 목록 (백틱 제외 대상)
SQL_KEYWORDS: Set[str] = {
    'SELECT', 'FROM', 'WHERE', 'AND', 'OR', 'NOT', 'IN', 'IS', 'NULL',
    'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'FULL', 'CROSS', 'ON',
    'ORDER', 'BY', 'ASC', 'DESC', 'GROUP', 'HAVING', 'LIMIT', 'OFFSET',
    'AS', 'DISTINCT', 'ALL', 'UNION', 'INTERSECT', 'EXCEPT',
    'INSERT', 'INTO', 'VALUES', 'UPDATE', 'SET', 'DELETE',
    'CREATE', 'ALTER', 'DROP', 'TABLE', 'INDEX', 'VIEW',
    'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES', 'CONSTRAINT',
    'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'COALESCE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
    'LIKE', 'BETWEEN', 'EXISTS', 'TRUE', 'FALSE',
    'CAST', 'CONVERT', 'DATE', 'TIME', 'TIMESTAMP', 'INTERVAL',
    'WITH', 'RECURSIVE', 'CTE',
}


def quote_uppercase_identifiers(sql: str) -> str:
    """
    MindsDB SQL에서 대문자 스키마/테이블명/컬럼명/별칭에 백틱을 추가합니다.
    
    예시:
        FROM posgres.RWIS.RDWMSPLY_TB
        → FROM posgres.`RWIS`.`RDWMSPLY_TB`
        
        SELECT SUJ_SEQ, SUJ_WNAME
        → SELECT `SUJ_SEQ`, `SUJ_WNAME`
        
        SELECT VAL AS 평균탁도
        → SELECT VAL AS `평균탁도`
    """
    if settings.target_db_type != "mysql":
        return sql
    
    datasource_prefix = settings.mindsdb_datasource_prefix
    
    # 1단계: datasource.SCHEMA.TABLE 패턴 처리
    if datasource_prefix:
        def replace_table_ref(match):
            ds = match.group(1)
            schema = match.group(2)
            table = match.group(3)
            
            if '`' in schema or '`' in table:
                return match.group(0)
            
            has_upper_schema = any(c.isupper() for c in schema)
            has_upper_table = any(c.isupper() for c in table)
            
            if has_upper_schema:
                schema = f"`{schema}`"
            if has_upper_table:
                table = f"`{table}`"
            
            return f"{ds}.{schema}.{table}"
        
        pattern = rf'\b({re.escape(datasource_prefix)})\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b'
        sql = re.sub(pattern, replace_table_ref, sql)
    
    # 2단계: AS 별칭 처리 (한글, 대문자, 특수문자 포함 별칭)
    # AS 뒤에 오는 식별자가 백틱 없이 한글이나 대문자를 포함하면 백틱 추가
    def replace_alias(match):
        as_keyword = match.group(1)  # AS
        alias = match.group(2)       # 별칭
        
        # 이미 백틱으로 감싸진 경우 스킵
        if alias.startswith('`'):
            return match.group(0)
        
        # 한글 포함 여부 확인
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in alias)
        # 대문자 포함 여부 확인
        has_upper = any(c.isupper() for c in alias)
        
        if has_korean or has_upper:
            return f"{as_keyword}`{alias}`"
        
        return match.group(0)
    
    # AS 별칭 패턴: AS 뒤에 공백과 식별자 (한글, 영문, 숫자, 언더스코어)
    # (?i) - 대소문자 무시
    alias_pattern = r'(?i)\b(AS\s+)([^\s,`\'"()]+)'
    sql = re.sub(alias_pattern, replace_alias, sql)
    
    # 3단계: 대문자/한글 식별자(컬럼명 등)에 백틱 추가
    def replace_identifier(match):
        identifier = match.group(0)
        
        # SQL 키워드인 경우 스킵
        if identifier.upper() in SQL_KEYWORDS:
            return identifier
        
        # 한글 포함 여부 확인
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in identifier)
        # 대문자 포함 여부 확인
        has_upper = any(c.isupper() for c in identifier)
        
        # 한글이나 대문자가 포함되어 있고, 아직 백틱이 없으면 추가
        if (has_korean or has_upper) and not identifier.startswith('`'):
            return f"`{identifier}`"
        
        return identifier
    
    # 백틱으로 감싸지 않은 식별자 패턴 (한글, 대문자 포함)
    # 한글 문자 범위: \uac00-\ud7a3
    identifier_pattern = r'(?<!`)\b([A-Z가-힣][A-Z0-9_가-힣]*)\b(?!`)'
    sql = re.sub(identifier_pattern, replace_identifier, sql)
    
    return sql


def parse_keyword_list_string(keyword_list_string: str) -> List[str]:
    """
    JSON 배열 혹은 콤마 구분 문자열을 파싱하여 키워드 리스트를 반환한다.
    run_mocked_tools.py 와 공유하기 위해 별도 모듈로 분리했다.
    """
    try:
        keywords = json.loads(keyword_list_string)
        if not isinstance(keywords, list):
            keywords = [keyword_list_string]
    except json.JSONDecodeError:
        keywords = [kw.strip() for kw in keyword_list_string.split(",")]
    return [kw for kw in keywords if kw]


def to_cdata(value: str) -> str:
    """XML CDATA 블록으로 감싼 문자열을 반환한다."""
    return f"<![CDATA[{value}]]>"


def add_datasource_prefix(sql: str) -> str:
    """
    MindsDB용 SQL에서 schema.table 패턴을 datasource.schema.table로 변환합니다.
    
    LLM이 생성한 SQL이 schema.table 형식일 때, MindsDB가 요구하는 
    datasource.schema.table 형식으로 변환합니다.
    
    예시:
        FROM RWIS.RPT_TB2
        → FROM posgres.RWIS.RPT_TB2 (mindsdb_datasource_prefix가 "posgres"인 경우)
        
        SELECT * FROM "RWIS"."RPT_TB2"
        → SELECT * FROM posgres."RWIS"."RPT_TB2"
        
        FROM `RWIS`.`RPT_TB2`
        → FROM posgres.`RWIS`.`RPT_TB2`
    """
    if settings.target_db_type != "mysql":
        return sql
    
    datasource_prefix = settings.mindsdb_datasource_prefix
    if not datasource_prefix:
        return sql
    
    # 이미 datasource 프리픽스가 있는 경우 스킵
    if re.search(rf'\b{re.escape(datasource_prefix)}\.\w+\.\w+', sql, re.IGNORECASE):
        return sql
    
    # FROM/JOIN 절에서 schema.table 패턴 찾기
    def replace_table_ref_quoted(match):
        """따옴표가 있는 경우: FROM "schema"."table" 또는 FROM `schema`.`table`"""
        prefix = match.group(1)  # FROM, JOIN 등
        quote_char = match.group(2)  # " 또는 `
        schema = match.group(3)
        table = match.group(4)
        
        # 이미 datasource 프리픽스가 있으면 스킵
        if schema.lower() == datasource_prefix.lower():
            return match.group(0)
        
        # datasource 프리픽스 추가
        return f"{prefix} {datasource_prefix}.{quote_char}{schema}{quote_char}.{quote_char}{table}{quote_char}"
    
    def replace_table_ref_unquoted(match):
        """따옴표가 없는 경우: FROM schema.table"""
        prefix = match.group(1)  # FROM, JOIN 등
        schema = match.group(2)
        table = match.group(3)
        
        # 이미 datasource 프리픽스가 있으면 스킵
        if schema.lower() == datasource_prefix.lower():
            return match.group(0)
        
        # datasource 프리픽스 추가
        return f"{prefix} {datasource_prefix}.{schema}.{table}"
    
    # 패턴 1: FROM "schema"."table" 또는 FROM `schema`.`table` (둘 다 같은 따옴표)
    pattern1 = r'(\b(?:FROM|JOIN)\s+)(["`])([A-Za-z_][A-Za-z0-9_]*)\2\.\2([A-Za-z_][A-Za-z0-9_]*)\2'
    sql = re.sub(pattern1, replace_table_ref_quoted, sql, flags=re.IGNORECASE)
    
    # 패턴 2: FROM schema.table (따옴표 없음)
    # 이미 datasource 프리픽스가 있는 경우는 스킵
    def replace_unquoted_safe(match):
        prefix = match.group(1)
        schema = match.group(2)
        table = match.group(3)
        
        # 이미 datasource 프리픽스가 있으면 스킵
        if schema.lower() == datasource_prefix.lower():
            return match.group(0)
        
        # datasource 프리픽스 추가
        return f"{prefix} {datasource_prefix}.{schema}.{table}"
    
    pattern2 = r'(\b(?:FROM|JOIN)\s+)([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)'
    sql = re.sub(pattern2, replace_unquoted_safe, sql, flags=re.IGNORECASE)
    
    return sql

