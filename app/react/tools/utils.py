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
    또한 데이터소스명에 하이픈 등 특수문자가 포함된 경우에도 백틱을 추가합니다.
    
    주의: MINDSDB_DATASOURCE_PREFIX 설정과 무관하게 모든 datasource.schema.table 패턴을 처리합니다.
    
    예시:
        FROM robo-postgres.`RWIS`.`AAA`
        → FROM `robo-postgres`.`RWIS`.`AAA`
        
        FROM my-db.RWIS.RDWMSPLY_TB
        → FROM `my-db`.`RWIS`.`RDWMSPLY_TB`
        
        SELECT SUJ_SEQ, SUJ_WNAME
        → SELECT `SUJ_SEQ`, `SUJ_WNAME`
        
        SELECT VAL AS 평균탁도
        → SELECT VAL AS `평균탁도`
    """
    if settings.target_db_type != "mysql":
        return sql
    
    def _needs_backtick(identifier: str) -> bool:
        """식별자에 백틱이 필요한지 확인 (하이픈, 특수문자, 대문자 포함)"""
        if not identifier:
            return False
        for c in identifier:
            # 소문자, 숫자, 언더스코어만 안전한 문자
            if c not in 'abcdefghijklmnopqrstuvwxyz0123456789_':
                return True
        return False
    
    # 0단계: datasource.schema.table 패턴 처리
    # 하이픈/특수문자/대문자가 포함된 경우에만 백틱 추가
    # 모두 소문자이고 특수문자가 없으면 백틱 불필요
    def replace_three_part_table_ref(match):
        """3-part 테이블 참조 처리: datasource.schema.table"""
        ds = match.group(1)  # 데이터소스
        schema = match.group(2)  # 스키마
        table = match.group(3)  # 테이블
        
        # 데이터소스: 하이픈/특수문자/대문자가 있으면 백틱 추가
        if not ds.startswith('`') and _needs_backtick(ds):
            ds = f"`{ds}`"
        
        # 스키마: 대문자/특수문자가 있으면 백틱 추가
        if not schema.startswith('`') and _needs_backtick(schema):
            schema = f"`{schema}`"
        
        # 테이블: 대문자/특수문자가 있으면 백틱 추가
        if not table.startswith('`') and _needs_backtick(table):
            table = f"`{table}`"
        
        return f"{ds}.{schema}.{table}"
    
    # 모든 datasource.schema.table 패턴 처리
    # 백틱으로 감싸진 datasource 또는 일반 datasource 모두 매칭
    # 패턴: (`ds` 또는 ds).schema.table (하이픈 포함 데이터소스도 매칭)
    # \b 대신 (?![A-Za-z0-9_]) 사용 (백틱 뒤에서도 동작)
    pattern = r'(`[^`]+`|[A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*)\.(`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)\.(`[^`]+`|[A-Za-z_][A-Za-z0-9_]*)(?![A-Za-z0-9_])'
    sql = re.sub(pattern, replace_three_part_table_ref, sql)
    
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
    데이터소스명에 하이픈 등 특수문자가 포함된 경우 백틱으로 감싸줍니다.
    
    예시:
        FROM RWIS.RPT_TB2
        → FROM `robo-postgres`.RWIS.RPT_TB2 (mindsdb_datasource_prefix가 "robo-postgres"인 경우)
        
        SELECT * FROM "RWIS"."RPT_TB2"
        → SELECT * FROM `robo-postgres`."RWIS"."RPT_TB2"
        
        FROM `RWIS`.`RPT_TB2`
        → FROM `robo-postgres`.`RWIS`.`RPT_TB2`
    """
    if settings.target_db_type != "mysql":
        return sql
    
    datasource_prefix = settings.mindsdb_datasource_prefix
    if not datasource_prefix:
        return sql
    
    # 데이터소스명에 하이픈/특수문자가 있으면 백틱으로 감싸기
    def _needs_backtick(identifier: str) -> bool:
        """식별자에 백틱이 필요한지 확인"""
        if not identifier:
            return False
        for c in identifier:
            if c not in 'abcdefghijklmnopqrstuvwxyz0123456789_':
                return True
        return False
    
    # 백틱이 필요한 경우 감싸서 사용
    ds_with_backtick = f"`{datasource_prefix}`" if _needs_backtick(datasource_prefix) else datasource_prefix
    
    # 이미 datasource 프리픽스가 있는 경우 스킵 (백틱 있거나 없는 경우 모두 체크)
    escaped_ds = re.escape(datasource_prefix)
    if re.search(rf'(?:`{escaped_ds}`|{escaped_ds})\.\w+\.\w+', sql, re.IGNORECASE):
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
        
        # datasource 프리픽스 추가 (백틱 적용)
        return f"{prefix} {ds_with_backtick}.{quote_char}{schema}{quote_char}.{quote_char}{table}{quote_char}"
    
    def replace_table_ref_unquoted(match):
        """따옴표가 없는 경우: FROM schema.table"""
        prefix = match.group(1)  # FROM, JOIN 등
        schema = match.group(2)
        table = match.group(3)
        
        # 이미 datasource 프리픽스가 있으면 스킵
        if schema.lower() == datasource_prefix.lower():
            return match.group(0)
        
        # datasource 프리픽스 추가 (백틱 적용)
        return f"{prefix} {ds_with_backtick}.{schema}.{table}"
    
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
        
        # datasource 프리픽스 추가 (백틱 적용)
        return f"{prefix} {ds_with_backtick}.{schema}.{table}"
    
    pattern2 = r'(\b(?:FROM|JOIN)\s+)([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)'
    sql = re.sub(pattern2, replace_unquoted_safe, sql, flags=re.IGNORECASE)
    
    return sql

