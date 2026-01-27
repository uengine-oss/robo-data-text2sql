"""SQL validation and security guards"""
import re
from typing import List, Tuple
import sqlglot
from sqlglot import exp

from app.config import settings


# Forbidden SQL keywords (DML/DDL)
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "REPLACE", "MERGE", "EXEC",
    "EXECUTE", "CALL", "PROCEDURE", "FUNCTION"
}

# Dangerous patterns
DANGEROUS_PATTERNS = [
    r";\s*\w+",  # Multiple statements
    r"--",  # SQL comments
    r"/\*.*?\*/",  # Block comments
    r"xp_cmdshell",  # SQL Server command execution
    r"sp_executesql",  # Dynamic SQL
]


class SQLValidationError(Exception):
    """Raised when SQL validation fails"""
    pass


class SQLGuard:
    """SQL validation and security guard"""
    
    def __init__(self):
        self.max_limit = settings.sql_row_limit
        self.max_join_depth = settings.max_join_depth
        self.max_subquery_depth = settings.max_subquery_depth
    
    def validate(self, sql: str, allowed_tables: List[str] = None) -> Tuple[str, bool]:
        """
        Validate SQL for safety and policy compliance.
        Returns: (cleaned_sql, is_valid)
        Raises: SQLValidationError if validation fails
        """
        sql = sql.strip()
        
        # Check for dangerous patterns
        self._check_dangerous_patterns(sql)
        
        # MindsDB 쿼리 감지 (3-part 식별자: datasource.schema.table)
        is_mindsdb_query = self._is_mindsdb_query(sql)
        
        # Parse SQL - MindsDB 쿼리는 mysql 방언 사용 (더 관대함)
        try:
            dialect = "mysql" if is_mindsdb_query else "postgres"
            parsed = sqlglot.parse_one(sql, read=dialect)
        except Exception as e:
            # 파싱 실패 시 기본 검증만 수행하고 원본 SQL 사용
            if is_mindsdb_query:
                # MindsDB 쿼리는 기본 검증만 통과하면 허용
                if not sql.strip().upper().startswith("SELECT"):
                    raise SQLValidationError("Only SELECT statements are allowed")
                return self._ensure_limit_simple(sql), True
            raise SQLValidationError(f"Failed to parse SQL: {str(e)}")
        
        # Check if it's a SELECT statement
        if not isinstance(parsed, exp.Select):
            raise SQLValidationError("Only SELECT statements are allowed")
        
        # Check for forbidden keywords
        self._check_forbidden_keywords(parsed)
        
        # Check join depth
        self._check_join_depth(parsed)
        
        # Check subquery depth
        self._check_subquery_depth(parsed)
        
        # Check allowed tables
        if allowed_tables:
            self._check_allowed_tables(parsed, allowed_tables)
        
        # Ensure LIMIT clause - MindsDB 쿼리는 원본 SQL 유지
        if is_mindsdb_query:
            sql_with_limit = self._ensure_limit_simple(sql)
        else:
            sql_with_limit = self._ensure_limit(sql, parsed)
        
        return sql_with_limit, True
    
    def _is_mindsdb_query(self, sql: str) -> bool:
        """MindsDB federated 쿼리인지 감지 (3-part 식별자 패턴)"""
        # 패턴: datasource.schema.table 또는 datasource.table
        # 예: posgres.rwis."AAA", mysql_full.common_db.customers, posgres.`RWIS`.`AAA`
        # 백틱(`) 또는 큰따옴표(") 또는 일반 식별자 지원
        pattern = r'\bFROM\s+\w+\.[\w`"]+\.[\w`"\']+' 
        return bool(re.search(pattern, sql, re.IGNORECASE))
    
    def _ensure_limit_simple(self, sql: str) -> str:
        """원본 SQL을 유지하면서 LIMIT만 추가 (MindsDB용)"""
        sql_upper = sql.upper()
        if "LIMIT" not in sql_upper:
            return f"{sql.rstrip(';')} LIMIT {self.max_limit}"
        else:
            # 기존 LIMIT 값이 max_limit보다 크면 교체
            match = re.search(r'\bLIMIT\s+(\d+)\b', sql, re.IGNORECASE)
            if match:
                limit_val = int(match.group(1))
                if limit_val > self.max_limit:
                    return re.sub(
                        r'\bLIMIT\s+\d+\b',
                        f'LIMIT {self.max_limit}',
                        sql,
                        flags=re.IGNORECASE
                    )
        return sql
    
    def _check_dangerous_patterns(self, sql: str):
        """Check for dangerous SQL patterns"""
        sql_upper = sql.upper()
        
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                raise SQLValidationError(f"Dangerous pattern detected: {pattern}")
    
    def _check_forbidden_keywords(self, parsed: exp.Expression):
        """Check for forbidden keywords in parsed SQL"""
        for node in parsed.walk():
            if hasattr(node, 'key') and node.key.upper() in FORBIDDEN_KEYWORDS:
                raise SQLValidationError(f"Forbidden keyword: {node.key}")
            
            # Check for specific node types
            if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, 
                               exp.Create, exp.Alter)):
                raise SQLValidationError(f"Forbidden operation: {type(node).__name__}")
    
    def _check_join_depth(self, parsed: exp.Select):
        """Check join depth doesn't exceed limit"""
        joins = list(parsed.find_all(exp.Join))
        if len(joins) > self.max_join_depth:
            raise SQLValidationError(
                f"Too many joins: {len(joins)} (max: {self.max_join_depth})"
            )
    
    def _check_subquery_depth(self, parsed: exp.Expression, depth: int = 0):
        """Recursively check subquery depth"""
        if depth > self.max_subquery_depth:
            raise SQLValidationError(
                f"Subquery depth exceeds limit: {depth} (max: {self.max_subquery_depth})"
            )
        
        for subquery in parsed.find_all(exp.Subquery):
            inner_expr = subquery.this
            if not isinstance(inner_expr, exp.Expression):
                continue
            self._check_subquery_depth(inner_expr, depth + 1)
    
    def _check_allowed_tables(self, parsed: exp.Select, allowed_tables: List[str]):
        """Check that only allowed tables are referenced"""
        tables = {
            table.name.lower()
            for table in parsed.find_all(exp.Table)
        }
        
        allowed_set = {t.lower() for t in allowed_tables}
        unauthorized = tables - allowed_set
        
        if unauthorized:
            raise SQLValidationError(
                f"Unauthorized tables referenced: {', '.join(unauthorized)}"
            )
    
    def _ensure_limit(self, sql: str, parsed: exp.Select) -> str:
        """Ensure LIMIT clause is present and within bounds"""
        limit_node = parsed.find(exp.Limit)
        
        if limit_node:
            # Extract limit value
            limit_expr = limit_node.expression
            if isinstance(limit_expr, exp.Literal):
                try:
                    limit_val = int(limit_expr.this)
                    if limit_val > self.max_limit:
                        # Replace with max limit
                        sql = re.sub(
                            r'\bLIMIT\s+\d+\b',
                            f'LIMIT {self.max_limit}',
                            sql,
                            flags=re.IGNORECASE
                        )
                except (ValueError, AttributeError):
                    pass
        else:
            # Add LIMIT clause
            sql = f"{sql.rstrip(';')} LIMIT {self.max_limit}"
        
        return sql
    
    @staticmethod
    def sanitize_identifier(identifier: str) -> str:
        """Sanitize SQL identifier (table/column name)"""
        # Remove dangerous characters
        sanitized = re.sub(r'[^\w.]', '', identifier)
        return sanitized

