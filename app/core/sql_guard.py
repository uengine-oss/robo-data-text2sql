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
        
        # Parse SQL
        try:
            db_type = str(getattr(settings, "target_db_type", "") or "").strip().lower()
            # sqlglot dialect mapping
            if db_type in {"postgres", "postgresql"}:
                dialect = "postgres"
            elif db_type in {"mysql", "mariadb"}:
                dialect = "mysql"
            elif db_type in {"oracle"}:
                dialect = "oracle"
            else:
                # Safe fallback: postgres dialect tends to be strict enough for SELECT-only validation
                dialect = "postgres"
            parsed = sqlglot.parse_one(sql, read=dialect)
        except Exception as e:
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

        # IMPORTANT (policy):
        # Do NOT inject or rewrite LIMIT automatically.
        # Row limiting for preview/execution safety must be handled by the execution layer
        # (timeout, max rows, cursor fetch), not by mutating user SQL.

        return sql, True
    
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
    
    @staticmethod
    def sanitize_identifier(identifier: str) -> str:
        """Sanitize SQL identifier (table/column name)"""
        # Remove dangerous characters
        sanitized = re.sub(r'[^\w.]', '', identifier)
        return sanitized

