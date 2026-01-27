"""SQL execution with safety and timeout"""
import asyncio
from typing import List, Dict, Any, Optional, Union
import asyncpg

from app.config import settings

# MySQL support
try:
    import aiomysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False


class SQLExecutionError(Exception):
    """Raised when SQL execution fails"""
    pass


class SQLExecutor:
    """Execute SQL queries with safety constraints
    
    Supports both PostgreSQL (asyncpg) and MySQL (aiomysql) connections.
    """
    
    def __init__(self):
        self.timeout = settings.sql_timeout_seconds
        self.max_rows = settings.sql_max_rows
    
    def _is_mysql_connection(self, conn) -> bool:
        """Check if the connection is a MySQL connection"""
        return MYSQL_AVAILABLE and isinstance(conn, aiomysql.Connection)
    
    async def _execute_mysql(self, conn, sql: str, timeout: float) -> Dict[str, Any]:
        """Execute query on MySQL/MindsDB connection"""
        import time
        start_time = time.time()
        
        async with conn.cursor() as cursor:
            await asyncio.wait_for(
                cursor.execute(sql),
                timeout=timeout
            )
            rows = await cursor.fetchall()
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Check row count limit
            if len(rows) > self.max_rows:
                raise SQLExecutionError(
                    f"Query returned too many rows: {len(rows)} (max: {self.max_rows})"
                )
            
            # Extract columns from cursor description
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            data = [list(row) for row in rows]
            
            return {
                "columns": columns,
                "rows": data,
                "row_count": len(rows),
                "execution_time_ms": round(execution_time_ms, 2)
            }
    
    async def _execute_postgres(self, conn, sql: str, timeout: float) -> Dict[str, Any]:
        """Execute query on PostgreSQL connection"""
        import time
        start_time = time.time()
        
        rows = await asyncio.wait_for(
            conn.fetch(sql),
            timeout=timeout
        )
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        # Check row count limit
        if len(rows) > self.max_rows:
            raise SQLExecutionError(
                f"Query returned too many rows: {len(rows)} (max: {self.max_rows})"
            )
        
        # Extract columns and data
        columns = list(rows[0].keys()) if rows else []
        data = [list(row.values()) for row in rows]
        
        return {
            "columns": columns,
            "rows": data,
            "row_count": len(rows),
            "execution_time_ms": round(execution_time_ms, 2)
        }
    
    async def execute_query(
        self,
        conn: Union[asyncpg.Connection, "aiomysql.Connection"],
        sql: str,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute SQL query and return results with metadata.
        
        Supports both PostgreSQL and MySQL connections.
        
        Args:
            conn: asyncpg or aiomysql connection.
            sql: SQL query to execute.
            timeout: Optional timeout in seconds (defaults to instance timeout).
        
        Returns:
            {
                "columns": List[str],
                "rows": List[List[Any]],
                "row_count": int,
                "execution_time_ms": float
            }
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        
        try:
            if self._is_mysql_connection(conn):
                return await self._execute_mysql(conn, sql, effective_timeout)
            else:
                return await self._execute_postgres(conn, sql, effective_timeout)
            
        except asyncio.TimeoutError:
            raise SQLExecutionError(
                f"Query execution timeout after {effective_timeout} seconds (max_sql_seconds limit exceeded)"
            )
        except asyncpg.PostgresError as e:
            raise SQLExecutionError(f"Database error: {str(e)}")
        except Exception as e:
            raise SQLExecutionError(f"Execution failed: {str(e)}")
    
    async def execute_ddl(
        self,
        conn: Union[asyncpg.Connection, "aiomysql.Connection"],
        sql: str,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Execute DDL or utility SQL statement (CREATE, DROP, ANALYZE, etc.).
        
        This method is for statements that don't return data rows,
        such as table creation, index creation, statistics collection, etc.
        
        Supports both PostgreSQL and MySQL connections.
        
        Args:
            conn: asyncpg or aiomysql connection.
            sql: DDL or utility SQL statement to execute.
            timeout: Optional timeout in seconds (defaults to instance timeout).
        
        Raises:
            SQLExecutionError: If execution fails.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        
        try:
            if self._is_mysql_connection(conn):
                async with conn.cursor() as cursor:
                    await asyncio.wait_for(
                        cursor.execute(sql),
                        timeout=effective_timeout
                    )
            else:
                await asyncio.wait_for(
                    conn.execute(sql),
                    timeout=effective_timeout
                )
        except asyncio.TimeoutError:
            raise SQLExecutionError(
                f"DDL execution timeout after {effective_timeout} seconds"
            )
        except asyncpg.PostgresError as e:
            raise SQLExecutionError(f"Database error: {str(e)}")
        except Exception as e:
            raise SQLExecutionError(f"DDL execution failed: {str(e)}")
    
    @staticmethod
    def format_results_for_json(results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results for JSON serialization"""
        # Convert any non-serializable types
        formatted_rows = []
        for row in results["rows"]:
            formatted_row = []
            for value in row:
                if value is None:
                    formatted_row.append(None)
                elif isinstance(value, (str, int, float, bool)):
                    formatted_row.append(value)
                else:
                    # Convert other types to string
                    formatted_row.append(str(value))
            formatted_rows.append(formatted_row)
        
        return {
            **results,
            "rows": formatted_rows
        }

