"""SQL execution with safety and timeout"""
import asyncio
from typing import List, Dict, Any, Optional
import asyncpg

from app.config import settings


class SQLExecutionError(Exception):
    """Raised when SQL execution fails"""
    pass


class SQLExecutor:
    """Execute SQL queries with safety constraints"""
    
    def __init__(self):
        self.timeout = settings.sql_timeout_seconds
        self.max_rows = settings.sql_max_rows
    
    async def execute_query(
        self,
        conn: asyncpg.Connection,
        sql: str,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute SQL query and return results with metadata.
        
        Args:
            conn: asyncpg connection.
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
        import time
        start_time = time.time()
        effective_timeout = timeout if timeout is not None else self.timeout
        
        try:
            # Execute with timeout
            rows = await asyncio.wait_for(
                conn.fetch(sql),
                timeout=effective_timeout
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
        conn: asyncpg.Connection,
        sql: str,
        *,
        timeout: Optional[float] = None,
    ) -> None:
        """
        Execute DDL or utility SQL statement (CREATE, DROP, ANALYZE, etc.).
        
        This method is for statements that don't return data rows,
        such as table creation, index creation, statistics collection, etc.
        
        Args:
            conn: asyncpg connection.
            sql: DDL or utility SQL statement to execute.
            timeout: Optional timeout in seconds (defaults to instance timeout).
        
        Raises:
            SQLExecutionError: If execution fails.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        
        try:
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

