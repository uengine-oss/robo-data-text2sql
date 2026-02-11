"""SQL execution with safety and timeout.

Decision alignment (merge_plan/09_decisions_to_finalize.md):
- D6: Do NOT inject/modify LIMIT in user SQL.
- D11: Prevent result overflow by reading up to sql_max_rows using cursor fetch (fetchmany/iteration),
  and explicitly mark truncation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import asyncpg

# Optional MySQL driver (used for MindsDB MySQL endpoint)
try:
    import aiomysql  # type: ignore
except Exception:  # pragma: no cover
    aiomysql = None  # type: ignore[assignment]

from app.config import settings


class SQLExecutionError(Exception):
    """Raised when SQL execution fails"""
    pass


class SQLExecutor:
    """Execute SQL queries with safety constraints"""
    
    def __init__(self):
        self.timeout = settings.sql_timeout_seconds
        self.max_rows = settings.sql_max_rows

    def _is_mysql_connection(self, conn: Any) -> bool:
        if aiomysql is None:
            return False
        try:
            return isinstance(conn, aiomysql.Connection)  # type: ignore[attr-defined]
        except Exception:
            return False

    def _remaining_timeout(self, *, started: float, timeout: float) -> float:
        # Keep small buffer so next await doesn't start with near-zero budget.
        elapsed = time.perf_counter() - started
        return max(0.001, float(timeout) - float(elapsed) - 0.05)

    def _should_reconnect_mysql(self, exc: Exception) -> bool:
        """
        MindsDB(MySQL endpoint) 연결이 끊겼을 때 흔히 나오는 에러 문자열들을 감지한다.
        - 현상: usecase_2 로그에서 (0, 'Not connected')가 반복되며 스텝 폭증.
        - 대응: 동일 쿼리를 새 연결로 1회 재시도(베스트에포트).
        """
        msg = str(exc or "").lower()
        needles = [
            "not connected",
            "server has gone away",
            "lost connection",
            "connection reset",
            "broken pipe",
            "connection aborted",
            "connection closed",
        ]
        return any(n in msg for n in needles)

    async def _open_mysql_connection(self) -> Any:
        """
        (Best-effort) Open a fresh aiomysql connection using current settings.
        Intended as a fallback when an existing request-scoped connection becomes stale.
        """
        if aiomysql is None:  # pragma: no cover
            raise RuntimeError("aiomysql is required for mysql execution but not installed")
        return await aiomysql.connect(
            host=settings.target_db_host,
            port=int(settings.target_db_port),
            user=settings.target_db_user,
            password=settings.target_db_password,
            db=settings.target_db_name,
            autocommit=True,
        )
    
    async def execute_query(
        self,
        conn: Union[asyncpg.Connection, Any],
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
        started_perf = time.perf_counter()
        started_wall = time.time()
        effective_timeout = float(timeout if timeout is not None else self.timeout)

        # Fetch up to max_rows + 1 to detect truncation without loading everything.
        cap = max(1, int(self.max_rows)) + 1

        try:
            if self._is_mysql_connection(conn):
                return await self._execute_mysql_with_cap(
                    conn, sql, timeout_s=effective_timeout, cap_rows=cap, started_perf=started_perf, started_wall=started_wall
                )
            return await self._execute_postgres_with_cap(
                conn, sql, timeout_s=effective_timeout, cap_rows=cap, started_perf=started_perf, started_wall=started_wall
            )
        except asyncio.TimeoutError:
            raise SQLExecutionError(
                f"Query execution timeout after {effective_timeout} seconds (max_sql_seconds limit exceeded)"
            )
        except asyncpg.PostgresError as e:
            raise SQLExecutionError(f"Database error: {str(e)}")
        except Exception as e:
            raise SQLExecutionError(f"Execution failed: {str(e)}")

    async def preview_query(
        self,
        conn: Union[asyncpg.Connection, Any],
        sql: str,
        *,
        row_limit: int,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute a SELECT query in preview mode WITHOUT mutating the SQL.
        Uses server-side cursor fetch to retrieve up to row_limit rows.

        Returns the same structure as execute_query, but row_count is the number of fetched rows.
        """
        started_perf = time.perf_counter()
        started_wall = time.time()
        effective_timeout = float(timeout if timeout is not None else self.timeout)
        limit = max(1, int(row_limit))

        try:
            if self._is_mysql_connection(conn):
                res = await self._execute_mysql_with_cap(
                    conn, sql, timeout_s=effective_timeout, cap_rows=limit, started_perf=started_perf, started_wall=started_wall
                )
                # Preview does not need truncation metadata; keep shape consistent.
                return {
                    "columns": res.get("columns", []),
                    "rows": res.get("rows", []),
                    "row_count": int(res.get("row_count", 0) or 0),
                    "execution_time_ms": float(res.get("execution_time_ms", 0) or 0),
                }

            res = await self._execute_postgres_with_cap(
                conn, sql, timeout_s=effective_timeout, cap_rows=limit, started_perf=started_perf, started_wall=started_wall
            )
            return {
                "columns": res.get("columns", []),
                "rows": res.get("rows", []),
                "row_count": int(res.get("row_count", 0) or 0),
                "execution_time_ms": float(res.get("execution_time_ms", 0) or 0),
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
        conn: Union[asyncpg.Connection, Any],
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
        effective_timeout = float(timeout if timeout is not None else self.timeout)
        started = time.perf_counter()

        try:
            if self._is_mysql_connection(conn):
                # aiomysql: execute via cursor; autocommit should be enabled in deps.
                async with conn.cursor() as cur:
                    await asyncio.wait_for(
                        cur.execute(sql),
                        timeout=self._remaining_timeout(started=started, timeout=effective_timeout),
                    )
                return

            await asyncio.wait_for(conn.execute(sql), timeout=effective_timeout)
        except asyncio.TimeoutError:
            raise SQLExecutionError(
                f"DDL execution timeout after {effective_timeout} seconds"
            )
        except asyncpg.PostgresError as e:
            raise SQLExecutionError(f"Database error: {str(e)}")
        except Exception as e:
            raise SQLExecutionError(f"DDL execution failed: {str(e)}")

    async def _execute_postgres_with_cap(
        self,
        conn: asyncpg.Connection,
        sql: str,
        *,
        timeout_s: float,
        cap_rows: int,
        started_perf: float,
        started_wall: float,
    ) -> Dict[str, Any]:
        # asyncpg Connection.cursor() is an async iterator; must run inside a transaction.
        tx = conn.transaction(readonly=True)
        await tx.start()
        cur = conn.cursor(sql)

        async def _consume() -> List[asyncpg.Record]:
            fetched: List[asyncpg.Record] = []
            async for row in cur:
                fetched.append(row)
                if len(fetched) >= cap_rows:
                    break
            return fetched

        try:
            rows = await asyncio.wait_for(_consume(), timeout=timeout_s)
        finally:
            await tx.rollback()

        execution_time_ms = (time.time() - started_wall) * 1000.0
        columns = list(rows[0].keys()) if rows else []
        data = [list(row.values()) for row in rows]

        truncated = False
        if cap_rows > 0 and len(data) >= cap_rows and cap_rows == (int(self.max_rows) + 1):
            # execute_query path: cap = max_rows+1; truncate to max_rows
            truncated = True
            data = data[: int(self.max_rows)]

        return {
            "columns": columns,
            "rows": data,
            "row_count": len(data),
            "returned_row_count": len(data),
            "truncated": bool(truncated),
            "max_rows_cap": int(self.max_rows),
            "execution_time_ms": round(float(execution_time_ms), 2),
        }

    async def _execute_mysql_with_cap(
        self,
        conn: Any,
        sql: str,
        *,
        timeout_s: float,
        cap_rows: int,
        started_perf: float,
        started_wall: float,
    ) -> Dict[str, Any]:
        # aiomysql cursor returns tuples; use fetchmany to enforce cap.
        # NOTE: MindsDB(MySQL endpoint)는 idle timeout/내부 오류로 연결이 끊기는 경우가 있어
        #       "(0, 'Not connected')"가 발생하면 fresh connection으로 1회 재시도한다.
        last_exc: Optional[Exception] = None
        for attempt in range(1, 3):  # original + 1 reconnect retry
            effective_conn = conn
            temp_conn = None
            try:
                if attempt == 2:
                    temp_conn = await self._open_mysql_connection()
                    effective_conn = temp_conn

                # Best-effort ping: 일부 aiomysql 버전은 reconnect 파라미터를 지원한다.
                try:
                    if hasattr(effective_conn, "ping"):
                        try:
                            await effective_conn.ping(reconnect=True)  # type: ignore[call-arg]
                        except TypeError:
                            await effective_conn.ping()
                except Exception:
                    # ping 실패 자체는 execute에서 재시도/예외로 처리
                    pass

                rows: List[Any] = []
                columns: List[str] = []
                async with effective_conn.cursor() as cur:
                    await asyncio.wait_for(
                        cur.execute(sql),
                        timeout=self._remaining_timeout(started=started_perf, timeout=timeout_s),
                    )
                    columns = [d[0] for d in (cur.description or [])] if getattr(cur, "description", None) else []
                    while len(rows) < cap_rows:
                        rem = cap_rows - len(rows)
                        batch = await asyncio.wait_for(
                            cur.fetchmany(rem),
                            timeout=self._remaining_timeout(started=started_perf, timeout=timeout_s),
                        )
                        if not batch:
                            break
                        rows.extend(list(batch))

                execution_time_ms = (time.time() - started_wall) * 1000.0
                data = [list(r) for r in rows]

                truncated = False
                if cap_rows > 0 and len(data) >= cap_rows and cap_rows == (int(self.max_rows) + 1):
                    truncated = True
                    data = data[: int(self.max_rows)]

                return {
                    "columns": columns,
                    "rows": data,
                    "row_count": len(data),
                    "returned_row_count": len(data),
                    "truncated": bool(truncated),
                    "max_rows_cap": int(self.max_rows),
                    "execution_time_ms": round(float(execution_time_ms), 2),
                }
            except Exception as exc:
                last_exc = exc
                # Only retry once, and only for typical stale-connection cases.
                if attempt == 1 and self._should_reconnect_mysql(exc):
                    continue
                raise
            finally:
                if temp_conn is not None:
                    try:
                        temp_conn.close()
                        await temp_conn.wait_closed()
                    except Exception:
                        pass

        # Should be unreachable (loop returns or raises), but keep a safe guard.
        raise SQLExecutionError(f"MySQL execution failed (retries exhausted): {last_exc}")
    
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

