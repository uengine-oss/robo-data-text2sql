"""Base classes and interfaces for database query builders."""

from abc import ABC, abstractmethod
from typing import Any, List

from app.react.utils.db_query_builder.type import ExecutionPlanResult, TableMetadata

class ExecutionPlanQueryBuilder(ABC):
    """Abstract base class for database-specific query builders."""

    @abstractmethod
    async def fetch_execution_plan(
        self,
        conn: Any,
        sql: str,
        *,
        analyze: bool = False,
        verbose: bool = True,
        buffers: bool = False,
        format: str = "JSON",
    ) -> ExecutionPlanResult:
        """Collect execution plan metrics for the given SQL statement.
        
        Args:
            conn: Active database connection/handle.
            sql: SQL statement to explain.
            analyze: Include actual execution stats if supported.
            verbose: Include verbose plan details if supported.
            buffers: Include buffer usage stats if supported.
            format: Desired explain output format (default JSON).
        
        Returns:
            ExecutionPlanResult with normalized metrics.
        """
        pass

    @abstractmethod
    async def collect_table_metadata(
        self,
        conn: Any,
        raw_plan: Any,
    ) -> List[TableMetadata]:
        """Collect table metadata for the plan produced by this builder."""
        pass

