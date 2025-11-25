"""Factory for creating database-specific query builders."""

from typing import Dict, Type

from app.react.utils.db_query_builder.base import ExecutionPlanQueryBuilder
from app.react.utils.db_query_builder.postgresql import PostgreSQLQueryBuilder


# Registry mapping database types to their builder classes
_BUILDER_REGISTRY: Dict[str, Type[ExecutionPlanQueryBuilder]] = {
    "postgresql": PostgreSQLQueryBuilder,
    "postgres": PostgreSQLQueryBuilder,  # Alias
}

# Simple cache for builder instances
_BUILDER_CACHE: Dict[str, ExecutionPlanQueryBuilder] = {}


def get_query_builder(db_type: str) -> ExecutionPlanQueryBuilder:
    """Get a query builder instance for the specified database type.
    
    Args:
        db_type: Database type identifier (e.g., "postgresql", "mysql").
                Case-insensitive.
    
    Returns:
        An instance of ExecutionPlanQueryBuilder for the specified database.
    
    Raises:
        NotImplementedError: If the database type is not supported.
    
    Examples:
        >>> builder = get_query_builder("postgresql")
        >>> execution_plan = builder.fetch_execution_plan(...)
    """
    normalized_type = db_type.lower().strip()
    
    # Check cache first
    if normalized_type in _BUILDER_CACHE:
        return _BUILDER_CACHE[normalized_type]
    
    # Look up builder class in registry
    builder_class = _BUILDER_REGISTRY.get(normalized_type)
    
    if builder_class is None:
        supported = ", ".join(sorted(set(_BUILDER_REGISTRY.keys())))
        raise NotImplementedError(
            f"Database type '{db_type}' is not supported. "
            f"Supported types: {supported}"
        )
    
    # Create and cache the builder instance
    builder = builder_class()
    _BUILDER_CACHE[normalized_type] = builder
    
    return builder


def clear_builder_cache() -> None:
    """Clear the builder cache. Useful for testing."""
    _BUILDER_CACHE.clear()

