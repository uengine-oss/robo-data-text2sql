from typing import Any, Dict, List
from dataclasses import dataclass, field

@dataclass
class ExecutionPlanResult:
    """Normalized execution plan metrics returned by builders."""

    total_cost: float
    execution_time_ms: float
    row_count: int
    raw_plan: Dict[str, Any]

@dataclass
class TableIndexMetadata:
    """Metadata for a single index."""

    index_name: str
    is_unique: bool
    columns: List[str] = field(default_factory=list)
    definition: str = ""


@dataclass
class TableMetadata:
    """Basic metadata for a referenced table."""

    schema_name: str
    table_name: str
    row_count: int = 0
    indexes: List[TableIndexMetadata] = field(default_factory=list)