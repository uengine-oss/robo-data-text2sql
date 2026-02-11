from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TableCandidate:
    schema: str
    name: str
    description: str
    analyzed_description: str = ""
    score: float = 0.0

    @property
    def fqn(self) -> str:
        return f"{self.schema}.{self.name}" if self.schema else self.name


@dataclass
class ColumnCandidate:
    table_schema: str
    table_name: str
    name: str
    dtype: str
    description: str
    score: float

    @property
    def table_fqn(self) -> str:
        return f"{self.table_schema}.{self.table_name}" if self.table_schema else self.table_name

    @property
    def column_fqn(self) -> str:
        return f"{self.table_fqn}.{self.name}" if self.table_fqn else self.name


