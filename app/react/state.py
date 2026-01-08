import base64
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class MetadataParseError(Exception):
    """LLM 이 반환한 metadata XML 파싱 실패"""


@dataclass
class ReactMetadata:
    identified_tables: List[Dict[str, Any]] = field(default_factory=list)
    identified_columns: List[Dict[str, Any]] = field(default_factory=list)
    identified_values: List[Dict[str, Any]] = field(default_factory=list)
    identified_relationships: List[Dict[str, Any]] = field(default_factory=list)
    identified_constraints: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._normalize_entries()

    def _normalize_entries(self) -> None:
        self._normalize_table_entries(self.identified_tables)
        self._ensure_schema_field(self.identified_columns)
        self._ensure_schema_field(self.identified_values)

    @staticmethod
    def _normalize_table_entries(entries: List[Dict[str, Any]]) -> None:
        for entry in entries:
            entry["description"] = (entry.get("description") or "").strip()
            entry["schema"] = (entry.get("schema") or "").strip()

    @staticmethod
    def _ensure_schema_field(entries: List[Dict[str, Any]]) -> None:
        for entry in entries:
            entry["schema"] = (entry.get("schema") or "").strip()

    def update_from_xml(self, metadata_xml: str) -> None:
        if not metadata_xml.strip():
            return

        try:
            element = ET.fromstring(metadata_xml)
        except ET.ParseError as exc:
            raise MetadataParseError(f"Failed to parse metadata XML: {exc}") from exc

        self._normalize_entries()
        self._extend_collection(
            element.find("identified_tables"),
            self.identified_tables,
            ["schema", "name", "purpose", "key_columns", "description"],
        )
        self._extend_collection(
            element.find("identified_columns"),
            self.identified_columns,
            ["schema", "table", "name", "data_type", "purpose"],
        )
        self._extend_collection(
            element.find("identified_values"),
            self.identified_values,
            ["schema", "table", "column", "actual_value", "user_term"],
        )
        self._extend_collection(
            element.find("identified_relationships"),
            self.identified_relationships,
            ["type", "condition", "tables"],
        )
        self._extend_collection(
            element.find("identified_constraints"),
            self.identified_constraints,
            ["type", "condition", "status"],
        )

    def to_xml(self) -> str:
        root = ET.Element("collected_metadata")

        self._append_collection(root, "identified_tables", "table", self.identified_tables)
        self._append_collection(root, "identified_columns", "column", self.identified_columns)
        self._append_collection(root, "identified_values", "value", self.identified_values)
        self._append_collection(root, "identified_relationships", "relationship", self.identified_relationships)
        self._append_collection(root, "identified_constraints", "constraint", self.identified_constraints)

        return ET.tostring(root, encoding="unicode")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identified_tables": self.identified_tables,
            "identified_columns": self.identified_columns,
            "identified_values": self.identified_values,
            "identified_relationships": self.identified_relationships,
            "identified_constraints": self.identified_constraints,
        }

    def has_table(self, table_name: str, schema: Optional[str] = None) -> bool:
        normalized_name = self._normalize_value(table_name)
        if not normalized_name:
            return False
        normalized_schema = (
            self._normalize_value(schema) if schema is not None else None
        )
        for entry in self.identified_tables:
            if self._normalize_value(entry.get("name")) != normalized_name:
                continue
            if normalized_schema is None:
                return True
            if self._normalize_value(entry.get("schema")) == normalized_schema:
                return True
        return False

    def add_table_if_missing(
        self,
        table_name: str,
        schema: str = "",
        purpose: str = "",
        key_columns: str = "",
        description: str = "",
    ) -> bool:
        normalized_name = self._normalize_value(table_name)
        if not normalized_name:
            return False
        schema_text = (schema or "").strip()
        lookup_schema = schema_text if schema_text else None
        if self.has_table(table_name, lookup_schema):
            return False

        if schema_text:
            for entry in self.identified_tables:
                if self._normalize_value(entry.get("name")) != normalized_name:
                    continue
                if not self._normalize_value(entry.get("schema")):
                    entry["schema"] = schema_text
                    return False

        purpose_text = (purpose or "").strip()
        description_text = (description or "").strip()
        key_columns_text = (key_columns or "").strip()

        self.identified_tables.append(
            {
                "schema": schema_text,
                "name": table_name,
                "purpose": purpose_text,
                "key_columns": key_columns_text,
                "description": description_text,
            }
        )
        return True

    @staticmethod
    def _normalize_value(value: Optional[str]) -> str:
        return (value or "").strip().lower()

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReactMetadata":
        if not data:
            return cls()
        return cls(
            identified_tables=data.get("identified_tables", []),
            identified_columns=data.get("identified_columns", []),
            identified_values=data.get("identified_values", []),
            identified_relationships=data.get("identified_relationships", []),
            identified_constraints=data.get("identified_constraints", []),
        )

    @staticmethod
    def _extend_collection(
        parent: Optional[ET.Element],
        target: List[Dict[str, Any]],
        fields: List[str],
    ) -> None:
        if parent is None:
            return
        for child in list(parent):
            entry: Dict[str, Any] = {}
            for field_name in fields:
                entry[field_name] = (child.findtext(field_name) or "").strip()
            if any(entry.values()):
                target.append(entry)

    @staticmethod
    def _append_collection(
        root: ET.Element,
        collection_tag: str,
        item_tag: str,
        items: List[Dict[str, Any]],
    ) -> None:
        collection_el = ET.SubElement(root, collection_tag)
        for item in items:
            item_el = ET.SubElement(collection_el, item_tag)
            for key, value in item.items():
                child = ET.SubElement(item_el, key)
                child.text = str(value)


@dataclass
class ReactSessionState:
    user_query: str
    dbms: str
    remaining_tool_calls: int
    partial_sql: str
    current_tool_result: str
    previous_reasonings: List[Dict[str, Any]] = field(default_factory=list)
    metadata: ReactMetadata = field(default_factory=ReactMetadata)
    iteration: int = 0
    step_confirmation_mode: bool = False
    awaiting_step_confirmation: bool = False
    search_table_candidates: List[Dict[str, str]] = field(default_factory=list)
    max_sql_seconds: int = 60
    prefer_language: str = "ko"
    explained_sqls: List[str] = field(default_factory=list)
    pending_agent_question: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_query": self.user_query,
            "dbms": self.dbms,
            "remaining_tool_calls": self.remaining_tool_calls,
            "partial_sql": self.partial_sql,
            "current_tool_result": self.current_tool_result,
            "previous_reasonings": self.previous_reasonings,
            "metadata": self.metadata.to_dict(),
            "iteration": self.iteration,
            "step_confirmation_mode": self.step_confirmation_mode,
            "awaiting_step_confirmation": self.awaiting_step_confirmation,
            "search_table_candidates": self.search_table_candidates,
            "max_sql_seconds": self.max_sql_seconds,
            "prefer_language": self.prefer_language,
            "explained_sqls": self.explained_sqls,
            "pending_agent_question": self.pending_agent_question,
        }

    def add_previous_reasoning(self, step: int, reasoning: str, limit: int) -> None:
        reasoning_text = (reasoning or "").strip()
        if limit <= 0:
            self.previous_reasonings.clear()
            return
        if not reasoning_text:
            return
        try:
            step_value = int(step)
        except (TypeError, ValueError):
            step_value = 0
        self.previous_reasonings.append({"step": step_value, "reasoning": reasoning_text})
        overflow = len(self.previous_reasonings) - limit
        if overflow > 0:
            del self.previous_reasonings[:overflow]

    def to_token(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False)
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReactSessionState":
        metadata = ReactMetadata.from_dict(data.get("metadata"))
        previous_reasonings = cls._extract_previous_reasonings(data)
        search_table_candidates = cls._normalize_search_table_candidates(
            data.get("search_table_candidates", [])
        )
        explained_sqls = cls._normalize_explained_sqls(data.get("explained_sqls", []))
        pending_agent_question = (data.get("pending_agent_question") or "").strip()
        return cls(
            user_query=data["user_query"],
            dbms=data["dbms"],
            remaining_tool_calls=data["remaining_tool_calls"],
            partial_sql=data.get("partial_sql", "SELECT PLACEHOLDER_COLUMNS FROM PLACEHOLDER_TABLE"),
            current_tool_result=data.get("current_tool_result", "<tool_result/>"),
            previous_reasonings=previous_reasonings,
            metadata=metadata,
            iteration=data.get("iteration", 0),
            step_confirmation_mode=data.get("step_confirmation_mode", False),
            awaiting_step_confirmation=data.get("awaiting_step_confirmation", False),
            search_table_candidates=search_table_candidates,
            max_sql_seconds=data.get("max_sql_seconds", 60),
            prefer_language=data.get("prefer_language", "ko"),
            explained_sqls=explained_sqls,
            pending_agent_question=pending_agent_question,
        )

    @staticmethod
    def dict_to_token(state_dict: Dict[str, Any]) -> str:
        payload = json.dumps(state_dict, ensure_ascii=False)
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("utf-8")

    @classmethod
    def from_token(cls, token: str) -> "ReactSessionState":
        payload = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        data = json.loads(payload)
        metadata = ReactMetadata.from_dict(data.get("metadata"))
        previous_reasonings = cls._extract_previous_reasonings(data)
        search_table_candidates = cls._normalize_search_table_candidates(
            data.get("search_table_candidates", [])
        )
        explained_sqls = cls._normalize_explained_sqls(data.get("explained_sqls", []))
        pending_agent_question = (data.get("pending_agent_question") or "").strip()
        return cls(
            user_query=data["user_query"],
            dbms=data["dbms"],
            remaining_tool_calls=data["remaining_tool_calls"],
            partial_sql=data["partial_sql"],
            current_tool_result=data.get("current_tool_result", "<tool_result/>"),
            previous_reasonings=previous_reasonings,
            metadata=metadata,
            iteration=data.get("iteration", 0),
            step_confirmation_mode=data.get("step_confirmation_mode", False),
            awaiting_step_confirmation=data.get("awaiting_step_confirmation", False),
            search_table_candidates=search_table_candidates,
            max_sql_seconds=data.get("max_sql_seconds", 60),
            prefer_language=data.get("prefer_language", "ko"),
            explained_sqls=explained_sqls,
            pending_agent_question=pending_agent_question,
        )

    @classmethod
    def new(
        cls,
        user_query: str,
        dbms: str,
        remaining_tool_calls: int,
        partial_sql: str = "SELECT PLACEHOLDER_COLUMNS FROM PLACEHOLDER_TABLE",
        step_confirmation_mode: bool = False,
        max_sql_seconds: int = 60,
        prefer_language: str = "ko",
    ) -> "ReactSessionState":
        return cls(
            user_query=user_query,
            dbms=dbms,
            remaining_tool_calls=remaining_tool_calls,
            partial_sql=partial_sql,
            current_tool_result="<tool_result/>",
            step_confirmation_mode=step_confirmation_mode,
            search_table_candidates=[],
            max_sql_seconds=max_sql_seconds,
            prefer_language=prefer_language,
            explained_sqls=[],
        )

    @staticmethod
    def _extract_previous_reasonings(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries = data.get("previous_reasonings")
        normalized: List[Dict[str, Any]] = []
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                reasoning_text = str(entry.get("reasoning", "")).strip()
                if not reasoning_text:
                    continue
                step_raw = entry.get("step")
                try:
                    step_value = int(step_raw)
                except (TypeError, ValueError):
                    step_value = 0
                normalized.append({"step": step_value, "reasoning": reasoning_text})
        if normalized:
            return normalized
        return []

    @staticmethod
    def _normalize_search_table_candidates(
        entries: Any,
    ) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        if not isinstance(entries, list):
            return normalized

        for entry in entries:
            if isinstance(entry, str):
                name = entry.strip()
                if name:
                    normalized.append({"name": name, "schema": "", "description": ""})
                continue

            if isinstance(entry, dict):
                name = str(entry.get("name", "")).strip()
                if not name:
                    continue
                schema_value = entry.get("schema")
                schema_text = (
                    str(schema_value).strip() if schema_value is not None else ""
                )
                description_value = entry.get("description")
                description_text = (
                    str(description_value).strip()
                    if description_value is not None
                    else ""
                )
                normalized.append(
                    {"name": name, "schema": schema_text, "description": description_text}
                )

        return normalized

    @staticmethod
    def _normalize_explained_sqls(entries: Any) -> List[str]:
        """Normalize explained SQLs list."""
        if not isinstance(entries, list):
            return []
        return [str(sql).strip() for sql in entries if sql]

    def add_explained_sql(self, sql: str) -> None:
        """Add a SQL that has been explained."""
        normalized = self._normalize_sql_for_comparison(sql)
        if normalized and normalized not in [
            self._normalize_sql_for_comparison(s) for s in self.explained_sqls
        ]:
            self.explained_sqls.append(sql.strip())

    def has_any_explained_sql(self) -> bool:
        """Return True if this session has successfully run explain at least once."""
        return bool(self.explained_sqls)

    def is_sql_explained(self, sql: str) -> bool:
        """Check if a SQL has been explained."""
        normalized = self._normalize_sql_for_comparison(sql)
        if not normalized:
            return False
        for explained_sql in self.explained_sqls:
            if self._normalize_sql_for_comparison(explained_sql) == normalized:
                return True
        return False

    @staticmethod
    def _normalize_sql_for_comparison(sql: str) -> str:
        """Normalize SQL for comparison (remove extra whitespace)."""
        if not sql:
            return ""
        import re
        # Collapse all whitespace to single spaces and strip
        return re.sub(r'\s+', ' ', sql.strip()).lower()