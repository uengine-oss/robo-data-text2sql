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
    pending_agent_question: str = ""
    # validate_sql 기반 게이트/재사용을 위한 상태
    last_validated_sql: str = ""
    last_validation_preview: Dict[str, Any] = field(default_factory=dict)
    validation_fail_count: int = 0
    # Auto context-refresh loop control (stop-condition-based). Best-effort; safe to ignore by callers.
    auto_context_refresh_count: int = 0
    auto_context_refresh_tried_queries: List[str] = field(default_factory=list)
    auto_context_refresh_no_progress_streak: int = 0
    auto_context_refresh_last_context_hash: str = ""

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
            "pending_agent_question": self.pending_agent_question,
            "last_validated_sql": self.last_validated_sql,
            "last_validation_preview": self.last_validation_preview,
            "validation_fail_count": self.validation_fail_count,
            "auto_context_refresh_count": self.auto_context_refresh_count,
            "auto_context_refresh_tried_queries": list(self.auto_context_refresh_tried_queries or []),
            "auto_context_refresh_no_progress_streak": self.auto_context_refresh_no_progress_streak,
            "auto_context_refresh_last_context_hash": self.auto_context_refresh_last_context_hash,
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
        pending_agent_question = (data.get("pending_agent_question") or "").strip()
        last_validated_sql = (data.get("last_validated_sql") or "").strip()
        last_validation_preview = data.get("last_validation_preview") or {}
        if not isinstance(last_validation_preview, dict):
            last_validation_preview = {}
        try:
            validation_fail_count = int(data.get("validation_fail_count", 0) or 0)
        except (TypeError, ValueError):
            validation_fail_count = 0
        try:
            auto_count = int(data.get("auto_context_refresh_count", 0) or 0)
        except (TypeError, ValueError):
            auto_count = 0
        tried_raw = data.get("auto_context_refresh_tried_queries") or []
        tried_list: List[str] = []
        if isinstance(tried_raw, list):
            for x in tried_raw[:50]:
                s = str(x or "").strip()
                if s:
                    tried_list.append(s[:500])
        try:
            no_prog = int(data.get("auto_context_refresh_no_progress_streak", 0) or 0)
        except (TypeError, ValueError):
            no_prog = 0
        last_hash = str(data.get("auto_context_refresh_last_context_hash") or "").strip()
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
            pending_agent_question=pending_agent_question,
            last_validated_sql=last_validated_sql,
            last_validation_preview=last_validation_preview,
            validation_fail_count=validation_fail_count,
            auto_context_refresh_count=auto_count,
            auto_context_refresh_tried_queries=tried_list,
            auto_context_refresh_no_progress_streak=no_prog,
            auto_context_refresh_last_context_hash=last_hash[:120],
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
        pending_agent_question = (data.get("pending_agent_question") or "").strip()
        last_validated_sql = (data.get("last_validated_sql") or "").strip()
        last_validation_preview = data.get("last_validation_preview") or {}
        if not isinstance(last_validation_preview, dict):
            last_validation_preview = {}
        try:
            validation_fail_count = int(data.get("validation_fail_count", 0) or 0)
        except (TypeError, ValueError):
            validation_fail_count = 0
        try:
            auto_count = int(data.get("auto_context_refresh_count", 0) or 0)
        except (TypeError, ValueError):
            auto_count = 0
        tried_raw = data.get("auto_context_refresh_tried_queries") or []
        tried_list: List[str] = []
        if isinstance(tried_raw, list):
            for x in tried_raw[:50]:
                s = str(x or "").strip()
                if s:
                    tried_list.append(s[:500])
        try:
            no_prog = int(data.get("auto_context_refresh_no_progress_streak", 0) or 0)
        except (TypeError, ValueError):
            no_prog = 0
        last_hash = str(data.get("auto_context_refresh_last_context_hash") or "").strip()
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
            pending_agent_question=pending_agent_question,
            last_validated_sql=last_validated_sql,
            last_validation_preview=last_validation_preview,
            validation_fail_count=validation_fail_count,
            auto_context_refresh_count=auto_count,
            auto_context_refresh_tried_queries=tried_list,
            auto_context_refresh_no_progress_streak=no_prog,
            auto_context_refresh_last_context_hash=last_hash[:120],
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
            last_validated_sql="",
            last_validation_preview={},
            validation_fail_count=0,
            auto_context_refresh_count=0,
            auto_context_refresh_tried_queries=[],
            auto_context_refresh_no_progress_streak=0,
            auto_context_refresh_last_context_hash="",
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

    def record_validation_pass(self, *, sql: str, preview: Optional[Dict[str, Any]] = None) -> None:
        self.last_validated_sql = (sql or "").strip()
        self.last_validation_preview = preview or {}
        self.validation_fail_count = 0

    def record_validation_fail(self) -> None:
        try:
            self.validation_fail_count = int(self.validation_fail_count or 0) + 1
        except Exception:
            self.validation_fail_count = 1