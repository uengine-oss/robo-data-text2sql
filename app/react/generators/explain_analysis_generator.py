import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape as xml_escape

import asyncpg
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.core.sql_exec import SQLExecutionError, SQLExecutor
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.react.prompts import get_prompt_text
from app.react.utils import XmlUtil
from app.react.utils.db_query_builder import ExecutionPlanResult, get_query_builder, TableMetadata

logger = logging.getLogger(__name__)


def _to_cdata(value: str) -> str:
    return f"<![CDATA[{value}]]>"


class ExplainAnalysisGeneratorError(Exception):
    """Raised when explain analysis generation fails."""


@dataclass
class ValidationQuery:
    query_id: str
    reason: str
    sql: str


@dataclass
class ValidationQueryResult:
    query: ValidationQuery
    success: bool
    row_count: Optional[int] = None
    execution_time_ms: Optional[float] = None
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    error_message: Optional[str] = None


@dataclass
class ExplainAnalysisLLMResponse:
    risk_summary: str
    validation_queries: List[ValidationQuery]

    @classmethod
    def from_xml(cls, xml_text: str) -> "ExplainAnalysisLLMResponse":
        sanitized = XmlUtil.sanitize_xml_text(xml_text.strip())
        try:
            root = ET.fromstring(sanitized)
        except ET.ParseError as exc:
            raise ExplainAnalysisGeneratorError(
                f"LLM 응답 XML 파싱 실패: {exc}"
            ) from exc

        if root.tag != "validation_plan":
            candidate = root.find("validation_plan")
            if candidate is None:
                raise ExplainAnalysisGeneratorError(
                    "LLM 응답에서 <validation_plan> 노드를 찾을 수 없습니다."
                )
            root = candidate

        summary = (root.findtext("./risk_analysis/summary") or "").strip()
        queries: List[ValidationQuery] = []
        for idx, query_el in enumerate(root.findall("./queries/query"), start=1):
            query_id = query_el.get("id") or str(idx)
            reason = (query_el.findtext("reason") or "").strip()
            sql_text = (query_el.findtext("sql") or "").strip()
            if not sql_text:
                logger.warning("LLM validation query #%s 에 SQL 내용이 없습니다.", query_id)
                continue
            queries.append(
                ValidationQuery(
                    query_id=query_id,
                    reason=reason,
                    sql=sql_text,
                )
            )

        if not queries:
            logger.warning("LLM 응답에서 실행할 검증 쿼리가 생성되지 않았습니다.")

        return cls(risk_summary=summary, validation_queries=queries)


@dataclass
class ExplainAnalysisResult:
    input_sql: str
    execution_plan: ExecutionPlanResult
    table_metadata: List[TableMetadata]
    risk_analysis_summary: str
    validation_results: List[ValidationQueryResult]
    llm_raw_response: str

    def to_xml_str(self) -> str:
        parts: List[str] = ["<explain_analysis_result>"]
        parts.append(f"<input_sql>{_to_cdata(self.input_sql)}</input_sql>")
        parts.append("<execution_plan>")
        if self.execution_plan.total_cost is not None and self.execution_plan.total_cost > 0:
            parts.append(f"<total_cost>{self.execution_plan.total_cost}</total_cost>")
        if self.execution_plan.execution_time_ms is not None and self.execution_plan.execution_time_ms > 0:
            parts.append(
                f"<execution_time_ms>{self.execution_plan.execution_time_ms}</execution_time_ms>"
            )
        parts.append(f"<row_count>{self.execution_plan.row_count}</row_count>")
        raw_plan_str = json.dumps(self.execution_plan.raw_plan)
        parts.append(f"<raw_plan>{_to_cdata(raw_plan_str)}</raw_plan>")
        parts.append("</execution_plan>")

        parts.append("<table_metadata>")
        for meta in self.table_metadata:
            schema_attr = xml_escape(meta.schema_name, {'"': "&quot;"})
            table_attr = xml_escape(meta.table_name, {'"': "&quot;"})
            parts.append(f'<table schema="{schema_attr}" name="{table_attr}">')
            parts.append(f"<row_count>{meta.row_count}</row_count>")
            if meta.indexes:
                parts.append("<indexes>")
                for index in meta.indexes:
                    idx_name = xml_escape(index.index_name, {'"': "&quot;"})
                    parts.append(
                        f'<index name="{idx_name}" unique="{str(index.is_unique).lower()}">'
                    )
                    if index.columns:
                        parts.append("<columns>")
                        for column in index.columns:
                            parts.append(
                                f"<column>{xml_escape(column)}</column>"
                            )
                        parts.append("</columns>")
                    else:
                        parts.append("<columns />")
                    parts.append(f"<definition>{_to_cdata(index.definition)}</definition>")
                    parts.append("</index>")
                parts.append("</indexes>")
            else:
                parts.append("<indexes />")
            parts.append("</table>")
        parts.append("</table_metadata>")

        parts.append("<risk_analysis>")
        parts.append(f"<summary>{_to_cdata(self.risk_analysis_summary)}</summary>")
        parts.append("</risk_analysis>")

        parts.append("<validation_queries>")
        for result in self.validation_results:
            parts.append(f'<validation_query id="{xml_escape(result.query.query_id)}">')
            parts.append(f"<reason>{_to_cdata(result.query.reason)}</reason>")
            parts.append(f"<sql>{_to_cdata(result.query.sql)}</sql>")
            parts.append("<execution>")
            parts.append(
                f"<status>{'success' if result.success else 'error'}</status>"
            )
            if result.row_count is not None:
                parts.append(f"<row_count>{result.row_count}</row_count>")
            if result.execution_time_ms is not None:
                parts.append(
                    f"<execution_time_ms>{result.execution_time_ms}</execution_time_ms>"
                )
            if result.columns:
                parts.append("<columns>")
                for column in result.columns:
                    parts.append(f"<column>{xml_escape(column)}</column>")
                parts.append("</columns>")
            if result.rows:
                parts.append("<rows>")
                for idx, row in enumerate(result.rows, start=1):
                    parts.append(f'<row index="{idx}">')
                    for col_idx, value in enumerate(row):
                        column_name = (
                            result.columns[col_idx]
                            if col_idx < len(result.columns)
                            else f"column_{col_idx+1}"
                        )
                        cell_value = "" if value is None else str(value)
                        parts.append(
                            f'<value column="{xml_escape(column_name, {"\"": "&quot;"})}">{_to_cdata(cell_value)}</value>'
                        )
                    parts.append("</row>")
                parts.append("</rows>")
            if result.error_message:
                parts.append(
                    f"<error_message>{_to_cdata(result.error_message)}</error_message>"
                )
            parts.append("</execution>")
            parts.append("</validation_query>")
        parts.append("</validation_queries>")
        parts.append("</explain_analysis_result>")
        return "\n".join(parts)


class ExplainAnalysisGenerator:
    """Generator that leverages explain_analysis_prompt to validate plans."""

    def __init__(
        self,
        *,
        llm: Optional[ChatOpenAI] = None,
        db_type: Optional[str] = None,
    ):
        self.prompt_text = get_prompt_text("explain_analysis_prompt.xml")
        self.llm = llm or ChatOpenAI(
            model=settings.react_openai_llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
            reasoning_effort="medium",
        )
        self.db_type = db_type or settings.target_db_type

    async def generate(
        self,
        *,
        sql: str,
        db_conn: asyncpg.Connection,
    ) -> ExplainAnalysisResult:
        sql_text = (sql or "").strip()
        if not sql_text:
            raise ValueError("SQL 문이 비어 있습니다.")

        builder = get_query_builder(self.db_type)
        execution_plan = await builder.fetch_execution_plan(db_conn, sql_text, analyze=False)
        table_metadata = await builder.collect_table_metadata(
            db_conn,
            execution_plan.raw_plan,
        )

        prompt_input = self._build_prompt_input(
            sql=sql_text,
            execution_plan=execution_plan,
            table_metadata=table_metadata,
        )

        llm_response_text = await self._call_llm(prompt_input)
        parsed_response = ExplainAnalysisLLMResponse.from_xml(llm_response_text)

        validation_results = await self._execute_validation_queries(
            parsed_response.validation_queries,
            db_conn=db_conn,
        )

        return ExplainAnalysisResult(
            input_sql=sql_text,
            execution_plan=execution_plan,
            table_metadata=table_metadata,
            risk_analysis_summary=parsed_response.risk_summary,
            validation_results=validation_results,
            llm_raw_response=llm_response_text,
        )

    async def _call_llm(self, input_xml: str) -> str:
        messages = [
            SystemMessage(content=self.prompt_text),
            HumanMessage(content=input_xml),
        ]
        response = await self.llm.ainvoke(messages)
        if isinstance(response.content, str):
            return response.content
        if isinstance(response.content, list):
            return "\n".join(
                part.get("text", "")
                if isinstance(part, dict)
                else str(part)
                for part in response.content
            )
        return str(response.content)

    async def _execute_validation_queries(
        self,
        queries: List[ValidationQuery],
        *,
        db_conn: asyncpg.Connection,
    ) -> List[ValidationQueryResult]:
        if not queries:
            return []

        guard = SQLGuard()
        executor = SQLExecutor()
        executor.timeout = min(executor.timeout, settings.explain_analysis_timeout_seconds)

        results: List[ValidationQueryResult] = []
        for query in queries:
            result = await self._run_single_validation_query(
                query,
                guard=guard,
                executor=executor,
                db_conn=db_conn,
            )
            results.append(result)
        return results

    async def _run_single_validation_query(
        self,
        query: ValidationQuery,
        *,
        guard: SQLGuard,
        executor: SQLExecutor,
        db_conn: asyncpg.Connection,
    ) -> ValidationQueryResult:
        try:
            validated_sql, _ = guard.validate(query.sql)
            execution_result = await executor.execute_query(db_conn, validated_sql)
            formatted = SQLExecutor.format_results_for_json(execution_result)
            return ValidationQueryResult(
                query=query,
                success=True,
                row_count=formatted["row_count"],
                execution_time_ms=formatted["execution_time_ms"],
                columns=formatted["columns"],
                rows=formatted["rows"],
            )
        except (SQLValidationError, SQLExecutionError) as exc:
            logger.warning(
                "검증 쿼리 실행 실패 (%s): %s", query.query_id, exc, exc_info=True
            )
            return ValidationQueryResult(
                query=query,
                success=False,
                error_message=str(exc),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("검증 쿼리 실행 중 알 수 없는 오류 발생")
            return ValidationQueryResult(
                query=query,
                success=False,
                error_message=f"Unexpected error: {exc}",
            )

    def _build_prompt_input(
        self,
        *,
        sql: str,
        execution_plan: ExecutionPlanResult,
        table_metadata: List[TableMetadata],
    ) -> str:
        plan_json = json.dumps(execution_plan.raw_plan)
        parts: List[str] = ["<input>"]
        parts.append(f"<sql>{_to_cdata(sql)}</sql>")
        parts.append("<execution_plan>")
        parts.append(f"<total_cost>{execution_plan.total_cost}</total_cost>")
        parts.append(
            f"<execution_time_ms>{execution_plan.execution_time_ms}</execution_time_ms>"
        )
        parts.append(f"<row_count>{execution_plan.row_count}</row_count>")
        parts.append(f"<raw_plan>{_to_cdata(plan_json)}</raw_plan>")
        parts.append("</execution_plan>")

        parts.append("<table_metadata>")
        for meta in table_metadata:
            schema_attr = xml_escape(meta.schema_name, {'"': "&quot;"})
            table_attr = xml_escape(meta.table_name, {'"': "&quot;"})
            parts.append(f'<table schema="{schema_attr}" name="{table_attr}">')
            parts.append(f"<row_count>{meta.row_count}</row_count>")
            if meta.indexes:
                parts.append("<indexes>")
                for index in meta.indexes:
                    idx_name = xml_escape(index.index_name, {'"': "&quot;"})
                    parts.append(
                        f'<index name="{idx_name}" unique="{str(index.is_unique).lower()}">'
                    )
                    if index.columns:
                        parts.append("<columns>")
                        for column in index.columns:
                            parts.append(f"<column>{xml_escape(column)}</column>")
                        parts.append("</columns>")
                    else:
                        parts.append("<columns />")
                    parts.append(f"<definition>{_to_cdata(index.definition)}</definition>")
                    parts.append("</index>")
                parts.append("</indexes>")
            else:
                parts.append("<indexes />")
            parts.append("</table>")
        parts.append("</table_metadata>")
        parts.append("</input>")
        return "\n".join(parts)