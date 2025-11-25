import ast
import asyncio
import contextlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.react.enrich_react_metadata import (
    auto_enrich_tables_from_reasoning
)
from app.react.state import ReactMetadata, ReactSessionState, MetadataParseError
from app.react.tools import ToolContext, ToolExecutionError, execute_tool
from app.react.utils import XmlUtil
from app.react.prompts import get_prompt_text


class ReactAgentError(Exception):
    """ReAct 에이전트 실행 중 발생한 일반적인 오류"""


class LLMResponseFormatError(ReactAgentError):
    """LLM 이 잘못된 XML 포맷을 반환했을 때"""


class SQLNotExplainedError(ReactAgentError):
    """explain 없이 execute_sql_preview 또는 submit_sql이 호출되었을 때"""


@dataclass
class SQLCompleteness:
    is_complete: bool
    missing_info: str
    confidence_level: str


@dataclass
class ToolCall:
    name: str
    raw_parameters_xml: str
    parsed_parameters: Dict[str, Any]


@dataclass
class ReactStep:
    iteration: int
    reasoning: str
    metadata_xml: str
    partial_sql: str
    sql_completeness: SQLCompleteness
    tool_call: ToolCall
    tool_result: Optional[str] = None
    llm_output: str = ""


@dataclass
class AgentOutcome:
    status: str
    steps: List[ReactStep]
    final_sql: Optional[str] = None
    final_sql_validated: Optional[str] = None
    metadata: ReactMetadata = field(default_factory=ReactMetadata)
    question_to_user: Optional[str] = None
    was_last_call_submission: bool = False
    sql_not_explained: bool = False


def _to_cdata(value: str) -> str:
    return f"<![CDATA[{value}]]>"


class ReactAgent:
    def __init__(self):
        self.prompt_text = get_prompt_text("react_prompt.xml")
        self.llm = ChatOpenAI(
            model=settings.react_openai_llm_model,
            temperature=0,
            api_key=settings.openai_api_key,
            reasoning_effort="medium"
        )

    async def run(
        self,
        state: ReactSessionState,
        tool_context: ToolContext,
        max_iterations: Optional[int] = None,
        user_response: Optional[str] = None,
        on_step: Optional[Callable[[ReactStep, ReactSessionState], Awaitable[None]]] = None,
        on_outcome: Optional[Callable[[AgentOutcome, ReactSessionState], Awaitable[None]]] = None,
    ) -> AgentOutcome:
        """
        ReAct 루프를 돌며 툴을 실행한다.
        user_response 가 주어지면 가장 최근 tool_result 로 전달한다.
        """
        if user_response:
            parts = ["<tool_result>"]
            if state.pending_agent_question:
                parts.append(
                    f"<agent_question>{_to_cdata(state.pending_agent_question)}</agent_question>"
                )
                state.pending_agent_question = ""  # Clear after use
            parts.append(f"<user_response>{_to_cdata(user_response)}</user_response>")
            parts.append("</tool_result>")
            state.current_tool_result = "".join(parts)

        steps: List[ReactStep] = []
        iteration_limit = max_iterations or (state.remaining_tool_calls + 10)

        for _ in range(iteration_limit):
            if state.remaining_tool_calls <= 0:
                raise ReactAgentError("No remaining tool calls.")

            if max_iterations and len(steps) >= max_iterations:
                break

            state.iteration += 1
            input_xml = self._build_input_xml(state)
            llm_response = await self._call_llm(input_xml)
            parsed_step = self._parse_llm_response(llm_response, state.iteration)

            # Update metadata and partial SQL based on LLM output
            try:
                state.metadata.update_from_xml(parsed_step.metadata_xml)
            except MetadataParseError as exc:
                raise ReactAgentError(str(exc)) from exc
            auto_enrich_tables_from_reasoning(parsed_step.reasoning, state)

            state.partial_sql = parsed_step.partial_sql or state.partial_sql
            state.add_previous_reasoning(
                step=parsed_step.iteration,
                reasoning=parsed_step.reasoning,
                limit=settings.previous_reasoning_limit_steps,
            )

            # Defer tool result assignment; handle by tool execution
            step = parsed_step
            steps.append(step)

            tool_name = step.tool_call.name

            if tool_name == "submit_sql":
                sql_text = step.tool_call.parsed_parameters.get("sql", "")
                is_last_call = state.remaining_tool_calls <= 1
                sql_not_explained = not state.is_sql_explained(sql_text)
                
                # remaining_tool_calls > 1일 때만 explain 검증
                if not is_last_call and sql_not_explained:
                    raise SQLNotExplainedError(
                        "Before calling submit_sql, you must first validate the SQL with the explain tool. "
                        "Please call the explain tool first and then try again."
                    )
                
                outcome = AgentOutcome(
                    status="submit_sql",
                    steps=steps,
                    final_sql=sql_text,
                    metadata=state.metadata,
                    was_last_call_submission=is_last_call,
                    sql_not_explained=sql_not_explained,
                )
                if on_step:
                    await on_step(step, state)
                if on_outcome:
                    await on_outcome(outcome, state)
                return outcome

            if tool_name == "ask_user":
                question = step.tool_call.parsed_parameters.get("question", "")
                state.remaining_tool_calls -= 1
                outcome = AgentOutcome(
                    status="ask_user",
                    steps=steps,
                    metadata=state.metadata,
                    question_to_user=question,
                )
                if on_step:
                    await on_step(step, state)
                if on_outcome:
                    await on_outcome(outcome, state)
                return outcome

            # execute_sql_preview는 반드시 explain이 선행되어야 함
            if tool_name == "execute_sql_preview":
                sql_text = step.tool_call.parsed_parameters.get("sql", "")
                if not state.is_sql_explained(sql_text):
                    raise SQLNotExplainedError(
                        "Before calling execute_sql_preview, you must first validate the SQL with the explain tool. "
                        "Please call the explain tool first and then try again."
                    )

            # Execute actual tool
            try:
                tool_result = await execute_tool(
                    tool_name=tool_name,
                    context=tool_context,
                    parameters=step.tool_call.parsed_parameters,
                )
            except ToolExecutionError as exc:
                raise ReactAgentError(str(exc)) from exc
            
            # explain 호출 시 SQL을 추적
            if tool_name == "explain":
                sql_text = step.tool_call.parsed_parameters.get("sql", "")
                if sql_text:
                    state.add_explained_sql(sql_text)
            
            state.current_tool_result = tool_result
            state.remaining_tool_calls -= 1

            # Update step with tool result
            step.tool_result = tool_result
            if on_step:
                await on_step(step, state)

        raise ReactAgentError("Iteration limit reached without completion.")

    async def stream(
        self,
        state: ReactSessionState,
        tool_context: ToolContext,
        max_iterations: Optional[int] = None,
        user_response: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue()

        async def on_step_callback(step: ReactStep, state_snapshot: ReactSessionState) -> None:
            await queue.put(
                {
                    "type": "step",
                    "step": step,
                    "state": state_snapshot.to_dict(),
                }
            )

        async def on_outcome_callback(outcome: AgentOutcome, state_snapshot: ReactSessionState) -> None:
            payload: Dict[str, Any] = {
                "type": "outcome",
                "outcome": outcome,
                "state": state_snapshot.to_dict(),
            }
            await queue.put(payload)

        async def runner() -> None:
            try:
                await self.run(
                    state=state,
                    tool_context=tool_context,
                    max_iterations=max_iterations,
                    user_response=user_response,
                    on_step=on_step_callback,
                    on_outcome=on_outcome_callback,
                )
            except Exception as exc:
                await queue.put({"type": "error", "error": exc})
            finally:
                await queue.put({"type": "done"})

        task = asyncio.create_task(runner())

        try:
            while True:
                event = await queue.get()
                if event["type"] == "done":
                    break
                yield event
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    def _build_input_xml(self, state: ReactSessionState) -> str:
        user_query = state.user_query
        if settings.is_add_mocked_db_caution:
            user_query = f"""{user_query}

(Important Caution: The database currently in use is a mocked DB with limited data.
- If a tool call fails or returns empty results, it means the data does NOT exist in this mocked environment.
- DO NOT retry the same failed tool call with the same parameters. It will fail again.
- DO NOT assume failures are transient errors. They are permanent limitations of the mocked DB.
- Instead, try alternative approaches: search for different tables, use different keywords, or work with the data you already have.)"""

        parts = ["<input>"]
        parts.append(f"<user_query><![CDATA[{user_query}]]></user_query>")
        parts.append(f"<dbms>{state.dbms}</dbms>")
        parts.append(f"<max_sql_seconds>{state.max_sql_seconds}</max_sql_seconds>")
        parts.append(f"<prefer_language>{state.prefer_language}</prefer_language>")
        parts.append(f"<remaining_tool_calls>{state.remaining_tool_calls}</remaining_tool_calls>")
        parts.append("<current_tool_result>")
        parts.append(state.current_tool_result or "<tool_result/>")
        parts.append("</current_tool_result>")
        parts.append("<previous_reasonings>")
        for entry in state.previous_reasonings:
            step_value = entry.get("step", "")
            step_attr = str(step_value) if step_value is not None else ""
            reasoning_text = entry.get("reasoning", "")
            parts.append(
                f'<previous_reasoning previous_step="{step_attr}"><![CDATA[{reasoning_text}]]></previous_reasoning>'
            )
        parts.append("</previous_reasonings>")
        parts.append(state.metadata.to_xml())
        parts.append(f"<partial_sql><![CDATA[{state.partial_sql}]]></partial_sql>")
        parts.append("</input>")
        return "\n".join(parts)

    async def _call_llm(self, input_xml: str) -> str:
        messages = [
            SystemMessage(content=self.prompt_text),
            HumanMessage(content=input_xml),
        ]
        response = await self.llm.ainvoke(messages)
        if settings.is_add_delay_after_react_generator:
            await asyncio.sleep(settings.delay_after_react_generator_seconds)

        if isinstance(response.content, str):
            return response.content
        if isinstance(response.content, list):
            return "\n".join(
                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                for part in response.content
            )
        return str(response.content)

    def _parse_llm_response(self, response_text: str, iteration: int) -> ReactStep:
        cleaned = response_text.strip()
        start_idx = cleaned.find("<")
        if start_idx > 0:
            cleaned = cleaned[start_idx:]

        cleaned = XmlUtil.sanitize_xml_text(cleaned)
        root = self._parse_xml_with_recovery(cleaned, response_text)

        reasoning = (root.findtext("reasoning") or "").strip()
        metadata_el = root.find("collected_metadata")
        if metadata_el is None:
            raise LLMResponseFormatError("Missing <collected_metadata> section.")
        metadata_xml = ET.tostring(metadata_el, encoding="unicode")

        partial_sql = (root.findtext("partial_sql") or "").strip()

        sql_check_el = root.find("sql_completeness_check")
        if sql_check_el is None:
            raise LLMResponseFormatError("Missing <sql_completeness_check> section.")

        is_complete_text = (sql_check_el.findtext("is_complete") or "").strip().lower()
        is_complete = is_complete_text == "true"
        missing_info = (sql_check_el.findtext("missing_info") or "").strip()
        confidence_level = (sql_check_el.findtext("confidence_level") or "").strip()
        completeness = SQLCompleteness(
            is_complete=is_complete,
            missing_info=missing_info,
            confidence_level=confidence_level,
        )

        tool_call_el = root.find("tool_call")
        if tool_call_el is None:
            raise LLMResponseFormatError("Missing <tool_call> section.")

        tool_name = (tool_call_el.findtext("tool_name") or "").strip()
        parameters_el = tool_call_el.find("parameters")
        if not tool_name or parameters_el is None:
            raise LLMResponseFormatError("Invalid tool_call structure.")

        raw_parameters_xml = ET.tostring(parameters_el, encoding="unicode")
        parsed_parameters = self._parse_tool_parameters(tool_name, parameters_el)

        return ReactStep(
            iteration=iteration,
            reasoning=reasoning,
            metadata_xml=metadata_xml,
            partial_sql=partial_sql,
            sql_completeness=completeness,
            tool_call=ToolCall(
                name=tool_name,
                raw_parameters_xml=raw_parameters_xml,
                parsed_parameters=parsed_parameters,
            ),
            llm_output=response_text,
        )

    def _parse_tool_parameters(
        self,
        tool_name: str,
        parameters_el: ET.Element,
    ) -> Dict[str, Any]:
        if tool_name in {"search_tables", "get_table_schema"}:
            text = (parameters_el.text or "").strip()
            keywords = self._parse_list_literal(text)
            key = "keywords" if tool_name == "search_tables" else "table_names"
            return {key: keywords}

        if tool_name == "search_column_values":
            table_name = (parameters_el.findtext("table") or "").strip()
            schema_name = (parameters_el.findtext("schema") or "").strip()
            column_name = (parameters_el.findtext("column") or "").strip()
            keywords_text = (parameters_el.findtext("search_keywords") or "").strip()
            keywords = self._parse_list_literal(keywords_text)
            return {
                "table": table_name,
                "schema": schema_name,
                "column": column_name,
                "search_keywords": keywords,
            }

        if tool_name in {"execute_sql_preview", "submit_sql", "explain"}:
            # parameters element may include nested tags; we take inner text
            sql_text = "".join(parameters_el.itertext()).strip()
            return {"sql": sql_text}

        if tool_name == "ask_user":
            question = "".join(parameters_el.itertext()).strip()
            return {"question": question}

        raise LLMResponseFormatError(f"Unsupported tool requested: {tool_name}")

    @staticmethod
    def _parse_list_literal(literal_text: str) -> List[str]:
        if not literal_text:
            return []
        try:
            parsed = ast.literal_eval(literal_text)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
        return [literal_text]


    def _parse_xml_with_recovery(self, xml_text: str, raw_response: str) -> ET.Element:
        try:
            return ET.fromstring(xml_text)
        except ET.ParseError as exc:
            wrapped_xml = f"<output>{xml_text}</output>"
            try:
                return ET.fromstring(wrapped_xml)
            except ET.ParseError:
                raise LLMResponseFormatError(
                    f"Invalid XML from LLM: {exc}\nRaw: {raw_response}"
                ) from exc
