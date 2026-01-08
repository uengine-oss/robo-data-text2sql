import ast
import asyncio
import contextlib
import time
import traceback
import uuid
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

from app.config import settings
from app.react.enrich_react_metadata import (
    auto_enrich_tables_from_reasoning
)
from app.react.state import ReactMetadata, ReactSessionState, MetadataParseError
from app.react.llm_factory import create_react_llm
from app.react.tools import ToolContext, ToolExecutionError, execute_tool
from app.react.utils import XmlUtil
from app.react.prompts import get_prompt_text
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


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


def _new_react_run_id() -> str:
    # Keep it short-but-unique for correlation across logs.
    return f"react_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _extract_first_output_xml(xml_text: str) -> Dict[str, Any]:
    """
    Extract the first <output>...</output> block from a potentially noisy LLM response.

    LLMs sometimes append markdown fences or extra tool-call markup after the closing </output>.
    This helper trims the payload to a single, well-formed XML document for robust parsing.
    """
    text = (xml_text or "").strip()
    start = text.find("<output")
    if start < 0:
        return {"xml": text, "trimmed": False, "start": -1, "end": -1}

    # Find the matching </output> for the first <output ...> considering nested <output> blocks.
    import re

    tag_iter = re.finditer(r"</?output\b", text[start:])
    depth = 0
    end_inclusive = -1
    for m in tag_iter:
        is_close = m.group(0).startswith("</")
        if not is_close:
            depth += 1
            continue
        depth -= 1
        if depth <= 0:
            # We matched the closing tag for the first output block.
            end_inclusive = start + m.start() + len("</output>")
            break

    if end_inclusive < 0:
        extracted = text[start:]
        trimmed = start > 0
        return {"xml": extracted, "trimmed": trimmed, "start": start, "end": -1}

    extracted = text[start:end_inclusive]
    trimmed = start > 0 or end_inclusive < len(text)
    return {"xml": extracted, "trimmed": trimmed, "start": start, "end": end_inclusive}


def _build_parameters_xml_text(value: str) -> str:
    """Build a <parameters>...</parameters> XML string with proper escaping."""
    el = ET.Element("parameters")
    el.text = value or ""
    return ET.tostring(el, encoding="unicode")


def _tool_result_has_error(tool_result_xml: Optional[str]) -> bool:
    """
    Return True if <tool_result> contains an <error> element.
    If parsing fails, treat as error (conservative).
    """
    text = (tool_result_xml or "").strip()
    if not text:
        return True
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return True
    return root.find("error") is not None or root.find(".//error") is not None


def _rewrite_step_tool_call(
    *,
    step: ReactStep,
    to_tool_name: str,
    sql_text: str,
) -> None:
    """Rewrite the parsed tool_call in-place (name + parameters + raw XML)."""
    step.tool_call.name = to_tool_name
    step.tool_call.parsed_parameters = {"sql": sql_text or ""}
    step.tool_call.raw_parameters_xml = _build_parameters_xml_text(sql_text or "")


def _maybe_rewrite_tool_call_for_policy(
    *,
    step: ReactStep,
    state: ReactSessionState,
) -> Optional[Dict[str, Any]]:
    """
    Runtime tool_call rewrite policy.
    - If remaining_tool_calls <= 1 and tool is execute_sql_preview -> submit_sql
    - Else if no successful explain yet in session and tool is execute_sql_preview/submit_sql -> explain
    Keeps 'has_any_explain' relaxed policy: once at least one explain succeeded, allow mismatches.
    """
    tool_name = step.tool_call.name
    if tool_name not in {"execute_sql_preview", "submit_sql"}:
        return None

    sql_text = (step.tool_call.parsed_parameters or {}).get("sql", "") or ""
    is_last_call = state.remaining_tool_calls <= 1

    if is_last_call and tool_name == "execute_sql_preview":
        _rewrite_step_tool_call(step=step, to_tool_name="submit_sql", sql_text=sql_text)
        return {
            "from": "execute_sql_preview",
            "to": "submit_sql",
            "reason": "last_call_force_submit",
        }

    if (not is_last_call) and tool_name in {"execute_sql_preview", "submit_sql"}:
        has_any_explain = state.has_any_explained_sql()
        is_explained = state.is_sql_explained(sql_text)
        if (not has_any_explain) and (not is_explained):
            _rewrite_step_tool_call(step=step, to_tool_name="explain", sql_text=sql_text)
            return {
                "from": tool_name,
                "to": "explain",
                "reason": "require_explain_first",
            }

    return None


class ReactAgent:
    def __init__(self):
        self.prompt_text = get_prompt_text("react_prompt.xml")
        self.llm = create_react_llm()

    async def _call_llm_xml_reprint(
        self,
        raw_llm_text: str,
        *,
        react_run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> str:
        """
        One-shot repair: ask the LLM to reprint ONLY valid XML (no extra text).
        Used when parsing still fails even after local best-effort repairs.
        """
        system_prompt = (
            "You are a strict XML formatter.\n"
            "You will receive text that was supposed to be a single valid XML document.\n"
            "Return ONLY the corrected XML.\n"
            "- Output must be a single <output>...</output> document.\n"
            "- Do NOT include markdown fences or any text outside XML.\n"
            "- Do NOT output a <note> element.\n"
            "- For any free-text fields, NEVER output raw '<' or '&' characters.\n"
            "  If you must include them, use CDATA or escape as &lt; &gt; &amp;.\n"
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "Fix the following content into a valid <output> XML only:\n\n"
                    f"{raw_llm_text}"
                )
            ),
        ]
        SmartLogger.log(
            "ERROR",
            "react.llm.reprint.request",
            category="react.llm.reprint.request",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "iteration": iteration,
                    "phase": "thinking",
                    "model": getattr(self.llm, "model_name", None)
                    or getattr(self.llm, "model", None),
                    "raw_llm_text": raw_llm_text,
                }
            ),
        )
        started = time.perf_counter()
        response = await self.llm.ainvoke(messages)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if isinstance(response.content, str):
            content = response.content
        elif isinstance(response.content, list):
            content = "\n".join(
                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                for part in response.content
            )
        else:
            content = str(response.content)

        SmartLogger.log(
            "ERROR",
            "react.llm.reprint.response",
            category="react.llm.reprint.response",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "iteration": iteration,
                    "phase": "thinking",
                    "elapsed_ms": elapsed_ms,
                    "assistant_response": content,
                }
            ),
        )
        return content

    async def run(
        self,
        state: ReactSessionState,
        tool_context: ToolContext,
        max_iterations: Optional[int] = None,
        user_response: Optional[str] = None,
        react_run_id: Optional[str] = None,
        api_started_perf_counter: Optional[float] = None,
        on_step: Optional[Callable[[ReactStep, ReactSessionState], Awaitable[None]]] = None,
        on_outcome: Optional[Callable[[AgentOutcome, ReactSessionState], Awaitable[None]]] = None,
        on_phase: Optional[Callable[[str, int, Dict[str, Any], ReactSessionState], Awaitable[None]]] = None,
        on_token: Optional[Callable[[int, str], Awaitable[None]]] = None,
        on_format_repair: Optional[Callable[[int, str], Awaitable[None]]] = None,
    ) -> AgentOutcome:
        """
        ReAct 루프를 돌며 툴을 실행한다.
        user_response 가 주어지면 가장 최근 tool_result 로 전달한다.
        """
        run_id = react_run_id or _new_react_run_id()
        run_started = time.perf_counter()
        last_input_xml: Optional[str] = None
        last_llm_response: Optional[str] = None
        last_tool_name: Optional[str] = None
        last_tool_parameters: Optional[Dict[str, Any]] = None
        last_tool_result: Optional[str] = None
        outcome: Optional[AgentOutcome] = None
        final_exception: Optional[BaseException] = None
        final_traceback: Optional[str] = None
        cancelled: bool = False

        # Per-iteration profile summary (level-0: no raw SQL/question)
        iteration_profiles: List[Dict[str, Any]] = []
        totals = {
            "llm_elapsed_ms_sum": 0.0,
            "llm_reprint_elapsed_ms_sum": 0.0,
            "tool_elapsed_ms_sum": 0.0,
        }

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

        SmartLogger.log(
            "INFO",
            "react.run.start",
            category="react.run.start",
            params=sanitize_for_log(
                {
                    "react_run_id": run_id,
                    "iteration_limit": iteration_limit,
                    "max_iterations": max_iterations,
                    "user_response": user_response,
                    "state_snapshot": state.to_dict(),
                }
            ),
        )

        try:
            for _ in range(iteration_limit):
                if state.remaining_tool_calls <= 0:
                    raise ReactAgentError("No remaining tool calls.")

                if max_iterations and len(steps) >= max_iterations:
                    break

                state.iteration += 1
            
                # Phase: thinking - LLM 호출 시작
                if on_phase:
                    await on_phase("thinking", state.iteration, {}, state)
            
                input_xml = self._build_input_xml(state)
                last_input_xml = input_xml

                iter_profile: Dict[str, Any] = {
                    "iteration": state.iteration,
                    "llm_elapsed_ms": None,
                    "llm_reprint_elapsed_ms": None,
                    "tool_name": None,
                    "tool_executed": None,
                    "tool_elapsed_ms": None,
                }
                iteration_profiles.append(iter_profile)

                llm_started = time.perf_counter()
                llm_response: Optional[str] = None
                try:
                    if on_token:
                        # 스트리밍 LLM 호출 (토큰 단위로 콜백)
                        llm_response = await self._call_llm_streaming(
                            input_xml,
                            on_token=on_token,
                            iteration=state.iteration,
                            react_run_id=run_id,
                        )
                    else:
                        llm_response = await self._call_llm(
                            input_xml,
                            react_run_id=run_id,
                            iteration=state.iteration,
                        )
                    last_llm_response = llm_response
                finally:
                    llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000.0
                    iter_profile["llm_elapsed_ms"] = llm_elapsed_ms
                    totals["llm_elapsed_ms_sum"] += llm_elapsed_ms
                if llm_response is None:
                    raise ReactAgentError("LLM call did not return a response.")

                try:
                    parsed_step = self._parse_llm_response(
                        llm_response,
                        state.iteration,
                        react_run_id=run_id,
                        state_snapshot=state.to_dict(),
                    )
                except Exception as parse_exc:
                    # Notify stream consumer that we are about to perform a format repair (LLM reprint).
                    if on_format_repair:
                        await on_format_repair(state.iteration, "parse_retry")
                    # F3: ask LLM to reprint XML once, then re-parse.
                    SmartLogger.log(
                        "ERROR",
                        "react.llm.parse_retry",
                        category="react.llm.parse_retry",
                        params=sanitize_for_log(
                            {
                                "react_run_id": run_id,
                                "iteration": state.iteration,
                                "exception": repr(parse_exc),
                                "traceback": traceback.format_exc(),
                                "assistant_response_raw": llm_response,
                            }
                        ),
                    )
                    reprint_started = time.perf_counter()
                    reprinted: Optional[str] = None
                    try:
                        reprinted = await self._call_llm_xml_reprint(
                            llm_response,
                            react_run_id=run_id,
                            iteration=state.iteration,
                        )
                        last_llm_response = reprinted
                    finally:
                        reprint_elapsed_ms = (time.perf_counter() - reprint_started) * 1000.0
                        iter_profile["llm_reprint_elapsed_ms"] = reprint_elapsed_ms
                        totals["llm_reprint_elapsed_ms_sum"] += reprint_elapsed_ms
                    if reprinted is None:
                        raise ReactAgentError("LLM reprint did not return a response.")
                    parsed_step = self._parse_llm_response(
                        reprinted,
                        state.iteration,
                        react_run_id=run_id,
                        state_snapshot=state.to_dict(),
                    )
            
                # Runtime tool_call rewrite (explain gate / last call)
                rewrite_info = _maybe_rewrite_tool_call_for_policy(step=parsed_step, state=state)
                if rewrite_info:
                    SmartLogger.log(
                        "WARNING",
                        "react.explain_gate.rewritten",
                        category="react.explain_gate.rewritten",
                        params=sanitize_for_log(
                            {
                                "react_run_id": run_id,
                                "iteration": state.iteration,
                                "from_tool": rewrite_info.get("from"),
                                "to_tool": rewrite_info.get("to"),
                                "reason": rewrite_info.get("reason"),
                            }
                        ),
                    )

                # Phase: reasoning - LLM 응답 파싱 완료
                if on_phase:
                    await on_phase(
                        "reasoning",
                        state.iteration,
                        {
                            "reasoning": parsed_step.reasoning,
                            "partial_sql": parsed_step.partial_sql,
                            "sql_completeness": {
                                "is_complete": parsed_step.sql_completeness.is_complete,
                                "missing_info": parsed_step.sql_completeness.missing_info,
                                "confidence_level": parsed_step.sql_completeness.confidence_level,
                            },
                            "tool_name": parsed_step.tool_call.name,
                            "tool_parameters": parsed_step.tool_call.parsed_parameters,
                        },
                        state,
                    )

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
                iter_profile["tool_name"] = tool_name

                if tool_name == "submit_sql":
                    sql_text = step.tool_call.parsed_parameters.get("sql", "")
                    is_last_call = state.remaining_tool_calls <= 1
                    sql_not_explained = not state.is_sql_explained(sql_text)
                    has_any_explain = state.has_any_explained_sql()
                
                    # Safety net: never hard-fail. If policy requires explain, rewrite and fall through.
                    if (not is_last_call) and sql_not_explained and (not has_any_explain):
                        _rewrite_step_tool_call(step=step, to_tool_name="explain", sql_text=sql_text)
                        tool_name = step.tool_call.name
                        iter_profile["tool_name"] = tool_name
                    if not is_last_call and sql_not_explained and has_any_explain:
                        SmartLogger.log(
                            "WARNING",
                            "react.explain_gate.bypassed",
                            category="react.explain_gate.bypassed",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": run_id,
                                    "iteration": state.iteration,
                                    "tool_name": tool_name,
                                    "reason": "sql_not_exactly_explained_but_session_has_any_explain",
                                }
                            ),
                        )
                
                    if tool_name == "submit_sql":
                        # submit_sql: no actual tool execution
                        iter_profile["tool_executed"] = False
                        iter_profile["tool_elapsed_ms"] = None

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
                    # ask_user: no actual tool execution
                    iter_profile["tool_executed"] = False
                    iter_profile["tool_elapsed_ms"] = None

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
                    is_explained = state.is_sql_explained(sql_text)
                    has_any_explain = state.has_any_explained_sql()
                    if not is_explained and not has_any_explain:
                        _rewrite_step_tool_call(step=step, to_tool_name="explain", sql_text=sql_text)
                        tool_name = step.tool_call.name
                        iter_profile["tool_name"] = tool_name
                    if not is_explained and has_any_explain:
                        SmartLogger.log(
                            "WARNING",
                            "react.explain_gate.bypassed",
                            category="react.explain_gate.bypassed",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": run_id,
                                    "iteration": state.iteration,
                                    "tool_name": tool_name,
                                    "reason": "sql_not_exactly_explained_but_session_has_any_explain",
                                }
                            ),
                        )

                # Phase: acting - 도구 실행 시작
                if on_phase:
                    await on_phase(
                        "acting",
                        state.iteration,
                        {
                            "tool_name": tool_name,
                            "tool_parameters": step.tool_call.parsed_parameters,
                        },
                        state,
                    )

                # Execute actual tool (log request/response + elapsed_ms)
                last_tool_name = tool_name
                last_tool_parameters = step.tool_call.parsed_parameters
                SmartLogger.log(
                    "INFO",
                    "react.tool.request",
                    category="react.tool.request",
                    params=sanitize_for_log(
                        {
                            "react_run_id": run_id,
                            "iteration": state.iteration,
                            "phase": "acting",
                            "tool_name": tool_name,
                            "tool_parameters": step.tool_call.parsed_parameters,
                        }
                    ),
                )
                tool_started = time.perf_counter()
                tool_result: Optional[str] = None
                synthetic_tool_result = False

                # Soft-fail for missing SQL (prevents ToolExecutionError -> run termination)
                if tool_name in {"explain", "execute_sql_preview"}:
                    sql_text = (step.tool_call.parsed_parameters or {}).get("sql", "") or ""
                    if not sql_text.strip():
                        synthetic_tool_result = True
                        tool_result = (
                            '<tool_result><error code="missing_sql">sql parameter is required</error></tool_result>'
                        )
                        SmartLogger.log(
                            "WARNING",
                            "react.tool.synthetic_error",
                            category="react.tool.synthetic_error",
                            params=sanitize_for_log(
                                {
                                    "react_run_id": run_id,
                                    "iteration": state.iteration,
                                    "tool_name": tool_name,
                                    "reason": "missing_sql",
                                }
                            ),
                        )

                try:
                    if not synthetic_tool_result:
                        tool_result = await execute_tool(
                            tool_name=tool_name,
                            context=tool_context,
                            parameters=step.tool_call.parsed_parameters,
                        )
                except ToolExecutionError as exc:
                    tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000.0
                    iter_profile["tool_executed"] = True
                    iter_profile["tool_elapsed_ms"] = tool_elapsed_ms
                    totals["tool_elapsed_ms_sum"] += tool_elapsed_ms
                    SmartLogger.log(
                        "ERROR",
                        "react.tool.error",
                        category="react.tool.error",
                        params=sanitize_for_log(
                            {
                                "react_run_id": run_id,
                                "iteration": state.iteration,
                                "tool_name": tool_name,
                                "tool_parameters": step.tool_call.parsed_parameters,
                                "exception": repr(exc),
                                "traceback": traceback.format_exc(),
                                "state_snapshot": state.to_dict(),
                            }
                        ),
                    )
                    raise ReactAgentError(str(exc)) from exc
                except Exception as exc:
                    tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000.0
                    iter_profile["tool_executed"] = True
                    iter_profile["tool_elapsed_ms"] = tool_elapsed_ms
                    totals["tool_elapsed_ms_sum"] += tool_elapsed_ms
                    SmartLogger.log(
                        "ERROR",
                        "react.tool.error",
                        category="react.tool.error",
                        params=sanitize_for_log(
                            {
                                "react_run_id": run_id,
                                "iteration": state.iteration,
                                "tool_name": tool_name,
                                "tool_parameters": step.tool_call.parsed_parameters,
                                "exception": repr(exc),
                                "traceback": traceback.format_exc(),
                                "state_snapshot": state.to_dict(),
                            }
                        ),
                    )
                    raise
                tool_elapsed_ms = (time.perf_counter() - tool_started) * 1000.0
                last_tool_result = tool_result
                iter_profile["tool_executed"] = not synthetic_tool_result
                iter_profile["tool_elapsed_ms"] = None if synthetic_tool_result else tool_elapsed_ms
                if not synthetic_tool_result:
                    totals["tool_elapsed_ms_sum"] += tool_elapsed_ms
                SmartLogger.log(
                    "INFO",
                    "react.tool.response",
                    category="react.tool.response",
                    params=sanitize_for_log(
                        {
                            "react_run_id": run_id,
                            "iteration": state.iteration,
                            "phase": "observing",
                            "tool_name": tool_name,
                            "elapsed_ms": tool_elapsed_ms,
                            "tool_result": tool_result,
                        }
                    ),
                )
            
                # explain 호출 시 SQL을 추적
                if tool_name == "explain":
                    sql_text = step.tool_call.parsed_parameters.get("sql", "")
                    if sql_text and tool_result and (not _tool_result_has_error(tool_result)):
                        state.add_explained_sql(sql_text)
            
                state.current_tool_result = tool_result
                state.remaining_tool_calls -= 1

                # Update step with tool result
                step.tool_result = tool_result
            
                # Phase: observing - 도구 실행 완료
                if on_phase:
                    await on_phase(
                        "observing",
                        state.iteration,
                        {
                            "tool_name": tool_name,
                            "tool_result_preview": tool_result[:500] if tool_result else "",
                        },
                        state,
                    )
            
                if on_step:
                    await on_step(step, state)

            raise ReactAgentError("Iteration limit reached without completion.")
        except asyncio.CancelledError as exc:
            cancelled = True
            final_exception = exc
            final_traceback = traceback.format_exc()
            raise
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "react.run.error",
                category="react.run.error",
                params=sanitize_for_log(
                    {
                        "react_run_id": run_id,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                        "state_snapshot": state.to_dict(),
                        "last_input_xml": last_input_xml,
                        "last_llm_response": last_llm_response,
                        "last_tool_name": last_tool_name,
                        "last_tool_parameters": last_tool_parameters,
                        "last_tool_result": last_tool_result,
                    }
                ),
            )
            final_exception = exc
            final_traceback = traceback.format_exc()
            raise
        finally:
            agent_elapsed_ms = (time.perf_counter() - run_started) * 1000.0
            api_end_to_end_elapsed_ms: Optional[float] = None
            if api_started_perf_counter is not None:
                api_end_to_end_elapsed_ms = (
                    time.perf_counter() - float(api_started_perf_counter)
                ) * 1000.0
            total_elapsed_ms = (
                api_end_to_end_elapsed_ms
                if api_end_to_end_elapsed_ms is not None
                else agent_elapsed_ms
            )

            # Determine final status for summary
            if outcome is not None:
                status = outcome.status
            elif cancelled:
                status = "await_step_confirmation" if state.awaiting_step_confirmation else "cancelled"
            elif final_exception is not None:
                status = "error"
            else:
                status = "unknown"

            SmartLogger.log(
                "INFO",
                "react.run.done",
                category="react.run.done",
                params=sanitize_for_log(
                    {
                        "react_run_id": run_id,
                        "status": status,
                        "steps_count": len(steps),
                        "iterations_count": len(iteration_profiles),
                        "total_elapsed_ms": total_elapsed_ms,
                        "agent_elapsed_ms": agent_elapsed_ms,
                        "api_end_to_end_elapsed_ms": api_end_to_end_elapsed_ms,
                        "totals": totals,
                        "iterations": iteration_profiles,
                        # Minimal safe snapshot (level-0; excludes user_query/sql/tool_result/metadata)
                        "state": {
                            "dbms": state.dbms,
                            "prefer_language": state.prefer_language,
                            "max_sql_seconds": state.max_sql_seconds,
                            "remaining_tool_calls": state.remaining_tool_calls,
                            "iteration": state.iteration,
                            "step_confirmation_mode": state.step_confirmation_mode,
                            "awaiting_step_confirmation": state.awaiting_step_confirmation,
                        },
                        "exception": repr(final_exception) if final_exception else None,
                        "traceback": final_traceback,
                    }
                ),
            )

    async def stream(
        self,
        state: ReactSessionState,
        tool_context: ToolContext,
        max_iterations: Optional[int] = None,
        user_response: Optional[str] = None,
        react_run_id: Optional[str] = None,
        api_started_perf_counter: Optional[float] = None,
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

        async def on_phase_callback(
            phase: str, 
            iteration: int, 
            data: Dict[str, Any], 
            state_snapshot: ReactSessionState
        ) -> None:
            await queue.put({
                "type": "phase",
                "phase": phase,
                "iteration": iteration,
                "data": data,
                "state": state_snapshot.to_dict(),
            })

        async def on_token_callback(iteration: int, token: str) -> None:
            await queue.put({
                "type": "token",
                "iteration": iteration,
                "token": token,
            })

        async def on_format_repair_callback(iteration: int, reason: str) -> None:
            await queue.put({
                "type": "format_repair",
                "iteration": iteration,
                "reason": reason,
            })

        async def runner() -> None:
            try:
                await self.run(
                    state=state,
                    tool_context=tool_context,
                    max_iterations=max_iterations,
                    user_response=user_response,
                    react_run_id=react_run_id,
                    api_started_perf_counter=api_started_perf_counter,
                    on_step=on_step_callback,
                    on_outcome=on_outcome_callback,
                    on_phase=on_phase_callback,
                    on_token=on_token_callback,
                    on_format_repair=on_format_repair_callback,
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

    async def _call_llm(
        self,
        input_xml: str,
        *,
        react_run_id: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> str:
        messages = [
            SystemMessage(content=self.prompt_text),
            HumanMessage(content=input_xml),
        ]
        SmartLogger.log(
            "INFO",
            "react.llm.request",
            category="react.llm.request",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "iteration": iteration,
                    "phase": "thinking",
                    "model": getattr(self.llm, "model_name", None)
                    or getattr(self.llm, "model", None),
                    "user_prompt": input_xml,
                }
            ),
        )
        llm_started = time.perf_counter()
        try:
            response = await self.llm.ainvoke(messages)
        except Exception as exc:
            llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000.0
            SmartLogger.log(
                "ERROR",
                "react.llm.error",
                category="react.llm.error",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "iteration": iteration,
                        "phase": "thinking",
                        "elapsed_ms": llm_elapsed_ms,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                        "user_prompt": input_xml,
                    }
                ),
            )
            raise
        llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000.0
        if settings.is_add_delay_after_react_generator:
            delay_started = time.perf_counter()
            await asyncio.sleep(settings.delay_after_react_generator_seconds)
            delay_elapsed_ms = (time.perf_counter() - delay_started) * 1000.0
            SmartLogger.log(
                "INFO",
                "react.llm.post_delay",
                category="react.llm.post_delay",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "iteration": iteration,
                        "elapsed_ms": delay_elapsed_ms,
                        "delay_seconds": settings.delay_after_react_generator_seconds,
                    }
                ),
            )

        if isinstance(response.content, str):
            content = response.content
        elif isinstance(response.content, list):
            content = "\n".join(
                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                for part in response.content
            )
        else:
            content = str(response.content)

        SmartLogger.log(
            "INFO",
            "react.llm.response",
            category="react.llm.response",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "iteration": iteration,
                    "phase": "thinking",
                    "elapsed_ms": llm_elapsed_ms,
                    "assistant_response": content,
                }
            ),
        )
        return content

    async def _call_llm_streaming(
        self,
        input_xml: str,
        *,
        on_token: Callable[[int, str], Awaitable[None]],
        iteration: int,
        react_run_id: Optional[str] = None,
    ) -> str:
        """LLM을 스트리밍 모드로 호출하여 토큰 단위로 콜백합니다."""
        messages = [
            SystemMessage(content=self.prompt_text),
            HumanMessage(content=input_xml),
        ]
        SmartLogger.log(
            "INFO",
            "react.llm.request.streaming",
            category="react.llm.request.streaming",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "iteration": iteration,
                    "phase": "thinking",
                    "model": getattr(self.llm, "model_name", None)
                    or getattr(self.llm, "model", None)
                }
            ),
        )
        
        def _coerce_chunk_content_to_text(content: Any) -> str:
            """
            Normalize streaming chunk content into text.
            Some providers (e.g., Gemini via LangChain) may emit content as a list of dict parts like:
              [{'type': 'text', 'text': '...'}]
            """
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                out: List[str] = []
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        out.append(str(part.get("text") or ""))
                    else:
                        out.append(str(part))
                return "".join(out)
            if isinstance(content, dict) and "text" in content:
                return str(content.get("text") or "")
            return str(content)

        full_content_parts: List[str] = []
        try:
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    token_text = _coerce_chunk_content_to_text(chunk.content)
                    if not token_text:
                        continue
                    full_content_parts.append(token_text)
                    await on_token(iteration, token_text)
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "react.llm.streaming.error",
                category="react.llm.streaming.error",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "iteration": iteration,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
            )
            raise
        
        full_content = "".join(full_content_parts)
        if settings.is_add_delay_after_react_generator:
            await asyncio.sleep(settings.delay_after_react_generator_seconds)
        
        SmartLogger.log(
            "INFO",
            "react.llm.response.streaming",
            category="react.llm.response.streaming",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "iteration": iteration,
                    "total_length": len(full_content),
                }
            ),
        )
        return full_content

    def _parse_llm_response(
        self,
        response_text: str,
        iteration: int,
        *,
        react_run_id: Optional[str] = None,
        state_snapshot: Optional[Dict[str, Any]] = None,
    ) -> ReactStep:
        cleaned = response_text.strip()
        start_idx = cleaned.find("<")
        if start_idx > 0:
            cleaned = cleaned[start_idx:]

        cleaned = XmlUtil.sanitize_xml_text(cleaned)
        extract_info = _extract_first_output_xml(cleaned)
        if extract_info.get("trimmed"):
            start = int(extract_info.get("start") or 0)
            end = int(extract_info.get("end") or 0)
            SmartLogger.log(
                "WARNING",
                "react.llm.response.trimmed",
                category="react.llm.response.trimmed",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "iteration": iteration,
                        "original_len": len(cleaned),
                        "extracted_len": len(extract_info.get("xml") or ""),
                        "trim_prefix_len": start if start > 0 else 0,
                        "trim_suffix_len": (len(cleaned) - end) if end > 0 else 0,
                        "trim_suffix_preview": (cleaned[end : end + 200] if end > 0 else ""),
                    }
                ),
            )
        cleaned = str(extract_info.get("xml") or cleaned)
        extracted_before_repair = cleaned
        repaired = XmlUtil.repair_llm_xml_text(
            cleaned,
            text_tag_names=["note", "reasoning", "missing_info", "partial_sql"],
            repair_parameters_text_only=True,
        )
        if repaired != cleaned:
            # Log at ERROR so it's visible under default SmartLogger thresholds.
            SmartLogger.log(
                "ERROR",
                "react.llm.response.repaired",
                category="react.llm.response.repaired",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "iteration": iteration,
                        "assistant_response_raw": response_text,
                        "extracted_xml_before_repair": extracted_before_repair,
                        "repaired_xml": repaired,
                    }
                ),
            )
            cleaned = repaired
        try:
            root = self._parse_xml_with_recovery(cleaned, response_text)

            # Recovery wrapper may produce nested <output><output>...</output></output>.
            # If key sections are missing, unwrap to the inner output for best-effort parsing.
            if root.find("collected_metadata") is None:
                inner_output = root.find("output")
                if inner_output is not None:
                    SmartLogger.log(
                        "WARNING",
                        "react.llm.response.nested_output_unwrapped",
                        category="react.llm.response.nested_output_unwrapped",
                        params=sanitize_for_log(
                            {
                                "react_run_id": react_run_id,
                                "iteration": iteration,
                            }
                        ),
                    )
                    root = inner_output

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
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "react.llm.parse_error",
                category="react.llm.parse_error",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "iteration": iteration,
                        "assistant_response_raw": response_text,
                        "sanitized_xml": cleaned,
                        "extracted_xml_before_repair": extracted_before_repair,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                        "state_snapshot": state_snapshot,
                    }
                ),
            )
            raise

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
