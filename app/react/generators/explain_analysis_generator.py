import json
import logging
import re
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape as xml_escape

import asyncpg
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.react.llm_factory import ReactLLMHandle, create_react_llm
from app.react.prompts import get_prompt_text
from app.react.utils import XmlUtil
from app.react.utils.db_query_builder import ExecutionPlanResult, get_query_builder, TableMetadata
from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger

logger = logging.getLogger(__name__)


def _to_cdata(value: str) -> str:
    return f"<![CDATA[{value}]]>"


class ExplainAnalysisGeneratorError(Exception):
    """Raised when explain analysis generation fails."""


@dataclass
class ExplainAnalysisLLMResponse:
    verdict: str
    reason: str
    suggested_fixes: List[str]

    @classmethod
    def from_xml(
        cls,
        xml_text: str,
        *,
        react_run_id: Optional[str] = None,
    ) -> "ExplainAnalysisLLMResponse":
        raw = (xml_text or "").strip()

        normalized_info = _normalize_llm_explain_verdict_xml(raw)
        if normalized_info.get("trimmed") or normalized_info.get("steps"):
            SmartLogger.log(
                "WARNING",
                "react.explain.llm.response.normalized",
                category="react.explain.llm.response.normalized",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "steps": normalized_info.get("steps", []),
                        "trimmed": normalized_info.get("trimmed", False),
                        "original_len": len(raw),
                        "normalized_len": len(normalized_info.get("xml") or ""),
                        "trim_prefix_len": normalized_info.get("trim_prefix_len", 0),
                        "trim_suffix_len": normalized_info.get("trim_suffix_len", 0),
                        "trim_suffix_preview": normalized_info.get("trim_suffix_preview", ""),
                    }
                ),
            )

        raw = str(normalized_info.get("xml") or raw).strip()
        sanitized = XmlUtil.sanitize_xml_text(raw)
        repaired = XmlUtil.repair_llm_xml_text(
            sanitized,
            text_tag_names=["note", "verdict", "reason", "fix"],
            repair_parameters_text_only=False,
        )
        if repaired != sanitized:
            # Log at ERROR so it is visible under default SmartLogger thresholds.
            SmartLogger.log(
                "ERROR",
                "react.explain.llm.response.repaired",
                category="react.explain.llm.response.repaired",
                params=sanitize_for_log(
                    {
                        "repaired": True,
                        "raw_llm_text": raw,
                        "sanitized_xml_before_repair": sanitized,
                        "repaired_xml": repaired,
                    }
                ),
            )
            sanitized = repaired
        try:
            root = ET.fromstring(sanitized)
        except ET.ParseError as exc:
            raise ExplainAnalysisGeneratorError(
                f"LLM 응답 XML 파싱 실패: {exc}"
            ) from exc

        # New schema: <explain_verdict>...</explain_verdict> (no validation queries)
        if root.tag != "explain_verdict":
            candidate = root.find("explain_verdict")
            if candidate is None:
                # Backward compatibility: accept old validation_plan and map to FAIL with reason only.
                legacy = root.find("validation_plan")
                if legacy is not None:
                    legacy_reason = (legacy.findtext("./risk_analysis/summary") or "").strip()
                    return cls(
                        verdict="FAIL" if legacy_reason else "",
                        reason=legacy_reason,
                        suggested_fixes=[],
                    )
                raise ExplainAnalysisGeneratorError(
                    "LLM 응답에서 <explain_verdict> 노드를 찾을 수 없습니다."
                )
            root = candidate

        verdict = (root.findtext("verdict") or "").strip().upper()
        reason = (root.findtext("reason") or "").strip()
        fixes: List[str] = []
        for fix_el in root.findall("./suggested_fixes/fix"):
            text = (fix_el.text or "").strip()
            if text:
                fixes.append(text)

        # Per policy: reason/fixes should be present only when FAIL.
        if verdict == "PASS":
            reason = ""
            fixes = []

        return cls(verdict=verdict, reason=reason, suggested_fixes=fixes)


@dataclass
class ExplainAnalysisResult:
    input_sql: str
    execution_plan: ExecutionPlanResult
    table_metadata: List[TableMetadata]
    verdict: str
    fail_reason: str
    suggested_fixes: List[str] = field(default_factory=list)
    llm_raw_response: str = ""

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

        parts.append("<explain_verdict>")
        parts.append(f"<verdict>{xml_escape(self.verdict or '')}</verdict>")
        if (self.verdict or "").upper() == "FAIL":
            parts.append(f"<reason>{_to_cdata(self.fail_reason or '')}</reason>")
            parts.append("<suggested_fixes>")
            for fix in self.suggested_fixes or []:
                parts.append(f"<fix>{_to_cdata(str(fix))}</fix>")
            parts.append("</suggested_fixes>")
        parts.append("</explain_verdict>")
        parts.append("</explain_analysis_result>")
        return "\n".join(parts)


class ExplainAnalysisGenerator:
    """Generator that leverages explain_analysis_prompt to validate plans."""

    def __init__(self):
        self.prompt_text = get_prompt_text("explain_analysis_prompt.xml")
        self.prompt_meta = PromptMeta(prompt_file="explain_analysis_prompt.xml", prompt_text=self.prompt_text)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="explain-analysis",
            thinking_level=None,
            system_prompt=self.prompt_text,
            allow_context_cache=False,
            include_thoughts=False,
            temperature=0.0,
            use_light=True
        )
        self.db_type = settings.target_db_type

    async def generate(
        self,
        *,
        sql: str,
        db_conn: asyncpg.Connection,
        react_run_id: Optional[str] = None,
        max_sql_seconds: Optional[int] = None,
    ) -> ExplainAnalysisResult:
        sql_text = (sql or "").strip()
        if not sql_text:
            raise ValueError("SQL 문이 비어 있습니다.")

        builder = get_query_builder(self.db_type)
        plan_started = time.perf_counter()
        execution_plan = await builder.fetch_execution_plan(db_conn, sql_text, analyze=False)
        plan_elapsed_ms = (time.perf_counter() - plan_started) * 1000.0
        SmartLogger.log(
            "INFO",
            "react.explain.db_execution_plan",
            category="react.explain.db_execution_plan",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "elapsed_ms": plan_elapsed_ms,
                    "input_sql": sql_text,
                    "execution_plan_raw": execution_plan.raw_plan,
                }
            ),
        )

        meta_started = time.perf_counter()
        table_metadata = await builder.collect_table_metadata(db_conn, execution_plan.raw_plan)
        meta_elapsed_ms = (time.perf_counter() - meta_started) * 1000.0
        SmartLogger.log(
            "INFO",
            "react.explain.db_table_metadata",
            category="react.explain.db_table_metadata",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "elapsed_ms": meta_elapsed_ms,
                    "input_sql": sql_text,
                    "table_metadata": [asdict(m) for m in table_metadata],
                }
            ),
        )

        # Policy: decide PASS/FAIL by time limit. Only call LLM when FAIL to get reason/fixes.
        limit_ms: Optional[int] = None
        if max_sql_seconds is not None:
            try:
                limit_ms = int(max(1, int(max_sql_seconds)) * 1000)
            except Exception:
                limit_ms = None

        exec_ms = execution_plan.execution_time_ms
        is_over_limit = bool(limit_ms and exec_ms and exec_ms > limit_ms)

        llm_response_text = ""
        verdict = "FAIL" if is_over_limit else "PASS"
        fail_reason = ""
        suggested_fixes: List[str] = []
        if verdict == "FAIL":
            prompt_input = self._build_prompt_input(
                sql=sql_text,
                execution_plan=execution_plan,
                table_metadata=table_metadata,
                max_sql_seconds=int(max_sql_seconds or 1),
            )
            llm_response_text = await self._call_llm(prompt_input, react_run_id=react_run_id)
            parsed = ExplainAnalysisLLMResponse.from_xml(
                llm_response_text,
                react_run_id=react_run_id,
            )
            fail_reason = (parsed.reason or "").strip()
            suggested_fixes = list(parsed.suggested_fixes or [])

        return ExplainAnalysisResult(
            input_sql=sql_text,
            execution_plan=execution_plan,
            table_metadata=table_metadata,
            verdict=verdict,
            fail_reason=fail_reason,
            suggested_fixes=suggested_fixes,
            llm_raw_response=llm_response_text,
        )

    async def _call_llm(self, input_xml: str, *, react_run_id: Optional[str] = None) -> str:
        llm = self.llm_handle.llm
        messages = [SystemMessage(content=self.prompt_text), HumanMessage(content=input_xml)]
        SmartLogger.log(
            "INFO",
            "react.explain.llm.request",
            category="react.explain.llm.request",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "model": getattr(llm, "model_name", None) or getattr(llm, "model", None),
                    "user_prompt": input_xml,
                }
            ),
            max_inline_chars=0,
        )
        llm_started = time.perf_counter()
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000.0
            SmartLogger.log(
                "ERROR",
                "react.explain.llm.error",
                category="react.explain.llm.error",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "elapsed_ms": llm_elapsed_ms,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                        "user_prompt": input_xml,
                    }
                ),
                max_inline_chars=0,
            )
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.explain_analysis.error",
                category="react.llm.repro.explain_analysis",
                react_run_id=react_run_id,
                generator="explain_analysis_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload={"input_xml": input_xml},
                messages_payload={"system": self.prompt_text, "human": input_xml},
                mode="xml_text",
                elapsed_ms=llm_elapsed_ms,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            raise
        llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000.0
        if isinstance(response.content, str):
            content = response.content
        elif isinstance(response.content, list):
            content = "\n".join(
                part.get("text", "")
                if isinstance(part, dict)
                else str(part)
                for part in response.content
            )
        else:
            content = str(response.content)

        SmartLogger.log(
            "INFO",
            "react.explain.llm.response",
            category="react.explain.llm.response",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "elapsed_ms": llm_elapsed_ms,
                    "assistant_response": content,
                }
            ),
            max_inline_chars=0,
        )
        log_llm_repro(
            level="INFO",
            message="react.llm.repro.explain_analysis.ok",
            category="react.llm.repro.explain_analysis",
            react_run_id=react_run_id,
            generator="explain_analysis_generator",
            llm_provider=self.llm_handle.provider,
            llm_model=self.llm_handle.model,
            prompt=self.prompt_meta,
            input_payload={"input_xml": input_xml},
            messages_payload={"system": self.prompt_text, "human": input_xml},
            mode="xml_text",
            elapsed_ms=llm_elapsed_ms,
            response_raw=content,
            parsed={"assistant_response": content},
        )
        return content

    def _build_prompt_input(
        self,
        *,
        sql: str,
        execution_plan: ExecutionPlanResult,
        table_metadata: List[TableMetadata],
        max_sql_seconds: int,
    ) -> str:
        plan_json = json.dumps(execution_plan.raw_plan)
        parts: List[str] = ["<input>"]
        parts.append(f"<sql>{_to_cdata(sql)}</sql>")
        parts.append(f"<max_sql_seconds>{int(max_sql_seconds)}</max_sql_seconds>")
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


def _normalize_llm_explain_verdict_xml(text: str) -> dict:
    """
    Best-effort normalization for LLM responses that are supposed to contain a single
    <explain_verdict>...</explain_verdict> XML document.

    Handles common wrappers/noise:
    - Top-level <![CDATA[ ... ]]>
    - Markdown fences (```xml ... ```)
    - Outer wrapper tags like <output>...</output>
    - Prefix/suffix non-XML text before/after the desired block
    """
    original = (text or "")
    s = original.strip()
    steps: List[str] = []
    trimmed = False
    trim_prefix_len = 0
    trim_suffix_len = 0
    trim_suffix_preview = ""

    # 1) Extract first markdown fenced block if present.
    if "```" in s:
        fence_match = re.search(
            r"```(?:xml)?\s*[\r\n]+([\s\S]*?)\s*```",
            s,
            flags=re.IGNORECASE,
        )
        if fence_match:
            extracted = fence_match.group(1) or ""
            # Determine if we trimmed anything.
            start, end = fence_match.span()
            trimmed = trimmed or (start > 0 or end < len(s))
            trim_prefix_len += start
            trim_suffix_len += (len(s) - end)
            trim_suffix_preview = (s[end : end + 200] if end < len(s) else "")
            s = extracted.strip()
            steps.append("code_fence_extracted")

    # 2) Unwrap top-level CDATA.
    if s.startswith("<![CDATA[") and s.endswith("]]>"):
        inner = s[len("<![CDATA[") : -len("]]>")]
        # CDATA could contain leading/trailing newlines; keep trimming consistent.
        s = inner.strip()
        steps.append("top_level_cdata_unwrapped")

    # 3) Extract the first <explain_verdict>...</explain_verdict> block anywhere in the text.
    extract_info = _extract_first_tag_block(s, "explain_verdict")
    if extract_info.get("found"):
        extracted_xml = str(extract_info.get("xml") or "").strip()
        if extracted_xml and extracted_xml != s:
            trimmed = True
            # Prefer the extractor's prefix/suffix lengths for better diagnostics.
            trim_prefix_len = int(extract_info.get("trim_prefix_len") or 0)
            trim_suffix_len = int(extract_info.get("trim_suffix_len") or 0)
            trim_suffix_preview = str(extract_info.get("trim_suffix_preview") or "")
            s = extracted_xml
            steps.append("explain_verdict_extracted")

    return {
        "xml": s,
        "steps": steps,
        "trimmed": trimmed,
        "trim_prefix_len": trim_prefix_len,
        "trim_suffix_len": trim_suffix_len,
        "trim_suffix_preview": trim_suffix_preview,
    }


def _extract_first_tag_block(text: str, tag_name: str) -> dict:
    """
    Extract the first <tag_name>...</tag_name> block from text, even if surrounded by noise.
    Uses a depth counter over start/end tags (best-effort; tag nesting is not expected).
    """
    s = (text or "")
    start = s.find(f"<{tag_name}")
    if start < 0:
        return {"found": False, "xml": s, "trim_prefix_len": 0, "trim_suffix_len": 0, "trim_suffix_preview": ""}

    # Scan tags from the first occurrence.
    tag_iter = re.finditer(rf"</?{re.escape(tag_name)}\b", s[start:], flags=re.IGNORECASE)
    depth = 0
    end_inclusive = -1
    for m in tag_iter:
        is_close = (m.group(0) or "").startswith("</")
        if not is_close:
            depth += 1
            continue
        depth -= 1
        if depth <= 0:
            # Find the closing '>' for the end tag to be robust to whitespace like </tag  >
            close_tag_name_end = start + m.end()
            gt_idx = s.find(">", close_tag_name_end)
            if gt_idx >= 0:
                end_inclusive = gt_idx + 1
            else:
                # Fallback: assume canonical </tag_name> length
                end_inclusive = start + m.start() + len(f"</{tag_name}>")
            break

    if end_inclusive < 0:
        extracted = s[start:]
        trim_prefix_len = start if start > 0 else 0
        return {
            "found": True,
            "xml": extracted,
            "trim_prefix_len": trim_prefix_len,
            "trim_suffix_len": 0,
            "trim_suffix_preview": "",
        }

    extracted = s[start:end_inclusive]
    trim_prefix_len = start if start > 0 else 0
    trim_suffix_len = (len(s) - end_inclusive) if end_inclusive < len(s) else 0
    trim_suffix_preview = s[end_inclusive : end_inclusive + 200] if trim_suffix_len > 0 else ""
    return {
        "found": True,
        "xml": extracted,
        "trim_prefix_len": trim_prefix_len,
        "trim_suffix_len": trim_suffix_len,
        "trim_suffix_preview": trim_suffix_preview,
    }