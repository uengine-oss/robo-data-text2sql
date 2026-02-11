from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, ConfigDict

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.llm_factory import ReactLLMHandle, create_react_llm
from app.react.prompts import get_prompt_text
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        t = content.get("text")
        return str(t) if isinstance(t, str) else ""
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return "\n".join(lines[1:]).strip()


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = str(text or "").strip()
    if not s:
        return None
    left = s.find("{")
    right = s.rfind("}")
    if left < 0 or right <= left:
        return None
    cand = s[left : right + 1]
    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _clamp01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = float(default)
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class QueryQualityGateLLMOutput(BaseModel):
    """
    Structured output from LLM judge. Extra keys are forbidden to reduce prompt drift.
    """

    model_config = ConfigDict(extra="forbid")

    accept: bool = Field(default=False)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    summary: str = Field(default="")


@dataclass
class QueryQualityJudgeResult:
    accept: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    summary: str = ""
    # Diagnostics
    parse_error: str = ""


class QueryQualityGateGenerator:
    """
    Query quality gate (LLM judge) generator.
    Prompt: react/prompts/query_quality_gate_prompt.md
    Output: strict JSON parsed into a Pydantic model.
    """

    _PROMPT_FILE = "query_quality_gate_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)
        # Always use light model (cost/latency) for judge.
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="query_quality_gate",
            # Some light Gemini models reject thinking_level; omit it by passing None.
            thinking_level=None,
            include_thoughts=False,
            temperature=0.0,
            max_output_tokens=700,
            use_light=True,
        )

    async def judge_round(
        self,
        *,
        question: str,
        sql: str,
        row_count: Optional[int],
        execution_time_ms: Optional[float],
        metadata: Dict[str, Any],
        steps_tail: List[Dict[str, Any]],
        preview: Optional[Dict[str, Any]],
        round_idx: int,
        react_run_id: Optional[str] = None,
        purpose: str = "query_quality_gate",
    ) -> QueryQualityJudgeResult:
        q = (question or "").strip()
        s = (sql or "").strip()
        payload: Dict[str, Any] = {
            "question": q,
            "sql": s,
            "signals": {
                "row_count": row_count,
                "execution_time_ms": execution_time_ms,
                "preview": preview or {},
            },
            "metadata": metadata or {},
            "steps_tail": list(steps_tail or [])[:8],
            "round_idx": int(round_idx),
        }
        human_text = json.dumps(payload, ensure_ascii=False, default=str)
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=human_text)]

        started = time.perf_counter()
        try:
            # Keep purpose explicit for tracing even though llm handle is cached.
            resp = await self.llm_handle.llm.ainvoke(messages)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.query_quality_gate.error",
                category="react.llm.repro.query_quality_gate",
                react_run_id=react_run_id,
                generator="query_quality_gate_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=self.prompt_meta,
                input_payload=payload,
                messages_payload={"system": self.system_prompt, "human": human_text},
                mode="json_text",
                elapsed_ms=elapsed_ms,
                response_raw=None,
                parsed=None,
                exception=exc,
                extra={"purpose": purpose, "round_idx": int(round_idx)},
            )
            SmartLogger.log(
                "WARNING",
                "react.query_quality_gate.llm_failed",
                category="react.query_quality_gate",
                params=sanitize_for_log(
                    {
                        "react_run_id": react_run_id,
                        "purpose": purpose,
                        "round_idx": int(round_idx),
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                max_inline_chars=0,
            )
            return QueryQualityJudgeResult(
                accept=False,
                confidence=0.0,
                reasons=["llm_error"],
                risk_flags=["llm_error"],
                summary="",
                parse_error="llm_error",
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        text = _strip_code_fences(_content_to_text(getattr(resp, "content", None)).strip())
        obj = _extract_first_json_object(text) or _extract_first_json_object(text.replace("\n", " ")) or {}

        parsed: Optional[QueryQualityGateLLMOutput] = None
        parse_error = ""
        try:
            parsed = QueryQualityGateLLMOutput.model_validate(obj)
        except ValidationError as ve:
            parse_error = f"validation_error:{ve.errors()[0].get('type') if ve.errors() else 'unknown'}"
        except Exception as e:
            parse_error = f"parse_error:{type(e).__name__}"

        if parsed is None:
            # Fail-closed on parse errors.
            out = QueryQualityJudgeResult(
                accept=False,
                confidence=0.0,
                reasons=["parse_failed"],
                risk_flags=["parse_failed"],
                summary="",
                parse_error=parse_error or "parse_failed",
            )
        else:
            reasons = [str(x or "").strip()[:240] for x in (parsed.reasons or []) if str(x or "").strip()]
            risk_flags = [str(x or "").strip()[:120] for x in (parsed.risk_flags or []) if str(x or "").strip()]
            out = QueryQualityJudgeResult(
                accept=bool(parsed.accept),
                confidence=_clamp01(parsed.confidence, 0.0),
                reasons=reasons[:12],
                risk_flags=risk_flags[:12],
                summary=str(parsed.summary or "").strip()[:400],
                parse_error="",
            )

        SmartLogger.log(
            "INFO",
            "react.query_quality_gate.llm",
            category="react.query_quality_gate",
            params=sanitize_for_log(
                {
                    "react_run_id": react_run_id,
                    "purpose": purpose,
                    "round_idx": int(round_idx),
                    "elapsed_ms": elapsed_ms,
                    "ok": bool(out.accept),
                    "confidence": float(out.confidence),
                    "parse_error": out.parse_error,
                    "llm_provider": self.llm_handle.provider,
                    "llm_model": self.llm_handle.model,
                }
            ),
            max_inline_chars=0,
        )

        log_llm_repro(
            level=("INFO" if not out.parse_error else "WARNING"),
            message=("react.llm.repro.query_quality_gate.ok" if not out.parse_error else "react.llm.repro.query_quality_gate.parse_failed"),
            category="react.llm.repro.query_quality_gate",
            react_run_id=react_run_id,
            generator="query_quality_gate_generator",
            llm_provider=self.llm_handle.provider,
            llm_model=self.llm_handle.model,
            prompt=self.prompt_meta,
            input_payload=payload,
            messages_payload={"system": self.system_prompt, "human": human_text},
            mode="json_text",
            elapsed_ms=elapsed_ms,
            response_raw=text,
            parsed={"obj": obj, "result": out.__dict__},
            extra={"purpose": purpose, "round_idx": int(round_idx)},
        )
        return out


@lru_cache(maxsize=1)
def get_query_quality_gate_generator() -> QueryQualityGateGenerator:
    return QueryQualityGateGenerator()

