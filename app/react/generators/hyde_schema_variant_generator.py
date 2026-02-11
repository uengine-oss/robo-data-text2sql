from __future__ import annotations

import time
import traceback
from functools import lru_cache
from typing import Any, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.generators.hyde_schema_generator import (
    HydeSchemaOut,
    _sanitize_hyde_structured,
    _try_parse_hyde_json,
    build_hyde_embedding_text,
)
from app.react.llm_factory import ReactLLMHandle, create_react_llm
from app.react.prompts import get_prompt_text
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


class HydeSchemaVariantGenerator:
    """
    Generate HyDE schema hints with optional extra instructions appended to the base prompt.

    This exists mainly for p_terminal experiments (e.g., multiple HyDE variants).
    Base prompt is loaded from prompts/hyde_schema_prompt.md.
    """

    _PROMPT_FILE = "hyde_schema_prompt.md"

    def __init__(self) -> None:
        self.base_system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.base_system_prompt)
        self.llm_handle: ReactLLMHandle = create_react_llm(
            purpose="vector-hyde-rank.hyde-variants",
            thinking_level=None,
            include_thoughts=False,
            temperature=0.0,
            max_output_tokens=700,
            use_light=True,
        )

    async def generate(
        self,
        *,
        question: str,
        variant_instruction: str,
        react_run_id: Optional[str] = None,
    ) -> Tuple[str, str]:
        q = (question or "").strip()
        if not q:
            return "", "empty_question"
        system_prompt = (self.base_system_prompt or "").strip()
        inst = (variant_instruction or "").strip()
        if inst:
            system_prompt = system_prompt + "\n\n" + inst

        messages = [SystemMessage(content=system_prompt), HumanMessage(content=q)]
        mode = "json_text"

        started = time.perf_counter()
        try:
            llm = self.llm_handle.llm
            resp: Any
            try:
                if hasattr(llm, "with_structured_output"):
                    structured_llm = llm.with_structured_output(HydeSchemaOut)  # type: ignore[attr-defined]
                    resp = await structured_llm.ainvoke(messages)
                    mode = "structured"
                else:
                    resp = await llm.ainvoke(messages)
                    mode = "json_text"
            except Exception:
                resp = await llm.ainvoke(messages)
                mode = "json_text"
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.hyde_schema_variant.error",
                category="react.llm.repro.hyde_schema_variant",
                react_run_id=react_run_id,
                generator="hyde_schema_variant_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=system_prompt),
                input_payload={"question": q, "variant_instruction": inst},
                messages_payload={"system": system_prompt, "human": q},
                mode=mode,
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            SmartLogger.log(
                "WARNING",
                "react.hyde_schema_variant.llm_failed",
                category="react.hyde_schema_variant",
                params=sanitize_for_log(
                    {"react_run_id": react_run_id, "exception": repr(exc), "traceback": traceback.format_exc()}
                ),
                max_inline_chars=0,
            )
            return "", "llm_error"
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        out: Optional[HydeSchemaOut] = None
        if isinstance(resp, HydeSchemaOut):
            out = resp
        elif isinstance(resp, dict):
            try:
                out = HydeSchemaOut.model_validate(resp)
            except Exception:
                out = None
        if out is None:
            content = getattr(resp, "content", None)
            text = content if isinstance(content, str) else str(content or "")
            out = _try_parse_hyde_json(text)
        if out is None:
            log_llm_repro(
                level="WARNING",
                message="react.llm.repro.hyde_schema_variant.llm_empty",
                category="react.llm.repro.hyde_schema_variant",
                react_run_id=react_run_id,
                generator="hyde_schema_variant_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=system_prompt),
                input_payload={"question": q, "variant_instruction": inst},
                messages_payload={"system": system_prompt, "human": q},
                mode=mode,
                elapsed_ms=elapsed_ms,
                response_raw=getattr(resp, "content", None),
                parsed=None,
            )
            return "", "llm_empty"
        out = _sanitize_hyde_structured(out)
        embed_text = build_hyde_embedding_text(out)
        if not embed_text or len(embed_text) < 20:
            log_llm_repro(
                level="WARNING",
                message="react.llm.repro.hyde_schema_variant.embed_empty",
                category="react.llm.repro.hyde_schema_variant",
                react_run_id=react_run_id,
                generator="hyde_schema_variant_generator",
                llm_provider=self.llm_handle.provider,
                llm_model=self.llm_handle.model,
                prompt=PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=system_prompt),
                input_payload={"question": q, "variant_instruction": inst},
                messages_payload={"system": system_prompt, "human": q},
                mode=mode,
                elapsed_ms=elapsed_ms,
                response_raw=getattr(resp, "content", None),
                parsed={"structured": (out.model_dump() if hasattr(out, "model_dump") else str(out))},
            )
            return "", "llm_empty"

        SmartLogger.log(
            "INFO",
            "react.hyde_schema_variant.llm",
            category="react.hyde_schema_variant",
            params=sanitize_for_log(
                {"react_run_id": react_run_id, "elapsed_ms": elapsed_ms, "hyde_len": len(embed_text or ""), "mode": mode}
            ),
            max_inline_chars=0,
        )
        # Repro log: include raw response as best-effort (content or dict/pydantic dump)
        raw_content = getattr(resp, "content", None)
        if raw_content is None and isinstance(resp, dict):
            raw_content = resp
        elif raw_content is None and hasattr(resp, "model_dump"):
            try:
                raw_content = resp.model_dump()  # type: ignore[attr-defined]
            except Exception:
                raw_content = str(resp)
        log_llm_repro(
            level="INFO",
            message="react.llm.repro.hyde_schema_variant.ok",
            category="react.llm.repro.hyde_schema_variant",
            react_run_id=react_run_id,
            generator="hyde_schema_variant_generator",
            llm_provider=self.llm_handle.provider,
            llm_model=self.llm_handle.model,
            prompt=PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=system_prompt),
            input_payload={"question": q, "variant_instruction": inst},
            messages_payload={"system": system_prompt, "human": q},
            mode=mode,
            elapsed_ms=elapsed_ms,
            response_raw=raw_content,
            parsed={
                "structured": (out.model_dump() if hasattr(out, "model_dump") else str(out)),
                "hyde_embedding_text": embed_text,
            },
            extra={"hyde_len": len(embed_text or "")},
        )
        return (embed_text or "").strip(), ("llm_ok_structured" if mode == "structured" else "llm_ok_json")


@lru_cache(maxsize=1)
def get_hyde_schema_variant_generator() -> HydeSchemaVariantGenerator:
    return HydeSchemaVariantGenerator()

