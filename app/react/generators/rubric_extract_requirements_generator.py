from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.react.generators._repro_log import PromptMeta, log_llm_repro
from app.react.prompts import get_prompt_text


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
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


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    cand = s[start : end + 1]
    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class RubricExtractRequirementsGenerator:
    _PROMPT_FILE = "rubric_extract_requirements_prompt.md"

    def __init__(self) -> None:
        self.system_prompt: str = get_prompt_text(self._PROMPT_FILE).strip()
        self.prompt_meta = PromptMeta(prompt_file=self._PROMPT_FILE, prompt_text=self.system_prompt)

    async def generate(self, *, llm: Any, question: str) -> Optional[Dict[str, Any]]:
        q = (question or "").strip()
        payload_obj = {"question": q}
        human_text = json.dumps(payload_obj, ensure_ascii=False)
        messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=human_text)]
        try:
            resp = await llm.ainvoke(messages)
        except Exception as exc:
            log_llm_repro(
                level="ERROR",
                message="react.llm.repro.rubric_extract_requirements.error",
                category="react.llm.repro.rubric",
                react_run_id=None,
                generator="rubric_extract_requirements_generator",
                llm_provider=getattr(llm, "provider", None),
                llm_model=getattr(llm, "model_name", None) or getattr(llm, "model", None),
                prompt=self.prompt_meta,
                input_payload=payload_obj,
                messages_payload={"system": self.system_prompt, "human": human_text},
                mode="json_text",
                elapsed_ms=None,
                response_raw=None,
                parsed=None,
                exception=exc,
            )
            raise
        text = _content_to_text(getattr(resp, "content", ""))
        obj = _extract_first_json_object(text)
        log_llm_repro(
            level="INFO",
            message="react.llm.repro.rubric_extract_requirements.ok",
            category="react.llm.repro.rubric",
            react_run_id=None,
            generator="rubric_extract_requirements_generator",
            llm_provider=getattr(llm, "provider", None),
            llm_model=getattr(llm, "model_name", None) or getattr(llm, "model", None),
            prompt=self.prompt_meta,
            input_payload=payload_obj,
            messages_payload={"system": self.system_prompt, "human": human_text},
            mode="json_text",
            elapsed_ms=None,
            response_raw=text,
            parsed=obj,
        )
        return obj


@lru_cache(maxsize=1)
def get_rubric_extract_requirements_generator() -> RubricExtractRequirementsGenerator:
    return RubricExtractRequirementsGenerator()

