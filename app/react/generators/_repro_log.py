from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger


def _stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON dump for hashing & replay.
    - sort_keys: stable ordering
    - separators: compact but stable
    - default=str: avoid serialization failures
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(_stable_json_dumps(obj))


@dataclass(frozen=True)
class PromptMeta:
    prompt_file: str
    prompt_text: str

    @property
    def prompt_sha256(self) -> str:
        return sha256_text(self.prompt_text or "")

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_text or "")


def log_llm_repro(
    *,
    level: str,
    message: str,
    category: str,
    react_run_id: Optional[str],
    generator: str,
    llm_provider: Optional[str],
    llm_model: Optional[str],
    prompt: Optional[PromptMeta],
    input_payload: Any,
    messages_payload: Any,
    mode: Optional[str],
    elapsed_ms: Optional[float],
    response_raw: Any,
    parsed: Any,
    extra: Optional[Dict[str, Any]] = None,
    exception: Optional[BaseException] = None,
) -> None:
    """
    Reproducibility-first LLM logging.

    - Always includes:
      - prompt text + sha256
      - input payload + sha256
      - raw response + sha256 (best-effort string conversion)
      - parsed output (best-effort)
    - Uses max_inline_chars=0 so SmartLogger stores the full payload as a detail file (when file_output enabled).
    """
    try:
        raw_text = "" if response_raw is None else str(response_raw)
    except Exception:
        raw_text = "<unprintable_response>"

    params: Dict[str, Any] = {
        "react_run_id": react_run_id,
        "generator": generator,
        "llm": {"provider": llm_provider, "model": llm_model},
        "mode": mode,
        "elapsed_ms": elapsed_ms,
        "prompt": (
            {
                "file": prompt.prompt_file,
                "sha256": prompt.prompt_sha256,
                "len": prompt.prompt_len,
                "text": prompt.prompt_text,
            }
            if prompt is not None
            else None
        ),
        "input": {
            "sha256": sha256_json(input_payload),
            "payload": input_payload,
        },
        "messages": messages_payload,
        "response": {
            "sha256": sha256_text(raw_text),
            "raw": raw_text,
            "raw_type": (type(response_raw).__name__ if response_raw is not None else "None"),
        },
        "parsed": parsed,
    }
    if extra:
        params["extra"] = extra
    if exception is not None:
        params["exception"] = {
            "repr": repr(exception),
            "type": type(exception).__name__,
            "traceback": traceback.format_exc(),
        }

    SmartLogger.log(
        level,
        message,
        category=category,
        params=sanitize_for_log(params),
        max_inline_chars=0,
    )

