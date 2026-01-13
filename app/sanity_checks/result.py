from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SanityCheckResult:
    name: str
    ok: bool
    detail: str = ""
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def to_log_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "name": self.name,
            "ok": self.ok,
            "detail": self.detail,
        }
        if self.data is not None:
            params["data"] = self.data
        if self.error:
            params["error"] = self.error
        return params


