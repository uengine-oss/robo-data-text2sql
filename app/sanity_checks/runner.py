from __future__ import annotations

import traceback
from typing import List

from app.smart_logger import SmartLogger
from app.sanity_checks.result import SanityCheckResult
from app.sanity_checks.checks.check_neo4j import check_neo4j
from app.sanity_checks.checks.check_db import check_target_db
from app.sanity_checks.checks.check_openai import check_openai
from app.sanity_checks.checks.check_google import check_google_if_react_provider_is_google


async def run_startup_sanity_checks_or_raise() -> List[SanityCheckResult]:
    """
    Run startup sanity checks (fail-fast).

    Raises:
        RuntimeError: if any required check fails.
    """
    checks = [
        check_neo4j(),
        check_target_db(),
        check_openai(),
        check_google_if_react_provider_is_google(),
    ]

    results: List[SanityCheckResult] = []
    for coro in checks:
        try:
            results.append(await coro)
        except Exception as exc:
            # Defensive: a check should return a failed result rather than raise,
            # but we still want a clean, logged failure mode.
            results.append(
                SanityCheckResult(
                    name="sanity_check_internal_error",
                    ok=False,
                    detail="A sanity check raised unexpectedly",
                    data=None,
                    error=repr(exc) + "\n" + traceback.format_exc(),
                )
            )

    failed = [r for r in results if not r.ok]

    for r in results:
        SmartLogger.log(
            "INFO" if r.ok else "ERROR",
            f"startup.sanity.{r.name}." + ("ok" if r.ok else "fail"),
            category="startup.sanity",
            params=r.to_log_params(),
            max_inline_chars=0,
        )

    if failed:
        SmartLogger.log(
            "CRITICAL",
            "startup.sanity.failed",
            category="startup.sanity",
            params={"failed": [f.name for f in failed]},
            max_inline_chars=0,
        )
        raise RuntimeError("Startup sanity checks failed. See logs for details.")

    SmartLogger.log("INFO", "startup.sanity.passed", category="startup.sanity", params=None, max_inline_chars=0)
    return results


