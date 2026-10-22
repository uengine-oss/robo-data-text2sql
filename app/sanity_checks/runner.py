from __future__ import annotations

import traceback
from typing import List

from app.config import settings
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
    # spec 007 T3 / SC-001 — 무엇이 기동을 막을 수 있는가.
    #
    # 인프라(neo4j·target_db)가 없으면 어떤 요청도 처리할 수 없으므로 치명으로 남긴다.
    # LLM·임베딩은 다르다 — 없으면 **데이터소스 질의만** 안 되고, 온톨로지 설계와
    # 그래프 조회는 계속 된다. 그것을 이유로 서버 전체를 안 띄우던 것이
    # 데스크톱 크래시 루프의 원인이었다 (CR-5b 결과 3).
    #
    # 되돌리기: STARTUP_SANITY_STRICT=true (설정 한 줄).
    fatal_checks = [
        ("neo4j", check_neo4j()),
        ("target_db", check_target_db()),
    ]
    advisory_checks = [
        ("openai", check_openai()),
        ("google", check_google_if_react_provider_is_google()),
    ]
    strict = bool(getattr(settings, "startup_sanity_strict", False))
    advisory_names = set() if strict else {name for name, _ in advisory_checks}
    checks = [coro for _, coro in fatal_checks] + [coro for _, coro in advisory_checks]

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

    failed = [r for r in results if not r.ok and r.name not in advisory_names]
    advisory_failed = [r for r in results if not r.ok and r.name in advisory_names]

    for r in results:
        SmartLogger.log(
            "INFO" if r.ok else ("WARNING" if r.name in advisory_names else "ERROR"),
            f"startup.sanity.{r.name}." + ("ok" if r.ok else "fail"),
            category="startup.sanity",
            params=r.to_log_params(),
            max_inline_chars=0,
        )

    if advisory_failed:
        # 조용히 넘기지 않는다. 무엇이 안 되는지 기동 로그에 그대로 적는다 —
        # 이 상태에서는 데이터소스 자연어 질의가 거부된다(FR-031).
        names = ", ".join(r.name for r in advisory_failed)
        print(
            f"[startup] LLM/임베딩 검사 실패({names}) — 서비스는 기동합니다. "
            "데이터소스 자연어 질의는 색인 상태에 따라 거부되고, "
            "온톨로지 설계·그래프 조회는 계속 사용할 수 있습니다. "
            "(엄격 모드로 되돌리려면 STARTUP_SANITY_STRICT=true)"
        )
        SmartLogger.log(
            "WARNING",
            "startup.sanity.advisory_failed",
            category="startup.sanity",
            params={"advisory_failed": [r.name for r in advisory_failed]},
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


