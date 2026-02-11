"""
In-process background job queue for post-processing cache artifacts.

Design goals:
- Must not block the user response path (best-effort enqueue).
- Must not reuse request-scoped DB/Neo4j sessions; worker opens its own sessions.
- Simple in-process asyncio Queue (jobs may be lost on process restart).
"""

from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from app.config import settings
from app.smart_logger import SmartLogger
from app.react.utils.log_sanitize import sanitize_for_log


@dataclass
class CachePostprocessJob:
    payload: Dict[str, Any]


_queue: Optional[asyncio.Queue[Optional[CachePostprocessJob]]] = None
_worker_tasks: List[asyncio.Task] = []


def is_started() -> bool:
    return _queue is not None and len(_worker_tasks) > 0


async def start_cache_postprocess_workers() -> None:
    """
    Start background workers (idempotent).
    Must be called from FastAPI lifespan startup.
    """
    global _queue, _worker_tasks
    if is_started():
        return

    _queue = asyncio.Queue(maxsize=int(settings.cache_postprocess_queue_maxsize))
    worker_count = max(1, int(settings.cache_postprocess_worker_count))

    for idx in range(worker_count):
        _worker_tasks.append(asyncio.create_task(_worker_loop(idx), name=f"cache_postprocess_worker_{idx}"))

    SmartLogger.log(
        "INFO",
        "cache_postprocess.workers.started",
        category="cache_postprocess.workers",
        params=sanitize_for_log(
            {
                "worker_count": worker_count,
                "queue_maxsize": int(settings.cache_postprocess_queue_maxsize),
            }
        ),
        max_inline_chars=0,
    )


async def stop_cache_postprocess_workers() -> None:
    """Stop background workers (best-effort, idempotent)."""
    global _queue, _worker_tasks
    if _queue is None:
        return

    # Signal graceful shutdown via sentinel jobs.
    try:
        for _ in _worker_tasks:
            try:
                _queue.put_nowait(None)
            except asyncio.QueueFull:
                # If full, fall back to cancelling workers.
                break
    except Exception:
        pass

    # Cancel as a fallback (also handles the QueueFull case).
    for task in _worker_tasks:
        if not task.done():
            task.cancel()

    await asyncio.gather(*_worker_tasks, return_exceptions=True)
    _worker_tasks = []
    _queue = None

    SmartLogger.log(
        "INFO",
        "cache_postprocess.workers.stopped",
        category="cache_postprocess.workers",
        params=None,
        max_inline_chars=0,
    )


def enqueue_cache_postprocess(payload: Dict[str, Any]) -> bool:
    """Best-effort enqueue. Returns True if queued, False otherwise."""
    if _queue is None:
        SmartLogger.log(
            "WARNING",
            "cache_postprocess.enqueue.skipped_not_started",
            category="cache_postprocess.enqueue",
            params=sanitize_for_log({"reason": "not_started"}),
            max_inline_chars=0,
        )
        return False

    job = CachePostprocessJob(payload=payload)
    try:
        _queue.put_nowait(job)
        SmartLogger.log(
            "INFO",
            "cache_postprocess.enqueue.ok",
            category="cache_postprocess.enqueue",
            params=sanitize_for_log(
                {
                    "question": (payload.get("question") or "")[:80],
                    "react_run_id": payload.get("react_run_id"),
                }
            ),
            max_inline_chars=0,
        )
        return True
    except asyncio.QueueFull:
        SmartLogger.log(
            "WARNING",
            "cache_postprocess.enqueue.dropped_queue_full",
            category="cache_postprocess.enqueue",
            params=sanitize_for_log(
                {
                    "question": (payload.get("question") or "")[:80],
                    "react_run_id": payload.get("react_run_id"),
                }
            ),
            max_inline_chars=0,
        )
        return False
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "cache_postprocess.enqueue.error",
            category="cache_postprocess.enqueue",
            params=sanitize_for_log({"exception": repr(exc), "traceback": traceback.format_exc()}),
            max_inline_chars=0,
        )
        return False


async def _worker_loop(worker_idx: int) -> None:
    assert _queue is not None
    from app.core.cache_postprocess import process_cache_postprocess_payload

    while True:
        job: Optional[CachePostprocessJob] = None
        try:
            job = await _queue.get()
            if job is None:
                return

            await process_cache_postprocess_payload(job.payload)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            SmartLogger.log(
                "ERROR",
                "cache_postprocess.worker.error",
                category="cache_postprocess.worker",
                params=sanitize_for_log(
                    {
                        "worker_idx": worker_idx,
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                        "job_summary": {
                            "react_run_id": (job.payload.get("react_run_id") if job else None),
                            "question": ((job.payload.get("question") or "")[:80] if job else None),
                        },
                    }
                ),
                max_inline_chars=0,
            )
        finally:
            # Best-effort bookkeeping.
            try:
                if _queue is not None and job is not None:
                    _queue.task_done()
            except Exception:
                pass


