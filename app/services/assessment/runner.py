"""Running the held-out bank against the loaded checkpoint, one item at a time.

A run is a background job like an upload: it answers every ``eval`` item, marks
each answer as it arrives, and publishes its measured progress to the run row
and to memory. The row also records the provenance a score needs to mean
anything — which checkpoint, which tokenizer, which generation settings, which
bank version and salt, and whether retrieval was disabled.

Two modes are kept apart on purpose. ``unaided`` never asks the corpus anything;
the evidence step exists only in the ``assisted`` branch, so no cache, fallback
or convenience call can quietly lend the model the answer. A rubric item is
answered and then waits for a trainer: its score stays null, which is
unassessed, not zero.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.db import database
from app.db.database import AssessmentResult, AssessmentRun
from app.services.assessment import bank, profiles, scoring
from app.services.assessment.conditions import (
    SAFE_FAILURE,
)
from app.services.assessment.conditions import (
    evidence as _evidence,
)
from app.services.assessment.conditions import (
    failure_message as _failure_message,
)
from app.services.assessment.conditions import (
    generate as _generate,
)
from app.services.assessment.conditions import (
    generation as _generation,
)
from app.services.assessment.conditions import (
    load_model as _load_model,
)
from app.services.assessment.conditions import (
    messages as _messages,
)
from app.services.assessment.conditions import (
    provenance as _provenance,
)

logger = logging.getLogger(__name__)

UNAIDED = "unaided"
ASSISTED = "assisted"
MODES = (UNAIDED, ASSISTED)

TERMINAL = frozenset({"done", "cancelled", "failed"})
WAITING = "Waiting to start"
INTERRUPTED = "Interrupted by a restart. Start a new run."

POLL_SECONDS = 0.5
MAX_POLLS = 1200
RECENT_RUNS = 20

_POOL: list[ThreadPoolExecutor] = []
_LIVE: dict[int, dict[str, Any]] = {}
_CANCELS: dict[int, threading.Event] = {}
_LOCK = threading.Lock()


def start(tenant_id: int | None, mode: str) -> int:
    """Create a run and hand it to the pool; the request returns its id."""
    run_id = create(tenant_id, mode)
    submit(run_id)
    return run_id


def create(tenant_id: int | None, mode: str) -> int:
    """Write the queued run with its provenance, before any model is loaded."""
    if mode not in MODES:
        raise ValueError(f"unknown assessment mode '{mode}'")
    total = len(bank.items(split="eval"))
    with database.SessionLocal() as db:
        run = AssessmentRun(
            mode=mode,
            status="queued",
            step=WAITING,
            total=total,
            tenant_id=tenant_id,
            **_provenance(),
        )
        db.add(run)
        db.commit()
        return int(run.id)


def submit(run_id: int) -> None:
    """Run one run on the bounded pool, and mark it failed if the work escapes."""
    with _LOCK:
        if not _POOL:
            _POOL.append(ThreadPoolExecutor(max_workers=1, thread_name_prefix="assessment"))
        pool = _POOL[0]
    pool.submit(_guarded, run_id)


def _guarded(run_id: int) -> None:
    try:
        execute(run_id)
    except Exception:  # the worker boundary
        # Anything that escaped execute still ends on the run row as a failure
        # instead of dying silently on the pool thread.
        logger.exception("Assessment run %s failed", run_id)
        mark_failed(run_id, SAFE_FAILURE)


def shutdown() -> None:
    """Stop accepting work and wait for the running run to finish."""
    with _LOCK:
        pool = _POOL.pop() if _POOL else None
    if pool is not None:
        pool.shutdown(wait=True)


def recover() -> int:
    """Mark the runs a previous process left queued or running as failed."""
    with database.SessionLocal() as db:
        statement = select(AssessmentRun).where(AssessmentRun.status.in_(("queued", "running")))
        runs = list(db.execute(statement).scalars())
        for run in runs:
            run.status = "failed"
            run.step = "Interrupted"
            run.error = INTERRUPTED
            run.finished_at = datetime.utcnow()
        db.commit()
    return len(runs)


def execute(run_id: int) -> None:
    """Answer every held-out item once, storing each result as it is marked."""
    run = _get(run_id)
    if run is None or run.status in TERMINAL:
        return
    mode = run.mode
    tenant_id = run.tenant_id
    total = int(run.total or 0)
    cancel = _cancel_event(run_id)
    if not _set_running(run_id):
        _drop_cancel(run_id)
        return
    done = 0
    try:
        model = _load_model()
        for item in bank.items(split="eval"):
            if cancel.is_set():
                break
            evidence = _evidence(item["prompt"], tenant_id) if mode == ASSISTED else ()
            started = time.perf_counter()
            answer, seconds = _generate(model, _messages(item, evidence), cancel)
            result = _record(item, answer, seconds, run_id, tenant_id)
            done += 1
            elapsed = round(time.perf_counter() - started, 4)
            step = f"{item['skill']} · item {done} of {total} · {elapsed:.1f}s"
            _store(run_id, result, done, total, step)
        _finish(run_id, "cancelled" if cancel.is_set() else "done")
    except Exception as error:  # noqa: BLE001 -- the worker boundary
        # The run boundary absorbs the failure, says which failure it was and
        # keeps the results already stored instead of discarding them.
        logger.warning("Assessment run %s stopped after %s items: %s", run_id, done, error)
        mark_failed(run_id, _failure_message(error))
    finally:
        _drop_cancel(run_id)


def as_dict(run: AssessmentRun) -> dict[str, Any]:
    """The run in the shape the page and the event stream read."""
    with _LOCK:
        live = dict(_LIVE.get(run.id, {}))
    return {
        "id": run.id,
        "mode": run.mode,
        "status": run.status,
        "step": str(live.get("step", run.step)),
        "progress": float(live.get("progress", run.progress)),
        "done": int(live.get("done", run.done)),
        "total": int(live.get("total", run.total)),
        "model_id": run.model_id,
        "model_artifact": run.model_artifact,
        "tokenizer_id": run.tokenizer_id,
        "generation": _generation(run),
        "bank": {
            "version": run.bank_version,
            "salt": run.bank_salt,
            "eval_items": int(run.total or 0),
        },
        "error": run.error,
        "created_at": _iso(run.created_at),
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
    }


def snapshot(run_id: int, tenant_id: int | None) -> dict[str, Any]:
    """The run as it is now, with the live progress of this process."""
    with database.SessionLocal() as db:
        return as_dict(profiles.owned(db, run_id, tenant_id))


def recent(tenant_id: int | None, limit: int = RECENT_RUNS) -> list[dict[str, Any]]:
    """The most recent runs of the workspace, newest first."""
    statement = select(AssessmentRun).order_by(AssessmentRun.id.desc()).limit(limit)
    if tenant_id is not None:
        statement = statement.where(AssessmentRun.tenant_id == tenant_id)
    with database.SessionLocal() as db:
        return [as_dict(run) for run in db.execute(statement).scalars()]


def cancel(run_id: int, tenant_id: int | None) -> dict[str, Any]:
    """Stop a queued or running run; a finished run is returned unchanged."""
    with database.SessionLocal() as db:
        run = profiles.owned(db, run_id, tenant_id)
        stopped_queued = False
        if run.status not in TERMINAL:
            stopped_queued = run.status != "running"
            run.status = "cancelled"
            run.step = "Stopped by the trainer."
            run.finished_at = datetime.utcnow()
            if not stopped_queued:
                _cancel_event(run_id).set()
            db.commit()
        with _LOCK:
            _LIVE.pop(run_id, None)
        if stopped_queued:
            _drop_cancel(run_id)
        return as_dict(run)


def mark_failed(run_id: int, message: str) -> None:
    """Record a failure that escaped the run's own handling."""
    with database.SessionLocal() as db:
        run = db.get(AssessmentRun, run_id)
        if run is None or run.status in TERMINAL:
            return
        run.status = "failed"
        run.step = "Failed"
        run.error = message
        run.finished_at = datetime.utcnow()
        db.commit()
    with _LOCK:
        _LIVE.pop(run_id, None)


def events(run_id: int, tenant_id: int | None) -> Iterator[tuple[str, dict[str, Any]]]:
    """The run as server-sent events, from the first read to its terminal state."""
    for _ in range(MAX_POLLS):
        payload = snapshot(run_id, tenant_id)
        yield "run", payload
        if payload["status"] in TERMINAL:
            return
        time.sleep(POLL_SECONDS)


def results(run_id: int, tenant_id: int | None) -> list[AssessmentResult]:
    """Every stored result of one run, in item order."""
    profiles.owns(run_id, tenant_id)
    statement = (
        select(AssessmentResult)
        .where(AssessmentResult.run_id == run_id)
        .order_by(AssessmentResult.item_id)
    )
    with database.SessionLocal() as db:
        return list(db.execute(statement).scalars())


def _record(
    item: dict[str, Any],
    answer: str,
    seconds: float,
    run_id: int,
    tenant_id: int | None,
) -> AssessmentResult:
    mark = scoring.mark(item, answer)
    return AssessmentResult(
        run_id=run_id,
        item_id=str(item["id"]),
        skill=str(item["skill"]),
        split=str(item["split"]),
        kind=str((item.get("expected") or {}).get("kind") or ""),
        stage=int(item["stage"]),
        topic=str(item["topic"]),
        response=answer[:8000],
        method=mark.method,
        score=mark.score,
        correct=mark.correct,
        latency_ms=round(seconds * 1000, 3),
        tenant_id=tenant_id,
    )


def _store(run_id: int, result: AssessmentResult, done: int, total: int, step: str) -> None:
    progress = round(done / total, 4) if total else 0.0
    with database.SessionLocal() as db:
        db.add(result)
        run = db.get(AssessmentRun, run_id)
        if run is not None and run.status not in TERMINAL:
            run.done = done
            run.progress = progress
            run.step = step
        db.commit()
    with _LOCK:
        _LIVE[run_id] = {"step": step, "done": done, "total": total, "progress": progress}


def _set_running(run_id: int) -> bool:
    """Mark the run running, unless a cancel finished it first."""
    with database.SessionLocal() as db:
        run = db.get(AssessmentRun, run_id)
        if run is None or run.status != "queued":
            return False
        run.status = "running"
        run.step = "Starting"
        run.started_at = datetime.utcnow()
        db.commit()
        return True


def _finish(run_id: int, status: str) -> None:
    with database.SessionLocal() as db:
        run = db.get(AssessmentRun, run_id)
        if run is None or run.status in TERMINAL:
            return
        run.status = status
        run.step = "Done" if status == "done" else "Stopped by the trainer."
        run.finished_at = datetime.utcnow()
        db.commit()
    with _LOCK:
        _LIVE.pop(run_id, None)


def _get(run_id: int) -> AssessmentRun | None:
    with database.SessionLocal() as db:
        return db.get(AssessmentRun, run_id)


def _cancel_event(run_id: int) -> threading.Event:
    with _LOCK:
        event = _CANCELS.get(run_id)
        if event is None:
            event = _CANCELS[run_id] = threading.Event()
        return event


def _drop_cancel(run_id: int) -> None:
    with _LOCK:
        event = _CANCELS.pop(run_id, None)
    if event is not None:
        event.set()


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None
