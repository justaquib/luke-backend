"""The checklist the training works through, kept where the page can read it.

A training runs in its own process, so the page cannot ask it how far it has
come: every part of the job is noted in a small file beside the model, and
the brain page reads that file while the training runs. It says which parts
are done, which one the training is on, how many of the model's steps have
happened, and how long the rest will take. When the training is finished the
file keeps the answer, so the panel can say how long it took.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from app.services.brain_paths import brain_dir, tenant_key

logger = logging.getLogger(__name__)

# The parts of a training, in the order they happen.
STAGES: tuple[str, ...] = ("words", "emotion", "intent", "feedback")

# What each part does, in plain words.
LABELS: dict[str, str] = {
    "words": "Trained the neural word model",
    "emotion": "Taught the emotion head",
    "intent": "Taught the intent head",
    "feedback": "Checked the answers it was told were wrong",
}

# The status a part can be in.
WAITING = "waiting"
WORKING = "working"
DONE = "done"


def progress_path(tenant_id: int | None) -> Path:
    """Where this tenant's training checklist is kept."""
    return brain_dir() / f"progress-{tenant_key(tenant_id)}.json"


def start(tenant_id: int | None) -> None:
    """Write a fresh checklist for a training that is beginning."""
    _write(
        tenant_id,
        {
            "started": time.time(),
            "stages": {key: WAITING for key in STAGES},
            "steps_done": 0,
            "steps_total": 0,
            "eta_seconds": None,
        },
    )


def stage(tenant_id: int | None, key: str, status: str) -> None:
    """Set one part of the training to working or done."""
    record = _read_raw(tenant_id)
    stages = record.get("stages")
    if isinstance(stages, dict) and key in stages:
        stages[key] = status
        _write(tenant_id, record)


def steps(
    tenant_id: int | None,
    done: int,
    total: int,
    eta_seconds: float | None,
    loss: float | None = None,
) -> None:
    """Note how many of the model's steps have happened, and what is left."""
    record = _read_raw(tenant_id)
    record["steps_done"] = int(done)
    record["steps_total"] = int(total)
    record["eta_seconds"] = round(eta_seconds, 1) if eta_seconds is not None else None
    if loss is not None:
        record["loss"] = loss
    _write(tenant_id, record)


def finish(tenant_id: int | None, stop_reason: str | None = None) -> float:
    """Close the checklist and return how long the training took, in seconds."""
    record = _read_raw(tenant_id)
    started = float(record.get("started") or 0)
    seconds = max(time.time() - started, 0.0) if started else 0.0
    record["finished"] = time.time()
    record["seconds"] = round(seconds, 1)
    record["eta_seconds"] = None
    if stop_reason:
        record["stop_reason"] = stop_reason
    _write(tenant_id, record)
    return seconds


def read(tenant_id: int | None) -> dict[str, Any]:
    """The checklist as the brain page reads it.

    An empty answer means a training has not begun since the file was last
    tidied, so the page says nothing about one.
    """
    record = _read_raw(tenant_id)
    if not record:
        return {}
    started = float(record.get("started") or 0)
    return {
        "started": round(started, 1),
        "finished": record.get("finished"),
        "seconds": record.get("seconds"),
        "eta_seconds": record.get("eta_seconds"),
        "steps_done": int(record.get("steps_done") or 0),
        "steps_total": int(record.get("steps_total") or 0),
        "loss": record.get("loss"),
        "stop_reason": record.get("stop_reason"),
        "stages": [
            {
                "key": key,
                "label": LABELS.get(key, key),
                "status": record["stages"].get(key, WAITING),
            }
            for key in STAGES
        ],
    }


def clear(tenant_id: int | None) -> None:
    """Tidy the checklist away, so the last training is not read as a new one."""
    progress_path(tenant_id).unlink(missing_ok=True)


def _read_raw(tenant_id: int | None) -> dict[str, Any]:
    path = progress_path(tenant_id)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return record if isinstance(record, dict) else {}


def _write(tenant_id: int | None, record: dict[str, Any]) -> None:
    path = progress_path(tenant_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
    except OSError as error:
        # A note that cannot be written is not worth stopping the training for.
        logger.warning("Could not write the training checklist %s: %s", path, error)
