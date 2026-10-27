"""The Hugging Face teacher service: what is trained, what is running, what serves.

The routes read one view of the machine, start one run at a time, and ask a
teacher to answer or predict. The weights never leave this package's care.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from app.services.huggingface import catalog, serving, store, tuning


class BaseModelError(Exception):
    """The run cannot start, or the teacher cannot be read."""


class BaseModelBusyError(BaseModelError):
    """A run is already training on this machine."""


class BaseModelMissingError(BaseModelError):
    """The teacher the caller named is not installed."""


_LOCK = threading.Lock()
_RUNNING: dict[int, str] = {}


def running(tenant_id: int | None) -> bool:
    """Whether a run of this tenant is going."""
    with _LOCK:
        return (tenant_id or 0) in _RUNNING


def _machine() -> dict[str, Any]:
    """The measured machine, without promising more than it has."""
    import torch
    import transformers

    return {
        "device": tuning.device_name(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def view(tenant_id: int | None) -> dict[str, Any]:
    """The machine, the offered models, the teachers and the last run."""
    teachers = store.list_teachers(tenant_id)
    trained = {str(row.get("model_id") or "") for row in teachers}
    models = [{**model, "trained": model["id"] in trained} for model in catalog.base_models()]
    return {
        "machine": _machine(),
        "base_models": models,
        "teachers": teachers,
        "last_run": store.read_last_run(tenant_id),
        "progress": store.read_progress(tenant_id),
        "running": running(tenant_id),
    }


def start(
    tenant_id: int | None,
    model_id: str,
    steps: int,
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Fine-tune a base model in the background and return the run's state."""
    from app.services.huggingface import transfer

    if not catalog.known(model_id):
        raise BaseModelError(f"{model_id} is not one of the offered base models.")
    if transfer.running(tenant_id):
        raise BaseModelBusyError(
            "A knowledge transfer is reading the machine; wait for it to finish."
        )
    key = tenant_id or 0
    with _LOCK:
        if key in _RUNNING:
            raise BaseModelBusyError("A base model is already training on this machine.")
        _RUNNING[key] = model_id
    _record(tenant_id, model_id, steps)
    thread = threading.Thread(
        target=_run,
        args=(tenant_id, model_id, steps, list(pairs)),
        daemon=True,
        name=f"hf-teacher-{key}",
    )
    thread.start()
    return view(tenant_id)


def _record(tenant_id: int | None, model_id: str, steps: int) -> None:
    """Write the run's first frame before the thread starts."""
    store.write_json(
        store.progress_path(tenant_id),
        {
            "model_id": model_id,
            "started": store.now(),
            "finished": None,
            "stage": "starting",
            "stages": {"download": "waiting", "tune": "waiting"},
            "steps_done": 0,
            "steps_total": steps,
            "eta_seconds": None,
            "error": None,
            "loss": None,
            "seconds": 0.0,
        },
    )


def _run(
    tenant_id: int | None,
    model_id: str,
    steps: int,
    pairs: list[tuple[str, str]],
) -> None:
    """Train, save the weights, then record what the run measured."""
    folder = store.teacher_folder(model_id)
    started = store.now()
    stages = {"download": "waiting", "tune": "waiting"}

    def report(step: int, stage: str, loss: float | None) -> None:
        if stage in stages:
            stages[stage] = "running"
        if stage == "tune":
            stages["download"] = "done"
        elapsed = store.now() - started
        done = max(step, 0)
        eta = round(elapsed / done * (steps - done), 1) if done and stage == "tune" else None
        store.write_json(
            store.progress_path(tenant_id),
            {
                "model_id": model_id,
                "started": started,
                "finished": None,
                "stage": stage,
                "stages": dict(stages),
                "steps_done": done,
                "steps_total": steps,
                "eta_seconds": eta,
                "error": None,
                "loss": loss,
                "seconds": round(elapsed, 1),
            },
        )

    try:
        result = tuning.tune(
            model_id,
            store.teacher_path(tenant_id, folder),
            pairs,
            steps,
            report,
        )
        _finish(tenant_id, started, result)
    except Exception as error:  # the run reports any failure to the trainer
        _fail(tenant_id, model_id, started, error)
    finally:
        with _LOCK:
            _RUNNING.pop(tenant_id or 0, None)


def _finish(tenant_id: int | None, started: float, result: dict[str, Any]) -> None:
    finished = store.now()
    payload = {
        **result,
        "started": started,
        "finished": finished,
        "stages": {"download": "done", "tune": "done"},
        "steps_done": result.get("steps"),
        "steps_total": result.get("steps"),
        "eta_seconds": None,
        "error": None,
    }
    store.write_json(store.progress_path(tenant_id), payload)
    store.write_json(store.last_run_path(tenant_id), payload)
    store.write_json(
        store.teacher_path(tenant_id, store.teacher_folder(str(result["model_id"])))
        / "training.json",
        payload,
    )
    serving.forget()


def _fail(tenant_id: int | None, model_id: str, started: float, error: Exception) -> None:
    payload = {
        "model_id": model_id,
        "started": started,
        "finished": store.now(),
        "stage": "failed",
        "stages": {"download": "failed", "tune": "failed"},
        "steps_done": 0,
        "steps_total": 0,
        "eta_seconds": None,
        "error": str(error) or error.__class__.__name__,
        "loss": None,
        "seconds": round(store.now() - started, 1),
    }
    store.write_json(store.progress_path(tenant_id), payload)
    store.write_json(store.last_run_path(tenant_id), payload)


def remove_teacher(tenant_id: int | None, folder: str) -> dict[str, Any]:
    """Remove one teacher's weights from this machine."""
    from app.services.huggingface import transfer

    if running(tenant_id):
        raise BaseModelBusyError("A base model is training; the run is using the machine.")
    if transfer.running(tenant_id):
        raise BaseModelBusyError("A knowledge transfer is reading the teacher; wait for it.")
    if not store.remove_teacher(tenant_id, folder):
        raise BaseModelMissingError(f"{folder} is not installed.")
    serving.forget(folder)
    return view(tenant_id)


def teacher(tenant_id: int | None, model_id: str) -> dict[str, Any] | None:
    """The installed teacher the chat selected, when it is there."""
    if not model_id.startswith("hf:"):
        return None
    folder = model_id.removeprefix("hf:")
    for row in store.list_teachers(tenant_id):
        if row["folder"] == folder:
            return row
    return None


def answer(tenant_id: int | None, folder: str, question: str, reference: str) -> str | None:
    """The answer the teacher writes from the reference it was given."""
    loaded = serving.load(tenant_id, folder)
    if loaded is None:
        return None
    try:
        return loaded.generate(question, reference)
    except (OSError, RuntimeError, ValueError):
        return None


def predict(tenant_id: int | None, folder: str, text: str, limit: int) -> dict[str, Any] | None:
    """The next words and the attention of the teacher."""
    loaded = serving.load(tenant_id, folder)
    if loaded is None:
        return None
    try:
        return loaded.predict(text, limit)
    except (OSError, RuntimeError, ValueError):
        return None
