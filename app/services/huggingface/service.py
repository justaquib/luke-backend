"""The Hugging Face teacher service: what is trained, what is running, what serves.

The routes read one view of the machine, start one run at a time, and ask a
teacher to answer or predict. Every run is one of the Phase 7 operations, given
a dataset whose held-out side is checked before a weight moves; an adapter is
written to its own folder so it can never be mistaken for a teacher.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from app.services.huggingface import catalog, serving, store, tuning
from app.services.training import loop, operations
from app.services.training.dataset import Dataset

logger = logging.getLogger(__name__)

# The stages a run passes through, in order, as the page reads them.
STAGES = ("load", "tune", "save")


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


def _folder_of(tenant_id: int | None, model_id: str, operation: str) -> Path:
    """Where one operation writes, with an adapter kept out of the teacher's folder."""
    name = store.teacher_folder(model_id)
    if operation == operations.LORA_OPERATION:
        name += "-lora"
    return store.teacher_path(tenant_id, name)


def start(
    tenant_id: int | None,
    model_id: str,
    operation: str,
    steps: int,
    dataset: Dataset,
) -> dict[str, Any]:
    """Carry out one named operation in the background and return the run's state."""
    from app.services.huggingface import transfer

    if not catalog.known(model_id):
        raise BaseModelError(f"{model_id} is not one of the offered base models.")
    if transfer.running(tenant_id):
        raise BaseModelBusyError(
            "A knowledge transfer is reading the machine; wait for it to finish."
        )
    plan = operations.Plan(
        operation,
        model_id,
        dataset,
        _folder_of(tenant_id, model_id, operation),
        loop.Settings(steps=steps),
    )
    try:
        operations.review(plan)
    except operations.TrainingError as error:
        raise BaseModelError(str(error)) from error
    key = tenant_id or 0
    with _LOCK:
        if key in _RUNNING:
            raise BaseModelBusyError("A base model is already training on this machine.")
        _RUNNING[key] = model_id
    _record(tenant_id, plan)
    thread = threading.Thread(
        target=_run,
        args=(tenant_id, plan),
        daemon=True,
        name=f"hf-teacher-{key}",
    )
    thread.start()
    return view(tenant_id)


def _frame(plan: operations.Plan, started: float) -> dict[str, Any]:
    """What every progress frame of this run says about which run it is."""
    return {
        "operation": plan.operation,
        "kind": plan.kind,
        "model_id": plan.model_id,
        "name": catalog.name_of(plan.model_id),
        "folder": plan.folder.name,
        "started": started,
    }


def _record(tenant_id: int | None, plan: operations.Plan) -> None:
    """Write the run's first frame before the thread starts."""
    store.write_json(
        store.progress_path(tenant_id),
        {
            **_frame(plan, store.now()),
            "finished": None,
            "stage": "starting",
            "stages": dict.fromkeys(STAGES, "waiting"),
            "steps_done": 0,
            "steps_total": plan.settings.steps,
            "eta_seconds": None,
            "error": None,
            "loss": None,
            "seconds": 0.0,
        },
    )


def _run(tenant_id: int | None, plan: operations.Plan) -> None:
    """Train, save what the operation produces, then record what the run measured."""
    started = store.now()
    stages = dict.fromkeys(STAGES, "waiting")

    def report(step: int, stage: str, loss: float | None) -> None:
        if stage in stages:
            stages[stage] = "running"
        if stage in ("tune", "save"):
            stages["load"] = "done"
        if stage == "save":
            stages["tune"] = "done"
        elapsed = store.now() - started
        done = max(step, 0)
        total = plan.settings.steps
        eta = round(elapsed / done * (total - done), 1) if done and stage == "tune" else None
        store.write_json(
            store.progress_path(tenant_id),
            {
                **_frame(plan, started),
                "finished": None,
                "stage": stage,
                "stages": dict(stages),
                "steps_done": done,
                "steps_total": total,
                "eta_seconds": eta,
                "error": None,
                "loss": loss,
                "seconds": round(elapsed, 1),
            },
        )

    try:
        result = operations.run(plan, report)
        _finish(tenant_id, plan, started, result)
    except Exception as error:  # noqa: BLE001 -- the teacher training run boundary
        # The run reports any failure to the trainer.
        logger.warning("the %s %s run failed: %s", plan.model_id, plan.operation, error)
        _fail(tenant_id, plan, started, error, stages)
    finally:
        with _LOCK:
            _RUNNING.pop(tenant_id or 0, None)


def _finish(
    tenant_id: int | None,
    plan: operations.Plan,
    started: float,
    result: dict[str, Any],
) -> None:
    """Record a run that reached its last step, and the weights it wrote."""
    finished = store.now()
    written = bool(result.get("written"))
    steps = result.get("steps")
    payload = {
        **_frame(plan, started),
        **result,
        "finished": finished,
        "stage": "done" if written else str(result.get("stop") or "stopped"),
        "stages": (
            dict.fromkeys(STAGES, "done")
            if written
            else {"load": "done", "tune": str(result.get("stop") or "stopped"), "save": "waiting"}
        ),
        "steps_done": steps,
        "steps_total": plan.settings.steps,
        "eta_seconds": None,
        "error": None,
    }
    store.write_json(store.progress_path(tenant_id), payload)
    store.write_json(store.last_run_path(tenant_id), payload)
    if written:
        store.write_json(plan.folder / "training.json", payload)
    if plan.kind == operations.FULL_WEIGHTS:
        serving.forget()


def _failure_stages(stages: dict[str, str]) -> dict[str, str]:
    """The stages as they stood, with the one that was running marked failed."""
    marked = dict(stages)
    for key in STAGES:
        if marked.get(key) != "done":
            marked[key] = "failed"
            break
    return marked


def _fail(
    tenant_id: int | None,
    plan: operations.Plan,
    started: float,
    error: Exception,
    stages: dict[str, str],
) -> None:
    """Record a run that failed, keeping what it had measured by then."""
    standing = store.read_progress(tenant_id)
    payload = {
        **_frame(plan, started),
        "finished": store.now(),
        "stage": "failed",
        "stages": _failure_stages(stages),
        "steps_done": int(standing.get("steps_done") or 0),
        "steps_total": plan.settings.steps,
        "eta_seconds": None,
        "error": str(error) or error.__class__.__name__,
        "loss": standing.get("loss"),
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


def search_models(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Search HuggingFace for models matching the query."""
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        models = api.list_models(search=query, limit=limit)
        return [
            {
                "model_id": model.modelId,
                "name": model.modelId.split("/")[-1],
                "parameters": getattr(model, "parameters", None) or "Unknown",
                "downloads": getattr(model, "downloads", 0),
            }
            for model in models
        ]
    except Exception as error:
        logger.warning("HuggingFace search failed: %s", error)
        raise BaseModelError(f"Could not search HuggingFace: {error}") from error


def download_teacher(tenant_id: int | None, model_id: str) -> dict[str, Any]:
    """Download a teacher model from HuggingFace and register it."""
    if running(tenant_id):
        raise BaseModelBusyError("A base model is training; the machine is busy.")
    try:
        from huggingface_hub import snapshot_download

        folder_name = store.teacher_folder(model_id)
        folder = store.teacher_path(tenant_id, folder_name)
        folder.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_id, local_dir=folder, local_dir_use_symlinks=False)
        metadata = {
            "model_id": model_id,
            "name": model_id.rsplit("/", maxsplit=1)[-1],
            "folder": folder_name,
            "downloaded": store.now(),
        }
        store.write_json(folder / "metadata.json", metadata)
        logger.info("Downloaded teacher %s to %s", model_id, folder)
        return {"folder": folder_name, "model_id": model_id, "name": metadata["name"]}
    except Exception as error:
        logger.warning("HuggingFace download failed: %s", error)
        raise BaseModelError(f"Could not download model: {error}") from error
