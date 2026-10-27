"""Knowledge transfer: move what a fine-tuned teacher knows into Luke.

A teacher was fine-tuned on this tenant's material and writes answers in its own
words. The transfer asks the teacher every question the material can ask, keeps
the answers that stay grounded in the reference they were given, and stores them
as approved FAQs. From then on Luke answers those questions without the teacher
loaded, so the knowledge outlives the weights.

One transfer runs at a time per tenant. Its progress and its report are plain
JSON, and the report remembers the FAQs it created so the trainer can undo it.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.db.database import Faq, SessionLocal
from app.services import vector_store
from app.services.huggingface import seeds as seed_service
from app.services.huggingface import serving, store

logger = logging.getLogger(__name__)

# What one transfer asks when the trainer does not choose a number.
DEFAULT_LIMIT = 40

# The most questions one transfer asks, so a run stays a reviewable batch.
MAX_LIMIT = 200

# How many sample questions the plan shows before the trainer starts.
SAMPLE_COUNT = 6

# A written answer must share at least this much of its content with the
# reference, or the teacher continued with facts of its own.
_MIN_MATERIAL_SHARE = 1 / 3

# A word long enough to name something in a reference or a written answer.
_CONTENT_WORD = re.compile(r"[a-z]{4,}")

_LOCK = threading.Lock()
_RUNNING: set[int] = set()


class TransferError(Exception):
    """The transfer cannot start or cannot be undone."""


class TransferBusyError(TransferError):
    """The machine is already busy with a run of this tenant."""


class TransferMissingError(TransferError):
    """The teacher the trainer named is not installed."""


def running(tenant_id: int | None) -> bool:
    """Whether a transfer of this tenant is going."""
    with _LOCK:
        return (tenant_id or 0) in _RUNNING


def plan(tenant_id: int | None, db: Session, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """The questions a transfer would ask, and the runs this tenant has had."""
    teachers = store.list_teachers(tenant_id)
    seeds = seed_service.seeds_for(tenant_id, db)
    active = running(tenant_id)
    record = store.read_json(store.transfer_path(tenant_id))
    return {
        "teachers": teachers,
        "available": len(seeds),
        "limit": _limit(limit),
        "by_kind": seed_service.count_kinds(seeds),
        "samples": [
            {"kind": seed.kind, "question": seed.question} for seed in seeds[:SAMPLE_COUNT]
        ],
        "running": active,
        "progress": record if active else None,
        "last": record if record and not active else None,
    }


def start(tenant_id: int | None, db: Session, folder: str, limit: int) -> dict[str, Any]:
    """Ask one teacher every question of this tenant's material in the background."""
    from app.services.huggingface import service as teacher_service

    if teacher_service.running(tenant_id):
        raise TransferBusyError("A base model is training; the transfer waits for it to finish.")
    teacher = next((row for row in store.list_teachers(tenant_id) if row["folder"] == folder), None)
    if teacher is None:
        raise TransferMissingError(f"{folder} is not installed.")
    seeds = seed_service.seeds_for(tenant_id, db)[: _limit(limit)]
    if not seeds:
        raise TransferError("There is no new material for the teacher to answer.")
    preview = plan(tenant_id, db, limit)
    key = tenant_id or 0
    with _LOCK:
        if key in _RUNNING:
            raise TransferBusyError("A knowledge transfer is already going.")
        _RUNNING.add(key)
    record = _first_frame(teacher, seeds)
    store.write_json(store.transfer_path(tenant_id), record)
    threading.Thread(
        target=_run,
        args=(tenant_id, folder, seeds, record),
        daemon=True,
        name=f"hf-transfer-{key}",
    ).start()
    return {**preview, "running": True, "progress": record, "last": None}


def undo(tenant_id: int | None, db: Session) -> dict[str, Any]:
    """Remove the FAQs the last transfer created, and nothing else."""
    if running(tenant_id):
        raise TransferBusyError("The transfer is going; undo it when it finishes.")
    record = store.read_json(store.transfer_path(tenant_id))
    ids = [int(value) for value in record.get("faq_ids") or []]
    if not ids:
        raise TransferError("The last transfer has no answers left to undo.")
    removed = 0
    for faq_id in ids:
        faq = db.get(Faq, faq_id)
        if faq is None or faq.tenant_id != tenant_id:
            continue
        db.delete(faq)
        vector_store.delete_faq(faq_id)
        removed += 1
    db.commit()
    record.update({"faq_ids": [], "undone": removed, "stage": "undone", "finished": store.now()})
    store.write_json(store.transfer_path(tenant_id), record)
    return plan(tenant_id, db)


def _limit(value: int) -> int:
    """The asked count, kept inside the range one run covers."""
    return min(MAX_LIMIT, max(1, int(value)))


def _first_frame(teacher: dict[str, Any], seeds: Sequence[seed_service.Seed]) -> dict[str, Any]:
    """The run's first frame, written before the thread starts."""
    return {
        "folder": str(teacher.get("folder") or ""),
        "model_id": str(teacher.get("model_id") or ""),
        "model_name": str(teacher.get("name") or teacher.get("folder") or ""),
        "started": store.now(),
        "finished": None,
        "stage": "loading",
        "total": len(seeds),
        "done": 0,
        "kept": 0,
        "skipped": 0,
        "by_kind": {},
        "last_question": "",
        "last_answer": "",
        "faq_ids": [],
        "undone": 0,
        "error": None,
        "seconds": 0.0,
    }


def _run(
    tenant_id: int | None,
    folder: str,
    seeds: Sequence[seed_service.Seed],
    record: dict[str, Any],
) -> None:
    """Ask the teacher each seed and keep the answers that stay on its material."""
    try:
        teacher = serving.load(tenant_id, folder)
        if teacher is None:
            raise TransferMissingError(f"{record['model_name']} could not be read.")
        record["stage"] = "asking"
        _write(tenant_id, record)
        db = SessionLocal()
        try:
            for seed in seeds:
                _step(db, tenant_id, teacher, seed, record)
        finally:
            db.close()
        record["stage"] = "done"
        _close(tenant_id, record)
    except Exception as error:  # a background boundary reports the failure on its record
        logger.exception("Knowledge transfer failed for tenant %s", tenant_id)
        record["stage"] = "failed"
        record["error"] = str(error) or error.__class__.__name__
        _close(tenant_id, record)
    finally:
        with _LOCK:
            _RUNNING.discard(tenant_id or 0)


def _step(
    db: Session,
    tenant_id: int | None,
    teacher: serving.Teacher,
    seed: seed_service.Seed,
    record: dict[str, Any],
) -> None:
    """Ask one question, keep its answer when it is grounded, and record the step."""
    record["done"] += 1
    record["last_question"] = seed.question
    answer = _written(teacher, seed)
    if answer and _grounded(answer, seed.reference):
        _keep(db, tenant_id, seed, answer, record)
    else:
        record["skipped"] += 1
    record["seconds"] = round(store.now() - float(record["started"]), 1)
    _write(tenant_id, record)


def _written(teacher: serving.Teacher, seed: seed_service.Seed) -> str:
    """The answer the teacher writes for one question, or nothing when it fails."""
    try:
        return (teacher.generate(seed.question, seed.reference) or "").strip()
    except (OSError, RuntimeError, ValueError):
        return ""


def _grounded(written: str, reference: str) -> bool:
    """Whether the written answer actually rewrites what it was given.

    A short reference holds one fact, so a written answer only has to share one
    of its words. A longer reference must share two, and a third of the answer.
    """
    material = set(_CONTENT_WORD.findall(reference.casefold()))
    words = set(_CONTENT_WORD.findall(written.casefold()))
    if not material:
        return bool(words)
    if not words:
        return False
    shared = len(words & material)
    if len(material) < 3:
        return shared >= 1
    return shared >= 2 and shared / len(words) >= _MIN_MATERIAL_SHARE


def _keep(
    db: Session,
    tenant_id: int | None,
    seed: seed_service.Seed,
    answer: str,
    record: dict[str, Any],
) -> None:
    """Store one answer as an approved FAQ, indexed with the other FAQs."""
    faq = Faq(
        question=seed.question.strip(),
        answer=answer,
        status="approved",
        tenant_id=tenant_id,
    )
    db.add(faq)
    db.flush()
    faq_id = faq.id
    vector_store.upsert_faq(faq_id, seed.question.strip(), answer, "approved", tenant_id)
    db.commit()
    record["kept"] += 1
    by_kind = record["by_kind"]
    by_kind[seed.kind] = by_kind.get(seed.kind, 0) + 1
    record["faq_ids"].append(faq_id)
    record["last_answer"] = answer


def _close(tenant_id: int | None, record: dict[str, Any]) -> None:
    """Write the last frame of a run."""
    record["finished"] = store.now()
    record["seconds"] = round(float(record["finished"]) - float(record["started"]), 1)
    _write(tenant_id, record)


def _write(tenant_id: int | None, record: dict[str, Any]) -> None:
    """Write the run's frame where the trainer reads it."""
    store.write_json(store.transfer_path(tenant_id), record)
