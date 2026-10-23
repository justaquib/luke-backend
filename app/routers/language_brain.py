"""HTTP routes for the assistance brain: what it knows and how it learns.

The routes stay thin: they check the caller, read the request, and hand the
work to the service that owns it.
"""

from __future__ import annotations

import multiprocessing
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.files import folder_size
from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import Faq, PendingQuestion, User, get_session
from app.schemas import CorrectionBody
from app.services import brain as brain_service
from app.services import documents as documents_service
from app.services import storage as storage_service
from app.services.language import corrections as correction_service
from app.services.language import lessons as lesson_service
from app.services.language import reporting
from app.services.language import runtime as runtime_service
from app.services.language import training_materials as materials_service
from app.services.language import training_report as training_report_service

router = APIRouter(prefix="/language-model", tags=["language-model"])


@router.get("/store")
def language_model_store(
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """What the tenant has stored: the files, the numbers and the size on disk."""
    tenant_id = tenant_id_of(user)
    documents = documents_service.list_documents(user, db)
    approved = db.execute(select(func.count(Faq.id)).where(Faq.status == "approved")).scalar_one()
    pending = db.execute(
        select(func.count(PendingQuestion.id)).where(PendingQuestion.resolved.is_(False))
    ).scalar_one()
    state = runtime_service.get_state(tenant_id)
    brain = reporting.brain_state(tenant_id)
    lessons = lesson_service.all_lesson_cards(tenant_id)
    model_report = storage_service.storage_report(tenant_id)
    return {
        "documents": documents,
        "totals": {
            "files": len(documents),
            "chunks": sum(int(item["chunks"]) for item in documents),
            "answers": int(approved),
            "pending": int(pending),
            "lessons": len(lessons),
            "vocabulary": int(brain.get("vocabulary", 0)),
        },
        "model": {
            "name": state.get("model_name", ""),
            "level": brain.get("model_level", {}).get("level", "0.0.0"),
            "title": brain.get("model_level", {}).get("title", ""),
            "vocabulary": int(brain.get("vocabulary", 0)),
            "parameters": int((brain.get("transformer") or {}).get("parameters", 0)),
            "trained": bool(brain.get("transformer_ready")),
            "steps": int((brain.get("transformer") or {}).get("steps", 0)),
            "loss": (brain.get("transformer") or {}).get("loss"),
            "corpus_sentences": int(model_report["model"]["corpus"]["sentences"]),
            "corpus_tokens": int(model_report["model"]["corpus"]["tokens"]),
        },
        "disk": {
            "learner_bytes": folder_size(runtime_service._DATA_DIR),
            "vector_store_bytes": folder_size(settings.chroma_persist_dir),
            "uploads_bytes": folder_size(settings.upload_dir),
        },
    }


@router.get("/brain")
def language_model_brain(user: User = Depends(get_bearer_user)) -> dict[str, Any]:
    """What the brain knows about itself, and the level it has reached."""
    return reporting.brain_state(tenant_id_of(user))


@router.post("/brain/test")
def test_language_model_level(
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Run the level tests again and report how the model did on each rung."""
    tenant_id = tenant_id_of(user)
    instruction = brain_service.remeasure_instruction(tenant_id)
    rungs = brain_service.rung_results(tenant_id, force=True)
    passed = sum(1 for rung in rungs if float(rung.get("rate", 0)) >= 0.6)
    return {
        "state": reporting.brain_state(tenant_id, force=True),
        "rungs": rungs,
        "instruction": instruction,
        "passed": passed,
        "total": len(rungs),
    }


@router.post("/brain/level/approve")
def approve_language_model_level(
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Promote the brain to the level it has measured, on the trainer's word."""
    tenant_id = tenant_id_of(user)
    try:
        promoted = brain_service.approve_level(tenant_id)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return {"state": reporting.brain_state(tenant_id, force=True), **promoted}


@router.post("/brain/train")
def train_language_model_brain(
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Start training the brain on this tenant's own material.

    The training takes a while, so it runs in its own process: the app stays
    usable while the brain learns, and the page is told that the brain is
    training so it can ask again until it is done.
    """
    tenant_id = tenant_id_of(user)
    materials = materials_service.gather(tenant_id, db)
    training = materials_service.training_set(materials, tenant_id)
    brain_service.mark_training(tenant_id, True)
    process = multiprocessing.get_context("spawn").Process(
        target=brain_service.train, args=(tenant_id, training), daemon=True
    )
    process.start()
    return {
        "state": reporting.brain_state(tenant_id),
        "sentence": None,
        "training": True,
    }


@router.get("/brain/sources")
def language_model_brain_sources(
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """What the brain has trained from, and how the training works."""
    tenant_id = tenant_id_of(user)
    writer = brain_service.writer_state(tenant_id)
    return training_report_service.tenant_report(tenant_id, db, writer)


@router.post("/brain/correction")
def correct_brain_answer(
    body: CorrectionBody,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Teach the assistant the right answer to a question it answered wrongly."""
    tenant_id = tenant_id_of(user)
    try:
        return correction_service.correct(
            tenant_id, body.question, body.answer, body.correction, db
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
