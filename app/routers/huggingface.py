"""HTTP routes for the Hugging Face teachers, the model choice and predictions.

The routes stay thin: they check the caller, read the request, gather the
material a run trains on, and hand the work to the services that own it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import User, get_session
from app.schemas import HuggingFaceTrainBody, PredictBody
from app.schemas.models import TransferBody
from app.services import model_choice
from app.services.huggingface import service, transfer
from app.services.language import training_materials as materials_service

router = APIRouter(prefix="/language-model", tags=["language-model"])


@router.get("/huggingface")
def huggingface_state(user: User = Depends(get_bearer_user)) -> dict[str, Any]:
    """The measured machine, the offered base models and the last run."""
    return service.view(tenant_id_of(user))


@router.post("/huggingface/train")
def train_with_huggingface(
    body: HuggingFaceTrainBody,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Fine-tune a base model on this tenant's material in the background."""
    tenant_id = tenant_id_of(user)
    materials = materials_service.gather(tenant_id, db)
    training = materials_service.training_set(materials, tenant_id)
    try:
        return service.start(tenant_id, body.model_id, body.steps, training.instructions)
    except service.BaseModelBusyError as error:
        raise HTTPException(409, str(error)) from error
    except service.BaseModelError as error:
        raise HTTPException(400, str(error)) from error


@router.get("/huggingface/transfer")
def transfer_state(
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """The questions a transfer would ask, and the runs this tenant has had."""
    return transfer.plan(tenant_id_of(user), db)


@router.post("/huggingface/transfer")
def start_transfer(
    body: TransferBody,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Ask one installed teacher this tenant's material in the background."""
    try:
        return transfer.start(tenant_id_of(user), db, body.folder, body.limit)
    except transfer.TransferBusyError as error:
        raise HTTPException(409, str(error)) from error
    except transfer.TransferMissingError as error:
        raise HTTPException(404, str(error)) from error
    except transfer.TransferError as error:
        raise HTTPException(400, str(error)) from error


@router.post("/huggingface/transfer/undo")
def undo_transfer(
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Remove the answers the last transfer added, and nothing else."""
    try:
        return transfer.undo(tenant_id_of(user), db)
    except transfer.TransferBusyError as error:
        raise HTTPException(409, str(error)) from error
    except transfer.TransferError as error:
        raise HTTPException(400, str(error)) from error


@router.delete("/huggingface/teachers/{folder}")
def remove_huggingface_teacher(
    folder: str,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Remove one fine-tuned teacher's weights from this machine."""
    try:
        return service.remove_teacher(tenant_id_of(user), folder)
    except service.BaseModelBusyError as error:
        raise HTTPException(409, str(error)) from error
    except service.BaseModelMissingError as error:
        raise HTTPException(404, str(error)) from error
    except service.BaseModelError as error:
        raise HTTPException(400, str(error)) from error


@router.get("/models")
def answer_models(user: User = Depends(get_bearer_user)) -> dict[str, Any]:
    """The models the chat can answer with: Luke and the installed teachers."""
    return model_choice.catalog(tenant_id_of(user))


@router.post("/predict")
def predict_next_words(
    body: PredictBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """The next words and the attention of the selected model."""
    result = model_choice.predict(tenant_id_of(user), body.model, body.text, body.limit)
    if result is None:
        raise HTTPException(409, "This model has no trained weights to predict with yet.")
    return result
