"""HTTP routes for the Hugging Face teachers, the model choice and predictions.

The routes stay thin: they check the caller, read the request, gather the
material a run trains on, and hand the work to the services that own it.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import User, get_session
from app.schemas import HuggingFaceTrainBody, PredictBody
from app.schemas.models import TransferBody
from app.services import model_choice
from app.services.huggingface import dataset_import, service, transfer
from app.services.training import dataset as datasets

router = APIRouter(prefix="/language-model", tags=["language-model"])


@router.get("/huggingface")
def huggingface_state(user: User = Depends(get_bearer_user)) -> dict[str, Any]:
    """The measured machine, the offered base models and the last run."""
    return service.view(tenant_id_of(user))


@router.get("/huggingface/dataset")
def huggingface_dataset(
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """The material a run would read, the split it is tagged with, and any leak."""
    prepared = datasets.prepared(tenant_id_of(user), db)
    return prepared.report() | {"leakage": list(datasets.leakage(prepared))}


@router.post("/huggingface/dataset")
async def upload_training_dataset(
    file: Annotated[UploadFile, File()],
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Upload a JSONL or CSV dataset of question-answer pairs for training."""
    if not file.filename:
        raise HTTPException(400, "No file provided.")
    filename = file.filename.lower()
    if not (filename.endswith(".jsonl") or filename.endswith(".csv")):
        raise HTTPException(400, "File must be .jsonl or .csv format.")
    try:
        content = await file.read()
        text = content.decode("utf-8")
        return dataset_import.import_dataset(tenant_id_of(user), text, filename, db)
    except UnicodeDecodeError as error:
        raise HTTPException(400, "File must be UTF-8 encoded.") from error
    except dataset_import.DatasetImportError as error:
        raise HTTPException(400, str(error)) from error


@router.post("/huggingface/train")
def train_with_huggingface(
    body: HuggingFaceTrainBody,
    user: User = Depends(get_bearer_user),
    db: Session = Depends(get_session),
) -> dict[str, Any]:
    """Carry out one named training operation on this tenant's material."""
    tenant_id = tenant_id_of(user)
    prepared = datasets.prepared(tenant_id, db)
    try:
        return service.start(tenant_id, body.model_id, body.operation, body.steps, prepared)
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


@router.get("/huggingface/search")
def search_huggingface_models(
    q: str,
    limit: int = 10,
    user: User = Depends(get_bearer_user),
) -> list[dict[str, Any]]:
    """Search HuggingFace for teacher models."""
    if not q or len(q.strip()) < 2:
        raise HTTPException(400, "Search query must be at least 2 characters.")
    try:
        return service.search_models(q.strip(), min(limit, 20))
    except service.BaseModelError as error:
        raise HTTPException(500, str(error)) from error


@router.post("/huggingface/teacher/download")
def download_huggingface_teacher(
    body: dict[str, str],
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Download a teacher model from HuggingFace."""
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(400, "model_id is required.")
    try:
        return service.download_teacher(tenant_id_of(user), model_id)
    except service.BaseModelBusyError as error:
        raise HTTPException(409, str(error)) from error
    except service.BaseModelError as error:
        raise HTTPException(500, str(error)) from error
