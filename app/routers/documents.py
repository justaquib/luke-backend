"""Document upload, preview, download, and feedback routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.files import convert_for_preview
from app.core.security import get_bearer_user
from app.db.database import get_session
from app.schemas import (
    DocumentDeleteResponse,
    DocumentInfo,
    FeedbackCreate,
    FeedbackOut,
    IngestRequest,
    IngestResponse,
)
from app.services import documents

router = APIRouter(tags=["documents"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    req: IngestRequest, authorization: str = Header(""), db: Session = Depends(get_session)
) -> IngestResponse:
    return documents.ingest_document(get_bearer_user(authorization, db), req, db)


@router.post("/upload")
def upload_file(
    file: UploadFile = File(...),
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> dict:
    return documents.upload_document(get_bearer_user(authorization, db), file, db)


@router.get("/documents", response_model=list[DocumentInfo])
def list_documents(
    authorization: str = Header(""), db: Session = Depends(get_session)
) -> list[DocumentInfo]:
    return documents.list_documents(get_bearer_user(authorization, db), db)


@router.delete("/documents/{filename}", response_model=DocumentDeleteResponse)
def delete_document(
    filename: str, authorization: str = Header(""), db: Session = Depends(get_session)
) -> DocumentDeleteResponse:
    return documents.delete_document(get_bearer_user(authorization, db), filename, db)


@router.get("/documents/{filename}/view")
def view_document(filename: str) -> FileResponse:
    path = documents.uploaded_file(filename)
    if path.suffix.lower() == ".pdf":
        return _inline(path, filename)
    preview = convert_for_preview(path)
    if preview is not None:
        return _inline(preview, filename)
    return FileResponse(
        str(path), filename=filename, media_type="text/plain", headers=_inline_headers()
    )


def _inline(path: Path, filename: str) -> FileResponse:
    return FileResponse(
        str(path),
        filename=filename,
        media_type="application/pdf",
        headers=_inline_headers(),
    )


def _inline_headers() -> dict[str, str]:
    return {"Content-Disposition": "inline"}


@router.get("/documents/{filename}/download")
def download_document(filename: str) -> FileResponse:
    path = documents.uploaded_file(filename)
    return FileResponse(
        str(path),
        filename=filename,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/feedback", response_model=FeedbackOut)
def create_feedback(
    req: FeedbackCreate,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> FeedbackOut:
    return documents.create_feedback(get_bearer_user(authorization, db), req, db)
