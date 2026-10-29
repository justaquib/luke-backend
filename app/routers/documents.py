"""Document upload, preview, download, and feedback routes."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.files import convert_for_preview
from app.core.security import get_bearer_user
from app.core.workspace import current_workspace
from app.db.database import User, get_session
from app.schemas import (
    DocumentDeleteResponse,
    DocumentInfo,
    FeedbackCreate,
    FeedbackOut,
    IngestJobOut,
    IngestRequest,
    IngestResponse,
)
from app.services import documents, ingest_jobs, streaming, upload_jobs

router = APIRouter(tags=["documents"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    req: IngestRequest, authorization: str = Header(""), db: Session = Depends(get_session)
) -> IngestResponse:
    return documents.ingest_document(get_bearer_user(authorization, db), req, db)


@router.post("/upload", status_code=202)
def upload_file(
    file: UploadFile = File(...),
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> dict:
    """Accept the file and queue it; the page follows the job from there."""
    return upload_jobs.queue_upload(get_bearer_user(authorization, db), file, db)


@router.get("/ingest/jobs", response_model=list[IngestJobOut])
def list_ingest_jobs(user: User = Depends(get_bearer_user)) -> list[dict]:
    return ingest_jobs.recent(current_workspace())


@router.get("/ingest/jobs/{job_id}", response_model=IngestJobOut)
def get_ingest_job(job_id: int, user: User = Depends(get_bearer_user)) -> dict:
    return ingest_jobs.snapshot(job_id, current_workspace())


@router.get("/ingest/jobs/{job_id}/events")
def ingest_job_events(job_id: int, user: User = Depends(get_bearer_user)) -> StreamingResponse:
    """The job as server-sent events until it reaches a terminal state."""
    tenant_id = current_workspace()
    ingest_jobs.owns(job_id, tenant_id)
    return StreamingResponse(
        streaming.frames(ingest_jobs.events(job_id, tenant_id)),
        media_type="text/event-stream",
        headers=streaming.SSE_HEADERS,
    )


@router.post("/ingest/jobs/{job_id}/cancel", response_model=IngestJobOut)
def cancel_ingest_job(job_id: int, user: User = Depends(get_bearer_user)) -> dict:
    return ingest_jobs.cancel(job_id, current_workspace())


@router.post("/ingest/jobs/{job_id}/retry", response_model=IngestJobOut)
def retry_ingest_job(job_id: int, user: User = Depends(get_bearer_user)) -> dict:
    return upload_jobs.retry_upload(job_id, current_workspace())


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
def view_document(
    filename: str,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> FileResponse:
    user = get_bearer_user(authorization, db)
    path = documents.uploaded_file(user, filename, db)
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
def download_document(
    filename: str,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> FileResponse:
    user = get_bearer_user(authorization, db)
    path = documents.uploaded_file(user, filename, db)
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
