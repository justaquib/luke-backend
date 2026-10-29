"""Upload validation and preview conversion helpers."""

from __future__ import annotations

import io
import subprocess
import tempfile
import zipfile
from pathlib import Path

from fastapi import HTTPException, UploadFile

from app.core.extraction import extract_document, extract_text_from_file, run_libreoffice

__all__ = [
    "ALLOWED_UPLOAD_EXTENSIONS",
    "convert_for_preview",
    "extract_document",
    "extract_text_from_file",
    "folder_size",
    "read_upload",
    "sniff_extension",
]

ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".pdf"}

_ZIP_MARKERS = {".docx": "word/", ".xlsx": "xl/"}
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def read_upload(upload: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing anything larger than the limit."""
    content = upload.file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(
            413,
            f"{Path(upload.filename or 'the file').name} is larger than "
            f"{limit // (1024 * 1024)} MB.",
        )
    return content


def sniff_extension(filename: str, content: bytes) -> str | None:
    """The extension the bytes really are, or None when they do not match."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        return extension if content.startswith(b"%PDF-") else None
    if extension in _ZIP_MARKERS:
        if not content.startswith(b"PK\x03\x04"):
            return None
        return extension if _zip_contains(content, _ZIP_MARKERS[extension]) else None
    if extension in {".doc", ".xls"}:
        return extension if content.startswith(_OLE_MAGIC) else None
    if extension in {".txt", ".csv"}:
        return extension if b"\x00" not in content else None
    return None


def _zip_contains(content: bytes, marker: str) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            return any(name.startswith(marker) for name in archive.namelist())
    except zipfile.BadZipFile:
        return False


def folder_size(path: str | Path) -> int:
    """The number of bytes held in a folder, or 0 when it is not there."""
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def convert_for_preview(path: Path) -> Path | None:
    """Convert an office document to a temporary PDF when possible."""
    extension = path.suffix.lower()
    if extension not in {".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}:
        return None
    with tempfile.TemporaryDirectory() as directory:
        try:
            run_libreoffice(path, "pdf", directory)
        except (subprocess.CalledProcessError, HTTPException):
            return None
        converted = Path(directory) / f"{path.stem}.pdf"
        return converted if converted.exists() else None
