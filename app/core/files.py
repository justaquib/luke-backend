"""File upload, text extraction, and preview conversion helpers."""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import HTTPException

ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".csv", ".xlsx", ".xls", ".docx", ".doc", ".pdf"}


def folder_size(path: str | Path) -> int:
    """The number of bytes held in a folder, or 0 when it is not there."""
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def extract_text_from_file(path: str, content_type: str | None = None) -> str:
    """Read the plain text held by a supported upload."""
    file_path = Path(path)
    extension = file_path.suffix.lower()
    if extension == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if extension == ".csv":
        with file_path.open(encoding="utf-8", errors="ignore", newline="") as handle:
            return "\n".join(" | ".join(row) for row in csv.reader(handle))
    if extension in (".xlsx", ".xls"):
        import pandas as pd

        frame = pd.read_excel(file_path)
        return "\n".join(" | ".join(map(str, row)) for row in frame.values.tolist())
    if extension == ".docx":
        return _docx_text(file_path)
    if extension == ".doc":
        return _convert_doc_to_text(file_path)
    if extension == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    raise HTTPException(415, f"Unsupported file type: {extension}")


def _docx_text(path: Path) -> str:
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())


def _libreoffice() -> str:
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if executable is None:
        raise HTTPException(500, "LibreOffice is required to read this file type")
    return executable


def _run_libreoffice(source: Path, target: str, output_dir: str) -> None:
    command = [
        _libreoffice(),
        "--headless",
        "--convert-to",
        target,
        "--outdir",
        output_dir,
        str(source),
    ]
    subprocess.run(command, check=True, capture_output=True)  # noqa: S603 -- fixed binary, no shell


def _convert_doc_to_text(path: Path) -> str:
    with tempfile.TemporaryDirectory() as directory:
        _run_libreoffice(path, "docx", directory)
        converted = Path(directory) / f"{path.stem}.docx"
        if not converted.exists():
            raise HTTPException(500, "Failed to convert DOC file")
        return _docx_text(converted)


def convert_for_preview(path: Path) -> Path | None:
    """Convert an office document to a temporary PDF when possible."""
    extension = path.suffix.lower()
    if extension not in {".docx", ".doc", ".xlsx", ".xls", ".csv", ".txt"}:
        return None
    with tempfile.TemporaryDirectory() as directory:
        try:
            _run_libreoffice(path, "pdf", directory)
        except (subprocess.CalledProcessError, HTTPException):
            return None
        converted = Path(directory) / f"{path.stem}.pdf"
        return converted if converted.exists() else None
