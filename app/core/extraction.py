"""The text of an upload, with the pages and characters it came from.

An index that only keeps the text cannot say where a passage was found. The
extractor therefore returns every page of a PDF, and the single run of text
every other format yields, each with the characters it occupies in the joined
text, so a passage can name its page and its offsets.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fastapi import HTTPException


@dataclass(frozen=True)
class PageText:
    """One page, and the characters of the joined text it holds."""

    page: int | None
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ExtractedText:
    """Every page of a document, joined into one text with their spans."""

    text: str
    pages: tuple[PageText, ...] = ()

    def page_at(self, offset: int) -> int | None:
        """The page a character offset falls on; None when no page span holds it."""
        for page in self.pages:
            if page.char_start <= offset < page.char_end:
                return page.page
        return None


def extract_document(path: str, content_type: str | None = None) -> ExtractedText:
    """Read a supported upload as text plus the pages it was read from."""
    file_path = Path(path)
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return _pdf_document(file_path)
    if extension == ".txt":
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    elif extension == ".csv":
        with file_path.open(encoding="utf-8", errors="ignore", newline="") as handle:
            text = "\n".join(" | ".join(row) for row in csv.reader(handle))
    elif extension in (".xlsx", ".xls"):
        import pandas as pd

        frame = pd.read_excel(file_path)
        text = "\n".join(" | ".join(map(str, row)) for row in frame.values.tolist())
    elif extension == ".docx":
        text = _docx_text(file_path)
    elif extension == ".doc":
        text = _convert_doc_to_text(file_path)
    else:
        raise HTTPException(415, f"Unsupported file type: {extension}")
    return _one_run(text)


def extract_text_from_file(path: str, content_type: str | None = None) -> str:
    """Read the plain text held by a supported upload."""
    return extract_document(path, content_type).text


def _one_run(text: str) -> ExtractedText:
    """A format without pages: its text is one span with no page number."""
    return ExtractedText(text=text, pages=(PageText(None, text, 0, len(text)),))


def _pdf_document(path: Path) -> ExtractedText:
    """Every page of a PDF, joined by one newline, with its own span."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[PageText] = []
    parts: list[str] = []
    offset = 0
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if parts:
            offset += 1
        pages.append(PageText(number, text, offset, offset + len(text)))
        parts.append(text)
        offset += len(text)
    joined = "\n".join(parts)
    return ExtractedText(text=joined, pages=tuple(pages))


def _docx_text(path: Path) -> str:
    from docx import Document as DocxDocument

    document = DocxDocument(str(path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())


def _libreoffice() -> str:
    executable = shutil.which("libreoffice") or shutil.which("soffice")
    if executable is None:
        raise HTTPException(500, "LibreOffice is required to read this file type")
    return executable


def run_libreoffice(source: Path, target: str, output_dir: str) -> None:
    """Convert a document with the installed LibreOffice, without a shell."""
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
        run_libreoffice(path, "docx", directory)
        converted = Path(directory) / f"{path.stem}.docx"
        if not converted.exists():
            raise HTTPException(500, "Failed to convert DOC file")
        return _docx_text(converted)
