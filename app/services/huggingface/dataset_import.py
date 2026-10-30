"""Dataset import for the Hugging Face trainer.

Parses JSONL and CSV files of question-answer pairs and imports them as FAQs.
"""

from __future__ import annotations

import csv
import io
import json
import logging
from typing import Any

from app.schemas import FaqCreate
from app.services import faq

logger = logging.getLogger(__name__)


class DatasetImportError(Exception):
    """The dataset could not be parsed or imported."""


def import_dataset(tenant_id: int | None, text: str, filename: str, db: Any) -> dict[str, Any]:
    """Import a JSONL or CSV dataset of question-answer pairs."""
    pairs = _parse_dataset(text, filename)
    if not pairs:
        raise DatasetImportError("The file contains no question-answer pairs.")
    imported = 0
    errors = []
    for idx, pair in enumerate(pairs, start=1):
        question = (pair.get("question") or "").strip()
        answer = (pair.get("answer") or "").strip()
        if not question or not answer:
            errors.append(f"Row {idx}: missing question or answer")
            continue
        try:
            faq.create_faq(
                tenant_id,
                FaqCreate(question=question, answer=answer, status="approved"),
                db,
            )
            imported += 1
        except (ValueError, TypeError) as error:
            errors.append(f"Row {idx}: {error}")
            logger.warning("Failed to import row %d: %s", idx, error)
    return {"imported": imported, "errors": errors}


def _parse_dataset(text: str, filename: str) -> list[dict[str, Any]]:
    """Parse a JSONL or CSV file into a list of question-answer pairs."""
    if filename.endswith(".jsonl"):
        return _parse_jsonl(text)
    if filename.endswith(".csv"):
        return _parse_csv(text)
    raise DatasetImportError(f"Unsupported file format: {filename}")


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of question-answer pairs."""
    pairs = []
    for line_num, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                raise DatasetImportError(f"Line {line_num}: expected JSON object")
            pairs.append(obj)
        except json.JSONDecodeError as error:
            raise DatasetImportError(f"Line {line_num}: invalid JSON: {error}") from error
    return pairs


def _parse_csv(text: str) -> list[dict[str, Any]]:
    """Parse a CSV file into a list of question-answer pairs."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise DatasetImportError("CSV file has no header row")
    required = {"question", "answer"}
    if not required.issubset(set(reader.fieldnames)):
        missing = required - set(reader.fieldnames)
        raise DatasetImportError(f"CSV missing required columns: {', '.join(missing)}")
    return [dict(row) for row in reader]
