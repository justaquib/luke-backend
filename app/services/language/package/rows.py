"""Validate dataset values and provenance before replacing stored rows."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Float, Integer, String

from app.services.language.package import dataset
from app.services.language.package.manifest import PackageError, safe_member, sha256_path


def validate(root: Path, expected: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    """Read the corpus and verify its counts, types and local references."""
    rows = dataset.load_rows(root)
    for name, table in dataset.TABLES.items():
        count = "pending_total" if name == "pending.jsonl" else name.removesuffix(".jsonl")
        if expected[count] != len(rows[name]):
            raise PackageError(f"The dataset count for {name} does not match the manifest.")
        for row in rows[name]:
            values = dataset._values(table, row)
            for field, value in values.items():
                _field(table.model.__table__.columns[field].type, field, value)
    _links(rows, "documents.jsonl", "document_id", "uploads.jsonl", "upload_id")
    _links(rows, "intents.jsonl", "faq_id", "faqs.jsonl", "faq_id")
    for upload in rows["uploads.jsonl"]:
        source = upload["source"]
        if not safe_member(source) or "/" in source:
            raise PackageError("An upload has an unsafe source name.")
        path = root / "raw" / source
        if not path.is_file() or path.stat().st_size != upload["size_bytes"]:
            raise PackageError(f"The raw upload {source} is missing or has the wrong size.")
        if sha256_path(path) != upload["content_hash"]:
            raise PackageError(f"The raw upload {source} does not match its content hash.")
    return rows


def _field(kind: Any, field: str, value: Any) -> None:
    valid = True
    if isinstance(kind, Boolean):
        valid = type(value) is bool
    elif isinstance(kind, Integer):
        valid = type(value) is int and value >= 0
    elif isinstance(kind, Float):
        valid = type(value) in (int, float) and math.isfinite(value)
    elif isinstance(kind, String):
        valid = isinstance(value, str)
    if not valid:
        raise PackageError(f"A dataset row has an invalid {field} value.")


def _links(rows: dict, source: str, key: str, target: str, target_key: str) -> None:
    identifiers = []
    for row in rows[target]:
        value = row.get(target_key)
        if type(value) is not int or value < 1 or value in identifiers:
            raise PackageError(f"The dataset {target} has an invalid or duplicate {target_key}.")
        identifiers.append(value)
    for row in rows[source]:
        value = row.get(key)
        if value is not None and (type(value) is not int or value not in identifiers):
            raise PackageError(f"A row in {source} refers to a missing {key}.")
