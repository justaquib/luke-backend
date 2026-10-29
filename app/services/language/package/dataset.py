"""The database rows a package carries, written as one JSON object per line.

The rows are kept whole, with their provenance and dates, so a restore rebuilds
the same corpus. The order is fixed and the natural key of each table decides
which repeated line is kept, so importing the same package twice leaves the
same rows. An upload's old id is carried so every chunk keeps pointing at the
file it came from after the rows are inserted again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.database import Document, Faq, Feedback, Intent, PendingQuestion, UploadedFile
from app.services.language.package.manifest import PackageError

DATASET_DIR = "dataset"


@dataclass(frozen=True)
class Table:
    """One table, the columns worth carrying, and the columns that are dates.

    ``blank`` names the text columns the app may store empty, so a row written
    without them comes back with ``""`` instead of breaking the restore.
    """

    model: type
    fields: tuple[str, ...]
    dates: tuple[str, ...]
    required: tuple[str, ...]
    blank: tuple[str, ...] = ()

    def dump(self, row: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        values = {name: _text(getattr(row, name, None)) for name in self.fields}
        return {**(extra or {}), **values}


DOCUMENTS = Table(
    Document,
    (
        "source",
        "chunk_index",
        "text",
        "document_id",
        "char_start",
        "char_end",
        "page",
        "content_hash",
        "language",
        "quality",
        "created_at",
    ),
    ("created_at",),
    ("source", "chunk_index", "text"),
)
FAQS = Table(
    Faq,
    ("question", "answer", "status", "confidence", "created_at", "updated_at"),
    ("created_at", "updated_at"),
    ("question", "answer", "status"),
)
PENDING = Table(
    PendingQuestion,
    ("question", "occurrences", "resolved", "created_at", "last_seen_at"),
    ("created_at", "last_seen_at"),
    ("question",),
)
FEEDBACK = Table(
    Feedback,
    ("message_id", "question", "answer", "source", "rating", "created_at"),
    ("created_at",),
    ("answer", "source", "rating"),
    blank=("question",),
)
UPLOADS = Table(
    UploadedFile,
    ("source", "content_hash", "size_bytes", "created_at"),
    ("created_at",),
    ("source", "content_hash", "size_bytes"),
)

INTENTS = Table(
    Intent,
    ("name", "response_template", "faq_id", "created_at", "updated_at"),
    ("created_at", "updated_at"),
    ("name", "response_template"),
)
TABLES = {
    "documents.jsonl": DOCUMENTS,
    "faqs.jsonl": FAQS,
    "pending.jsonl": PENDING,
    "feedback.jsonl": FEEDBACK,
    "uploads.jsonl": UPLOADS,
    "intents.jsonl": INTENTS,
}
INSERT_ORDER = (
    "uploads.jsonl",
    "documents.jsonl",
    "faqs.jsonl",
    "pending.jsonl",
    "feedback.jsonl",
    "intents.jsonl",
)


def jsonl(rows: list[dict[str, Any]]) -> bytes:
    """One JSON object per line, in the order given."""
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def parse(data: bytes, name: str) -> list[dict[str, Any]]:
    """The rows of one dataset file, refusing a line that is not an object."""
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PackageError(f"{name} line {number} is not valid JSON.") from error
        if not isinstance(row, dict):
            raise PackageError(f"{name} line {number} is not a JSON object.")
        rows.append(row)
    return rows


def dump_rows(
    db: Session, tenant_id: int | None, include_ids: bool = False
) -> dict[str, list[dict[str, Any]]]:
    """Every workspace row, grouped by the file it is written to.

    ``include_ids`` is used only by a rollback snapshot, so a failed import can
    put every row back under the id it already had.
    """
    rows: dict[str, list[dict[str, Any]]] = {}
    for name, table in TABLES.items():
        statement = select(table.model)
        statement = _scope(statement, table.model.tenant_id, tenant_id)
        dumped = []
        for row in db.execute(statement.order_by(table.model.id)).scalars():
            extra = {"upload_id": row.id} if table is UPLOADS else {}
            if table is FAQS:
                extra["faq_id"] = row.id
            if include_ids:
                extra["id"] = row.id
            dumped.append(table.dump(row, extra))
        rows[name] = dumped
    return rows


def load_rows(root: Path) -> dict[str, list[dict[str, Any]]]:
    """The rows a staged package carries, named by their dataset file."""
    folder = root / DATASET_DIR
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in TABLES:
        path = folder / name
        if not path.is_file():
            raise PackageError(f"The package is missing dataset/{name}.")
        rows[name] = parse(path.read_bytes(), name)
    return rows


def snapshot(db: Session, tenant_id: int | None) -> dict[str, list[dict[str, Any]]]:
    """The workspace rows as they are now, so a failed restore can put them back."""
    return dump_rows(db, tenant_id, include_ids=True)


def clear(db: Session, tenant_id: int | None) -> None:
    """Remove every workspace row the package owns, leaving other workspaces alone."""
    for name in reversed(INSERT_ORDER):
        model = TABLES[name].model
        db.execute(delete(model).where(_condition(model.tenant_id, tenant_id)))
    db.flush()


def restore(
    db: Session,
    tenant_id: int | None,
    rows: dict[str, list[dict[str, Any]]],
    dedupe: bool = False,
    keep_ids: bool = False,
) -> dict[str, int]:
    """Insert the package's rows in dependency order, skipping duplicates."""
    writer = _Writer(db, tenant_id, dedupe, keep_ids)
    counts = {"uploads": writer.insert_uploads(rows.get("uploads.jsonl", []))}
    for name in INSERT_ORDER[1:]:
        counts[name.removesuffix(".jsonl")] = writer.insert(TABLES[name], rows.get(name, []))
    db.flush()
    return counts


@dataclass
class _Writer:
    """Insert one workspace's rows, keeping the new upload ids for its chunks."""

    db: Session
    tenant_id: int | None
    dedupe: bool = True
    keep_ids: bool = False
    uploads: dict[Any, int] = field(default_factory=dict)
    faqs: dict[int, int] = field(default_factory=dict)

    def insert_uploads(self, rows: list[dict[str, Any]]) -> int:
        seen: set[tuple[str, str]] = set()
        kept = 0
        for row in rows:
            key = (str(row.get("source") or ""), str(row.get("content_hash") or ""))
            if not key[0] or (self.dedupe and key in seen):
                continue
            record = UploadedFile(tenant_id=self.tenant_id, **_values(UPLOADS, row))
            self._keep_id(record, row)
            self.db.add(record)
            self.db.flush()
            seen.add(key)
            self.uploads[row.get("upload_id")] = record.id
            kept += 1
        return kept

    def insert(self, table: Table, rows: list[dict[str, Any]]) -> int:
        seen: set[Any] = set()
        kept = 0
        for row in rows:
            values = _values(table, row)
            key = _natural_key(table, values)
            if self.dedupe and key is not None and key in seen:
                continue
            if table is DOCUMENTS and values.get("document_id") is not None:
                values["document_id"] = self.uploads[values["document_id"]]
            if table is INTENTS and values.get("faq_id") is not None:
                values["faq_id"] = self.faqs[values["faq_id"]]
            record = table.model(tenant_id=self.tenant_id, **values)
            self._keep_id(record, row)
            self.db.add(record)
            if table is FAQS and row.get("faq_id") is not None:
                self.db.flush()
                self.faqs[row["faq_id"]] = record.id
            if key is not None:
                seen.add(key)
            kept += 1
        self.db.flush()
        return kept

    def _keep_id(self, record: Any, row: dict[str, Any]) -> None:
        """A rollback restores the id a row already had, so nothing else moves."""
        if self.keep_ids and type(row.get("id")) is int:
            record.id = row["id"]


def _natural_key(table: Table, values: dict[str, Any]) -> Any:
    if table is DOCUMENTS:
        return (values.get("source"), values.get("chunk_index"))
    if table is FAQS or table is PENDING:
        return str(values.get("question") or "").strip().casefold()
    return None


def _values(table: Table, row: dict[str, Any]) -> dict[str, Any]:
    values = {name: row.get(name) for name in table.fields if row.get(name) is not None}
    missing = [name for name in table.required if values.get(name) in (None, "")]
    if missing:
        raise PackageError(f"A {table.model.__name__} row is missing {', '.join(missing)}.")
    for name in table.blank:
        values.setdefault(name, "")
    for name in table.dates:
        if isinstance(values.get(name), str):
            values[name] = _moment(values[name])
    return values


def _moment(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise PackageError(f"A package row has an unreadable date: {value!r}.") from error


def _text(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


def counts(db: Session, tenant_id: int | None) -> dict[str, int]:
    """The rows a workspace holds, by table, for the manifest and the check."""
    return {
        "documents": _count(db, Document, Document.tenant_id, tenant_id),
        "sources": _distinct_sources(db, tenant_id),
        "faqs": _count(db, Faq, Faq.tenant_id, tenant_id),
        "approved_faqs": _count(db, Faq, Faq.tenant_id, tenant_id, Faq.status == "approved"),
        "pending": _count(
            db,
            PendingQuestion,
            PendingQuestion.tenant_id,
            tenant_id,
            PendingQuestion.resolved.is_(False),
        ),
        "pending_total": _count(db, PendingQuestion, PendingQuestion.tenant_id, tenant_id),
        "feedback": _count(db, Feedback, Feedback.tenant_id, tenant_id),
        "uploads": _count(db, UploadedFile, UploadedFile.tenant_id, tenant_id),
        "intents": _count(db, Intent, Intent.tenant_id, tenant_id),
    }


def _count(db: Session, model: type, column: Any, tenant_id: int | None, extra: Any = None) -> int:
    statement = select(func.count()).select_from(model).where(_condition(column, tenant_id))
    if extra is not None:
        statement = statement.where(extra)
    return int(db.execute(statement).scalar_one() or 0)


def _distinct_sources(db: Session, tenant_id: int | None) -> int:
    statement = select(func.count(func.distinct(Document.source))).where(
        _condition(Document.tenant_id, tenant_id)
    )
    return int(db.execute(statement).scalar_one() or 0)


def _condition(column: Any, tenant_id: int | None) -> Any:
    return column.is_(None) if tenant_id is None else column == tenant_id


def _scope(statement: Any, column: Any, tenant_id: int | None) -> Any:
    return statement.where(_condition(column, tenant_id))
