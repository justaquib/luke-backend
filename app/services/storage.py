"""How much data the assistant holds, and where it sits.

The trainer's Storage page reads this: the records and bytes held in the
database, the vector index, the uploaded files, the chunks that were cut out
of them, and the files that hold the model itself.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sqlalchemy import inspect, text

from app.config import settings
from app.db.database import engine
from app.services import brain, language_logic, vector_store


def _directory_bytes(path: Path) -> tuple[int, int]:
    """The size of a folder on disk, and how many files it holds."""
    if not path.is_dir():
        return 0, 0
    total = 0
    count = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _database_report() -> dict[str, Any]:
    """The rows held in the database, and how big the database is."""
    tables: list[dict[str, Any]] = []
    total_rows = 0
    try:
        names = inspect(engine).get_table_names()
    except Exception:
        names = []
    with engine.connect() as connection:
        for name in names:
            try:
                rows = int(connection.execute(text(f'SELECT COUNT(*) FROM "{name}"')).scalar() or 0)
            except Exception:
                continue
            total_rows += rows
            tables.append({"name": name, "rows": rows})
        size_bytes = 0
        try:
            size_bytes = int(connection.execute(text("SELECT pg_database_size(current_database())")).scalar() or 0)
        except Exception:
            # Not a PostgreSQL server, so the size is not read this way.
            size_bytes = 0
    url = settings.database_url
    engine_name = "postgresql" if url.startswith("postgres") else "sqlite"
    return {
        "engine": engine_name,
        "tables": sorted(tables, key=lambda item: (-item["rows"], item["name"])),
        "rows": total_rows,
        "size_bytes": size_bytes,
    }


def _vector_report() -> dict[str, Any]:
    """What the vector index holds, and how much space it takes."""
    collections: list[dict[str, Any]] = []
    total_records = 0
    try:
        client = vector_store.get_client()
        for collection in client.list_collections():
            try:
                count = int(collection.count())
            except Exception:
                count = 0
            dimensions = 0
            try:
                sample = collection.get(limit=1, include=["embeddings"])
                embeddings = sample.get("embeddings") if isinstance(sample, dict) else None
                if embeddings is not None and len(embeddings):
                    dimensions = len(embeddings[0])
            except Exception:
                dimensions = 0
            total_records += count
            collections.append(
                {"name": collection.name, "records": count, "dimensions": dimensions}
            )
    except Exception:
        collections = []
    size_bytes, files = _directory_bytes(Path(settings.chroma_persist_dir))
    return {
        "collections": sorted(collections, key=lambda item: (-item["records"], item["name"])),
        "records": total_records,
        "size_bytes": size_bytes,
        "files": files,
        "directory": settings.chroma_persist_dir,
    }


def _files_report() -> dict[str, Any]:
    """The files that were uploaded, and how much room they take."""
    upload_dir = Path(settings.upload_dir)
    files: list[dict[str, Any]] = []
    total = 0
    if upload_dir.is_dir():
        for path in sorted(upload_dir.iterdir()):
            if not path.is_file():
                continue
            size = _file_bytes(path)
            total += size
            files.append({"name": path.name, "size_bytes": size})
    return {
        "files": files,
        "count": len(files),
        "size_bytes": total,
        "directory": settings.upload_dir,
    }


def _chunks_report(tenant_id: int | None) -> dict[str, Any]:
    """The chunks cut out of the documents, and the words they hold."""
    rows = 0
    characters = 0
    pages = 0
    try:
        with engine.connect() as connection:
            query = text("SELECT COUNT(*), COALESCE(SUM(LENGTH(text)), 0) FROM documents")
            result = connection.execute(query).first()
            if result:
                rows = int(result[0] or 0)
                characters = int(result[1] or 0)
            pages = int(
                connection.execute(text("SELECT COUNT(DISTINCT page_number) FROM documents")).scalar() or 0
            )
    except Exception:
        rows = 0
    sources = 0
    try:
        with engine.connect() as connection:
            sources = int(connection.execute(text("SELECT COUNT(DISTINCT source) FROM documents")).scalar() or 0)
    except Exception:
        sources = 0
    return {
        "chunks": rows,
        "characters": characters,
        "pages": pages,
        "sources": sources,
    }


def _model_report(tenant_id: int | None) -> dict[str, Any]:
    """The files that hold the model, and what is inside them."""
    brain_dir = brain.BRAIN_DIR
    files: list[dict[str, Any]] = []
    total = 0
    if brain_dir.is_dir():
        for path in sorted(brain_dir.iterdir()):
            if not path.is_file():
                continue
            size = _file_bytes(path)
            total += size
            files.append({"name": path.name, "size_bytes": size})

    state = brain.writer_state(tenant_id)
    transformer_bytes = _file_bytes(brain.writer_path(tenant_id))

    corpus = language_logic.training_corpus(tenant_id)
    tokens = sum(len(text.split()) for text in corpus)
    characters = sum(len(text) for text in corpus)

    return {
        "files": files,
        "file_count": len(files),
        "size_bytes": total,
        "directory": str(brain_dir),
        "transformer": {
            "trained": bool(state.get("trained")),
            "parameters": int(state.get("parameters") or 0),
            "vocabulary": int(state.get("vocabulary") or 0),
            "steps": int(state.get("steps") or 0),
            "loss": state.get("loss"),
            "device": state.get("device") or "cpu",
            "size_bytes": transformer_bytes,
        },
        "corpus": {
            "sentences": len(corpus),
            "tokens": tokens,
            "characters": characters,
        },
    }


def storage_report(tenant_id: int | None = None) -> dict[str, Any]:
    """The whole picture: what is held, where, and how big it is."""
    database = _database_report()
    vector = _vector_report()
    files = _files_report()
    chunks = _chunks_report(tenant_id)
    model = _model_report(tenant_id)

    total_bytes = (
        database["size_bytes"]
        + vector["size_bytes"]
        + files["size_bytes"]
        + model["size_bytes"]
    )
    return {
        "database": database,
        "vector": vector,
        "files": files,
        "chunks": chunks,
        "model": model,
        "totals": {
            "bytes": total_bytes,
            "records": database["rows"] + vector["records"],
            "files": files["count"] + vector["files"] + model["file_count"],
        },
    }
