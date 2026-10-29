"""Rebuild the vector index from Postgres with cosine distance and stable ids.

The live collections were built before the index config, so they report the L2
space and search a slightly different notion of nearness than the embedding is
calibrated for. This reads the rows Postgres keeps, writes the same text into
fresh collections with the cosine space and UUID5 ids, checks the counts and a
sample question, then swaps the canonical collections. The old index directory
is copied under ``backups/`` first, and the sqlite file is vacuumed afterwards.

Run ``--dry-run`` first: it reports exactly what would be rebuilt and writes
only the report. ``--yes`` takes the backup, rebuilds and vacuums. If the
result is wrong, ``--rollback <report folder> --yes`` parks the current index
and copies the backed-up one back.

    .venv/bin/python scripts/rebuild_index.py --dry-run
    .venv/bin/python scripts/rebuild_index.py --yes
    .venv/bin/python scripts/rebuild_index.py --rollback backups/index-rebuild-... --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SUFFIX = "-rebuild"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, no backup")
    parser.add_argument("--yes", action="store_true", help="back up, then rebuild")
    parser.add_argument("--rollback", metavar="FOLDER", help="restore that report folder")
    return parser.parse_args()


def _tenant(tenant_id: int | None) -> dict[str, str]:
    return {} if tenant_id is None else {"tenant_id": str(tenant_id)}


def _document_payload() -> dict[str, Any]:
    from sqlalchemy import select

    from app.db.database import Document, SessionLocal
    from app.services import chunk_vectors
    from app.services.provenance import content_hash

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows = 0
    with SessionLocal() as db:
        for row in db.execute(
            select(Document).order_by(Document.source, Document.chunk_index, Document.id)
        ).scalars():
            rows += 1
            identifier = chunk_vectors.chunk_id(row.tenant_id, row.source, row.text)
            if identifier in seen:
                continue
            seen.add(identifier)
            metadata: dict[str, Any] = {"source": row.source, **_tenant(row.tenant_id)}
            for key, value in (
                ("document_id", row.document_id),
                ("chunk_index", row.chunk_index),
                ("char_start", row.char_start),
                ("char_end", row.char_end),
                ("page_number", row.page),
                ("content_hash", row.content_hash or content_hash(row.text)),
                ("language", row.language),
                ("quality", row.quality),
            ):
                if value is not None:
                    metadata[key] = value
            ids.append(identifier)
            texts.append(row.text)
            metadatas.append(metadata)
    return {
        "ids": ids,
        "documents": texts,
        "metadatas": metadatas,
        "rows": rows,
        "duplicates": rows - len(ids),
    }


def _faq_payload() -> dict[str, Any]:
    from sqlalchemy import select

    from app.db.database import Faq, SessionLocal

    ids: list[str] = []
    questions: list[str] = []
    metadatas: list[dict[str, Any]] = []
    with SessionLocal() as db:
        for row in db.execute(select(Faq).order_by(Faq.id)).scalars():
            ids.append(str(row.id))
            questions.append(row.question)
            metadatas.append(
                {
                    "answer": row.answer,
                    "faq_id": row.id,
                    "status": row.status,
                    "confidence": row.confidence,
                    **_tenant(row.tenant_id),
                }
            )
    return {"ids": ids, "documents": questions, "metadatas": metadatas, "rows": len(ids)}


def _intent_payload() -> dict[str, Any]:
    from sqlalchemy import select

    from app.db.database import Intent, PendingQuestion, SessionLocal

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    with SessionLocal() as db:
        for row in db.execute(
            select(PendingQuestion)
            .where(PendingQuestion.resolved.is_(False))
            .order_by(PendingQuestion.id)
        ).scalars():
            ids.append(f"pending-{row.id}")
            documents.append(row.question)
            metadatas.append({"kind": "pending", "pending_id": row.id, **_tenant(row.tenant_id)})
        for row in db.execute(select(Intent).order_by(Intent.id)).scalars():
            metadata = {
                "kind": "intent",
                "name": row.name,
                "response_template": row.response_template,
                **_tenant(row.tenant_id),
            }
            if row.faq_id is not None:
                metadata["faq_id"] = row.faq_id
            ids.append(f"intent-{row.id}")
            documents.append(row.name)
            metadatas.append(metadata)
    return {"ids": ids, "documents": documents, "metadatas": metadatas, "rows": len(ids)}


def _payloads() -> dict[str, dict[str, Any]]:
    return {
        "docs": _document_payload(),
        "faq": _faq_payload(),
        "intents": _intent_payload(),
    }


def _tree_size(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def _collection_state(client: Any, names: list[str]) -> dict[str, Any]:
    from app.services.retrieval import similarity_from_distance

    state: dict[str, Any] = {}
    existing = {collection.name for collection in client.list_collections()}
    for name in names:
        if name not in existing:
            state[name] = {"present": False}
            continue
        collection = client.get_collection(name)
        space = str((collection.metadata or {}).get("hnsw:space", "l2"))
        state[name] = {
            "present": True,
            "count": collection.count(),
            "space": space,
            "metadata": collection.metadata or {},
            "sample_similarity": similarity_from_distance(0.4, space),
        }
    return state


def _drop(client: Any, name: str) -> None:
    try:
        client.delete_collection(name)
    except ValueError:
        return


def _temp_name(name: str) -> str:
    return f"{name}{_SUFFIX}"


def _build(client: Any, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from app.services.chroma import COLLECTION_CONFIG

    built: dict[str, Any] = {}
    for name, payload in payloads.items():
        _drop(client, _temp_name(name))
        collection = client.create_collection(name=_temp_name(name), metadata=COLLECTION_CONFIG)
        for start in range(0, len(payload["ids"]), 256):
            end = start + 256
            collection.add(
                ids=payload["ids"][start:end],
                documents=payload["documents"][start:end],
                metadatas=payload["metadatas"][start:end],
            )
        built[name] = collection
    return built


def _verify(collection: Any, payload: dict[str, Any]) -> dict[str, Any]:
    expected = len(payload["ids"])
    stored = collection.count()
    result: dict[str, Any] = {"expected": expected, "stored": stored, "ok": expected == stored}
    if not expected:
        return result
    question = " ".join(payload["documents"][0].split()[:12])
    found = collection.query(query_texts=[question], n_results=min(5, expected))
    ids = (found.get("ids") or [[]])[0]
    result["sample_question"] = question
    result["sample_top"] = ids[0] if ids else None
    result["sample_rank"] = payload["ids"][0] in ids
    result["ok"] = result["ok"] and bool(ids) and result["sample_rank"]
    return result


def _copy(source: Any, target: Any, size: int = 256) -> int:
    offset = 0
    while True:
        batch = source.get(
            limit=size,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        ids = batch.get("ids") or []
        if not ids:
            return target.count()
        target.add(
            ids=ids,
            documents=batch.get("documents"),
            metadatas=batch.get("metadatas"),
            embeddings=batch.get("embeddings"),
        )
        offset += len(ids)


def _swap(client: Any, name: str, temp: str) -> dict[str, Any]:
    from app.services.chroma import COLLECTION_CONFIG
    from app.services.retrieval import similarity_from_distance

    _drop(client, name)
    target = client.create_collection(name=name, metadata=COLLECTION_CONFIG)
    count = _copy(client.get_collection(temp), target)
    _drop(client, temp)
    space = str((target.metadata or {}).get("hnsw:space", "l2"))
    return {
        "count": count,
        "space": space,
        "sample_similarity": similarity_from_distance(0.4, space),
    }


def _vacuum(chroma_dir: Path) -> dict[str, Any]:
    from app.services import chroma

    chroma.reset_clients()
    database = chroma_dir / "chroma.sqlite3"
    if not database.is_file():
        return {"present": False}
    before = database.stat().st_size
    connection = sqlite3.connect(database, isolation_level=None, timeout=30)
    try:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        connection.execute("VACUUM")
    except sqlite3.Error as error:
        return {"present": True, "before": before, "error": str(error)}
    finally:
        connection.close()
    after = database.stat().st_size
    return {"present": True, "before": before, "after": after, "saved": before - after}


def _write(folder: Path, payload: dict[str, Any]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "report.json"
    target.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    return target


def _new_folder() -> Path:
    from scripts.single_user import backup

    return backup.backup_directory(ROOT, "index-rebuild")


def _names() -> list[str]:
    from app.config import settings

    return [
        settings.chroma_collection,
        settings.chroma_faq_collection,
        settings.chroma_intents_collection,
    ]


def _planned(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "rows": payloads[name]["rows"],
            "unique": len(payloads[name]["ids"]),
            "duplicates": payloads[name].get("duplicates", 0),
        }
        for name in _names()
    }


def _rebuild(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from app.config import settings
    from app.services import chroma

    client = chroma.get_client()
    names = _names()
    report: dict[str, Any] = {
        "taken_at": datetime.now(UTC).isoformat(),
        "collections": _planned(payloads),
    }
    report["old"] = _collection_state(client, names)
    report["old_size"] = _tree_size(Path(settings.chroma_persist_dir))
    built = _build(client, payloads)
    report["verify"] = {name: _verify(built[name], payloads[name]) for name in names}
    report["swapped"] = {name: _swap(client, name, _temp_name(name)) for name in names}
    del client
    report["vacuum"] = _vacuum(Path(settings.chroma_persist_dir))
    report["new_size"] = _tree_size(Path(settings.chroma_persist_dir))
    report["new"] = _collection_state(chroma.get_client(), names)
    report["size_saved"] = report["old_size"]["bytes"] - report["new_size"]["bytes"]
    return report


def _rollback(folder: Path) -> dict[str, Any]:
    from app.config import settings
    from app.services import chroma

    current = Path(settings.chroma_persist_dir)
    saved = folder / "chroma"
    if not saved.is_dir():
        raise SystemExit(f"{saved} does not hold a backed-up index")
    chroma.reset_clients()
    parked = current.with_name(f"{current.name}.replaced-{datetime.now():%Y%m%d-%H%M%S}")
    if current.exists():
        current.rename(parked)
    shutil.copytree(saved, current)
    return {"restored": str(saved), "parked": str(parked)}


def backup_folder(folder: Path) -> dict[str, Any]:
    """Copy the live index under the report folder before anything changes."""
    from app.config import settings
    from app.services import chroma
    from scripts.single_user import backup

    chroma.reset_clients()
    return backup.tree_manifest(Path(settings.chroma_persist_dir), folder / "chroma")


def main() -> int:
    args = _arguments()
    os.chdir(ROOT)
    if args.rollback:
        if not args.yes:
            raise SystemExit("--rollback needs --yes: it replaces the live index")
        payload = {
            "taken_at": datetime.now(UTC).isoformat(),
            "rollback": _rollback(Path(args.rollback)),
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.dry_run == args.yes:
        raise SystemExit("choose --dry-run or --yes")
    folder = _new_folder()
    if args.dry_run:
        from app.config import settings
        from app.services import chroma

        payloads = _payloads()
        report: dict[str, Any] = {
            "mode": "dry-run",
            "taken_at": datetime.now(UTC).isoformat(),
            "collections": _planned(payloads),
            "old": _collection_state(chroma.get_client(), _names()),
            "old_size": _tree_size(Path(settings.chroma_persist_dir)),
        }
        print(json.dumps({"report": str(_write(folder, report)), **report}, indent=2))
        return 0
    manifest = backup_folder(folder)
    report = _rebuild(_payloads())
    report["mode"] = "apply"
    report["backup"] = manifest
    print(json.dumps({"report": str(_write(folder, report)), **report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
