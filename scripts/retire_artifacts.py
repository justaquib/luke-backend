"""Retire the lessons collection and the writer weights no code reads.

The language learner used to keep its passages in a ``lessons`` vector
collection and its writer in ``.language_model/transformer-0.pt``. The lessons
live in the learner's own file now, and the writer was replaced by the word
model, so both are kept only as dead weight. Nothing in ``app/`` reads either.

Run ``--dry-run`` first: it prints the plan and writes it as a report.
``--yes`` moves the checkpoint into the backup folder, then deletes the
collection. ``--rollback FOLDER`` puts both back, so the change can be tested.

    .venv/bin/python scripts/retire_artifacts.py --dry-run
    .venv/bin/python scripts/retire_artifacts.py --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

COLLECTION = "lessons"
CHECKPOINT = "transformer-0.pt"
BACKUP_FILE = "lessons.json"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="inspect and report only")
    mode.add_argument("--yes", action="store_true", help="back up and remove both")
    mode.add_argument("--rollback", metavar="FOLDER", help="restore from a backup folder")
    return parser.parse_args()


def _folder(client: Any) -> Any | None:
    """The stale collection, or None when it is already gone."""
    for collection in client.list_collections():
        if collection.name == COLLECTION:
            return collection
    return None


def _checkpoint() -> Path:
    from app.services.brain_paths import brain_dir

    return brain_dir() / CHECKPOINT


def plan(client: Any, checkpoint: Path) -> dict[str, Any]:
    """What is there to remove, without changing anything."""
    collection = _folder(client)
    return {
        "collection": {
            "name": COLLECTION,
            "present": collection is not None,
            "records": int(collection.count()) if collection is not None else 0,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "present": checkpoint.is_file(),
            "bytes": checkpoint.stat().st_size if checkpoint.is_file() else 0,
        },
    }


def backup(client: Any, folder: Path, checkpoint: Path) -> dict[str, Any]:
    """Keep the collection's text and move the checkpoint into the backup."""
    folder.mkdir(parents=True, exist_ok=True)
    kept: dict[str, Any] = {"collection": None, "checkpoint": None}
    collection = _folder(client)
    if collection is not None:
        stored = collection.get(include=["documents", "metadatas"])
        data = {
            "name": collection.name,
            "ids": list(stored.get("ids") or []),
            "documents": list(stored.get("documents") or []),
            "metadatas": [metadata or {} for metadata in stored.get("metadatas") or []],
        }
        (folder / BACKUP_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
        kept["collection"] = {"file": BACKUP_FILE, "records": len(data["ids"])}
    if checkpoint.is_file():
        target = folder / checkpoint.name
        shutil.move(str(checkpoint), target)
        kept["checkpoint"] = {"file": target.name, "bytes": target.stat().st_size}
    return kept


def remove(client: Any, checkpoint: Path) -> dict[str, bool]:
    """Delete the collection and any checkpoint the backup did not move."""
    removed = {"collection": False, "checkpoint": False}
    if _folder(client) is not None:
        client.delete_collection(COLLECTION)
        removed["collection"] = True
    if checkpoint.is_file():
        checkpoint.unlink()
        removed["checkpoint"] = True
    return removed


def restore(client: Any, folder: Path, checkpoint: Path) -> dict[str, bool]:
    """Put the collection's text and the checkpoint back from a backup."""
    restored = {"collection": False, "checkpoint": False}
    path = folder / BACKUP_FILE
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        collection = _folder(client) or client.create_collection(name=data["name"])
        if data["ids"]:
            collection.upsert(
                ids=data["ids"], documents=data["documents"], metadatas=data["metadatas"]
            )
        restored["collection"] = True
    source = folder / checkpoint.name
    if source.is_file():
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), checkpoint)
        restored["checkpoint"] = True
    return restored


def _report(payload: dict[str, Any], folder: Path | None = None) -> Path:
    target = folder or (ROOT / "backups" / f"retired-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
    target.mkdir(parents=True, exist_ok=True)
    path = target / "report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    arguments = _arguments()
    os.chdir(ROOT)
    from app.services.chroma import get_client

    client = get_client()
    checkpoint = _checkpoint()
    payload: dict[str, Any] = {
        "mode": "rollback" if arguments.rollback else "apply" if arguments.yes else "dry_run",
        "started_at": datetime.now(UTC).isoformat(),
        "before": plan(client, checkpoint),
    }
    folder: Path | None = None
    if arguments.yes:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        folder = ROOT / "backups" / f"retired-{stamp}"
        payload["backup"] = backup(client, folder, checkpoint)
        payload["removed"] = remove(client, checkpoint)
    if arguments.rollback:
        folder = Path(arguments.rollback)
        payload["restored"] = restore(client, folder, checkpoint)
    payload["after"] = plan(client, checkpoint)
    path = _report(payload, folder)
    print(json.dumps(payload, indent=2))
    print(f"\nReport: {path}")
    if arguments.dry_run:
        print("Nothing changed. Run with --yes to move the backup and remove both.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
