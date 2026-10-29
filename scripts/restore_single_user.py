"""Restore a single-user migration backup, or rehearse that restore elsewhere.

    .venv/bin/python scripts/restore_single_user.py backups/single-user-<stamp> --yes
    .venv/bin/python scripts/restore_single_user.py backups/single-user-<stamp> --yes \\
        --database-url postgresql+psycopg2://postgres:postgres@localhost:5433/luke_restore_test \\
        --files-target /tmp/luke-restore-test

The second form leaves the live database and folders untouched and is how the
rollback is rehearsed on a copy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", help="backup folder name under backups/ or a full path")
    parser.add_argument("--yes", action="store_true", help="perform the restore")
    parser.add_argument("--database-url", default="", help="restore into this database instead")
    parser.add_argument("--files-target", default="", help="restore the folders here instead")
    return parser.parse_args()


def _folder(name: str) -> Path:
    candidate = Path(name)
    if candidate.is_dir():
        return candidate.resolve()
    return ROOT / "backups" / name


def _engine(url: str, source: Any) -> Any:
    from sqlalchemy import create_engine

    if not url:
        return source
    return create_engine(url, pool_pre_ping=True)


def _restore_files(folder: Path, target: Path) -> dict[str, Any]:
    restored = {}
    for name in (".chroma", ".language_model", "uploads"):
        source = folder / name
        if not source.is_dir():
            restored[name] = {"present": False}
            continue
        destination = target / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        restored[name] = {"present": True, "files": sum(1 for _ in destination.rglob("*"))}
    return restored


def _compare(folder: Path, counts: dict[str, int]) -> dict[str, Any]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["database"]["rows"]
    differences = {
        name: {"expected": expected.get(name), "restored": counts.get(name)}
        for name in expected
        if expected.get(name) != counts.get(name)
    }
    return {"matches": not differences, "differences": differences}


def main() -> int:
    arguments = _arguments()
    if not arguments.yes:
        print("Nothing done: pass --yes to restore.")
        return 2
    folder = _folder(arguments.backup)
    if not (folder / "database.json").is_file():
        print(f"No database.json in {folder}")
        return 1
    os.chdir(ROOT)
    from app.db.database import engine as live
    from scripts.single_user import backup

    target = _engine(arguments.database_url, live)
    if arguments.database_url:
        backup.copy_schema(live, target)
    counts = backup.restore_tables(target, folder / "database.json")
    files = _restore_files(folder, Path(arguments.files_target) if arguments.files_target else ROOT)
    check = _compare(folder, counts)
    result = {"database": counts, "files": files, "verify": check}
    print(json.dumps(result, indent=2))
    return 0 if check["matches"] else 1


if __name__ == "__main__":
    sys.exit(main())
