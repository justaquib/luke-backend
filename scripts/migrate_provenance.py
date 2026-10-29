"""Add the provenance columns to the documents table.

A chunk now remembers the upload it came from, the characters it covers in the
extracted text, the page it was found on, the hash of its own text, the
language it is written in and how much of it reads as prose. Every column is
nullable, so adding them does not rewrite a row and no existing text moves.

Run ``--dry-run`` first: it prints the plan and writes it as a report.
``--yes`` adds the missing columns. ``--rollback`` drops them again; it only
exists to test the change on a copy, and it discards the stored provenance.

    .venv/bin/python scripts/migrate_provenance.py --dry-run
    .venv/bin/python scripts/migrate_provenance.py --yes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The column and its SQL type. The order is the order they are added in.
COLUMNS = (
    ("document_id", "INTEGER"),
    ("char_start", "INTEGER"),
    ("char_end", "INTEGER"),
    ("page", "INTEGER"),
    ("content_hash", "VARCHAR(64)"),
    ("language", "VARCHAR(16)"),
    ("quality", "FLOAT"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="inspect and report only")
    mode.add_argument("--yes", action="store_true", help="add the missing columns")
    mode.add_argument("--rollback", action="store_true", help="drop the provenance columns")
    return parser.parse_args()


def plan(engine: Any) -> dict[str, Any]:
    """The columns that are there, the ones that are missing, and the row count."""
    from sqlalchemy import inspect, text

    present = {column["name"] for column in inspect(engine).get_columns("documents")}
    with engine.connect() as connection:
        rows = int(connection.execute(text("SELECT COUNT(*) FROM documents")).scalar_one())
    return {
        "columns": sorted(present),
        "missing": [name for name, _ in COLUMNS if name not in present],
        "rows": rows,
    }


def add(engine: Any) -> list[str]:
    """Add every missing column once; a column already there is left alone."""
    from sqlalchemy import inspect, text

    present = {column["name"] for column in inspect(engine).get_columns("documents")}
    added: list[str] = []
    with engine.begin() as connection:
        for name, kind in COLUMNS:
            if name not in present:
                connection.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {kind}"))
                added.append(name)
    return added


def drop(engine: Any) -> list[str]:
    """Drop the provenance columns, for a rollback test on a copy."""
    from sqlalchemy import inspect, text

    present = {column["name"] for column in inspect(engine).get_columns("documents")}
    dropped: list[str] = []
    with engine.begin() as connection:
        for name, _ in reversed(COLUMNS):
            if name in present:
                connection.execute(text(f"ALTER TABLE documents DROP COLUMN {name}"))
                dropped.append(name)
    return dropped


def _report(payload: dict[str, Any]) -> Path:
    folder = ROOT / "backups" / f"provenance-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "report.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    arguments = _arguments()
    os.chdir(ROOT)
    from sqlalchemy import create_engine

    from app.config import settings

    engine = create_engine(settings.database_url)
    before = plan(engine)
    payload: dict[str, Any] = {
        "mode": "rollback" if arguments.rollback else "apply" if arguments.yes else "dry_run",
        "started_at": datetime.now(UTC).isoformat(),
        "before": before,
    }
    if arguments.yes:
        payload["added"] = add(engine)
    if arguments.rollback:
        payload["dropped"] = drop(engine)
    payload["after"] = plan(engine)
    path = _report(payload)
    print(json.dumps(payload, indent=2))
    print(f"\nReport: {path}")
    if arguments.dry_run:
        print("Nothing changed. Run with --yes to add the columns.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
