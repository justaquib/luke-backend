"""Add the trainer note column to the pending questions table.

A contradictory answer is shown in the trainer inbox with the two answers it
compares, and that explanation is stored beside the question. The column is
nullable, so adding it rewrites no row.

Run ``--dry-run`` first: it prints the plan and writes it as a report.
``--yes`` adds the missing column. ``--rollback`` drops it again; it only
exists to test the change on a copy.

    .venv/bin/python scripts/migrate_pending_note.py --dry-run
    .venv/bin/python scripts/migrate_pending_note.py --yes
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

COLUMN = ("note", "TEXT")
TABLE = "pending_questions"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="inspect and report only")
    mode.add_argument("--yes", action="store_true", help="add the column")
    mode.add_argument("--rollback", action="store_true", help="drop the column again")
    return parser.parse_args()


def _present(engine: Any) -> set[str]:
    from sqlalchemy import inspect

    return {column["name"] for column in inspect(engine).get_columns(TABLE)}


def plan(engine: Any) -> dict[str, Any]:
    """The columns that are there and whether the note is missing."""
    from sqlalchemy import text

    columns = _present(engine)
    name, _ = COLUMN
    with engine.connect() as connection:
        rows = int(connection.execute(text("SELECT COUNT(*) FROM pending_questions")).scalar_one())
    return {
        "columns": sorted(columns),
        "missing": [] if name in columns else [name],
        "rows": rows,
    }


def add(engine: Any) -> list[str]:
    """Add the note column once; a column already there is left alone."""
    from sqlalchemy import text

    present = _present(engine)
    added: list[str] = []
    name, kind = COLUMN
    if name not in present:
        with engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {name} {kind}"))
        added.append(name)
    return added


def drop(engine: Any) -> list[str]:
    """Drop the note column, for a rollback test on a copy."""
    from sqlalchemy import text

    present = _present(engine)
    name, _ = COLUMN
    if name not in present:
        return []
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {TABLE} DROP COLUMN {name}"))
    return [name]


def _report(payload: dict[str, Any]) -> Path:
    folder = ROOT / "backups" / f"pending-note-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
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
    payload: dict[str, Any] = {
        "mode": "rollback" if arguments.rollback else "apply" if arguments.yes else "dry_run",
        "started_at": datetime.now(UTC).isoformat(),
        "before": plan(engine),
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
        print("Nothing changed. Run with --yes to add the column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
