"""Add the chunk fingerprint column to the documents table.

Every chunk now remembers a 64-bit SimHash of its wording, so a near repeat of
a passage can be recognised at ingest. The column is nullable, so adding it
does not rewrite a row and no existing text moves; ``--yes`` also fills the
rows that have text and no fingerprint yet.

Run ``--dry-run`` first: it prints the plan and writes it as a report.
``--yes`` adds the missing column and backfills it. ``--rollback`` drops the
column again; it only exists to test the change on a copy, and it discards
the stored fingerprints.

    .venv/bin/python scripts/migrate_simhash.py --dry-run
    .venv/bin/python scripts/migrate_simhash.py --yes
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

COLUMN = ("simhash", "BIGINT")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="inspect and report only")
    mode.add_argument("--yes", action="store_true", help="add and backfill the column")
    mode.add_argument("--rollback", action="store_true", help="drop the fingerprint column")
    return parser.parse_args()


def _present(engine: Any) -> set[str]:
    from sqlalchemy import inspect

    return {column["name"] for column in inspect(engine).get_columns("documents")}


def plan(engine: Any) -> dict[str, Any]:
    """The columns that are there, whether simhash is missing, and the row count."""
    from sqlalchemy import text

    columns = _present(engine)
    name, _ = COLUMN
    with engine.connect() as connection:
        rows = int(connection.execute(text("SELECT COUNT(*) FROM documents")).scalar_one())
        unbackfilled = (
            int(
                connection.execute(
                    text("SELECT COUNT(*) FROM documents WHERE simhash IS NULL")
                ).scalar_one()
            )
            if name in columns
            else rows
        )
    return {
        "columns": sorted(columns),
        "missing": [] if name in columns else [name],
        "rows": rows,
        "unbackfilled": unbackfilled,
    }


def add(engine: Any) -> list[str]:
    """Add the fingerprint column once; a column already there is left alone."""
    from sqlalchemy import text

    present = _present(engine)
    added: list[str] = []
    name, kind = COLUMN
    if name not in present:
        with engine.begin() as connection:
            connection.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {kind}"))
        added.append(name)
    return added


def backfill(engine: Any) -> int:
    """Fingerprint every row that has text and no fingerprint yet."""
    from sqlalchemy import text

    from app.services.quality.simhash import simhash, to_signed

    with engine.begin() as connection:
        rows = connection.execute(
            text("SELECT id, text FROM documents WHERE simhash IS NULL")
        ).all()
        for row in rows:
            connection.execute(
                text("UPDATE documents SET simhash = :value WHERE id = :id"),
                {"value": to_signed(simhash(row.text or "")), "id": row.id},
            )
    return len(rows)


def drop(engine: Any) -> list[str]:
    """Drop the fingerprint column, for a rollback test on a copy."""
    from sqlalchemy import text

    present = _present(engine)
    name, _ = COLUMN
    if name not in present:
        return []
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE documents DROP COLUMN {name}"))
    return [name]


def _report(payload: dict[str, Any]) -> Path:
    folder = ROOT / "backups" / f"simhash-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
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
        payload["backfilled"] = backfill(engine)
    if arguments.rollback:
        payload["dropped"] = drop(engine)
    payload["after"] = plan(engine)
    path = _report(payload)
    print(json.dumps(payload, indent=2))
    print(f"\nReport: {path}")
    if arguments.dry_run:
        print("Nothing changed. Run with --yes to add and backfill the column.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
