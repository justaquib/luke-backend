"""Score the passages that were stored before the score existed.

Every chunk now carries the language it is written in and the share of it that
reads as prose, and a training run prefers the passages that score high. Rows
written before that change have neither, so they would all be preferred last.
``--yes`` measures the stored text again and fills the rows that are missing a
value; it never rewrites a row that already has one.

Run ``--dry-run`` first: it counts the rows and writes the plan as a report.

    .venv/bin/python scripts/migrate_quality.py --dry-run
    .venv/bin/python scripts/migrate_quality.py --yes
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

COLUMNS = ("language", "quality")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="count and report only")
    mode.add_argument("--yes", action="store_true", help="measure and fill the missing rows")
    return parser.parse_args()


def _present(engine: Any) -> set[str]:
    from sqlalchemy import inspect

    return {column["name"] for column in inspect(engine).get_columns("documents")}


COUNTS = {
    "rows": "SELECT COUNT(*) FROM documents",
    "language": "SELECT COUNT(*) FROM documents WHERE language IS NULL",
    "quality": "SELECT COUNT(*) FROM documents WHERE quality IS NULL",
}


def plan(engine: Any) -> dict[str, Any]:
    """The columns that are there and how many rows still lack each value."""
    from sqlalchemy import text

    columns = _present(engine)
    missing = [name for name in COLUMNS if name not in columns]
    counts: dict[str, int] = {}
    with engine.connect() as connection:
        for name, statement in COUNTS.items():
            counts[name] = int(connection.execute(text(statement)).one()[0])
        for name in COLUMNS:
            if name not in columns:
                counts[name] = counts["rows"]
    return {"columns": sorted(columns), "missing": missing, **counts}


def fill(engine: Any) -> int:
    """Measure every row that has no language or no quality yet."""
    from sqlalchemy import text

    from app.services.language.identify import guess
    from app.services.quality.score import passage_quality

    update = text("UPDATE documents SET language = :language, quality = :quality WHERE id = :id")
    with engine.begin() as connection:
        rows = connection.execute(
            text(
                "SELECT id, text FROM documents "
                "WHERE language IS NULL OR quality IS NULL ORDER BY id"
            )
        ).all()
        for row in rows:
            connection.execute(
                update,
                {
                    "language": guess(row.text or ""),
                    "quality": passage_quality(row.text or ""),
                    "id": row.id,
                },
            )
    return len(rows)


def _report(payload: dict[str, Any]) -> Path:
    folder = ROOT / "backups" / f"quality-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
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
        "mode": "apply" if arguments.yes else "dry_run",
        "started_at": datetime.now(UTC).isoformat(),
        "before": plan(engine),
    }
    if arguments.yes:
        payload["filled"] = fill(engine)
    payload["after"] = plan(engine)
    path = _report(payload)
    print(json.dumps(payload, indent=2))
    print(f"\nReport: {path}")
    if arguments.dry_run:
        print("Nothing changed. Run with --yes to measure and fill the rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
