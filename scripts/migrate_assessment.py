"""Create the two assessment tables on a database that predates Phase 6.

``init_db()`` creates them on a fresh install; an existing workspace needs them
added in place. Both tables start empty — a run is only written when an
assessment is started — so adding them neither rewrites nor moves any row.

Run ``--dry-run`` first: it prints the plan and writes it as a report.
``--yes`` creates the missing tables. ``--rollback`` drops them again; it only
exists to test the change on a copy, and it discards assessment history.

    .venv/bin/python scripts/migrate_assessment.py --dry-run
    .venv/bin/python scripts/migrate_assessment.py --yes
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

TABLES = ("assessment_runs", "assessment_results")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="inspect and report only")
    mode.add_argument("--yes", action="store_true", help="create the missing tables")
    mode.add_argument("--rollback", action="store_true", help="drop the assessment tables")
    return parser.parse_args()


def _definitions() -> dict[str, Any]:
    """The table objects, imported late so the script can set the path first."""
    from app.db.database import AssessmentResult, AssessmentRun

    return {
        "assessment_runs": AssessmentRun.__table__,
        "assessment_results": AssessmentResult.__table__,
    }


COUNTS = {
    "assessment_runs": "SELECT COUNT(*) FROM assessment_runs",
    "assessment_results": "SELECT COUNT(*) FROM assessment_results",
}


def plan(engine: Any) -> dict[str, Any]:
    """Which tables are there, which are missing, and how many rows each holds."""
    from sqlalchemy import inspect, text

    present = set(inspect(engine).get_table_names())
    rows: dict[str, int] = {}
    with engine.connect() as connection:
        for name, statement in COUNTS.items():
            count = 0
            if name in present:
                count = int(connection.execute(text(statement)).scalar_one())
            rows[name] = count
    return {
        "tables": [name for name in TABLES if name in present],
        "missing": [name for name in TABLES if name not in present],
        "rows": rows,
    }


def create(engine: Any) -> list[str]:
    """Create every missing table once; a table already there is left alone."""
    from app.db.database import Base

    definitions = _definitions()
    missing = plan(engine)["missing"]
    if missing:
        Base.metadata.create_all(engine, tables=[definitions[name] for name in missing])
    return missing


def drop(engine: Any) -> list[str]:
    """Drop the assessment tables, for a rollback test on a copy."""
    from app.db.database import Base

    definitions = _definitions()
    present = plan(engine)["tables"]
    if present:
        Base.metadata.drop_all(engine, tables=[definitions[name] for name in reversed(present)])
    return present


def _report(payload: dict[str, Any]) -> Path:
    folder = ROOT / "backups" / f"assessment-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
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
        "mode": "rollback" if arguments.rollback else ("apply" if arguments.yes else "dry_run"),
        "started_at": datetime.now(UTC).isoformat(),
        "before": plan(engine),
    }
    if arguments.yes:
        payload["created"] = create(engine)
    if arguments.rollback:
        payload["dropped"] = drop(engine)
    payload["after"] = plan(engine)
    path = _report(payload)
    print(json.dumps(payload, indent=2))
    print(f"\nReport: {path}")
    if arguments.dry_run:
        print("Nothing changed. Run with --yes to create the assessment tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
