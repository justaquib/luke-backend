"""Move the existing data onto the single Luke Lab workspace.

Run ``--dry-run`` first: it prints and saves exactly what would change and
writes nothing else. ``--yes`` takes a full backup, then applies the change.

    .venv/bin/python scripts/migrate_single_user.py --dry-run
    .venv/bin/python scripts/migrate_single_user.py --yes
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
FIXED_QUESTION = "What is there to know about wind?"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="inspect and report only")
    parser.add_argument("--yes", action="store_true", help="back up, then apply the change")
    parser.add_argument("--question", default=FIXED_QUESTION, help="the fixed retrieval check")
    return parser.parse_args()


def _snapshot(engine: Any, question: str) -> dict[str, Any]:
    from scripts.single_user import database, vectors

    with engine.connect() as connection:
        tables = database.reflect(connection)
        plan = database.inspect(connection, tables)
        counts = database.counts(connection, tables)
    return {
        "counts": counts,
        "outside": sum(entry["outside"] for entry in plan["tables"]),
        "faq_groups": len(plan["faqs"]),
        "vectors_stale": sum(entry["stale"] for entry in vectors.inspect()["collections"]),
        "search": vectors.sample(question),
    }


def _inspect_all(engine: Any, root: Path, question: str) -> dict[str, Any]:
    from scripts.single_user import database, files, state, vectors

    with engine.connect() as connection:
        tables = database.reflect(connection)
        plan = database.inspect(connection, tables)
    return {
        "plan": plan,
        "state": state.plan_state(root / ".language_model"),
        "vectors": vectors.inspect(),
        "uploads": files.plan(root / "uploads"),
        "before": _snapshot(engine, question),
    }


def _backup(root: Path, engine: Any) -> dict[str, Any]:
    from scripts.single_user import backup

    folder = backup.backup_directory(root)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "root": str(folder),
        "taken_at": datetime.now(UTC).isoformat(),
        "database": backup.dump_tables(engine, folder / "database.json"),
        "chroma": backup.tree_manifest(root / ".chroma", folder / ".chroma"),
        "language_model": backup.tree_manifest(
            root / ".language_model", folder / ".language_model"
        ),
        "uploads": backup.tree_manifest(root / "uploads", folder / "uploads"),
    }
    (folder / "manifest.json").write_text(
        json.dumps(manifest, default=str, indent=2), encoding="utf-8"
    )
    return manifest


def _apply_all(engine: Any, root: Path, inspected: dict, question: str) -> dict[str, Any]:
    from scripts.single_user import database, files, state, vectors

    with engine.begin() as connection:
        tables = database.reflect(connection)
        undo = database.apply_plan(connection, tables, inspected["plan"])
    dropped = [int(identifier) for identifier in undo.get("faq_map", {})]
    stale = [entry for entry in inspected["vectors"]["collections"] if entry["ids"]]
    vector_result = vectors.apply_all({"collections": stale}, dropped)
    upload_result = files.apply(inspected["uploads"], root / "uploads")
    after = _snapshot(engine, question)
    after["uploads_loose"] = len(files.plan(root / "uploads")["moves"])
    return {
        "undo": undo,
        "state": state.apply_state(inspected["state"]),
        "vectors_applied": vector_result,
        "uploads_applied": upload_result,
        "after": after,
    }


def _conflicts(plan: dict[str, Any], learner: dict, uploads: dict) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    for group in plan["faqs"]:
        for row in group["drop"]:
            conflicts.append(
                {
                    "kind": "faq_duplicate",
                    "question": group["question"],
                    "dropped_id": row["id"],
                    "kept_id": group["keep"]["id"],
                    "kept_status": group["keep"]["status"],
                }
            )
    for row in plan["users"]["drop"]:
        conflicts.append({"kind": "user_removed", "email": row["email"], "id": row["id"]})
    for row in plan["tenants"]["drop"]:
        conflicts.append({"kind": "tenant_removed", "slug": row["slug"], "id": row["id"]})
    for note in learner["notes"]:
        if note["action"] != "duplicate":
            conflicts.append({"kind": "learner_card", **note})
    for entry in uploads["moves"]:
        if entry["action"] == "conflict":
            conflicts.append({"kind": "upload_name_conflict", "source": entry["source"]})
    return conflicts


def _problems(after: dict[str, Any]) -> list[str]:
    problems = []
    if after["outside"]:
        problems.append(f"{after['outside']} rows outside workspace 0")
    if after["faq_groups"]:
        problems.append(f"{after['faq_groups']} duplicate FAQ groups remain")
    if after["vectors_stale"]:
        problems.append(f"{after['vectors_stale']} vectors lack the workspace")
    if after["uploads_loose"]:
        problems.append(f"{after['uploads_loose']} uploads remain at the root")
    return problems


def _database_section(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "tables": plan["tables"],
        "faqs": plan["faqs"],
        "users": plan["users"],
        "tenants": plan["tenants"],
    }


def main() -> int:
    arguments = _arguments()
    if arguments.dry_run == arguments.yes:
        print("Choose one: --dry-run to inspect, or --yes to apply.")
        return 2
    os.chdir(ROOT)
    from app.db.database import engine
    from scripts.single_user import report

    payload: dict[str, Any] = {
        "dry_run": arguments.dry_run,
        "started_at": datetime.now(UTC).isoformat(),
        "root": str(ROOT),
    }
    folder = ROOT / "backups" / f"preflight-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    if arguments.yes:
        payload["backup"] = _backup(ROOT, engine)
        folder = Path(payload["backup"]["root"])
    inspected = _inspect_all(engine, ROOT, arguments.question)
    payload.update(
        {
            "database": _database_section(inspected["plan"]),
            "learner": {
                "sources": inspected["state"]["sources"],
                "notes": inspected["state"]["notes"],
                "measurements": inspected["state"]["measurements"],
            },
            "vectors": inspected["vectors"],
            "uploads": inspected["uploads"],
            "before": inspected["before"],
        }
    )
    if arguments.yes:
        payload.update(_apply_all(engine, ROOT, inspected, arguments.question))
        payload["problems"] = _problems(payload["after"])
    folder.mkdir(parents=True, exist_ok=True)
    report.write_conflicts(
        folder, _conflicts(inspected["plan"], inspected["state"], inspected["uploads"])
    )
    if arguments.yes:
        report.write_rollback(folder, payload["undo"], payload["backup"])
    report.write_report(folder, payload)
    print(report.summary(payload))
    print(f"\nReport: {folder / 'report.json'}")
    if payload.get("problems"):
        print("\nProblems:")
        for problem in payload["problems"]:
            print(f"  - {problem}")
        return 1
    if arguments.dry_run:
        print("Nothing changed. Run with --yes to apply after reviewing the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
