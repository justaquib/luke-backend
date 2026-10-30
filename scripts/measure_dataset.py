"""What the training dataset holds for one workspace, measured from the store.

The Phase 7 operations train on this, so the row it prints is the row that
belongs beside a measurement: how many sources, how many examples of each task,
and how the split fell out. Nothing here is invented — a workspace with no
uploaded documents reports no document source, and the split says so.

Run with:
    .venv/bin/python scripts/measure_dataset.py --tenant 0
"""

from __future__ import annotations

import argparse
import importlib
import sys
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@cache
def _modules() -> dict[str, Any]:
    """The app modules, loaded once with the repository on the import path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {
        "database": importlib.import_module("app.db.database"),
        "dataset": importlib.import_module("app.services.training.dataset"),
        "splits": importlib.import_module("app.services.training.splits"),
    }


def dataset_of(tenant_id: int | None, share: float) -> Any:
    """The dataset of one workspace, read from the store as it stands."""
    dataset = _modules()["dataset"]
    with _modules()["database"].SessionLocal() as db:
        return dataset.build(dataset.corpus(tenant_id, db), share=share)


def kinds(report: dict[str, Any]) -> str:
    """The example counts of each task and side, in one line."""
    return " ".join(f"{key}={value}" for key, value in sorted(report["counts"].items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", type=int, default=None)
    parser.add_argument("--share", type=float, default=_modules()["splits"].DEFAULT_SHARE)
    arguments = parser.parse_args()
    built = dataset_of(arguments.tenant, arguments.share)
    report = built.report()
    sides = report["splits"]
    print(f"workspace {arguments.tenant}")
    print(
        f"  sources       {report['sources']}"
        f" (train {sides['train_sources']}, held out {sides['held_out_sources']})"
    )
    print(f"  split taken   {report['split_taken']}  salt {report['salt']} share {report['share']}")
    print(f"  examples      {report['examples']}  {kinds(report)}")
    print(
        f"  by task       passages from {report['passage_sources']} sources,"
        f" answers from {report['answer_sources']} sources"
    )
    leaked = _modules()["dataset"].leakage(built)
    print(f"  leakage       {len(leaked)}" + (f" {list(leaked)}" if leaked else ""))
    if not report["sources"]:
        print("  nothing to read: the workspace holds no material for the dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
