"""Run one Phase 7 training operation for real and print what it measured.

The operations in ``app/services/training/operations.py`` are the code the
trainer page will call; this runs the same code once, outside the app, and
reports the numbers a measurement row needs: what the corpus gave, what the
model let be adapted, how long the steps took, what the loss fell to, how much
memory the process needed and what the run wrote.

Nothing is written where the app reads its teachers from. The default output is
a scratch folder, so a measurement run cannot become the model the chat answers
with; promoting a run is a separate, deliberate action.

Run with:
    .venv/bin/python scripts/measure_training.py --operation lora --tenant 0 --steps 20
"""

from __future__ import annotations

import argparse
import importlib
import json
import resource
import sys
import time
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / ".scratch"


@cache
def _modules() -> dict[str, Any]:
    """The app modules, loaded once with the repository on the import path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {
        "database": importlib.import_module("app.db.database"),
        "dataset": importlib.import_module("app.services.training.dataset"),
        "formats": importlib.import_module("app.services.training.formats"),
        "loop": importlib.import_module("app.services.training.loop"),
        "operations": importlib.import_module("app.services.training.operations"),
        "splits": importlib.import_module("app.services.training.splits"),
    }


def dataset_of(tenant_id: int | None, share: float) -> Any:
    """The dataset of one workspace, read from the store as it stands."""
    dataset = _modules()["dataset"]
    with _modules()["database"].SessionLocal() as db:
        return dataset.build(dataset.corpus(tenant_id, db), share=share)


def report(step: int, stage: str, loss: float | None) -> None:
    """The run's own progress, written as it happens so a long step is not silence."""
    tail = f" loss {loss}" if loss is not None else ""
    print(f"  {stage}{tail}", file=sys.stderr, flush=True)


def plan_of(arguments: argparse.Namespace, built: Any) -> Any:
    """The run the command line asked for."""
    modules = _modules()
    settings = modules["loop"].Settings(
        steps=arguments.steps,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        max_tokens=arguments.max_tokens,
        seed=arguments.seed,
    )
    adapter = modules["operations"].Adapter(
        rank=arguments.rank, alpha=arguments.alpha, dropout=arguments.dropout
    )
    return modules["operations"].Plan(
        arguments.operation,
        arguments.model,
        built,
        Path(arguments.folder),
        settings,
        adapter,
    )


def bytes_of(folder: Path) -> int:
    """The bytes a run wrote, counted from the files it left behind."""
    return sum(item.stat().st_size for item in folder.rglob("*") if item.is_file())


def peak_memory() -> int:
    """The largest resident set this process reached, in bytes."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def main() -> int:
    modules = _modules()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation", choices=modules["operations"].OPERATIONS, default="lora")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tenant", type=int, default=None)
    parser.add_argument("--share", type=float, default=modules["splits"].DEFAULT_SHARE)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=modules["loop"].BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=modules["loop"].LEARNING_RATE)
    parser.add_argument("--max-tokens", type=int, default=modules["formats"].DEFAULT_MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=modules["loop"].DEFAULT_SEED)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--folder", default="")
    arguments = parser.parse_args()
    arguments.folder = arguments.folder or str(
        SCRATCH / f"measure-{arguments.operation}-{int(time.time())}"
    )

    built = dataset_of(arguments.tenant, arguments.share)
    summary = built.report()
    print(
        f"corpus: {summary['sources']} sources, {summary['examples']} examples, "
        f"{json.dumps(summary['counts'])}",
        file=sys.stderr,
    )
    started = time.perf_counter()
    try:
        record = modules["operations"].run(plan_of(arguments, built), report)
    except modules["operations"].TrainingError as error:
        print(json.dumps({"operation": arguments.operation, "refused": str(error)}, indent=2))
        return 1
    record["wall_seconds"] = round(time.perf_counter() - started, 1)
    record["peak_memory_bytes"] = peak_memory()
    record["folder"] = str(Path(arguments.folder))
    record["bytes"] = bytes_of(Path(arguments.folder)) if Path(arguments.folder).is_dir() else 0
    record["machine"] = {
        "platform": sys.platform,
        "python": sys.version.split()[0],
    }
    print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
