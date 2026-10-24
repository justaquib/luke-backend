"""Measure the answer model in memory, without touching the live checkpoint.

The script builds the corpus exactly as production does, trains a fresh model
with the honest held-out split, and prints one JSON object. It never saves a
checkpoint and never asks the engine for a model, so a measurement run cannot
overwrite what a tenant is serving.

Run with:
    .venv/bin/python scripts/measure_writer.py --tenant 0 --source files --steps 0 --seed 7
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@cache
def _imports() -> tuple[Any, Any]:
    """The app modules, loaded once with the repository on the import path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return (
        importlib.import_module("app.services.language.training_materials"),
        importlib.import_module("app.services.neural"),
    )


def _materials(tenant_id: int, source: str) -> list[Any]:
    materials = _imports()[0]
    if source == "files":
        return materials.collect(tenant_id)
    from app.db.database import SessionLocal

    with SessionLocal() as db:
        return materials.gather(tenant_id, db)


def _settings(args: argparse.Namespace, neural: Any) -> Any:
    return neural.TrainingSettings(
        steps=args.steps,
        passes=None if args.passes == 0 else args.passes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        warmup_steps=0 if args.no_warmup else args.warmup,
        check_every=args.check_every,
        max_seconds=args.seconds,
        max_steps=args.max_steps,
    )


def _corpus(tenant_id: int, source: str) -> tuple[Any, Any]:
    materials, neural = _imports()
    training = materials.training_set(_materials(tenant_id, source), tenant_id)
    corpus = neural.AnswerCorpus(
        training.sentences, list(training.instructions), list(training.contexts)
    )
    return training, corpus


def _evaluation(tenant_id: int, task: Any) -> dict[str, Any]:
    """The taught-request, refusal and contamination probes on the trained model."""
    from app.services.language import evaluation as evaluation_service
    from app.services.language.lessons import all_lesson_cards

    return evaluation_service.evaluate(tenant_id, all_lesson_cards(tenant_id), task)


def _report(args: argparse.Namespace) -> dict[str, Any]:
    _, neural = _imports()
    settings = _settings(args, neural)
    training, corpus = _corpus(args.tenant, args.source)
    task = neural.AnswerTask()
    result = task.fit(corpus, settings=settings)
    report = {
        "tenant": args.tenant,
        "source": args.source,
        "seed": args.seed,
        "settings": {
            "steps": settings.steps,
            "passes": result.get("passes", settings.passes),
            "batch_size": settings.batch_size,
            "learning_rate": settings.learning_rate,
            "warmup_steps": settings.warmup_steps,
            "check_every": settings.check_every,
            "max_steps": settings.max_steps,
            "max_seconds": settings.max_seconds,
        },
        "corpus": {
            "materials": len(training.materials),
            "sentences": len(training.sentences),
            "instructions": len(training.instructions),
            "contexts": len(training.contexts),
        },
        "device": neural.device_name(),
        "report": result,
    }
    if args.evaluate:
        report["evaluation"] = _evaluation(args.tenant, task)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", type=int, default=0)
    parser.add_argument("--source", choices=("files", "all"), default="files")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--passes", type=int, default=0, help="0 keeps the production default")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--check-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--seconds", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="also measure the held-out requests and refusals on the trained model",
    )
    args = parser.parse_args()
    print(json.dumps(_report(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
