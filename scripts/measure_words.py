"""Measure the neural word model on held-out learner sentences.

The model learns from the learner corpus, the held-out sentences are never
trained on, and the measured numbers are compared with a model that guesses
one of the known words evenly. The script prints one JSON object: top-1
next-word accuracy, mean log-probability and perplexity against chance. It
never saves a checkpoint.

Run with:
    .venv/bin/python scripts/measure_words.py --tenant 0 --seed 7
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from functools import cache
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


@cache
def _modules() -> dict[str, Any]:
    """The app modules, loaded once with the repository on the import path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {
        "materials": importlib.import_module("app.services.language.training_materials"),
        "neural": importlib.import_module("app.services.neural"),
        "runtime": importlib.import_module("app.services.language.runtime"),
        "serving": importlib.import_module("app.services.neural.serving"),
        "tokens": importlib.import_module("app.services.neural.tokens"),
        "words": importlib.import_module("app.services.neural.words"),
    }


def _sentences(tenant_id: int) -> list[str]:
    modules = _modules()
    materials = modules["materials"]
    training = materials.training_set(materials.collect(tenant_id), tenant_id)
    corpus = [
        *training.sentences,
        *modules["runtime"].learning_examples(tenant_id),
        *(response for _, response in training.instructions),
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for sentence in corpus:
        key = " ".join(sentence.split()).casefold()
        if key and key not in seen:
            seen.add(key)
            unique.append(sentence)
    return unique


def _split(sentences: list[str], share: float, seed: int) -> tuple[list[str], list[str]]:
    """A deterministic split where a whole sentence is on one side."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(sentences), generator=generator).tolist()
    shuffled = [sentences[index] for index in order]
    held_count = max(1, round(len(shuffled) * share))
    return shuffled[held_count:], shuffled[:held_count]


def _words(sentence: str) -> list[str]:
    return [token for token in _modules()["neural"].tokenize(sentence) if token[:1].isalpha()]


def _positions(
    held: list[str], neural_index: dict[str, int], limit: int
) -> list[tuple[list[str], str]]:
    """The next-word positions whose target the model knows."""
    positions: list[tuple[list[str], str]] = []
    for sentence in held:
        words = _words(sentence)
        for index in range(1, len(words)):
            target = words[index]
            if target in neural_index:
                positions.append((words[:index], target))
                if len(positions) >= limit:
                    return positions
    return positions


def _neural_scores(
    task: Any, positions: list[tuple[list[str], str]], modules: dict[str, Any]
) -> dict[str, Any]:
    tokens = modules["tokens"]
    model = task.model
    device = next(model.parameters()).device
    context_size = int(task.config["context"])
    word_ids = _word_ids(task, modules)
    positions_in_row = {token: row for row, token in enumerate(word_ids)}
    hits = 0
    log_total = 0.0
    scored = 0
    for context, target in positions:
        target_id = task.index.get(target)
        if target_id not in positions_in_row:
            continue
        ids = [task.index[tokens.START]]
        ids += [task.index[word] for word in context if word in task.index]
        window = ids[-context_size:]
        with torch.no_grad():
            logits = model(torch.tensor([window], dtype=torch.long, device=device))[0, -1]
            chosen = torch.log_softmax(logits[word_ids], dim=-1)
        scored += 1
        if int(chosen.argmax().item()) == positions_in_row[target_id]:
            hits += 1
        log_total += float(chosen[positions_in_row[target_id]].item())
    return _scores(hits, scored, log_total)


def _scores(hits: int, scored: int, log_total: float) -> dict[str, Any]:
    mean = log_total / scored if scored else None
    return {
        "positions": scored,
        "accuracy": round(hits / scored, 4) if scored else None,
        "mean_log_probability": round(mean, 4) if mean is not None else None,
        "perplexity": round(math.exp(-mean), 4) if mean is not None else None,
    }


def _report(args: argparse.Namespace) -> dict[str, Any]:
    modules = _modules()
    neural = modules["neural"]
    train, held = _split(_sentences(args.tenant), args.share, args.seed)
    task = neural.WordTask()
    settings = neural.TrainingSettings(
        steps=args.steps,
        passes=None if args.passes == 0 else args.passes,
        learning_rate=args.learning_rate,
        seed=args.seed,
        warmup_steps=0 if args.no_warmup else args.warmup,
        max_seconds=args.seconds,
    )
    report = task.fit(train, settings=settings)
    positions = _positions(held, task.index, args.max_positions)
    result: dict[str, Any] = {
        "tenant": args.tenant,
        "seed": args.seed,
        "corpus": {"sentences": len(train) + len(held), "train": len(train), "held_out": len(held)},
        "positions": len(positions),
        "settings": {
            "steps": settings.steps,
            "passes": report.get("passes", settings.passes),
            "learning_rate": settings.learning_rate,
            "warmup_steps": settings.warmup_steps,
            "max_seconds": settings.max_seconds,
        },
        "neural": {"report": report},
    }
    if task.trained and task.model is not None:
        result["neural"].update(_neural_scores(task, positions, modules))
        classes = max(len(_word_ids(task, modules)), 1)
        result["chance"] = {
            "words": classes,
            "accuracy": round(1 / classes, 4),
            "mean_log_probability": round(-math.log(classes), 4),
            "perplexity": round(float(classes), 4),
        }
        measured, reason = modules["serving"].word_gate(task.report)
        result["measured_well"] = {"ok": measured, "reason": reason}
    return result


def _word_ids(task: Any, modules: dict[str, Any]) -> list[int]:
    tokens = modules["tokens"]
    words = modules["words"]
    return [
        index
        for index, token in enumerate(task.vocabulary)
        if tokens.is_a_word(token) and token not in words.PUNCTUATION
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--share", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--passes", type=int, default=0, help="0 keeps the production default")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--max-positions", type=int, default=5_000)
    args = parser.parse_args()
    print(json.dumps(_report(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
