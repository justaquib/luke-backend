"""Measure byte-level BPE vocabularies against the word-level baseline.

The corpus is the material the scratch model already trains from. The held-out
sentences are chosen by the same seed the training uses, and the BPE tokenizer
is built from the training side only. Every candidate is trained with the same
settings, and the report gives the numbers a choice may be based on: tokens
per character, fertility, truncation, parameters, artifact and checkpoint
bytes, throughput, held-out loss, perplexity, next-token accuracy and bits per
character.

Run with:
    .venv/bin/python scripts/measure_tokenizer.py --tenant 0 --dry-run
    .venv/bin/python scripts/measure_tokenizer.py --tenant 0
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import tempfile
import time
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@cache
def _modules() -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {
        "database": importlib.import_module("app.db.database"),
        "materials": importlib.import_module("app.services.language.training_materials"),
        "neural": importlib.import_module("app.services.neural"),
        "quality": importlib.import_module("app.services.neural.quality"),
        "runtime": importlib.import_module("app.services.language.runtime"),
        "tokenizer": importlib.import_module("app.services.neural.tokenizer"),
        "tokens": importlib.import_module("app.services.neural.tokens"),
        "corpus": importlib.import_module("app.services.neural.tokenizer.corpus"),
        "spec": importlib.import_module("app.services.neural.tokenizer.spec"),
        "artifacts": importlib.import_module("app.services.neural.tokenizer.artifacts"),
        "words": importlib.import_module("app.services.neural.words"),
    }


def _sentences(tenant_id: int) -> list[str]:
    """The corpus the scratch model trains from, deduplicated."""
    modules = _modules()
    session_factory = modules["database"].SessionLocal
    db = session_factory()
    try:
        materials = modules["materials"].gather(tenant_id, db)
    finally:
        db.close()
    training = modules["materials"].training_set(materials, tenant_id)
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


def _settings(args: argparse.Namespace) -> Any:
    neural = _modules()["neural"]
    return neural.TrainingSettings(
        steps=args.steps,
        passes=None,
        learning_rate=args.learning_rate,
        seed=args.seed,
        warmup_steps=0,
        max_seconds=args.seconds,
        validation_share=args.share,
    )


def _kept(sentences: list[str]) -> list[str]:
    return _modules()["quality"].keep(sentences)


def _token_stats(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    """Tokens per character, fertility and length, measured on a fixed text set."""
    characters = sum(len(text) for text in texts)
    words = sum(len(text.split()) for text in texts)
    started = time.perf_counter()
    encoded = [tokenizer.encode(text) for text in texts]
    elapsed = max(time.perf_counter() - started, 1e-9)
    tokens = sum(len(ids) for ids in encoded)
    return {
        "texts": len(texts),
        "characters": characters,
        "tokens": tokens,
        "tokens_per_character": round(tokens / characters, 4) if characters else None,
        "fertility": round(tokens / words, 4) if words else None,
        "average_tokens": round(tokens / len(texts), 2) if texts else None,
        "truncated_64": _truncated(encoded, 64),
        "truncated_128": _truncated(encoded, 128),
        "truncated_256": _truncated(encoded, 256),
        "encode_characters_per_second": round(characters / elapsed, 1),
        "encode_tokens_per_second": round(tokens / elapsed, 1),
    }


def _truncated(encoded: list[list[int]], limit: int) -> float | None:
    if not encoded:
        return None
    over = sum(1 for ids in encoded if len(ids) + 2 > limit)
    return round(over / len(encoded), 4)


def _checkpoint_bytes(task: Any) -> int:
    folder = Path(tempfile.mkdtemp())
    path = folder / "words.pt"
    task.save(path)
    size = path.stat().st_size if path.is_file() else 0
    path.unlink(missing_ok=True)
    return size


def _measure(
    task: Any, report: dict[str, Any], train: list[str], held: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    """The row a baseline and a candidate report, measured the same way."""
    modules = _modules()
    tokenizer = task.tokenizer
    held_stats = _token_stats(tokenizer, held)
    characters = sum(len(text) for text in held)
    validation = report.get("validation_loss")
    bits_per_character = (
        round(float(validation) * held_stats["tokens"] / characters / math.log(2), 4)
        if validation is not None and characters
        else None
    )
    return {
        "vocab_size": tokenizer.vocab_size,
        "trained": True,
        "tokenizer": task.tokenizer_record,
        "artifact_bytes": modules["artifacts"].artifact_bytes(task.tokenizer_record),
        "parameters": report.get("parameters"),
        # The width, not the requested size: a small corpus cannot fill 8k or 16k.
        "embedding_parameters": tokenizer.vocab_size * args.width,
        "checkpoint_bytes": _checkpoint_bytes(task),
        "tokens_per_character_train": _token_stats(tokenizer, train)["tokens_per_character"],
        "held_out": held_stats,
        "held_out_tokens_per_character": held_stats["tokens_per_character"],
        "bits_per_character": bits_per_character,
        "held_out_loss": validation,
        "held_out_perplexity": report.get("perplexity"),
        "next_token_accuracy": report.get("next_word_accuracy"),
        "chance_loss": round(math.log(max(tokenizer.vocab_size, 2)), 4),
        "chance_perplexity": round(float(tokenizer.vocab_size), 2),
        "train_seconds": report.get("seconds"),
    }


def _trained_row(
    kind: str, requested: int | None, train: list[str], held: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    """Train one model the way production does, then measure it on the held-out side."""
    modules = _modules()
    config: dict[str, Any] = {"tokenizer_kind": kind, "context": args.context}
    if requested is not None:
        config["vocabulary_size"] = requested
    task = modules["words"].WordTask(config)
    started = time.perf_counter()
    report = task.fit(train + held, settings=_settings(args))
    elapsed = round(time.perf_counter() - started, 1)
    if not report.get("trained"):
        return {
            "kind": kind,
            "requested_vocab_size": requested,
            "trained": False,
            "reason": report.get("reason"),
        }
    return {
        "kind": kind,
        "requested_vocab_size": requested,
        "measured_seconds": elapsed,
        **_measure(task, report, train, held, args),
    }


def _candidate(
    size: int, train: list[str], held: list[str], args: argparse.Namespace
) -> dict[str, Any]:
    return _trained_row(_modules()["tokenizer"].BPE_KIND, size, train, held, args)


def _unknown_words(train: list[str], held: list[str]) -> dict[str, Any]:
    """How many held-out words a train-only word vocabulary could not hold.

    The word model builds its vocabulary from every kept sentence, so this
    share says how much that asymmetry flatters it against BPE, whose
    vocabulary is built from the training side only.
    """
    modules = _modules()
    tokens_module = modules["tokens"]
    vocabulary = set(modules["spec"].build(modules["tokenizer"].WORD_KIND, train).vocabulary())
    specials = set(tokens_module.SPECIAL_TOKENS)
    words = [
        token for text in held for token in tokens_module.tokenize(text) if token not in specials
    ]
    unknown = sum(1 for word in words if word not in vocabulary)
    return {
        "held_out_words": len(words),
        "unknown_words": unknown,
        "unknown_share": round(unknown / len(words), 4) if words else None,
    }


def _baseline(train: list[str], held: list[str], args: argparse.Namespace) -> dict[str, Any]:
    row = _trained_row(_modules()["tokenizer"].WORD_KIND, None, train, held, args)
    row["held_out_unknown_words"] = _unknown_words(train, held)
    row["recorded_on_2026_10_27"] = {
        "held_out_loss": 3.1008,
        "chance_loss": 7.3601,
        "parameters": 606208,
        "vocabulary": 1872,
    }
    return row


def _report(args: argparse.Namespace) -> dict[str, Any]:
    modules = _modules()
    sentences = _sentences(args.tenant)
    kept = _kept(sentences)
    settings = _settings(args)
    train, held = modules["corpus"].split_training(kept, settings)
    result: dict[str, Any] = {
        "tenant": args.tenant,
        "seed": args.seed,
        "corpus": {
            "sentences": len(sentences),
            "kept": len(kept),
            "train_sentences": len(train),
            "held_out_sentences": len(held),
            "train_characters": sum(len(text) for text in train),
            "held_out_characters": sum(len(text) for text in held),
        },
        "settings": {
            "steps": args.steps,
            "context": args.context,
            "width": args.width,
            "learning_rate": args.learning_rate,
            "seconds": args.seconds,
        },
        "baseline": _baseline(train, held, args),
        "candidates": [],
    }
    if args.dry_run:
        return result
    for size in args.vocab_size:
        result["candidates"].append(_candidate(size, train, held, args))
        print(json.dumps(result["candidates"][-1], sort_keys=True), file=sys.stderr)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--share", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--vocab-size",
        type=int,
        action="append",
        default=None,
        help="a BPE vocabulary to test; repeat for several (default 2048/4096/8192/16384)",
    )
    args = parser.parse_args()
    if args.vocab_size is None:
        args.vocab_size = [2048, 4096, 8192, 16384]
    print(json.dumps(_report(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
