"""Measure the assessment engine before any of its numbers are trusted.

Five reports, only the last of which reads a run row:

- Bank: the shipped files, their digests, and what each skill's splits hold.
- Scorers: how long each deterministic scorer takes on real items, and what the
  same item scores when it is answered with its own expectation and when it is
  answered wrong.
- Generation: a timed pilot of the runner's own call -- the installed
  checkpoint, greedy decoding, the run's message shape -- over ``--items``
  held-out items spread across the skills, plus the retrieval read that only the
  assisted mode makes.
- Recall: how many held-out answers the checkpoint produces from memory, with
  the task framing and the options removed. A probe only counts when its answer
  is a phrase long enough to survive inside prose and the bare prompt no longer
  shows it. This is the pretraining-contamination probe: the base model's
  pretraining corpus cannot be audited on this machine, so the leak is measured
  on the items instead.
- Run: what a finished run of the engine measured, read back from the rows it
  wrote -- its wall time, the per-item latency percentiles, what each skill
  scored, and the rubric rows still waiting for a trainer.

Run with:
    .venv/bin/python scripts/measure_assessment.py
    .venv/bin/python scripts/measure_assessment.py --items 12 --recall 12
    .venv/bin/python scripts/measure_assessment.py --run 2 --tenant 0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BANK_DIR = ROOT / "app" / "data" / "assessment"
KINDS = ("option", "span", "number", "literal", "rules", "rubric")
WRONG = {
    "option": "F",
    "span": "the wrong end of the boat",
    "number": -917.5,
    "literal": "None",
    "rules": "",
    "rubric": "",
}
_INSTRUCTION = re.compile(r"\n[ \t]*\n")
_SPACES = re.compile(r"\s+")
_OPTIONS = re.compile(r"\n\s*[A-F][.)].*", re.DOTALL)
_LABELS = re.compile(r"\n\s*([A-F])[.)]\s*(.+)")
# A one- or two-word answer sits inside ordinary prose, so naming it says nothing
# about what the checkpoint memorised; those probes are kept apart from the share.
MIN_RECALL_ANSWER_CHARS = 5


@cache
def _modules() -> dict[str, Any]:
    """The app modules, loaded once with the repository on the import path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {
        "bank": importlib.import_module("app.services.assessment.bank"),
        "conditions": importlib.import_module("app.services.assessment.conditions"),
        "profiles": importlib.import_module("app.services.assessment.profiles"),
        "runner": importlib.import_module("app.services.assessment.runner"),
        "scoring": importlib.import_module("app.services.assessment.scoring"),
        "skills": importlib.import_module("app.services.assessment.skills"),
    }


def _percentile(values: list[float], share: float) -> float:
    """The ordered value at ``share`` of the sample, nearest-rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(share * (len(ordered) - 1))))
    return ordered[index]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _file_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"]
    kinds = Counter(str(item["expected"]["kind"]) for item in items)
    return {
        "skill": path.stem,
        "bytes": path.stat().st_size,
        "sha256_16": _digest(path),
        "items": len(items),
        "eval": sum(1 for item in items if item["split"] == "eval"),
        "train": sum(1 for item in items if item["split"] == "train"),
        "stages": sorted({int(item["stage"]) for item in items}),
        "topics": len({str(item["topic"]) for item in items}),
        "kinds": dict(kinds),
        "datasets": sorted({str(item["source"]["dataset"]) for item in items}),
        "licences": sorted({str(item["source"]["licence"]) for item in items}),
    }


def _bank_report() -> dict[str, Any]:
    bank = _modules()["bank"]
    files = [
        _file_report(path) for path in sorted(BANK_DIR.glob("*.json")) if path.stem != "manifest"
    ]
    manifest = bank.manifest()
    eval_items = bank.items(split="eval")
    return {
        "version": bank.version(),
        "salt": bank.salt(),
        "check": bank.check(),
        "sources": len(manifest.get("sources") or ()),
        "manifest_bytes": (BANK_DIR / "manifest.json").stat().st_size,
        "bank_bytes": sum(row["bytes"] for row in files)
        + (BANK_DIR / "manifest.json").stat().st_size,
        "items": sum(row["items"] for row in files),
        "eval_items": len(eval_items),
        "eval_kinds": dict(Counter(str(item["expected"]["kind"]) for item in eval_items)),
        "eval_rules": dict(
            Counter(
                str(rule["id"])
                for item in eval_items
                for rule in item["expected"].get("rules") or ()
            )
        ),
        "files": files,
    }


def _expected_answer(item: dict[str, Any]) -> Any:
    expected = item["expected"]
    kind = str(expected["kind"])
    if kind == "rubric":
        return str((expected.get("reference") or "").split("\n")[0][:80])
    if kind == "rules":
        return "An answer that follows the stated rules."
    if kind == "span":
        return str(expected.get("answer") or expected.get("acceptable", [""])[0])
    return expected.get("answer")


def _scorer_report(repeats: int) -> dict[str, Any]:
    scoring = _modules()["scoring"]
    bank = _modules()["bank"]
    rows: dict[str, Any] = {}
    for kind in KINDS:
        items = [item for item in bank.items(split="eval") if item["expected"]["kind"] == kind][:6]
        if not items:
            continue
        answers = [(_expected_answer(item), item) for item in items]
        wrong = [(WRONG[kind], item) for item in items]
        cases = answers + wrong
        mark = scoring.mark
        start = time.perf_counter()
        for _ in range(repeats):
            for response, item in cases:
                mark(item, str(response))
        per_call = (time.perf_counter() - start) / (repeats * len(cases))
        rows[kind] = {
            "items": len(items),
            "microseconds_per_answer": round(per_call * 1e6, 2),
            "expected_answer_scores": [mark(item, str(r)).score for r, item in answers],
            "wrong_answer_scores": [mark(item, str(r)).score for r, item in wrong],
        }
    return rows


def _sample_items(count: int) -> list[dict[str, Any]]:
    """The first held-out items of each skill, taken in turn until ``count``."""
    bank = _modules()["bank"]
    by_skill: dict[str, list[dict[str, Any]]] = {}
    for item in bank.items(split="eval"):
        by_skill.setdefault(str(item["skill"]), []).append(item)
    chosen: list[dict[str, Any]] = []
    round_number = 0
    while len(chosen) < count:
        added = False
        for items in by_skill.values():
            if round_number < len(items):
                chosen.append(items[round_number])
                added = True
                if len(chosen) >= count:
                    break
        if not added:
            break
        round_number += 1
    return chosen


def _speed_row(seconds: list[float], tokens: list[float]) -> dict[str, Any]:
    return {
        "items": len(seconds),
        "p50_ms": round(_percentile(seconds, 0.50) * 1000, 1),
        "p95_ms": round(_percentile(seconds, 0.95) * 1000, 1),
        "mean_ms": round(statistics.fmean(seconds) * 1000, 1) if seconds else None,
        "min_ms": round(min(seconds) * 1000, 1) if seconds else None,
        "max_ms": round(max(seconds) * 1000, 1) if seconds else None,
        "new_tokens_mean": round(statistics.fmean(tokens), 1) if tokens else None,
        "tokens_per_second": (
            round(sum(tokens) / sum(seconds), 2) if seconds and sum(seconds) else None
        ),
    }


def _generation_report(count: int, tenant: int | None) -> dict[str, Any]:
    conditions = _modules()["conditions"]
    scoring = _modules()["scoring"]
    loaded_at = time.perf_counter()
    model = conditions.load_model()
    load_seconds = time.perf_counter() - loaded_at
    params = conditions.generation_params()
    per_item: list[dict[str, Any]] = []
    for item in _sample_items(count):
        messages = conditions.messages(item, ())
        start = time.perf_counter()
        result = model.generate(messages, params, None)
        seconds = time.perf_counter() - start
        mark = scoring.mark(item, result.text)
        evidence_start = time.perf_counter()
        hits = conditions.evidence(str(item["prompt"]), tenant)
        evidence_seconds = time.perf_counter() - evidence_start
        per_item.append(
            {
                "item_id": item["id"],
                "skill": item["skill"],
                "kind": item["expected"]["kind"],
                "seconds": round(seconds, 3),
                "prompt_tokens": result.prompt_tokens,
                "new_tokens": result.new_tokens,
                "stop_reason": result.stop_reason,
                "score": mark.score,
                "method": mark.method,
                "evidence_seconds": round(evidence_seconds, 4),
                "evidence_hits": len(hits),
            }
        )
    seconds = [float(row["seconds"]) for row in per_item]
    tokens = [float(row["new_tokens"]) for row in per_item]
    evidence = [float(row["evidence_seconds"]) for row in per_item]
    return {
        "device": type(model).__name__,
        "load_seconds": round(load_seconds, 3),
        "max_new_tokens": params.max_new_tokens,
        "temperature": params.temperature,
        "unaided_generation": _speed_row(seconds, tokens),
        "evidence_read_ms": {
            "p50": round(_percentile(evidence, 0.50) * 1000, 2),
            "p95": round(_percentile(evidence, 0.95) * 1000, 2),
        },
        "full_run_extrapolation_seconds": round(
            statistics.fmean(seconds) * len(_modules()["bank"].items(split="eval")), 1
        )
        if seconds
        else None,
        "rows": per_item,
    }


def _bare_prompt(item: dict[str, Any]) -> str:
    """The item without its task framing, and without its options."""
    body = _INSTRUCTION.split(str(item["prompt"]), maxsplit=1)[-1]
    if item["expected"]["kind"] == "option":
        body = _OPTIONS.split(body, maxsplit=1)[0]
    return str(body).strip()


def _wanted_phrase(item: dict[str, Any]) -> str:
    """The words a response must contain for this probe to count as recalled."""
    answer = str(_expected_answer(item)).strip().strip("\"'`")
    if str(item["expected"]["kind"]) == "option":
        labels = dict(_LABELS.findall(str(item["prompt"])))
        return str(labels.get(answer, "")).strip().strip("\"'`")
    return answer


def _says(text: str, wanted: str) -> bool:
    """Whether the wording stands as its own word, not as part of a longer one."""
    haystack = _SPACES.sub(" ", text).casefold()
    folded = _SPACES.sub(" ", wanted).casefold().strip(" .,;:!?")
    if not folded:
        return False
    return re.search(rf"(?<!\w){re.escape(folded)}(?!\w)", haystack) is not None


def _informative(item: dict[str, Any], bare: str) -> bool:
    """Whether this probe can say anything at all about memorisation.

    A probe fails that test when its answer is a common short word, and when the
    bare prompt still shows the answer, because then a response that copies what
    it is reading looks exactly like a response that remembers it.
    """
    wanted = _wanted_phrase(item)
    if len(wanted) < MIN_RECALL_ANSWER_CHARS:
        return False
    return not _says(bare, wanted)


def _names_answer(item: dict[str, Any], text: str) -> bool:
    scoring = _modules()["scoring"]
    kind = str(item["expected"]["kind"])
    if kind == "number":
        wanted = float(_expected_answer(item))
        return scoring._same_number(float(scoring._number(text) or 0.0), wanted)
    return _says(text, _wanted_phrase(item))


def _recall_probes(count: int) -> tuple[list[dict[str, Any]], int]:
    """The held-out items worth probing, and how many this pass left out."""
    kept: list[dict[str, Any]] = []
    skipped = 0
    for item in _sample_items(count * 8):
        if item["expected"]["kind"] not in {"option", "span", "number", "literal"}:
            continue
        bare = _bare_prompt(item)
        if not _informative(item, bare):
            skipped += 1
            continue
        kept.append({"item": item, "bare": bare})
        if len(kept) >= count:
            break
    return kept, skipped


def _recall_report(count: int) -> dict[str, Any]:
    conditions = _modules()["conditions"]
    model = conditions.load_model()
    params = conditions.generation_params()
    probes, skipped = _recall_probes(count)
    rows = []
    for probe in probes:
        item, bare = probe["item"], probe["bare"]
        result = model.generate([{"role": "user", "content": bare}], params, None)
        rows.append(
            {
                "item_id": item["id"],
                "kind": item["expected"]["kind"],
                "bare_prompt": bare[:120],
                "answer": str(_expected_answer(item))[:60],
                "wanted": _wanted_phrase(item)[:60],
                "response": result.text[:120],
                "names_the_answer": _names_answer(item, result.text),
            }
        )
    recalled = sum(1 for row in rows if row["names_the_answer"])
    return {
        "probes": len(rows),
        "uninformative_items_left_out": skipped,
        "min_answer_chars": MIN_RECALL_ANSWER_CHARS,
        "recalled": recalled,
        "recall_share": round(recalled / len(rows), 4) if rows else None,
        "rows": rows,
    }


def _skill_rows(rows: list[Any]) -> dict[str, Any]:
    """What one skill's stored results say, kept apart from the other seven."""
    scores = [float(row.score) for row in rows if row.score is not None]
    return {
        "items": len(rows),
        "scored": len(scores),
        "unassessed": len(rows) - len(scores),
        "correct": sum(1 for row in rows if row.correct is True),
        "mean_score": round(statistics.fmean(scores), 4) if scores else None,
        "passed": sum(1 for score in scores if score >= 0.8) if scores else 0,
        "empty_responses": sum(1 for row in rows if not (row.response or "").strip()),
    }


def _run_report(run_id: int, tenant: int | None) -> dict[str, Any]:
    """Everything a finished run measured, read back from the rows it wrote."""
    runner, profiles = _modules()["runner"], _modules()["profiles"]
    run = runner.snapshot(run_id, tenant)
    rows = runner.results(run_id, tenant)
    latencies = [float(row.latency_ms) for row in rows if row.latency_ms is not None]
    scores = [float(row.score) for row in rows if row.score is not None]
    wall = None
    if run["started_at"] and run["finished_at"]:
        started = datetime.fromisoformat(str(run["started_at"]))
        wall = round((datetime.fromisoformat(str(run["finished_at"])) - started).total_seconds(), 1)
    sheet = profiles.sheet(run_id, str(run["mode"]), tenant)
    waiting = [row.item_id for row in rows if row.kind == "rubric" and row.score is None]
    by_skill: dict[str, list[Any]] = {}
    for row in rows:
        by_skill.setdefault(row.skill, []).append(row)
    return {
        "run": run_id,
        "mode": run["mode"],
        "status": run["status"],
        "items": len(rows),
        "wall_seconds": wall,
        "provenance": {
            key: run[key]
            for key in ("model_id", "model_artifact", "tokenizer_id", "generation", "bank")
        },
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 1),
            "p95": round(_percentile(latencies, 0.95), 1),
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "min": round(min(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "scoring": {
            "scored": len(scores),
            "unassessed": len(rows) - len(scores),
            "mean_score": round(statistics.fmean(scores), 4) if scores else None,
            "passed": sum(1 for score in scores if score >= 0.8),
            "methods": dict(Counter(str(row.method) for row in rows)),
            "empty_responses": sum(1 for row in rows if not (row.response or "").strip()),
            "waiting_for_review": sorted(waiting),
        },
        "skills": {skill: _skill_rows(by_skill[skill]) for skill in sorted(by_skill)},
        "profiles": sheet["profiles"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=0, help="held-out items to generate for")
    parser.add_argument(
        "--recall", type=int, default=0, help="items to probe for memorised answers"
    )
    parser.add_argument("--repeats", type=int, default=200, help="scorer timing repeats")
    parser.add_argument("--run", type=int, default=0, help="a stored run to read back")
    parser.add_argument("--tenant", type=int, default=None)
    args = parser.parse_args()

    report: dict[str, Any] = {"bank": _bank_report(), "scorers": _scorer_report(args.repeats)}
    if args.items:
        report["generation"] = _generation_report(args.items, args.tenant)
    if args.recall:
        report["recall"] = _recall_report(args.recall)
    if args.run:
        report["run"] = _run_report(args.run, args.tenant)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
