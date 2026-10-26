"""Measure how Luke reads, connects and finishes a question.

Three probe sets, all on the tenant's real material and with no checkpoint
written:

- Reading: fixed questions read through ``nlu.read_question``; reports the
  words read, how many carry a dictionary meaning and synonyms, the question
  type and the key words.
- Synonym fallback: definition questions built from real lessons, where a
  WordNet synonym of a lesson's title word appears in that lesson's answer,
  the exact lesson lookup does not already answer the question, and the
  dictionary holds the question word. Reports the probes that reached the
  intended lesson, a different lesson, or none.
- Finishing: every taught lesson answer, finished twice; reports how many
  changed, how many were already finished, and whether finishing is
  idempotent.

Run with:
    .venv/bin/python scripts/measure_reading.py --tenant 0
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

READING_PROBES = (
    "What is a circle?",
    "What does the head do?",
    "Why is the sky blue?",
    "Write a short poem about the star.",
    "Tell me about the eye.",
    "Who is your creator?",
    "How many legs does a dog have?",
    "Where is the sun?",
    "Give me three fruits.",
    "What is a clarinet?",
    "How do I say hello in French?",
    "Tell me a story about the wind.",
)

_WORD = re.compile(r"[a-z]+")


@cache
def _modules() -> dict[str, Any]:
    """The app modules, loaded once with the repository on the import path."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return {
        "dictionary": importlib.import_module("app.services.dictionary"),
        "fallback": importlib.import_module("app.services.language.fallback"),
        "finishing": importlib.import_module("app.services.language.finishing"),
        "lessons": importlib.import_module("app.services.language.lessons"),
        "matching": importlib.import_module("app.services.language.matching"),
        "nlu": importlib.import_module("app.services.nlu"),
        "topics": importlib.import_module("app.services.language.topics"),
    }


def _reading(question: str) -> dict[str, Any]:
    modules = _modules()
    reading = modules["nlu"].read_question(question)
    words = [
        {
            "word": word.word,
            "lemma": word.lemma,
            "part_of_speech": word.part_of_speech,
            "meaning": word.meaning,
            "synonyms": list(word.synonyms),
        }
        for word in reading.words
    ]
    return {
        "question": question,
        "question_type": reading.question_type,
        "key_words": list(reading.key_words),
        "words": words,
        "with_meaning": sum(1 for word in words if word["meaning"]),
        "with_synonyms": sum(1 for word in words if word["synonyms"]),
    }


def _reading_report() -> dict[str, Any]:
    rows = [_reading(question) for question in READING_PROBES]
    types: dict[str, int] = {}
    for row in rows:
        kind = row["question_type"] or "(not read)"
        types[kind] = types.get(kind, 0) + 1
    return {
        "questions": len(rows),
        "words": sum(len(row["words"]) for row in rows),
        "with_meaning": sum(row["with_meaning"] for row in rows),
        "with_synonyms": sum(row["with_synonyms"] for row in rows),
        "question_types": types,
        "rows": rows,
    }


def _probe_for(card: dict, modules: dict[str, Any], seen: set[str]) -> dict[str, Any] | None:
    """One definition question whose synonym names this lesson, if it exists."""
    dictionary = modules["dictionary"]
    title = modules["topics"].lesson_title(str(card.get("question") or ""))
    title_words = set(_WORD.findall(title.lower()))
    answer_words = set(_WORD.findall(str(card.get("answer") or "").lower()))
    for term in sorted(title_words):
        if len(term) < 4:
            continue
        for synonym in dictionary.search_expansions(term, per_word=6, max_terms=12):
            if " " in synonym or synonym in title_words or synonym not in answer_words:
                continue
            question = f"What is a {synonym}?"
            if question in seen or dictionary.lookup_word(synonym) is None:
                continue
            if modules["matching"]._matched_learned_lesson(question, modules["cards"]) is not None:
                continue
            seen.add(question)
            return {
                "question": question,
                "synonym": synonym,
                "term": term,
                "lesson": card.get("question"),
            }
    return None


def _synonym_probes(cards: list[dict], modules: dict[str, Any], limit: int) -> list[dict]:
    probes: list[dict] = []
    seen: set[str] = set()
    for card in cards:
        probe = _probe_for(card, modules, seen)
        if probe is not None:
            probes.append(probe)
            if len(probes) >= limit:
                break
    return probes


def _fallback_report(probes: list[dict]) -> dict[str, Any]:
    fallback = _modules()["fallback"]
    rows: list[dict[str, Any]] = []
    counts = {"intended_lesson": 0, "other_lesson": 0, "no_lesson": 0}
    for probe in probes:
        attempt = fallback.find_fallback(probe["question"], _modules()["cards"])
        if attempt is not None and attempt.answer is not None:
            reached = attempt.answer.matched_question
            outcome = "intended_lesson" if reached == probe["lesson"] else "other_lesson"
            detail = attempt.connection.detail
        else:
            reached = ""
            outcome = "no_lesson"
            detail = (
                attempt.connection.detail if attempt is not None else "the fallback did not run"
            )
        counts[outcome] += 1
        rows.append({**probe, "outcome": outcome, "reached": reached, "detail": detail})
    total = len(rows)
    return {
        "probes": total,
        **counts,
        "intended_share": round(counts["intended_lesson"] / total, 4) if total else None,
        "rows": rows,
    }


def _finishing_report(cards: list[dict]) -> dict[str, Any]:
    finish = _modules()["finishing"].finish
    changed: list[str] = []
    already = 0
    idempotent = 0
    answers = [str(card.get("answer") or "") for card in cards]
    for answer in answers:
        once = finish(answer)
        if once == answer:
            already += 1
        elif len(changed) < 5:
            changed.append(f"{answer!r} -> {once!r}")
        if finish(once) == once:
            idempotent += 1
    return {
        "answers": len(answers),
        "already_finished": already,
        "changed": len(answers) - already,
        "idempotent": idempotent,
        "examples": changed,
    }


def _report(tenant: int, limit: int) -> dict[str, Any]:
    modules = _modules()
    modules["cards"] = modules["lessons"].all_lesson_cards(tenant)
    probes = _synonym_probes(modules["cards"], modules, limit)
    return {
        "tenant": tenant,
        "lesson_cards": len(modules["cards"]),
        "reading": _reading_report(),
        "synonym_fallback": _fallback_report(probes),
        "finishing": _finishing_report(modules["cards"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", type=int, default=0)
    parser.add_argument("--probes", type=int, default=40)
    args = parser.parse_args()
    print(json.dumps(_report(args.tenant, args.probes), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
