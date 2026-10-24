"""The rung results the builder shows: what the model can do, measured.

The question rungs read the tenant's indexed lessons; the capability rungs
read the training state. They are kept together here so the ladder has one
place that decides whether a rung passed, and one cache to keep the page
from re-asking every probe on every load.
"""

from __future__ import annotations

import time
from typing import Any

from app.services import brain_instruction, vector_store
from app.services.brain_levels import level_order
from app.services.brain_rungs import (
    CORRECTION_LIMIT,
    PROBE_CACHE,
    RUNG_CACHE,
    RUNG_TESTS,
    answer_matches,
    cache_key,
    cached_rungs,
    rung_probes,
    subject_coverage,
)


def _probe_answer(tenant_id: int | None, question: str) -> str:
    """The answer the index gives a probe, cached because it never changes."""
    key = f"{cache_key(tenant_id)}|{question}"
    hits = PROBE_CACHE.get(key)
    if hits is None:
        try:
            hits = vector_store.search_lessons(question, n_results=1, tenant_id=tenant_id)
        except (OSError, ValueError):
            hits = []
        PROBE_CACHE[key] = hits
    if hits and hits[0].get("distance") is not None:
        return str(hits[0].get("answer", ""))
    return ""


def _probe_entries(tenant_id: int | None, probes: list[dict[str, Any]]) -> dict[str, Any]:
    """How many of the probes an answer matched, and the entries for each."""
    passed = 0
    passes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for probe in probes:
        answer = _probe_answer(tenant_id, probe["question"])
        entry = {
            "question": probe["question"],
            "expected": probe["answer"],
            "form": probe["form"],
            "answered": answer,
        }
        if answer_matches(answer, probe["answer"]):
            passed += 1
            passes.append(entry)
        else:
            failures.append(entry)
    return {"passed": passed, "passes": passes, "failures": failures[:4]}


def _probe_results(tenant_id: int | None) -> list[dict[str, Any]]:
    """The question rungs, one result for each rung that has probes."""
    results = []
    for rung, probes in rung_probes(tenant_id).items():
        counted = _probe_entries(tenant_id, probes)
        total = len(probes)
        results.append(
            {
                "rung": rung,
                "probes": total,
                "passed": counted["passed"],
                "rate": round(counted["passed"] / total, 2) if total else 0.0,
                "passes": counted["passes"],
                "failures": counted["failures"],
            }
        )
    return results


def _writing_rung(tenant_id: int | None) -> dict[str, Any]:
    """The writing rung: whether the writer wrote a held-out answer itself."""
    measurement = brain_instruction.stored_instruction(tenant_id)
    measured = bool(measurement.get("measured"))
    written = int(measurement.get("written") or 0)
    total = int(measurement.get("total") or 0)
    wrote = measured and written > 0
    detail = (
        f"the writer wrote {written} of {total} held-out answers"
        if measured
        else "no held-out writing measurement yet"
    )
    entry = {
        "question": "Write a new sentence about what you know",
        "expected": "an answer the writer produced itself",
        "form": "held-out request",
        "answered": f"the writer produced {written} of {total} answers itself",
    }
    failure = {
        "question": "Write a new sentence about what you know",
        "expected": "an answer the writer produced itself",
        "form": "not measured yet" if not measured else "looked up instead",
    }
    return {
        "rung": "0.0.4",
        "probes": 1,
        "passed": 1 if wrote else 0,
        "rate": 1.0 if wrote else 0.0,
        "passes": [entry] if wrote else [],
        "failures": [] if wrote else [failure],
        "detail": detail,
    }


def _correction_rung(brain_state: dict[str, Any]) -> dict[str, Any]:
    """The correction rung: how many taught corrections were kept."""
    kept = int(brain_state.get("learned_corrections", 0))
    entry = {
        "question": "Learn from a correction and keep it",
        "expected": "the corrected answer is used afterwards",
        "form": "kept correction",
        "answered": "it kept a correction and answers with it",
    }
    return {
        "rung": "0.0.6",
        "probes": CORRECTION_LIMIT,
        "passed": kept,
        "rate": round(kept / CORRECTION_LIMIT, 2) if CORRECTION_LIMIT else 0.0,
        "passes": [entry for _ in range(min(kept, CORRECTION_LIMIT))],
        "failures": []
        if kept >= CORRECTION_LIMIT
        else [
            {
                "question": "Learn from a correction and keep it",
                "expected": f"{CORRECTION_LIMIT} corrections answered correctly after being taught",
                "form": "never taught",
            }
        ],
        "detail": f"{kept} of {CORRECTION_LIMIT} corrections kept",
    }


def _exam_rung(knowledge: dict[str, Any]) -> dict[str, Any]:
    """The exam rung: whether a school grade has been passed."""
    grades = int(knowledge.get("passed_grades", 0))
    entry = {
        "question": "Pass the final exam",
        "expected": "one grade exam passed",
        "form": "from the training",
        "answered": f"{grades} grade exam(s) passed",
    }
    return {
        "rung": "0.0.7",
        "probes": 1,
        "passed": 1 if grades else 0,
        "rate": 1.0 if grades else 0.0,
        "passes": [entry] if grades else [],
        "failures": []
        if grades
        else [
            {
                "question": "Pass the final exam",
                "expected": "one grade exam passed",
                "form": "never taught",
            }
        ],
        "detail": f"{grades} grade exams passed" if grades else "no final exam passed yet",
    }


def _grounded_rung(brain_state: dict[str, Any]) -> dict[str, Any]:
    """The grounded rung: whether the confidence head knows the edge."""
    grounded = bool(brain_state.get("confidence_ready"))
    entry = {
        "question": "Say when you do not know",
        "expected": "a trained confidence head",
        "form": "from the training",
        "answered": "it says when it does not know",
    }
    return {
        "rung": "0.0.8",
        "probes": 1,
        "passed": 1 if grounded else 0,
        "rate": 1.0 if grounded else 0.0,
        "passes": [entry] if grounded else [],
        "failures": []
        if grounded
        else [
            {
                "question": "Say when you do not know",
                "expected": "a trained confidence head",
                "form": "never taught",
            }
        ],
        "detail": (
            "it knows the edge of its knowledge"
            if grounded
            else "it has not learned where its knowledge ends"
        ),
    }


def _depth_rung(subjects: dict[str, int]) -> dict[str, Any]:
    """The depth rung: whether one subject holds five questions."""
    deepest = max(subjects, key=lambda key: subjects[key]) if subjects else ""
    count = max(subjects.values()) if subjects else 0
    deep = 1 if any(value >= 5 for value in subjects.values()) else 0
    entry = {
        "question": "Hold one subject in depth",
        "expected": "five questions on one topic",
        "form": "from the training",
        "answered": f"deepest subject: {deepest} ({count}) questions",
    }
    return {
        "rung": "0.0.9",
        "probes": 1,
        "passed": deep,
        "rate": 1.0 if deep else 0.0,
        "passes": [entry] if deep else [],
        "failures": []
        if deep
        else [
            {
                "question": "Hold one subject in depth",
                "expected": "five questions on one topic",
                "form": "never taught",
            }
        ],
        "detail": f"deepest subject: {deepest} ({count})" if subjects else "no subject yet",
    }


def rung_results(
    tenant_id: int | None,
    force: bool = False,
    brain_state: dict[str, Any] | None = None,
    knowledge: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the rung tests and report how the model did on each."""
    if brain_state is None:
        from app.services.brain_state import state

        brain_state = state(tenant_id)
    knowledge = knowledge or {"lessons": 0, "passed_grades": 0, "stage": "nursery"}
    if not force:
        cached = cached_rungs(tenant_id)
        if cached is not None:
            return cached
    results = [
        *_probe_results(tenant_id),
        _writing_rung(tenant_id),
        _correction_rung(brain_state),
        _exam_rung(knowledge),
        _grounded_rung(brain_state),
        _depth_rung(subject_coverage(tenant_id)),
    ]
    for rung in results:
        rung["test"] = RUNG_TESTS.get(rung["rung"], "")
    results.sort(key=lambda item: level_order(item["rung"]))
    RUNG_CACHE[cache_key(tenant_id)] = (time.time(), results)
    return results
