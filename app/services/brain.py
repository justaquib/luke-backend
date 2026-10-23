"""A small neural brain for the local assistant.

The assistant already has a neural cortex: the MiniLM transformer that
ChromaDB uses to embed every taught question, answer and document. This
module adds the missing parts of a brain on top of it:

* a **sentence writer** - a transformer trained on the material the tenant
  taught, so the assistant can write a sentence instead of only quoting one,
* an **emotion head** - a trained classifier that reads the feeling in a
  message,
* a **clarify head** - a trained classifier that decides whether the
  assistant has enough to answer or should ask a question back,
* a **confidence head** - a trained classifier that estimates whether the
  answer it is about to give is likely to be right.

All of it is trained from the tenant's own material and from the corrections
the trainer gives, and it runs locally on PyTorch. No model is downloaded.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from app.services import embeddings, training_progress, vector_store
from app.services.brain_paths import brain_dir, brain_path, writer_path
from app.services.brain_paths import tenant_key as _tenant_key
from app.services.heads import LinearHead
from app.services.language import runtime as language_runtime

if TYPE_CHECKING:
    from app.services.language.training_materials import TrainingSet


_LOCK = threading.RLock()

LOGGER = logging.getLogger(__name__)




_EMOTION_WORDS: dict[str, tuple[str, ...]] = {
    "happy": (
        "happy",
        "glad",
        "pleased",
        "joy",
        "joyful",
        "excited",
        "cheerful",
        "smile",
        "delight",
    ),
    "sad": ("sad", "unhappy", "upset", "cry", "crying", "lonely", "sorry", "hurt"),
    "angry": ("angry", "mad", "annoyed", "annoying", "cross", "furious", "upset"),
    "tired": ("tired", "sleepy", "exhausted", "weary", "sleep"),
    "scared": ("scared", "afraid", "frightened", "nervous", "worried", "fear"),
    "surprised": ("surprised", "shock", "shocked", "amazed", "wow", "astonished"),
    "calm": ("calm", "quiet", "peaceful", "relaxed", "fine", "good", "gentle"),
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*|[.,!?]")


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in _WORD_RE.findall(text or "")]


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]


# ---------------------------------------------------------------- the writer


# ------------------------------------------------------------- the levels


LUKE_LEVELS: list[dict[str, Any]] = [
    {
        "level": "0.0.0",
        "name": "Seed",
        "description": "It holds what it was taught and can look it up, but it cannot answer anything on its own.",
        "reference": "The floor of the scale: material without general answering.",
    },
    {
        "level": "0.0.1",
        "name": "Cued",
        "description": "It finds the answer when asked exactly as it was taught.",
        "reference": "Teach it something and it can give that back.",
    },
    {
        "level": "0.0.2",
        "name": "Paraphrased",
        "description": "It finds the answer when the question is worded differently.",
        "reference": "It survives a rewording of the question.",
    },
    {
        "level": "0.0.3",
        "name": "Generalised",
        "description": "It answers about the topic without the exact question.",
        "reference": "It knows the topic, not only the sentence.",
    },
    {
        "level": "0.0.4",
        "name": "Written",
        "description": "It writes new sentences in the style it was taught.",
        "reference": "It composes instead of quoting.",
    },
    {
        "level": "0.0.5",
        "name": "Explained",
        "description": "It explains an answer when it is asked to.",
        "reference": "It can say why, not only what.",
    },
    {
        "level": "0.0.6",
        "name": "Corrected",
        "description": "It learns from a correction and keeps it.",
        "reference": "A wrong answer turns into a right one, and stays right.",
    },
    {
        "level": "0.0.7",
        "name": "Examined",
        "description": "It passes the final exam of a school grade.",
        "reference": "The school says it is ready for the next grade.",
    },
    {
        "level": "0.0.8",
        "name": "Grounded",
        "description": "It says it does not know instead of guessing.",
        "reference": "Knowing the edge of its knowledge is a kind of intelligence.",
    },
    {
        "level": "0.0.9",
        "name": "Teacher",
        "description": "It holds a whole subject: many questions, one topic.",
        "reference": "Depth, not a list of facts.",
    },
    {
        "level": "0.1.0",
        "name": "Independent",
        "description": "It answers a question it was never taught, in its own words.",
        "reference": "The long road: every lesson, every correction and every exam widens what it can do.",
    },
]

# The level the brain starts on, and the one it can climb to today. Further
# rungs are added between the milestones (0.1.1, 0.2.0 and so on) as the
# brain learns to do more, so the numbers keep the same three-segment shape.
_SEED_LEVEL = "0.0.0"

BENCHMARK_TITLE = "Luna 6 · Frontier"
BENCHMARK_NOTE = "Luna 6 is the class of a frontier chat model such as GPT or Claude at the top of its range. Luke is not one of those: it is built from scratch and trained only on what you teach it, so it is measured against that bar honestly."

GENERALITY_PROBES = [
    "Who wrote the play Romeo and Juliet?",
    "Write a short poem about the rain.",
    "How many continents are there?",
    "What is the boiling point of water?",
    # Most AI models cannot answer this one, so it is a fair test of what this
    # model really knows for itself.
    "What are the rudiment?",
]

_PROBE_LIMIT = 6
CORRECTION_LIMIT = 5
_RUNG_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_RUNG_CACHE_TTL = 1800.0
# The same probe questions are asked every time, so their answers are kept.
_PROBE_CACHE: dict[str, list[dict[str, Any]]] = {}


def _reframe(question: str) -> str:
    """Ask the same question in a different way."""
    text = question.strip().rstrip("?.! ")
    if not text:
        return question
    return f"Could you tell me {text[0].lower() + text[1:]}?"


def _swap_synonym(question: str) -> str | None:
    """Ask the question with a word replaced by a synonym."""
    for word in re.findall(r"[A-Za-z]{4,}", question):
        try:
            from app.services.dictionary import search_expansions

            synonyms = [item for item in search_expansions(word) if item != word.lower()]
        except Exception:
            synonyms = []
        if not synonyms:
            continue
        replacement = synonyms[0]
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        swapped = pattern.sub(replacement, question, count=1)
        if swapped.lower() != question.lower():
            return swapped
    return None


def _answer_matches(produced: str, expected: str) -> bool:
    """Whether the answer the model gave says the same thing."""
    if not produced or not expected:
        return False
    expected_tokens = set(_tokenize(expected))
    produced_tokens = set(_tokenize(produced))
    if not expected_tokens:
        return False
    overlap = len(expected_tokens & produced_tokens)
    return overlap / len(expected_tokens) >= 0.6


def rung_probes(tenant_id: int | None) -> dict[str, list[dict[str, Any]]]:
    """The question tests for each rung, built from what the tenant has taught.

    Each rung asks for the taught material in a harder way, so teaching more
    (and teaching it in more ways) is what moves the level up.
    """
    from app.services import language_logic

    try:
        cards = language_logic.all_lesson_cards(tenant_id)
    except Exception:
        cards = []

    taught = [
        {"question": card["question"], "answer": card["answer"]}
        for card in cards
        if card.get("question") and card.get("answer")
    ][:_PROBE_LIMIT]

    exact = [
        {"question": item["question"], "answer": item["answer"], "form": "asked as taught"}
        for item in taught
    ]
    reworded = [
        {
            "question": _reframe(item["question"]),
            "answer": item["answer"],
            "form": "asked in other words",
        }
        for item in taught
    ]
    swapped: list[dict[str, Any]] = []
    for item in taught:
        alternative = _swap_synonym(item["question"])
        if alternative:
            swapped.append(
                {
                    "question": alternative,
                    "answer": item["answer"],
                    "form": "asked with another word",
                }
            )

    explained = [
        {
            "question": f"Explain: {item['question']}",
            "answer": item["answer"],
            "form": "asked for an explanation",
        }
        for item in taught
    ]

    return {
        "0.0.1": exact,
        "0.0.2": reworded,
        "0.0.3": swapped,
        "0.0.5": explained,
    }


_RUNG_TESTS = {
    "0.0.1": "Can it find the answer when asked exactly as it was taught?",
    "0.0.2": "Can it find the answer when the question is worded differently?",
    "0.0.3": "Can it answer about the topic without the exact question?",
    "0.0.4": "Can it write new sentences in the style you taught?",
    "0.0.5": "Can it explain an answer when it is asked to?",
    "0.0.6": "Has it learned from a correction and kept it?",
    "0.0.7": "Has it passed the final exam of a grade?",
    "0.0.8": "Does it say it does not know instead of guessing?",
    "0.0.9": "Does it hold one whole subject in depth?",
    "0.1.0": "Can it answer a question it was never taught, in its own words?",
}


def rung_results(
    tenant_id: int | None,
    force: bool = False,
    brain_state: dict[str, Any] | None = None,
    knowledge: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run the rung tests and report how the model did on each.

    The rungs are the ones a model built from scratch can climb: the question
    tests, plus the capabilities that come from the training itself.
    """
    import time

    brain_state = brain_state if brain_state is not None else state(tenant_id)
    knowledge = (
        knowledge
        if knowledge is not None
        else {"lessons": 0, "passed_grades": 0, "stage": "nursery"}
    )

    cache_key = f"{_tenant_key(tenant_id)}"
    if not force:
        cached = _RUNG_CACHE.get(cache_key)
        if cached and (time.time() - cached[0]) < _RUNG_CACHE_TTL:
            return cached[1]

    results: list[dict[str, Any]] = []
    for rung, probes in rung_probes(tenant_id).items():
        passed = 0
        passes: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for probe in probes:
            answer = ""
            cache_key = f"{_tenant_key(tenant_id)}|{probe['question']}"
            hits = _PROBE_CACHE.get(cache_key)
            if hits is None:
                try:
                    hits = vector_store.search_lessons(
                        probe["question"],
                        n_results=1,
                        tenant_id=tenant_id,
                    )
                except Exception:
                    hits = []
                _PROBE_CACHE[cache_key] = hits
            if hits and hits[0].get("distance") is not None:
                answer = hits[0].get("answer", "")
            if _answer_matches(answer, probe["answer"]):
                passed += 1
                passes.append(
                    {
                        "question": probe["question"],
                        "expected": probe["answer"],
                        "form": probe["form"],
                        "answered": answer,
                    }
                )
            else:
                failures.append(
                    {
                        "question": probe["question"],
                        "expected": probe["answer"],
                        "form": probe["form"],
                        "answered": answer,
                    }
                )
        total = len(probes)
        results.append(
            {
                "rung": rung,
                "probes": total,
                "passed": passed,
                "rate": round(passed / total, 2) if total else 0.0,
                "passes": passes,
                "failures": failures[:4],
            }
        )

    # The capabilities that come from the training itself.
    subjects = _subject_coverage(tenant_id)
    # The transformer is the writer now, so it counts as the writing model.
    written = bool(brain_state.get("trained")) or load_writer(tenant_id) is not None
    results.append(
        {
            "rung": "0.0.4",
            "probes": 1,
            "passed": 1 if written else 0,
            "rate": 1.0 if written else 0.0,
            "passes": [
                {
                    "question": "Write a new sentence about what you know",
                    "expected": "a sentence in the taught style",
                    "form": "from the training",
                    "answered": "it writes new sentences in the style it was taught",
                }
            ]
            if written
            else [],
            "failures": []
            if written
            else [
                {
                    "question": "Write a new sentence about what you know",
                    "expected": "a sentence in the taught style",
                    "form": "never taught",
                }
            ],
            "detail": "the writing model is trained"
            if written
            else "the writing model is not trained yet",
        }
    )
    corrections = brain_state.get("corrections", 0)
    kept = brain_state.get("learned_corrections", 0)
    results.append(
        {
            "rung": "0.0.6",
            "probes": CORRECTION_LIMIT,
            "passed": kept,
            "rate": round(kept / CORRECTION_LIMIT, 2) if CORRECTION_LIMIT else 0.0,
            "passes": [
                {
                    "question": "Learn from a correction and keep it",
                    "expected": "the corrected answer is used afterwards",
                    "form": "kept correction",
                    "answered": "it kept a correction and answers with it",
                }
                for _ in range(min(kept, CORRECTION_LIMIT))
            ],
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
    )
    grades = knowledge.get("passed_grades", 0)
    results.append(
        {
            "rung": "0.0.7",
            "probes": 1,
            "passed": 1 if grades else 0,
            "rate": 1.0 if grades else 0.0,
            "passes": [
                {
                    "question": "Pass the final exam",
                    "expected": "one grade exam passed",
                    "form": "from the training",
                    "answered": f"{grades} grade exam(s) passed",
                }
            ]
            if grades
            else [],
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
    )
    grounded = bool(brain_state.get("confidence_ready"))
    results.append(
        {
            "rung": "0.0.8",
            "probes": 1,
            "passed": 1 if grounded else 0,
            "rate": 1.0 if grounded else 0.0,
            "passes": [
                {
                    "question": "Say when you do not know",
                    "expected": "a trained confidence head",
                    "form": "from the training",
                    "answered": "it says when it does not know",
                }
            ]
            if grounded
            else [],
            "failures": []
            if grounded
            else [
                {
                    "question": "Say when you do not know",
                    "expected": "a trained confidence head",
                    "form": "never taught",
                }
            ],
            "detail": "it knows the edge of its knowledge"
            if grounded
            else "it has not learned where its knowledge ends",
        }
    )
    deep = 1 if any(count >= 5 for count in subjects.values()) else 0
    results.append(
        {
            "rung": "0.0.9",
            "probes": 1,
            "passed": deep,
            "rate": 1.0 if deep else 0.0,
            "passes": [
                {
                    "question": "Hold one subject in depth",
                    "expected": "five questions on one topic",
                    "form": "from the training",
                    "answered": f"deepest subject: {max(subjects, key=lambda key: subjects[key])} ({max(subjects.values()) if subjects else 0}) questions",
                }
            ]
            if deep
            else [],
            "failures": []
            if deep
            else [
                {
                    "question": "Hold one subject in depth",
                    "expected": "five questions on one topic",
                    "form": "never taught",
                }
            ],
            "detail": f"deepest subject: {max(subjects, key=lambda key: subjects[key])} ({max(subjects.values()) if subjects else 0})"
            if subjects
            else "no subject yet",
        }
    )

    for rung in results:
        rung["test"] = _RUNG_TESTS.get(rung["rung"], "")
    results.sort(key=lambda item: level_order(item["rung"]))
    _RUNG_CACHE[cache_key] = (time.time(), results)
    return results


def _subject_coverage(tenant_id: int | None) -> dict[str, int]:
    """How many questions the tenant holds for each subject."""
    from app.services import language_logic

    try:
        cards = language_logic.all_lesson_cards(tenant_id)
    except Exception:
        return {}
    counts: dict[str, int] = {}
    for card in cards:
        subject = (card.get("subject") or "General").strip() or "General"
        counts[subject] = counts.get(subject, 0) + 1
    return counts


def _instruction_path(tenant_id: int | None) -> Path:
    """Where the last instruction measurement is kept."""
    return brain_dir() / f"instruction-{_tenant_key(tenant_id)}.json"


def _unmeasured_instruction() -> dict[str, Any]:
    """The instruction score before the trainer has run the first test."""
    return {
        "score": 0.0,
        "passed": 0,
        "total": 0,
        "probes": [],
        "measured": False,
        "measured_at": None,
    }


def stored_instruction(tenant_id: int | None) -> dict[str, Any]:
    """The last instruction measurement that was stored, if there is one."""
    try:
        data = json.loads(_instruction_path(tenant_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _unmeasured_instruction()
    return data if isinstance(data, dict) else _unmeasured_instruction()


def save_instruction(tenant_id: int | None, result: dict[str, Any]) -> None:
    """Keep the instruction measurement beside the tenant's other numbers."""
    path = _instruction_path(tenant_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **result,
        "measured": True,
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def instruction_score(tenant_id: int | None) -> dict[str, Any]:
    """How well the model answers the requests it was never trained on.

    The measurement takes about a minute, so the last one is stored and the
    brain page reads the stored number instead of measuring on every load.
    A fresh measurement is taken when the trainer runs the level test, or
    when a training has just finished.
    """
    return stored_instruction(tenant_id)


def remeasure_instruction(tenant_id: int | None) -> dict[str, Any]:
    """Run the instruction test again and keep the measurement."""
    from app.services.language import evaluation as evaluation_service
    from app.services.language.lessons import all_lesson_cards

    try:
        cards = all_lesson_cards(tenant_id)
    except (OSError, ValueError):
        cards = []
    result = evaluation_service.evaluate(tenant_id, cards, load_writer(tenant_id))
    save_instruction(tenant_id, result)
    return result


def generality_score(tenant_id: int | None) -> dict[str, Any]:
    """How far the brain can answer beyond what it was taught.

    This is what separates a small local model from a frontier one: a frontier
    model answers questions it has never seen, this one cannot.
    """
    from app.services import language_logic

    answered = 0
    probes: list[dict[str, Any]] = []
    for probe in GENERALITY_PROBES:
        try:
            result = language_logic.preview_answer(probe, tenant_id)
        except Exception:
            result = {"answer": "", "source": "not_learned"}
        source = result.get("source", "not_learned")
        # A dictionary lookup is a lookup, not general reasoning.
        general = source not in {"not_learned", "dictionary"}
        answered += 1 if general else 0
        probes.append({"question": probe, "answered": general, "source": source})
    score = answered / max(len(GENERALITY_PROBES), 1)
    if score >= 0.75:
        cap = 6
    elif score >= 0.5:
        cap = 5
    elif score >= 0.25:
        cap = 4
    else:
        cap = 3
    return {"score": round(score, 2), "cap": cap, "probes": probes}


# The ladder is measured against the transformer's vocabulary, the one live
# token list. The threshold counts the words the model writes with.
_CANONICAL_VOCABULARY_FLOOR = 400


def model_capabilities(
    tenant_id: int | None, brain_state: dict[str, Any], knowledge: dict[str, Any]
) -> list[dict[str, Any]]:
    """What the brain can actually do, in the order the levels need it.

    The level follows the capabilities, not the size of the data set: a large
    collection of documents does not make a model reason.

    Every vocabulary check here reads the transformer's vocabulary, so the
    words it counts are the words the answering model actually uses.
    """
    from app.services import vocabulary as vocabulary_service

    canonical = vocabulary_service.canonical(tenant_id)
    vocabulary_ready = len(canonical) > 0 and len(canonical.index) == len(canonical)
    return [
        {
            "key": "taught",
            "name": "Taught",
            "ready": knowledge.get("lessons", 0) > 0,
            "detail": f"{knowledge.get('lessons', 0)} lessons taught",
        },
        {
            "key": "recalled",
            "name": "Recalled",
            "ready": (
                knowledge.get("lessons", 0) > 0
                and brain_state.get("trained", False)
                and vocabulary_ready
            ),
            "detail": f"{len(canonical)} words it can write with",
        },
        {
            "key": "searching",
            "name": "Searching",
            "ready": knowledge.get("lessons", 0) > 0,
            "detail": "searches lessons, answers, intents and documents",
        },
        {
            "key": "explaining",
            "name": "Explaining",
            "ready": True,
            "detail": "shows the reasoning and the nodes it visited",
        },
        {
            "key": "writing",
            "name": "Writing",
            "ready": bool(brain_state.get("trained")),
            "detail": "writes its own sentences",
        },
        {
            "key": "empathetic",
            "name": "Empathetic",
            "ready": bool(brain_state.get("emotion_ready")),
            "detail": "reads the feeling in a message",
        },
        {
            "key": "curious",
            "name": "Curious",
            "ready": bool(brain_state.get("clarify_ready")),
            "detail": "ask back when the evidence is thin",
        },
        {
            "key": "self_correcting",
            "name": "Self-correcting",
            "ready": brain_state.get("corrections", 0) > 0,
            "detail": "learn from the trainer's corrections and remember them",
        },
        {
            "key": "examined",
            "name": "Examined",
            "ready": knowledge.get("passed_grades", 0) > 0,
            "detail": f"{knowledge.get('passed_grades', 0)} grade exams passed",
        },
        {
            "key": "autonomous",
            "name": "Autonomous",
            "ready": (
                knowledge.get("passed_grades", 0) > 0
                and brain_state.get("corrections", 0) > 0
                and brain_state.get("clarify_ready")
                and len(canonical) >= _CANONICAL_VOCABULARY_FLOOR
            ),
            "detail": "keep its own knowledge in order",
        },
    ]


# Stages of growth on the maturity side: how much material the brain has
# taken in. They sit on their own human scale, like the stages of a growing
# child, and never borrow the names of the model levels, because volume is
# not intelligence.
_MATURITY_STAGES: tuple[tuple[int, str, str], ...] = (
    (0, "Newborn", "Nothing is indexed yet; the brain has not started to take in the world."),
    (1, "First words", "The first lesson or answer is in; recognition begins."),
    (25, "Toddler", "A handful of words and lessons; it repeats what it was taught."),
    (100, "Schoolchild", "Enough material to answer across several subjects."),
    (500, "Scholar", "Deep material with documents, many topics and whole subjects."),
)


def training_volume_level(items: int) -> dict[str, Any]:
    """Where the amount of learned material sits on the maturity scale.

    Volume is not intelligence, so it never touches the model level: a
    hundred thousand documents still do not make a model reason. The stages
    follow the human brain instead, from newborn to scholar.
    """
    index = 0
    for position, stage in enumerate(_MATURITY_STAGES):
        if items >= stage[0]:
            index = position
    stages = [
        {
            "title": title,
            "items": threshold,
            "detail": detail,
            "reached": position <= index,
            "current": position == index,
        }
        for position, (threshold, title, detail) in enumerate(_MATURITY_STAGES)
    ]
    _, title, detail = _MATURITY_STAGES[index]
    next_stage: dict[str, Any] | None = None
    if index + 1 < len(_MATURITY_STAGES):
        threshold, next_title, _next_detail = _MATURITY_STAGES[index + 1]
        next_stage = {"title": next_title, "items": threshold}
    return {
        "title": title,
        "detail": detail,
        "items": items,
        "stages": stages,
        "next_stage": next_stage,
    }


def level_order(level: str) -> tuple[int, int, int]:
    """The ladder id as three numbers, so ids sort as 0.0.1, 0.0.9, 0.1.0."""
    values: list[int] = []
    for part in str(level).split("."):
        try:
            values.append(int(part))
        except ValueError:
            values.append(0)
    while len(values) < 3:
        values.append(0)
    return (values[0], values[1], values[2])


def level_value(level: str) -> float:
    """The same ladder id as one number, for bars and progress shares."""
    major, minor, patch = level_order(level)
    return major + minor / 10 + patch / 100


def _level_title(level: str) -> str:
    return f"Luke {level}"


def _level_entry(level: str) -> dict[str, Any]:
    """The ladder entry for a level id."""
    for entry in LUKE_LEVELS:
        if str(entry["level"]) == level:
            return {**entry, "title": _level_title(level)}
    return {**LUKE_LEVELS[0], "title": _level_title(_SEED_LEVEL)}


def _next_level_entry(level: str) -> dict[str, Any] | None:
    order = level_order(level)
    for entry in LUKE_LEVELS:
        if level_order(str(entry["level"])) > order:
            return {**entry, "title": f"{_level_title(str(entry['level']))} · {entry['name']}"}
    return None


def _coverage(tenant_id: int | None) -> dict[str, Any]:
    """What the assistant can read today: books, trained answers, lessons."""
    coverage = {
        "documents": 0,
        "chunks": 0,
        "trained_answers": 0,
        "lessons": 0,
        "passed_grades": 0,
    }
    try:
        from app.db.database import Document, Faq, SessionLocal
        from sqlalchemy import select

        with SessionLocal() as db:
            document_query = select(Document)
            answer_query = select(Faq).where(Faq.status == "approved")
            if tenant_id is not None:
                document_query = document_query.where(Document.tenant_id == tenant_id)
                answer_query = answer_query.where(Faq.tenant_id == tenant_id)
            coverage["documents"] = len(
                {row.source for row in db.execute(document_query).scalars().all()}
            )
            coverage["trained_answers"] = len(db.execute(answer_query).scalars().all())
    except Exception:
        pass
    try:
        coverage["chunks"] = vector_store.collection_size()
    except Exception:
        pass
    return coverage


def next_steps(
    tenant_id: int | None,
    brain_state: dict[str, Any],
    knowledge: dict[str, Any],
    rungs: list[dict[str, Any]],
    level: str,
    coverage: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """What to do next to advance, in the order it helps most.

    Each step points at the part of the app that feeds the assistant: the
    documents it reads, the answers the Trainer stores, or the lessons it
    learns in the school.
    """
    coverage = coverage or {}
    kept = brain_state.get("learned_corrections", 0)
    documents = coverage.get("documents", 0)
    chunks = coverage.get("chunks", 0)
    answers = coverage.get("trained_answers", 0)
    steps: list[dict[str, Any]] = []

    if documents == 0:
        steps.append(
            {
                "action": "add_document",
                "title": "Give it a book to read",
                "detail": "Luke has no document to read yet. Add a PDF, a novel or any related file in the Trainer's Documents tab: the pages are split into chunks, every chunk is embedded for meaning, and the passages are read back word by word when a question needs them.",
                "link": "/trainer?tab=documents",
                "link_label": "Add a document",
            }
        )
    elif chunks < 40:
        steps.append(
            {
                "action": "add_document",
                "title": "Give it more pages to read",
                "detail": f"Only {chunks} chunks are indexed. More pages means more text chunked, embedded and read word by word.",
                "link": "/trainer?tab=documents",
                "link_label": "Add a document",
            }
        )

    if answers < 5:
        steps.append(
            {
                "action": "add_answer",
                "title": "Store more trained answers",
                "detail": "The Trainer keeps the question and answer pairs. The more it holds, the more it can answer without guessing.",
                "link": "/trainer?tab=new",
                "link_label": "Add a Q&A",
            }
        )

    for rung in rungs:
        if rung["probes"] and rung["rate"] < 0.6:
            hints = {
                "0.0.1": "Teach Luke a question and answer it cannot already answer.",
                "0.0.2": "Teach the same answers again in different words, so it can find them when the question is reworded.",
                "0.0.3": "Teach the topic rather than the exact question, and add a few phrasings of it.",
                "0.0.4": "Train the brain so its writing model learns the style of your lessons.",
                "0.0.5": "Teach the reason as well as the answer, so it can explain it when asked.",
                "0.0.6": f"Mark an answer wrong in the chat and give the right one. {max(0, CORRECTION_LIMIT - kept)} more kept corrections and Luke passes this rung.",
                "0.0.7": "Pass the final exam of the grade it is studying.",
                "0.0.8": "Give it a few corrections so it learns where its knowledge ends and says so instead of guessing.",
                "0.0.9": "Teach more questions on one topic so it holds a whole subject.",
            }
            steps.append(
                {
                    "action": "add_lesson",
                    "title": f"Pass the Luke {rung['rung']} test",
                    "detail": hints.get(rung["rung"], "Teach more of what Luke should know."),
                    "link": "/trainer?tab=language",
                    "link_label": "Teach a lesson",
                }
            )
            break

    if knowledge.get("lessons", 0) == 0:
        steps.append(
            {
                "action": "add_lesson",
                "title": "Add your first lesson",
                "detail": "Luke starts empty. Teach it a question and the answer it should give.",
                "link": "/trainer?tab=language",
                "link_label": "Open the learner",
            }
        )

    grade = knowledge.get("stage", "nursery")
    if grade and grade != "complete":
        from app.services import language_logic

        grade_name = next(
            (stage["title"] for stage in language_logic.STAGES if stage["id"] == grade),
            str(grade).replace("_", " ").title(),
        )
        steps.append(
            {
                "action": "mock_test",
                "title": f"Take a mock test for {grade_name}",
                "detail": "A mock test is practice. It never advances the grade, so you can see what still needs learning.",
                "link": "/trainer?tab=language",
                "link_label": "Open the learner",
            }
        )
        steps.append(
            {
                "action": "final_exam",
                "title": f"Pass the {grade_name} final exam to advance the school grade",
                "detail": "The final exam moves Luke to the next grade once it answers at least 80% of the questions correctly.",
                "link": "/trainer?tab=language",
                "link_label": "Open the learner",
            }
        )
    if knowledge.get("lessons", 0) < 40:
        steps.append(
            {
                "action": "add_lesson",
                "title": "Add more knowledge to the school",
                "detail": f"{knowledge.get('lessons', 0)} lessons are taught. Every lesson is a question and answer the learner can use, and the school grades are built on them.",
                "link": "/trainer?tab=language",
                "link_label": "Teach a lesson",
            }
        )

    if brain_state.get("corrections", 0) == 0:
        steps.append(
            {
                "action": "rate_answers",
                "title": "Mark one answer wrong and correct it",
                "detail": "In the chat, mark an answer wrong and type the right one. Luke learns it, and the correction becomes part of its memory.",
                "link": "/chat",
                "link_label": "Open the chat",
            }
        )

    steps.append(
        {
            "action": "train_brain",
            "title": "Train the brain",
            "detail": "Training rebuilds the writing model and the emotion head from everything Luke has been taught, then tests the level again.",
            "link": "",
            "link_label": "",
        }
    )

    return steps


def _level_state_path(tenant_id: int | None) -> Path:
    """Where the trainer's approved model level is kept."""
    return brain_dir() / f"level-{_tenant_key(tenant_id)}.json"


def approved_level(tenant_id: int | None) -> str:
    """The level the trainer has approved, or the seed if none was."""
    path = _level_state_path(tenant_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _SEED_LEVEL
    level = str(data.get("approved_level", _SEED_LEVEL)) if isinstance(data, dict) else _SEED_LEVEL
    if level not in {str(entry["level"]) for entry in LUKE_LEVELS}:
        return _SEED_LEVEL
    return level


def approve_level(tenant_id: int | None) -> dict[str, Any]:
    """Promote the brain to the level its measurements have reached.

    Nothing is promoted on its own: the measured level can run ahead of the
    displayed one, and this call from the trainer closes the gap. The
    approval is kept with a small history, so the climb is on record.
    """
    level = model_level(tenant_id)
    measured = str(level["measured_level"])
    previous = approved_level(tenant_id)
    if level_order(measured) <= level_order(previous):
        raise ValueError("Luke has not measured a higher level than is approved.")
    path = _level_state_path(tenant_id)
    history: list[Any] = []
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        history = list(stored.get("history", [])) if isinstance(stored, dict) else []
    except (OSError, json.JSONDecodeError):
        history = []
    history.append({"level": measured, "approved_at": datetime.now(timezone.utc).isoformat()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"approved_level": measured, "history": history[-20:]}, indent=2),
        encoding="utf-8",
    )
    return {"approved_level": measured, "previous_level": previous, "history": history}


def _knowledge(tenant_id: int | None) -> dict[str, Any]:
    """What the school has taught: lessons, grades and the current one."""
    knowledge: dict[str, Any] = {"lessons": 0, "passed_grades": 0, "items": 0}
    try:
        from app.services import language_logic

        knowledge["lessons"] = len(language_logic.all_lesson_cards(tenant_id))
        model_state = language_logic.get_state(tenant_id)
        knowledge["passed_grades"] = len(model_state.get("passed_stages", []))
        knowledge["items"] = knowledge["lessons"]
        knowledge["stage"] = model_state.get("current_stage_id")
    except Exception:
        pass
    return knowledge


def _measured_level(rungs: list[dict[str, Any]], score: float) -> str:
    """The level the measurements qualify, before any approval.

    The rungs are taken in order from the seed, so a gap stops the climb,
    and the top rung answers a question it was never taught: the stretch a
    model reaches when its own words hold up without being taught.
    """
    ladder_ids = [str(entry["level"]) for entry in LUKE_LEVELS]
    by_id = {str(rung["rung"]): rung for rung in rungs}
    measured = _SEED_LEVEL
    for level_id in ladder_ids[1:]:
        rung = by_id.get(level_id)
        if rung is None or not rung["probes"] or rung["rate"] < 0.6:
            break
        measured = level_id
    if score >= 0.5:
        measured = ladder_ids[-1]
    return measured


def model_level(tenant_id: int | None, force: bool = False) -> dict[str, Any]:
    """The level the brain has reached, measured by what it can do.

    A local model is held to the same scale as a frontier one, and it is
    capped by how far it can answer on its own: a retriever cannot reach the
    levels that need real reasoning, however much it was taught. What it
    measures as and what is displayed are two different numbers: the level
    only moves when the trainer approves the promotion.
    """
    brain_state = state(tenant_id)
    knowledge = _knowledge(tenant_id)
    capabilities = model_capabilities(tenant_id, brain_state, knowledge)
    generality = generality_score(tenant_id)
    instructions = instruction_score(tenant_id)
    coverage = _coverage(tenant_id)
    knowledge["items"] = coverage.get("chunks", 0)
    coverage["lessons"] = knowledge.get("lessons", 0)
    coverage["passed_grades"] = knowledge.get("passed_grades", 0)
    coverage["current_stage"] = knowledge.get("stage", "nursery")

    rungs = rung_results(tenant_id, force=force, brain_state=brain_state, knowledge=knowledge)
    measured = _measured_level(rungs, generality["score"])

    # The displayed level never moves on its own: the trainer approves it.
    level = approved_level(tenant_id)
    current = _level_entry(level)
    following = _next_level_entry(measured if level_order(measured) > level_order(level) else level)
    steps = next_steps(tenant_id, brain_state, knowledge, rungs, level, coverage)
    missing = [capability["name"] for capability in capabilities if not capability["ready"]]
    next_detail = next(
        (step["detail"] for step in steps if step["title"].startswith("Pass the Luke")),
        steps[0]["detail"] if steps else "",
    )

    return {
        "level": level,
        "measured_level": measured,
        "approval_ready": level_order(measured) > level_order(level),
        "level_value": round(level_value(level), 4),
        "name": current["name"],
        "title": current["title"],
        "description": current["description"],
        "reference": current["reference"],
        "next_level": following["title"] if following else None,
        "next_requirement": next_detail if missing else "It has reached the highest level.",
        "capabilities": capabilities,
        "max_level": LUKE_LEVELS[-1]["level"],
        "ladder_size": len(LUKE_LEVELS),
        "levels": LUKE_LEVELS,
        "training_volume": training_volume_level(knowledge.get("items", 0)),
        "rungs": rungs,
        "next_steps": steps,
        "coverage": coverage,
        "generality": generality["score"],
        "generality_cap": generality["cap"],
        "generality_probes": generality["probes"],
        "instruction_following": instructions["score"],
        "instruction_probes": instructions["probes"],
        "instruction_total": instructions["total"],
        "instruction_measured": bool(instructions.get("measured")),
        "frontier_level": 6.0,
        "frontier_title": BENCHMARK_TITLE,
        "frontier_note": BENCHMARK_NOTE,
        "model_name": "Luke",
    }


# ------------------------------------------------------------- the brain


_WRITER_CACHE: dict[str, tuple[Any, float]] = {}


def load_writer(tenant_id: int | None):
    """The trained transformer of this tenant, if it has one.

    It is kept in memory, so the answers do not have to load it every time.
    The file is watched, so a model that was trained in another process is
    picked up as soon as it lands.
    """
    from app.services import transformer

    key = _tenant_key(tenant_id)
    path = writer_path(tenant_id)
    if not path.exists():
        return None
    stamp = path.stat().st_mtime
    cached = _WRITER_CACHE.get(key)
    if cached is not None and cached[1] == stamp:
        return cached[0]
    writer = transformer.TransformerWriter.load(path)
    if writer is None:
        return None
    _WRITER_CACHE[key] = (writer, stamp)
    return writer


def train_writer(
    tenant_id: int | None,
    corpus: list[str],
    steps: int = 0,
    instructions: list[tuple[str, str]] | None = None,
    contexts: list[tuple[str, str, str]] | None = None,
    config: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Train the transformer from scratch on the tenant's own sentences.

    ``steps`` of 0 means the training is measured against the material:
    the passes it has to make decide how long it trains.

    The sentences teach the model the trainer's words, the instructions teach
    it to answer a request, and the grounded pairs teach it to answer from the
    material it was given. ``progress`` is told how far the training has
    come, so a training in its own process can be watched from the page.
    """
    from app.services import transformer
    from app.services.transformer.settings import TrainingSettings

    writer = transformer.TransformerWriter(config=config)
    report = writer.fit(
        corpus,
        instructions=instructions or [],
        contexts=contexts or [],
        settings=TrainingSettings(steps=steps),
        progress=progress,
    )
    if report.get("trained"):
        path = writer_path(tenant_id)
        writer.save(path)
        _WRITER_CACHE[_tenant_key(tenant_id)] = (writer, path.stat().st_mtime)
    return report


TRAINING_TIMEOUT_SECONDS = 30 * 60


def training_path(tenant_id: int | None) -> Path:
    """The marker that says this tenant's brain is training."""
    return brain_dir() / f"training-{_tenant_key(tenant_id)}.json"


def mark_training(tenant_id: int | None, value: bool) -> None:
    """Note that the brain is training, so the page can say so.

    The note is written next to the model, so a training that runs in another
    process is seen as well.
    """
    path = training_path(tenant_id)
    if value:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"started": time.time()}), encoding="utf-8")
        return
    path.unlink(missing_ok=True)


def clear_training(tenant_id: int | None = None) -> None:
    if tenant_id is None:
        for path in brain_dir().glob("training-*.json"):
            path.unlink(missing_ok=True)
        return
    training_path(tenant_id).unlink(missing_ok=True)


def is_training(tenant_id: int | None) -> bool:
    """Whether this tenant's brain is being trained right now.

    The marker on disk is the truth: the training runs in its own process, so
    the flag cannot be kept in memory alone.
    """
    path = training_path(tenant_id)
    if not path.exists():
        return False
    try:
        started = float(json.loads(path.read_text(encoding="utf-8")).get("started", 0))
    except (OSError, ValueError, TypeError):
        started = 0
    if started and time.time() - started > TRAINING_TIMEOUT_SECONDS:
        # A training that was interrupted leaves its marker behind.
        path.unlink(missing_ok=True)
        return False
    return True


def writer_state(tenant_id: int | None) -> dict[str, Any]:
    """What the writer knows about itself, for the brain page."""
    writer = load_writer(tenant_id)
    if writer is None:
        return {"trained": False}
    return {
        "trained": True,
        "loss": writer.loss,
        "steps": writer.steps,
        "vocabulary": len(writer.vocabulary),
        "parameters": sum(parameter.numel() for parameter in writer.model.parameters())
        if writer.model
        else 0,
        "device": transformer_device(),
    }


def transformer_device() -> str:
    from app.services import transformer

    return transformer.device_name()


def write_with_transformer(tenant_id: int | None, prompt: str, max_tokens: int = 24) -> str | None:
    """Let the trained transformer write what comes after the prompt."""
    from app.services import transformer
    from app.services.transformer.settings import WritingSettings

    writer = load_writer(tenant_id)
    if writer is None:
        return None
    return writer.generate(prompt, settings=WritingSettings(max_words=max_tokens))


def feedback_path(tenant_id: int | None) -> Path:
    return brain_dir() / f"feedback-{_tenant_key(tenant_id)}.json"


def load_feedback(tenant_id: int | None) -> list[dict[str, Any]]:
    path = feedback_path(tenant_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_feedback(tenant_id: int | None, entries: list[dict[str, Any]]) -> None:
    path = feedback_path(tenant_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def record_feedback(
    tenant_id: int | None,
    question: str,
    answer: str,
    correct: bool,
    correction: str = "",
    learned: bool | None = None,
) -> dict[str, Any]:
    """Remember whether an answer was right, and what the right answer was.

    ``learned`` is set once the correction has been re-asked and checked, so
    the trainer can see that Luke really kept it.
    """
    with _LOCK:
        entries = load_feedback(tenant_id)
        entries.append(
            {
                "question": (question or "").strip()[:600],
                "answer": (answer or "").strip()[:1200],
                "correct": bool(correct),
                "correction": (correction or "").strip()[:1200],
                "learned": learned,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        save_feedback(tenant_id, entries)
    return {
        "entries": len(entries),
        "correct": sum(1 for entry in entries if entry.get("correct")),
        "learned_corrections": learned_corrections(tenant_id),
    }


def learned_corrections(tenant_id: int | None) -> int:
    """How many corrections Luke has answered correctly after being taught."""
    return sum(1 for entry in load_feedback(tenant_id) if entry.get("learned"))


def mark_correction_learned(
    tenant_id: int | None, question: str, correction: str
) -> bool:
    """Mark the correction as kept, when the assistant now answers it right.

    The entry that was kept is the last one for this question and this
    correction, so a correction that is taught twice is not counted twice.
    """
    with _LOCK:
        entries = load_feedback(tenant_id)
        for entry in reversed(entries):
            if entry.get("question") == question and entry.get("correction") == correction:
                entry["learned"] = True
                save_feedback(tenant_id, entries)
                return True
    return False


def verify_correction(tenant_id: int | None, question: str, correction: str) -> dict[str, Any]:
    """Ask the corrected question again and check that Luke keeps the correction."""
    answer = ""
    try:
        hits = vector_store.search_faq(question, n_results=1, tenant_id=tenant_id)
        if hits and hits[0].get("distance") is not None:
            answer = hits[0].get("answer", "")
    except Exception:
        answer = ""
    return {
        "learned": _answer_matches(answer, correction),
        "answer": answer,
    }


def emotion_of(text: str) -> str:
    """The feeling a piece of text carries, using the taught emotion words."""
    words = set(_tokenize(text))
    for emotion, markers in _EMOTION_WORDS.items():
        if words & set(markers):
            return emotion
    return "neutral"


def emotion_training_sentences(
    sentences: list[str], lessons: list[dict] | None = None
) -> tuple[list[str], list[str]]:
    """Label the taught sentences and lessons by the feeling they express."""
    labelled: list[str] = []
    labels: list[str] = []
    for sentence in sentences:
        emotion = emotion_of(sentence)
        if emotion == "neutral" and len(sentence.split()) > 3:
            continue
        labelled.append(sentence)
        labels.append(emotion)
    for lesson in lessons or []:
        for text in (lesson.get("question", ""), lesson.get("answer", "")):
            if not text:
                continue
            emotion = emotion_of(text)
            if emotion == "neutral":
                continue
            labelled.append(text)
            labels.append(emotion)
    return labelled, labels


def _embedding_features(texts: list[str]) -> list[list[float]]:
    """The embedding of every text, in the space the collections are stored in."""
    if not texts:
        return []
    try:
        return embeddings.embed_texts(texts)
    except Exception:
        return []


def train(
    tenant_id: int | None,
    materials: TrainingSet,
    config: dict[str, Any] | None = None,
    steps: int = 0,
) -> dict[str, Any]:
    """Train the writer and the heads from the tenant's material and feedback.

    ``steps`` of 0 means the writer trains as long as the material asks for
    (see ``train_writer``). The transformer is the writer, so it is trained
    again on everything the tenant has taught: every lesson, every approved
    answer, every correction and every document is in the material, and the
    vocabulary grows with it. The questions are turned into requests as well,
    so the writer learns to answer a request and not only to be looked up.
    """
    with _LOCK:
        feedback = load_feedback(tenant_id)
    corrections = [entry["correction"] for entry in feedback if entry.get("correction")]
    training_sentences = [sentence for sentence in materials.sentences if sentence]

    # The transformer is the writer now, so it is trained again on everything
    # the tenant has taught: the vocabulary grows with every lesson. The
    # checklist is written beside the model, so the page can watch it move.
    training_progress.start(tenant_id)
    mark_training(tenant_id, True)
    try:
        writer_report = train_writer(
            tenant_id,
            training_sentences,
            instructions=list(materials.instructions),
            contexts=list(materials.contexts),
            config=config,
            steps=steps,
            progress=_follow_writer_steps(tenant_id),
        )
        training_progress.stage(tenant_id, "writer", training_progress.DONE)
        emotion_examples = _train_emotion_head(tenant_id, training_sentences)
        training_progress.stage(tenant_id, "emotion", training_progress.DONE)
        intent_examples = _train_intent_head(tenant_id, materials)
        training_progress.stage(tenant_id, "intent", training_progress.DONE)
        if feedback:
            _train_feedback_heads(tenant_id, feedback)
        training_progress.stage(tenant_id, "feedback", training_progress.DONE)
    finally:
        clear_training(tenant_id)
        training_progress.finish(tenant_id)
    # The trainer asked for this run, so the instruction measurement is worth
    # a fresh one: the model it measures has just changed.
    try:
        remeasure_instruction(tenant_id)
    except (OSError, ValueError, RuntimeError) as error:
        LOGGER.warning("The instruction test could not rerun after training: %s", error)
    return {
        "writer": writer_report,
        "emotion_examples": emotion_examples,
        "intent_examples": intent_examples,
        "feedback_entries": len(feedback),
        "corrections": len(corrections),
    }


def _follow_writer_steps(tenant_id: int | None) -> Callable[[int, int], None]:
    """The steps of the writer, noted in the checklist as they happen."""
    started = time.time()

    def follow(done: int, total: int) -> None:
        elapsed = max(time.time() - started, 0.001)
        rate = max(done, 1) / elapsed
        remaining = max(total - done, 0)
        training_progress.steps(tenant_id, done, total, eta_seconds=remaining / rate)

    return follow


def _train_intent_head(tenant_id: int | None, materials: TrainingSet) -> int:
    """Teach the intent head from the tenant's own lessons.

    The head reads the kind of a question from its meaning, the same way the
    emotion head reads the feeling in a sentence.
    """
    from app.services.reasoning import intent as intent_service

    cards = [
        material.as_card() for material in materials.materials if material.question.strip()
    ]
    return intent_service.train(tenant_id, cards)


def _train_emotion_head(tenant_id: int | None, training_sentences: list[str]) -> int:
    """Teach the emotion head from the tenant's own sentences."""
    emotion_labels = ["neutral", *sorted(_EMOTION_WORDS)]
    head = LinearHead(emotion_labels)
    sentences, emotions = emotion_training_sentences(training_sentences, None)
    features = _embedding_features(sentences)
    targets = [emotion_labels.index(emotion) for emotion in emotions]
    if features and len(features) == len(targets) and head.fit(features, targets):
        head.save(brain_path(tenant_id, "emotion"))
    return len(sentences)


def _train_feedback_heads(tenant_id: int | None, feedback: list[dict[str, Any]]) -> None:
    """Teach the heads that read a question and judge the answer."""
    clarify = LinearHead(["answer", "ask_back"])
    correct = LinearHead(["correct", "wrong"])
    questions = [entry["question"] for entry in feedback if entry.get("question")]
    features = _embedding_features(questions)
    if not features or len(features) != len(feedback):
        return
    labels = [0 if entry.get("correct") else 1 for entry in feedback]
    clarify.fit(features, labels)
    correct.fit(features, labels)
    clarify.save(brain_path(tenant_id, "clarify"))
    correct.save(brain_path(tenant_id, "correct"))


def state(tenant_id: int | None) -> dict[str, Any]:
    """What the brain currently knows about itself.

    The vocabulary here is the transformer's: it is the one live token list,
    so the numbers and the ids always mean the same words.
    """
    from app.services import vocabulary as vocabulary_service

    feedback = load_feedback(tenant_id)
    emotion = LinearHead.load(brain_path(tenant_id, "emotion"))
    clarify = LinearHead.load(brain_path(tenant_id, "clarify"))
    correct = LinearHead.load(brain_path(tenant_id, "correct"))
    intent = LinearHead.load(brain_path(tenant_id, "intent"))
    transformer = load_writer(tenant_id)
    canonical = vocabulary_service.canonical(tenant_id)
    return {
        "trained": transformer is not None,
        "vocabulary": len(canonical),
        "transformer": writer_state(tenant_id),
        "transformer_ready": transformer is not None,
        "training": is_training(tenant_id),
        "progress": training_progress.read(tenant_id),
        "emotions": emotion.labels if emotion else [],
        "emotion_ready": bool(emotion and emotion.fitted),
        "clarify_ready": bool(clarify and clarify.fitted),
        "confidence_ready": bool(correct and correct.fitted),
        "intent_ready": bool(intent and intent.fitted),
        "intents": intent.labels if intent else [],
        "feedback_entries": len(feedback),
        "correct_answers": sum(1 for entry in feedback if entry.get("correct")),
        "wrong_answers": sum(1 for entry in feedback if not entry.get("correct")),
        "corrections": sum(1 for entry in feedback if entry.get("correction")),
        "learned_corrections": sum(1 for entry in feedback if entry.get("learned")),
    }


def write_sentence(tenant_id: int | None, seed_word: str = "") -> str | None:
    # The transformer is the only writer. The old numpy writer is retired.
    return write_with_transformer(tenant_id, seed_word)


def feel(tenant_id: int | None, text: str) -> dict[str, Any]:
    """The emotion in a message, from the trained head (or the taught words)."""
    head = LinearHead.load(brain_path(tenant_id, "emotion"))
    if head is not None:
        features = _embedding_features([text])
        if features:
            label, confidence, scores = head.predict(features)
            if confidence >= 0.45:
                return {
                    "emotion": label,
                    "confidence": confidence,
                    "scores": scores,
                    "source": "trained",
                }
    return {"emotion": emotion_of(text), "confidence": 0.4, "scores": {}, "source": "words"}


def should_ask_back(tenant_id: int | None, question: str) -> dict[str, Any]:
    """Whether the assistant should ask a question back instead of answering."""
    head = LinearHead.load(brain_path(tenant_id, "clarify"))
    if head is not None:
        features = _embedding_features([question])
        if features:
            label, confidence, scores = head.predict(features)
            if label == "ask_back" and confidence >= 0.6:
                return {"ask_back": True, "confidence": confidence, "source": "trained"}
            return {"ask_back": False, "confidence": confidence, "source": "trained"}
    return {"ask_back": False, "confidence": 0.0, "source": "untrained"}


def answer_confidence(tenant_id: int | None, question: str) -> dict[str, Any]:
    """How likely the answer is to be right, from the trainer's corrections."""
    head = LinearHead.load(brain_path(tenant_id, "correct"))
    if head is not None:
        features = _embedding_features([question])
        if features:
            label, confidence, scores = head.predict(features)
            return {
                "likely_correct": label == "correct",
                "confidence": confidence,
                "source": "trained",
            }
    return {"likely_correct": True, "confidence": 0.5, "source": "untrained"}
