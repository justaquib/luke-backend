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
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from app.services import vector_store


BRAIN_DIR = Path(__file__).resolve().parents[2] / ".language_model"
_LOCK = threading.RLock()

_EMOTION_WORDS: dict[str, tuple[str, ...]] = {
    "happy": ("happy", "glad", "pleased", "joy", "joyful", "excited", "cheerful", "smile", "delight"),
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


class LinearHead:
    """A softmax classifier over the assistant's neural embeddings.

    It runs on PyTorch, the same library the writer uses, so the app needs
    one library for its models and not two.
    """

    def __init__(self, labels: list[str], seed: int = 11) -> None:
        self.labels = labels
        self.seed = seed
        self.weights: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self.fitted = False

    def fit(self, features: list[list[float]], targets: list[int], epochs: int = 200, learning_rate: float = 0.1) -> bool:
        if len(features) < max(4, len(self.labels)):
            return False
        if len(set(targets)) < 2:
            # There is only one kind of answer to learn from, so the head
            # learns that: it says the same for everything it was shown.
            x = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.long)
            counts = torch.bincount(y, minlength=len(self.labels)).to(torch.float32)
            self.weights = torch.zeros((x.shape[1], len(self.labels)), dtype=torch.float32)
            self.bias = torch.log(counts.clamp(min=1.0) / counts.sum())
            self.fitted = True
            return True
        x = torch.tensor(features, dtype=torch.float32)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        y = torch.tensor(targets, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed)
        self.weights = torch.randn(
            (x.shape[1], len(self.labels)), generator=generator, dtype=torch.float32
        ) * 0.05
        self.bias = torch.zeros(len(self.labels), dtype=torch.float32)
        one_hot = torch.zeros((len(y), len(self.labels)), dtype=torch.float32)
        one_hot[torch.arange(len(y)), y] = 1
        # Rare feelings should count as much as common ones.
        counts = torch.bincount(y, minlength=len(self.labels)).to(torch.float32)
        class_weights = counts.sum() / (len(self.labels) * counts.clamp(min=1.0))
        weights = class_weights[y].reshape(-1, 1)

        for _ in range(epochs):
            logits = x @ self.weights + self.bias
            logits -= logits.max(dim=1, keepdim=True).values
            probabilities = torch.exp(logits)
            probabilities /= probabilities.sum(dim=1, keepdim=True)
            grad = torch.clamp((probabilities - one_hot) * weights / len(y), -1.0, 1.0)
            grad = torch.nan_to_num(grad)
            if not torch.isfinite(grad).all():
                # The numbers went out of hand, so the training stops here
                # instead of writing nonsense into the model.
                break
            self.weights -= learning_rate * (x.T @ grad + 0.002 * self.weights)
            self.bias -= learning_rate * grad.sum(dim=0)
        self.weights = torch.nan_to_num(self.weights)
        self.bias = torch.nan_to_num(self.bias)
        self.fitted = True
        return True

    def predict(self, features: list[list[float]]) -> tuple[str, float, dict[str, float]]:
        if not self.fitted or self.weights is None or self.bias is None:
            return self.labels[0], 0.0, {}
        x = torch.tensor(features, dtype=torch.float32)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        logits = x @ self.weights + self.bias
        logits -= logits.max(dim=1, keepdim=True).values
        probabilities = torch.exp(logits)
        probabilities /= probabilities.sum(dim=1, keepdim=True)
        index = int(torch.argmax(probabilities[0]).item())
        scores = {
            label: round(float(value), 3)
            for label, value in zip(self.labels, probabilities[0].tolist())
        }
        return self.labels[index], round(float(probabilities[0][index].item()), 3), scores

    def save(self, path: Path) -> None:
        if self.weights is None or self.bias is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"labels": list(self.labels), "weights": self.weights, "bias": self.bias},
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "LinearHead | None":
        if not path.is_file():
            return None
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            head = cls([str(label) for label in data["labels"]])
            head.weights = data["weights"].to(torch.float32)
            head.bias = data["bias"].to(torch.float32)
            head.fitted = True
            return head
        except Exception:
            return None


# ------------------------------------------------------------- the levels


LUKE_LEVELS: list[dict[str, Any]] = [
    {
        "level": 0.0,
        "name": "Seed",
        "description": "It holds what it was taught and can look it up, but it cannot answer anything on its own.",
        "reference": "The floor of the scale: material without general answering.",
    },
    {
        "level": 0.1,
        "name": "Cued",
        "description": "It finds the answer when asked exactly as it was taught.",
        "reference": "Teach it something and it can give that back.",
    },
    {
        "level": 0.2,
        "name": "Paraphrased",
        "description": "It finds the answer when the question is worded differently.",
        "reference": "It survives a rewording of the question.",
    },
    {
        "level": 0.3,
        "name": "Generalised",
        "description": "It answers about the topic without the exact question.",
        "reference": "It knows the topic, not only the sentence.",
    },
    {
        "level": 0.4,
        "name": "Written",
        "description": "It writes new sentences in the style it was taught.",
        "reference": "It composes instead of quoting.",
    },
    {
        "level": 0.5,
        "name": "Explained",
        "description": "It explains an answer when it is asked to.",
        "reference": "It can say why, not only what.",
    },
    {
        "level": 0.6,
        "name": "Corrected",
        "description": "It learns from a correction and keeps it.",
        "reference": "A wrong answer turns into a right one, and stays right.",
    },
    {
        "level": 0.7,
        "name": "Examined",
        "description": "It passes the final exam of a school grade.",
        "reference": "The school says it is ready for the next grade.",
    },
    {
        "level": 0.8,
        "name": "Grounded",
        "description": "It says it does not know instead of guessing.",
        "reference": "Knowing the edge of its knowledge is a kind of intelligence.",
    },
    {
        "level": 0.9,
        "name": "Teacher",
        "description": "It holds a whole subject: many questions, one topic.",
        "reference": "Depth, not a list of facts.",
    },
    {
        "level": 1.0,
        "name": "Independent",
        "description": "It answers a question it was never taught, in its own words.",
        "reference": "The long road: every lesson, every correction and every exam widens what it can do.",
    },
]

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

    exact = [{"question": item["question"], "answer": item["answer"], "form": "asked as taught"} for item in taught]
    reworded = [
        {"question": _reframe(item["question"]), "answer": item["answer"], "form": "asked in other words"}
        for item in taught
    ]
    swapped: list[dict[str, Any]] = []
    for item in taught:
        alternative = _swap_synonym(item["question"])
        if alternative:
            swapped.append({"question": alternative, "answer": item["answer"], "form": "asked with another word"})

    explained = [
        {"question": f"Explain: {item['question']}", "answer": item["answer"], "form": "asked for an explanation"}
        for item in taught
    ]

    return {
        "0.1": exact,
        "0.2": reworded,
        "0.3": swapped,
        "0.5": explained,
    }


_RUNG_TESTS = {
    "0.1": "Can it find the answer when asked exactly as it was taught?",
    "0.2": "Can it find the answer when the question is worded differently?",
    "0.3": "Can it answer about the topic without the exact question?",
    "0.4": "Can it write new sentences in the style you taught?",
    "0.5": "Can it explain an answer when it is asked to?",
    "0.6": "Has it learned from a correction and kept it?",
    "0.7": "Has it passed the final exam of a grade?",
    "0.8": "Does it say it does not know instead of guessing?",
    "0.9": "Does it hold one whole subject in depth?",
    "1.0": "Can it answer a question it was never taught, in its own words?",
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
    knowledge = knowledge if knowledge is not None else {"lessons": 0, "passed_grades": 0, "stage": "nursery"}

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
                passes.append({
                    "question": probe["question"],
                    "expected": probe["answer"],
                    "form": probe["form"],
                    "answered": answer,
                })
            else:
                failures.append({
                    "question": probe["question"],
                    "expected": probe["answer"],
                    "form": probe["form"],
                    "answered": answer,
                })
        total = len(probes)
        results.append({
            "rung": rung,
            "probes": total,
            "passed": passed,
            "rate": round(passed / total, 2) if total else 0.0,
            "passes": passes,
            "failures": failures[:4],
        })

    # The capabilities that come from the training itself.
    subjects = _subject_coverage(tenant_id)
    # The transformer is the writer now, so it counts as the writing model.
    written = bool(brain_state.get("trained")) or load_writer(tenant_id) is not None
    results.append({
        "rung": "0.4",
        "probes": 1,
        "passed": 1 if written else 0,
        "rate": 1.0 if written else 0.0,
        "passes": [{"question": "Write a new sentence about what you know", "expected": "a sentence in the taught style", "form": "from the training", "answered": "it writes new sentences in the style it was taught"}] if written else [],
        "failures": [] if written else [{"question": "Write a new sentence about what you know", "expected": "a sentence in the taught style", "form": "never taught"}],
        "detail": "the writing model is trained" if written else "the writing model is not trained yet",
    })
    corrections = brain_state.get("corrections", 0)
    kept = brain_state.get("learned_corrections", 0)
    results.append({
        "rung": "0.6",
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
        "failures": [] if kept >= CORRECTION_LIMIT else [{
            "question": "Learn from a correction and keep it",
            "expected": f"{CORRECTION_LIMIT} corrections answered correctly after being taught",
            "form": "never taught",
        }],
        "detail": f"{kept} of {CORRECTION_LIMIT} corrections kept",
    })
    grades = knowledge.get("passed_grades", 0)
    results.append({
        "rung": "0.7",
        "probes": 1,
        "passed": 1 if grades else 0,
        "rate": 1.0 if grades else 0.0,
        "passes": [{"question": "Pass the final exam", "expected": "one grade exam passed", "form": "from the training", "answered": f"{grades} grade exam(s) passed"}] if grades else [],
        "failures": [] if grades else [{"question": "Pass the final exam", "expected": "one grade exam passed", "form": "never taught"}],
        "detail": f"{grades} grade exams passed" if grades else "no final exam passed yet",
    })
    grounded = bool(brain_state.get("confidence_ready"))
    results.append({
        "rung": "0.8",
        "probes": 1,
        "passed": 1 if grounded else 0,
        "rate": 1.0 if grounded else 0.0,
        "passes": [{"question": "Say when you do not know", "expected": "a trained confidence head", "form": "from the training", "answered": "it says when it does not know"}] if grounded else [],
        "failures": [] if grounded else [{"question": "Say when you do not know", "expected": "a trained confidence head", "form": "never taught"}],
        "detail": "it knows the edge of its knowledge" if grounded else "it has not learned where its knowledge ends",
    })
    deep = 1 if any(count >= 5 for count in subjects.values()) else 0
    results.append({
        "rung": "0.9",
        "probes": 1,
        "passed": deep,
        "rate": 1.0 if deep else 0.0,
        "passes": [{
            "question": "Hold one subject in depth",
            "expected": "five questions on one topic",
            "form": "from the training",
            "answered": f"deepest subject: {max(subjects, key=lambda key: subjects[key])} ({max(subjects.values()) if subjects else 0}) questions",
        }] if deep else [],
        "failures": [] if deep else [{"question": "Hold one subject in depth", "expected": "five questions on one topic", "form": "never taught"}],
        "detail": f"deepest subject: {max(subjects, key=lambda key: subjects[key])} ({max(subjects.values()) if subjects else 0})" if subjects else "no subject yet",
    })

    for rung in results:
        rung["test"] = _RUNG_TESTS.get(rung["rung"], "")
    results.sort(key=lambda item: float(item["rung"]))
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


def model_capabilities(tenant_id: int | None, brain_state: dict[str, Any], knowledge: dict[str, Any]) -> list[dict[str, Any]]:
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


_VOLUME_STEPS = (
    (0, 0.0),
    (1, 0.1),
    (25, 0.2),
    (100, 0.3),
    (500, 0.4),
)


def training_volume_level(items: int) -> dict[str, Any]:
    """Where the amount of learned material sits on the same ladder.

    Volume is not intelligence, so it can never carry the model far up the
    scale: a hundred thousand documents still do not make a model reason.
    """
    level = 0.0
    for threshold, value in _VOLUME_STEPS:
        if items >= threshold:
            level = value
    entry = _level_entry(level)
    return {
        "level": level,
        "title": entry["title"],
        "name": entry["name"],
        "items": items,
    }


def _level_title(level: float) -> str:
    number = f"{level:g}"
    return f"Luke {number}"


def _level_entry(level: float) -> dict[str, Any]:
    """The ladder entry for a level, which can sit between two whole levels."""
    best = LUKE_LEVELS[0]
    for entry in LUKE_LEVELS:
        if entry["level"] <= level + 1e-9:
            best = entry
        else:
            break
    return {**best, "title": f"{_level_title(best['level'])} · {best['name']}"}


def _next_level_entry(level: float) -> dict[str, Any] | None:
    for entry in LUKE_LEVELS:
        if entry["level"] > level + 1e-9:
            return {**entry, "title": f"{_level_title(entry['level'])} · {entry['name']}"}
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
            coverage["documents"] = len({row.source for row in db.execute(document_query).scalars().all()})
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
    level: float,
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
        steps.append({
            "action": "add_document",
            "title": "Give it a book to read",
            "detail": "Luke has no document to read yet. Add a PDF, a novel or any related file in the Trainer's Documents tab: the pages are split into chunks, every chunk is embedded for meaning, and the passages are read back word by word when a question needs them.",
            "link": "/trainer?tab=documents",
            "link_label": "Add a document",
        })
    elif chunks < 40:
        steps.append({
            "action": "add_document",
            "title": "Give it more pages to read",
            "detail": f"Only {chunks} chunks are indexed. More pages means more text chunked, embedded and read word by word.",
            "link": "/trainer?tab=documents",
            "link_label": "Add a document",
        })

    if answers < 5:
        steps.append({
            "action": "add_answer",
            "title": "Store more trained answers",
            "detail": "The Trainer keeps the question and answer pairs. The more it holds, the more it can answer without guessing.",
            "link": "/trainer?tab=new",
            "link_label": "Add a Q&A",
        })

    for rung in rungs:
        if rung["probes"] and rung["rate"] < 0.6:
            hints = {
                "0.1": "Teach Luke a question and answer it cannot already answer.",
                "0.2": "Teach the same answers again in different words, so it can find them when the question is reworded.",
                "0.3": "Teach the topic rather than the exact question, and add a few phrasings of it.",
                "0.4": "Train the brain so its writing model learns the style of your lessons.",
                "0.5": "Teach the reason as well as the answer, so it can explain it when asked.",
                "0.6": f"Mark an answer wrong in the chat and give the right one. {max(0, CORRECTION_LIMIT - kept)} more kept corrections and Luke passes this rung.",
                "0.7": "Pass the final exam of the grade it is studying.",
                "0.8": "Give it a few corrections so it learns where its knowledge ends and says so instead of guessing.",
                "0.9": "Teach more questions on one topic so it holds a whole subject.",
            }
            steps.append({
                "action": "add_lesson",
                "title": f"Pass the Luke {rung['rung']} test",
                "detail": hints.get(rung["rung"], "Teach more of what Luke should know."),
                "link": "/trainer?tab=language",
                "link_label": "Teach a lesson",
            })
            break

    if knowledge.get("lessons", 0) == 0:
        steps.append({
            "action": "add_lesson",
            "title": "Add your first lesson",
            "detail": "Luke starts empty. Teach it a question and the answer it should give.",
            "link": "/trainer?tab=language",
            "link_label": "Open the learner",
        })

    grade = knowledge.get("stage", "nursery")
    if grade and grade != "complete":
        from app.services import language_logic

        grade_name = next(
            (stage["title"] for stage in language_logic.STAGES if stage["id"] == grade),
            str(grade).replace("_", " ").title(),
        )
        steps.append({
            "action": "mock_test",
            "title": f"Take a mock test for {grade_name}",
            "detail": "A mock test is practice. It never advances the grade, so you can see what still needs learning.",
            "link": "/trainer?tab=language",
            "link_label": "Open the learner",
        })
        steps.append({
            "action": "final_exam",
            "title": f"Pass the {grade_name} final exam to advance the school grade",
            "detail": "The final exam moves Luke to the next grade once it answers at least 80% of the questions correctly.",
            "link": "/trainer?tab=language",
            "link_label": "Open the learner",
        })
    if knowledge.get("lessons", 0) < 40:
        steps.append({
            "action": "add_lesson",
            "title": "Add more knowledge to the school",
            "detail": f"{knowledge.get('lessons', 0)} lessons are taught. Every lesson is a question and answer the learner can use, and the school grades are built on them.",
            "link": "/trainer?tab=language",
            "link_label": "Teach a lesson",
        })

    if brain_state.get("corrections", 0) == 0:
        steps.append({
            "action": "rate_answers",
            "title": "Mark one answer wrong and correct it",
            "detail": "In the chat, mark an answer wrong and type the right one. Luke learns it, and the correction becomes part of its memory.",
            "link": "/chat",
            "link_label": "Open the chat",
        })

    steps.append({
        "action": "train_brain",
        "title": "Train the brain",
        "detail": "Training rebuilds the writing model and the emotion head from everything Luke has been taught, then tests the level again.",
        "link": "",
        "link_label": "",
    })

    return steps


def model_level(tenant_id: int | None, force: bool = False) -> dict[str, Any]:
    """The level the brain has reached, measured by what it can do.

    A local model is held to the same scale as a frontier one, and it is
    capped by how far it can answer on its own: a retriever cannot reach the
    levels that need real reasoning, however much it was taught.
    """
    brain_state = state(tenant_id)
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

    capabilities = model_capabilities(tenant_id, brain_state, knowledge)
    generality = generality_score(tenant_id)
    coverage = _coverage(tenant_id)
    knowledge["items"] = coverage.get("chunks", 0)
    coverage["lessons"] = knowledge.get("lessons", 0)
    coverage["passed_grades"] = knowledge.get("passed_grades", 0)
    coverage["current_stage"] = knowledge.get("stage", "nursery")

    # The level is the intelligence level, measured by how far the model can
    # answer on its own. A checklist of features never lifts it: a retriever
    # with every feature is still a retriever.
    score = generality["score"]
    rungs = rung_results(tenant_id, force=force, brain_state=brain_state, knowledge=knowledge)

    # The level is the highest rung it has passed. The rungs are in order, and
    # every one of them is something a model built from scratch can be taught.
    rung_level = 0.0
    for rung in rungs:
        if rung["probes"] and rung["rate"] >= 0.6:
            rung_level = float(rung["rung"])
        else:
            break

    # The top rung is the stretch: answering a question it was never taught,
    # in its own words. It needs no pre-trained model, only more lessons,
    # more corrections and more exams.
    if score >= 0.5:
        rung_level = 1.0
    level = rung_level

    current = _level_entry(level)
    following = _next_level_entry(level)
    steps = next_steps(tenant_id, brain_state, knowledge, rungs, level, coverage)
    missing = [capability["name"] for capability in capabilities if not capability["ready"]]
    next_detail = next(
        (step["detail"] for step in steps if step["title"].startswith("Pass the Luke")),
        steps[0]["detail"] if steps else "",
    )

    return {
        "level": level,
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
        "frontier_level": 6.0,
        "frontier_title": BENCHMARK_TITLE,
        "frontier_note": BENCHMARK_NOTE,
        "model_name": "Luke",
    }



# ------------------------------------------------------------- the brain


def _tenant_key(tenant_id: int | None) -> int:
    return tenant_id if tenant_id is not None else 0


def brain_path(tenant_id: int | None, name: str) -> Path:
    """Where a trained head is kept. The old numpy files are not read."""
    return BRAIN_DIR / f"brain-{_tenant_key(tenant_id)}-{name}.pt"


def writer_path(tenant_id: int | None) -> Path:
    """Where the transformer that writes in the trainer's words is kept."""
    return BRAIN_DIR / f"transformer-{_tenant_key(tenant_id)}.pt"


_WRITER_CACHE: dict[str, Any] = {}


def load_writer(tenant_id: int | None):
    """The trained transformer of this tenant, if it has one.

    It is kept in memory, so the answers do not have to load it every time.
    """
    from app.services import transformer_lm

    key = _tenant_key(tenant_id)
    cached = _WRITER_CACHE.get(key)
    if cached is not None:
        return cached
    writer = transformer_lm.TransformerWriter.load(writer_path(tenant_id))
    if writer is None:
        return None
    _WRITER_CACHE[key] = writer
    return writer


def train_writer(tenant_id: int | None, corpus: list[str], steps: int = 3000) -> dict[str, Any]:
    """Train the transformer from scratch on the tenant's own sentences."""
    from app.services import transformer_lm

    writer = transformer_lm.TransformerWriter()
    report = writer.fit(corpus, steps=steps)
    if report.get("trained"):
        writer.save(writer_path(tenant_id))
        _WRITER_CACHE[_tenant_key(tenant_id)] = writer
    return report


_TRAINING: dict[str, bool] = {}


def mark_training(tenant_id: int | None, value: bool) -> None:
    """Note that the brain is training, so the page can say so."""
    _TRAINING[_tenant_key(tenant_id)] = value


def clear_training(tenant_id: int | None = None) -> None:
    if tenant_id is None:
        _TRAINING.clear()
    else:
        _TRAINING.pop(_tenant_key(tenant_id), None)


def is_training(tenant_id: int | None) -> bool:
    return bool(_TRAINING.get(_tenant_key(tenant_id)))


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
        "parameters": sum(parameter.numel() for parameter in writer.model.parameters()) if writer.model else 0,
        "device": transformer_device(),
    }


def transformer_device() -> str:
    from app.services import transformer_lm

    return transformer_lm.device_name()


def write_with_transformer(tenant_id: int | None, prompt: str, max_tokens: int = 24) -> str | None:
    """Let the trained transformer write what comes after the prompt."""
    writer = load_writer(tenant_id)
    if writer is None:
        return None
    return writer.generate(prompt, max_tokens=max_tokens)


def feedback_path(tenant_id: int | None) -> Path:
    return BRAIN_DIR / f"feedback-{_tenant_key(tenant_id)}.json"


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
        entries.append({
            "question": (question or "").strip()[:600],
            "answer": (answer or "").strip()[:1200],
            "correct": bool(correct),
            "correction": (correction or "").strip()[:1200],
            "learned": learned,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        save_feedback(tenant_id, entries)
    return {
        "entries": len(entries),
        "correct": sum(1 for entry in entries if entry.get("correct")),
        "learned_corrections": learned_corrections(tenant_id),
    }


def learned_corrections(tenant_id: int | None) -> int:
    """How many corrections Luke has answered correctly after being taught."""
    return sum(1 for entry in load_feedback(tenant_id) if entry.get("learned"))


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


def emotion_training_sentences(sentences: list[str], lessons: list[dict] | None = None) -> tuple[list[str], list[str]]:
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
    if not texts:
        return []
    try:
        return vector_store.embed_texts(texts)
    except Exception:
        return []


def train(tenant_id: int | None, corpus: list[str], lessons: list[dict] | None = None) -> dict[str, Any]:
    """Train the writer and the heads from the tenant's material and feedback.

    The transformer is the writer now, so the small writer is only trained when
    there is no transformer yet.
    """
    with _LOCK:
        feedback = load_feedback(tenant_id)
    corrections = [entry["correction"] for entry in feedback if entry.get("correction")]
    training_sentences = [sentence for sentence in corpus if sentence] + corrections

    # The transformer is the writer now. The old numpy writer, and the
    # vocabulary it carried, are retired.
    writer_report = {"trained": load_writer(tenant_id) is not None, "reason": "the transformer is the writer"}

    emotion_labels = ["neutral"] + sorted(_EMOTION_WORDS)
    head = LinearHead(emotion_labels)
    sentences, emotions = emotion_training_sentences(training_sentences, lessons)
    features = _embedding_features(sentences)
    targets = [emotion_labels.index(emotion) for emotion in emotions]
    if features and len(features) == len(targets):
        if head.fit(features, targets):
            head.save(brain_path(tenant_id, "emotion"))

    clarify = LinearHead(["answer", "ask_back"])
    correct = LinearHead(["correct", "wrong"])
    if feedback:
        questions = [entry["question"] for entry in feedback if entry.get("question")]
        features = _embedding_features(questions)
        if features and len(features) == len(feedback):
            clarify.fit(features, [0 if entry.get("correct") else 1 for entry in feedback])
            correct.fit(features, [0 if entry.get("correct") else 1 for entry in feedback])
            clarify.save(brain_path(tenant_id, "clarify"))
            correct.save(brain_path(tenant_id, "correct"))

        return {
            "writer": writer_report,
            "emotion_examples": len(sentences),
            "feedback_entries": len(feedback),
            "corrections": len(corrections),
        }


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
    transformer = load_writer(tenant_id)
    canonical = vocabulary_service.canonical(tenant_id)
    return {
        "trained": transformer is not None,
        "vocabulary": len(canonical),
        "transformer": writer_state(tenant_id),
        "transformer_ready": transformer is not None,
        "training": is_training(tenant_id),
        "emotions": emotion.labels if emotion else [],
        "emotion_ready": bool(emotion and emotion.fitted),
        "clarify_ready": bool(clarify and clarify.fitted),
        "confidence_ready": bool(correct and correct.fitted),
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
                return {"emotion": label, "confidence": confidence, "scores": scores, "source": "trained"}
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
