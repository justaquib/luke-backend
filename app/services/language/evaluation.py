"""How well the model answers requests it was never worded for.

A model that only looks things up answers the request it was taught and
nothing else. This measures the other half: the requests the model was never
trained on, phrased in words the trainer never used, about material the
trainer did teach.

Three sets of probes are used, and each of them keeps something out of the
training:

* the **cross-form** probes ask for a lesson in a wording the model was
  never trained on,
* the **cross-topic** probes ask about a whole subject the model was never
  trained on,
* the **adversarial** probes ask about things that were never taught, and
  the model has to refuse them instead of answering with something else.

The score is the honest one. The requests used here are kept out of the
training, so a model that only memorises scores nothing.
"""

from __future__ import annotations

from typing import Any

from app.services.language import instructions as instruction_service
from app.services.language import requests as request_service
from app.services.language.topics import topic_words

# The number of held-out requests the model is measured on.
EVALUATION_SIZE = 12

# The share of the words that must match for an answer to count as right.
MATCH_SHARE = 0.6

# How many probes the adversarial set holds.
ADVERSARIAL_SIZE = 8

# The small model the cross-topic probe is measured with.
PROBE_MODEL = {"d_model": 64, "layers": 2, "heads": 4, "context": 128, "dropout": 0.1}

# The steps the cross-topic probe is trained for.
PROBE_STEPS = 400

# The answer the model gives when it was taught nothing about the request.
REFUSAL = "I do not know anything about that."


def _words(text: str) -> set[str]:
    """The words of a text, in lower case."""
    return set(topic_words(text))


def _matches(answer: str, expected: str) -> bool:
    """Whether the answer says what the taught answer says."""
    expected_words = _words(expected)
    if not expected_words:
        return False
    answer_words = _words(answer)
    return len(expected_words & answer_words) / len(expected_words) >= MATCH_SHARE


def _probes(
    tenant_id: int | None,
    cards: list[dict[str, Any]],
    items: list[Any],
) -> list[dict[str, Any]]:
    """The score of the model on the requests it was given."""
    probes: list[dict[str, Any]] = []
    passed = 0
    for item in items:
        request = request_service.parse_request(item.instruction)
        if request is None:
            continue
        card = request_service.find_material(request, cards)
        if card is None:
            continue
        # The answer comes from the same path the chat uses, so the score is
        # the score of the assistant and not of the model alone.
        result = request_service.answer_request(item.instruction, tenant_id, cards)
        answer = result["answer"] if result else ""
        correct = _matches(answer, item.response)
        passed += 1 if correct else 0
        probes.append(
            {
                "question": item.instruction,
                "expected": item.response,
                "answered": answer,
                "correct": correct,
                "kind": item.kind,
            }
        )
    return [{"passed": passed, "total": len(probes), "probes": probes}]


def _cross_form(tenant_id: int | None, cards: list[dict[str, Any]]) -> dict[str, Any]:
    """The score on the requests the model was never worded for."""
    held_out = instruction_service.held_out_requests(cards, limit=EVALUATION_SIZE)
    result = _probes(tenant_id, cards, held_out)
    return _score(result[0])


def _score(result: dict[str, Any]) -> dict[str, Any]:
    """The score of a set of probes."""
    total = result["total"]
    passed = result["passed"]
    return {
        "score": round(passed / total, 2) if total else 0.0,
        "passed": passed,
        "total": total,
        "probes": result["probes"],
    }


def _adversarial(cards: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The requests about things that were never taught, and the negation."""
    from app.services.language.training_materials import _negative_examples

    index = instruction_service.topic_index(cards)
    negatives = _negative_examples(cards, index)
    topics = next(iter(index.values()), [])
    negated = [
        (f"Write a poem that is not about {topic}.", REFUSAL)
        for topic in topics[:ADVERSARIAL_SIZE]
        if topic
    ]
    return (negatives + negated)[:ADVERSARIAL_SIZE]


def _held_out_subject(cards: list[dict[str, Any]]) -> str:
    """The subject with the most lessons, which is the one kept out."""
    counts: dict[str, int] = {}
    for card in cards:
        subject = str(card.get("subject") or "General").strip() or "General"
        counts[subject] = counts.get(subject, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: item[1])[0]


def _cross_topic(cards: list[dict[str, Any]]) -> dict[str, Any]:
    """The score of a model that never saw one of the subjects.

    A model is trained on every subject but one, and is then asked about the
    one it did not see. The score says what it learned from the rest.
    """
    subject = _held_out_subject(cards)
    if not subject:
        return {"score": 0.0, "passed": 0, "total": 0, "subject": "", "probes": []}
    kept = [card for card in cards if str(card.get("subject") or "").strip() != subject]
    held = [card for card in cards if str(card.get("subject") or "").strip() == subject]
    if not kept or not held:
        return {"score": 0.0, "passed": 0, "total": 0, "subject": subject, "probes": []}
    writer = _probe_writer(kept)
    if writer is None:
        return {"score": 0.0, "passed": 0, "total": 0, "subject": subject, "probes": []}
    index = instruction_service.topic_index(held)
    items = [
        item
        for card in held
        for item in instruction_service.held_out_instructions_for(
            card, index.get(str(card.get("id") or card.get("question") or ""), [])
        )
    ][:EVALUATION_SIZE]
    result = _probes(None, held, items)
    return {"subject": subject, **_score(result[0])}


def _probe_writer(cards: list[dict[str, Any]]) -> Any:
    """A small model trained on the cards it is given."""
    from app.services import transformer
    from app.services.language import training_materials as materials_service
    from app.services.transformer.settings import TrainingSettings

    materials = [
        materials_service.Material(
            materials_service.LESSON,
            str(card.get("question") or ""),
            str(card.get("answer") or ""),
            str(card.get("subject") or ""),
        )
        for card in cards
    ]
    training = materials_service.training_set(materials)
    writer = transformer.TransformerWriter(config=PROBE_MODEL)
    report = writer.fit(
        training.sentences,
        instructions=training.instructions,
        contexts=training.contexts,
        settings=TrainingSettings(steps=PROBE_STEPS),
    )
    return writer if report.get("trained") else None


def evaluate(tenant_id: int | None, cards: list[dict[str, Any]], writer: Any) -> dict[str, Any]:
    """The score of the model on the requests it was never trained on.

    Only the requests are measured. The held-out questions are answered by
    looking them up, which is what the paraphrasing rung measures already.
    """
    if writer is None:
        return {"score": 0.0, "passed": 0, "total": 0, "probes": []}
    cross_form = _cross_form(tenant_id, cards)
    adversarial = _adversarial(cards)
    return {
        **cross_form,
        "cross_form": cross_form,
        "adversarial": _refusals(writer, adversarial),
        "cross_topic": _cross_topic(cards),
    }


def _refusals(writer: Any, requests: list[tuple[str, str]]) -> dict[str, Any]:
    """How often the model refuses what it was never taught."""
    probes: list[dict[str, Any]] = []
    passed = 0
    for request, expected in requests:
        answer = writer.respond(request) or ""
        refused = REFUSAL.strip().casefold() in answer.casefold() or not answer.strip()
        passed += 1 if refused else 0
        probes.append(
            {
                "question": request,
                "expected": expected,
                "answered": answer,
                "correct": refused,
            }
        )
    return {
        "score": round(passed / len(requests), 2) if requests else 0.0,
        "passed": passed,
        "total": len(requests),
        "probes": probes,
    }
