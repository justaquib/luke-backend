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
training, so a model that only memorises scores nothing. The adversarial
score is the refusal the user gets from the product: the request is read, the
material is looked up, and nothing is answered when the topic was never
taught. The writer's own refusal is counted beside it, because the boundary
is what keeps the chat honest and the model does not learn that on its own.
"""

from __future__ import annotations

from typing import Any

from app.services.language import instructions as instruction_service
from app.services.language import requests as request_service
from app.services.language.negatives import REFUSAL, build_held_out_negatives
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
    writer: Any = None,
) -> list[dict[str, Any]]:
    """The score of the model on the requests it was given.

    The writer is threaded through the same path the chat uses, so the score
    is the score of the model writing an answer and not of the lookup alone.
    Every probe says whether the answer was written or looked up.
    """
    probes: list[dict[str, Any]] = []
    passed = 0
    for item in items:
        request = request_service.parse_request(item.instruction)
        if request is None:
            continue
        card = request_service.find_material(request, cards)
        if card is None:
            continue
        result = request_service.answer_request(item.instruction, tenant_id, cards, writer=writer)
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
                "written": bool(result and result.get("source") == "written"),
            }
        )
    return [{"passed": passed, "total": len(probes), "probes": probes}]


def _cross_form(
    tenant_id: int | None,
    cards: list[dict[str, Any]],
    writer: Any = None,
    held_out: list[Any] | None = None,
) -> dict[str, Any]:
    """The score on the requests the model was never worded for."""
    if held_out is None:
        held_out = instruction_service.held_out_requests(cards, limit=EVALUATION_SIZE)
    result = _probes(tenant_id, cards, held_out, writer)
    return _score(result[0])


def _contamination(cards: list[dict[str, Any]], held_out: list[Any]) -> dict[str, Any]:
    """Whether a held-out request was also part of the training wording."""
    trained = {
        item.instruction.strip().casefold()
        for item in instruction_service.training_instructions(cards)
    }
    overlaps = [
        item.instruction for item in held_out if item.instruction.strip().casefold() in trained
    ]
    return {
        "checked": len(held_out),
        "overlaps": len(overlaps),
        "examples": overlaps[:3],
        "clean": not overlaps,
    }


def _score(result: dict[str, Any]) -> dict[str, Any]:
    """The score of a set of probes."""
    total = result["total"]
    passed = result["passed"]
    return {
        "score": round(passed / total, 2) if total else 0.0,
        "passed": passed,
        "total": total,
        "written": sum(1 for probe in result["probes"] if probe.get("written")),
        "looked_up": sum(1 for probe in result["probes"] if not probe.get("written")),
        "probes": result["probes"],
    }


def _taught_topics(cards: list[dict[str, Any]], limit: int) -> list[str]:
    """Distinct topics the lessons are about, taken across the cards."""
    seen: list[str] = []
    for topics in instruction_service.topic_index(cards).values():
        for topic in topics:
            if topic and topic not in seen:
                seen.append(topic)
            if len(seen) >= limit:
                return seen
    return seen


def _adversarial(cards: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The requests about things that were never taught, and the negation."""
    half = ADVERSARIAL_SIZE // 2
    negatives = build_held_out_negatives(cards, half)
    negated = [
        (f"Write a poem that is not about {topic}.", REFUSAL)
        for topic in _taught_topics(cards, half)
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
    result = _probes(None, held, items, writer)
    return {"subject": subject, **_score(result[0])}


def _probe_writer(cards: list[dict[str, Any]]) -> Any:
    """A small model trained on the cards it is given."""
    from app.services import neural
    from app.services.language import training_materials as materials_service
    from app.services.neural.settings import TrainingSettings

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
    writer = neural.AnswerTask(PROBE_MODEL)
    corpus = neural.AnswerCorpus(training.sentences, training.instructions, training.contexts)
    report = writer.fit(corpus, settings=TrainingSettings(steps=PROBE_STEPS))
    return writer if report.get("trained") else None


def evaluate(tenant_id: int | None, cards: list[dict[str, Any]], writer: Any) -> dict[str, Any]:
    """The score of the model on the requests it was never trained on.

    Only the requests are measured. The held-out questions are answered by
    looking them up, which is what the paraphrasing rung measures already.
    """
    if writer is None:
        return {"score": 0.0, "passed": 0, "total": 0, "probes": []}
    held_out = instruction_service.held_out_requests(cards, limit=EVALUATION_SIZE)
    cross_form = _cross_form(tenant_id, cards, writer, held_out)
    adversarial = _adversarial(cards)
    return {
        **cross_form,
        "cross_form": cross_form,
        "adversarial": _refusals(tenant_id, cards, writer, adversarial),
        "cross_topic": _cross_topic(cards),
        "contamination": _contamination(cards, held_out),
    }


def _says_refusal(answer: str) -> bool:
    """Whether the writer itself said it does not know."""
    return not answer.strip() or REFUSAL.strip().casefold() in answer.casefold()


def _refusals(
    tenant_id: int | None,
    cards: list[dict[str, Any]],
    writer: Any,
    requests: list[tuple[str, str]],
) -> dict[str, Any]:
    """How often the product refuses what it was never taught.

    The score is the refusal the user gets: no answer is produced when the
    request names a topic that was never taught. ``model`` is how often the
    writer refused on its own, which is a different and much weaker number:
    the material boundary is the guard, and the writer is never asked to
    invent outside it.
    """
    probes: list[dict[str, Any]] = []
    refused = 0
    model_refused = 0
    for question, expected in requests:
        result = request_service.answer_request(question, tenant_id, cards, writer=writer)
        raw = writer.respond(question) or ""
        says_no = _says_refusal(raw)
        refused += 1 if result is None else 0
        model_refused += 1 if says_no else 0
        probes.append(
            {
                "question": question,
                "expected": expected,
                "answered": result["answer"] if result else raw,
                "correct": result is None,
                "model_refused": says_no,
            }
        )
    return {
        "score": round(refused / len(requests), 2) if requests else 0.0,
        "passed": refused,
        "total": len(requests),
        "model": model_refused,
        "probes": probes,
    }
