"""How well the model answers requests it was never worded for.

A model that only looks things up answers the request it was taught and
nothing else. This measures the other half: the requests the model was never
trained on, phrased in words the trainer never used, about material the
trainer did teach.

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


def evaluate(tenant_id: int | None, cards: list[dict[str, Any]], writer: Any) -> dict[str, Any]:
    """The score of the model on the requests it was never trained on.

    Only the requests are measured. The held-out questions are answered by
    looking them up, which is what the paraphrasing rung measures already.
    """
    if writer is None:
        return {"score": 0.0, "passed": 0, "total": 0, "probes": []}
    held_out = instruction_service.held_out_requests(cards, limit=EVALUATION_SIZE)
    probes: list[dict[str, Any]] = []
    passed = 0
    for item in held_out:
        request = request_service.parse_request(item.instruction)
        if request is None:
            continue
        card = request_service.find_material(request, cards)
        if card is None:
            continue
        # The answer comes from the same path the chat uses, so the score is
        # the score of the assistant and not of the model alone.
        result = request_service.answer_request(
            item.instruction, tenant_id, cards, writer=writer
        )
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
    total = len(probes)
    score = passed / total if total else 0.0
    return {
        "score": round(score, 2),
        "passed": passed,
        "total": total,
        "probes": probes,
    }
