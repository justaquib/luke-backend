"""The gate a grounded answer must pass before it is shown as grounded.

A citation must point at evidence that was actually supplied, and the written
answer may not introduce a number, name or measurement that the evidence does
not contain. When the gate fails, the caller shows the extractive answer again
instead of the written one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

CITATION = re.compile(r"\[(\d{1,2})\]")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
ENTITY_LABELS = frozenset({"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "DATE", "TIME"})


def cited(answer: str) -> list[int]:
    """The evidence numbers an answer cites, in the order they appear."""
    return [int(value) for value in CITATION.findall(answer or "")]


def verify_grounded(answer: str, evidence: Sequence[str]) -> tuple[bool, str]:
    """Whether an answer is genuinely grounded on the supplied evidence."""
    text = (answer or "").strip()
    if not text:
        return False, "The written answer is empty."
    numbers = cited(text)
    if not numbers:
        return False, "The written answer did not cite the supplied evidence."
    if any(number < 1 or number > len(evidence) for number in numbers):
        return False, "A citation points outside the supplied evidence."
    without_citations = CITATION.sub("", text)
    material = " ".join(evidence)
    introduced = sorted(set(NUMBER.findall(without_citations)) - set(NUMBER.findall(material)))
    if introduced:
        return False, (
            "The written answer introduced a number the evidence does not contain: "
            f"{', '.join(introduced)}."
        )
    unknown = _unknown_entities(without_citations, material)
    if unknown:
        return False, (
            "The written answer named something the evidence does not contain: "
            f"{', '.join(unknown)}."
        )
    return True, "Every citation resolves to the supplied evidence and nothing was added."


def _unknown_entities(answer: str, material: str) -> list[str]:
    """Names in the answer that the evidence never mentions."""
    from app.services import nlp

    try:
        entities = nlp.extract_entities(answer)
    except (OSError, ValueError, RuntimeError):
        return []
    known = material.casefold()
    unknown: list[str] = []
    for entity in entities:
        if str(entity.get("label") or "") not in ENTITY_LABELS:
            continue
        text = str(entity.get("text") or "").strip()
        if text and text.casefold() not in known and text not in unknown:
            unknown.append(text)
    return unknown[:4]
