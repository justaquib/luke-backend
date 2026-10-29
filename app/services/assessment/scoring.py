"""Marking an answer against what its item expects, without running the answer.

Deterministic kinds are marked here: a chosen letter, an extractive span, a
final number and a Python literal. Each scorer normalises the way a person
would read the answer — case, punctuation, articles, surrounding prose — but
the comparison itself is exact, and nothing the model writes is ever executed:
a literal answer goes through ``ast.literal_eval`` only.

A rubric is not scored here. It waits for a trainer's review, and until then
its score is ``None``, which means unassessed, never zero.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.services.assessment import intake, skills

ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCTUATION = re.compile(r"[!\"#$%&'()*+,./:;<=>?@\[\\\]^_`{|}~“”‘’–—…]")
SPACES = re.compile(r"\s+")
OPTION_BARE = re.compile(r"\s*([A-F])\s*", re.IGNORECASE)
OPTION_LABELLED = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:|=)?\s*[\(\[]?\s*([A-F])(?![A-Za-z])", re.IGNORECASE
)
OPTION_DECORATED = re.compile(r"(?:^|\s)[\(\[]?([A-F])[\).:\]]")
NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
MARKED_NUMBER = re.compile(r"####\s*(-?\$?\d[\d,]*(?:\.\d+)?)")
LABEL = re.compile(r"^\s*(?:final\s+)?(?:answer|output|result)\s*(?:is|:|=)?\s*", re.IGNORECASE)
FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]*)\n?(.*?)```", re.DOTALL)
INLINE = re.compile(r"`([^`\n]+)`")

UNASSESSED = "unassessed"


@dataclass(frozen=True)
class Mark:
    """What one answer earned: a score, whether it was right, and how it was read."""

    score: float | None
    correct: bool | None
    method: str


def _words(text: str) -> str:
    """SQuAD-style normal form: no case, punctuation or articles."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    stripped = PUNCTUATION.sub(" ", folded)
    return SPACES.sub(" ", ARTICLES.sub(" ", stripped)).strip()


def _body(response: str) -> str:
    """The answer without a leading label, a code fence or wrapping quotes."""
    text = response.strip()
    fenced = FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    text = LABEL.sub("", text).strip()
    return text.strip(" \t\r\n\"'`")


def _number(text: str) -> float | None:
    """The number an answer ends on, marked with ``####`` when the text has it."""
    marked = MARKED_NUMBER.findall(text)
    if marked:
        return _parse_number(marked[-1])
    body = _body(text)
    only = NUMBER.fullmatch(body)
    if only:
        return _parse_number(only.group(0))
    found = NUMBER.findall(body)
    if found:
        return _parse_number(found[-1])
    return None


def _parse_number(text: str) -> float | None:
    try:
        return float(text.replace("$", "").replace(",", ""))
    except ValueError:
        return None


def _same_number(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(right) * 1e-9)


def _letter(response: str, choices: Sequence[str]) -> str | None:
    """The option letter the answer settles on, if it names one unambiguously.

    A bare letter or a decorated one — ``B``, ``(b)``, ``B.``, ``Answer: B`` —
    counts. A letter that is only a word, as the article in "A dog barked", is
    not a choice, so it does not count.
    """
    allowed = intake.CHOICE_LABELS[: len(choices)] if choices else intake.CHOICE_LABELS
    body = _body(response)
    bare = OPTION_BARE.fullmatch(body)
    if bare is not None and bare.group(1).upper() in allowed:
        return bare.group(1).upper()
    for pattern in (OPTION_LABELLED, OPTION_DECORATED):
        found = pattern.search(body)
        if found is None:
            continue
        letter = found.group(1).upper()
        if letter in allowed:
            return letter
    return None


def _literal(text: str) -> Any:
    """A Python literal from the answer, or ``MISSING`` when it is not one.

    The answer is read both as written and without its labels and wrapping
    quotes, because a quoted value — ``'No such key!'`` — is the value, and the
    quote marks are part of how the bank writes it. A value the model set in
    backticks inside a sentence is read too, last one first.
    """
    body = _body(text)
    stripped = text.strip()
    spanned = [found.strip() for found in reversed(INLINE.findall(text))]
    lines = [
        stripped,
        body,
        *spanned,
        *(line.strip() for line in reversed(body.splitlines())),
    ]
    for candidate in lines:
        if not candidate:
            continue
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
    return _MISSING


class _Missing:
    """A literal that could not be read, kept apart from a literal ``None``."""


_MISSING = _Missing()


def _same_literal(left: Any, right: Any) -> bool:
    if isinstance(right, bool) or isinstance(left, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _same_number(float(left), float(right))
    return left == right


def _option(item: Mapping[str, Any], response: str) -> Mark:
    expected = item.get("expected") or {}
    choices = tuple(str(choice) for choice in item.get("choices") or ())
    answer = str(expected.get("answer") or "").strip().upper()
    chosen = _letter(response, choices)
    if chosen is None:
        return Mark(0.0, False, "option")
    return Mark(1.0 if chosen == answer else 0.0, chosen == answer, "option")


def _span(item: Mapping[str, Any], response: str) -> Mark:
    expected = item.get("expected") or {}
    accepted = expected.get("acceptable") or (expected.get("answer") or "",)
    body = _words(_body(response))
    wanted = {_words(str(text)) for text in accepted}
    wanted.discard("")
    correct = bool(body) and body in wanted
    return Mark(1.0 if correct else 0.0, correct, "span")


def _numeric(item: Mapping[str, Any], response: str) -> Mark:
    target = (item.get("expected") or {}).get("answer")
    actual = _number(response)
    if actual is None or not isinstance(target, (int, float)) or isinstance(target, bool):
        return Mark(0.0, False, "number")
    correct = _same_number(actual, float(target))
    return Mark(1.0 if correct else 0.0, correct, "number")


def _target_literal(target: Any) -> Any:
    """The expected value, read out of the text the bank keeps it in.

    A code dataset stores ``[15, 15]`` as a string, so the answer it holds has
    to be parsed the same way the response is. Text that is not a literal at all
    is compared as the text it is.
    """
    if not isinstance(target, str):
        return target
    parsed = _literal(target)
    return target if parsed is _MISSING else parsed


def _spoken(value: str) -> str:
    """The value with only its case and its spacing smoothed, for a text read."""
    folded = SPACES.sub(" ", unicodedata.normalize("NFKC", value).casefold())
    return folded.strip(" \t\r\n\"'`")


def _described(item: Mapping[str, Any], response: str) -> Mark:
    target = (item.get("expected") or {}).get("answer")
    wanted = _target_literal(target)
    actual = _literal(response)
    if actual is not _MISSING and wanted is not _MISSING:
        correct = _same_literal(actual, wanted)
    elif isinstance(wanted, str):
        correct = bool(_body(response)) and _spoken(_body(response)) == _spoken(wanted)
    else:
        correct = False
    return Mark(1.0 if correct else 0.0, correct, "literal")


def _rules(item: Mapping[str, Any], response: str) -> Mark:
    from app.services.assessment.instruction_rules import check_rules

    rules = (item.get("expected") or {}).get("rules") or ()
    result = check_rules(rules, response)
    if result is None:
        return Mark(None, None, UNASSESSED)
    correct, passed, total = result
    return Mark(round(passed / total, 4), correct, "rules")


def _rubric(item: Mapping[str, Any], response: str) -> Mark:
    return Mark(None, None, "rubric")


SCORERS = {
    skills.OPTION: _option,
    skills.SPAN: _span,
    skills.NUMBER: _numeric,
    skills.LITERAL: _described,
    skills.RULES: _rules,
    skills.RUBRIC: _rubric,
}


def mark(item: Mapping[str, Any], response: str | None) -> Mark:
    """Score one answer against its item's expected block."""
    kind = str((item.get("expected") or {}).get("kind") or "")
    scorer = SCORERS.get(kind)
    if scorer is None:
        return Mark(None, None, UNASSESSED)
    return scorer(item, response or "")
