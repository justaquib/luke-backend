"""The IFEval instruction checks, ported to plain Python.

Ported from ``google/IFEval``'s ``instructions_registry.py`` so an instruction
item can be marked offline: no network, no model and no ``nltk``, ``absl``,
``immutabledict`` or ``langdetect`` package. The checks keep the original
meaning. Two divergences are deliberate and documented:

- sentences are split on whitespace after ``.``, ``!`` or ``?`` instead of with
  ``nltk.sent_tokenize``, so a run needs no corpus download;
- ``language:response_language`` is not ported, because deciding the language
  of an answer honestly needs a detector. A rule this module cannot check
  makes its whole item unassessed, never failed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

Rule = Mapping[str, Any]

SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
BULLET = re.compile(r"^\* ", re.MULTILINE)
HIGHLIGHT = re.compile(r"\*[^*\n]+\*")
PLACEHOLDER = re.compile(r"\[[^\[\]]*\]")
JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
TWO_RESPONSE_MARK = "******"
CONSTRAINED = frozenset({"my answer is yes.", "my answer is no.", "my answer is maybe."})
UNSUPPORTED = frozenset({"language:response_language"})


def _relation(actual: int, target: int, relation: str) -> bool:
    if relation == "less than":
        return actual < target
    if relation == "at least":
        return actual >= target
    if relation == "exactly":
        return actual == target
    raise ValueError(f"unknown comparison '{relation}'")


def _keyword_count(text: str, keyword: str) -> int:
    pattern = rf"\b{re.escape(keyword)}\b"
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _existence(kwargs: Rule, response: str) -> bool:
    words = [str(word) for word in kwargs.get("keywords") or ()]
    return bool(words) and all(word.casefold() in response.casefold() for word in words)


def _frequency(kwargs: Rule, response: str) -> bool:
    count = _keyword_count(response, str(kwargs.get("keyword") or ""))
    return _relation(count, int(kwargs.get("frequency", 0)), str(kwargs.get("relation", "")))


def _forbidden(kwargs: Rule, response: str) -> bool:
    words = [str(word) for word in kwargs.get("forbidden_words") or ()]
    return not any(_keyword_count(response, word) for word in words)


def _letter_frequency(kwargs: Rule, response: str) -> bool:
    letter = str(kwargs.get("letter") or "").casefold()
    if not letter:
        return False
    count = response.casefold().count(letter)
    target = int(kwargs.get("let_frequency", 0))
    return _relation(count, target, str(kwargs.get("let_relation", "")))


def _lowercase(kwargs: Rule, response: str) -> bool:
    return response == response.lower()


def _capital_word_frequency(kwargs: Rule, response: str) -> bool:
    capitals = sum(1 for word in response.split() if word[:1].isupper())
    relation = str(kwargs.get("capital_relation", ""))
    return _relation(capitals, int(kwargs.get("capital_frequency", 0)), relation)


def _words(kwargs: Rule, response: str) -> bool:
    count = len(response.split())
    return _relation(count, int(kwargs.get("num_words", 0)), str(kwargs.get("relation", "")))


def _sentences(kwargs: Rule, response: str) -> bool:
    parts = [part for part in SENTENCE_BREAK.split(response.strip()) if part.strip()]
    target = int(kwargs.get("num_sentences", 0))
    return _relation(len(parts), target, str(kwargs.get("relation", "")))


def _paragraphs(kwargs: Rule, response: str) -> bool:
    parts = [part for part in PARAGRAPH_BREAK.split(response.strip()) if part.strip()]
    return len(parts) == int(kwargs.get("num_paragraphs", 0))


def _placeholders(kwargs: Rule, response: str) -> bool:
    return len(PLACEHOLDER.findall(response)) >= int(kwargs.get("num_placeholders", 0))


def _postscript(kwargs: Rule, response: str) -> bool:
    marker = str(kwargs.get("postscript_marker") or "P.S.")
    return response.rstrip().endswith(marker)


def _json_format(kwargs: Rule, response: str) -> bool:
    body = response.strip()
    fenced = JSON_FENCE.search(body)
    if fenced is not None:
        body = fenced.group(1).strip()
    try:
        json.loads(body)
    except (TypeError, ValueError):
        return False
    return True


def _constrained_response(kwargs: Rule, response: str) -> bool:
    return response.strip().casefold() in CONSTRAINED


def _bullets(kwargs: Rule, response: str) -> bool:
    return len(BULLET.findall(response)) >= int(kwargs.get("num_bullets", 0))


def _highlights(kwargs: Rule, response: str) -> bool:
    return len(HIGHLIGHT.findall(response)) >= int(kwargs.get("num_highlights", 0))


def _sections(kwargs: Rule, response: str) -> bool:
    splitter = str(kwargs.get("section_spliter") or "SECTION").casefold()
    return response.casefold().count(splitter) >= int(kwargs.get("num_sections", 0))


def _two_responses(kwargs: Rule, response: str) -> bool:
    parts = response.split(TWO_RESPONSE_MARK)
    return len(parts) == 2 and all(part.strip() for part in parts)


def _repeat_prompt(kwargs: Rule, response: str) -> bool:
    prompt = str(kwargs.get("prompt_to_repeat") or "")
    return bool(prompt) and response.lstrip().startswith(prompt)


def _end_phrase(kwargs: Rule, response: str) -> bool:
    phrase = str(kwargs.get("end_phrase") or "")
    return bool(phrase) and response.rstrip().endswith(phrase)


def _quotation(kwargs: Rule, response: str) -> bool:
    body = response.strip()
    return body.startswith('"') and body.endswith('"') and len(body) > 1


def _no_comma(kwargs: Rule, response: str) -> bool:
    return "," not in response


CHECKS = {
    "keywords:existence": _existence,
    "keywords:frequency": _frequency,
    "keywords:forbidden_words": _forbidden,
    "keywords:letter_frequency": _letter_frequency,
    "change_case:english_lowercase": _lowercase,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "length_constraints:number_words": _words,
    "length_constraints:number_sentences": _sentences,
    "length_constraints:number_paragraphs": _paragraphs,
    "detectable_content:number_placeholders": _placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:json_format": _json_format,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:number_bullet_lists": _bullets,
    "detectable_format:number_highlighted_sections": _highlights,
    "detectable_format:multiple_sections": _sections,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:end_checker": _end_phrase,
    "startend:quotation": _quotation,
    "punctuation:no_comma": _no_comma,
}

SUPPORTED = frozenset(CHECKS)


def can_check(rules: Sequence[Rule]) -> bool:
    """Whether every rule on an item is one this module can decide."""
    return bool(rules) and all(str(rule.get("id") or "") in SUPPORTED for rule in rules)


def check_rules(rules: Sequence[Rule], response: str) -> tuple[bool, int, int] | None:
    """How many rules the answer obeyed, or ``None`` when any rule is unported."""
    if not can_check(rules):
        return None
    results = [CHECKS[str(rule["id"])](rule.get("kwargs") or {}, response) for rule in rules]
    passed = sum(1 for result in results if result)
    return passed == len(results), passed, len(results)
