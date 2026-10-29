"""A deterministic answer is marked exactly, and never by running it."""

from __future__ import annotations

from typing import Any

import pytest

from app.services.assessment import scoring


def _item(kind: str, choices: list[str] | None = None, **expected: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"expected": {"kind": kind, **expected}}
    if choices is not None:
        item["choices"] = choices
    return item


def test_an_option_letter_is_read_from_surrounding_prose() -> None:
    item = _item("option", answer="B", choices=["Red", "Blue", "Green"])

    assert scoring.mark(item, "The answer is B.") == scoring.Mark(1.0, True, "option")
    assert scoring.mark(item, "(c)") == scoring.Mark(0.0, False, "option")


def test_a_letter_inside_a_word_is_not_an_option() -> None:
    item = _item("option", answer="A", choices=["Red", "Blue"])

    assert scoring.mark(item, "A dog barked") == scoring.Mark(0.0, False, "option")
    assert scoring.mark(item, "B is the one") == scoring.Mark(0.0, False, "option")


def test_an_empty_option_answer_is_wrong_not_crashing() -> None:
    item = _item("option", answer="A", choices=["Red", "Blue"])

    assert scoring.mark(item, None) == scoring.Mark(0.0, False, "option")
    assert scoring.mark(item, "   ") == scoring.Mark(0.0, False, "option")


def test_a_span_ignores_case_punctuation_and_articles() -> None:
    item = _item("span", answer="the wet grass", acceptable=["the wet grass", "wet grass"])

    assert scoring.mark(item, "Wet grass.") == scoring.Mark(1.0, True, "span")
    assert scoring.mark(item, "Answer: the wet grass") == scoring.Mark(1.0, True, "span")
    assert scoring.mark(item, "dry grass") == scoring.Mark(0.0, False, "span")


def test_a_span_needs_the_whole_span_not_a_piece_of_it() -> None:
    item = _item("span", answer="a small blue ball")

    assert scoring.mark(item, "blue ball") == scoring.Mark(0.0, False, "span")


def test_a_marked_number_is_the_final_answer_even_after_reasoning() -> None:
    item = _item("number", answer=9)
    answer = "She had 4, then bought 5 more.\n#### 9"

    assert scoring.mark(item, answer) == scoring.Mark(1.0, True, "number")


def test_a_number_accepts_commas_decimals_and_a_dollar_sign() -> None:
    item = _item("number", answer=1234.5)

    assert scoring.mark(item, "$1,234.50") == scoring.Mark(1.0, True, "number")
    assert scoring.mark(item, "The total is 1234.5 dollars") == scoring.Mark(1.0, True, "number")
    assert scoring.mark(item, "1235") == scoring.Mark(0.0, False, "number")


def test_an_unreadable_number_is_wrong() -> None:
    item = _item("number", answer=3)

    assert scoring.mark(item, "not a number") == scoring.Mark(0.0, False, "number")


def test_a_literal_matches_its_value_and_type() -> None:
    item = _item("literal", answer=[1, 2])

    assert scoring.mark(item, "[1, 2]") == scoring.Mark(1.0, True, "literal")
    assert scoring.mark(item, "```python\n[1, 2]\n```") == scoring.Mark(1.0, True, "literal")
    assert scoring.mark(item, "[1, 3]") == scoring.Mark(0.0, False, "literal")


def test_a_literal_boolean_is_not_a_number() -> None:
    item = _item("literal", answer=True)

    assert scoring.mark(item, "True") == scoring.Mark(1.0, True, "literal")
    assert scoring.mark(item, "1") == scoring.Mark(0.0, False, "literal")


def test_a_literal_the_bank_writes_as_text_is_still_matched() -> None:
    item = _item("literal", answer="[15, 15]")

    assert scoring.mark(item, "[15, 15]") == scoring.Mark(1.0, True, "literal")
    assert scoring.mark(item, "The output is `[15, 15]`") == scoring.Mark(1.0, True, "literal")
    assert scoring.mark(item, "```\n[15, 15]\n```") == scoring.Mark(1.0, True, "literal")
    assert scoring.mark(item, "[15, 16]") == scoring.Mark(0.0, False, "literal")


def test_a_literal_string_keeps_its_quotes_on_both_sides() -> None:
    item = _item("literal", answer="'No such key!'")

    assert scoring.mark(item, "'No such key!'") == scoring.Mark(1.0, True, "literal")
    assert scoring.mark(item, "No such key!") == scoring.Mark(1.0, True, "literal")
    assert scoring.mark(item, "no such key !") == scoring.Mark(0.0, False, "literal")
    assert scoring.mark(item, "Another key") == scoring.Mark(0.0, False, "literal")


def test_an_answer_that_is_code_is_never_run(tmp_path) -> None:
    item = _item("literal", answer=3)
    target = tmp_path / "pwned"
    answer = f"__import__('pathlib').Path({str(target)!r}).touch()"

    assert scoring.mark(item, answer) == scoring.Mark(0.0, False, "literal")
    assert not target.exists()


def test_a_rubric_waits_for_review_and_is_not_zero() -> None:
    mark = scoring.mark(_item("rubric", criteria=["It keeps to the passage."]), "A summary.")

    assert mark.score is None
    assert mark.correct is None
    assert mark.method == "rubric"


@pytest.mark.parametrize("kind", ["vibes", ""])
def test_an_unknown_kind_is_unassessed(kind: str) -> None:
    assert scoring.mark(_item(kind), "anything") == scoring.Mark(None, None, "unassessed")
