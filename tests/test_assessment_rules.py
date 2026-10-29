"""Every ported IFEval check decides the answers it should, offline."""

from __future__ import annotations

import pytest

from app.services.assessment import instruction_rules as rules
from app.services.assessment import scoring


def _rule(rule_id: str, **kwargs: object) -> dict[str, object]:
    return {"id": rule_id, "kwargs": kwargs}


CASES = [
    (_rule("keywords:existence", keywords=["apple", "banana"]), "Apple and banana", "apple only"),
    (_rule("keywords:frequency", keyword="sun", frequency=2, relation="exactly"), "sun sun", "sun"),
    (
        _rule("keywords:forbidden_words", forbidden_words=["however"]),
        "It rained.",
        "However, rain.",
    ),
    (
        _rule("keywords:letter_frequency", letter="a", let_frequency=2, let_relation="at least"),
        "banana",
        "fig",
    ),
    (_rule("change_case:english_lowercase"), "all lower", "All lower"),
    (
        _rule(
            "change_case:capital_word_frequency",
            capital_relation="exactly",
            capital_frequency=2,
        ),
        "The Cat sat",
        "the cat sat",
    ),
    (
        _rule("length_constraints:number_words", num_words=3, relation="less than"),
        "one two",
        "one two three",
    ),
    (
        _rule("length_constraints:number_sentences", num_sentences=2, relation="exactly"),
        "One. Two.",
        "One.",
    ),
    (_rule("length_constraints:number_paragraphs", num_paragraphs=2), "One\n\nTwo", "One"),
    (
        _rule("detectable_content:number_placeholders", num_placeholders=2),
        "Hi [name], [date]",
        "Hi [name]",
    ),
    (_rule("detectable_content:postscript", postscript_marker="P.S."), "Bye.\nP.S.", "Bye."),
    (_rule("detectable_format:json_format"), '{"a": 1}', "not json"),
    (_rule("detectable_format:constrained_response"), "My answer is yes.", "Yes."),
    (_rule("detectable_format:number_bullet_lists", num_bullets=2), "* one\n* two", "* one"),
    (
        _rule("detectable_format:number_highlighted_sections", num_highlights=1),
        "A *key* point",
        "A key point",
    ),
    (
        _rule("detectable_format:multiple_sections", section_spliter="SECTION", num_sections=2),
        "SECTION one\nSECTION two",
        "one section",
    ),
    (_rule("combination:two_responses"), "first\n******\nsecond", "only one"),
    (
        _rule("combination:repeat_prompt", prompt_to_repeat="Write a poem"),
        "Write a poem about rain",
        "A poem",
    ),
    (_rule("startend:end_checker", end_phrase="Amen."), "Pray. Amen.", "Pray."),
    (_rule("startend:quotation"), '"Hello."', "Hello."),
    (_rule("punctuation:no_comma"), "one two", "one, two"),
]


@pytest.mark.parametrize(("rule", "good", "bad"), CASES, ids=[c[0]["id"] for c in CASES])
def test_a_rule_passes_what_it_should_and_fails_what_it_should(
    rule: dict[str, object], good: str, bad: str
) -> None:
    checked = rules.check_rules([rule], good)
    missed = rules.check_rules([rule], bad)

    assert checked == (True, 1, 1)
    assert missed == (False, 0, 1)


def test_a_fenced_json_answer_counts() -> None:
    answer = '```json\n{"format": "json"}\n```'

    assert rules.check_rules([_rule("detectable_format:json_format")], answer) == (True, 1, 1)


def test_the_language_rule_is_not_ported_and_leaves_the_item_unassessed() -> None:
    unported = [_rule("language:response_language", language="en")]
    item = {"expected": {"kind": "rules", "rules": unported}}

    mark = scoring.mark(item, "This is English.")

    assert mark.score is None
    assert mark.correct is None
    assert mark.method == "unassessed"
    assert rules.check_rules(item["expected"]["rules"], "This is English.") is None


def test_one_unported_rule_makes_a_mixed_item_unassessed() -> None:
    mixed = [
        _rule("punctuation:no_comma"),
        _rule("language:response_language", language="en"),
    ]

    assert rules.can_check(mixed) is False
    assert rules.check_rules(mixed, "one two") is None


def test_a_mixed_item_scores_the_share_of_rules_it_kept() -> None:
    item = {
        "expected": {
            "kind": "rules",
            "rules": [
                _rule("punctuation:no_comma"),
                _rule("startend:quotation"),
            ],
        }
    }

    kept = scoring.mark(item, '"one two"')
    lost = scoring.mark(item, "one two")

    assert kept == scoring.Mark(1.0, True, "rules")
    assert lost == scoring.Mark(0.5, False, "rules")


def test_an_empty_rule_list_is_unassessed() -> None:
    assert rules.can_check([]) is False
    assert rules.check_rules([], "anything") is None


def test_an_unknown_relation_is_a_programming_error_not_a_failed_answer() -> None:
    rule = _rule("length_constraints:number_words", num_words=3, relation="about")

    with pytest.raises(ValueError, match="unknown comparison"):
        rules.check_rules([rule], "one")
