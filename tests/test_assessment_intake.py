"""Every source record is shaped into a question with an answer that can be scored."""

from __future__ import annotations

import pytest

from app.services.assessment import intake, skills
from app.services.assessment.intake_grammar import blimp_draft, ifeval_draft
from app.services.assessment.intake_reading import boolq_draft, squad_draft
from app.services.assessment.intake_reasoning import (
    arc_easy_draft,
    cruxeval_draft,
    gsm8k_draft,
)
from app.services.assessment.intake_writing import dolly_creative_draft, dolly_summary_draft
from app.services.assessment.sources import BLIMP, SOURCES

BLIMP_RECORD = {
    "sentence_good": "Susan revealed herself.",
    "sentence_bad": "Susan revealed themselves.",
    "UID": "anaphor_number_agreement",
    "linguistics_term": "anaphor_agreement",
    "field": "morphology",
    "pair_id": 4,
}

SQUAD_RECORD = {
    "id": "56be4db0acb8001400a502ec",
    "title": "Super_Bowl_50",
    "context": "The Denver Broncos defeated the Carolina Panthers.",
    "question": "Which team won?",
    "answers": {"text": ["Denver Broncos", "Denver", "the Broncos"], "answer_start": [4, 4, 4]},
}


def _letter_of(draft: intake.Draft, text: str) -> str:
    return intake.CHOICE_LABELS[draft.choices.index(text)]


def test_blimp_points_at_the_grammatical_sentence() -> None:
    draft = blimp_draft(BLIMP_RECORD, 0)
    assert draft is not None
    assert draft.choices[0] != draft.choices[1]
    assert draft.answer == _letter_of(draft, BLIMP_RECORD["sentence_good"])
    assert BLIMP_RECORD["sentence_bad"] in draft.choices


def test_blimp_structural_paradigm_is_the_hardest_stage() -> None:
    plain = blimp_draft(BLIMP_RECORD, 0)
    island = blimp_draft(BLIMP_RECORD | {"UID": "complex_NP_island"}, 0)
    assert plain is not None and island is not None
    assert plain.stage < skills.MAX_STAGE
    assert island.stage == skills.MAX_STAGE


def test_ifeval_keeps_every_constraint() -> None:
    draft = ifeval_draft(
        {
            "key": 1000,
            "prompt": "Write a hundred words without commas.",
            "instruction_id_list": ["length_constraints:number_words", "punctuation:no_comma"],
            "kwargs": [{"num_words": 100, "relation": "at least"}, {"num_comma": None}],
        },
        0,
    )
    assert draft is not None
    assert [rule["id"] for rule in draft.rules] == [
        "length_constraints:number_words",
        "punctuation:no_comma",
    ]
    assert draft.rules[0]["kwargs"] == {"num_words": 100, "relation": "at least"}
    assert draft.rules[1]["kwargs"] == {}
    assert draft.stage == 2
    assert draft.topic == "length_constraints"


def test_ifeval_skips_a_record_whose_kwargs_do_not_match() -> None:
    assert (
        ifeval_draft(
            {
                "key": 1,
                "prompt": "Do the thing.",
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [],
            },
            0,
        )
        is None
    )


def test_squad_answers_with_the_shortest_accepted_span() -> None:
    draft = squad_draft(SQUAD_RECORD, 0)
    assert draft is not None
    assert draft.answer == "Denver"
    assert "Denver Broncos" in draft.acceptable
    assert draft.stage == 1
    assert draft.topic == "which"
    assert SQUAD_RECORD["context"] in draft.prompt


@pytest.mark.parametrize(("answer", "word"), [(True, "Yes"), (False, "No")])
def test_boolq_letters_the_answer_it_was_given(answer: bool, word: str) -> None:
    draft = boolq_draft({"question": "Is it?", "answer": answer, "passage": "It is."}, 7)
    assert draft is not None
    assert draft.answer == _letter_of(draft, word)
    assert draft.record == "7"


def test_gsm8k_reads_the_final_number_and_counts_the_steps() -> None:
    draft = gsm8k_draft(
        {
            "question": "Janet has 16 eggs and eats three.",
            "answer": "She has 16 - 3 = <<16-3=13>>13.\nThen 13 - 4 = <<13-4=9>>9.\n#### 9",
        },
        0,
    )
    assert draft is not None
    assert draft.answer == 9
    assert draft.extra["steps"] == 2
    assert draft.stage == 2


def test_gsm8k_refuses_a_record_without_an_answer_marker() -> None:
    assert gsm8k_draft({"question": "How many?", "answer": "Nine."}, 0) is None


def test_arc_maps_a_numeric_answer_key_to_its_letter() -> None:
    draft = arc_easy_draft(
        {
            "id": "Mercury_1",
            "question": "Which planet is hottest?",
            "choices": {"text": ["Venus", "Mars"], "label": ["1", "2"]},
            "answerKey": "2",
        },
        0,
    )
    assert draft is not None
    assert draft.answer == "B"
    assert draft.skill == "general_knowledge"
    assert draft.source == "arc_easy"


def test_cruxeval_skips_a_program_that_does_not_parse() -> None:
    broken = {"code": "def f(x):\n    return (", "input": "1", "output": "1", "id": "sample_9"}
    assert cruxeval_draft(broken, 0) is None


def test_cruxeval_asks_for_the_exact_return_value() -> None:
    draft = cruxeval_draft(
        {"code": "def f(x):\n    return x + 1", "input": "1", "output": "2", "id": "sample_0"},
        0,
    )
    assert draft is not None
    assert draft.answer == "2"
    assert "f(1) = ?" in draft.prompt
    assert draft.topic == "sequence of statements"


def test_dolly_keeps_each_category_in_its_own_skill() -> None:
    creative = dolly_creative_draft(
        {
            "instruction": "Write a short poem about the sea.",
            "context": "",
            "response": "The sea is wide.",
            "category": "creative_writing",
        },
        3,
    )
    summary = dolly_summary_draft(
        {
            "instruction": "Summarise the passage.",
            "context": "The sea is wide and deep.",
            "response": "The sea is large.",
            "category": "summarization",
        },
        4,
    )
    assert creative is not None and summary is not None
    assert creative.skill == "creative_writing" and creative.topic == "poem"
    assert summary.skill == "reading"
    assert creative.criteria and summary.criteria
    assert (
        dolly_creative_draft({"category": "open_qa", "instruction": "x", "response": "y"}, 5)
        is None
    )
    assert (
        dolly_summary_draft({"category": "open_qa", "instruction": "x", "response": "y"}, 5) is None
    )


def test_every_registered_source_names_a_shaper_and_a_licence() -> None:
    assert set(SOURCES) == {
        "blimp",
        "ifeval",
        "squad",
        "boolq",
        "dolly_summary",
        "gsm8k",
        "cruxeval",
        "arc_easy",
        "arc_challenge",
        "dolly_creative",
    }
    for source in SOURCES.values():
        assert source.meta.licence and source.meta.url.startswith("https://")
        assert source.meta.skill in skills.SKILL_IDS
        assert source.fetches
    assert len(BLIMP.fetches) > 1
