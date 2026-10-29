"""The bank loads offline, and a profile never turns missing evidence into zero."""

from __future__ import annotations

from typing import Any

import pytest

from app.services.assessment import bank, skills
from app.services.assessment.skills import Outcome


def _outcome(
    score: float | None,
    *,
    skill: str = "grammar",
    stage: int = 1,
    topic: str = "agreement",
    kind: str = "option",
) -> Outcome:
    return Outcome(skill=skill, stage=stage, topic=topic, kind=kind, score=score)


def _many(count: int, score: float, **values: Any) -> list[Outcome]:
    return [_outcome(score, **values) for _ in range(count)]


def test_every_skill_file_holds_the_promised_items() -> None:
    for skill in skills.SKILL_IDS:
        found = bank.items(skill)

        assert len(found) == 45, skill
        assert sum(1 for item in found if item["split"] == "eval") == 30
        assert sum(1 for item in found if item["split"] == "train") == 15


def test_the_split_filter_returns_only_that_side() -> None:
    held = bank.items(split="eval")
    taught = bank.items(split="train")

    assert len(held) == 240
    assert len(taught) == 120
    assert {item["split"] for item in held} == {"eval"}


def test_an_item_can_be_found_by_the_id_its_result_stores() -> None:
    first = bank.items("mathematics")[0]

    assert bank.item(first["id"]) == first
    assert bank.item("mathematics:not-a-source:0") is None


def test_the_manifest_carries_a_version_and_the_split_salt() -> None:
    assert bank.version() >= 1
    assert bank.salt() == "luke-assessment-v1"


def test_the_whole_bank_passes_its_offline_check() -> None:
    assert bank.check() == []
    assert bank.counts()["reading"] == {"eval": 30, "train": 15}


def test_an_unknown_skill_is_refused() -> None:
    with pytest.raises(bank.BankError, match="unknown skill"):
        bank.items("handwriting")

    with pytest.raises(bank.BankError, match="unknown split"):
        bank.items(split="holdout")


def test_a_skill_with_no_evidence_is_unassessed_not_zero() -> None:
    profile = skills.build_profile("grammar", [], bank_items=[{"topic": "agreement"}])

    assert profile.status == "unassessed"
    assert profile.score is None
    assert profile.stage is None
    assert profile.confidence == "none"
    assert profile.verdict == "not measured yet"
    assert profile.items_scored == 0
    assert profile.next_step.startswith("Run the held-out items")


def test_a_rubric_waiting_for_review_keeps_the_skill_unassessed() -> None:
    waiting = [_outcome(None, skill="creative_writing", topic="poem", kind="rubric")]

    profile = skills.build_profile("creative_writing", waiting, bank_items=[])

    assert profile.status == "unassessed"
    assert profile.waiting == 1
    assert profile.next_step.startswith("A trainer needs to review")


def test_a_strong_run_clears_stages_and_names_its_strengths() -> None:
    outcomes = _many(20, 1.0, stage=1) + _many(5, 1.0, stage=2, topic="islands")

    profile = skills.build_profile("grammar", outcomes, bank_items=[])

    assert profile.status == "assessed"
    assert profile.score == 1.0
    assert profile.stage == 2
    assert profile.confidence == "high"
    assert profile.verdict == "benchmark passed"
    assert profile.strengths == ("agreement", "islands")
    assert profile.weaknesses == ()
    assert profile.next_step == "Move to stage 3."


def test_a_weak_run_names_the_topic_to_practise() -> None:
    outcomes = _many(4, 0.0, topic="scope")

    profile = skills.build_profile("grammar", outcomes, bank_items=[])

    assert profile.score == 0.0
    assert profile.stage is None
    assert profile.confidence == "low"
    assert profile.verdict == "benchmark not yet passed"
    assert profile.weaknesses == ("scope",)
    assert profile.next_step == "Practise scope at stage 1."


def test_two_correct_items_are_not_a_cleared_stage() -> None:
    outcomes = _many(2, 1.0)

    profile = skills.build_profile("grammar", outcomes, bank_items=[])

    assert profile.stage is None
    assert profile.next_step == "Repeat the stage 1 items; 100% passed."


def test_confidence_rises_with_the_number_of_scored_items() -> None:
    assert skills.build_profile("grammar", _many(9, 1.0), bank_items=[]).confidence == "low"
    assert skills.build_profile("grammar", _many(10, 1.0), bank_items=[]).confidence == "medium"
    assert skills.build_profile("grammar", _many(19, 1.0), bank_items=[]).confidence == "medium"
    assert skills.build_profile("grammar", _many(20, 1.0), bank_items=[]).confidence == "high"


def test_the_delta_is_measured_against_the_previous_run() -> None:
    profile = skills.build_profile("grammar", _many(10, 0.8), previous=0.5, bank_items=[])
    first = skills.build_profile("grammar", _many(10, 0.8), bank_items=[])

    assert profile.delta == 0.3
    assert first.delta is None


def test_untested_areas_are_the_bank_topics_the_run_never_asked() -> None:
    topics = [{"topic": "agreement"}, {"topic": "scope"}, {"topic": "islands"}]
    outcomes = _many(3, 1.0, topic="agreement")

    profile = skills.build_profile("grammar", outcomes, bank_items=topics)

    assert profile.untested == ("islands", "scope")


def test_the_profile_carries_the_skill_title() -> None:
    profile = skills.build_profile("science", [], bank_items=[])

    assert profile.title == skills.SKILL_TITLES["science"]
