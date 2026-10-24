"""Hybrid retrieval keeps the embedding's meaning and the question's terms."""

from app.services.retrieval import rerank_hits


def _hit(text: str, distance: float | None) -> dict:
    return {"text": text, "distance": distance, "metadata": {}}


def test_a_semantic_hit_stays_first_when_no_terms_are_shared() -> None:
    hits = [_hit("Ferry timetable to the island.", 0.1), _hit("Hotel booking policy.", 0.9)]

    assert rerank_hits("schedule", hits)[0] is hits[0]


def test_a_named_term_beats_a_closer_embedding() -> None:
    hits = [_hit("Ferry timetable to the island.", 0.2), _hit("The voucher holds it.", 0.8)]

    assert rerank_hits("voucher", hits)[0] is hits[1]


def test_stopwords_alone_do_not_reorder_hits() -> None:
    hits = [_hit("Ferry timetable to the island.", 0.1), _hit("What is the policy there.", 0.9)]

    assert rerank_hits("what is the schedule", hits)[0] is hits[0]


def test_missing_distances_do_not_break_the_ranking() -> None:
    hits = [_hit("Voucher details.", None), _hit("Ferry timetable.", None)]

    assert rerank_hits("voucher", hits)[0] is hits[0]


def test_empty_hits_stay_empty() -> None:
    assert rerank_hits("anything", []) == []


def test_a_question_of_only_stopwords_keeps_semantic_order() -> None:
    hits = [_hit("First passage.", 0.2), _hit("Second passage.", 0.8)]

    assert rerank_hits("what is it", hits)[0] is hits[0]
