"""Hybrid retrieval keeps the embedding's meaning and the question's terms."""

from app.services.retrieval import (
    bm25_ranking,
    bm25_scores,
    fuse_rankings,
    similarity_from_distance,
)


def test_a_semantic_ranking_survives_when_no_term_is_shared() -> None:
    assert fuse_rankings([["a", "b"], []]) == ["a", "b"]


def test_a_named_term_beats_a_closer_embedding() -> None:
    assert fuse_rankings([["a", "b"], ["b"]])[0] == "b"


def test_a_passage_found_only_by_its_words_is_kept() -> None:
    fused = fuse_rankings([["semantic"], ["lexical"]])

    assert set(fused) == {"semantic", "lexical"}


def test_stopwords_alone_do_not_rank_a_passage() -> None:
    assert bm25_ranking("what is the schedule", ["Ferry timetable to the island."]) == []


def test_bm25_ranks_the_passage_that_uses_the_named_word() -> None:
    documents = ["Ferry timetable to the island.", "The voucher holds it."]

    assert bm25_ranking("voucher", documents) == [1]


def test_bm25_prefers_the_passage_that_uses_the_word_more() -> None:
    documents = ["The voucher is here.", "The voucher holds the voucher number."]
    scores = bm25_scores("voucher", documents)

    assert scores[1] > scores[0] > 0


def test_a_question_of_only_stopwords_keeps_every_score_at_zero() -> None:
    assert bm25_scores("what is it", ["First passage.", "Second passage."]) == [0.0, 0.0]


def test_empty_rankings_stay_empty() -> None:
    assert fuse_rankings([]) == []
    assert fuse_rankings([[], []]) == []


def test_a_fused_tie_follows_the_first_ranking() -> None:
    assert fuse_rankings([["first"], ["second"]]) == ["first", "second"]


def test_a_squared_l2_distance_is_read_as_a_cosine_similarity() -> None:
    assert similarity_from_distance(0.0, "l2") == 1.0
    assert similarity_from_distance(0.4, "l2") == 0.8
    assert similarity_from_distance(2.0, "l2") == 0.0


def test_a_cosine_distance_is_read_as_one_minus_the_distance() -> None:
    assert similarity_from_distance(0.0, "cosine") == 1.0
    assert similarity_from_distance(0.4, "cosine") == 0.6


def test_a_missing_distance_stays_missing() -> None:
    assert similarity_from_distance(None, "l2") is None
