"""Chunking keeps words, questions and sentence boundaries whole."""

import itertools
import re

from app.services.nlp import chunk_text

WORDS = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"}


def _first_token(chunk: str) -> str:
    return re.split(r"\s+", chunk.strip())[0].strip(".,;:")


def _last_token(chunk: str) -> str:
    return re.split(r"\s+", chunk.strip())[-1].strip(".,;:")


def test_no_chunk_starts_or_ends_mid_word() -> None:
    text = " ".join(["alpha beta gamma delta epsilon zeta eta theta iota kappa."] * 40)

    chunks = chunk_text(text, max_chars=120, overlap=30)

    assert len(chunks) > 2
    for chunk in chunks[1:]:
        assert _first_token(chunk).lower() in WORDS
    for chunk in chunks[:-1]:
        assert _last_token(chunk).lower() in WORDS
    assert all(len(chunk) <= 120 for chunk in chunks)


def test_a_long_unpunctuated_sentence_is_split_between_words() -> None:
    text = " ".join(["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 12)

    chunks = chunk_text(text, max_chars=80, overlap=0)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 80
        assert _first_token(chunk).lower() in WORDS
        assert _last_token(chunk).lower() in WORDS


def test_a_question_starts_a_new_chunk_and_stays_whole() -> None:
    lead = " ".join(["Context before the question."] * 8)
    question = "Q: What is a closure in JavaScript?"
    answer = "A closure is a function that remembers its outer scope."
    text = f"{lead} {question} {answer}"

    chunks = chunk_text(text, max_chars=160, overlap=20)

    holding = [chunk for chunk in chunks if "What is a closure" in chunk]
    assert len(holding) == 1
    assert question in holding[0]
    assert answer in holding[0]


def _sentence_numbers(chunk: str) -> set[int]:
    return {int(number) for number in re.findall(r"Sentence number (\d+)", chunk)}


def test_overlap_repeats_whole_sentences_from_the_previous_chunk() -> None:
    sentences = [f"Sentence number {number} carries some words." for number in range(12)]

    chunks = chunk_text(" ".join(sentences), max_chars=120, overlap=60)

    assert len(chunks) > 2
    for previous, following in itertools.pairwise(chunks):
        assert max(_sentence_numbers(previous)) in _sentence_numbers(following)


def test_without_overlap_neighbouring_chunks_do_not_repeat_text() -> None:
    sentences = [f"Sentence number {number} carries some words." for number in range(12)]

    chunks = chunk_text(" ".join(sentences), max_chars=120, overlap=0)

    assert len(chunks) > 2
    for previous, following in itertools.pairwise(chunks):
        assert max(_sentence_numbers(previous)) < min(_sentence_numbers(following))


def test_text_that_fits_is_returned_unchanged_in_one_chunk() -> None:
    text = "A short document about ferries. It has two sentences."

    assert chunk_text(text, max_chars=800, overlap=100) == [text]


def test_whitespace_only_text_returns_one_chunk() -> None:
    chunks = chunk_text("   \n\n  ")

    assert len(chunks) == 1
