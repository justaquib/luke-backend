"""Tests for the scratch tokenizers and the artifacts they are read from.

The word adapter must keep every legacy id; byte-level BPE must round-trip any
text; an artifact must be found by its hash and refused when it is missing or
changed; and a held-out sentence must never shape a tokenizer's vocabulary.
"""

from __future__ import annotations

import unicodedata

import pytest

from app.services import vocabulary
from app.services.neural import checkpoint
from app.services.neural.model import TinyTransformer
from app.services.neural.settings import TrainingSettings
from app.services.neural.tokenizer import artifacts, corpus, spec
from app.services.neural.tokenizer.base import (
    BPE_KIND,
    SPECIAL_NAMES,
    WORD_KIND,
    CorruptArtifactError,
    MissingArtifactError,
    VocabularyMismatchError,
)
from app.services.neural.tokenizer.bpe import BpeTokenizer
from app.services.neural.tokenizer.word import WordTokenizer
from app.services.neural.tokens import SPECIAL_TOKENS, tokenize
from app.services.neural.words import WordTask

CORPUS = [
    "The cow is an animal.",
    "The cow is a farm animal that gives milk.",
    "A cow gives milk every day.",
    "The moon shines at night.",
    "Water is the clear liquid people drink.",
    "People drink water every day.",
] * 8

ROUND_TRIP = [
    "Hello, world!",
    "दूध गाय से आता है",
    "世界 🙂 café",
    "a\u0301 combining mark",
    "مرحبا بالعالم",
    "123.45 and 6,789",
    "two  spaces\tand a tab",
    "line one\nline two\n\nline four",
    "<pad> is just text here",
    "x" * 200,
    "",
]


def _bpe(size: int = 300) -> BpeTokenizer:
    return BpeTokenizer.train(CORPUS, size)


@pytest.fixture(autouse=True)
def _artifacts_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "root", lambda: tmp_path / "tokenizers")


def test_the_word_adapter_keeps_the_legacy_ids():
    text = "The cow is an animal."
    expected = sorted(set(tokenize(text)) | set(SPECIAL_TOKENS))
    tokenizer = WordTokenizer.build([text])

    assert tokenizer.vocabulary() == expected
    assert tokenizer.encode(text) == [expected.index(token) for token in tokenize(text)]
    assert tokenizer.special_ids == {name: expected.index(name) for name in SPECIAL_NAMES}


def test_a_saved_word_vocabulary_reads_back_the_same_ids(tmp_path):
    tokenizer = WordTokenizer.build(CORPUS)
    tokenizer.save(tmp_path)

    loaded = WordTokenizer.load(tmp_path)

    assert loaded.vocabulary() == tokenizer.vocabulary()
    assert loaded.encode("The cow is an animal.") == tokenizer.encode("The cow is an animal.")


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_bpe_round_trips_every_text(text):
    tokenizer = _bpe()

    assert tokenizer.decode(tokenizer.encode(text)) == unicodedata.normalize("NFC", text)


def test_bpe_fixes_its_special_ids_and_never_encodes_one():
    tokenizer = _bpe()
    text = "<pad> <start> <end> <unk> <br>"

    ids = tokenizer.encode(text)

    assert tokenizer.special_ids == {name: position for position, name in enumerate(SPECIAL_NAMES)}
    assert not any(tokenizer.is_special(token_id) for token_id in ids)
    assert tokenizer.decode(ids) == text


def test_a_line_break_is_one_mark_and_keeps_the_poem_shape():
    tokenizer = _bpe()

    ids = tokenizer.encode("first line\nsecond line")

    assert ids.count(tokenizer.special_ids["<br>"]) == 1
    assert tokenizer.decode(ids) == "first line\nsecond line"


def test_the_same_corpus_trains_the_same_artifact():
    first = _bpe(400)
    second = _bpe(400)

    assert first.digest() == second.digest()
    assert first.encode("The cow is a farm animal.") == second.encode("The cow is a farm animal.")


def test_a_saved_bpe_tokenizer_reads_back_the_same_ids(tmp_path):
    tokenizer = _bpe()
    tokenizer.save(tmp_path)

    loaded = BpeTokenizer.load(tmp_path)

    assert loaded.encode("The cow is an animal.") == tokenizer.encode("The cow is an animal.")
    assert loaded.digest() == tokenizer.digest()


def test_an_artifact_is_found_by_its_hash_and_reads_back(tmp_path):
    tokenizer = _bpe()
    record = artifacts.record(tokenizer, {"passages": 48})

    loaded = artifacts.load(record)

    assert record["id"] == tokenizer.digest()
    assert (tmp_path / "tokenizers" / record["id"] / "tokenizer.json").is_file()
    assert loaded.encode("The moon shines at night.") == tokenizer.encode(
        "The moon shines at night."
    )
    assert artifacts.meta(record["id"])["corpus"] == {"passages": 48}
    assert artifacts.describe(record)["present"] is True


def test_a_missing_artifact_is_refused():
    record = {"kind": BPE_KIND, "id": "0" * 64, "vocab_size": 300}

    with pytest.raises(MissingArtifactError):
        artifacts.load(record)
    assert artifacts.describe(record)["error"] is not None


def test_a_changed_artifact_is_refused(tmp_path):
    tokenizer = _bpe()
    record = artifacts.record(tokenizer)
    path = tmp_path / "tokenizers" / record["id"] / "tokenizer.json"
    path.write_text("changed", encoding="utf-8")

    with pytest.raises(CorruptArtifactError):
        artifacts.load(record)
    assert artifacts.describe(record)["present"] is False


def test_a_foreign_or_unknown_tokenizer_is_refused():
    with pytest.raises(CorruptArtifactError):
        artifacts.load({"kind": "teacher", "id": "0" * 64, "vocab_size": 1})
    with pytest.raises(VocabularyMismatchError):
        artifacts.load({"kind": WORD_KIND, "vocab_size": 1}, vocabulary=[])
    with pytest.raises(VocabularyMismatchError):
        artifacts.verify({"kind": BPE_KIND, "vocab_size": 10, "special_ids": {}}, 12)


def test_a_vocabulary_too_small_for_bpe_is_refused():
    with pytest.raises(VocabularyMismatchError):
        spec.build(BPE_KIND, CORPUS, 16)


def test_no_held_out_sentence_reaches_the_tokenizer_corpus():
    sentences = [f"sentence number {index} is a small test." for index in range(40)]
    settings = TrainingSettings(min_validation=2, min_validation_groups=2, validation_share=0.2)

    train, held = corpus.split_training(sentences, settings)

    assert held
    assert not set(train) & set(held)
    held_groups = {corpus.group_of(sentence) for sentence in held}
    assert held_groups.isdisjoint({corpus.group_of(sentence) for sentence in train})


def test_a_bpe_training_records_that_it_read_only_the_training_side(tmp_path):
    sentences = [f"sentence number {index} is a small test." for index in range(40)]
    settings = TrainingSettings(
        steps=10,
        batch_size=4,
        min_steps=2,
        min_validation=2,
        min_validation_groups=2,
        validation_share=0.2,
        max_seconds=30,
    )
    config = {"d_model": 16, "layers": 1, "heads": 2, "context": 16, "tokenizer_kind": BPE_KIND}
    task = WordTask({**config, "vocabulary_size": 280})

    report = task.fit(sentences, settings=settings)

    assert report["trained"] is True
    train, held = corpus.split_training(sentences, settings)
    stored = artifacts.meta(task.tokenizer_record["id"])["corpus"]
    assert stored["tokenizer_sentences"] == len(train)
    assert stored["held_out_sentences"] == len(held)


def test_a_bpe_checkpoint_loads_and_is_refused_without_its_artifact(tmp_path):
    config = {"d_model": 16, "layers": 1, "heads": 2, "context": 16, "tokenizer_kind": BPE_KIND}
    task = WordTask({**config, "vocabulary_size": 280})
    settings = TrainingSettings(
        steps=10, batch_size=4, min_steps=2, min_validation=2, min_validation_groups=2
    )
    assert task.fit(CORPUS, settings=settings)["trained"] is True
    path = tmp_path / "words.pt"
    task.save(path)
    stored = checkpoint.load(path)

    assert stored is not None
    assert stored.schema == checkpoint.SCHEMA
    assert stored.tokenizer["kind"] == BPE_KIND
    loaded = WordTask.load(path)
    assert loaded is not None
    assert loaded.tokenizer is not None and loaded.tokenizer.kind == BPE_KIND

    artifacts.folder(stored.tokenizer["id"]).rename(tmp_path / "moved")
    assert WordTask.load(path) is None


def test_a_checkpoint_without_a_tokenizer_uses_the_word_adapter(tmp_path):
    vocabulary = ["<br>", "<end>", "<pad>", "<start>", "<unk>", "a", "animal", "cow"]
    model = TinyTransformer(len(vocabulary), {"d_model": 16, "layers": 1, "heads": 2, "context": 8})
    checkpoint.save(
        tmp_path / "old.pt",
        checkpoint.Checkpoint(
            task="words",
            model=model,
            vocabulary=vocabulary,
            config={"d_model": 16, "layers": 1, "heads": 2, "context": 8},
            metrics={},
        ),
    )

    task = WordTask.load(tmp_path / "old.pt")

    assert task is not None
    assert task.tokenizer is not None
    assert task.tokenizer.kind == WORD_KIND
    assert task.vocabulary == vocabulary


def test_a_written_piece_sentence_holds_no_special_mark():
    config = {"d_model": 16, "layers": 1, "heads": 2, "context": 16, "tokenizer_kind": BPE_KIND}
    task = WordTask({**config, "vocabulary_size": 280})
    settings = TrainingSettings(
        steps=20, batch_size=4, min_steps=2, min_validation=2, min_validation_groups=2
    )
    assert task.fit(CORPUS, settings=settings)["trained"] is True

    written = task.generate_sentence("the cow", max_words=8)

    if written:
        for mark in SPECIAL_NAMES:
            assert mark not in written


def test_a_bpe_model_writes_pieces_and_predicts_them():
    config = {"d_model": 24, "layers": 2, "heads": 2, "context": 32, "tokenizer_kind": BPE_KIND}
    task = WordTask({**config, "vocabulary_size": 320})
    settings = TrainingSettings(
        steps=150, batch_size=4, min_steps=2, min_validation=2, min_validation_groups=2
    )
    assert task.fit(CORPUS, settings=settings)["trained"] is True
    assert task.tokenizer is not None

    rows = task.predict_next("The cow", limit=3)
    written = task.generate_sentence("the cow", max_words=12)

    assert rows, "a trained piece model must offer a next piece"
    assert all(row["piece"] is True for row in rows)
    assert all(row["word"] not in SPECIAL_NAMES for row in rows)
    assert written, "a trained piece model must write past its seed"
    for mark in SPECIAL_NAMES:
        assert mark not in written
    assert task.score_words(["The", "cow", "gives", "milk"]) < 0.0


def test_the_page_reads_the_measured_tokenizer_record():
    config = {"d_model": 16, "layers": 1, "heads": 2, "context": 16, "tokenizer_kind": BPE_KIND}
    task = WordTask({**config, "vocabulary_size": 300})
    settings = TrainingSettings(
        steps=10, batch_size=4, min_steps=2, min_validation=2, min_validation_groups=2
    )
    assert task.fit(CORPUS, settings=settings)["trained"] is True

    seen = artifacts.describe(task.tokenizer_record)

    assert seen["kind"] == BPE_KIND
    assert seen["default_kind"] == WORD_KIND
    assert seen["present"] is True
    assert seen["artifact_bytes"] > 0
    assert seen["corpus"]["characters"] > 0
    assert seen["tokens_per_character"] is not None
    assert seen["short_id"] == seen["id"][:12]


def test_a_word_checkpoint_names_no_separate_artifact():
    task = WordTask(
        {
            "d_model": 16,
            "layers": 1,
            "heads": 2,
            "context": 16,
            "tokenizer_kind": WORD_KIND,
        }
    )
    settings = TrainingSettings(
        steps=10, batch_size=4, min_steps=2, min_validation=2, min_validation_groups=2
    )
    assert task.fit(CORPUS, settings=settings)["trained"] is True

    seen = artifacts.describe(task.tokenizer_record)

    assert seen["kind"] == WORD_KIND
    assert seen["present"] is True
    assert seen["artifact_bytes"] == 0
    assert seen["error"] is None


def test_the_canonical_vocabulary_reads_ids_through_its_own_tokenizer():
    tokenizer = _bpe()
    canonical = vocabulary.Vocabulary(
        vocabulary.CANONICAL_OWNER, tokenizer.vocabulary(), tokenizer=tokenizer
    )
    text = "The cow is an animal."

    assert canonical.kind == BPE_KIND
    assert canonical.encode(text) == tokenizer.encode(text)
    assert canonical.decode(canonical.encode(text)) == text


def test_the_startup_guard_checks_the_live_tokenizer():
    tokenizer = _bpe()
    good = vocabulary.Vocabulary(
        vocabulary.CANONICAL_OWNER, tokenizer.vocabulary(), tokenizer=tokenizer
    )

    assert vocabulary._tokenizer_problems(good) == []

    class Broken:
        kind = "bpe"

        def __init__(self) -> None:
            self.special_ids = {name: 9999 for name in SPECIAL_NAMES}

    broken = vocabulary.Vocabulary(
        vocabulary.CANONICAL_OWNER, tokenizer.vocabulary(), tokenizer=Broken()
    )

    assert vocabulary._tokenizer_problems(broken) != []
