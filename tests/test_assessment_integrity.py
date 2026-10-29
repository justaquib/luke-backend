"""The held-out items stay out of everything the brain can learn from."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.database import Document, Faq
from app.services.assessment import bank, integrity
from app.services.language import corpus, training_materials, writing

CLEAN = "The little dog ran across the wet grass and into the tall grass again."


@pytest.fixture(scope="module")
def held_out() -> dict[str, Any]:
    return next(item for item in bank.items(split="eval") if item["skill"] == "grammar")


@pytest.fixture()
def session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database.Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    with factory() as db:
        yield db
    engine.dispose()


@pytest.fixture()
def only_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave the live lesson cards out, so the database is all that is scanned."""
    monkeypatch.setattr(integrity, "lesson_sources", lambda tenant_id=None: [])


def test_a_prompt_in_a_stored_passage_is_found(
    session: Session, held_out: dict[str, Any], only_stored: None
) -> None:
    session.add(
        Document(
            source="leak.txt",
            chunk_index=0,
            text=f"Notes.\n{held_out['prompt'].upper()}\nEnd.",
            tenant_id=0,
        )
    )
    session.commit()

    found = integrity.scan(session, 0)

    assert [(leak.item_id, leak.how, leak.where) for leak in found] == [
        (held_out["id"], "prompt", "document:1 leak.txt")
    ]


def test_an_item_id_in_a_stored_answer_is_found(
    session: Session, held_out: dict[str, Any], only_stored: None
) -> None:
    session.add(
        Faq(
            question="Where is this from?",
            answer=f"Copied from {held_out['id']}.",
            tenant_id=0,
        )
    )
    session.commit()

    found = integrity.scan(session, 0)

    assert [(leak.item_id, leak.how) for leak in found] == [(held_out["id"], "id")]


def test_clean_stored_material_has_no_leak(session: Session, only_stored: None) -> None:
    session.add_all(
        [
            Document(source="clean.txt", chunk_index=0, text=CLEAN, tenant_id=0),
            Faq(question="What did the dog do?", answer="It ran across the grass.", tenant_id=0),
        ]
    )
    session.commit()

    assert integrity.scan(session, 0) == []


def test_a_held_out_lesson_card_is_a_leak(
    held_out: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    card = {"id": "copied", "question": held_out["prompt"], "answer": "A copied answer."}
    monkeypatch.setattr("app.services.language.lessons.all_lesson_cards", lambda tenant_id: [card])
    monkeypatch.setattr(
        "app.services.language.runtime._read",
        lambda tenant_id: {"lessons": [], "word_cards": [], "passed_stages": []},
    )

    found = integrity.leaks(integrity.lesson_sources(0))

    assert [(leak.item_id, leak.how, leak.where) for leak in found] == [
        (held_out["id"], "prompt", "lesson:copied")
    ]


def test_the_training_loader_drops_a_held_out_document(
    held_out: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training_materials, "all_lesson_cards", lambda tenant_id: [])

    kept = training_materials.collect(
        0, documents=[("page.txt", f"Notes. {held_out['prompt']}", 0.9)]
    )

    assert all(held_out["prompt"] not in material.answer for material in kept)
    assert all(material.kind != training_materials.DOCUMENT for material in kept)


def test_the_training_set_keeps_no_held_out_sentence(held_out: dict[str, Any]) -> None:
    planted = training_materials.Material(
        training_materials.DOCUMENT, "page.txt", f"Notes. {held_out['prompt']}"
    )
    clean = training_materials.Material(training_materials.DOCUMENT, "page.txt", CLEAN)

    training = training_materials.training_set([planted, clean])

    assert training.sentences
    assert not any(integrity.is_held_out(sentence) for sentence in training.sentences)


def test_the_tokenizer_corpus_keeps_no_held_out_text(held_out: dict[str, Any]) -> None:
    from app.services.neural.tokenizer import corpus as tokenizer_corpus

    planted = training_materials.Material(
        training_materials.DOCUMENT, "page.txt", f"Notes. {held_out['prompt']}"
    )
    clean = training_materials.Material(training_materials.DOCUMENT, "page.txt", CLEAN)
    training = training_materials.training_set([planted, clean])

    passages = tokenizer_corpus.passages_of(training.sentences)

    assert passages
    assert not any(integrity.is_held_out(passage.text) for passage in passages)


def test_the_word_corpus_drops_a_held_out_lesson(
    held_out: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    card = {"kind": "lesson", "question": held_out["prompt"], "answer": "A copied answer."}
    monkeypatch.setattr(corpus, "all_lesson_cards", lambda tenant_id: [card])
    monkeypatch.setattr(corpus, "_repair_sentences", list)
    monkeypatch.setattr(corpus.runtime, "_read", lambda tenant_id: {"lessons": []})

    sentences = corpus.training_corpus(0)

    assert not any(integrity.is_held_out(sentence) for sentence in sentences)


def test_the_writing_corpus_drops_a_held_out_sentence(
    held_out: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(writing, "_taught_sentences", lambda tenant_id: [held_out["prompt"], CLEAN])
    monkeypatch.setattr(writing, "_taught_answers", lambda tenant_id: [])
    monkeypatch.setattr(writing, "_tenant_lesson_answers", lambda tenant_id: [])

    assert writing._sentence_corpus(0) == [CLEAN]


def test_the_train_half_holds_none_of_the_eval_half() -> None:
    held = bank.items(split="eval")
    train = bank.items(split="train")

    assert {item["id"] for item in held}.isdisjoint(item["id"] for item in train)
    assert {integrity.normalise(item["prompt"]) for item in held}.isdisjoint(
        integrity.normalise(item["prompt"]) for item in train
    )
