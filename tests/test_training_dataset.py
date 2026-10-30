"""The Phase 7 dataset: provenance on every example and a split taken by source."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_bearer_user
from app.db import database
from app.db.database import Document, Faq, User, get_session
from app.routers import huggingface as huggingface_router
from app.services.language.training_materials import (
    APPROVED,
    CONVERSATION,
    DOCUMENT,
    LESSON,
    SENTENCE,
    WORD,
    Material,
)
from app.services.training import dataset, splits

DOC_A = "alpha.pdf"
DOC_B = "beta.pdf"
DOC_C = "gamma.pdf"
PASSAGE = "Water the soil twice a week and keep it out of direct sun."
OTHER = "Volcanoes open where the crust is thin. Lava cools into dark rock."


def document(source: str, text: str) -> Material:
    return Material(DOCUMENT, source, text, quality=1.0)


def lesson(question: str, answer: str) -> Material:
    return Material(LESSON, question, answer)


def materials() -> list[Material]:
    """Two documents with more than one chunk each, and the taught material."""
    return [
        Material(DOCUMENT, DOC_A, f"{PASSAGE} {OTHER}", quality=1.0),
        Material(DOCUMENT, DOC_A, OTHER, quality=1.0),
        document(DOC_B, OTHER),
        lesson("What does a plant need?", "A plant needs light to make its food."),
        Material(APPROVED, "Why is the sky blue?", "Air scatters blue light."),
        Material(CONVERSATION, "hello", "Hello. I am Luke, your study partner."),
        Material(WORD, "volcano", "an opening in the crust"),
    ]


def test_a_whole_document_sits_on_one_side_of_the_split() -> None:
    built = dataset.build(materials())
    tags: dict[str, set[str]] = {}
    for example in built.examples:
        tags.setdefault(example.source, set()).add(example.split)
    assert all(len(side) == 1 for side in tags.values())


def test_the_split_holds_a_source_out_instead_of_a_sentence() -> None:
    built = dataset.build(materials(), share=0.5)
    held = {e.source for e in built.examples if e.split == splits.HELD_OUT}
    trained = {e.source for e in built.examples if e.split == splits.TRAIN}
    assert held and trained
    assert held.isdisjoint(trained)


def test_a_held_out_source_gives_the_training_side_nothing() -> None:
    built = dataset.build(materials(), share=0.5)
    held = {source for source, tag in built.assignment.items() if tag == splits.HELD_OUT}
    trained = {e.source for e in built.examples if e.split == splits.TRAIN}
    assert held.isdisjoint(trained)


def test_a_document_becomes_passages_to_read_and_never_a_question() -> None:
    built = dataset.build([document(DOC_A, f"{PASSAGE} {OTHER}")])
    assert {example.task for example in built.examples} == {dataset.PRETRAIN}
    assert all(example.prompt == "" for example in built.examples)
    assert built.counts()["sft_train"] + built.counts()["sft_held_out"] == 0


def test_a_taught_answer_becomes_a_pair_with_its_question() -> None:
    built = dataset.build([lesson("What does a plant need?", "Light.")])
    example = built.examples[0]
    assert (example.task, example.prompt, example.target) == (
        dataset.SFT,
        "What does a plant need?",
        "Light.",
    )
    assert example.source == "lesson:What does a plant need?"


def test_the_provenance_names_the_unit_the_example_came_from() -> None:
    built = dataset.build(materials())
    assert any(example.source == f"document:{DOC_A}" for example in built.examples)
    assert any(example.source == dataset.CONVERSATION_SOURCE for example in built.examples)


def test_the_taught_voice_is_one_source_and_never_split() -> None:
    built = dataset.build(
        [
            Material(CONVERSATION, "hello", "Hello. I am Luke."),
            Material(CONVERSATION, "thanks", "You are welcome."),
        ]
    )
    assert built.sources == (dataset.CONVERSATION_SOURCE,)


def test_a_words_card_and_a_written_sentence_are_not_examples_of_either_task() -> None:
    built = dataset.build(
        [
            Material(WORD, "volcano", "an opening in the crust"),
            Material(SENTENCE, "", "The volcano opened over the ridge."),
        ]
    )
    assert built.examples == ()
    assert built.assignment == {}


def test_the_split_does_not_move_when_a_document_is_added() -> None:
    before = dataset.build(materials(), share=0.5)
    after = dataset.build([*materials(), document(DOC_C, OTHER)], share=0.5)
    for source, tag in before.assignment.items():
        assert after.assignment[source] == tag
    assert after.assignment[f"document:{DOC_C}"] in splits.SPLITS


def test_the_same_salt_always_gives_the_same_tags_and_another_salt_does_not() -> None:
    sources = [f"document:doc{i}.pdf" for i in range(40)]
    assert splits.assign(sources) == splits.assign(list(reversed(sources)))
    assert splits.assign(sources, salt="luke-training-v2") != splits.assign(sources)


def test_both_sides_survive_a_share_that_asks_for_all_or_nothing() -> None:
    many = splits.assign([f"document:doc{i}.pdf" for i in range(30)], share=0.999)
    assert splits.summary(many)["train_sources"] == 1
    few = splits.assign(["document:a.pdf", "document:b.pdf"], share=0.001)
    assert splits.summary(few)["held_out_sources"] == 1


def test_a_share_outside_the_split_is_refused() -> None:
    with pytest.raises(ValueError):
        splits.assign(["a", "b"], share=0.0)
    with pytest.raises(ValueError):
        splits.assign(["a", "b"], share=1.5)


def test_a_split_is_not_taken_until_there_are_two_sources() -> None:
    built = dataset.build([document(DOC_A, PASSAGE)])
    assert not built.taken()
    assert built.report()["split_taken"] is False
    assert dataset.build(materials(), share=0.5).taken()


def test_the_same_words_under_two_labels_are_kept_on_one_side() -> None:
    built = dataset.build([document(DOC_A, PASSAGE), lesson("How often?", PASSAGE)])
    tags = {source: built.assignment[source] for source in built.assignment}
    assert len(set(tags.values())) == 1
    assert dataset.leakage(built) == ()
    assert built.taken() is False


def test_a_repeated_answer_across_cards_cannot_be_trained_and_measured() -> None:
    built = dataset.build(
        [
            lesson("How often is soil watered?", PASSAGE),
            lesson("When does a gardener water?", PASSAGE),
            Material(APPROVED, "Why is the sky blue?", "Air scatters blue light."),
            Material(
                APPROVED, "Why is grass green?", "Light reaches the leaf and comes back green."
            ),
        ]
    )
    assert dataset.leakage(built) == ()
    for example in built.examples:
        assert example.split == built.assignment[example.source]
    repeated = {example.source for example in built.examples if example.target == PASSAGE}
    assert len(repeated) == 2
    assert len({built.assignment[source] for source in repeated}) == 1


def test_a_document_that_only_repeats_its_own_passage_is_not_a_leak() -> None:
    built = dataset.build(
        [document(DOC_A, PASSAGE), lesson("What is lava?", "Lava cooled into dark rock.")]
    )
    assert dataset.leakage(built) == ()


def test_the_report_counts_the_sides_it_measured() -> None:
    built = dataset.build(materials(), share=0.5)
    report = built.report()
    assert report["examples"] == len(built.examples)
    assert report["sources"] == len(built.assignment)
    assert sum(report["counts"].values()) == len(built.examples)
    assert report["salt"] == splits.DEFAULT_SALT
    assert report["passage_sources"] + report["answer_sources"] <= report["sources"]


@pytest.fixture()
def files_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the learner's own files out, so the database is all the dataset sees."""
    from app.services import brain

    monkeypatch.setattr(brain, "load_feedback", lambda tenant_id: [])
    for name in ("_lesson_materials", "_conversation_materials", "_word_materials"):
        monkeypatch.setattr(dataset.training_materials, name, lambda tenant_id=None: [])
    monkeypatch.setattr(
        dataset.training_materials, "_sentence_materials", lambda tenant_id=None: []
    )


@pytest.fixture()
def workspace(files_are_empty: None) -> Iterator[Session]:
    """A database with two documents and one approved answer."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database.Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, autoflush=False)() as db:
        db.add_all(
            [
                Document(source=DOC_A, chunk_index=0, text=PASSAGE, tenant_id=3, quality=1.0),
                Document(source=DOC_A, chunk_index=1, text=OTHER, tenant_id=3, quality=1.0),
                Document(source=DOC_B, chunk_index=0, text=OTHER, tenant_id=3, quality=1.0),
                Faq(
                    question="Why is the sky blue?",
                    answer="Air scatters blue light.",
                    status="approved",
                    tenant_id=3,
                ),
            ]
        )
        db.commit()
        yield db
    engine.dispose()


def test_the_workspace_reads_into_examples_tagged_by_source(workspace: Session) -> None:
    built = dataset.build(dataset.corpus(3, workspace), share=0.5)
    assert set(built.assignment) == {
        f"{DOCUMENT}:{DOC_A}",
        f"{DOCUMENT}:{DOC_B}",
        f"{APPROVED}:Why is the sky blue?",
    }
    assert all(
        example.source.startswith((f"{DOCUMENT}:", f"{APPROVED}:")) for example in built.examples
    )
    assert built.counts()["pretrain_train"] + built.counts()["pretrain_held_out"] == 3
    assert built.counts()["sft_train"] + built.counts()["sft_held_out"] == 1
    assert dataset.leakage(built) == ()


def test_the_workspace_prepares_the_dataset_a_run_is_given(workspace: Session) -> None:
    built = dataset.prepared(3, workspace)
    assert built.sources == (
        f"{APPROVED}:Why is the sky blue?",
        f"{DOCUMENT}:{DOC_A}",
        f"{DOCUMENT}:{DOC_B}",
    )
    assert built.taken() is True
    assert dataset.leakage(built) == ()


def test_the_dataset_route_reports_the_split_and_no_leak(
    workspace: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(huggingface_router, "tenant_id_of", lambda user: 3)
    app = FastAPI()
    app.include_router(huggingface_router.router)
    app.dependency_overrides[get_bearer_user] = lambda: User(
        id=1,
        email="trainer@example.com",
        name="Trainer",
        is_systemadmin=False,
        tenant_id=3,
    )
    app.dependency_overrides[get_session] = lambda: workspace
    with TestClient(app) as client:
        response = client.get("/language-model/huggingface/dataset")
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == 3
    assert body["split_taken"] is True
    assert body["counts"]["pretrain_train"] + body["counts"]["pretrain_held_out"] == 3
    assert body["leakage"] == []


def test_a_held_out_document_takes_every_one_of_its_passages_with_it(workspace: Session) -> None:
    built = dataset.build(dataset.corpus(3, workspace), share=0.5)
    per_side = Counter(
        (example.source, example.split)
        for example in built.examples
        if example.task == dataset.PRETRAIN
    )
    alpha = {tag: per_side[(f"{DOCUMENT}:{DOC_A}", tag)] for tag in splits.SPLITS}
    assert alpha[splits.HELD_OUT] + alpha[splits.TRAIN] == 2
    assert 0 in alpha.values()
