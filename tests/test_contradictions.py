"""A second answer to the same question waits for the trainer, not the first one's place."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import database
from app.db.database import Faq, PendingQuestion, Tenant, User
from app.schemas import FaqCreate, FaqUpdate
from app.services import faq, vector_store

WORKSPACE = 0


@pytest.fixture()
def db(monkeypatch: pytest.MonkeyPatch) -> Session:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    database.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False)()
    session.add_all(
        [
            Tenant(id=WORKSPACE, name="Luke Lab", slug="luke-lab"),
            Tenant(id=1, name="Other", slug="other"),
        ]
    )
    session.commit()
    for name in ("upsert_faq", "delete_pending_index", "delete_faq", "index_pending"):
        monkeypatch.setattr(vector_store, name, lambda *args, **kwargs: None)
    yield session
    session.close()


def _user() -> User:
    return User(id=1, email="person@example.com", name="Person", tenant_id=WORKSPACE)


def _approved(db: Session, question: str, answer: str, tenant_id: int = WORKSPACE) -> Faq:
    row = Faq(question=question, answer=answer, status="approved", tenant_id=tenant_id)
    db.add(row)
    db.commit()
    return row


def _inbox(db: Session) -> list[PendingQuestion]:
    return list(db.execute(select(PendingQuestion)).scalars())


def test_a_different_answer_to_the_same_question_reaches_the_inbox(db: Session) -> None:
    _approved(db, "What is rain?", "Water falling from clouds.")

    faq.create_faq(_user(), FaqCreate(question="what is rain", answer="Snow falls."), db)

    entry = db.execute(select(PendingQuestion)).scalar_one()
    assert entry.question == "what is rain"
    assert entry.note is not None
    assert "Snow falls." in entry.note
    assert "Water falling from clouds." in entry.note
    assert entry.occurrences == 1


def test_a_rephrased_answer_is_not_a_conflict(db: Session) -> None:
    _approved(db, "What is rain?", "Water falls from clouds.")

    faq.create_faq(
        _user(), FaqCreate(question="What is rain?", answer="Water falls from the clouds."), db
    )

    assert _inbox(db) == []


def test_a_different_question_is_not_a_conflict(db: Session) -> None:
    _approved(db, "What is rain?", "Water falling from clouds.")

    faq.create_faq(_user(), FaqCreate(question="What is snow?", answer="Frozen water."), db)

    assert _inbox(db) == []


def test_the_same_conflict_counts_up_instead_of_filling_the_inbox(db: Session) -> None:
    _approved(db, "What is rain?", "Water falling from clouds.")
    request = FaqCreate(question="What is rain?", answer="Snow falls.")

    faq.create_faq(_user(), request, db)
    faq.create_faq(_user(), request, db)

    entry = db.execute(select(PendingQuestion)).scalar_one()
    assert entry.occurrences == 2


def test_answering_a_pending_question_flags_its_other_answer(db: Session) -> None:
    _approved(db, "What is rain?", "Water falling from clouds.")
    pending = PendingQuestion(question="what is rain?", tenant_id=WORKSPACE)
    db.add(pending)
    db.commit()

    faq.answer_pending(_user(), pending.id, FaqUpdate(answer="Snow falls."), db)

    assert db.get(PendingQuestion, pending.id).resolved is True
    entry = db.execute(
        select(PendingQuestion).where(PendingQuestion.note.is_not(None))
    ).scalar_one()
    assert "Snow falls." in (entry.note or "")


def test_the_first_answer_to_a_question_is_not_a_conflict(db: Session) -> None:
    faq.create_faq(_user(), FaqCreate(question="What is rain?", answer="Water falls."), db)

    assert _inbox(db) == []


def test_another_workspace_cannot_answer_its_way_into_this_inbox(db: Session) -> None:
    _approved(db, "What is rain?", "Water falling from clouds.", tenant_id=1)

    faq.create_faq(_user(), FaqCreate(question="What is rain?", answer="Snow falls."), db)

    assert _inbox(db) == []


def test_the_note_column_migration_adds_and_drops_once() -> None:
    from sqlalchemy import create_engine, text

    from scripts.migrate_pending_note import add, drop, plan

    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE pending_questions (id INTEGER PRIMARY KEY)"))

    assert "note" in plan(engine)["missing"]
    assert add(engine) == ["note"]
    assert add(engine) == []
    assert plan(engine)["missing"] == []
    assert drop(engine) == ["note"]
    assert drop(engine) == []
