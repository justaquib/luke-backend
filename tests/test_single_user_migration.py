"""The single-user migration plans the change without surprising anyone."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, create_engine

from scripts.single_user import database, files, state


def _engine():
    engine = create_engine("sqlite://")
    metadata = MetaData()
    Table(
        "faqs",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("question", String),
        Column("answer", String),
        Column("status", String),
        Column("tenant_id", Integer, nullable=True),
        Column("created_at", DateTime),
        Column("updated_at", DateTime, nullable=True),
    )
    Table(
        "users",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("email", String),
        Column("is_systemadmin", Integer),
        Column("tenant_id", Integer, nullable=True),
    )
    Table(
        "tenants",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("name", String),
        Column("slug", String),
    )
    metadata.create_all(engine)
    tables = {name: metadata.tables[name] for name in ("faqs", "users", "tenants")}
    return engine, tables


def _faq(identifier: int, question: str, status: str, tenant_id, moment: datetime) -> dict:
    return {
        "id": identifier,
        "question": question,
        "answer": f"answer {identifier}",
        "status": status,
        "tenant_id": tenant_id,
        "created_at": moment,
    }


def _insert(engine, tables, rows: list[dict]) -> None:
    with engine.begin() as connection:
        connection.execute(tables["faqs"].insert(), rows)


def test_duplicate_questions_keep_the_approved_answer_of_any_tenant():
    engine, tables = _engine()
    now = datetime(2026, 10, 27, tzinfo=UTC)
    _insert(
        engine,
        tables,
        [
            _faq(2, "What is rain?", "pending", 0, now),
            _faq(3, "  what   is RAIN? ", "approved", 1, now - timedelta(days=1)),
            _faq(4, "What is snow?", "approved", None, now),
        ],
    )

    with engine.connect() as connection:
        groups = database.faq_groups(connection, tables["faqs"])

    assert len(groups) == 1
    assert groups[0]["keep"]["id"] == 3
    assert [row["id"] for row in groups[0]["drop"]] == [2]


def test_the_oldest_system_admin_is_the_one_login_kept():
    engine, tables = _engine()
    with engine.begin() as connection:
        connection.execute(
            tables["users"].insert(),
            [
                {"id": 7, "email": "admin@example.com", "is_systemadmin": True, "tenant_id": 0},
                {"id": 9, "email": "second@example.com", "is_systemadmin": True, "tenant_id": 1},
                {"id": 12, "email": "user@example.com", "is_systemadmin": False, "tenant_id": 1},
            ],
        )

    with engine.connect() as connection:
        plan = database.user_plan(connection, tables["users"])

    assert plan["keep"]["email"] == "admin@example.com"
    assert [row["email"] for row in plan["drop"]] == ["second@example.com", "user@example.com"]


def test_every_tenant_except_workspace_zero_is_removed():
    engine, tables = _engine()
    with engine.begin() as connection:
        connection.execute(
            tables["tenants"].insert(),
            [
                {"id": 0, "name": "Luke Lab", "slug": "luke-lab"},
                {"id": 1, "name": "Old", "slug": "old"},
            ],
        )

    with engine.connect() as connection:
        plan = database.tenant_plan(connection, tables["tenants"])

    assert [row["id"] for row in plan["keep"]] == [0]
    assert [row["id"] for row in plan["drop"]] == [1]


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_learner_files_merge_by_their_words_not_their_ids(tmp_path: Path):
    _write(
        tmp_path / "tenant-0.json",
        {"lesson_cards": [{"id": "a", "question": "What is rain?", "answer": "Water."}]},
    )
    _write(
        tmp_path / "tenant-1.json",
        {
            "lesson_cards": [
                {"id": "b", "question": " what IS rain? ", "answer": "water."},
                {"id": "a", "question": "What is snow?", "answer": "Frozen water."},
            ]
        },
    )
    _write(tmp_path / "instruction-7.json", {"measured_at": "2026-10-01"})

    plan = state.plan_state(tmp_path)

    cards = plan["merged_state"]["lesson_cards"]
    assert len(cards) == 2
    assert plan["notes"][0]["action"] == "duplicate"
    assert plan["notes"][1]["action"] == "reassigned"
    assert cards[1]["id"] != "a"
    assert plan["changed"] is True
    assert plan["measurements"][0]["file"] == "instruction-7.json"


def test_a_file_with_nothing_new_is_not_rewritten(tmp_path: Path):
    _write(
        tmp_path / "tenant-0.json",
        {"lessons": [{"id": "a", "question": "What is rain?", "answer": "Water."}]},
    )
    _write(
        tmp_path / "tenant-1.json",
        {"lessons": [{"id": "z", "question": "What is rain?", "answer": "Water."}]},
    )

    plan = state.plan_state(tmp_path)

    assert plan["changed"] is False
    assert state.apply_state(plan) == {"written": False}


def test_uploads_move_into_workspace_zero_and_conflicts_stay_put(tmp_path: Path):
    workspace = tmp_path / "0"
    workspace.mkdir()
    (tmp_path / "loose.pdf").write_text("loose", encoding="utf-8")
    (tmp_path / "same.pdf").write_text("same", encoding="utf-8")
    (workspace / "same.pdf").write_text("same", encoding="utf-8")
    (tmp_path / "clash.pdf").write_text("new", encoding="utf-8")
    (workspace / "clash.pdf").write_text("old", encoding="utf-8")

    plan = files.plan(tmp_path)
    actions = {entry["source"]: entry["action"] for entry in plan["moves"]}
    assert actions == {"clash.pdf": "conflict", "loose.pdf": "move", "same.pdf": "same"}

    files.apply(plan, tmp_path)

    assert (workspace / "loose.pdf").read_text(encoding="utf-8") == "loose"
    assert not (tmp_path / "loose.pdf").exists()
    assert not (tmp_path / "same.pdf").exists()
    assert (tmp_path / "clash.pdf").read_text(encoding="utf-8") == "new"
    assert (workspace / "clash.pdf").read_text(encoding="utf-8") == "old"
