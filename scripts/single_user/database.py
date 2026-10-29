"""Read the tenant-shaped tables, plan the remap, and apply it.

Every table keeps its ``tenant_id`` column; the migration only changes the
values to the one workspace. Rows that are merged or removed are recorded
first so the change can be undone.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Connection, MetaData, Table, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

WORKSPACE = 0
WORKSPACE_NAME = "Luke Lab"
WORKSPACE_SLUG = "luke-lab"

# Every table whose rows name a tenant. ``tenants`` is handled apart because
# it is the table the others point at.
TENANT_TABLES = (
    "documents",
    "uploaded_files",
    "faqs",
    "feedback",
    "pending_questions",
    "intents",
    "users",
    "invites",
    "rules",
    "website_crawls",
    "website_pages",
    "web_sources",
    "web_snapshots",
)


def reflect(connection: Connection) -> dict[str, Table]:
    """The live tables this migration knows, reflected once and keyed by name.

    A table the app has not created yet is simply absent, so a dry run can
    inspect a database from before a schema addition.
    """
    metadata = MetaData()
    metadata.reflect(bind=connection)
    known = {*TENANT_TABLES, "tenants"}
    return {name: table for name, table in metadata.tables.items() if name in known}


def _row(row: Any) -> dict[str, Any]:
    return {key: value for key, value in dict(row._mapping).items()}


def table_state(connection: Connection, table: Table) -> dict[str, Any]:
    """One table's total and the rows that do not already belong to workspace 0."""
    rows = connection.execute(select(table)).all()
    outside = [_row(row) for row in rows if row._mapping["tenant_id"] != WORKSPACE]
    return {
        "table": table.name,
        "total": len(rows),
        "outside": len(outside),
        "outside_ids": [row["id"] for row in outside],
    }


def _question_key(question: Any) -> str:
    return " ".join(str(question).split()).casefold()


def _keep_rank(row: dict[str, Any]) -> tuple:
    moment = row.get("updated_at") or row.get("created_at") or datetime.min
    return (row.get("status") == "approved", moment)


def faq_groups(connection: Connection, table: Table) -> list[dict[str, Any]]:
    """Saved answers that ask the same question, with the one to keep."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in connection.execute(select(table).order_by(table.c.id)).all():
        grouped.setdefault(_question_key(row._mapping["question"]), []).append(_row(row))
    groups = []
    for key, items in grouped.items():
        if len(items) < 2:
            continue
        keep = max(items, key=_keep_rank)
        dropped = [row for row in items if row["id"] != keep["id"]]
        groups.append({"question": keep["question"], "key": key, "keep": keep, "drop": dropped})
    return groups


def user_plan(connection: Connection, users: Table) -> dict[str, Any]:
    """Which login is kept, and which ones the single-user build removes."""
    rows = [_row(row) for row in connection.execute(select(users).order_by(users.c.id)).all()]
    admins = [row for row in rows if row.get("is_systemadmin")]
    keep = min(admins or rows, key=lambda row: row["id"]) if rows else None
    dropped = [row for row in rows if keep is not None and row["id"] != keep["id"]]
    return {"keep": keep, "drop": dropped}


def tenant_plan(connection: Connection, tenants: Table) -> dict[str, Any]:
    """The tenants that disappear once every row belongs to workspace 0."""
    rows = [_row(row) for row in connection.execute(select(tenants).order_by(tenants.c.id)).all()]
    return {
        "keep": [row for row in rows if row["id"] == WORKSPACE],
        "drop": [row for row in rows if row["id"] != WORKSPACE],
    }


def inspect(connection: Connection, tables: dict[str, Table]) -> dict[str, Any]:
    """Everything the migration would change, without changing it."""
    users = user_plan(connection, tables["users"])
    return {
        "tables": [
            table_state(connection, tables[name]) for name in TENANT_TABLES if name in tables
        ],
        "faqs": faq_groups(connection, tables["faqs"]),
        "users": users,
        "tenants": tenant_plan(connection, tables["tenants"]),
    }


def _ensure_workspace(connection: Connection, tenants: Table) -> None:
    statement = pg_insert(tenants).values(id=WORKSPACE, name=WORKSPACE_NAME, slug=WORKSPACE_SLUG)
    connection.execute(statement.on_conflict_do_nothing(index_elements=["id"]))


def _remap(connection: Connection, tables: dict[str, Table]) -> dict[str, Any]:
    undo: dict[str, Any] = {}
    for name in TENANT_TABLES:
        if name not in tables:
            continue
        table = tables[name]
        rows = connection.execute(select(table.c.id, table.c.tenant_id)).all()
        changed = {str(row.id): row.tenant_id for row in rows if row.tenant_id != WORKSPACE}
        if changed:
            ids = [int(identifier) for identifier in changed]
            connection.execute(update(table).where(table.c.id.in_(ids)).values(tenant_id=WORKSPACE))
        undo[name] = changed
    return undo


def _merge_faqs(
    connection: Connection, tables: dict[str, Table], groups: list[dict[str, Any]]
) -> dict[str, Any]:
    faqs = tables["faqs"]
    intents = tables["intents"]
    faq_map: dict[str, int] = {}
    moved: dict[str, int] = {}
    for group in groups:
        kept = group["keep"]["id"]
        for row in group["drop"]:
            faq_map[str(row["id"])] = kept
        dropped = [row["id"] for row in group["drop"]]
        pointers = connection.execute(
            select(intents.c.id, intents.c.faq_id).where(intents.c.faq_id.in_(dropped))
        ).all()
        moved.update({str(row.id): row.faq_id for row in pointers})
        connection.execute(update(intents).where(intents.c.faq_id.in_(dropped)).values(faq_id=kept))
        connection.execute(delete(faqs).where(faqs.c.id.in_(dropped)))
    return {"faq_map": faq_map, "moved_intents": moved}


def _remove_users(connection: Connection, tables: dict[str, Table], plan: dict[str, Any]) -> None:
    keep = plan["keep"]
    dropped = [row["id"] for row in plan["drop"]]
    if not dropped:
        return
    if keep is not None:
        invites = tables["invites"]
        connection.execute(
            update(invites).where(invites.c.created_by.in_(dropped)).values(created_by=keep["id"])
        )
    connection.execute(delete(tables["users"]).where(tables["users"].c.id.in_(dropped)))


def _remove_tenants(connection: Connection, tables: dict[str, Table], plan: dict[str, Any]) -> None:
    dropped = [row["id"] for row in plan["drop"]]
    if dropped:
        connection.execute(delete(tables["tenants"]).where(tables["tenants"].c.id.in_(dropped)))


def apply_plan(connection: Connection, tables: dict[str, Table], plan: dict[str, Any]) -> dict:
    """Move every row onto workspace 0 and return the record that undoes it."""
    _ensure_workspace(connection, tables["tenants"])
    tenant_columns = _remap(connection, tables)
    merged = _merge_faqs(connection, tables, plan["faqs"])
    _remove_users(connection, tables, plan["users"])
    _remove_tenants(connection, tables, plan["tenants"])
    return {
        "tenant_columns": tenant_columns,
        "faqs": [row for group in plan["faqs"] for row in group["drop"]],
        **merged,
        "users": plan["users"]["drop"],
        "tenants": plan["tenants"]["drop"],
    }


def counts(connection: Connection, tables: dict[str, Table]) -> dict[str, int]:
    """The row count of every migrated table, for the before/after check."""
    return {
        name: connection.execute(select(func.count()).select_from(tables[name])).scalar_one()
        for name in (*TENANT_TABLES, "tenants")
        if name in tables
    }


def restore(connection: Connection, undo: dict[str, Any], tables: dict[str, Table]) -> None:
    """Put the removed rows and old tenant ids back after a failed migration."""
    for row in undo.get("tenants", []):
        _insert_row(connection, tables["tenants"], row)
    for row in undo.get("users", []):
        _insert_row(connection, tables["users"], row)
    for row in undo.get("faqs", []):
        _insert_row(connection, tables["faqs"], row)
    intents = tables["intents"]
    for identifier, faq_id in undo.get("moved_intents", {}).items():
        statement = update(intents).where(intents.c.id == int(identifier))
        connection.execute(statement.values(faq_id=faq_id))
    for name, before in undo.get("tenant_columns", {}).items():
        table = tables[name]
        for identifier, tenant_id in before.items():
            connection.execute(
                update(table).where(table.c.id == int(identifier)).values(tenant_id=tenant_id)
            )


def _insert_row(connection: Connection, table: Table, row: dict[str, Any]) -> None:
    connection.execute(insert(table).values(**row))
