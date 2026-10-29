"""Copy every data store to a timestamped folder before the migration.

The server runs PostgreSQL 16 and the installed ``pg_dump`` is 14, which
refuses a newer server, so the database backup is a full JSON row dump with
the same guarantee: every table, every row, and a hash of the file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, MetaData, func, select, text


def backup_directory(root: Path, prefix: str = "single-user") -> Path:
    """A new timestamped folder under ``backups`` that does not exist yet."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = root / "backups" / f"{prefix}-{stamp}"
    suffix = 1
    while target.exists():
        target = root / "backups" / f"single-user-{stamp}-{suffix}"
        suffix += 1
    return target


def sha256_of(path: Path) -> str:
    """The SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(source: Path, target: Path) -> dict:
    """Copy a file or folder and record its bytes, files and hashes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target)
        files = [path for path in target.rglob("*") if path.is_file()]
    elif source.is_file():
        shutil.copy2(source, target)
        files = [target]
    else:
        return {"source": str(source), "present": False}
    return {
        "source": str(source),
        "target": str(target),
        "present": True,
        "bytes": sum(path.stat().st_size for path in files),
        "files": len(files),
        "sha256": {str(path.relative_to(target)): sha256_of(path) for path in files},
    }


def dump_tables(engine: Engine, target: Path) -> dict[str, Any]:
    """Write every table and row to one JSON file, and record its hash."""
    metadata = MetaData()
    metadata.reflect(bind=engine)
    tables: dict[str, list[dict[str, Any]]] = {}
    with engine.connect() as connection:
        for table in metadata.sorted_tables:
            rows = connection.execute(select(table)).all()
            tables[table.name] = [dict(row._mapping) for row in rows]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "taken_at": datetime.now(UTC).isoformat(),
                "tables": tables,
            },
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "target": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_of(target),
        "rows": {name: len(rows) for name, rows in tables.items()},
    }


def _reset_sequence(connection: Any, table: Any) -> None:
    sequence = connection.execute(
        text("SELECT pg_get_serial_sequence(:table, 'id')"), {"table": table.name}
    ).scalar_one_or_none()
    if sequence is None:
        return
    highest = connection.execute(select(func.max(table.c.id))).scalar_one()
    if highest is not None:
        connection.execute(
            text("SELECT setval(CAST(:sequence AS regclass), :value)"),
            {"sequence": sequence, "value": highest},
        )


def restore_tables(engine: Engine, target: Path) -> dict[str, int]:
    """Replace every table with the rows in a backup file."""
    payload = json.loads(target.read_text(encoding="utf-8"))
    rows: dict[str, list[dict[str, Any]]] = payload["tables"]
    metadata = MetaData()
    metadata.reflect(bind=engine)
    with engine.begin() as connection:
        for table in reversed(metadata.sorted_tables):
            connection.execute(table.delete())
        for table in metadata.sorted_tables:
            stored = rows.get(table.name, [])
            if stored:
                connection.execute(table.insert(), stored)
        for table in metadata.sorted_tables:
            _reset_sequence(connection, table)
    return {name: len(rows.get(name, [])) for name in metadata.tables}


def copy_schema(source: Engine, target: Engine) -> list[str]:
    """Create the source database's tables in an empty target database."""
    metadata = MetaData()
    metadata.reflect(bind=source)
    metadata.create_all(target)
    return sorted(metadata.tables)
