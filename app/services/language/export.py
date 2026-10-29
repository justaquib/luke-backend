"""The download endpoint's entry point.

The package itself lives in ``app/services/language/package/``; this module
keeps the two functions the routes and the older tests call, so the package
format can change without moving the download route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.services.language.package import bundle
from app.services.language.package.manifest import PackageError

# The routes and the older tests catch this name.
ExportError = PackageError


def preview(tenant_id: int | None, db: Session) -> dict[str, Any]:
    """What the package would hold, in both modes, without building it."""
    return bundle.preview(tenant_id, db)


def build(tenant_id: int | None, db: Session, mode: str = "full") -> tuple[Path, str]:
    """Write the package to a temporary zip and return it with its file name."""
    return bundle.build(tenant_id, db, mode)
