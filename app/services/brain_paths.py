"""Where the trained models of a tenant are kept.

They live beside the learner's own file, so a tenant's models are found next
to its own lessons and the two can be kept together. The paths are gathered
here so every part of the brain writes and reads in the same place.
"""

from __future__ import annotations

from pathlib import Path


def brain_dir() -> Path:
    """The folder the trained models are kept in."""
    from app.services.language import runtime as language_runtime

    return language_runtime.data_dir()


def tenant_key(tenant_id: int | None) -> int:
    """The number a tenant is written under."""
    return tenant_id if tenant_id is not None else 0


def brain_path(tenant_id: int | None, name: str) -> Path:
    """Where a trained head is kept. The old numpy files are not read."""
    return brain_dir() / f"brain-{tenant_key(tenant_id)}-{name}.pt"
