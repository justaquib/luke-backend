"""What Luke's foundation is, measured from the files on disk.

The page reads this route instead of a hard-coded name: whether the pinned
checkpoint is installed, and the identity the loader will verify before it
answers with it.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from app.core.security import get_bearer_user
from app.db.database import User
from app.services.foundation import identity, runtime

router = APIRouter(prefix="/language-model/foundation", tags=["foundation"])


@router.get("")
def foundation_status(user: User = Depends(get_bearer_user)) -> dict[str, Any]:
    """The installed checkpoint, its identity and whether it is loaded."""
    status = identity.status()
    return {
        **status,
        "loaded": runtime.loaded(),
        "device": runtime.loaded_device(),
        "memory_bytes": runtime.memory_bytes() if runtime.loaded() else None,
    }
