"""The one workspace every record belongs to in Luke Lab.

The product runs for a single person, so there is one workspace. The
tenant-shaped columns stay in place, defaulted to this value, so a future
multi-user path does not need a schema rewrite; nothing outside this module
decides which workspace a request reads or writes.
"""

from __future__ import annotations

CANONICAL_WORKSPACE = 0
WORKSPACE_NAME = "Luke Lab"
WORKSPACE_SLUG = "luke-lab"


def current_workspace() -> int:
    """The workspace every read and write uses in this single-user build."""
    return CANONICAL_WORKSPACE
