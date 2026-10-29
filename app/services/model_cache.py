"""A cache of loaded models, bounded by the bytes their files take.

A local install serves one workspace, but the trainer visits several in a
session. The models that were used last stay while their files fit the cap,
so a returning workspace is served without a reload and a burst of teachers
cannot quietly fill the memory.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from app.config import settings


def file_size(path: Path) -> int:
    """How many bytes a file takes, or zero when it is not there."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


class ByteBoundedCache:
    """Least-recently-used entries, kept while their sizes fit the cap.

    The entry asked for last is never evicted, even when it alone is larger
    than the cap: the model that serves the current request must stay.
    """

    def __init__(self, limit: int | None = None) -> None:
        self._limit = limit
        self._entries: OrderedDict[Any, tuple[float | None, Any, int]] = OrderedDict()

    def cap(self) -> int:
        """The size the cache may reach, from the settings unless fixed."""
        return settings.weights_cache_bytes if self._limit is None else self._limit

    def find(self, key: Any, stamp: float | None) -> Any | None:
        """The entry kept under a key, if it was built from the same files."""
        entry = self._entries.get(key)
        if entry is None or entry[0] != stamp:
            return None
        self._entries.move_to_end(key)
        return entry[1]

    def keep(self, key: Any, stamp: float | None, value: Any, size: int) -> None:
        """Keep an entry, dropping the least recently used ones that no longer fit."""
        self._entries.pop(key, None)
        self._entries[key] = (stamp, value, max(int(size), 0))
        self._evict()

    def drop(self, key: Any) -> None:
        self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()

    def bytes(self) -> int:
        """How many bytes the kept files take together."""
        return sum(entry[2] for entry in self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)

    def _evict(self) -> None:
        while len(self._entries) > 1 and self.bytes() > self.cap():
            self._entries.popitem(last=False)
