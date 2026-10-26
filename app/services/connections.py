"""A dot the answer connected, and what it found there.

Every connection names a source the answer flow consulted and the outcome it
had, in the words of the measurement it made: a distance, a threshold, a
lesson. The answer pages show these records as they are produced; a page never
writes a connection of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Outcome = Literal["match", "miss", "pass", "fail", "skipped"]


@dataclass(frozen=True)
class Connection:
    """One source, what it returned, and the measured reason why."""

    source: str
    outcome: Outcome
    detail: str
    matched_question: str = ""

    def as_dict(self) -> dict[str, str]:
        """The shape the routes and the answer pages read."""
        return {
            "source": self.source,
            "outcome": self.outcome,
            "detail": self.detail,
            "matched_question": self.matched_question,
        }
