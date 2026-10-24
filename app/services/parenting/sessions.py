"""The coaching sessions of the parenting chat.

A session is the parent and the coach at the table: the turns they take, the
correction the parent gave, and whether the coach has to be handed to a human
trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4


@dataclass
class CoachingSession:
    """One conversation between a parent and the coaching coach."""

    id: str
    topic: str
    child_age_band: str
    status: str = "open"
    turns: int = 0
    corrections_used: int = 0
    scaffold_failures: int = 0
    last_message: str = ""
    last_correction: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class CoachingSessionStore:
    """The coaching sessions of the app, held while the chat runs."""

    def __init__(self) -> None:
        self._sessions: dict[str, CoachingSession] = {}

    def create(self, topic: str = "General", child_age_band: str = "") -> CoachingSession:
        """Start a coaching session on a topic and an age band."""
        session = CoachingSession(
            id=f"parent-{uuid4().hex[:12]}",
            topic=(topic or "General").strip() or "General",
            child_age_band=(child_age_band or "").strip(),
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> CoachingSession | None:
        """The session with this id, or None when it is not held."""
        return self._sessions.get(session_id)

    def save(self, session: CoachingSession) -> None:
        """Keep a session, and stamp the time of the change."""
        session.updated_at = datetime.now(UTC)
        self._sessions[session.id] = session

    def clear(self) -> None:
        """Close every coaching session."""
        self._sessions.clear()


_store = CoachingSessionStore()


def coaching_store() -> CoachingSessionStore:
    """The store the coaching sessions are held in."""
    return _store
