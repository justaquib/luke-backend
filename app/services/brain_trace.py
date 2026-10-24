"""The measured reading behind one chat answer.

The chat answer itself may come from a lesson, a document or the writer. This
service only measures the writer: it reads the same question and reports its
real tokens, attention and choices, so a page can show the model's work
without pretending that every answer was generated.
"""

from __future__ import annotations

from typing import Any

from app.services import brain
from app.services.neural import tracing


def answer_trace(tenant_id: int | None, question: str) -> dict[str, Any]:
    """Trace the writer reading the question the assistant answered."""
    return tracing.trace(brain.load_writer(tenant_id), question.strip())
