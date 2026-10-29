"""Request body for the conversation expressions the trainer teaches."""

from __future__ import annotations

from pydantic import BaseModel


class ExpressionBody(BaseModel):
    """One taught phrase and the reply Luke gives it."""

    category: str
    phrase: str
    reply: str
