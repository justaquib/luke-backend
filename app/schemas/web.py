"""The web evidence settings the lab page reads and writes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WebSettingsBody(BaseModel):
    """What the trainer may change about reading the web."""

    enabled: bool = False
    provider: str = "searxng"
    endpoint: str = ""
    allowlist: list[str] = Field(default_factory=list)
    ttl_hours: int = 168
    min_score: float = 0.45
    results: int = 3


class WebSourceBody(BaseModel):
    """One page the trainer wants to allowlist and refresh."""

    url: str
