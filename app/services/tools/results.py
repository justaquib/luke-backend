"""The result of running a tool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """What a tool returns after it runs.

    The output is any JSON-serializable value. The error is a human-readable
    message when the tool failed. The requires_confirmation flag is set when
    the tool needs the user to approve before running.
    """

    output: Any = None
    error: str | None = None
    requires_confirmation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the API response."""
        return {
            "output": self.output,
            "error": self.error,
            "requires_confirmation": self.requires_confirmation,
            "metadata": self.metadata,
        }
