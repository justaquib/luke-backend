"""The tool registry: every tool Luke knows about.

Tools are registered at import time by the builtin modules. The registry is a
module-level singleton so every part of the app sees the same list.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.services.tools.errors import ToolNotFound


@dataclass(frozen=True)
class Tool:
    """One registered tool: its name, description, parameters, and handler."""

    name: str
    description: str
    parameters: dict[str, Any]
    requires_confirmation: bool
    handler: Callable[..., Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the API response."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_confirmation": self.requires_confirmation,
        }


class ToolRegistry:
    """The singleton list of registered tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool to the registry."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Look up a tool by name."""
        if name not in self._tools:
            raise ToolNotFound(f"Unknown tool: {name}")
        return self._tools[name]

    def list(self) -> list[Tool]:
        """Every registered tool, in registration order."""
        return list(self._tools.values())

    def __contains__(self, name: str) -> bool:
        return name in self._tools


_REGISTRY = ToolRegistry()


def register(tool: Tool) -> None:
    """Add a tool to the global registry."""
    _REGISTRY.register(tool)


def get_tool(name: str) -> Tool:
    """Look up a tool in the global registry."""
    return _REGISTRY.get(name)


def list_tools() -> list[Tool]:
    """Every tool in the global registry."""
    return _REGISTRY.list()
