"""Errors the tool system can raise."""

from __future__ import annotations


class ToolNotFound(LookupError):
    """The requested tool is not registered."""


class ToolPermissionDenied(PermissionError):
    """The tenant does not have permission to use this tool."""


class ToolRequiresConfirmation(RuntimeError):
    """The tool needs explicit user confirmation before it runs."""


class ToolExecutionFailed(RuntimeError):
    """The tool handler raised an exception."""
