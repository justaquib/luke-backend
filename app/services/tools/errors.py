"""Errors the tool system can raise."""

from __future__ import annotations


class ToolNotFoundError(LookupError):
    """The requested tool is not registered."""


class ToolPermissionDeniedError(PermissionError):
    """The tenant does not have permission to use this tool."""


class ToolRequiresConfirmationError(RuntimeError):
    """The tool needs explicit user confirmation before it runs."""


class ToolExecutionFailedError(RuntimeError):
    """The tool handler raised an exception."""
