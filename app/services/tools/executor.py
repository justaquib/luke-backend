"""The tool executor: runs a tool after checking permissions.

The executor is the single entry point for running a tool. It checks that the
tenant has permission, that the tool exists, and that any required
confirmation has been given. It then calls the tool's handler and wraps the
result in a ToolResult.
"""

from __future__ import annotations

from typing import Any

from app.services.tools.errors import (
    ToolExecutionFailed,
    ToolPermissionDenied,
    ToolRequiresConfirmation,
)
from app.services.tools.permissions import can_use
from app.services.tools.registry import get_tool
from app.services.tools.results import ToolResult


def execute(
    tool_name: str,
    params: dict[str, Any],
    tenant_id: int | None,
    confirmed: bool = False,
) -> ToolResult:
    """Run the named tool with the given parameters.

    The executor checks permissions, looks up the tool, and calls its handler.
    If the tool requires confirmation and the caller has not given it, the
    executor returns a result with requires_confirmation=True instead of
    running the handler.
    """
    if not can_use(tool_name, tenant_id):
        raise ToolPermissionDenied(f"Tool not permitted: {tool_name}")
    tool = get_tool(tool_name)
    if tool.requires_confirmation and not confirmed:
        return ToolResult(
            output=None,
            error=None,
            requires_confirmation=True,
            metadata={"tool": tool_name, "params": params},
        )
    try:
        output = tool.handler(**params)
    except ToolRequiresConfirmation:
        raise
    except Exception as error:
        raise ToolExecutionFailed(f"Tool {tool_name} failed: {error}") from error
    return ToolResult(output=output)
