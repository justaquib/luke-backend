"""Tool-calling system for Luke.

Tools are user-triggered actions: the user invokes a tool from the chat
composer, the tool runs, and the result is fed back to the LLM as conversation
context. The LLM does not decide to call tools; it reads their output.

Every tool is registered in the registry at import time. The executor checks
permissions, runs the tool, and returns a result. Write and launch operations
require explicit user confirmation before they execute.
"""

from app.services.tools.errors import (
    ToolExecutionFailed,
    ToolNotFound,
    ToolPermissionDenied,
    ToolRequiresConfirmation,
)
from app.services.tools.executor import execute
from app.services.tools.permissions import can_use, set_permission
from app.services.tools.registry import Tool, ToolRegistry, get_tool, list_tools, register
from app.services.tools.results import ToolResult

__all__ = [
    "Tool",
    "ToolExecutionFailed",
    "ToolNotFound",
    "ToolPermissionDenied",
    "ToolRegistry",
    "ToolRequiresConfirmation",
    "ToolResult",
    "can_use",
    "execute",
    "get_tool",
    "list_tools",
    "register",
    "set_permission",
]
