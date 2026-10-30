"""Pydantic schemas for the tools endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolOut(BaseModel):
    """One registered tool: its name, description, parameters, and confirmation flag."""

    name: str
    description: str
    parameters: dict[str, Any]
    requires_confirmation: bool


class ToolExecuteRequest(BaseModel):
    """A request to run a tool."""

    tool: str = Field(..., min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None
    confirmed: bool = False


class ToolExecuteResponse(BaseModel):
    """The result of running a tool."""

    status: str
    tool: str
    params: dict[str, Any] | None = None
    output: Any = None
    error: str | None = None
    preview: dict[str, Any] | None = None


class ToolPermissionOut(BaseModel):
    """Whether a tool is enabled for the tenant."""

    tool: str
    enabled: bool


class ToolPermissionsBody(BaseModel):
    """A request to set a tool's permission."""

    tool: str = Field(..., min_length=1)
    enabled: bool


class AppOut(BaseModel):
    """One installed application."""

    name: str
    path: str


class AppAllowlistBody(BaseModel):
    """A request to set the app allowlist."""

    apps: list[str] = Field(default_factory=list)
