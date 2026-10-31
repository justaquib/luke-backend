"""HTTP routes for the tool system: listing, executing, permissions, and apps.

The router stays thin: it checks the caller, reads the request, and hands the
work to the tool service. Errors from the service are translated into HTTP
responses the frontend can show.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import User
from app.schemas.tools import (
    AppAllowlistBody,
    AppAllowlistEntry,
    AppOut,
    ToolExecuteRequest,
    ToolExecuteResponse,
    ToolOut,
    ToolPermissionOut,
    ToolPermissionsBody,
)
from app.services import tool_settings
from app.services.tools import (
    ToolExecutionFailedError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
    can_use,
    execute,
    list_tools,
    set_permission,
)
from app.services.tools.builtin.launch_app import list_installed_apps

router = APIRouter(prefix="/tools", tags=["tools"])


def _require_admin(user: User = Depends(get_bearer_user)) -> User:
    """Check that the caller is an admin."""
    if user.is_systemadmin or user.role in ("superadmin", "admin"):
        return user
    raise HTTPException(403, "Only admins can manage tool permissions")


@router.get("", response_model=list[ToolOut])
def tools_list(
    user: User = Depends(get_bearer_user),
) -> list[ToolOut]:
    """Every registered tool the tenant has permission to see."""
    tenant_id = tenant_id_of(user)
    return [
        ToolOut.model_validate(tool.to_dict())
        for tool in list_tools()
        if can_use(tool.name, tenant_id)
    ]


@router.post("/execute", response_model=ToolExecuteResponse)
def tools_execute(
    body: ToolExecuteRequest,
    user: User = Depends(get_bearer_user),
) -> ToolExecuteResponse:
    """Run a tool with the given parameters.

    If the tool requires confirmation and the caller has not given it, the
    response status is 'awaiting_confirmation' and the frontend shows a
    confirmation card. Otherwise the tool runs and the result is returned.
    """
    tenant_id = tenant_id_of(user)
    try:
        result = execute(body.tool, body.params, tenant_id, confirmed=body.confirmed)
    except ToolNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    except ToolPermissionDeniedError as error:
        raise HTTPException(403, str(error)) from error
    except ToolExecutionFailedError as error:
        raise HTTPException(400, str(error)) from error
    if result.requires_confirmation:
        return ToolExecuteResponse(
            status="awaiting_confirmation",
            tool=body.tool,
            params=body.params,
            output=None,
            error=None,
            preview=result.metadata,
        )
    return ToolExecuteResponse(
        status="succeeded" if result.error is None else "failed",
        tool=body.tool,
        output=result.output,
        error=result.error,
    )


@router.get("/permissions", response_model=list[ToolPermissionOut])
def tools_permissions(
    user: User = Depends(get_bearer_user),
) -> list[ToolPermissionOut]:
    """Every tool permission for this tenant."""
    from app.services.tools.permissions import all_permissions

    tenant_id = tenant_id_of(user)
    permissions = all_permissions(tenant_id)
    return [ToolPermissionOut(tool=tool, enabled=enabled) for tool, enabled in permissions.items()]


@router.put("/permissions", response_model=list[ToolPermissionOut])
def tools_set_permission(
    body: ToolPermissionsBody,
    user: User = Depends(_require_admin),
) -> list[ToolPermissionOut]:
    """Set whether a tool is enabled for this tenant. Admin only."""
    tenant_id = tenant_id_of(user)
    try:
        set_permission(body.tool, body.enabled, tenant_id)
    except ToolNotFoundError as error:
        raise HTTPException(404, str(error)) from error
    return tools_permissions(user)


@router.get("/apps", response_model=list[AppOut])
def tools_apps(
    user: User = Depends(get_bearer_user),
) -> list[AppOut]:
    """Every installed application on the OS."""
    return [AppOut.model_validate(app) for app in list_installed_apps()]


@router.get("/apps/allowlist", response_model=list[AppAllowlistEntry])
def tools_app_allowlist(
    user: User = Depends(get_bearer_user),
) -> list[AppAllowlistEntry]:
    """The apps the tenant has allowed Luke to launch."""
    tenant_id = tenant_id_of(user)
    return [
        AppAllowlistEntry.model_validate(entry) for entry in tool_settings.app_allowlist(tenant_id)
    ]


@router.put("/apps/allowlist", response_model=list[AppAllowlistEntry])
def tools_set_app_allowlist(
    body: AppAllowlistBody,
    user: User = Depends(_require_admin),
) -> list[AppAllowlistEntry]:
    """Set the app allowlist. Admin only."""
    tenant_id = tenant_id_of(user)
    entries = [entry.model_dump() for entry in body.apps]
    tool_settings.set_app_allowlist(entries, tenant_id)
    return body.apps
