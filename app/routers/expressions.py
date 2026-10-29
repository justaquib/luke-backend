"""HTTP routes for the conversation expressions the trainer teaches.

The routes stay thin: they check the caller and hand the work to the
expressions service, which owns the file, the validation and the matching.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.core.security import get_bearer_user, tenant_id_of
from app.db.database import User
from app.schemas.expressions import ExpressionBody
from app.services.language import expressions

router = APIRouter(prefix="/language-model/expressions", tags=["expressions"])


@router.get("")
def list_expressions(user: User = Depends(get_bearer_user)) -> dict[str, Any]:
    """The taught expressions, grouped by their category."""
    return expressions.view(tenant_id_of(user))


@router.post("", status_code=201)
def teach_expression(
    body: ExpressionBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Teach one opening, closing, pausing or resuming expression."""
    try:
        expression = expressions.add(tenant_id_of(user), body.category, body.phrase, body.reply)
    except expressions.ExpressionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return expression.as_dict()


@router.put("/{expression_id}")
def update_expression(
    expression_id: str,
    body: ExpressionBody,
    user: User = Depends(get_bearer_user),
) -> dict[str, Any]:
    """Change a taught expression in place."""
    try:
        expression = expressions.update(
            tenant_id_of(user), expression_id, body.category, body.phrase, body.reply
        )
    except expressions.ExpressionError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return expression.as_dict()


@router.delete("/{expression_id}")
def remove_expression(
    expression_id: str,
    user: User = Depends(get_bearer_user),
) -> dict[str, str]:
    """Forget one taught expression."""
    try:
        expressions.remove(tenant_id_of(user), expression_id)
    except expressions.ExpressionError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"removed": expression_id}
