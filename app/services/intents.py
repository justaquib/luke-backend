"""Intent CRUD flows."""

from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import require_tenant_user
from app.db.database import Intent, User
from app.schemas import IntentCreate
from app.services import vector_store


def _tenant(user: User, action: str) -> int:
    return require_tenant_user(user, action)


def create_intent(user: User, req: IntentCreate, db: Session) -> Intent:
    """Create an intent for the current tenant."""
    tenant_id = _tenant(user, "create intents")
    intent = Intent(
        name=req.name.strip(),
        response_template=req.response_template.strip(),
        faq_id=req.faq_id,
        tenant_id=tenant_id,
    )
    db.add(intent)
    db.commit()
    db.refresh(intent)
    vector_store.upsert_intent(
        intent.id, intent.name, intent.response_template, intent.faq_id, tenant_id
    )
    return intent


def list_intents(user: User, db: Session) -> list[Intent]:
    """List intents visible to the current user."""
    statement = select(Intent).order_by(Intent.updated_at.desc())
    if not user.is_systemadmin:
        statement = statement.where(Intent.tenant_id == _tenant(user, "view intents"))
    return list(db.execute(statement).scalars().all())


def update_intent(user: User, intent_id: int, req: IntentCreate, db: Session) -> Intent:
    """Update an intent's wording and linked FAQ."""
    tenant_id = _tenant(user, "update intents")
    intent = db.get(Intent, intent_id)
    if intent is None or intent.tenant_id != tenant_id:
        raise HTTPException(404, "Intent not found")
    intent.name = req.name.strip()
    intent.response_template = req.response_template.strip()
    intent.faq_id = req.faq_id
    db.commit()
    db.refresh(intent)
    vector_store.upsert_intent(
        intent.id, intent.name, intent.response_template, intent.faq_id, tenant_id
    )
    return intent


def remove_intent(user: User, intent_id: int, db: Session) -> dict:
    """Delete an intent and its vector entry."""
    tenant_id = _tenant(user, "delete intents")
    intent = db.get(Intent, intent_id)
    if intent is None or intent.tenant_id != tenant_id:
        raise HTTPException(404, "Intent not found")
    db.delete(intent)
    db.commit()
    vector_store.delete_intent(intent_id)
    return {"deleted": intent_id}
