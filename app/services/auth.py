"""Authentication flows for Google sign-in, invitations, and system admin."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi import HTTPException
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.security import create_access_token, require_admin
from app.db.database import Invite, Tenant, User
from app.schemas import (
    AcceptInviteRequest,
    GoogleAuthRequest,
    InviteCreate,
    SystemAdminLogin,
    TokenResponse,
)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _google_payload(token: str) -> dict:
    try:
        return id_token.verify_token(token, google_requests.Request(), audience=None)
    except ValueError as error:
        raise HTTPException(401, "Invalid Google ID token") from error


def _identity(payload: dict) -> tuple[str, str, str | None, str | None]:
    google_id = payload.get("sub")
    email = payload.get("email")
    if not google_id or not email:
        raise HTTPException(401, "Invalid Google ID token payload")
    return google_id, email, payload.get("name") or email, payload.get("picture")


def _find_user(db: Session, google_id: str, email: str) -> User | None:
    user = db.execute(select(User).where(User.google_id == google_id)).scalar_one_or_none()
    if user is None:
        user = db.execute(select(User).where(User.email == email)).scalar_one_or_none()
    return user


def _token_for(user: User) -> TokenResponse:
    token = create_access_token(
        {
            "sub": str(user.id),
            "email": user.email,
            "tenant_id": user.tenant_id,
            "role": user.role,
            "is_systemadmin": user.is_systemadmin,
        }
    )
    return TokenResponse(access_token=token)


def _invite_for_token(db: Session, token: str) -> Invite:
    invite = db.execute(select(Invite).where(Invite.token == token)).scalar_one_or_none()
    if invite is None:
        raise HTTPException(404, "Invite not found")
    if invite.used_at is not None:
        raise HTTPException(400, "Invite already used")
    if _now() > invite.expires_at:
        raise HTTPException(400, "Invite expired")
    return invite


def google_login(req: GoogleAuthRequest, db: Session) -> TokenResponse:
    """Sign in with Google, consuming an invite when one is supplied."""
    payload = _google_payload(req.id_token)
    identity = _identity(payload)
    google_id, email, name, picture = identity
    user = _find_user(db, google_id, email)
    invite = _invite_for_token(db, req.invite_token) if req.invite_token else None
    if user is not None:
        user.name = name
        user.picture = picture
        if invite is not None and not user.tenant_id:
            user.tenant_id = invite.tenant_id
            user.role = invite.role
            invite.used_at = _now()
    else:
        user = _create_google_user(db, identity, invite)
    db.commit()
    db.refresh(user)
    return _token_for(user)


def _create_google_user(
    db: Session,
    identity: tuple[str, str, str | None, str | None],
    invite: Invite | None,
) -> User:
    google_id, email, name, picture = identity
    if invite is not None:
        user = User(
            google_id=google_id,
            email=email,
            name=name,
            picture=picture,
            tenant_id=invite.tenant_id,
            role=invite.role,
        )
        invite.used_at = _now()
    else:
        tenant = Tenant(name=name or email, slug=email.partition("@")[0])
        db.add(tenant)
        db.flush()
        user = User(
            google_id=google_id,
            email=email,
            name=name,
            picture=picture,
            tenant_id=tenant.id,
            role="superadmin",
        )
    db.add(user)
    return user


def systemadmin_status(db: Session) -> dict:
    """Report whether a system administrator exists and which email it uses."""
    user = db.execute(select(User).where(User.is_systemadmin.is_(True))).scalar_one_or_none()
    if user is None:
        return {"configured": False}
    return {"configured": True, "email": user.email}


def systemadmin_login(req: SystemAdminLogin, db: Session) -> TokenResponse:
    """Sign in the system administrator with the configured PIN."""
    if not settings.systemadmin_pin:
        raise HTTPException(401, "System admin PIN not configured")
    user = db.execute(select(User).where(User.is_systemadmin.is_(True))).scalar_one_or_none()
    if user is None:
        raise HTTPException(401, "System admin not configured")
    if not req.pin or not bcrypt.checkpw(
        req.pin.encode("utf-8"), user.pin_hash.encode("utf-8")
    ):
        raise HTTPException(401, "Invalid PIN")
    return _token_for(user)


def refresh_session(user: User) -> TokenResponse:
    """Give a signed-in user a new token, so the session keeps going.

    The token is short-lived enough that a user who opens the app every day
    should not have to sign in again every day. A session that is still good
    is renewed here.
    """
    return _token_for(user)


def accept_invite(req: AcceptInviteRequest, invite_token: str, db: Session) -> TokenResponse:
    """Accept an invitation using a Google identity."""
    invite = _invite_for_token(db, invite_token)
    payload = _google_payload(req.id_token)
    google_id, email, name, picture = _identity(payload)
    user = _find_user(db, google_id, email)
    if user is not None:
        user.name = name
        user.picture = picture
        if not user.tenant_id:
            user.tenant_id = invite.tenant_id
            user.role = invite.role
    else:
        user = User(
            google_id=google_id,
            email=email,
            name=name,
            picture=picture,
            tenant_id=invite.tenant_id,
            role=invite.role,
        )
        db.add(user)
    invite.used_at = _now()
    db.commit()
    db.refresh(user)
    return _token_for(user)


def create_invite(user: User, req: InviteCreate, db: Session) -> Invite:
    """Create an invitation for the current administrator's tenant."""
    tenant_id = require_admin(user, "create invites")
    invite = Invite(
        tenant_id=tenant_id,
        email=req.email,
        token=secrets.token_urlsafe(32),
        role=req.role,
        created_by=user.id,
        expires_at=_now() + timedelta(hours=settings.invite_expire_hours),
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    return invite


def list_invites(user: User, db: Session) -> list[Invite]:
    """List invitations for the current administrator's tenant."""
    tenant_id = require_admin(user, "view invites")
    return list(
        db.execute(
            select(Invite)
            .where(Invite.tenant_id == tenant_id)
            .order_by(Invite.created_at.desc())
        )
        .scalars()
        .all()
    )


def list_tenants(user: User, db: Session) -> list[Tenant]:
    """List every tenant for the system administrator."""
    if not user.is_systemadmin:
        raise HTTPException(403, "Only system admin can list tenants")
    return list(db.execute(select(Tenant).order_by(Tenant.created_at.desc())).scalars().all())
