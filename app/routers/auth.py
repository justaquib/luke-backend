"""Authentication and administration routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from app.core.security import get_bearer_user
from app.db.database import get_session
from app.schemas import (
    AcceptInviteRequest,
    GoogleAuthRequest,
    InviteCreate,
    InviteOut,
    SystemAdminLogin,
    TokenResponse,
    UserOut,
)
from app.services import auth

router = APIRouter(tags=["auth"])


@router.post("/auth/google", response_model=TokenResponse)
def auth_google(req: GoogleAuthRequest, db: Session = Depends(get_session)) -> TokenResponse:
    return auth.google_login(req, db)


@router.get("/auth/systemadmin/status")
def systemadmin_status(db: Session = Depends(get_session)) -> dict:
    return auth.systemadmin_status(db)


@router.post("/auth/systemadmin", response_model=TokenResponse)
def auth_systemadmin(req: SystemAdminLogin, db: Session = Depends(get_session)) -> TokenResponse:
    return auth.systemadmin_login(req, db)


@router.post("/auth/accept-invite", response_model=TokenResponse)
def auth_accept_invite(
    req: AcceptInviteRequest, invite_token: str, db: Session = Depends(get_session)
) -> TokenResponse:
    return auth.accept_invite(req, invite_token, db)


@router.post("/auth/invites", response_model=InviteOut)
def create_invite(
    req: InviteCreate,
    authorization: str = Header(""),
    db: Session = Depends(get_session),
) -> InviteOut:
    return auth.create_invite(get_bearer_user(authorization, db), req, db)


@router.get("/auth/invites", response_model=list[InviteOut])
def list_invites(
    authorization: str = Header(""), db: Session = Depends(get_session)
) -> list[InviteOut]:
    return auth.list_invites(get_bearer_user(authorization, db), db)


@router.get("/auth/me", response_model=UserOut)
def auth_me(authorization: str = Header(""), db: Session = Depends(get_session)) -> UserOut:
    return get_bearer_user(authorization, db)


@router.post("/auth/refresh", response_model=TokenResponse)
def auth_refresh(
    authorization: str = Header(""), db: Session = Depends(get_session)
) -> TokenResponse:
    """Renew the session of the signed-in user, so they do not sign in again."""
    return auth.refresh_session(get_bearer_user(authorization, db))
