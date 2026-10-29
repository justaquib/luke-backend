"""Sign-in, the signed-in user and the invite a workspace hands a newcomer."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr


class GoogleAuthRequest(BaseModel):
    id_token: str
    invite_token: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105 -- OAuth token type label, not a secret


class UserOut(BaseModel):
    id: int
    google_id: str | None
    email: str
    name: str
    picture: str | None = None
    tenant_id: int | None = None
    role: str = "user"
    is_systemadmin: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class InviteCreate(BaseModel):
    email: EmailStr
    role: str = "user"


class InviteOut(BaseModel):
    id: int
    tenant_id: int
    email: str
    role: str
    token: str
    expires_at: datetime
    used_at: datetime | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class SystemAdminLogin(BaseModel):
    pin: str


class AcceptInviteRequest(BaseModel):
    id_token: str
