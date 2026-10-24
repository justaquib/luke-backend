"""Authentication and password helpers shared by routers and services."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import bcrypt
from fastapi import Depends, Header, HTTPException
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import SessionLocal, User, get_session


def hash_pin(pin: str) -> str:
    """Return a bcrypt hash for an administrator PIN."""
    return bcrypt.hashpw(pin.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def ensure_systemadmin() -> None:
    """Create or refresh the configured system administrator account."""
    with SessionLocal() as db:
        user = db.execute(select(User).where(User.is_systemadmin.is_(True))).scalar_one_or_none()
        if user is not None:
            if settings.systemadmin_pin:
                user.pin_hash = hash_pin(settings.systemadmin_pin)
                db.commit()
            return
        pin = settings.systemadmin_pin or "123456"
        db.add(
            User(
                email=settings.systemadmin_email,
                name="System Admin",
                is_systemadmin=True,
                pin_hash=hash_pin(pin),
            )
        )
        db.commit()


def create_access_token(data: dict[str, Any]) -> str:
    """Encode a signed access token containing the given claims."""
    to_encode = data.copy()
    to_encode.update({"exp": datetime.now(UTC).timestamp() + settings.jwt_expire_minutes * 60})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def get_bearer_user(authorization: str = Header(""), db: Session = Depends(get_session)) -> User:
    """Resolve the signed-in user from an Authorization header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = int(payload.get("sub"))
    except (JWTError, ValueError, TypeError) as error:
        raise HTTPException(401, "Invalid token") from error
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(401, "User not found")
    return user


def tenant_id_of(user: User) -> int | None:
    """The tenant whose lessons and answers this user may read and teach."""
    return user.tenant_id


def require_tenant_user(user: User, action: str) -> int:
    """Return the tenant id or fail when the user cannot perform the action."""
    if user.is_systemadmin:
        # The system administrator has no tenant of their own and looks after
        # every tenant, so they may perform cross-tenant governance actions.
        return user.tenant_id or 0
    if not user.tenant_id:
        raise HTTPException(403, "User is not assigned to a tenant")
    return user.tenant_id


def require_admin(user: User, action: str) -> int:
    """Return the tenant id for an administrator-only action."""
    if user.is_systemadmin:
        raise HTTPException(403, f"System admin cannot {action}")
    if not user.tenant_id:
        raise HTTPException(403, "User is not assigned to a tenant")
    if user.role not in ("superadmin", "admin"):
        raise HTTPException(403, f"Only admins can {action}")
    return user.tenant_id
