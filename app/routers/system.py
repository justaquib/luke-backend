"""Health and statistics routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.database import get_session
from app.schemas import StatsResponse
from app.services import system

router = APIRouter(tags=["system"])


@router.get("/health")
def health() -> dict:
    return system.health_status()


@router.get("/stats", response_model=StatsResponse)
def stats(db: Session = Depends(get_session)) -> StatsResponse:
    return system.stats(db)
