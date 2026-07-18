from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    model_type: str
    version: str


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        model_type=settings.model_type,
        version="0.1.0",
    )
