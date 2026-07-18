from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints import analysis, courses, health, jobs

router = APIRouter()

router.include_router(health.router)
router.include_router(analysis.router)
router.include_router(jobs.router)
router.include_router(courses.router)
