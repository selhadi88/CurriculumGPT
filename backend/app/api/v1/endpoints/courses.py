from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.db.models import Course
from app.schemas.course import CourseListResponse, CourseResponse

router = APIRouter()


@router.get("/courses", response_model=CourseListResponse, tags=["courses"])
def list_courses(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    keyword: Optional[str] = None,
    db: Session = Depends(get_db),
) -> CourseListResponse:
    query = db.query(Course)

    if category:
        query = query.filter(Course.category == category)
    if keyword:
        query = query.filter(
            Course.title.ilike(f"%{keyword}%") | Course.description.ilike(f"%{keyword}%")
        )

    total = query.count()
    courses = query.offset((page - 1) * page_size).limit(page_size).all()

    return CourseListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[CourseResponse.model_validate(c) for c in courses],
    )
