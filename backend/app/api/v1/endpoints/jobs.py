from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.db.models import Job
from app.schemas.job import DomainStats, JobListResponse, JobResponse

router = APIRouter()


@router.get("/jobs", response_model=JobListResponse, tags=["jobs"])
def list_jobs(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    domain: Optional[str] = None,
    keyword: Optional[str] = None,
    db: Session = Depends(get_db),
) -> JobListResponse:
    query = db.query(Job)

    if domain:
        query = query.filter(Job.domain == domain)
    if keyword:
        query = query.filter(
            Job.title.ilike(f"%{keyword}%") | Job.description.ilike(f"%{keyword}%")
        )

    total = query.count()
    jobs = query.offset((page - 1) * page_size).limit(page_size).all()

    return JobListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[JobResponse.model_validate(j) for j in jobs],
    )


@router.get("/jobs/search", response_model=JobListResponse, tags=["jobs"])
def search_jobs(
    q: str = Query(..., min_length=2),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
) -> JobListResponse:
    """Full-text keyword search across title and description."""
    query = db.query(Job).filter(
        Job.title.ilike(f"%{q}%") | Job.description.ilike(f"%{q}%")
    )
    total = query.count()
    jobs = query.offset((page - 1) * page_size).limit(page_size).all()

    return JobListResponse(
        total=total,
        page=page,
        page_size=page_size,
        items=[JobResponse.model_validate(j) for j in jobs],
    )


@router.get("/domains", response_model=list[DomainStats], tags=["jobs"])
def list_domains(db: Session = Depends(get_db)) -> list[DomainStats]:
    """Return job counts and top skills per domain."""
    rows = (
        db.query(Job.domain, func.count(Job.id).label("job_count"))
        .filter(Job.domain.isnot(None), Job.domain != "")
        .group_by(Job.domain)
        .order_by(func.count(Job.id).desc())
        .all()
    )

    result: list[DomainStats] = []
    for domain, count in rows:
        # Collect skills for this domain (simplified: aggregate from skills array)
        domain_jobs = db.query(Job).filter(Job.domain == domain).limit(50).all()
        skill_freq: dict[str, int] = {}
        for job in domain_jobs:
            for skill in (job.skills or []):
                skill_freq[skill] = skill_freq.get(skill, 0) + 1

        top_skills = sorted(skill_freq, key=lambda s: -skill_freq[s])[:5]

        # Average salaries
        salary_rows = (
            db.query(func.avg(Job.salary_min), func.avg(Job.salary_max))
            .filter(Job.domain == domain, Job.salary_min.isnot(None))
            .first()
        )

        result.append(DomainStats(
            domain=domain,
            job_count=count,
            top_skills=top_skills,
            avg_salary_min=round(salary_rows[0], 0) if salary_rows and salary_rows[0] else None,
            avg_salary_max=round(salary_rows[1], 0) if salary_rows and salary_rows[1] else None,
        ))

    return result
