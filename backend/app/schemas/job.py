from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class JobResponse(BaseModel):
    id: uuid.UUID
    title: str
    description: str
    company: Optional[str]
    domain: Optional[str]
    skills: Optional[list[str]]
    location: Optional[str]
    salary_min: Optional[float]
    salary_max: Optional[float]
    source: Optional[str]
    url: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[JobResponse]


class DomainStats(BaseModel):
    domain: str
    job_count: int
    top_skills: list[str]
    avg_salary_min: Optional[float]
    avg_salary_max: Optional[float]
