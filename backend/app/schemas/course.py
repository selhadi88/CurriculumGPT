from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CourseResponse(BaseModel):
    id: uuid.UUID
    title: str
    description: str
    category: Optional[str]
    skills: Optional[list[str]]
    source: Optional[str]
    url: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class CourseListResponse(BaseModel):
    total: int
    page: int
    page_size: int
    items: list[CourseResponse]
