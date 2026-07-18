from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    skills: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(100))
    url: Mapped[Optional[str]] = mapped_column(String(1000))
    scraped_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    company: Mapped[Optional[str]] = mapped_column(String(300))
    domain: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    skills: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(300))
    salary_min: Mapped[Optional[float]] = mapped_column(Float)
    salary_max: Mapped[Optional[float]] = mapped_column(Float)
    source: Mapped[Optional[str]] = mapped_column(String(100))
    url: Mapped[Optional[str]] = mapped_column(String(1000))
    # Real job posting date — powers the Recency (Srec) and Trend (Strend) components.
    posted_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    scraped_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    mode: Mapped[str] = mapped_column(String(50), nullable=False)  # "institution" | "individual"
    status: Mapped[str] = mapped_column(String(50), default="pending")  # pending | running | done | failed
    input_data: Mapped[Optional[dict]] = mapped_column(JSON)
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class SkillMapping(Base):
    __tablename__ = "skill_mappings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    skill_index: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    skill_name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    domain: Mapped[Optional[str]] = mapped_column(String(100))
    category: Mapped[Optional[str]] = mapped_column(String(100))
    onet_code: Mapped[Optional[str]] = mapped_column(String(50))
    keywords: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))
    weight: Mapped[float] = mapped_column(Float, default=1.0)
