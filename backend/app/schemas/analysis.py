from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    mode: Literal["institution", "individual"]
    # For individual mode: plain text list of courses/skills
    text: Optional[str] = Field(None, description="Comma or newline-separated courses/skills")
    # For institution mode: CSV content as a string (title,description columns)
    csv_content: Optional[str] = Field(None, description="CSV content with title,description columns")
    # Target domain filter (optional)
    target_domain: Optional[str] = Field(None, description="Filter jobs by domain (e.g. 'data_science')")
    # For individual: target a specific job title
    target_job_title: Optional[str] = None


class SkillGap(BaseModel):
    skill_index: int
    skill_name: str
    curriculum_score: float   # 0–1, how strongly the curriculum covers this skill
    job_score: float           # 0–1, how strongly jobs require this skill
    gap: float                 # job_score - curriculum_score
    severity: Literal["high", "medium", "low"]
    recommended_certifications: list[str] = []
    suggested_resources: list[str] = []


class JobMatch(BaseModel):
    job_id: str
    title: str
    company: Optional[str]
    domain: Optional[str]
    alignment_score: float    # 0–1
    matched_skills: list[str]
    missing_skills: list[str]
    url: Optional[str]


class AnalysisResult(BaseModel):
    analysis_id: str
    mode: Literal["institution", "individual"]
    status: Literal["done"]
    overall_score: float        # 0–100
    industry_coverage: float    # % of job requirements met
    curriculum_relevance: float # % of courses relevant to market

    skill_matrix: list[list[float]]   # (n_courses, n_jobs)
    skill_gaps: list[SkillGap]
    top_matching_jobs: list[JobMatch]

    curriculum_labels: list[str] = []   # display names for heatmap rows
    job_labels: list[str] = []          # display names for heatmap columns

    metrics: dict[str, Any]
    certifications: list[dict[str, Any]]
    onet_mapping: dict[str, Any]

    # Seven-component breakdown (populated when MODEL_TYPE=curriculum_gpt)
    component_scores: dict[str, float] = {}    # {"semantic": 0.82, "topic": 0.71, ...}
    component_weights: dict[str, float] = {}   # softmax(theta) per component
    uncertainty: float = 0.0                   # Monte Carlo dropout std

    created_at: datetime


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: Literal["pending", "running", "done", "failed"]
    progress_pct: Optional[float] = None
    error: Optional[str] = None
    result: Optional[AnalysisResult] = None
