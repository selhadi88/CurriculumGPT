"""
Initialize the database: create tables and optionally seed from processed CSVs.
Called on startup by the FastAPI lifespan handler.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from app.database import Base, _get_engine, _get_session_factory
from app.db.models import Course, Job, SkillMapping

logger = logging.getLogger(__name__)

_DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data"
_MAPPINGS_DIR = _DATA_ROOT / "mappings"
_PROCESSED_DIR = _DATA_ROOT / "processed"

# Cap how many rows to seed so first startup is fast. 0 = no cap (all rows).
# Set SEED_LIMIT in the environment (default 2000 — plenty for analysis).
_SEED_LIMIT = int(os.getenv("SEED_LIMIT", "2000"))


def create_tables() -> None:
    Base.metadata.create_all(bind=_get_engine())
    logger.info("Database tables created / verified")


def seed_skill_mappings(db: Session) -> None:
    if db.query(SkillMapping).count() > 0:
        return

    taxonomy_path = _MAPPINGS_DIR / "skill_taxonomy.json"
    if not taxonomy_path.exists():
        logger.warning("skill_taxonomy.json not found — skipping skill seed")
        return

    with taxonomy_path.open() as f:
        taxonomy = json.load(f)

    idx = 0
    for domain_key, domain_data in taxonomy.get("domains", {}).items():
        for cat_key, cat_data in domain_data.get("skills", {}).items():
            db.add(SkillMapping(
                skill_index=idx,
                skill_name=cat_data.get("label", cat_key),
                domain=domain_key,
                category=cat_key,
                keywords=cat_data.get("keywords", []),
                weight=cat_data.get("weight", 1.0),
            ))
            idx += 1

    db.commit()
    logger.info("Seeded %d skill mappings", idx)


def seed_courses(db: Session) -> None:
    if db.query(Course).count() > 0:
        return

    for csv_path in (_PROCESSED_DIR / "courses").glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, nrows=_SEED_LIMIT or None)
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_path, exc)
            continue

        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            description = str(row.get("description", row.get("content", ""))).strip()
            if not title or not description:
                continue

            skills_raw = row.get("skills", row.get("tags", ""))
            skills: list[str] = []
            if isinstance(skills_raw, str) and skills_raw:
                skills = [s.strip() for s in skills_raw.split(",") if s.strip()]

            db.add(Course(
                title=title,
                description=description,
                category=str(row.get("category", row.get("Category", ""))),
                skills=skills or None,
                source=str(row.get("source", csv_path.stem)),
                url=str(row.get("url", "")),
            ))

        db.commit()
        count = db.query(Course).count()
        logger.info("Seeded courses from %s — total: %d", csv_path.name, count)


def seed_jobs(db: Session) -> None:
    if db.query(Job).count() > 0:
        return

    for csv_path in (_PROCESSED_DIR / "jobs").glob("*.csv"):
        try:
            df = pd.read_csv(csv_path, nrows=_SEED_LIMIT or None)
        except Exception as exc:
            logger.warning("Could not read %s: %s", csv_path, exc)
            continue

        for _, row in df.iterrows():
            title = str(row.get("title", "")).strip()
            description = str(row.get("description", row.get("requirement", ""))).strip()
            if not title or not description:
                continue

            tags_raw = row.get("tags", row.get("skills", ""))
            skills: list[str] = []
            if isinstance(tags_raw, str) and tags_raw:
                skills = [s.strip() for s in tags_raw.split(",") if s.strip()]

            db.add(Job(
                title=title,
                description=description,
                company=str(row.get("company", "")),
                domain=str(row.get("domain", row.get("category", ""))),
                skills=skills or None,
                location=str(row.get("location", "")),
                source=str(row.get("source", csv_path.stem)),
                url=str(row.get("url", "")),
            ))

        db.commit()
        count = db.query(Job).count()
        logger.info("Seeded jobs from %s — total: %d", csv_path.name, count)


def init_db() -> None:
    """Called once at application startup."""
    create_tables()
    db = _get_session_factory()()
    try:
        seed_skill_mappings(db)
        seed_courses(db)
        seed_jobs(db)
    finally:
        db.close()
