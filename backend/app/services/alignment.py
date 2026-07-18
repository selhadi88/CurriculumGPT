"""
AlignmentService — orchestrates the full analysis pipeline:
  1. Parse curriculum input (text or CSV)
  2. Fetch job samples from DB (filtered by domain if requested)
  3. Run bi-encoder scoring
  4. Compute metrics
  5. Build skill gap list + certification recommendations
  6. Persist result to DB
"""
from __future__ import annotations

import csv
import io
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.db.models import Analysis, Job, SkillMapping
from app.schemas.analysis import AnalysisRequest, AnalysisResult, JobMatch, SkillGap
from app.services.embedding import embed_and_score, get_alignment_model
from app.services.recommendation import build_learning_roadmap, get_certifications_for_skills
from ml.evaluation.metrics import AlignmentMetrics

logger = logging.getLogger(__name__)

_ONET_PATH = Path(__file__).parent.parent.parent.parent / "data" / "mappings" / "onet_occupations.json"
_UI_OUTPUTS_DIR = Path(__file__).parent.parent.parent.parent / "UI-outputs"
_MAX_JOBS_PER_ANALYSIS = 20
_MAX_COURSES_PER_ANALYSIS = 100

# UI domain slugs → actual job categories present in the database.
_UI_DOMAIN_TO_DB = {
    "computer_science": ["Information Technology", "Engineering"],
    "data_science": ["Information Technology", "Finance"],
    "cloud_devops": ["Information Technology"],
    "web_development": ["Information Technology"],
    "cybersecurity": ["Information Technology"],
    "business": ["Finance", "Marketing", "Operations", "Human Resources"],
    "design": ["Marketing"],
}


def _parse_curriculum_text(text: str) -> tuple[list[str], list[str]]:
    """Split comma/newline-separated items. Returns (texts, labels)."""
    texts: list[str] = []
    labels: list[str] = []
    for part in text.replace(",", "\n").split("\n"):
        clean = part.strip()
        if clean:
            texts.append(clean)
            # Label = first 40 chars
            labels.append(clean[:40] if len(clean) > 40 else clean)
    return texts, labels


def _parse_curriculum_csv(csv_content: str) -> tuple[list[str], list[str]]:
    """
    Parse any CSV into texts + labels for display.
    Auto-detects title/description columns. Returns (texts, labels).
    """
    reader = csv.DictReader(io.StringIO(csv_content))
    if reader.fieldnames is None:
        return [], []

    fields = [f.strip().lower() for f in reader.fieldnames]

    _title_hints = {"title", "name", "course", "subject", "module", "course_title", "course name"}
    _desc_hints = {"description", "desc", "content", "summary", "detail", "overview",
                   "about", "syllabus", "objective", "requirement", "body", "text"}

    title_col = next(
        (reader.fieldnames[i] for i, f in enumerate(fields) if f in _title_hints),
        reader.fieldnames[0],
    )
    desc_col = next(
        (reader.fieldnames[i] for i, f in enumerate(fields) if f in _desc_hints),
        None,
    )

    texts: list[str] = []
    labels: list[str] = []
    for row in reader:
        title = (row.get(title_col) or "").strip()
        if not title:
            continue

        if desc_col:
            description = (row.get(desc_col) or "").strip()
        else:
            description = " ".join(
                str(v).strip() for k, v in row.items()
                if k != title_col and v and str(v).strip()
            )

        combined = f"{title}. {description}" if description else title
        texts.append(combined[:1000])
        # Short label for display
        labels.append(title[:40] if len(title) > 40 else title)

    return texts, labels


def _load_taxonomy() -> dict:
    path = Path(__file__).parent.parent.parent.parent / "data" / "mappings" / "skill_taxonomy.json"
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def _detect_skills_keyword(text: str, taxonomy: dict) -> list[float]:
    """
    Detect skill presence using keyword matching against the taxonomy.
    Returns a float vector (0–1) per skill: score > 0 means skill is mentioned.

    Matches against BOTH the skill's keywords AND its display label / key, so a
    short input like "Machine Learning, Python" reliably lights up the right
    skills. One keyword hit already gives a solid score (0.6); more hits → 1.0.
    """
    # Normalize hyphens/underscores to spaces so "big-data" still matches the
    # keyword "big data" — course titles and postings vary in punctuation.
    text_lower = text.lower().replace("-", " ").replace("_", " ")
    scores: list[float] = []
    for domain_data in taxonomy.get("domains", {}).values():
        for skill_key, skill_info in domain_data.get("skills", {}).items():
            # Build the full set of phrases that indicate this skill.
            phrases = list(skill_info.get("keywords", []))
            label = skill_info.get("label", "")
            if label:
                phrases.append(label)
            phrases.append(skill_key.replace("_", " "))

            phrases = [p.lower() for p in phrases if p]
            matches = sum(1 for p in phrases if p in text_lower)
            if matches == 0:
                scores.append(0.0)
            elif matches == 1:
                scores.append(0.6)   # one clear mention → strong signal
            else:
                scores.append(min(1.0, 0.6 + 0.2 * (matches - 1)))
    return scores


def _load_onet_mapping() -> dict[str, Any]:
    if _ONET_PATH.exists():
        with _ONET_PATH.open() as f:
            return json.load(f)
    return {}


def _fetch_relevant_jobs(db: Session, curriculum_texts: list[str], limit: int) -> list[Job]:
    """
    Retrieve jobs whose title/description mention terms from the curriculum, so
    the alignment comparison is against RELEVANT jobs (not random unrelated ones).
    Falls back to the first `limit` jobs if no keyword matches are found.
    """
    import re
    from sqlalchemy import or_

    # Extract distinctive terms (3+ chars) from the curriculum input.
    blob = " ".join(curriculum_texts).lower()
    terms = [t for t in re.findall(r"[a-zA-Z][a-zA-Z+#.]{2,}", blob)]
    # Keep the most informative unique terms (skip common filler words).
    stop = {"and", "the", "for", "with", "into", "from", "course", "skills", "data"}
    seen: list[str] = []
    for t in terms:
        if t not in stop and t not in seen:
            seen.append(t)
    keywords = seen[:12]

    if keywords:
        conds = [Job.title.ilike(f"%{kw}%") for kw in keywords]
        conds += [Job.description.ilike(f"%{kw}%") for kw in keywords]
        relevant = db.query(Job).filter(or_(*conds)).limit(limit).all()
        if relevant:
            return relevant

    return db.query(Job).limit(limit).all()


def _get_severity(gap: float) -> str:
    if gap >= 0.4:
        return "high"
    if gap >= 0.2:
        return "medium"
    return "low"


def run_analysis(
    request: AnalysisRequest,
    analysis_id: uuid.UUID,
    db: Session,
) -> None:
    """
    Background task: runs the full analysis pipeline and persists the result.
    Updates the Analysis row as it progresses.
    """
    _set_status(db, analysis_id, "running")

    try:
        result = _execute(request, analysis_id, db)
        _persist_result(db, analysis_id, result)
    except Exception as exc:
        logger.exception("Analysis %s failed", analysis_id)
        _set_status(db, analysis_id, "failed", error=str(exc))


def _execute(
    request: AnalysisRequest,
    analysis_id: uuid.UUID,
    db: Session,
) -> AnalysisResult:
    # 1. Parse curriculum
    if request.csv_content:
        curriculum_texts, curriculum_labels = _parse_curriculum_csv(request.csv_content)
    elif request.text:
        curriculum_texts, curriculum_labels = _parse_curriculum_text(request.text)
    else:
        raise ValueError("Either 'text' or 'csv_content' must be provided")

    curriculum_texts = curriculum_texts[:_MAX_COURSES_PER_ANALYSIS]
    curriculum_labels = curriculum_labels[:_MAX_COURSES_PER_ANALYSIS]
    if not curriculum_texts:
        raise ValueError("No valid curriculum items found in input")

    # 2. Fetch jobs from DB — prefer jobs RELEVANT to the curriculum so the
    # comparison is meaningful (otherwise we'd grab random unrelated jobs like
    # "security guard" for a tech curriculum and every gap would look tiny).
    jobs = []
    if request.target_domain:
        # UI domain slugs (e.g. "computer_science") don't match the DB's job
        # categories (e.g. "Information Technology"). Map them, and fall back to
        # curriculum-relevance retrieval if the mapped filter finds nothing.
        db_domains = _UI_DOMAIN_TO_DB.get(request.target_domain, [])
        if db_domains:
            jobs = db.query(Job).filter(Job.domain.in_(db_domains)).limit(_MAX_JOBS_PER_ANALYSIS).all()
    elif request.target_job_title:
        jobs = db.query(Job).filter(
            Job.title.ilike(f"%{request.target_job_title}%")
        ).limit(_MAX_JOBS_PER_ANALYSIS).all()

    if not jobs:
        # No domain/title filter (or it matched nothing) → relevance retrieval.
        jobs = _fetch_relevant_jobs(db, curriculum_texts, _MAX_JOBS_PER_ANALYSIS)

    if not jobs:
        jobs = db.query(Job).limit(_MAX_JOBS_PER_ANALYSIS).all()

    if not jobs:
        raise ValueError("No jobs found in the database. Run `python scripts/scrape.py` first.")

    job_texts = [f"{j.title}. {j.description[:800]}" for j in jobs]
    job_labels = [j.title for j in jobs]
    # Per-job posting years → powers the Recency component (Srec).
    job_years = [j.posted_date.year if getattr(j, "posted_date", None) else None for j in jobs]

    # 3. Encode unique texts once — full similarity matrix in 2 forward passes
    from ml.models.alignment import AlignmentModel
    model: AlignmentModel = get_alignment_model()  # type: ignore[assignment]
    model._load()

    assert model._model is not None
    from ml.models.bi_encoder import BGEBiEncoder
    import torch

    component_scores: dict[str, float] = {}
    component_weights: dict[str, float] = {}

    if isinstance(model._model, BGEBiEncoder):
        # Falsy check, not `is None` — MODEL_CHECKPOINT arrives as "" (not None) when
        # unset via docker-compose's `${MODEL_CHECKPOINT:-}`, and "" still means "no
        # checkpoint, use the pretrained backbone directly".
        use_backbone = not model._checkpoint
        batch = 16
        if use_backbone:
            curr_embs = model._model.encode_backbone_texts(curriculum_texts, batch_size=batch)
            job_embs_tensor = model._model.encode_backbone_texts(job_texts, batch_size=batch)
        else:
            curr_embs = model._model.encode_texts(curriculum_texts, encoder="curriculum", batch_size=batch)
            job_embs_tensor = model._model.encode_texts(job_texts, encoder="job", batch_size=batch)

        sim_matrix = torch.matmul(curr_embs, job_embs_tensor.T)
        all_scores = sim_matrix.tolist()
    else:
        # curriculum_gpt and custom backends: score per (curriculum, job) pair.
        # Pass each job's posting year (Recency) so the component is data-backed.
        all_scores = []
        agg_components: dict[str, list[float]] = {}
        for curr_text in curriculum_texts:
            results = model.score(
                [curr_text] * len(job_texts), job_texts, pub_years=job_years
            )
            all_scores.append([r.overall_score for r in results])  # type: ignore
            for r in results:  # type: ignore[assignment]
                for name, val in getattr(r, "component_scores", {}).items():
                    agg_components.setdefault(name, []).append(val)
                if getattr(r, "weights", None):
                    component_weights = r.weights  # type: ignore[attr-defined]
        # Mean component score across all scored pairs → dashboard summary.
        component_scores = {
            name: round(sum(vals) / len(vals), 4)
            for name, vals in agg_components.items() if vals
        }

    # 3b. Keyword-based skill detection (works for any model, no training needed)
    taxonomy = _load_taxonomy()
    all_curr_skills = [_detect_skills_keyword(text, taxonomy) for text in curriculum_texts]
    all_job_skills = [_detect_skills_keyword(text, taxonomy) for text in job_texts]

    # 4. Compute aggregate metrics
    evaluator = AlignmentMetrics()
    metrics_bundle = evaluator.compute(all_scores, all_curr_skills, all_job_skills)

    # 5. Build skill gap list
    skill_mappings = {sm.skill_index: sm for sm in db.query(SkillMapping).all()}
    skill_gaps = _build_skill_gaps(all_curr_skills, all_job_skills, skill_mappings, db)

    # 6. Top matching jobs
    top_jobs = _build_job_matches(all_scores, jobs, all_curr_skills, all_job_skills, skill_mappings)

    # 7. Certifications
    high_skill_names = [g.skill_name for g in skill_gaps if g.severity == "high"][:5]
    certifications = get_certifications_for_skills(high_skill_names, top_k=5)

    # 8. O*NET mapping
    onet = _build_onet_mapping(jobs)

    return AnalysisResult(
        analysis_id=str(analysis_id),
        mode=request.mode,
        status="done",
        overall_score=round(metrics_bundle.overall_score * 100, 2),
        industry_coverage=round(metrics_bundle.industry_coverage * 100, 2),
        curriculum_relevance=round(metrics_bundle.curriculum_relevance * 100, 2),
        skill_matrix=all_scores,
        skill_gaps=skill_gaps,
        top_matching_jobs=top_jobs,
        curriculum_labels=curriculum_labels,
        job_labels=job_labels,
        metrics=metrics_bundle.to_dict(),
        certifications=certifications,
        onet_mapping=onet,
        component_scores=component_scores,
        component_weights=component_weights,
        created_at=datetime.utcnow(),
    )


def _build_skill_gaps(
    curr_skills: list[list[float]],
    job_skills: list[list[float]],
    skill_mappings: dict[int, SkillMapping],
    db: Session,
) -> list[SkillGap]:
    if not curr_skills or not job_skills:
        return []

    n_skills = len(curr_skills[0])
    avg_curr = [
        sum(row[s] for row in curr_skills) / len(curr_skills)
        for s in range(n_skills)
    ]
    avg_job = [
        sum(row[s] for row in job_skills) / len(job_skills)
        for s in range(n_skills)
    ]

    gaps: list[SkillGap] = []
    for idx in range(n_skills):
        # Include any skill that is meaningfully present in EITHER the curriculum
        # or the jobs — so the radar shows a real "You vs Required" picture, not
        # only positive gaps. (Previously this filtered to gap>0.05, leaving the
        # radar/charts nearly empty.)
        if avg_curr[idx] < 0.05 and avg_job[idx] < 0.05:
            continue

        gap = max(0.0, avg_job[idx] - avg_curr[idx])
        sm = skill_mappings.get(idx)
        skill_name = sm.skill_name if sm else f"skill_{idx}"
        severity = _get_severity(gap)
        certs = get_certifications_for_skills([skill_name], top_k=2)

        gaps.append(SkillGap(
            skill_index=idx,
            skill_name=skill_name,
            curriculum_score=round(avg_curr[idx], 3),
            job_score=round(avg_job[idx], 3),
            gap=round(gap, 3),
            severity=severity,
            recommended_certifications=[c["name"] for c in certs],
            suggested_resources=[f"Coursera: {skill_name}", f"GitHub: awesome-{skill_name.lower().replace(' ', '-')}"],
        ))

    # Sort by how "interesting" the skill is: biggest gaps first, then strongest presence.
    gaps.sort(key=lambda g: (g.gap, g.job_score, g.curriculum_score), reverse=True)
    return gaps[:20]


def _build_job_matches(
    all_scores: list[list[float]],
    jobs: list[Job],
    all_curr_skills: list[list[float]] | None = None,
    all_job_skills: list[list[float]] | None = None,
    skill_mappings: dict | None = None,
) -> list[JobMatch]:
    if not all_scores or not jobs:
        return []

    job_max_scores = [
        max(all_scores[i][j] for i in range(len(all_scores)))
        for j in range(len(jobs))
    ]
    ranked_indices = sorted(range(len(jobs)), key=lambda j: -job_max_scores[j])

    # Average curriculum skill presence
    avg_curr: list[float] = []
    if all_curr_skills:
        n = len(all_curr_skills[0])
        avg_curr = [sum(r[s] for r in all_curr_skills) / len(all_curr_skills) for s in range(n)]

    matches: list[JobMatch] = []
    for j_idx in ranked_indices[:10]:
        job = jobs[j_idx]
        score = job_max_scores[j_idx]

        matched_skills: list[str] = []
        missing_skills: list[str] = []

        if all_job_skills and avg_curr and skill_mappings:
            job_skill_vec = all_job_skills[j_idx]
            for skill_idx, (j_score, c_score) in enumerate(zip(job_skill_vec, avg_curr)):
                sm = skill_mappings.get(skill_idx)
                if sm is None:
                    continue
                if j_score >= 0.2 and c_score >= 0.15:
                    matched_skills.append(sm.skill_name)
                elif j_score >= 0.2 and c_score < 0.1:
                    missing_skills.append(sm.skill_name)

        # Fall back to DB-stored skills if detection found nothing
        if not matched_skills and not missing_skills:
            matched_skills = (job.skills or [])[:5]

        matches.append(JobMatch(
            job_id=str(job.id),
            title=job.title,
            company=job.company,
            domain=job.domain,
            alignment_score=round(score, 3),
            matched_skills=matched_skills[:5],
            missing_skills=missing_skills[:5],
            url=job.url,
        ))

    return matches


def _build_onet_mapping(jobs: list[Job]) -> dict[str, Any]:
    onet_data = _load_onet_mapping()
    occupations = onet_data.get("occupations", {})

    domain_counts: dict[str, int] = {}
    for job in jobs:
        if job.domain:
            domain_counts[job.domain] = domain_counts.get(job.domain, 0) + 1

    top_domains = sorted(domain_counts, key=lambda d: -domain_counts[d])[:3]
    relevant_occupations: list[dict[str, Any]] = []

    for occ in occupations.values():
        if occ.get("domain") in top_domains:
            relevant_occupations.append({
                "title": occ["title"],
                "domain": occ["domain"],
                "growth_rate": occ.get("growth_rate_pct"),
                "median_salary": occ.get("median_salary_usd"),
                "top_skills": occ.get("skills", [])[:5],
            })

    return {
        "top_domains": top_domains,
        "relevant_occupations": relevant_occupations[:5],
    }


def _set_status(
    db: Session,
    analysis_id: uuid.UUID,
    status: str,
    error: Optional[str] = None,
) -> None:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if analysis:
        analysis.status = status
        if error:
            analysis.error_message = error
        if status in ("done", "failed"):
            analysis.completed_at = datetime.utcnow()
        db.commit()


def _persist_result(
    db: Session,
    analysis_id: uuid.UUID,
    result: AnalysisResult,
) -> None:
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if analysis:
        analysis.status = "done"
        result_dict = result.model_dump(mode="json")
        analysis.result = result_dict
        analysis.completed_at = datetime.utcnow()
        db.commit()

    _save_to_ui_outputs(analysis_id, result_dict)


def _save_to_ui_outputs(analysis_id: uuid.UUID, result: dict) -> None:
    try:
        _UI_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        mode = result.get("mode", "unknown")
        stem = f"{timestamp}_{mode}_{str(analysis_id)[:8]}"
        (_UI_OUTPUTS_DIR / f"{stem}.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8"
        )
        logger.info("UI output saved: %s.json", stem)
        _export_excel(result, _UI_OUTPUTS_DIR / f"{stem}.xlsx")
        _export_pdf(result, _UI_OUTPUTS_DIR / f"{stem}.pdf")
    except Exception as exc:
        logger.warning("Could not save UI output: %s", exc)


def _export_excel(result: dict, path: Path) -> None:
    """Excel export: metrics, skill matrix, gap list. Best-effort."""
    try:
        from openpyxl import Workbook
    except ImportError:
        return
    try:
        wb = Workbook()
        ws = wb.active
        ws.title = "Metrics"
        ws.append(["Metric", "Value"])
        for k, v in (result.get("metrics") or {}).items():
            ws.append([k, v])
        ws.append([])
        ws.append(["Component", "Score", "Weight"])
        weights = result.get("component_weights") or {}
        for name, score in (result.get("component_scores") or {}).items():
            ws.append([name, score, weights.get(name)])

        gaps = wb.create_sheet("Skill Gaps")
        gaps.append(["Skill", "Curriculum", "Job", "Gap", "Severity"])
        for g in result.get("skill_gaps", []):
            gaps.append([g.get("skill_name"), g.get("curriculum_score"),
                         g.get("job_score"), g.get("gap"), g.get("severity")])

        matrix = wb.create_sheet("Skill Matrix")
        matrix.append(["course \\ job"] + (result.get("job_labels") or []))
        for label, row in zip(result.get("curriculum_labels") or [],
                              result.get("skill_matrix") or []):
            matrix.append([label] + [round(v, 3) for v in row])
        wb.save(path)
        logger.info("Excel export saved: %s", path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Excel export failed: %s", exc)


def _export_pdf(result: dict, path: Path) -> None:
    """One-page PDF summary. Best-effort."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        return
    try:
        c = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 60
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y, "CurriculumGPT — Alignment Report")
        y -= 30
        c.setFont("Helvetica", 11)
        for key in ("overall_score", "industry_coverage", "curriculum_relevance", "uncertainty"):
            if key in result:
                c.drawString(50, y, f"{key.replace('_', ' ').title()}: {result[key]}")
                y -= 18
        y -= 10
        if result.get("component_scores"):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "Seven-Component Breakdown"); y -= 20
            c.setFont("Helvetica", 10)
            weights = result.get("component_weights") or {}
            for name, score in result["component_scores"].items():
                w = weights.get(name)
                c.drawString(60, y, f"{name:12s}  score={score:.3f}" +
                             (f"  weight={w:.3f}" if w is not None else ""))
                y -= 15
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Top Skill Gaps"); y -= 20
        c.setFont("Helvetica", 10)
        for g in result.get("skill_gaps", [])[:5]:
            c.drawString(60, y, f"{g.get('skill_name')}  (gap {g.get('gap')}, {g.get('severity')})")
            y -= 15
        c.showPage()
        c.save()
        logger.info("PDF export saved: %s", path.name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PDF export failed: %s", exc)
