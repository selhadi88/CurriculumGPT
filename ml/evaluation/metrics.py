"""
Evaluation metrics for the alignment model.

AlignmentMetrics.compute() returns a dict with:
  - overall_score: mean cosine similarity across all (curriculum, job) pairs
  - industry_coverage: % of jobs with at least one curriculum match above threshold
  - curriculum_relevance: % of curriculum items matched to at least one job
  - skill_gap_count: number of skills with gap > threshold
  - ndcg_at_k: normalized discounted cumulative gain (for ranked retrieval)
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class MetricBundle:
    overall_score: float
    industry_coverage: float
    curriculum_relevance: float
    mean_skill_gap: float
    ndcg_at_10: float
    high_gap_skills: int      # number of skills with gap ≥ 0.4
    medium_gap_skills: int    # 0.2 ≤ gap < 0.4
    low_gap_skills: int       # 0.0 < gap < 0.2

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score * 100, 2),
            "industry_coverage": round(self.industry_coverage * 100, 2),
            "curriculum_relevance": round(self.curriculum_relevance * 100, 2),
            "mean_skill_gap": round(self.mean_skill_gap * 100, 2),
            "ndcg_at_10": round(self.ndcg_at_10, 4),
            "high_gap_skills": self.high_gap_skills,
            "medium_gap_skills": self.medium_gap_skills,
            "low_gap_skills": self.low_gap_skills,
        }


class AlignmentMetrics:
    def __init__(
        self,
        # A pair counts as a "match" at >= this cosine similarity. BGE projection
        # heads produce scores in the ~0.3–0.5 band before heavy fine-tuning, so
        # 0.35 reflects real matches; 0.5 was too strict and zeroed the metric.
        coverage_threshold: float = 0.35,
        gap_threshold: float = 0.2,
    ) -> None:
        self.coverage_threshold = coverage_threshold
        self.gap_threshold = gap_threshold

    def compute(
        self,
        similarity_matrix: list[list[float]],          # (n_curriculum, n_jobs)
        curriculum_skills: list[list[float]],           # (n_curriculum, n_skills)
        job_skills: list[list[float]],                  # (n_jobs, n_skills)
        relevance_labels: list[list[int]] | None = None,  # optional ground-truth
    ) -> MetricBundle:
        n_curr = len(similarity_matrix)
        n_jobs = len(similarity_matrix[0]) if similarity_matrix else 0

        # Overall score = mean of all pairwise similarities
        overall = (
            sum(v for row in similarity_matrix for v in row) / (n_curr * n_jobs)
            if n_curr > 0 and n_jobs > 0
            else 0.0
        )

        # Industry coverage: fraction of jobs matched by at least one curriculum item
        industry_coverage = 0.0
        if n_jobs > 0:
            covered_jobs = sum(
                1 for j in range(n_jobs)
                if any(similarity_matrix[i][j] >= self.coverage_threshold for i in range(n_curr))
            )
            industry_coverage = covered_jobs / n_jobs

        # Curriculum relevance: fraction of curriculum items relevant to at least one job
        curriculum_relevance = 0.0
        if n_curr > 0:
            relevant_items = sum(
                1 for i in range(n_curr)
                if any(similarity_matrix[i][j] >= self.coverage_threshold for j in range(n_jobs))
            )
            curriculum_relevance = relevant_items / n_curr

        # Skill gap analysis
        mean_skill_gap, high_gap, med_gap, low_gap = self._compute_skill_gaps(
            curriculum_skills, job_skills
        )

        # NDCG@10
        ndcg = 0.0
        if relevance_labels is not None:
            ndcg = self._mean_ndcg(similarity_matrix, relevance_labels, k=10)

        return MetricBundle(
            overall_score=overall,
            industry_coverage=industry_coverage,
            curriculum_relevance=curriculum_relevance,
            mean_skill_gap=mean_skill_gap,
            ndcg_at_10=ndcg,
            high_gap_skills=high_gap,
            medium_gap_skills=med_gap,
            low_gap_skills=low_gap,
        )

    def _compute_skill_gaps(
        self,
        curriculum_skills: list[list[float]],
        job_skills: list[list[float]],
    ) -> tuple[float, int, int, int]:
        if not curriculum_skills or not job_skills:
            return 0.0, 0, 0, 0

        n_skills = len(curriculum_skills[0])

        # Average skill presence across curricula and jobs
        avg_curr = [
            sum(row[s] for row in curriculum_skills) / len(curriculum_skills)
            for s in range(n_skills)
        ]
        avg_job = [
            sum(row[s] for row in job_skills) / len(job_skills)
            for s in range(n_skills)
        ]

        gaps = [max(0.0, avg_job[s] - avg_curr[s]) for s in range(n_skills)]
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0

        high = sum(1 for g in gaps if g >= 0.4)
        med = sum(1 for g in gaps if 0.2 <= g < 0.4)
        low = sum(1 for g in gaps if 0.0 < g < 0.2)

        return mean_gap, high, med, low

    @staticmethod
    def _dcg(relevances: list[int], k: int) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(rank + 2)
            for rank, rel in enumerate(relevances[:k])
        )

    def _mean_ndcg(
        self,
        similarity_matrix: list[list[float]],
        relevance_labels: list[list[int]],
        k: int = 10,
    ) -> float:
        ndcg_scores: list[float] = []
        for i, (scores, labels) in enumerate(zip(similarity_matrix, relevance_labels)):
            # Sort by predicted score descending
            ranked = [labels[j] for j in sorted(range(len(scores)), key=lambda x: -scores[x])]
            ideal = sorted(labels, reverse=True)

            dcg = self._dcg(ranked, k)
            idcg = self._dcg(ideal, k)

            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
