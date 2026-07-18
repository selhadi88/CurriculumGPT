"""
RecommendationService — maps skill gaps to actionable certifications and resources.
Reads from data/mappings/certifications.json and skill_taxonomy.json.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAPPINGS = Path(__file__).parent.parent.parent.parent / "data" / "mappings"


@lru_cache(maxsize=1)
def _load_certifications() -> dict[str, Any]:
    path = _MAPPINGS / "certifications.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_taxonomy() -> dict[str, Any]:
    path = _MAPPINGS / "skill_taxonomy.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def get_certifications_for_skills(skill_names: list[str], top_k: int = 5) -> list[dict[str, Any]]:
    """Return the top-k certifications most relevant to the given skill names."""
    data = _load_certifications()
    certs_flat: list[dict[str, Any]] = []

    for category in data.get("certifications", {}).values():
        certs_flat.extend(category)

    def relevance(cert: dict[str, Any]) -> int:
        keywords = [kw.lower() for kw in cert.get("keywords", [])]
        return sum(
            1 for skill in skill_names
            if any(skill.lower() in kw or kw in skill.lower() for kw in keywords)
        )

    ranked = sorted(certs_flat, key=relevance, reverse=True)
    return ranked[:top_k]


def build_learning_roadmap(
    skill_gaps: list[dict[str, Any]],
    max_items: int = 10,
) -> list[dict[str, Any]]:
    """
    Given a list of skill gaps (each with 'skill_name', 'severity', 'gap'),
    return a prioritized learning roadmap with certifications and free resources.
    """
    taxonomy = _load_taxonomy()
    roadmap: list[dict[str, Any]] = []

    high_gaps = [g for g in skill_gaps if g.get("severity") == "high"][:5]
    medium_gaps = [g for g in skill_gaps if g.get("severity") == "medium"][:5]
    prioritized = (high_gaps + medium_gaps)[:max_items]

    for gap in prioritized:
        skill_name = gap.get("skill_name", "")
        certs = get_certifications_for_skills([skill_name], top_k=2)

        # Find free resources from taxonomy keywords
        free_resources: list[str] = []
        for domain in taxonomy.get("domains", {}).values():
            for cat_data in domain.get("skills", {}).values():
                if skill_name.lower() in cat_data.get("label", "").lower():
                    keywords = cat_data.get("keywords", [])[:3]
                    free_resources = [
                        f"Search Coursera for: {kw}" for kw in keywords
                    ]
                    break

        roadmap.append({
            "skill": skill_name,
            "severity": gap.get("severity"),
            "gap_pct": round(gap.get("gap", 0) * 100, 1),
            "certifications": [
                {"name": c["name"], "provider": c["provider"], "cost_usd": c["cost_usd"]}
                for c in certs
            ],
            "free_resources": free_resources[:3],
        })

    return roadmap
