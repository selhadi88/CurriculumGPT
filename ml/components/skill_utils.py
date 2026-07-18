"""
Shared skill-taxonomy utilities.

Single source of truth for:
  - loading the skill taxonomy,
  - the canonical ordered list of skills (index ⇄ name),
  - keyword-based skill detection (mirrors the backend's
    `_detect_skills_keyword` but lives in ml/ so training has no backend import).

The skill count is derived dynamically from the taxonomy so the model never
assumes a fixed 150 (the shipped taxonomy actually defines ~49 skills).
"""
from __future__ import annotations

import functools
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_TAXONOMY_PATH = Path(__file__).resolve().parents[2] / "data" / "mappings" / "skill_taxonomy.json"


@functools.lru_cache(maxsize=1)
def load_taxonomy() -> dict:
    if not _TAXONOMY_PATH.exists():
        logger.warning("skill_taxonomy.json not found at %s", _TAXONOMY_PATH)
        return {"domains": {}}
    with _TAXONOMY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _skill_records() -> tuple[tuple[str, str, list[str], str], ...]:
    """
    Ordered tuple of (skill_key, label, keywords, domain).
    Order is deterministic: domain insertion order, then skill insertion order —
    identical to how the backend keyword detector iterates, so indices align.
    """
    records: list[tuple[str, str, list[str], str]] = []
    for domain_name, domain_data in load_taxonomy().get("domains", {}).items():
        for skill_key, skill_info in domain_data.get("skills", {}).items():
            records.append((
                skill_key,
                skill_info.get("label", skill_key),
                [kw.lower() for kw in skill_info.get("keywords", [])],
                domain_name,
            ))
    return tuple(records)


def get_num_skills() -> int:
    """Number of skills actually defined in the taxonomy (derive, never hardcode)."""
    return len(_skill_records())


def skill_names() -> list[str]:
    """Display labels, ordered by skill index."""
    return [label for _, label, _, _ in _skill_records()]


def skill_keys() -> list[str]:
    return [key for key, _, _, _ in _skill_records()]


def detect_skill_vector(text: str) -> np.ndarray:
    """
    Soft skill-presence vector in [0, 1], one entry per taxonomy skill.
    Matches the backend's keyword scoring: min(1.0, matches / 3).
    """
    # Normalize hyphens/underscores to spaces so "big-data" still matches the
    # keyword "big data" — course titles and postings vary in punctuation.
    text_lower = text.lower().replace("-", " ").replace("_", " ")
    records = _skill_records()
    vec = np.zeros(len(records), dtype=np.float32)
    for i, (_, _, keywords, _) in enumerate(records):
        if not keywords:
            continue
        matches = sum(1 for kw in keywords if kw in text_lower)
        vec[i] = min(1.0, matches / 3.0)
    return vec


def present_skill_indices(text: str, threshold: float = 1e-6) -> list[int]:
    """Indices of skills whose soft score exceeds `threshold` (the set Φ in §3.8)."""
    vec = detect_skill_vector(text)
    return [int(i) for i in np.nonzero(vec > threshold)[0]]
