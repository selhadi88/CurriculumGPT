"""
Strend — industry trend-alignment component (paper §3.8).

A BiLSTM consumes 12 monthly skill-frequency vectors:
    F in R^{12 x num_skills},   h_trend = BiLSTM(F) in R^1024

The BiLSTM + projection live inside CurriculumGPT. This module is responsible
for loading the monthly frequency matrix (built by scripts/build_dataset.py from
dated postings) and a parameter-free fallback.

The current shipped data has no posting dates, so the monthly matrix is usually
absent and this contributes a constant 0.5.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import skill_utils

logger = logging.getLogger(__name__)

_FREQ_PATH = Path(__file__).resolve().parents[2] / "data" / "mappings" / "monthly_skill_freq.json"
N_MONTHS = 12

_warned = False
_cache: np.ndarray | None = None
_loaded = False


def load_monthly_frequencies() -> np.ndarray | None:
    """Returns (12, num_skills) matrix or None if unavailable."""
    global _cache, _loaded
    if _loaded:
        return _cache
    _loaded = True
    if not _FREQ_PATH.exists():
        _cache = None
        return None
    try:
        with _FREQ_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        matrix = np.array(data["matrix"], dtype=np.float32)  # (12, num_skills)
        _cache = matrix
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load monthly_skill_freq.json: %s", exc)
        _cache = None
    return _cache


def get_frequency_tensor() -> np.ndarray:
    """
    Always returns a (12, num_skills) float matrix — real data if present,
    otherwise zeros (with a one-time warning).
    """
    global _warned
    matrix = load_monthly_frequencies()
    if matrix is None:
        if not _warned:
            logger.warning("No monthly skill-frequency data — Strend uses a zero "
                           "matrix (constant 0.5). Re-scrape dated postings to enable.")
            _warned = True
        return np.zeros((N_MONTHS, skill_utils.get_num_skills()), dtype=np.float32)
    return matrix


def has_real_data() -> bool:
    return load_monthly_frequencies() is not None


def compute(curriculum_text: str, job_text: str, **kwargs) -> float:
    """
    Parameter-free fallback: constant 0.5 (the BiLSTM-based score is produced by
    CurriculumGPT when monthly data exists).
    """
    return 0.5
