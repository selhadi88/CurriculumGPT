"""
Sdep — Bloom's taxonomy cognitive-depth component (paper §3.7).

Action-verb matching produces a 6-dim Bloom histogram per document:
    b in R^6  (remember, understand, apply, analyze, evaluate, create)

The final score uses a tiny learnable layer that lives in CurriculumGPT:
    Sdep = sigmoid(Wd @ [b_c ; b_i] + bd),  Wd in R^{1x12}

This module only builds the histograms. compute() provides a *parameter-free*
fallback (mean of the upper-Bloom alignment) so the component is usable without
a trained Wd (e.g. in unit tests / the live backend before training).
"""
from __future__ import annotations

import functools
import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_VERBS_PATH = Path(__file__).resolve().parents[2] / "data" / "mappings" / "bloom_verbs.json"

LEVELS = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
N_LEVELS = 6

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


@functools.lru_cache(maxsize=1)
def _verb_lookup() -> dict[str, int]:
    """Map each action verb -> its Bloom level index (0..5)."""
    if not _VERBS_PATH.exists():
        logger.warning("bloom_verbs.json not found at %s", _VERBS_PATH)
        return {}
    with _VERBS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    lookup: dict[str, int] = {}
    for level_idx, level in enumerate(LEVELS):
        for verb in data.get("verbs", {}).get(level, []):
            lookup[verb.lower()] = level_idx
    return lookup


def compute_bloom_histogram(text: str) -> np.ndarray:
    """
    Normalized 6-dim Bloom histogram. Each entry = (verb matches at that level)
    / (total tokens). Sums to <= 1. Returns zeros for empty text.
    """
    lookup = _verb_lookup()
    tokens = _TOKEN_RE.findall(text.lower())
    hist = np.zeros(N_LEVELS, dtype=np.float32)
    if not tokens:
        return hist
    for tok in tokens:
        idx = lookup.get(tok)
        if idx is not None:
            hist[idx] += 1.0
    hist /= float(len(tokens))
    return hist


def compute(curriculum_text: str, job_text: str, **kwargs) -> float:
    """
    Parameter-free Bloom-alignment fallback in [0, 1]: cosine similarity of the
    two Bloom histograms (defaults to 0.5 when either histogram is all-zero).
    The trained version uses CurriculumGPT.bloom_layer instead.
    """
    b_c = compute_bloom_histogram(curriculum_text)
    b_i = compute_bloom_histogram(job_text)
    denom = float(np.linalg.norm(b_c) * np.linalg.norm(b_i))
    if denom == 0.0:
        return 0.5
    cos = float(np.dot(b_c, b_i) / denom)
    return max(0.0, min(1.0, cos))
