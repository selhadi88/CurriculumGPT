"""
Sstruct — curriculum structure-coherence component (paper §3.9).

    Sstruct(c, i) = (1 + rho(r_c, r_i)) / 2  in [0, 1]

rho is the Spearman rank correlation between the order in which taxonomy skills
appear in the curriculum vs. in the job description. When ordering can't be
established (fewer than 2 shared skills, or no sequencing metadata) it defaults
to 0.5 — as the paper does for ~30% of its syllabi.

The shipped dataset has no syllabus sequencing, so this almost always returns 0.5.
"""
from __future__ import annotations

import logging
import re

import numpy as np
from scipy.stats import spearmanr

from . import skill_utils

logger = logging.getLogger(__name__)

_DEFAULT = 0.5
_warned = False

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


def _skill_order(text: str) -> dict[int, int]:
    """
    Map skill index -> position of its first keyword occurrence in the text.
    Lower position = appears earlier.
    """
    text_lower = text.lower()
    order: dict[int, int] = {}
    records = skill_utils._skill_records()  # ordered (key, label, keywords, domain)
    for idx, (_, _, keywords, _) in enumerate(records):
        first = None
        for kw in keywords:
            pos = text_lower.find(kw)
            if pos != -1:
                first = pos if first is None else min(first, pos)
        if first is not None:
            order[idx] = first
    return order


def compute(curriculum_text: str | None = None, job_text: str | None = None,
            seq_c=None, seq_i=None, **kwargs) -> float:
    """
    Spearman-based structure coherence in [0, 1]. Returns 0.5 when the orderings
    can't be aligned. `seq_c`/`seq_i` may be provided directly (precomputed).
    """
    global _warned
    if curriculum_text is None or job_text is None:
        return _DEFAULT

    order_c = _skill_order(curriculum_text)
    order_i = _skill_order(job_text)
    shared = sorted(set(order_c) & set(order_i))
    if len(shared) < 2:
        if not _warned:
            logger.warning("Insufficient shared-skill ordering — Sstruct defaults to 0.5.")
            _warned = True
        return _DEFAULT

    # Rank the shared skills by their position in each text, then correlate the
    # two orderings. argsort gives strictly-distinct ranks, so the correlation is
    # always well-defined (no constant-vector NaN problem).
    pos_c = np.array([order_c[s] for s in shared], dtype=float)
    pos_i = np.array([order_i[s] for s in shared], dtype=float)
    rank_c = pos_c.argsort().argsort()
    rank_i = pos_i.argsort().argsort()
    with np.errstate(invalid="ignore"):
        rho, _ = spearmanr(rank_c, rank_i)
    if rho is None or rho != rho:  # NaN guard
        return _DEFAULT
    return max(0.0, min(1.0, (1.0 + float(rho)) / 2.0))
