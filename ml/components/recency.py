"""
Srec — temporal recency component (paper §3.6).

    Srec = exp(-lambda * (t_cur - t_pub)),   lambda = 0.1 / year,   clamp to [0.1, 1.0]

The current shipped dataset has no posting dates, so pub_year is usually None and
this returns 0.5 (with a one-time warning). Once dated postings exist (USAJobs
PositionStartDate), pass the integer year via `pub_year`.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime

logger = logging.getLogger(__name__)

LAMBDA = 0.1
_MIN, _MAX = 0.1, 1.0
_FALLBACK = 0.5

_warned = False


def compute(pub_year: int | None = None, t_cur: int | None = None, **kwargs) -> float:
    """
    Recency score in [0.1, 1.0]. Returns 0.5 when pub_year is unavailable.
    `curriculum_text` / `job_text` are accepted and ignored for a uniform
    component signature.
    """
    global _warned
    if pub_year is None:
        if not _warned:
            logger.warning("pub_year missing — Srec falls back to %.1f. "
                           "Re-scrape dated postings (USAJobs) to enable recency.",
                           _FALLBACK)
            _warned = True
        return _FALLBACK

    year_now = t_cur if t_cur is not None else datetime.now().year
    delta = max(0, year_now - int(pub_year))
    score = math.exp(-LAMBDA * delta)
    return max(_MIN, min(_MAX, score))
