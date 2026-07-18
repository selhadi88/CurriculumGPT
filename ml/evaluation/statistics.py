"""
Statistical validation utilities (paper §5.2).

  - Cohen's d with 95% bootstrap CI (10,000 resamples)
  - ICC(2,1) via pingouin (graceful fallback if unavailable)
  - Two-tailed paired t-test with Bonferroni correction
  - Fleiss' kappa via statsmodels (for multi-rater gold subsets)
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Standardized mean difference (pooled SD)."""
    a, b = np.asarray(group_a, float), np.asarray(group_b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else 0.0


def cohens_d_ci(group_a, group_b, n_boot: int = 10000, seed: int = 42) -> dict:
    """Cohen's d with a 95% bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(group_a, float), np.asarray(group_b, float)
    point = cohens_d(a, b)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ra = rng.choice(a, len(a), replace=True)
        rb = rng.choice(b, len(b), replace=True)
        boots[i] = cohens_d(ra, rb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"cohens_d": round(point, 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4), "n_boot": n_boot}


def paired_ttest_bonferroni(model_scores, baseline_scores, n_comparisons: int = 6) -> dict:
    """Two-tailed paired t-test with Bonferroni-corrected alpha."""
    from scipy import stats
    a = np.asarray(model_scores, float)
    b = np.asarray(baseline_scores, float)
    m = min(len(a), len(b))
    t, p = stats.ttest_rel(a[:m], b[:m])
    alpha_corr = 0.05 / n_comparisons
    return {"t_stat": round(float(t), 4), "p_value": float(p),
            "alpha_corrected": round(alpha_corr, 4),
            "significant": bool(p < alpha_corr)}


def icc_2_1(ratings_matrix: np.ndarray) -> dict:
    """
    ICC(2,1) — two-way random, single measures. `ratings_matrix` shape
    (n_subjects, n_raters). Uses pingouin; falls back to a manual ANOVA estimate.
    """
    data = np.asarray(ratings_matrix, float)
    try:
        import pandas as pd
        import pingouin as pg
        n_subj, n_rater = data.shape
        long = pd.DataFrame({
            "subject": np.repeat(np.arange(n_subj), n_rater),
            "rater": np.tile(np.arange(n_rater), n_subj),
            "rating": data.flatten(),
        })
        res = pg.intraclass_corr(data=long, targets="subject", raters="rater", ratings="rating")
        matched = res[res["Type"] == "ICC2"]
        if matched.empty:
            raise ValueError("ICC2 row not found")
        row = matched.iloc[0]
        ci = row["CI95%"]
        return {"icc": round(float(row["ICC"]), 4),
                "ci_low": round(float(ci[0]), 4) if ci is not None else None,
                "ci_high": round(float(ci[1]), 4) if ci is not None else None}
    except Exception as exc:  # noqa: BLE001
        logger.warning("pingouin ICC failed (%s) — manual estimate.", exc)
        return {"icc": _icc_manual(data), "ci_low": None, "ci_high": None}


def _icc_manual(data: np.ndarray) -> float:
    n, k = data.shape
    grand = data.mean()
    ms_rows = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return round(float((ms_rows - ms_err) / denom), 4) if denom else 0.0


def fleiss_kappa(rater_labels: np.ndarray) -> float:
    """
    Fleiss' kappa. `rater_labels` shape (n_items, n_raters), integer categories.
    """
    try:
        from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa as fk
        table, _ = aggregate_raters(np.asarray(rater_labels))
        return round(float(fk(table)), 4)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fleiss kappa failed: %s", exc)
        return 0.0
