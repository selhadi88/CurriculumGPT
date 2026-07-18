"""
Scov — LDA topic-coverage component (paper §3.5).

    Scov(c, i) = cosine(theta_c, theta_i),   theta in R^50

LDA (K=50) is fit ONLY on training-fold documents to prevent leakage, then the
fitted CountVectorizer + LDA are persisted with joblib. At inference / scoring
we load the artifacts and transform each document to its topic distribution.

If no fitted model exists, compute() falls back to 0.5 and warns once.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ARTIFACT_DIR = Path(__file__).resolve().parent / "lda_artifacts"
_ARTIFACT_PATH = _ARTIFACT_DIR / "lda.pkl"

N_TOPICS = 50
RANDOM_STATE = 42

_warned = False
_cache: dict | None = None


def fit_lda(texts: list[str], n_topics: int = N_TOPICS) -> None:
    """
    Fit CountVectorizer + LDA on the given (training-fold) texts and persist them.
    Call this once per CV fold with train-only documents.
    """
    import joblib
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        min_df=5,
        max_df=0.95,
        stop_words="english",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        dtm = vectorizer.fit_transform(texts)
    except ValueError:
        # Happens when the corpus is tiny (smoke tests): relax min_df.
        vectorizer = CountVectorizer(min_df=1, stop_words="english", lowercase=True)
        dtm = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        learning_method="online",
        max_iter=10,
    )
    lda.fit(dtm)

    _ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "lda": lda}, _ARTIFACT_PATH)
    _reset_cache()
    logger.info("LDA fitted (K=%d, vocab=%d) and saved to %s",
                n_topics, len(vectorizer.vocabulary_), _ARTIFACT_PATH)


def _load() -> dict | None:
    global _cache
    if _cache is not None:
        return _cache
    if not _ARTIFACT_PATH.exists():
        return None
    import joblib
    _cache = joblib.load(_ARTIFACT_PATH)
    return _cache


def _reset_cache() -> None:
    global _cache
    _cache = None


def transform(text: str) -> np.ndarray | None:
    """Topic distribution theta in R^K for one document, or None if unfitted."""
    artifacts = _load()
    if artifacts is None:
        return None
    dtm = artifacts["vectorizer"].transform([text])
    theta = artifacts["lda"].transform(dtm)[0]
    return theta.astype(np.float32)


def is_fitted() -> bool:
    return _load() is not None


def compute(curriculum_text: str, job_text: str, **kwargs) -> float:
    """Cosine of the two topic distributions. Range [0, 1]. Falls back to 0.5."""
    global _warned
    theta_c = transform(curriculum_text)
    theta_i = transform(job_text)
    if theta_c is None or theta_i is None:
        if not _warned:
            logger.warning("LDA model not fitted — Scov falls back to 0.5. "
                           "Call topic_coverage.fit_lda(train_texts) first.")
            _warned = True
        return 0.5

    denom = float(np.linalg.norm(theta_c) * np.linalg.norm(theta_i))
    if denom == 0.0:
        return 0.5
    cos = float(np.dot(theta_c, theta_i) / denom)
    return max(0.0, min(1.0, cos))
