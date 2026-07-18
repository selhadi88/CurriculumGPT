"""
Baseline aligners for Table 1 reproduction.

Each baseline exposes `.fit(train_rows)` (no-op for unsupervised ones) and
`.predict_scores(rows) -> np.ndarray` in [0, 1]. A 0.5 threshold yields labels.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _texts(rows):
    return ([r["curriculum_text"] for r in rows], [r["job_text"] for r in rows])


class TfidfBaseline:
    """TF-IDF keyword overlap via cosine similarity."""

    name = "TF-IDF Keyword Overlap"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", max_features=20000)

    def fit(self, rows) -> None:
        c, j = _texts(rows)
        self.vec.fit(c + j)

    def predict_scores(self, rows) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity
        c, j = _texts(rows)
        cv, jv = self.vec.transform(c), self.vec.transform(j)
        return np.array([float(cosine_similarity(cv[i], jv[i])[0, 0]) for i in range(len(rows))])


class SentenceBertBaseline:
    """SentenceBERT all-mpnet-base-v2 cosine (no fine-tuning)."""

    name = "SentenceBERT (all-mpnet-base-v2)"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def fit(self, rows) -> None:
        return None

    def predict_scores(self, rows) -> np.ndarray:
        c, j = _texts(rows)
        ec = self.model.encode(c, normalize_embeddings=True, show_progress_bar=False)
        ej = self.model.encode(j, normalize_embeddings=True, show_progress_bar=False)
        return np.clip((ec * ej).sum(axis=1), 0.0, 1.0)


def evaluate_scores(scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy / precision / recall / F1 / AUC from continuous scores."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    preds = (scores >= threshold).astype(int)
    out = {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
    }
    try:
        out["auc"] = round(float(roc_auc_score(labels, scores)), 4)
    except ValueError:
        out["auc"] = None
    return out
