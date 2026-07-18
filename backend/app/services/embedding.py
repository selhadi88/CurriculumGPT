"""
EmbeddingService — thin wrapper around AlignmentModel.
Lazy-loads the model on first call; reuses the singleton across requests.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_alignment_model() -> object:
    """Return the lazy-loaded AlignmentModel singleton."""
    from ml.models.alignment import AlignmentModel

    settings = get_settings()
    model = AlignmentModel(
        model_type=settings.model_type,
        checkpoint_path=settings.model_checkpoint,
        device=settings.device or None,
    )
    logger.info("AlignmentModel initialised (type=%s)", settings.model_type)
    return model


def embed_and_score(
    curriculum_texts: list[str],
    job_texts: list[str],
) -> list[object]:
    """Score all (curriculum[i], job[i]) pairs and return AlignmentResult list."""
    model = get_alignment_model()
    return model.score(curriculum_texts, job_texts)  # type: ignore[union-attr]
