"""
Sskill — O*NET skill-taxonomy matching component (paper §3.8).

    Sskill(c, i) = mean over (sc in Phi_c, si in Phi_i) of cos(e_sc, e_si)

Phi_x is the set of taxonomy skills detected in document x (keyword-based, via
skill_utils). e_s are GNN-refined skill embeddings when available; otherwise we
fall back to a parameter-free Jaccard-style overlap of the detected skill sets.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from . import skill_utils

logger = logging.getLogger(__name__)

# Optional in-memory cache of refined embeddings set by CurriculumGPT after the
# GCN forward pass: a Tensor of shape (num_skills, dim).
_refined_embeddings: torch.Tensor | None = None


def set_refined_embeddings(emb: torch.Tensor | None) -> None:
    global _refined_embeddings
    _refined_embeddings = emb.detach().cpu() if emb is not None else None


def _embedding_score(idx_c: list[int], idx_i: list[int]) -> float:
    emb = _refined_embeddings
    e_c = emb[idx_c]                       # (|Phi_c|, D)
    e_i = emb[idx_i]                       # (|Phi_i|, D)
    e_c = torch.nn.functional.normalize(e_c, dim=-1)
    e_i = torch.nn.functional.normalize(e_i, dim=-1)
    sim = e_c @ e_i.T                      # (|Phi_c|, |Phi_i|)
    return float(sim.mean().clamp(0.0, 1.0))


def _jaccard_fallback(idx_c: list[int], idx_i: list[int]) -> float:
    sc, si = set(idx_c), set(idx_i)
    union = sc | si
    if not union:
        return 0.5
    return len(sc & si) / len(union)


def compute(curriculum_text: str, job_text: str, **kwargs) -> float:
    """Mean pairwise cosine of detected skills' embeddings. Range [0, 1]."""
    idx_c = skill_utils.present_skill_indices(curriculum_text)
    idx_i = skill_utils.present_skill_indices(job_text)
    if not idx_c or not idx_i:
        return 0.5

    if _refined_embeddings is not None:
        try:
            return _embedding_score(idx_c, idx_i)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skill embedding score failed (%s) — using Jaccard.", exc)

    return _jaccard_fallback(idx_c, idx_i)
