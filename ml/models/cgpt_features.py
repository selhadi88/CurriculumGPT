"""
Feature builder for CurriculumGPT.

Turns raw (curriculum_text, job_text[, metadata]) into the batch dict that
`CurriculumGPT.forward` consumes:

    curriculum_ids/mask, job_ids/mask   — BGE tokenized
    bloom_c, bloom_i                    — (B, 6) Bloom histograms
    monthly_freq                        — (B, 12, num_skills) trend input
    det_features                        — (B, 3): [Scov, Srec, Sstruct]
    skill_idx_c, skill_idx_i            — detected skill indices per item

Deterministic components (topic, recency, structure) are computed here on CPU.
"""
from __future__ import annotations

import numpy as np
import torch

from ml.components import (
    bloom_depth,
    recency,
    skill_utils,
    structure,
    topic_coverage,
    trend,
)


def build_batch(
    curriculum_texts: list[str],
    job_texts: list[str],
    tokenizer,
    pub_years: list[int | None] | None = None,
    max_length: int = 256,
    device: str | torch.device = "cpu",
) -> dict:
    """Build the full feature batch for a list of (curriculum, job) pairs."""
    n = len(curriculum_texts)
    pub_years = pub_years or [None] * n
    dev = torch.device(device)

    curr_enc = tokenizer(curriculum_texts, padding="max_length", truncation=True,
                         max_length=max_length, return_tensors="pt")
    job_enc = tokenizer(job_texts, padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")

    bloom_c = np.stack([bloom_depth.compute_bloom_histogram(t) for t in curriculum_texts])
    bloom_i = np.stack([bloom_depth.compute_bloom_histogram(t) for t in job_texts])

    # Deterministic component features
    det = np.zeros((n, 3), dtype=np.float32)
    for k, (c, j, yr) in enumerate(zip(curriculum_texts, job_texts, pub_years)):
        det[k, 0] = topic_coverage.compute(c, j)
        det[k, 1] = recency.compute(yr)
        det[k, 2] = structure.compute(c, j)

    # Trend input: same monthly matrix per item (global signal); zeros if no data
    freq = trend.get_frequency_tensor()                       # (12, num_skills)
    monthly = np.broadcast_to(freq, (n, *freq.shape)).copy()

    skill_idx_c = [skill_utils.present_skill_indices(t) for t in curriculum_texts]
    skill_idx_i = [skill_utils.present_skill_indices(t) for t in job_texts]

    return {
        "curriculum_ids": curr_enc["input_ids"].to(dev),
        "curriculum_mask": curr_enc["attention_mask"].to(dev),
        "job_ids": job_enc["input_ids"].to(dev),
        "job_mask": job_enc["attention_mask"].to(dev),
        "bloom_c": torch.tensor(bloom_c, dtype=torch.float32, device=dev),
        "bloom_i": torch.tensor(bloom_i, dtype=torch.float32, device=dev),
        "monthly_freq": torch.tensor(monthly, dtype=torch.float32, device=dev),
        "det_features": torch.tensor(det, dtype=torch.float32, device=dev),
        "skill_idx_c": skill_idx_c,
        "skill_idx_i": skill_idx_i,
    }
