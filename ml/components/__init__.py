"""
ml.components — the six non-semantic similarity components of the
seven-component CurriculumGPT alignment architecture (paper §3.5–§3.9).

The seventh component, semantic similarity (Ssem, §3.4), is provided by
`ml.models.bi_encoder.BGEBiEncoder` and fused in `ml.models.curriculum_gpt`.

Each component module exposes a `compute(...)` function returning a score in a
documented range (mostly [0, 1]). Components that need a learnable layer
(Bloom depth, trend, skill GNN) expose helpers here and keep the trainable
parameters inside `CurriculumGPT`.

NOTE on skill count: the shipped `data/mappings/skill_taxonomy.json` declares
`total_skills: 150` but only defines ~49 skills. To avoid dimension mismatches,
all skill-vector logic derives the count dynamically via
`ml.components.skill_utils.get_num_skills()`. Do not hardcode 150.
"""
from __future__ import annotations

__all__ = [
    "topic_coverage",
    "recency",
    "bloom_depth",
    "skill_taxonomy",
    "skill_gnn",
    "trend",
    "structure",
    "skill_utils",
]
