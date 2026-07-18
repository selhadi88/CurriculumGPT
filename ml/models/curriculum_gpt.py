"""
CurriculumGPT — the paper's seven-component weighted-similarity model (§3.3).

    S(c, i) = sum_k  w_k * S_k(c, i),   w = softmax(theta),  theta in R^7

Components (in fixed order):
    0 semantic   Ssem    — BGE bi-encoder cosine (§3.4, trainable)
    1 topic      Scov    — LDA topic cosine (§3.5, fixed feature)
    2 recency    Srec    — exp decay on pub year (§3.6, fixed feature)
    3 depth      Sdep    — Bloom histograms → learned layer (§3.7, trainable)
    4 skill      Sskill  — O*NET skills + GCN (§3.8, trainable via GCN)
    5 trend      Strend  — BiLSTM over monthly skill freq (§3.8, trainable)
    6 structure  Sstruct — Spearman of skill orderings (§3.9, fixed feature)

Design: the *deterministic* components (topic, recency, structure) are computed
from text/metadata on CPU and fed in as a precomputed `feature` tensor. The
*learnable* components (semantic via BGE, depth via bloom_layer, skill via GCN,
trend via BiLSTM) are computed inside forward() so gradients flow. The fusion
weights `theta` are always learned.

A binary classification head on the fused score enables accuracy/F1 evaluation.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from ml.components import skill_gnn, skill_taxonomy, skill_utils
from ml.models.bi_encoder import BGEBiEncoder

logger = logging.getLogger(__name__)

COMPONENT_NAMES = ["semantic", "topic", "recency", "depth", "skill", "trend", "structure"]
N_COMPONENTS = 7

# Indices of components supplied as precomputed deterministic features.
IDX_SEMANTIC, IDX_TOPIC, IDX_RECENCY, IDX_DEPTH, IDX_SKILL, IDX_TREND, IDX_STRUCT = range(7)


class CurriculumGPT(nn.Module):
    def __init__(
        self,
        bge_model: BGEBiEncoder | None = None,
        num_skills: int | None = None,
        skill_emb_dim: int = 256,
        cache_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.num_skills = num_skills or skill_utils.get_num_skills()

        # Semantic core (component 0). Reuse a provided BGE to avoid double-loading.
        self.bge = bge_model if bge_model is not None else BGEBiEncoder(
            num_skills=self.num_skills, cache_dir=cache_dir
        )

        # Learned fusion weights θ → softmax (Eq. 1)
        self.theta = nn.Parameter(torch.zeros(N_COMPONENTS))

        # Depth (component 3): Wd ∈ R^{1×12} over [b_c ; b_i]
        self.bloom_layer = nn.Linear(12, 1)

        # Trend (component 5): BiLSTM over 12 monthly skill-frequency vectors
        self.trend_lstm = nn.LSTM(self.num_skills, 512, batch_first=True, bidirectional=True)
        self.trend_proj = nn.Linear(1024, 1)

        # Skill GNN (component 4): refines skill node embeddings
        self.skill_gnn = skill_gnn.TwoLayerGCN(skill_emb_dim, skill_emb_dim, skill_emb_dim)
        self.register_buffer("_skill_adj", skill_gnn.build_adjacency(), persistent=False)
        self._skill_node_features: torch.Tensor | None = None  # set lazily

        # Binary aligned/misaligned head on the fused scalar score
        self.cls_head = nn.Linear(1, 2)

    # ------------------------------------------------------------------ utils

    def fusion_weights(self) -> torch.Tensor:
        return torch.softmax(self.theta, dim=0)

    def _ensure_skill_features(self, device: torch.device) -> None:
        if self._skill_node_features is not None:
            return
        # Encode skill names with the BGE backbone (no grad) as node init features.
        def encode(sentences: list[str]) -> torch.Tensor:
            return self.bge.encode_backbone_texts(sentences, batch_size=16, device=str(device))
        feats = skill_gnn.init_node_features(encode, dim=256).to(device)
        self._skill_node_features = feats

    def refresh_skill_embeddings(self) -> torch.Tensor:
        """Run the GCN and publish refined embeddings to the skill component."""
        device = self._skill_adj.device
        self._ensure_skill_features(device)
        refined = self.skill_gnn(self._skill_node_features, self._skill_adj)
        skill_taxonomy.set_refined_embeddings(refined)
        return refined

    # ------------------------------------------------------------------ depth

    def _depth_score(self, bloom_c: torch.Tensor, bloom_i: torch.Tensor) -> torch.Tensor:
        x = torch.cat([bloom_c, bloom_i], dim=-1)            # (B, 12)
        return torch.sigmoid(self.bloom_layer(x)).squeeze(-1)  # (B,)

    # ------------------------------------------------------------------ trend

    def _trend_score(self, monthly_freq: torch.Tensor) -> torch.Tensor:
        # monthly_freq: (B, 12, num_skills)
        out, _ = self.trend_lstm(monthly_freq)               # (B, 12, 1024)
        h = out[:, -1, :]                                    # last step
        return torch.sigmoid(self.trend_proj(h)).squeeze(-1)  # (B,)

    # ------------------------------------------------------------------ skill

    def _skill_score_from_indices(
        self, idx_c: list[list[int]], idx_i: list[list[int]], device: torch.device
    ) -> torch.Tensor:
        refined = self.refresh_skill_embeddings()            # (N, D), grad-enabled
        scores = []
        for a, b in zip(idx_c, idx_i):
            if not a or not b:
                scores.append(torch.tensor(0.5, device=device))
                continue
            ec = refined[a]                                  # (|a|, D)
            ei = refined[b]                                  # (|b|, D)
            sim = (ec @ ei.T).clamp(0.0, 1.0)
            scores.append(sim.mean())
        return torch.stack(scores)                           # (B,)

    # ------------------------------------------------------------------ forward

    def forward(self, batch: dict) -> dict:
        """
        Expects a batch dict with:
          curriculum_ids, curriculum_mask, job_ids, job_mask   (tensors)
          bloom_c, bloom_i        (B, 6) float
          monthly_freq            (B, 12, num_skills) float
          det_features            (B, 3) float  → [Scov, Srec, Sstruct] precomputed
          skill_idx_c, skill_idx_i  list[list[int]] (len B)
        Returns alignment_score, per-component scores, weights, cls_logits.
        """
        device = batch["curriculum_ids"].device

        # 0 semantic — BGE cosine (trainable)
        curr_emb = self.bge.encode_curriculum(batch["curriculum_ids"], batch["curriculum_mask"])
        job_emb = self.bge.encode_job(batch["job_ids"], batch["job_mask"])
        s_sem = (curr_emb * job_emb).sum(dim=-1).clamp(0.0, 1.0)         # (B,)

        # 3 depth (trainable)
        s_dep = self._depth_score(batch["bloom_c"], batch["bloom_i"])

        # 4 skill (trainable via GCN)
        s_skill = self._skill_score_from_indices(
            batch["skill_idx_c"], batch["skill_idx_i"], device
        )

        # 5 trend (trainable)
        s_trend = self._trend_score(batch["monthly_freq"])

        # 1 topic, 2 recency, 6 structure — precomputed deterministic features
        det = batch["det_features"]                                      # (B, 3)
        s_cov, s_rec, s_struct = det[:, 0], det[:, 1], det[:, 2]

        scores = torch.stack(
            [s_sem, s_cov, s_rec, s_dep, s_skill, s_trend, s_struct], dim=-1
        )                                                                # (B, 7)
        weights = self.fusion_weights()                                  # (7,)
        S = (scores * weights).sum(dim=-1)                               # (B,)

        return {
            "alignment_score": S,
            "component_scores": scores,
            "weights": weights,
            "cls_logits": self.cls_head(S.unsqueeze(-1)),                # (B, 2)
            "curriculum_embedding": curr_emb,
            "job_embedding": job_emb,
        }
