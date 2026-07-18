"""
Two-layer GCN over the O*NET / taxonomy skill graph (paper §3.8).

Skills are nodes; an edge connects two skills that co-occur in the same O*NET
occupation. Node features start from BGE sentence embeddings of the skill names
and are refined by a 2-layer GCN. The refined embeddings feed Sskill scoring.

We implement the GCN by hand (no torch_geometric dependency):
    h^{l+1} = ReLU( A_norm @ h^l @ W^l ),   A_norm = D^{-1/2} (A + I) D^{-1/2}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import skill_utils

logger = logging.getLogger(__name__)

_ONET_PATH = Path(__file__).resolve().parents[2] / "data" / "mappings" / "onet_occupations.json"


def build_adjacency() -> torch.Tensor:
    """
    Symmetric-normalized adjacency (with self-loops) over taxonomy skills.
    Edges: skills co-mentioned in the same O*NET occupation's skill list.
    Falls back to identity (self-loops only) when O*NET data is sparse.
    """
    names = [n.lower() for n in skill_utils.skill_names()]
    keys = [k.lower() for k in skill_utils.skill_keys()]
    n = len(names)
    name_to_idx: dict[str, int] = {}
    for idx, (label, key) in enumerate(zip(names, keys)):
        name_to_idx.setdefault(label, idx)
        name_to_idx.setdefault(key.replace("_", " "), idx)

    A = np.eye(n, dtype=np.float32)

    if _ONET_PATH.exists():
        with _ONET_PATH.open(encoding="utf-8") as f:
            onet = json.load(f)
        for occ in onet.get("occupations", {}).values():
            present = []
            for skill in occ.get("skills", []):
                idx = name_to_idx.get(str(skill).lower())
                if idx is not None:
                    present.append(idx)
            for a in present:
                for b in present:
                    if a != b:
                        A[a, b] = 1.0
                        A[b, a] = 1.0

    # Symmetric normalization
    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        return F.relu(self.W(A_norm @ h))


class TwoLayerGCN(nn.Module):
    """2-layer GCN producing refined skill embeddings (N, out_dim)."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 256, out_dim: int = 256) -> None:
        super().__init__()
        self.l1 = GCNLayer(in_dim, hidden_dim)
        self.l2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, node_features: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h = self.l1(node_features, A_norm)
        h = self.l2(h, A_norm)
        return F.normalize(h, dim=-1)


@torch.no_grad()
def init_node_features(encode_fn, dim: int = 256) -> torch.Tensor:
    """
    Initialize node features by embedding each skill name with `encode_fn`
    (a callable: list[str] -> Tensor[N, D]). Projects/pads to `dim`.
    Falls back to random features if encode_fn is None.
    """
    names = skill_utils.skill_names()
    if encode_fn is None:
        logger.warning("No encoder for skill node features — using random init.")
        return F.normalize(torch.randn(len(names), dim), dim=-1)

    sentences = [f"Skill: {name}" for name in names]
    emb = encode_fn(sentences)  # (N, D)
    if emb.shape[-1] > dim:
        emb = emb[:, :dim]
    elif emb.shape[-1] < dim:
        pad = torch.zeros(emb.shape[0], dim - emb.shape[-1])
        emb = torch.cat([emb, pad], dim=-1)
    return F.normalize(emb, dim=-1)
