"""
BGEBiEncoder — fine-tunable bi-encoder built on BAAI/bge-large-en-v1.5.

Architecture:
  Shared BAAI/bge-large-en-v1.5 backbone (frozen or partially frozen)
  → Curriculum projection head  (1024 → 512 → 256, LayerNorm+GELU)
  → Job projection head          (1024 → 512 → 256, LayerNorm+GELU)
  → Skill classification head    (256 → num_skills, multi-label)

Training uses InfoNCE contrastive loss (see ml/training/losses.py).
Inference: cosine similarity on L2-normalized 256-dim projections.

BGE models prepend "Represent this sentence: " for asymmetric retrieval.
For symmetric tasks (both curriculum and job), no prefix is needed.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
_HIDDEN_DIM = 1024   # bge-large hidden size
_PROJ_DIM = 256      # final embedding dimension


class ProjectionHead(nn.Module):
    """Two-layer MLP projection: hidden → 512 → proj_dim, with LayerNorm."""

    def __init__(
        self,
        input_dim: int = _HIDDEN_DIM,
        proj_dim: int = _PROJ_DIM,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BGEBiEncoder(nn.Module):
    """
    Bi-encoder for curriculum-to-job alignment.

    Usage:
        model = BGEBiEncoder()
        curr_emb = model.encode_curriculum(curr_ids, curr_mask)   # (B, 256)
        job_emb  = model.encode_job(job_ids,  job_mask)            # (B, 256)
        scores   = (curr_emb * job_emb).sum(-1)                    # cosine, pre-normalized
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        proj_dim: int = _PROJ_DIM,
        num_skills: int = 150,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        cache_dir = cache_dir or os.getenv("MODEL_CACHE_DIR", ".model_cache")
        logger.info("Loading backbone: %s (cache=%s)", model_name, cache_dir)

        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen — only projection heads will be trained")

        hidden_size: int = self.backbone.config.hidden_size

        self.curriculum_head = ProjectionHead(hidden_size, proj_dim, dropout)
        self.job_head = ProjectionHead(hidden_size, proj_dim, dropout)
        self.skill_head = nn.Sequential(
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, num_skills),
        )

    def _cls_pooling(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """BGE uses [CLS] token representation (index 0)."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]   # (B, hidden_size)

    def encode_curriculum(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns L2-normalized (B, proj_dim) curriculum embeddings."""
        cls = self._cls_pooling(input_ids, attention_mask)
        proj = self.curriculum_head(cls)
        return F.normalize(proj, dim=-1)

    def encode_job(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns L2-normalized (B, proj_dim) job embeddings."""
        cls = self._cls_pooling(input_ids, attention_mask)
        proj = self.job_head(cls)
        return F.normalize(proj, dim=-1)

    def forward(
        self,
        curriculum_ids: torch.Tensor,
        curriculum_mask: torch.Tensor,
        job_ids: torch.Tensor,
        job_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        curr_emb = self.encode_curriculum(curriculum_ids, curriculum_mask)  # (B, D)
        job_emb = self.encode_job(job_ids, job_mask)                        # (B, D)

        # Cosine similarity is just dot-product when both are L2-normalized
        alignment_score = (curr_emb * job_emb).sum(dim=-1)                 # (B,)

        return {
            "alignment_score": alignment_score,
            "curriculum_embedding": curr_emb,
            "job_embedding": job_emb,
            "curriculum_skills": self.skill_head(curr_emb),
            "job_skills": self.skill_head(job_emb),
        }

    @torch.no_grad()
    def encode_backbone_texts(
        self,
        texts: list[str],
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Returns L2-normalized CLS embeddings directly from the backbone,
        bypassing the randomly-initialized projection heads.
        Use this for zero-shot inference (no fine-tuned checkpoint).
        """
        self.eval()
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(dev)

        all_embeddings: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            cls = self._cls_pooling(
                encoded["input_ids"].to(dev),
                encoded["attention_mask"].to(dev),
            )
            all_embeddings.append(F.normalize(cls, dim=-1).cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_texts(
        self,
        texts: list[str],
        encoder: str = "curriculum",
        max_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Convenience method: tokenize a list of strings and return embeddings.
        encoder: "curriculum" or "job"
        """
        self.eval()
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.to(dev)

        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(dev)
            attention_mask = encoded["attention_mask"].to(dev)

            if encoder == "curriculum":
                emb = self.encode_curriculum(input_ids, attention_mask)
            else:
                emb = self.encode_job(input_ids, attention_mask)

            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)
