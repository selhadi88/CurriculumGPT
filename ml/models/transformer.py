"""
CurriculumTransformer — full transformer encoder implemented from scratch.

Architecture:
  TokenEmbedding + PositionalEncoding
  → 6 × TransformerEncoderLayer (pre-norm, 8-head MHA + GELU FFN)
  → mean-pool with attention mask
  → AlignmentHead (pairwise cosine → score)
  → SkillHead (multi-label, 150 skills)

No HuggingFace dependency. ~10M parameters.
Use as a lightweight alternative to BGEBiEncoder or as a distillation target.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]  # type: ignore[index]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.W_q(query))
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask: (B, 1, 1, T) — 0 = attend, large-negative = ignore
            scores = scores + mask

        attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn_weights, V)
        return self.W_o(self._merge_heads(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm variant (more stable training than post-norm)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, normed, normed, mask))
        # FFN with residual
        x = x + self.ff(self.norm2(x))
        return x


class CurriculumTransformer(nn.Module):
    """
    6-layer transformer encoder for curriculum-job alignment.
    d_model=256, 8 heads, d_ff=1024 (~10M params).

    Outputs:
        encode(input_ids, attention_mask) → (B, d_model) mean-pooled embedding
        forward(curr_ids, curr_mask, job_ids, job_mask) → dict with scores + embeddings
    """

    def __init__(
        self,
        vocab_size: int = 30_522,   # BERT WordPiece vocab
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_len: int = 512,
        dropout: float = 0.1,
        num_skills: int = 150,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

        # Skill classification head (multi-label)
        self.skill_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_skills),
        )

        # Alignment scoring head (takes concatenated pair embeddings)
        self.alignment_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def _make_padding_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask: (B, T), 1=real token, 0=pad
        # returns additive mask: (B, 1, 1, T)
        return (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (B, d_model) mean-pooled representation."""
        # Scale embeddings as in original "Attention is All You Need"
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.input_norm(x)

        mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            mask = self._make_padding_mask(attention_mask)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.output_norm(x)

        # Masked mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)

        return pooled

    def forward(
        self,
        curriculum_ids: torch.Tensor,
        curriculum_mask: Optional[torch.Tensor],
        job_ids: torch.Tensor,
        job_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        curr_emb = self.encode(curriculum_ids, curriculum_mask)   # (B, D)
        job_emb = self.encode(job_ids, job_mask)                  # (B, D)

        # Cosine similarity (primary alignment signal)
        curr_norm = F.normalize(curr_emb, dim=-1)
        job_norm = F.normalize(job_emb, dim=-1)
        cosine_score = (curr_norm * job_norm).sum(dim=-1, keepdim=True)  # (B, 1)

        # Learned alignment score
        combined = torch.cat([curr_emb, job_emb], dim=-1)  # (B, 2D)
        learned_score = self.alignment_head(combined)       # (B, 1)

        # Blend cosine + learned (0.5/0.5 at init, can be tuned)
        alignment_score = 0.5 * (cosine_score + 1.0) / 2.0 + 0.5 * learned_score

        return {
            "alignment_score": alignment_score.squeeze(-1),   # (B,)
            "curriculum_embedding": curr_emb,
            "job_embedding": job_emb,
            "curriculum_skills": self.skill_head(curr_emb),   # (B, num_skills)
            "job_skills": self.skill_head(job_emb),            # (B, num_skills)
        }
