"""
Loss functions for training the alignment model.

InfoNCELoss: standard in-batch contrastive loss (SimCLR / CLIP style).
  - Positive pair: (curriculum_i, job_i)
  - Negatives: all other (curriculum_i, job_j≠i) in the batch
  - Temperature τ controls sharpness (default 0.07)

HardNegativeInfoNCE: same as InfoNCE but mines the hardest negatives
  from a larger candidate pool and includes them in the denominator.

SkillMultiLabelLoss: BCE loss for the skill classification heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    In-batch contrastive loss (symmetric).
    Both curriculum→job and job→curriculum directions are averaged.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        curriculum_embeddings: torch.Tensor,  # (B, D), L2-normalized
        job_embeddings: torch.Tensor,          # (B, D), L2-normalized
    ) -> torch.Tensor:
        batch_size = curriculum_embeddings.size(0)

        # Similarity matrix (B, B)
        logits = torch.matmul(curriculum_embeddings, job_embeddings.T) / self.temperature

        # Diagonal = positive pairs
        labels = torch.arange(batch_size, device=logits.device)

        loss_c2j = F.cross_entropy(logits, labels)
        loss_j2c = F.cross_entropy(logits.T, labels)

        return (loss_c2j + loss_j2c) / 2.0


class HardNegativeInfoNCE(nn.Module):
    """
    InfoNCE with hard negative augmentation.
    Mines top-k hard negatives per sample and adds them to the denominator.

    hard_neg_weight: extra weight for hard negatives (default 1.0 = equal weight)
    """

    def __init__(self, temperature: float = 0.07, hard_neg_weight: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_neg_weight = hard_neg_weight

    def forward(
        self,
        curriculum_embeddings: torch.Tensor,  # (B, D)
        job_embeddings: torch.Tensor,          # (B, D)
    ) -> torch.Tensor:
        B = curriculum_embeddings.size(0)
        logits = torch.matmul(curriculum_embeddings, job_embeddings.T) / self.temperature

        # Create labels (diagonal = positive)
        labels = torch.arange(B, device=logits.device)

        # Hard negatives: for each query, find the most similar non-positive
        with torch.no_grad():
            masked_logits = logits.clone()
            masked_logits.fill_diagonal_(float("-inf"))
            hard_neg_indices = masked_logits.argmax(dim=-1)  # (B,)

        # Build augmented logits: [pos, all-in-batch, hard-neg (reweighted)]
        pos_logits = logits[torch.arange(B), labels].unsqueeze(1)  # (B, 1)
        hard_neg_logits = (
            logits[torch.arange(B), hard_neg_indices].unsqueeze(1) * self.hard_neg_weight
        )  # (B, 1)

        augmented = torch.cat([logits, hard_neg_logits], dim=-1)  # (B, B+1)
        loss = F.cross_entropy(augmented, labels)
        return loss


class SkillMultiLabelLoss(nn.Module):
    """
    Binary cross-entropy for multi-label skill classification.
    pos_weight: handles class imbalance (most skills are absent).
    """

    def __init__(self, pos_weight: float = 5.0) -> None:
        super().__init__()
        self.pos_weight_val = pos_weight

    def forward(
        self,
        pred_logits: torch.Tensor,   # (B, num_skills)
        target: torch.Tensor,         # (B, num_skills) binary
    ) -> torch.Tensor:
        pw = torch.full_like(pred_logits[0], self.pos_weight_val)
        return F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pw)


class AlignmentClassificationLoss(nn.Module):
    """
    Cross-entropy on the binary aligned/misaligned head of CurriculumGPT.
    Enables accuracy / F1 / AUC evaluation against silver labels.
    """

    def forward(self, cls_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(cls_logits, labels.long())


class CurriculumGPTLoss(nn.Module):
    """
    Combined loss for the seven-component model:
        L = alpha * InfoNCE(curriculum, job) + beta * CE(cls_logits, label)

    InfoNCE shapes the BGE embedding space; the classification term aligns the
    fused score with the binary labels. Default alpha=0.7, beta=0.3.
    """

    def __init__(self, temperature: float = 0.07, alpha: float = 0.7, beta: float = 0.3) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.contrastive = InfoNCELoss(temperature)
        self.classification = AlignmentClassificationLoss()

    def forward(
        self,
        curriculum_embeddings: torch.Tensor,
        job_embeddings: torch.Tensor,
        cls_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        contrastive = self.contrastive(curriculum_embeddings, job_embeddings)
        classification = self.classification(cls_logits, labels)
        total = self.alpha * contrastive + self.beta * classification
        return {
            "loss": total,
            "contrastive_loss": contrastive,
            "classification_loss": classification,
        }


class AlignmentTrainingLoss(nn.Module):
    """
    Combined loss = α * InfoNCE + β * skill_multilabel.
    Default weights: α=1.0, β=0.3.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 1.0,
        beta: float = 0.3,
        use_hard_negatives: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.contrastive = (
            HardNegativeInfoNCE(temperature) if use_hard_negatives else InfoNCELoss(temperature)
        )
        self.skill_loss = SkillMultiLabelLoss()

    def forward(
        self,
        curriculum_embeddings: torch.Tensor,
        job_embeddings: torch.Tensor,
        curriculum_skill_logits: torch.Tensor,
        job_skill_logits: torch.Tensor,
        curriculum_skill_targets: torch.Tensor,
        job_skill_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        contrastive = self.contrastive(curriculum_embeddings, job_embeddings)

        skill_loss_curr = self.skill_loss(curriculum_skill_logits, curriculum_skill_targets)
        skill_loss_job = self.skill_loss(job_skill_logits, job_skill_targets)
        skill_loss = (skill_loss_curr + skill_loss_job) / 2.0

        total = self.alpha * contrastive + self.beta * skill_loss

        return {
            "loss": total,
            "contrastive_loss": contrastive,
            "skill_loss": skill_loss,
        }
