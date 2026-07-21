"""
Training pipeline for BGEBiEncoder and CurriculumTransformer.

Usage:
    python scripts/train.py --model bge --epochs 10 --batch-size 32

FrozenBackboneTrainer (below) is the memory-safe alternative to the plain
Trainer class above for the "bge" model type: it never loads BAAI/bge-large-en-v1.5
during training at all. Pair it with ml.training.precompute.precompute_and_cache(),
which runs the backbone exactly once in inference mode and caches the raw
embeddings; FrozenBackboneTrainer then trains only ProjectionHead instances
on those cached vectors via InfoNCE. See scripts/train_bge_frozen.py.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ml.models.alignment import AlignmentModel
from ml.models.bi_encoder import BGEBiEncoder, ProjectionHead
from ml.models.transformer import CurriculumTransformer
from ml.training.losses import AlignmentTrainingLoss, InfoNCELoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    model_type: str = "bge"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 256
    val_split: float = 0.1
    temperature: float = 0.07
    num_skills: int = 150
    output_dir: str = "checkpoints"
    save_every_n_epochs: int = 2
    device: str = "auto"


class AlignmentDataset(Dataset):
    """
    Expects a list of dicts: {"curriculum": str, "job": str, "skills": list[int]}
    skills is a multi-hot vector of length num_skills.
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
        num_skills: int = 150,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_skills = num_skills

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        curr_enc = self.tokenizer(
            sample["curriculum"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        job_enc = self.tokenizer(
            sample["job"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        skill_ids: list[int] = sample.get("skills", [])
        skill_vec = torch.zeros(self.num_skills)
        for s in skill_ids:
            if 0 <= s < self.num_skills:
                skill_vec[s] = 1.0

        return {
            "curriculum_ids": curr_enc["input_ids"].squeeze(0),
            "curriculum_mask": curr_enc["attention_mask"].squeeze(0),
            "job_ids": job_enc["input_ids"].squeeze(0),
            "job_mask": job_enc["attention_mask"].squeeze(0),
            "skill_targets": skill_vec,
        }


def collate_fn(batch: list[dict]) -> dict:
    return {
        "curriculum_ids": torch.stack([b["curriculum_ids"] for b in batch]),
        "curriculum_mask": torch.stack([b["curriculum_mask"] for b in batch]),
        "job_ids": torch.stack([b["job_ids"] for b in batch]),
        "job_mask": torch.stack([b["job_mask"] for b in batch]),
        "skill_targets": torch.stack([b["skill_targets"] for b in batch]),
    }


class Trainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto"
            else config.device
        )
        logger.info("Training on %s", self.device)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(self) -> nn.Module:
        if self.config.model_type == "bge":
            return BGEBiEncoder(num_skills=self.config.num_skills)
        elif self.config.model_type == "custom":
            return CurriculumTransformer(num_skills=self.config.num_skills)
        raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _get_tokenizer(self) -> AutoTokenizer:
        if self.config.model_type == "bge":
            return AutoTokenizer.from_pretrained(
                "BAAI/bge-large-en-v1.5",
                cache_dir=os.getenv("MODEL_CACHE_DIR", ".model_cache"),
            )
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def train(self, samples: list[dict]) -> None:
        cfg = self.config
        tokenizer = self._get_tokenizer()
        model = self._build_model().to(self.device)

        dataset = AlignmentDataset(samples, tokenizer, cfg.max_length, cfg.num_skills)
        val_size = max(1, int(len(dataset) * cfg.val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collate_fn)

        criterion = AlignmentTrainingLoss(temperature=cfg.temperature)
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        total_steps = len(train_loader) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        metrics_log: list[dict] = []

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._run_epoch(model, train_loader, criterion, optimizer, scheduler)
            val_loss = self._eval_epoch(model, val_loader, criterion)

            metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            metrics_log.append(metrics)
            logger.info("Epoch %d/%d — train_loss=%.4f val_loss=%.4f", epoch, cfg.epochs, train_loss, val_loss)

            if epoch % cfg.save_every_n_epochs == 0 or epoch == cfg.epochs:
                self._save_checkpoint(model, optimizer, epoch, val_loss)

        with (self.output_dir / "training_metrics.json").open("w") as f:
            json.dump(metrics_log, f, indent=2)

        logger.info("Training complete. Checkpoints saved to %s", self.output_dir)

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: AlignmentTrainingLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
    ) -> float:
        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = model(
                batch["curriculum_ids"], batch["curriculum_mask"],
                batch["job_ids"], batch["job_mask"],
            )

            losses = criterion(
                curriculum_embeddings=output["curriculum_embedding"],
                job_embeddings=output["job_embedding"],
                curriculum_skill_logits=output["curriculum_skills"],
                job_skill_logits=output["job_skills"],
                curriculum_skill_targets=batch["skill_targets"],
                job_skill_targets=batch["skill_targets"],
            )

            loss = losses["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            optimizer.step()
            scheduler.step()  # type: ignore[union-attr]

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: AlignmentTrainingLoss,
    ) -> float:
        model.eval()
        total_loss = 0.0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = model(
                batch["curriculum_ids"], batch["curriculum_mask"],
                batch["job_ids"], batch["job_mask"],
            )
            losses = criterion(
                output["curriculum_embedding"],
                output["job_embedding"],
                output["curriculum_skills"],
                output["job_skills"],
                batch["skill_targets"],
                batch["skill_targets"],
            )
            total_loss += losses["loss"].item()

        return total_loss / len(loader)

    def _save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
    ) -> None:
        path = self.output_dir / f"checkpoint_epoch{epoch:02d}_loss{val_loss:.4f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": asdict(self.config),
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)


# ============================================================================
# Frozen-backbone training: precompute (ml.training.precompute) + this trainer.
#
# The backbone (BAAI/bge-large-en-v1.5, ~335M params) is loaded ONLY by the
# precompute step, in inference mode, and is out of scope before this class
# is ever constructed. FrozenBackboneTrainer holds nothing but two small
# ProjectionHead instances (~656K params each) and trains them via InfoNCE
# on cached embedding vectors — no tokenizer, no backbone forward/backward,
# no backbone gradients or optimizer state at any point in this class.
# ============================================================================

@dataclass
class FrozenBackboneConfig:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    temperature: float = 0.07
    proj_dim: int = 256
    hidden_dim: int = 1024   # must match the cached embeddings' dimensionality
    val_split: float = 0.1
    output_dir: str = "checkpoints"
    save_every_n_epochs: int = 2
    device: str = "cpu"      # heads are tiny; CPU is fine and keeps this off the GPU/OOM budget entirely
    seed: int = 42


class CachedPairDataset(Dataset):
    """
    Positive-leaning (curriculum, job) pairs, resolved to their precomputed
    raw embeddings by exact text lookup against the cache. Only rows with
    train_weight > 0 belong here — InfoNCELoss treats the diagonal of each
    batch as the anchor-positive set and everything else in the batch as
    negatives, so a row with zero weight contributes nothing and should be
    excluded rather than included at weight 0 (matches InfoNCELoss's own
    docstring contract). Each row's optional `train_weight` (defaults to 1.0
    if absent, e.g. plain label==1 rows) scales how strongly that anchor pair
    is pulled together — see InfoNCELoss for the graded-target use case
    (e.g. aligned=1.0, partial=0.5).
    """

    def __init__(
        self,
        pairs: list[dict],
        curriculum_embeddings: torch.Tensor,
        job_embeddings: torch.Tensor,
        curriculum_text_to_idx: dict[str, int],
        job_text_to_idx: dict[str, int],
    ) -> None:
        self.curriculum_embeddings = curriculum_embeddings
        self.job_embeddings = job_embeddings
        self.samples: list[tuple[int, int]] = []
        self.weights: list[float] = []
        missing = 0
        for row in pairs:
            c_idx = curriculum_text_to_idx.get(row["curriculum_text"])
            j_idx = job_text_to_idx.get(row["job_text"])
            if c_idx is None or j_idx is None:
                missing += 1
                continue
            self.samples.append((c_idx, j_idx))
            self.weights.append(float(row.get("train_weight", 1.0)))
        if missing:
            logger.warning(
                "%d/%d pairs had no matching cached embedding (cache built from "
                "different texts?) — skipped", missing, len(pairs),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        c_idx, j_idx = self.samples[idx]
        return {
            "curriculum_embedding": self.curriculum_embeddings[c_idx],
            "job_embedding": self.job_embeddings[j_idx],
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
        }


def _cached_collate(batch: list[dict]) -> dict:
    return {
        "curriculum_embedding": torch.stack([b["curriculum_embedding"] for b in batch]),
        "job_embedding": torch.stack([b["job_embedding"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
    }


class FrozenBackboneTrainer:
    """
    Trains only BGEBiEncoder's curriculum_head/job_head projection heads,
    via InfoNCE, on embeddings produced by ml.training.precompute — the
    backbone never enters this class.

    The saved checkpoint stores ONLY the two heads' state dicts under
    "model_state_dict", using the exact key names BGEBiEncoder uses
    internally (curriculum_head.*, job_head.*). ml/models/alignment.py's
    existing checkpoint loader already calls
    `model.load_state_dict(state["model_state_dict"], strict=False)`, so it
    loads this partial checkpoint with no code changes: the trained heads
    are applied, and the pretrained backbone weights (loaded independently
    at inference time from BAAI/bge-large-en-v1.5) are left untouched.
    """

    def __init__(self, config: FrozenBackboneConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self._seed_everything(config.seed)

        self.curriculum_head = ProjectionHead(config.hidden_dim, config.proj_dim).to(self.device)
        self.job_head = ProjectionHead(config.hidden_dim, config.proj_dim).to(self.device)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "FrozenBackboneTrainer on %s — %d trainable params (heads only, no backbone)",
            self.device, self._n_trainable_params(),
        )

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import random
        random.seed(seed)
        torch.manual_seed(seed)

    def _n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.curriculum_head.parameters() if p.requires_grad) + \
               sum(p.numel() for p in self.job_head.parameters() if p.requires_grad)

    def train(self, cache: dict, pairs: list[dict]) -> dict:
        cfg = self.config

        curriculum_text_to_idx = {t: i for i, t in enumerate(cache["curriculum_texts"])}
        job_text_to_idx = {t: i for i, t in enumerate(cache["job_texts"])}

        dataset = CachedPairDataset(
            pairs, cache["curriculum_embeddings"], cache["job_embeddings"],
            curriculum_text_to_idx, job_text_to_idx,
        )
        if len(dataset) < 2:
            raise ValueError(
                f"Only {len(dataset)} positive pair(s) resolved against the cache — "
                "need at least 2 for InfoNCE's in-batch negatives to mean anything."
            )

        val_size = max(1, int(len(dataset) * cfg.val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        batch_size = min(cfg.batch_size, max(2, train_size))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=len(train_ds) > batch_size, collate_fn=_cached_collate)
        val_loader = DataLoader(val_ds, batch_size=min(cfg.batch_size, max(1, val_size)),
                                collate_fn=_cached_collate)

        criterion = InfoNCELoss(temperature=cfg.temperature)
        params = list(self.curriculum_head.parameters()) + list(self.job_head.parameters())
        optimizer = AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        total_steps = max(1, len(train_loader)) * cfg.epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        metrics_log: list[dict] = []
        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._run_epoch(train_loader, criterion, optimizer, scheduler)
            val_loss = self._eval_epoch(val_loader, criterion) if len(val_ds) > 0 else train_loss

            metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            metrics_log.append(metrics)
            logger.info("Epoch %d/%d — train_loss=%.4f val_loss=%.4f",
                        epoch, cfg.epochs, train_loss, val_loss)

            if epoch % cfg.save_every_n_epochs == 0 or epoch == cfg.epochs:
                self._save_checkpoint(epoch, val_loss)

        with (self.output_dir / "frozen_backbone_training_metrics.json").open("w") as f:
            json.dump(metrics_log, f, indent=2)

        logger.info("Frozen-backbone training complete. Checkpoints saved to %s", self.output_dir)
        return {"metrics": metrics_log, "output_dir": str(self.output_dir),
                "n_train_pairs": train_size, "n_val_pairs": val_size}

    def _run_epoch(self, loader, criterion, optimizer, scheduler) -> float:
        self.curriculum_head.train()
        self.job_head.train()
        total_loss = 0.0

        for batch in loader:
            curr_raw = batch["curriculum_embedding"].to(self.device)
            job_raw = batch["job_embedding"].to(self.device)
            weights = batch["weight"].to(self.device)

            curr_emb = F.normalize(self.curriculum_head(curr_raw), dim=-1)
            job_emb = F.normalize(self.job_head(job_raw), dim=-1)

            loss = criterion(curr_emb, job_emb, weights=weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.curriculum_head.parameters()) + list(self.job_head.parameters()),
                self.config.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / max(1, len(loader))

    @torch.no_grad()
    def _eval_epoch(self, loader, criterion) -> float:
        self.curriculum_head.eval()
        self.job_head.eval()
        total_loss = 0.0

        for batch in loader:
            curr_raw = batch["curriculum_embedding"].to(self.device)
            job_raw = batch["job_embedding"].to(self.device)
            weights = batch["weight"].to(self.device)
            curr_emb = F.normalize(self.curriculum_head(curr_raw), dim=-1)
            job_emb = F.normalize(self.job_head(job_raw), dim=-1)
            total_loss += criterion(curr_emb, job_emb, weights=weights).item()

        return total_loss / max(1, len(loader))

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        path = self.output_dir / f"frozen_epoch{epoch:02d}_loss{val_loss:.4f}.pt"
        # Deliberately partial state dict — only the two heads. BGEBiEncoder's
        # backbone.* and skill_head.* keys are absent on purpose; the existing
        # loader (ml/models/alignment.py) applies this with strict=False, so
        # the pretrained backbone is left exactly as loaded from Hugging Face.
        model_state_dict = {
            **{f"curriculum_head.{k}": v for k, v in self.curriculum_head.state_dict().items()},
            **{f"job_head.{k}": v for k, v in self.job_head.state_dict().items()},
        }
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "val_loss": val_loss,
                "config": asdict(self.config),
                "frozen_backbone": True,
            },
            path,
        )
        logger.info("Saved frozen-backbone checkpoint (heads only): %s", path)
