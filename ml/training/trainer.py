"""
Training pipeline for BGEBiEncoder and CurriculumTransformer.

Usage:
    python scripts/train.py --model bge --epochs 10 --batch-size 32
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
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from ml.models.alignment import AlignmentModel
from ml.models.bi_encoder import BGEBiEncoder
from ml.models.transformer import CurriculumTransformer
from ml.training.losses import AlignmentTrainingLoss

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
