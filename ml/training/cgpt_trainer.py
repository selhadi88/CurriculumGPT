"""
Trainer for the seven-component CurriculumGPT model.

Reads labeled pairs from `data/labeled/pairs.parquet` (built by
scripts/build_dataset.py) and trains with:
  - combined InfoNCE + classification loss (ml.training.losses.CurriculumGPTLoss)
  - curriculum learning: 3 progressive difficulty stages over the epoch budget
  - Monte Carlo dropout for uncertainty at eval (n_MC=10)
  - AdamW + cosine schedule with warmup

Kept separate from ml.training.trainer.Trainer so the BGE/custom paths are untouched.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from ml.components import topic_coverage
from ml.models.cgpt_features import build_batch
from ml.models.curriculum_gpt import CurriculumGPT
from ml.training.losses import CurriculumGPTLoss

logger = logging.getLogger(__name__)


@dataclass
class CGPTConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 256
    temperature: float = 0.07
    alpha: float = 0.7
    beta: float = 0.3
    n_mc: int = 10
    curriculum_learning: bool = True
    output_dir: str = "checkpoints"
    save_every_n_epochs: int = 10
    device: str = "auto"
    fit_lda: bool = True
    seed: int = 42


# ----------------------------------------------------------------------------- data

class PairDataset(Dataset):
    """Holds raw labeled rows; feature tensors are built per-batch in collate."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def _difficulty_rank(row: dict) -> str:
    """Easy = clearly aligned/misaligned; hard = ambiguous middle."""
    sim = float(row.get("sim_target", 0.5))
    if sim > 0.6 or sim < 0.15:
        return "easy"
    if sim < 0.35:
        return "medium"
    return "hard"


def _stage_rows(rows: list[dict], stage: int) -> list[dict]:
    """Curriculum-learning subset for a given stage (1=easy, 2=+medium, 3=all)."""
    if stage <= 1:
        return [r for r in rows if _difficulty_rank(r) == "easy"]
    if stage == 2:
        return [r for r in rows if _difficulty_rank(r) in ("easy", "medium")]
    return rows


# ----------------------------------------------------------------------------- trainer

class CGPTTrainer:
    def __init__(self, config: CGPTConfig) -> None:
        self.cfg = config
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if config.device == "auto" else config.device
        )
        self._seed_everything(config.seed)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("CGPTTrainer on %s", self.device)

    @staticmethod
    def _seed_everything(seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _collate(self, rows: list[dict], tokenizer) -> tuple[dict, torch.Tensor]:
        curr = [r["curriculum_text"] for r in rows]
        jobs = [r["job_text"] for r in rows]
        years = [r.get("pub_year") for r in rows]
        labels = torch.tensor([int(r["label"]) for r in rows], device=self.device)
        batch = build_batch(curr, jobs, tokenizer, pub_years=years,
                            max_length=self.cfg.max_length, device=self.device)
        return batch, labels

    def train(self, rows: list[dict]) -> CurriculumGPT:
        cfg = self.cfg
        model = CurriculumGPT().to(self.device)
        tokenizer = model.bge.tokenizer

        # Fit LDA on the training texts (leakage-safe: caller passes train-only rows).
        if cfg.fit_lda:
            texts = [r["curriculum_text"] for r in rows] + [r["job_text"] for r in rows]
            topic_coverage.fit_lda(texts)

        criterion = CurriculumGPTLoss(cfg.temperature, cfg.alpha, cfg.beta)
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

        # Total steps depend on curriculum stages; approximate with full set.
        steps_per_epoch = max(1, len(rows) // cfg.batch_size)
        total_steps = steps_per_epoch * cfg.epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, int(total_steps * cfg.warmup_ratio), total_steps
        )

        metrics_log: list[dict] = []
        for epoch in range(1, cfg.epochs + 1):
            stage = self._stage_for_epoch(epoch)
            epoch_rows = _stage_rows(rows, stage) if cfg.curriculum_learning else rows
            if not epoch_rows:
                epoch_rows = rows
            loader = DataLoader(
                PairDataset(epoch_rows), batch_size=cfg.batch_size, shuffle=True,
                collate_fn=lambda b: b,
            )
            loss = self._run_epoch(model, loader, tokenizer, criterion, optimizer, scheduler)
            metrics_log.append({"epoch": epoch, "stage": stage,
                                "n_pairs": len(epoch_rows), "train_loss": loss})
            logger.info("Epoch %d/%d [stage %d, %d pairs] loss=%.4f",
                        epoch, cfg.epochs, stage, len(epoch_rows), loss)
            if epoch % cfg.save_every_n_epochs == 0 or epoch == cfg.epochs:
                self._save(model, optimizer, epoch, loss)

        with (self.output_dir / "cgpt_training_metrics.json").open("w") as f:
            json.dump(metrics_log, f, indent=2)
        logger.info("CurriculumGPT training complete → %s", self.output_dir)
        return model

    def _stage_for_epoch(self, epoch: int) -> int:
        third = max(1, self.cfg.epochs // 3)
        if epoch <= third:
            return 1
        if epoch <= 2 * third:
            return 2
        return 3

    def _run_epoch(self, model, loader, tokenizer, criterion, optimizer, scheduler) -> float:
        model.train()
        total = 0.0
        n = 0
        for rows in loader:
            batch, labels = self._collate(rows, tokenizer)
            out = model(batch)
            losses = criterion(
                curriculum_embeddings=out["curriculum_embedding"],
                job_embeddings=out["job_embedding"],
                cls_logits=out["cls_logits"],
                labels=labels,
            )
            loss = losses["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            n += 1
        return total / max(1, n)

    @torch.no_grad()
    def mc_predict(self, model: CurriculumGPT, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo dropout prediction. Returns (mean_scores, std_uncertainty),
        each shape (len(rows),). Dropout stays active across n_MC passes.
        """
        tokenizer = model.bge.tokenizer
        model.train()  # keep dropout active
        loader = DataLoader(PairDataset(rows), batch_size=self.cfg.batch_size,
                            shuffle=False, collate_fn=lambda b: b)
        all_means: list[np.ndarray] = []
        all_stds: list[np.ndarray] = []
        for batch_rows in loader:
            batch, _ = self._collate(batch_rows, tokenizer)
            passes = []
            for _ in range(self.cfg.n_mc):
                out = model(batch)
                passes.append(out["alignment_score"].cpu().numpy())
            stacked = np.stack(passes)            # (n_mc, B)
            all_means.append(stacked.mean(axis=0))
            all_stds.append(stacked.std(axis=0))
        model.eval()
        return np.concatenate(all_means), np.concatenate(all_stds)

    def _save(self, model, optimizer, epoch: int, loss: float) -> None:
        path = self.output_dir / f"cgpt_epoch{epoch:02d}_loss{loss:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": asdict(self.cfg),
        }, path)
        logger.info("Saved checkpoint: %s", path)
