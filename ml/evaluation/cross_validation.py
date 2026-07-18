"""
Nested 6-fold cross-validation for CurriculumGPT (paper §5.2).

  - StratifiedKFold (6 outer folds) on the binary label.
  - Per fold: fit LDA on TRAIN ONLY (leakage prevention), train CurriculumGPT,
    evaluate on the held-out fold.
  - Reports accuracy / precision / recall / F1 / AUC, mean ± std across folds.

Run:
    python -m ml.evaluation.cross_validation --epochs 5 --folds 6
    python -m ml.evaluation.cross_validation --smoke      # tiny, fast CPU check
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from ml.components import topic_coverage
from ml.evaluation.baselines import evaluate_scores
from ml.models.cgpt_features import build_batch
from ml.training.cgpt_trainer import CGPTConfig, CGPTTrainer

logger = logging.getLogger(__name__)

_PAIRS = Path(__file__).resolve().parents[2] / "data" / "labeled" / "pairs.parquet"
_RESULTS = Path(__file__).resolve().parent / "results"


def _rows_from_df(df: pd.DataFrame) -> list[dict]:
    return df.to_dict("records")


@torch.no_grad()
def _predict(model, rows, device, batch_size=16) -> np.ndarray:
    model.eval()
    tok = model.bge.tokenizer
    scores: list[float] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = build_batch([r["curriculum_text"] for r in chunk],
                            [r["job_text"] for r in chunk], tok,
                            pub_years=[r.get("pub_year") for r in chunk], device=device)
        out = model(batch)
        scores.extend(out["alignment_score"].cpu().tolist())
    return np.array(scores)


def run_cv(epochs: int, folds: int, smoke: bool, seed: int = 42) -> dict:
    df = pd.read_parquet(_PAIRS)
    if smoke:
        df = df.sample(min(len(df), 60), random_state=seed).reset_index(drop=True)
        folds = min(folds, 3)
        epochs = min(epochs, 2)
    labels = df["label"].to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics: list[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(df, labels), start=1):
        train_df = df.iloc[tr_idx].reset_index(drop=True)
        test_df = df.iloc[te_idx].reset_index(drop=True)
        logger.info("Fold %d/%d — train=%d test=%d", fold, folds, len(train_df), len(test_df))

        # Leakage prevention: refit LDA on TRAIN ONLY for this fold.
        topic_coverage._reset_cache()
        train_texts = train_df["curriculum_text"].tolist() + train_df["job_text"].tolist()
        topic_coverage.fit_lda(train_texts)

        cfg = CGPTConfig(epochs=epochs, batch_size=8 if smoke else 16,
                        device=device, fit_lda=False, save_every_n_epochs=10**6,
                        curriculum_learning=not smoke)
        trainer = CGPTTrainer(cfg)
        model = trainer.train(_rows_from_df(train_df))

        scores = _predict(model, _rows_from_df(test_df), device)
        m = evaluate_scores(scores, test_df["label"].to_numpy())
        m["fold"] = fold
        fold_metrics.append(m)
        logger.info("Fold %d metrics: %s", fold, m)

    summary = _summarize(fold_metrics)
    _RESULTS.mkdir(parents=True, exist_ok=True)
    out = {"folds": fold_metrics, "summary": summary,
           "config": {"epochs": epochs, "n_folds": folds, "smoke": smoke,
                      "n_pairs": len(df), "device": device}}
    (_RESULTS / "cv_results.json").write_text(json.dumps(out, indent=2))
    logger.info("CV summary: %s", summary)
    return out


def _summarize(fold_metrics: list[dict]) -> dict:
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None]
        if vals:
            summary[k] = {"mean": round(float(np.mean(vals)), 4),
                          "std": round(float(np.std(vals)), 4)}
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="tiny fast CPU check")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_cv(args.epochs, args.folds, args.smoke, args.seed)


if __name__ == "__main__":
    main()
