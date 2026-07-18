"""
Permutation-based component ablation (paper §5.4, Table 2).

For each of the 7 components, replace its score with Gaussian noise (matched
mean/std), recompute the fused alignment score, and measure the accuracy drop
vs. the full system. A larger drop = more important component.

Run:
    python -m ml.evaluation.ablation --checkpoint checkpoints/cgpt_epochXX.pt
    python -m ml.evaluation.ablation --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.components import topic_coverage
from ml.evaluation.baselines import evaluate_scores
from ml.models.cgpt_features import build_batch
from ml.models.curriculum_gpt import COMPONENT_NAMES, CurriculumGPT

logger = logging.getLogger(__name__)

_PAIRS = Path(__file__).resolve().parents[2] / "data" / "labeled" / "pairs.parquet"
_RESULTS = Path(__file__).resolve().parent / "results"


@torch.no_grad()
def _component_matrix(model, rows, device, batch_size=16):
    """Return (N, 7) raw component scores and the fusion weights."""
    model.eval()
    tok = model.bge.tokenizer
    comps, weights = [], None
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = build_batch([r["curriculum_text"] for r in chunk],
                            [r["job_text"] for r in chunk], tok,
                            pub_years=[r.get("pub_year") for r in chunk], device=device)
        out = model(batch)
        comps.append(out["component_scores"].cpu().numpy())
        weights = out["weights"].cpu().numpy()
    return np.concatenate(comps), weights


def _fused(component_matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return component_matrix @ weights


def run_ablation(checkpoint: str | None, smoke: bool, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(_PAIRS)
    if smoke:
        df = df.sample(min(len(df), 60), random_state=seed).reset_index(drop=True)
    labels = df["label"].to_numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # LDA must be fitted for topic scores; fit on this data (eval-only context).
    if not topic_coverage.is_fitted():
        topic_coverage.fit_lda(df["curriculum_text"].tolist() + df["job_text"].tolist())

    model = CurriculumGPT().to(device)
    if checkpoint and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state["model_state_dict"], strict=False)
        logger.info("Loaded checkpoint %s", checkpoint)

    rows = df.to_dict("records")
    comp_matrix, weights = _component_matrix(model, rows, device)

    full_scores = _fused(comp_matrix, weights)
    full_acc = evaluate_scores(full_scores, labels)["accuracy"]
    logger.info("Full system accuracy: %.4f", full_acc)

    results = []
    for idx, name in enumerate(COMPONENT_NAMES):
        permuted = comp_matrix.copy()
        col = permuted[:, idx]
        permuted[:, idx] = rng.normal(col.mean(), col.std() + 1e-6, size=len(col)).clip(0, 1)
        acc = evaluate_scores(_fused(permuted, weights), labels)["accuracy"]
        results.append({"component": name, "weight": round(float(weights[idx]), 4),
                        "ablated_accuracy": acc, "delta_accuracy": round(full_acc - acc, 4)})
        logger.info("Ablate %-10s → acc %.4f (Δ %.4f)", name, acc, full_acc - acc)

    results.sort(key=lambda r: r["delta_accuracy"], reverse=True)
    out = {"full_accuracy": full_acc, "components": results}
    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / "ablation.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_ablation(args.checkpoint, args.smoke)


if __name__ == "__main__":
    main()
