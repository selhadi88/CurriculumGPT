"""
Evaluate the current alignment model against the LLM-graded gold set.

Ground truth: `llm_label` in data/labeled/gold_llm.parquet (aligned/partial/
not_aligned, from llm_label_graded()):
  - Binary target (accuracy/precision/recall/F1/AUC): aligned OR partial -> 1,
    not_aligned -> 0.
  - 3-level relevance (NDCG@10): aligned=2, partial=1, not_aligned=0.

Model scores: AlignmentModel.score() (ml/models/alignment.py) — the same
production scoring path the backend uses, respecting MODEL_TYPE/MODEL_CHECKPOINT
env vars (defaults to BGE zero-shot cosine, per docker-compose.yml's
MODEL_TYPE=bge / unset MODEL_CHECKPOINT).

Binary threshold: median of the model's own scores on the evaluated subset —
same convention as ml/evaluation/compare_models.py (`threshold=np.median(s)`),
applied once globally and once per domain (score distributions differ by
domain, so a single fixed cutoff wouldn't split each subset meaningfully).

Usage:
    python scripts/evaluate_gold.py
    python scripts/evaluate_gold.py --input data/labeled/gold_llm.parquet --out results/gold_evaluation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.evaluation.baselines import evaluate_scores  # noqa: E402
from ml.models.alignment import AlignmentModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("evaluate_gold")

_RELEVANCE = {"aligned": 2, "partial": 1, "not_aligned": 0}
_BINARY = {"aligned": 1, "partial": 1, "not_aligned": 0}


def _model_scores(df: pd.DataFrame) -> np.ndarray:
    model_type = os.getenv("MODEL_TYPE", "bge")
    checkpoint = os.getenv("MODEL_CHECKPOINT") or None
    logger.info("Scoring %d gold pairs with AlignmentModel(model_type=%r, checkpoint=%r)…",
                len(df), model_type, checkpoint)
    model = AlignmentModel(model_type=model_type, checkpoint_path=checkpoint)
    results = model.score(df["curriculum_text"].tolist(), df["job_text"].tolist(), batch_size=32)
    return np.array([r.overall_score for r in results], dtype=np.float64)


def _dcg(relevances: list[int], k: int) -> float:
    return sum((2 ** rel - 1) / math.log2(rank + 2) for rank, rel in enumerate(relevances[:k]))


def _ndcg_at_10(scores: np.ndarray, relevance: np.ndarray) -> float:
    order = np.argsort(-scores)
    ranked = relevance[order].tolist()
    ideal = sorted(relevance.tolist(), reverse=True)
    dcg = _dcg(ranked, 10)
    idcg = _dcg(ideal, 10)
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def _eval_subset(scores: np.ndarray, binary: np.ndarray, relevance: np.ndarray) -> dict:
    threshold = float(np.median(scores))
    out = evaluate_scores(scores, binary, threshold=threshold)
    out["threshold"] = round(threshold, 4)
    out["ndcg_at_10"] = _ndcg_at_10(scores, relevance)
    out["n"] = int(len(scores))
    out["positive_rate"] = round(float(binary.mean()), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/labeled/gold_llm.parquet")
    ap.add_argument("--out", type=str, default="results/gold_evaluation.json")
    args = ap.parse_args()

    in_path = Path(args.input)
    df = pd.read_parquet(in_path)
    if "llm_label" not in df.columns:
        raise RuntimeError(f"{in_path} has no llm_label column — run llm_label_graded() first.")
    missing = int(df["llm_label"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} rows in {in_path} have no llm_label (ungraded) — cannot evaluate.")

    relevance = df["llm_label"].map(_RELEVANCE).to_numpy()
    binary = df["llm_label"].map(_BINARY).to_numpy()

    scores = _model_scores(df)

    overall = _eval_subset(scores, binary, relevance)

    per_domain: dict[str, dict] = {}
    for domain in sorted(df["course_domain"].dropna().unique()):
        mask = (df["course_domain"] == domain).to_numpy()
        per_domain[domain] = _eval_subset(scores[mask], binary[mask], relevance[mask])

    result = {
        "input": str(in_path),
        "model_type": os.getenv("MODEL_TYPE", "bge"),
        "model_checkpoint": os.getenv("MODEL_CHECKPOINT") or None,
        "ground_truth": "llm_label (graded via LLM: aligned/partial/not_aligned)",
        "binary_mapping": {"aligned": 1, "partial": 1, "not_aligned": 0},
        "relevance_mapping": {"aligned": 2, "partial": 1, "not_aligned": 0},
        "threshold_method": ("median of model scores on the evaluated subset "
                              "(matches ml/evaluation/compare_models.py convention)"),
        "overall": overall,
        "per_domain": per_domain,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)

    logger.info("=== Overall (n=%d) ===", overall["n"])
    for k, v in overall.items():
        logger.info("  %s: %s", k, v)
    logger.info("=== Per-domain ===")
    for domain, m in per_domain.items():
        logger.info("  %-18s n=%-4d acc=%-7s prec=%-7s rec=%-7s f1=%-7s auc=%-7s ndcg@10=%-7s pos_rate=%s",
                    domain, m["n"], m["accuracy"], m["precision"], m["recall"], m["f1"],
                    m["auc"], m["ndcg_at_10"], m["positive_rate"])


if __name__ == "__main__":
    main()
