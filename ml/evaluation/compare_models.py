"""
Model comparison harness (paper Table 1) — the core research deliverable.

Trains/evaluates on a single held-out split (fast) or delegates to nested CV:
  - CurriculumGPT (full 7-component)
  - CurriculumGPT (4-component ablation: semantic+topic+skill+depth)
  - TF-IDF keyword overlap
  - SentenceBERT (all-mpnet-base-v2)
  - BGE semantic-only (zero-shot cosine)

Outputs a comparison table (JSON + console) into the run folder. Each method is
evaluated with accuracy / precision / recall / F1 / AUC on the same test set.

Run:
    python -m ml.evaluation.compare_models --epochs 30
    python -m ml.evaluation.compare_models --smoke
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from ml.components import topic_coverage
from ml.evaluation.baselines import (SentenceBertBaseline, TfidfBaseline,
                                     evaluate_scores)
from ml.evaluation.run_manifest import RunManifest
from ml.models.cgpt_features import build_batch
from ml.models.curriculum_gpt import COMPONENT_NAMES, CurriculumGPT
from ml.training.cgpt_trainer import CGPTConfig, CGPTTrainer

logger = logging.getLogger(__name__)

_PAIRS = None  # resolved at runtime


def _pairs_path():
    import os
    from pathlib import Path
    override = os.getenv("PAIRS_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data" / "labeled" / "pairs.parquet"


@torch.no_grad()
def _cgpt_scores(model, rows, device, active_components=None, batch_size=32):
    """Fused scores; if active_components given, zero out the others before fusing."""
    model.eval()
    tok = model.bge.tokenizer
    out_scores: list[float] = []
    weights = model.fusion_weights().cpu().numpy()
    mask = np.ones(len(COMPONENT_NAMES))
    if active_components is not None:
        mask = np.array([1.0 if n in active_components else 0.0 for n in COMPONENT_NAMES])
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = build_batch([r["curriculum_text"] for r in chunk],
                            [r["job_text"] for r in chunk], tok,
                            pub_years=[r.get("pub_year") for r in chunk], device=device)
        out = model(batch)
        comps = out["component_scores"].cpu().numpy()           # (B, 7)
        w = weights * mask
        w = w / w.sum() if w.sum() > 0 else weights
        out_scores.extend((comps * w).sum(axis=1).tolist())
    return np.array(out_scores)


def _bge_zero_shot_scores(rows, device, batch_size=64):
    from ml.models.bi_encoder import BGEBiEncoder
    model = BGEBiEncoder(); model.eval()
    c = [r["curriculum_text"] for r in rows]
    j = [r["job_text"] for r in rows]
    ce = model.encode_backbone_texts(c, batch_size=batch_size, device=device)
    je = model.encode_backbone_texts(j, batch_size=batch_size, device=device)
    return (ce * je).sum(-1).cpu().numpy().clip(0, 1)


def run(epochs: int, smoke: bool, seed: int = 42) -> dict:
    df = pd.read_parquet(_pairs_path())
    if smoke:
        df = df.sample(min(len(df), 80), random_state=seed).reset_index(drop=True)
        epochs = min(epochs, 3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Comparison on %d pairs (%.1f%% aligned) | device=%s | epochs=%d",
                len(df), 100 * df["label"].mean(), device, epochs)

    train_df, test_df = train_test_split(df, test_size=0.25, stratify=df["label"],
                                         random_state=seed)
    train_rows = train_df.to_dict("records")
    test_rows = test_df.to_dict("records")
    y_test = test_df["label"].to_numpy()

    manifest = RunManifest(config={"epochs": epochs, "smoke": smoke,
                                    "n_pairs": len(df), "device": device, "seed": seed})

    # Leakage-safe LDA: fit on train only.
    topic_coverage._reset_cache()
    topic_coverage.fit_lda(train_df["curriculum_text"].tolist() + train_df["job_text"].tolist())

    results: dict[str, dict] = {}

    # --- Train CurriculumGPT ---
    cfg = CGPTConfig(epochs=epochs, batch_size=16 if not smoke else 8, device=device,
                    fit_lda=False, output_dir=str(manifest.dir / "checkpoints"),
                    save_every_n_epochs=max(5, epochs // 2), curriculum_learning=not smoke)
    trainer = CGPTTrainer(cfg)
    model = trainer.train(train_rows)
    manifest.add_artifact("checkpoint_dir", str(manifest.dir / "checkpoints"))

    # CurriculumGPT (full 7-component)
    s_full = _cgpt_scores(model, test_rows, device)
    results["CurriculumGPT (7-component)"] = evaluate_scores(s_full, y_test, threshold=float(np.median(s_full)))

    # CurriculumGPT (4-component baseline)
    four = {"semantic", "topic", "skill", "depth"}
    s_four = _cgpt_scores(model, test_rows, device, active_components=four)
    results["CurriculumGPT (4-component)"] = evaluate_scores(s_four, y_test, threshold=float(np.median(s_four)))

    # --- Baselines ---
    try:
        bge = _bge_zero_shot_scores(test_rows, device)
        results["BGE semantic-only (zero-shot)"] = evaluate_scores(bge, y_test, threshold=float(np.median(bge)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("BGE baseline failed: %s", exc)

    try:
        tfidf = TfidfBaseline(); tfidf.fit(train_rows)
        s = tfidf.predict_scores(test_rows)
        results["TF-IDF keyword overlap"] = evaluate_scores(s, y_test, threshold=float(np.median(s)))
    except Exception as exc:  # noqa: BLE001
        logger.warning("TF-IDF baseline failed: %s", exc)

    if not smoke:
        try:
            sbert = SentenceBertBaseline()
            s = sbert.predict_scores(test_rows)
            results["SentenceBERT (mpnet)"] = evaluate_scores(s, y_test, threshold=float(np.median(s)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("SentenceBERT baseline failed: %s", exc)

    comparison = {"test_size": len(test_rows), "results": results}
    manifest.add_metrics("model_comparison", comparison)
    _print_table(results)
    logger.info("Comparison saved to %s", manifest.dir)
    return {"manifest_dir": str(manifest.dir), **comparison}


def _print_table(results: dict) -> None:
    print("\n" + "=" * 78)
    print(f"{'Method':<34}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}")
    print("-" * 78)
    ordered = sorted(results.items(), key=lambda kv: -(kv[1].get("accuracy") or 0))
    for name, m in ordered:
        print(f"{name:<34}{m['accuracy']:>8.3f}{m['precision']:>8.3f}"
              f"{m['recall']:>8.3f}{m['f1']:>8.3f}{(m.get('auc') or 0):>8.3f}")
    print("=" * 78 + "\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.epochs, args.smoke, args.seed)


if __name__ == "__main__":
    main()
