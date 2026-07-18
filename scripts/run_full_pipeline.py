"""
Full research pipeline orchestrator.

Runs the complete CurriculumGPT research workflow and saves every artifact into a
single timestamped run folder under outputs/runs/<run_id>/:

    1. (optional) build/label dataset        -> data/labeled/pairs.parquet
    2. train CurriculumGPT (GPU if available) -> <run>/checkpoints/
    3. model comparison vs baselines          -> <run>/model_comparison.json
    4. component ablation (Table 2)           -> <run>/ablation.json
    5. inference latency benchmark            -> <run>/latency.json
    6. statistical validation                 -> <run>/statistics.json
    7. visual HTML report                     -> <run>/report.html

Usage:
    python scripts/run_full_pipeline.py                 # full run (uses existing labels)
    python scripts/run_full_pipeline.py --epochs 30
    python scripts/run_full_pipeline.py --smoke         # fast end-to-end sanity run
    python scripts/run_full_pipeline.py --build-data --n-pairs 1200 --gemini
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("pipeline")

_PAIRS = Path(__file__).parent.parent / "data" / "labeled" / "pairs.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smoke", action="store_true", help="fast end-to-end sanity run")
    ap.add_argument("--build-data", action="store_true", help="(re)build the labeled dataset first")
    ap.add_argument("--n-pairs", type=int, default=1200)
    ap.add_argument("--gemini", action="store_true", help="use Gemini as primary labeler")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=== CurriculumGPT full pipeline | device=%s | smoke=%s ===", device, args.smoke)

    # 1. Dataset -----------------------------------------------------------------
    if args.build_data or not _PAIRS.exists():
        logger.info("[1/7] Building labeled dataset…")
        from scripts.build_dataset import build_pairs, gemini_label
        from ml.evaluation.run_manifest import RUNS_ROOT  # noqa: F401
        df = build_pairs(args.n_pairs, args.seed)
        if args.gemini:
            df = gemini_label(df)
        _PAIRS.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(_PAIRS, index=False)
        logger.info("Dataset written: %s (%d rows)", _PAIRS, len(df))
    else:
        logger.info("[1/7] Using existing dataset: %s", _PAIRS)

    df = pd.read_parquet(_PAIRS)
    logger.info("Dataset: %d pairs, %.1f%% aligned", len(df), 100 * df["label"].mean())

    # 2-3. Train + compare models ------------------------------------------------
    logger.info("[2-3/7] Training CurriculumGPT + model comparison…")
    from ml.evaluation.compare_models import run as run_comparison
    comparison = run_comparison(args.epochs, args.smoke, args.seed)
    run_dir = Path(comparison["manifest_dir"])
    logger.info("Run folder: %s", run_dir)

    # Find the best checkpoint produced during comparison training.
    ckpts = sorted((run_dir / "checkpoints").glob("cgpt_epoch*.pt"))
    checkpoint = str(ckpts[-1]) if ckpts else None

    from ml.evaluation.run_manifest import RunManifest  # noqa: E402
    # Reattach to the same run folder for the remaining metrics.
    manifest = RunManifest(run_id=run_dir.name)

    # 4. Ablation ----------------------------------------------------------------
    logger.info("[4/7] Component ablation…")
    try:
        from ml.evaluation.ablation import run_ablation
        abl = run_ablation(checkpoint, args.smoke, args.seed)
        manifest.add_metrics("ablation", abl)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ablation failed: %s", exc)

    # 5. Latency -----------------------------------------------------------------
    logger.info("[5/7] Latency benchmark…")
    try:
        from ml.evaluation.benchmark_latency import run_benchmark
        lat = run_benchmark(n_warmup=3, n_measure=20 if not args.smoke else 5)
        manifest.add_metrics("latency", lat)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Latency benchmark failed: %s", exc)

    # 6. Statistics --------------------------------------------------------------
    logger.info("[6/7] Statistical validation…")
    try:
        _compute_statistics(manifest, comparison)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Statistics failed: %s", exc)

    # 7. Visual report -----------------------------------------------------------
    logger.info("[7/7] Generating visual report…")
    try:
        from ml.evaluation.report import build_report
        report_path = build_report(run_dir)
        manifest.add_artifact("report", str(report_path))
        logger.info("Report: %s", report_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Report generation failed: %s", exc)

    # Update PROJECT.md results section
    try:
        _update_project_md(run_dir, comparison)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not update PROJECT.md: %s", exc)

    logger.info("=== PIPELINE COMPLETE === all artifacts in %s", run_dir)
    print(f"\nOpen the report:  {run_dir / 'report.html'}\n")


def _update_project_md(run_dir: Path, comparison: dict) -> None:
    """Append/replace the auto-generated results table in PROJECT.md."""
    proj = Path(__file__).parent.parent / "PROJECT.md"
    if not proj.exists():
        return
    results = comparison.get("results", {})
    if not results:
        return
    lines = ["| Method | Accuracy | Precision | Recall | F1 | AUC |",
             "|--------|----------|-----------|--------|----|----|"]
    for name, m in sorted(results.items(), key=lambda kv: -(kv[1].get("accuracy") or 0)):
        lines.append(f"| {name} | {m.get('accuracy',0):.3f} | {m.get('precision',0):.3f} "
                     f"| {m.get('recall',0):.3f} | {m.get('f1',0):.3f} | {(m.get('auc') or 0):.3f} |")
    table = "\n".join(lines)
    marker = "## 5. Results"
    text = proj.read_text(encoding="utf-8")
    head = text.split(marker)[0]
    new = (f"{head}{marker}\n\n"
           f"_Latest full run: `{run_dir.name}` — see "
           f"`{run_dir.as_posix()}/report.html` for charts._\n\n"
           f"{table}\n\n"
           f"Test pairs: {comparison.get('test_size','?')}. "
           f"Full metrics + ablation in the run folder.\n")
    proj.write_text(new, encoding="utf-8")


def _compute_statistics(manifest, comparison) -> None:
    """Cohen's d of CurriculumGPT vs the strongest baseline, on test scores."""
    from ml.evaluation import statistics as st
    results = comparison.get("results", {})
    accs = {k: v.get("accuracy", 0) for k, v in results.items()}
    if "CurriculumGPT (7-component)" not in accs:
        return
    # Bootstrap a per-fold-like comparison from accuracy deltas (indicative).
    cgpt = accs["CurriculumGPT (7-component)"]
    baselines = {k: a for k, a in accs.items() if "CurriculumGPT" not in k}
    summary = {
        "curriculum_gpt_accuracy": cgpt,
        "baselines": baselines,
        "best_baseline": max(baselines.items(), key=lambda kv: kv[1])[0] if baselines else None,
        "improvement_over_best_baseline_pp": round(
            (cgpt - max(baselines.values())) * 100, 2) if baselines else None,
        "note": ("Accuracy deltas on a single held-out split. For publication-grade "
                 "Cohen's d / ICC, run nested CV (ml.evaluation.cross_validation)."),
    }
    manifest.add_metrics("statistics", summary)


if __name__ == "__main__":
    main()
