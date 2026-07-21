"""
Frozen-backbone training pipeline for the BGE bi-encoder's projection heads.

Two strictly separate steps:
  1. Precompute — BAAI/bge-large-en-v1.5 runs once, in inference mode
     (no_grad, fp16 where supported), over every curriculum/job text in the
     labeled pairs. Embeddings are cached to disk. The backbone is dropped
     immediately after.
  2. Train — loads ONLY the cached embeddings and trains curriculum_head /
     job_head via InfoNCE. The backbone is never imported into this step.

Usage:
    python scripts/train_bge_frozen.py --epochs 10
    python scripts/train_bge_frozen.py --epochs 3 --smoke-test
"""
from __future__ import annotations

import argparse
import logging
import resource
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from ml.training.precompute import load_embedding_cache, precompute_and_cache  # noqa: E402
from ml.training.trainer import FrozenBackboneConfig, FrozenBackboneTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("train_bge_frozen")

_DEFAULT_PAIRS_PATH = Path(__file__).parent.parent / "data" / "labeled" / "pairs.parquet"

# aligned/partial/not_aligned -> continuous training weight. A weight of 0
# (not_aligned) excludes the row entirely — InfoNCE has nothing meaningful to
# pull together for a pair the LLM says isn't aligned. partial=0.5 still gets
# pulled together, just half as strongly as a full aligned=1.0 pair.
_SIM_TARGET_MAP = {"aligned": 1.0, "partial": 0.5, "not_aligned": 0.0}


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--precompute-batch-size", type=int, default=8)
    ap.add_argument("--no-fp16", action="store_true", help="Force fp32 in the precompute step")
    # `checkpoints/` is mounted read-only in docker-compose.yml (the app reads
    # exported checkpoints from there; it doesn't write them). Training output
    # goes under outputs/, which is read-write — same convention RunManifest
    # uses. Copy the resulting .pt file into checkpoints/ on the host if you
    # want `MODEL_CHECKPOINT` to pick it up from that directory specifically;
    # otherwise just point MODEL_CHECKPOINT at the outputs/ path directly.
    ap.add_argument("--output-dir", default="outputs/checkpoints/bge_frozen")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Recompute embeddings even if a cache file already exists")
    ap.add_argument("--smoke-test", action="store_true", help="1 epoch, tiny batch, fast sanity run")
    ap.add_argument("--pairs-path", default=str(_DEFAULT_PAIRS_PATH),
                    help="Labeled pairs parquet to train on (default: data/labeled/pairs.parquet)")
    ap.add_argument("--cache-path", default=None,
                    help="Embedding cache location (default: <pairs-path stem>_embedding_cache.pt "
                         "next to the pairs file — kept distinct per pairs file so a cache built "
                         "from one dataset's texts is never silently reused for another)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pairs_path = Path(args.pairs_path)
    cache_path = Path(args.cache_path) if args.cache_path else \
        pairs_path.parent / f"{pairs_path.stem}_embedding_cache.pt"

    if not pairs_path.exists():
        logger.error("No labeled pairs at %s. Run scripts/build_dataset.py first.", pairs_path)
        sys.exit(1)

    df = pd.read_parquet(pairs_path)
    if "llm_label" in df.columns and df["llm_label"].notna().all():
        df["train_weight"] = df["llm_label"].map(_SIM_TARGET_MAP)
        logger.info("Training signal: llm_label -> train_weight (aligned=1.0, partial=0.5, not_aligned=0.0)")
    else:
        df["train_weight"] = df["label"].astype(float)
        logger.info("No complete llm_label column found — falling back to heuristic `label` (0/1) as train_weight")
    positives = df[df["train_weight"] > 0]
    logger.info("Loaded %d pairs (%d with train_weight>0) from %s", len(df), len(positives), pairs_path)

    curriculum_texts = sorted(set(df["curriculum_text"].tolist()))
    job_texts = sorted(set(df["job_text"].tolist()))

    # ---------------------------------------------------------------- step 1
    if cache_path.exists() and not args.force_recompute:
        logger.info("[1/2] Using existing embedding cache: %s (pass --force-recompute to rebuild)",
                    cache_path)
        cache = load_embedding_cache(cache_path)
    else:
        logger.info("[1/2] Precompute step — loading backbone in inference mode only…")
        rss_before = _peak_rss_mib()
        report = precompute_and_cache(
            curriculum_texts, job_texts, cache_path,
            batch_size=args.precompute_batch_size,
            use_fp16=not args.no_fp16,
        )
        logger.info(
            "Precompute done: %d curriculum + %d job texts, dtype=%s%s, "
            "%.1fs, peak RSS %.0f MiB (was %.0f MiB before this step)",
            report.n_curriculum, report.n_job, report.dtype_used,
            f" (fp16 requested, fell back: {report.fp16_fallback_reason})"
            if report.fp16_fallback_reason else "",
            report.elapsed_seconds, report.peak_rss_mib, rss_before,
        )
        cache = load_embedding_cache(cache_path)

    # ---------------------------------------------------------------- step 2
    logger.info("[2/2] Training step — backbone NOT imported, heads only…")
    cfg = FrozenBackboneConfig(
        epochs=1 if args.smoke_test else args.epochs,
        batch_size=4 if args.smoke_test else args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        save_every_n_epochs=1 if args.smoke_test else 2,
        seed=args.seed,
    )
    trainer = FrozenBackboneTrainer(cfg)
    rss_before_train = _peak_rss_mib()
    result = trainer.train(cache, positives.to_dict("records"))
    rss_after_train = _peak_rss_mib()

    logger.info(
        "Training done: %d train / %d val pairs, final train_loss=%.4f, "
        "peak RSS %.0f MiB (was %.0f MiB before this step)",
        result["n_train_pairs"], result["n_val_pairs"],
        result["metrics"][-1]["train_loss"],
        rss_after_train, rss_before_train,
    )
    print(f"\nCheckpoints written to: {result['output_dir']}/")
    print("Set MODEL_CHECKPOINT to one of those .pt files to use the trained heads at inference.\n")


if __name__ == "__main__":
    main()
