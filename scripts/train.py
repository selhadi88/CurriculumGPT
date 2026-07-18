"""
CLI for training the alignment model.

Usage:
    python scripts/train.py --model bge --epochs 10 --batch-size 32
    python scripts/train.py --model custom --epochs 20 --lr 3e-4
    python scripts/train.py --model bge --checkpoint checkpoints/checkpoint_epoch05_loss0.1234.pt
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.pipeline import load_processed
from ml.training.trainer import Trainer, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("train")


def build_training_pairs(
    n_curriculum: int = 500,
    n_jobs: int = 500,
) -> list[dict]:
    """
    Build (curriculum, job) training pairs from processed data.
    In a real setup you'd have human-labeled or heuristically paired data.
    Here we use simple heuristic matching by category keyword overlap.
    """
    try:
        courses_df = load_processed("courses")
        jobs_df = load_processed("jobs")
    except FileNotFoundError as e:
        logger.error("%s", e)
        logger.error("Run `python scripts/scrape.py` and `python scripts/preprocess.py` first.")
        sys.exit(1)

    courses_df = courses_df.head(n_curriculum)
    jobs_df = jobs_df.head(n_jobs)

    pairs: list[dict] = []
    for _, course in courses_df.iterrows():
        curriculum_text = f"{course.get('title', '')}. {course.get('description', '')}"
        # Naive: pair each course with a random job (contrastive loss handles supervision)
        for _, job in jobs_df.sample(min(5, len(jobs_df))).iterrows():
            job_text = f"{job.get('title', '')}. {job.get('description', '')}"
            pairs.append({
                "curriculum": curriculum_text[:1000],
                "job": job_text[:1000],
                "skills": [],  # would be populated with real labels
            })

    logger.info("Built %d training pairs from processed data", len(pairs))
    return pairs


def _train_curriculum_gpt(args) -> None:
    """Train the seven-component CurriculumGPT model from the labeled parquet."""
    import pandas as pd
    from ml.training.cgpt_trainer import CGPTConfig, CGPTTrainer

    pairs_path = Path(__file__).parent.parent / "data" / "labeled" / "pairs.parquet"
    if not pairs_path.exists():
        logger.error("Labeled dataset not found at %s", pairs_path)
        logger.error("Run `python scripts/build_dataset.py` first.")
        sys.exit(1)

    df = pd.read_parquet(pairs_path)
    if args.smoke_test:
        df = df.sample(min(len(df), 40), random_state=42).reset_index(drop=True)
    rows = df.to_dict("records")
    logger.info("Loaded %d labeled pairs (%.1f%% aligned)", len(rows), 100 * df["label"].mean())

    cfg = CGPTConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device="auto",
        curriculum_learning=not args.smoke_test,
    )
    logger.info("CurriculumGPT config: %s", cfg)
    CGPTTrainer(cfg).train(rows)
    logger.info("CurriculumGPT training done. Checkpoints in %s/", args.output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GurriculumGPT alignment model")
    parser.add_argument("--model", choices=["bge", "custom", "curriculum_gpt"], default="bge")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--checkpoint", default=None, help="Resume from checkpoint")
    parser.add_argument("--n-curriculum", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=500)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Tiny fast CPU run to validate the pipeline")
    args = parser.parse_args()

    if args.model == "curriculum_gpt":
        _train_curriculum_gpt(args)
        return

    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
    )

    logger.info("Training config: %s", json.dumps(config.__dict__, indent=2))

    samples = build_training_pairs(args.n_curriculum, args.n_jobs)
    if not samples:
        logger.error("No training samples found.")
        sys.exit(1)

    trainer = Trainer(config)
    trainer.train(samples)

    logger.info("Training done. Checkpoints in %s/", args.output_dir)


if __name__ == "__main__":
    main()
