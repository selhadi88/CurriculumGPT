"""
CLI for preprocessing raw scraped data.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --force   # re-run even if output exists
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing.pipeline import run_full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("preprocess")


def main() -> None:
    parser = argparse.ArgumentParser(description="GurriculumGPT preprocessing pipeline")
    parser.add_argument("--force", action="store_true", help="Re-run even if output already exists")
    args = parser.parse_args()

    logger.info("Starting preprocessing pipeline (force=%s)", args.force)
    counts = run_full_pipeline(force=args.force)

    for record_type, count in counts.items():
        logger.info("  %s: %d records processed", record_type, count)

    total = sum(counts.values())
    logger.info("Pipeline complete — %d total records ready in data/processed/", total)


if __name__ == "__main__":
    main()
