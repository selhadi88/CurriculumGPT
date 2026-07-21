"""
CLI for data collection.

Usage:
    python scripts/scrape.py --source coursera --limit 2000
    python scripts/scrape.py --source onet --limit 500
    python scripts/scrape.py --source jobs --limit 3000
    python scripts/scrape.py --source all --limit 5000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.scrapers.coursera import CourseraScraper
from data.scrapers.jobs import MITOpenCourseWareScraper, RemoteOKScraper, USAJobsScraper
from data.scrapers.onet import ONetScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("scrape")

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Per-source counts for the final report, keyed by save filename.
_REPORT: dict[str, int] = {}


def save_jsonl(records: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.__dict__ if hasattr(rec, "__dict__") else rec) + "\n")
    logger.info("Saved %d records → %s", len(records), out_path)
    _REPORT[str(out_path.relative_to(RAW_DIR.parent.parent))] = len(records)


async def run_coursera(limit: int) -> None:
    try:
        async with CourseraScraper() as scraper:
            records = await scraper.scrape(limit=limit)
    except NotImplementedError as exc:
        logger.warning("Skipping coursera: %s", exc)
        _REPORT["data/raw/courses/coursera.jsonl (skipped)"] = 0
        return
    save_jsonl(
        [{"title": r.title, "description": r.description, "source": r.source, **r.metadata}
         for r in records],
        RAW_DIR / "courses" / "coursera.jsonl",
    )


async def run_mit_ocw(limit: int) -> None:
    try:
        async with MITOpenCourseWareScraper() as scraper:
            records = await scraper.scrape(limit=limit)
    except NotImplementedError as exc:
        logger.warning("Skipping mit_ocw: %s", exc)
        _REPORT["data/raw/courses/mit_ocw.jsonl (skipped)"] = 0
        return
    save_jsonl(
        [{"title": r.title, "description": r.description, "source": r.source, **r.metadata}
         for r in records],
        RAW_DIR / "courses" / "mit_ocw.jsonl",
    )


async def run_onet(limit: int) -> None:
    async with ONetScraper() as scraper:
        records = await scraper.scrape(limit=limit)
    save_jsonl(
        [{"title": r.title, "description": r.description, "source": r.source, **r.metadata}
         for r in records],
        RAW_DIR / "jobs" / "onet.jsonl",
    )


async def run_jobs(limit: int) -> None:
    all_records = []

    # RemoteOK (no auth)
    async with RemoteOKScraper() as scraper:
        records = await scraper.scrape(limit=min(limit // 2, 2000))
        logger.info("RemoteOK: %d records", len(records))
        all_records.extend(records)

    # USAJobs (requires API key)
    async with USAJobsScraper() as scraper:
        records = await scraper.scrape(limit=min(limit // 2, 2000))
        logger.info("USAJobs: %d records", len(records))
        all_records.extend(records)

    save_jsonl(
        [{"title": r.title, "description": r.description, "source": r.source, **r.metadata}
         for r in all_records],
        RAW_DIR / "jobs" / "jobs.jsonl",
    )


def print_report() -> None:
    print("\n" + "=" * 60)
    print(f"{'Source file':<45}{'Records':>15}")
    print("-" * 60)
    total = 0
    for name, count in _REPORT.items():
        print(f"{name:<45}{count:>15}")
        total += count
    print("-" * 60)
    print(f"{'TOTAL':<45}{total:>15}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="GurriculumGPT data scraper")
    parser.add_argument(
        "--source",
        choices=["coursera", "onet", "jobs", "mit_ocw", "all"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=1000, help="Max records per source")
    args = parser.parse_args()

    if args.source == "coursera":
        asyncio.run(run_coursera(args.limit))
    elif args.source == "onet":
        asyncio.run(run_onet(args.limit))
    elif args.source == "jobs":
        asyncio.run(run_jobs(args.limit))
    elif args.source == "mit_ocw":
        asyncio.run(run_mit_ocw(args.limit))
    elif args.source == "all":
        asyncio.run(run_coursera(args.limit))
        asyncio.run(run_mit_ocw(args.limit // 4))
        asyncio.run(run_onet(args.limit // 2))
        asyncio.run(run_jobs(args.limit))

    logger.info("Scraping complete.")
    print_report()


if __name__ == "__main__":
    main()
