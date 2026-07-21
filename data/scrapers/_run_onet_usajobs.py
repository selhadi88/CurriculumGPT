"""
One-off driver: run ONetScraper + USAJobsScraper for real against all 7
taxonomy domains, save raw JSONL, print per-source and per-domain counts.

Reads credentials from the environment (ONET_API_KEY, USAJOBS_API_KEY,
USAJOBS_EMAIL) — never hardcoded here. Not part of the permanent scripts/
CLI surface; scripts/scrape.py --source onet / --source jobs are the
supported entry points long-term. This exists just to drive one real run
and produce the count report for this conversation.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")
logger = logging.getLogger("run_onet_usajobs")

from data.scrapers.jobs import USAJobsScraper  # noqa: E402
from data.scrapers.onet import ONetScraper  # noqa: E402

RAW_DIR = Path(__file__).resolve().parents[1] / "raw"


def save_jsonl(records: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"title": rec.title, "description": rec.description,
                                 "source": rec.source, **rec.metadata}) + "\n")
    logger.info("Saved %d records -> %s", len(records), out_path)


async def main() -> None:
    report: dict[str, dict] = {}

    logger.info("=== O*NET Web Services ===")
    async with ONetScraper() as scraper:
        onet_records = await scraper.scrape(limit=500)
    save_jsonl(onet_records, RAW_DIR / "jobs" / "onet.jsonl")
    report["onet"] = {"total": len(onet_records), "by_domain": dict(scraper.domain_counts)}

    logger.info("=== USAJobs ===")
    async with USAJobsScraper() as scraper:
        usajobs_records = await scraper.scrape(limit=4000)
    save_jsonl(usajobs_records, RAW_DIR / "jobs" / "usajobs.jsonl")
    report["usajobs"] = {"total": len(usajobs_records), "by_domain": dict(scraper.domain_counts)}

    print("\n" + "=" * 66)
    print(f"{'Source':<12}{'Domain':<20}{'Count':>10}")
    print("-" * 66)
    grand_total = 0
    for source, info in report.items():
        for domain, count in info["by_domain"].items():
            print(f"{source:<12}{domain:<20}{count:>10}")
        print(f"{source:<12}{'TOTAL':<20}{info['total']:>10}")
        print("-" * 66)
        grand_total += info["total"]
    print(f"{'COMBINED':<32}{grand_total:>10}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
