"""
NOT IMPLEMENTED — out of scope for this task.

Coursera has no official public API for bulk course-catalog access. A real
implementation would need to scrape coursera.org's course pages directly,
which risks violating their Terms of Service — the same concern already
flagged earlier in this project for site-scraping generally. This task
implemented only the three sources with real, ToS-compliant APIs: O*NET Web
Services, USAJobs, and RemoteOK (see data/scrapers/onet.py, jobs.py).

Exists here only so `from data.scrapers.coursera import CourseraScraper`
(used by scripts/scrape.py) keeps importing cleanly; scripts/scrape.py
catches NotImplementedError from `--source all` and skips this source with a
warning rather than crashing the whole run.
"""
from __future__ import annotations

from data.scrapers.base import ScrapedRecord  # noqa: F401  (kept for symmetry/import parity)


class CourseraScraper:
    async def __aenter__(self) -> "CourseraScraper":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    async def scrape(self, limit: int = 500) -> list:
        raise NotImplementedError(
            "CourseraScraper is not implemented (no official bulk API; "
            "scraping coursera.org directly risks violating their Terms of "
            "Service — out of scope). Use --source onet or --source jobs instead."
        )
