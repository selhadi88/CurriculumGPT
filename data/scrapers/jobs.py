"""
Job-posting collectors: USAJobs (official API, needs a key) and RemoteOK
(public API, no auth). Also exposes MITOpenCourseWareScraper as an
importable-but-not-implemented stub — scripts/scrape.py imports all three
names from this module, and MIT OCW has no official bulk API (would need
site scraping, which is out of scope for this task; see the class docstring).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from data.scrapers.base import RateLimiter, ScrapedRecord, get_json_with_retry

logger = logging.getLogger(__name__)

# Search terms per taxonomy domain — mirrors data/scrapers/onet.py's
# DOMAIN_SEARCH_TERMS so both collectors can report per-domain counts on the
# same 7 domains (data/mappings/skill_taxonomy.json).
DOMAIN_SEARCH_TERMS: dict[str, list[str]] = {
    "computer_science": ["software engineer", "computer programmer", "systems analyst"],
    "data_science": ["data scientist", "data analyst", "machine learning engineer"],
    "cloud_devops": ["cloud engineer", "systems administrator", "network engineer"],
    "web_development": ["web developer", "front end developer"],
    "cybersecurity": ["information security", "cybersecurity analyst"],
    "business": ["business analyst", "management analyst", "financial analyst"],
    "design": ["graphic designer", "user experience designer"],
}


# --------------------------------------------------------------------------- USAJobs

class USAJobsScraper:
    """
    USAJobs Search API — official US federal government jobs API.
    Docs: https://developer.usajobs.gov/api-reference/get-api-search
    Auth: free registration at https://developer.usajobs.gov/ gives an
    Authorization-Key; USAJOBS_EMAIL must match the email you registered
    with (USAJobs requires it as the User-Agent on every request).
    """

    _BASE_URL = "https://data.usajobs.gov/api/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        email: Optional[str] = None,
        requests_per_second: float = 2.0,
        results_per_page: int = 500,
    ) -> None:
        self.api_key = api_key or os.getenv("USAJOBS_API_KEY", "")
        self.email = email or os.getenv("USAJOBS_EMAIL", "")
        if not self.api_key or not self.email:
            raise ValueError(
                "USAJOBS_API_KEY and USAJOBS_EMAIL are required. Register free at "
                "https://developer.usajobs.gov/ and set both."
            )
        self.results_per_page = min(results_per_page, 500)  # USAJobs hard cap
        self.rate_limiter = RateLimiter(requests_per_second)
        self._client: Optional[httpx.AsyncClient] = None
        self._headers = {
            "Host": "data.usajobs.gov",
            "User-Agent": self.email,
            "Authorization-Key": self.api_key,
        }

    async def __aenter__(self) -> "USAJobsScraper":
        self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def scrape(
        self,
        limit: int = 1000,
        domain_keywords: Optional[dict[str, list[str]]] = None,
    ) -> list[ScrapedRecord]:
        """
        domain_keywords: {domain_name: [search terms]}, defaults to
        DOMAIN_SEARCH_TERMS above (the 7 taxonomy domains). limit caps the
        total across all domains combined; each domain gets a roughly equal
        share so no single domain can starve the others.
        """
        assert self._client is not None, "use `async with USAJobsScraper() as scraper:`"

        domain_keywords = domain_keywords or DOMAIN_SEARCH_TERMS
        per_domain_limit = max(1, limit // max(1, len(domain_keywords)))

        records: list[ScrapedRecord] = []
        seen_ids: set[str] = set()
        self.domain_counts: dict[str, int] = {}

        for domain, keywords in domain_keywords.items():
            domain_start_count = len(records)
            for keyword in keywords:
                if len(records) - domain_start_count >= per_domain_limit:
                    break
                page = 1
                while len(records) - domain_start_count < per_domain_limit:
                    params = {
                        "Keyword": keyword,
                        "ResultsPerPage": self.results_per_page,
                        "Page": page,
                    }
                    data = await get_json_with_retry(
                        self._client, self._BASE_URL,
                        rate_limiter=self.rate_limiter, params=params, headers=self._headers,
                    )
                    items = data.get("SearchResult", {}).get("SearchResultItems", [])
                    if not items:
                        break

                    for item in items:
                        desc = item.get("MatchedObjectDescriptor", {})
                        job_id = desc.get("PositionID") or item.get("MatchedObjectId")
                        if not job_id or job_id in seen_ids:
                            continue
                        seen_ids.add(job_id)

                        title = desc.get("PositionTitle", "")
                        summary = desc.get("UserArea", {}).get("Details", {}).get("JobSummary", "")
                        if not title:
                            continue

                        records.append(ScrapedRecord(
                            title=title,
                            description=summary or title,
                            source="usajobs",
                            metadata={
                                "job_id": job_id,
                                "company": desc.get("OrganizationName", ""),
                                "location": desc.get("PositionLocationDisplay", ""),
                                "url": desc.get("PositionURI", ""),
                                "posted_date": desc.get("PublicationStartDate", ""),
                                "keyword_query": keyword,
                                "domain": domain,
                            },
                        ))
                        if len(records) - domain_start_count >= per_domain_limit:
                            break

                    total_available = int(data.get("SearchResult", {}).get("SearchResultCountAll", 0))
                    logger.info(
                        "USAJobs [%s] %-30s page %d: +%d (domain total %d/%d, %d available for this keyword)",
                        domain, repr(keyword), page, len(items),
                        len(records) - domain_start_count, per_domain_limit, total_available,
                    )
                    page += 1
                    if page * self.results_per_page > total_available:
                        break

            self.domain_counts[domain] = len(records) - domain_start_count

        return records[:limit]


# --------------------------------------------------------------------------- RemoteOK

class RemoteOKScraper:
    """
    RemoteOK public API — no authentication, but RemoteOK asks scrapers to
    identify themselves with a descriptive User-Agent (their site/API docs
    request this explicitly) rather than a generic client string.
    Docs: https://remoteok.com/api

    IMPORTANT — verified live before building on it: `?tags=X` genuinely
    filters (91/100 results tagged "design" vs. 16/100 in the unfiltered
    feed), but `page=`/`offset=` params are silently ignored — every page
    returns the identical ~100 results. RemoteOK's public API does not
    support pagination past that fixed page; scrape_by_tags() still loops
    defensively (stops as soon as a "page" brings back no new IDs) so it
    picks up real pagination automatically if RemoteOK ever adds it, but
    today each tag effectively contributes at most ~100 postings.
    """

    _URL = "https://remoteok.com/api"

    def __init__(self, requests_per_second: float = 1.0) -> None:
        self.rate_limiter = RateLimiter(requests_per_second)
        self._client: Optional[httpx.AsyncClient] = None
        self._headers = {
            "User-Agent": "CurriculumGPT-research-scraper/1.0 "
                          "(curriculum-industry alignment research; contact via repo)",
        }

    async def __aenter__(self) -> "RemoteOKScraper":
        self._client = httpx.AsyncClient(follow_redirects=True)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()

    @staticmethod
    def _to_record(item: dict) -> Optional[ScrapedRecord]:
        title = item.get("position") or item.get("title") or ""
        if not title:
            return None
        description = item.get("description", "") or title
        return ScrapedRecord(
            title=title,
            description=description,
            source="remoteok",
            metadata={
                "job_id": str(item.get("id", "")),
                "company": item.get("company", ""),
                "location": item.get("location", "") or "Remote",
                "url": item.get("url", ""),
                "tags": item.get("tags", []),
                "posted_date": item.get("date", ""),
            },
        )

    async def _fetch(self, params: Optional[dict] = None) -> list[dict]:
        data = await get_json_with_retry(
            self._client, self._URL, rate_limiter=self.rate_limiter,
            params=params, headers=self._headers,
        )
        if not isinstance(data, list):
            logger.warning("Unexpected RemoteOK response shape: %s", type(data))
            return []
        # First element is a legal/metadata notice, not a job posting.
        return [item for item in data if isinstance(item, dict) and item.get("id")]

    async def scrape(self, limit: int = 1000) -> list[ScrapedRecord]:
        assert self._client is not None, "use `async with RemoteOKScraper() as scraper:`"
        postings = await self._fetch()
        logger.info("RemoteOK: %d postings returned (before limit=%d)", len(postings), limit)
        records = [self._to_record(item) for item in postings[:limit]]
        return [r for r in records if r is not None]

    async def scrape_by_tags(
        self,
        domain_tags: dict[str, list[str]],
        existing_ids: Optional[set[str]] = None,
        max_pages_per_tag: int = 5,
    ) -> tuple[list[ScrapedRecord], dict[str, int]]:
        """
        Fetch RemoteOK's per-tag endpoint for every tag in every domain,
        deduping against `existing_ids` (postings already collected
        elsewhere) and across tags/domains within this call — the same
        posting commonly carries multiple tags (e.g. both "design" and
        "figma"), so without this a job would otherwise be double-counted.

        Returns (new_records, domain_counts) where domain_counts is how many
        *new, previously-unseen* postings each domain's tags contributed —
        i.e. additions, not raw per-tag totals.
        """
        assert self._client is not None, "use `async with RemoteOKScraper() as scraper:`"

        seen_ids: set[str] = set(existing_ids or set())
        new_records: list[ScrapedRecord] = []
        domain_counts: dict[str, int] = {}

        for domain, tags in domain_tags.items():
            domain_new = 0
            for tag in tags:
                page_seen_this_tag: set[str] = set()
                for page in range(1, max_pages_per_tag + 1):
                    postings = await self._fetch({"tags": tag, "page": page})
                    ids_this_page = {str(p.get("id")) for p in postings}
                    new_this_page = ids_this_page - page_seen_this_tag
                    if not new_this_page:
                        # No pagination support (confirmed live: page N ==
                        # page 1) or genuinely exhausted — stop either way.
                        break
                    page_seen_this_tag |= ids_this_page

                    for item in postings:
                        job_id = str(item.get("id", ""))
                        if not job_id or job_id in seen_ids:
                            continue
                        seen_ids.add(job_id)
                        record = self._to_record(item)
                        if record is None:
                            continue
                        record.metadata["domain"] = domain
                        record.metadata["tag_query"] = tag
                        new_records.append(record)
                        domain_new += 1

                logger.info("RemoteOK tag=%-10s page(s) fetched: %d, running domain total: %d",
                           repr(tag), page, domain_new)

            domain_counts[domain] = domain_new

        return new_records, domain_counts


# --------------------------------------------------------------------------- MIT OCW

class MITOpenCourseWareScraper:
    """
    MIT OpenCourseWare course metadata via MIT's own public "MIT Open"
    aggregation API — genuinely public and structured (not HTML scraping).

    ocw.mit.edu itself is a static Hugo site with no REST API (verified live:
    every guessed /api/, /*.json path 404s; only per-course Schema.org
    JSON-LD is embedded, and it lacks department/topics). MIT's separate
    open.mit.edu service — part of MIT Open Learning's public infrastructure
    — exposes exactly that structured metadata for every OCW course:
        GET https://open.mit.edu/api/v0/courses/?offered_by=OCW&limit=N&offset=M
    Verified live: 2,581 OCW-offered courses total, each with title,
    short_description/full_description, department + department_slug, and a
    topics list — precisely the fields requested (title, description,
    department, topics).
    """

    _URL = "https://open.mit.edu/api/v0/courses/"

    def __init__(self, requests_per_second: float = 3.0) -> None:
        self.rate_limiter = RateLimiter(requests_per_second)
        self._client: Optional[httpx.AsyncClient] = None
        self._headers = {
            "User-Agent": "CurriculumGPT-research-scraper/1.0 "
                          "(curriculum-industry alignment research; contact via repo)",
            "Accept": "application/json",
        }

    async def __aenter__(self) -> "MITOpenCourseWareScraper":
        self._client = httpx.AsyncClient(follow_redirects=True)
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()

    @staticmethod
    def _to_record(item: dict) -> Optional[ScrapedRecord]:
        title = item.get("title", "")
        if not title:
            return None
        description = item.get("full_description") or item.get("short_description") or title
        department_slug = item.get("department_slug", "") or ""
        department = department_slug.replace("-", " ").title() if department_slug else ""
        topics = [t.get("name", "") for t in item.get("topics", []) if t.get("name")]
        return ScrapedRecord(
            title=title,
            description=description,
            source="mit_ocw",
            metadata={
                "course_id": item.get("course_id", ""),
                "department": department,
                "department_codes": item.get("department", []),
                "topics": topics,
                "url": item.get("url", ""),
                "level": (item.get("runs") or [{}])[0].get("level", "") if item.get("runs") else "",
            },
        )

    async def scrape(self, limit: int = 500) -> list[ScrapedRecord]:
        assert self._client is not None, "use `async with MITOpenCourseWareScraper() as scraper:`"

        records: list[ScrapedRecord] = []
        offset = 0
        page_size = min(limit, 100)
        while len(records) < limit:
            data = await get_json_with_retry(
                self._client, self._URL, rate_limiter=self.rate_limiter,
                params={"offered_by": "OCW", "limit": page_size, "offset": offset},
                headers=self._headers,
            )
            results = data.get("results", [])
            if not results:
                break
            for item in results:
                record = self._to_record(item)
                if record is not None:
                    records.append(record)
                if len(records) >= limit:
                    break
            logger.info("MIT OCW: offset %d -> +%d (total %d/%d, %d available)",
                       offset, len(results), len(records), limit, data.get("count", 0))
            if data.get("next") is None:
                break
            offset += page_size

        return records[:limit]
