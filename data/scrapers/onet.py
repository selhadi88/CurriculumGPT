"""
O*NET Web Services collector — occupations + skills + technology skills for
every occupation relevant to the skill taxonomy's domains.

Auth: current (v2.0) O*NET Web Services accounts use a single API key sent
as an `X-API-Key` header — register free at https://services.onetcenter.org/.

IMPORTANT — v1.9 vs v2.0 are different hosts, not just different auth on the
same host. Confirmed directly from O*NET's own migration docs
(services.onetcenter.org/reference/start/migration) after v1.9's endpoint
returned 401 Basic Auth challenges for an X-API-Key-style key:
    v1.9 (legacy, username/password Basic Auth): https://services.onetcenter.org/ws/{path}
    v2.0 (current, X-API-Key header):            https://api-v2.onetcenter.org/{path}

Endpoints used (verified against live responses, not assumed from v1.9 docs):
    GET https://api-v2.onetcenter.org/online/search?keyword={term}&start=1&end=20
        -> {"total": N, "occupation": [{"code": "15-1252.00", "title": "Software Developers", ...}, ...]}
    GET https://api-v2.onetcenter.org/online/occupations/{code}/details/skills?end=50
        -> {"total": N, "element": [{"name": "Programming", ...}, ...]}
    GET https://api-v2.onetcenter.org/online/occupations/{code}/details/technology_skills?end=100
        -> {"total": N, "category": [{"title": "...", "example": [{"title": "Apache Kafka", ...}, ...]}, ...]}
        Note: each example's display name is under "title", not "name" — an
        assumption from v1.9-era knowledge that turned out wrong when checked
        against a real v2.0 response; fixed here after live verification.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

from data.scrapers.base import RateLimiter, ScrapedRecord, get_json_with_retry

logger = logging.getLogger(__name__)

_BASE_URL = "https://api-v2.onetcenter.org"

# Search terms per taxonomy domain (data/mappings/skill_taxonomy.json has 7
# domains, not 8 — verified against the shipped file before writing this).
# Each term is run through the O*NET keyword search; results are deduped by
# O*NET-SOC code across all domains before fetching skills/tech-skills.
DOMAIN_SEARCH_TERMS: dict[str, list[str]] = {
    "computer_science": ["software developer", "computer programmer", "computer systems analyst"],
    "data_science": ["data scientist", "statistician", "data analyst", "machine learning"],
    "cloud_devops": ["network and computer systems administrator", "computer network architect",
                     "database administrator"],
    "web_development": ["web developer", "web and digital interface designer"],
    "cybersecurity": ["information security analyst", "penetration tester"],
    "business": ["business analyst", "management analyst", "financial analyst", "marketing manager"],
    "design": ["graphic designer", "user experience designer"],
}


class ONetScraper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_second: float = 4.0,
        results_per_search: int = 20,
    ) -> None:
        self.api_key = api_key or os.getenv("ONET_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "ONET_API_KEY is not set. Register free at "
                "https://services.onetcenter.org/ and set ONET_API_KEY."
            )
        self.results_per_search = results_per_search
        self.rate_limiter = RateLimiter(requests_per_second)
        self._client: Optional[httpx.AsyncClient] = None
        self._headers = {
            "X-API-Key": self.api_key,
            "Accept": "application/json",
        }

    async def __aenter__(self) -> "ONetScraper":
        self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _search_occupations(self, keyword: str) -> list[dict]:
        url = f"{_BASE_URL}/online/search"
        data = await get_json_with_retry(
            self._client, url,
            rate_limiter=self.rate_limiter,
            params={"keyword": keyword, "start": 1, "end": self.results_per_search},
            headers=self._headers,
        )
        return data.get("occupation", [])

    async def _get_skills(self, code: str) -> list[str]:
        url = f"{_BASE_URL}/online/occupations/{code}/details/skills"
        try:
            data = await get_json_with_retry(
                self._client, url, rate_limiter=self.rate_limiter, headers=self._headers,
                # end=100 in one call rather than paginating — occupations have
                # well under 100 skill elements (35 on the occupation checked
                # live), so a single generous page covers all of them.
                params={"start": 1, "end": 100},
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("Skills fetch failed for %s: %s", code, exc)
            return []
        return [el.get("name", "") for el in data.get("element", []) if el.get("name")]

    async def _get_technology_skills(self, code: str) -> list[str]:
        url = f"{_BASE_URL}/online/occupations/{code}/details/technology_skills"
        try:
            data = await get_json_with_retry(
                self._client, url, rate_limiter=self.rate_limiter, headers=self._headers,
                params={"start": 1, "end": 100},
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("Technology skills fetch failed for %s: %s", code, exc)
            return []
        techs: list[str] = []
        for category in data.get("category", []):
            for example in category.get("example", []):
                # Live-verified field name is "title", not "name" (my
                # original assumption, carried over from v1.9-era knowledge,
                # was wrong — checked against a real v2.0 response).
                title = example.get("title")
                if title:
                    techs.append(title)
        return techs

    async def scrape(self, limit: int = 500) -> list[ScrapedRecord]:
        """
        limit caps the total number of occupations fetched in full detail
        (across all domains combined) — the search step itself is cheap and
        always covers every configured term.
        """
        assert self._client is not None, "use `async with ONetScraper() as scraper:`"

        occupations_by_code: dict[str, dict] = {}
        domain_by_code: dict[str, str] = {}

        for domain, terms in DOMAIN_SEARCH_TERMS.items():
            for term in terms:
                try:
                    results = await self._search_occupations(term)
                except httpx.HTTPStatusError as exc:
                    logger.error("O*NET search failed for %r: %s", term, exc)
                    continue
                for occ in results:
                    code = occ.get("code")
                    if not code:
                        continue
                    occupations_by_code.setdefault(code, occ)
                    domain_by_code.setdefault(code, domain)
                logger.info("O*NET search %-45s -> %d occupations (running total: %d unique)",
                           repr(term), len(results), len(occupations_by_code))

        codes = list(occupations_by_code.keys())[:limit]
        logger.info("Fetching skills + technology skills for %d unique occupations…", len(codes))

        records: list[ScrapedRecord] = []
        for i, code in enumerate(codes, start=1):
            occ = occupations_by_code[code]
            title = occ.get("title", "")
            skills = await self._get_skills(code)
            tech_skills = await self._get_technology_skills(code)
            description = (
                f"{title}. Core skills: {', '.join(skills[:12])}. "
                f"Technology skills: {', '.join(tech_skills[:12])}."
            ).strip()
            records.append(ScrapedRecord(
                title=title,
                description=description,
                source="onet",
                metadata={
                    "onet_code": code,
                    "domain": domain_by_code.get(code, ""),
                    "skills": skills,
                    "technology_skills": tech_skills,
                },
            ))
            if i % 10 == 0 or i == len(codes):
                logger.info("  …%d/%d occupations detailed", i, len(codes))

        self.domain_counts: dict[str, int] = {}
        for code in codes:
            d = domain_by_code.get(code, "")
            self.domain_counts[d] = self.domain_counts.get(d, 0) + 1

        return records
