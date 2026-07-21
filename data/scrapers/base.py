"""
Shared infrastructure for data/scrapers/*: a common record type, an async
rate limiter, and a retrying HTTP GET helper. Every real collector
(onet.py, jobs.py) builds on these instead of reimplementing backoff logic.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ScrapedRecord:
    """
    What every scraper yields. scripts/scrape.py accesses `.title`,
    `.description`, `.source`, and spreads `.metadata` into the saved JSONL
    record — keep this shape stable, it's the contract the rest of the
    pipeline (data/preprocessing/pipeline.py) depends on.
    """
    title: str
    description: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RateLimiter:
    """
    Simple fixed-interval async rate limiter (not a full token bucket —
    deliberately conservative and easy to reason about for a small research
    scraper, not a high-throughput production crawler).
    """

    def __init__(self, requests_per_second: float) -> None:
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            elapsed = time.monotonic() - self._last_call
            remaining = self.min_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_call = time.monotonic()


class RateLimitExceeded(RuntimeError):
    """Raised when retries are exhausted against a 429/5xx response."""


async def get_json_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    rate_limiter: RateLimiter,
    params: Optional[dict[str, Any]] = None,
    headers: Optional[dict[str, str]] = None,
    max_retries: int = 5,
    backoff_base: float = 1.5,
    timeout: float = 30.0,
) -> Any:
    """
    GET url, rate-limited, retrying on 429/5xx with exponential backoff +
    jitter (and honoring a numeric Retry-After header when the server sends
    one). Raises RateLimitExceeded / httpx.HTTPStatusError if retries are
    exhausted or a non-retryable status (4xx other than 429) comes back.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        await rate_limiter.wait()
        try:
            resp = await client.get(url, params=params, headers=headers, timeout=timeout)
        except httpx.TransportError as exc:
            last_exc = exc
            wait_s = backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning("Transport error on %s (attempt %d/%d): %s — retrying in %.1fs",
                           url, attempt + 1, max_retries, exc, wait_s)
            await asyncio.sleep(wait_s)
            continue

        if resp.status_code == 200:
            # Force UTF-8 decode explicitly rather than trusting httpx's
            # charset auto-detection — some sources (RemoteOK observed live)
            # don't declare charset correctly, and auto-detection silently
            # mojibake's non-ASCII text (e.g. Portuguese/accented job titles)
            # instead of raising, so this would go unnoticed without it.
            import json as _json
            return _json.loads(resp.content.decode("utf-8"))

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            wait_s = float(retry_after) if retry_after and retry_after.isdigit() else (
                backoff_base * (2 ** attempt) + random.uniform(0, 0.5)
            )
            logger.warning("%s %s (attempt %d/%d) — retrying in %.1fs",
                           resp.status_code, url, attempt + 1, max_retries, wait_s)
            await asyncio.sleep(wait_s)
            last_exc = RateLimitExceeded(f"{resp.status_code} from {url}")
            continue

        # Non-retryable client error (401/403/404/...) — fail fast with the
        # response body, since retrying won't fix a bad key or bad request.
        resp.raise_for_status()

    raise last_exc or RateLimitExceeded(f"Exhausted {max_retries} retries against {url}")
