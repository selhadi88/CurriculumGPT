"""
Smoke tests for data/scrapers/*. Split into two groups:

  - Deterministic, no real network: rate limiter timing, retry/backoff
    against a mocked transport, missing-credential errors. Always run these.
  - Live: RemoteOK (no auth) is hit for real. O*NET/USAJobs live calls are
    opt-in via env vars, since they need real registered credentials.

Usage:
    python data/scrapers/test_scrapers_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from data.scrapers.base import RateLimiter, get_json_with_retry  # noqa: E402
from data.scrapers.jobs import RemoteOKScraper, USAJobsScraper  # noqa: E402
from data.scrapers.onet import ONetScraper  # noqa: E402

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


async def test_missing_credentials() -> None:
    print("\n1. missing-credential guards (no real network)")
    for key in ("ONET_API_KEY",):
        os.environ.pop(key, None)
    try:
        ONetScraper()
        check("ONetScraper() raises without ONET_API_KEY", False)
    except ValueError:
        check("ONetScraper() raises without ONET_API_KEY", True)

    for key in ("USAJOBS_API_KEY", "USAJOBS_EMAIL"):
        os.environ.pop(key, None)
    try:
        USAJobsScraper()
        check("USAJobsScraper() raises without key+email", False)
    except ValueError:
        check("USAJobsScraper() raises without key+email", True)


async def test_rate_limiter() -> None:
    print("\n2. rate limiter timing (no real network)")
    limiter = RateLimiter(requests_per_second=5.0)  # 200ms min interval
    t0 = time.monotonic()
    for _ in range(4):
        await limiter.wait()
    elapsed = time.monotonic() - t0
    # 4 calls at 5/s should take >= 3 intervals = ~0.6s (first call is free)
    check("4 calls at 5 req/s take >= 0.55s", elapsed >= 0.55, f"took {elapsed:.2f}s")
    check("4 calls at 5 req/s don't take absurdly long", elapsed < 2.0, f"took {elapsed:.2f}s")


async def test_retry_on_429_then_success() -> None:
    print("\n3. retry/backoff against a mocked 429-then-200 transport")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate limited"})
        return httpx.Response(200, json={"ok": True, "attempt": calls["n"]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        limiter = RateLimiter(requests_per_second=50.0)
        result = await get_json_with_retry(
            client, "https://example.invalid/test", rate_limiter=limiter, max_retries=5, backoff_base=0.01,
        )
    check("succeeds after 2 retries on 429", result == {"ok": True, "attempt": 3}, str(result))
    check("made exactly 3 requests (2 failed + 1 success)", calls["n"] == 3, str(calls["n"]))


async def test_retry_exhausted_raises() -> None:
    print("\n4. retry exhaustion raises rather than hanging or returning junk")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        limiter = RateLimiter(requests_per_second=50.0)
        try:
            await get_json_with_retry(
                client, "https://example.invalid/test", rate_limiter=limiter, max_retries=3, backoff_base=0.01,
            )
            check("raises after exhausting retries against persistent 503", False)
        except Exception as exc:  # noqa: BLE001
            check("raises after exhausting retries against persistent 503", True, str(exc))


async def test_non_retryable_4xx_fails_fast() -> None:
    print("\n5. non-retryable 4xx (e.g. bad API key) fails immediately, not after 5 retries")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": "invalid key"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        limiter = RateLimiter(requests_per_second=50.0)
        try:
            await get_json_with_retry(
                client, "https://example.invalid/test", rate_limiter=limiter, max_retries=5, backoff_base=0.01,
            )
            check("401 raises", False)
        except httpx.HTTPStatusError:
            check("401 raises", True)
    check("401 did not retry 5 times (fail-fast)", calls["n"] == 1, f"made {calls['n']} calls")


async def test_remoteok_live() -> None:
    print("\n6. RemoteOK — live, real API (no auth needed)")
    try:
        async with RemoteOKScraper() as scraper:
            records = await scraper.scrape(limit=2000)
        check("returns a nonzero number of postings", len(records) > 0, f"got {len(records)}")
        check("records have non-empty titles", all(r.title for r in records))
        check("records tagged with source='remoteok'", all(r.source == "remoteok" for r in records))
    except httpx.HTTPError as exc:
        check("RemoteOK reachable", False, str(exc))


async def test_onet_usajobs_live_if_credentials_present() -> None:
    print("\n7. O*NET / USAJobs — live, only if real credentials are set in the environment")
    if os.getenv("ONET_API_KEY"):
        async with ONetScraper() as scraper:
            records = await scraper.scrape(limit=5)
        check("O*NET returns occupation records with real key", len(records) > 0, f"got {len(records)}")
    else:
        print("  [SKIP] ONET_API_KEY not set — not testing live")

    if os.getenv("USAJOBS_API_KEY") and os.getenv("USAJOBS_EMAIL"):
        async with USAJobsScraper() as scraper:
            records = await scraper.scrape(limit=5)
        check("USAJobs returns job records with real key", len(records) > 0, f"got {len(records)}")
    else:
        print("  [SKIP] USAJOBS_API_KEY/USAJOBS_EMAIL not set — not testing live")


async def main() -> None:
    await test_missing_credentials()
    await test_rate_limiter()
    await test_retry_on_429_then_success()
    await test_retry_exhausted_raises()
    await test_non_retryable_4xx_fails_fast()
    await test_remoteok_live()
    await test_onet_usajobs_live_if_credentials_present()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
