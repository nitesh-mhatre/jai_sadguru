"""
data/nse_session.py
--------------------
Singleton requests.Session that handles NSE India's cookie requirement.

NSE blocks direct API calls without a valid browser session cookie.
The pattern that works:
  1. GET https://www.nseindia.com           → sets nsit, nseappid cookies
  2. GET https://www.nseindia.com/option-chain  → refreshes session
  3. Now API calls succeed with those cookies

This module keeps ONE persistent session alive and auto-renews cookies
every 25 minutes (NSE sessions expire after ~30 min of inactivity).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import requests

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent":      (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer":         "https://www.nseindia.com/option-chain",
    "Origin":          "https://www.nseindia.com",
    "Connection":      "keep-alive",
    "DNT":             "1",
    "Sec-Fetch-Dest":  "empty",
    "Sec-Fetch-Mode":  "cors",
    "Sec-Fetch-Site":  "same-origin",
}

_WARMUP_URLS = [
    "https://www.nseindia.com",
    "https://www.nseindia.com/option-chain",
]

_COOKIE_TTL  = 25 * 60   # 25 minutes
_MAX_RETRIES = 3
_RETRY_DELAY = 2          # seconds


# ── Singleton session manager ─────────────────────────────────────────────────

class _NseSession:
    """Thread-safe NSE session with automatic cookie renewal."""

    def __init__(self):
        self._lock         = threading.Lock()
        self._session: requests.Session | None = None
        self._last_warmup: float = 0.0

    def _needs_warmup(self) -> bool:
        return time.time() - self._last_warmup > _COOKIE_TTL

    def _warmup(self) -> None:
        """Create a fresh session and hit NSE pages to get cookies."""
        log.debug("NSE session warmup starting…")
        sess = requests.Session()
        sess.headers.update(_HEADERS)

        for url in _WARMUP_URLS:
            try:
                r = sess.get(url, timeout=12, allow_redirects=True)
                log.debug("Warmup GET %s → %d", url, r.status_code)
                time.sleep(0.8)   # small delay between warmup requests
            except Exception as exc:
                log.warning("Warmup request failed for %s: %s", url, exc)

        self._session      = sess
        self._last_warmup  = time.time()
        log.debug("NSE session warmed up at %s", datetime.now().strftime("%H:%M:%S"))

    def get(self, url: str, **kwargs) -> requests.Response:
        """
        Perform a GET request with the managed NSE session.
        Auto-warms up cookies if needed and retries on 403/timeout.
        """
        kwargs.setdefault("timeout", 15)

        for attempt in range(_MAX_RETRIES):
            with self._lock:
                if self._session is None or self._needs_warmup():
                    self._warmup()
                sess = self._session

            try:
                resp = sess.get(url, **kwargs)

                if resp.status_code == 403:
                    log.warning(
                        "NSE 403 on %s (attempt %d) — forcing cookie refresh",
                        url, attempt + 1,
                    )
                    with self._lock:
                        self._last_warmup = 0   # force re-warmup
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_RETRY_DELAY * (attempt + 1))
                    continue

                resp.raise_for_status()
                return resp

            except requests.exceptions.Timeout:
                log.warning("Timeout on %s (attempt %d)", url, attempt + 1)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)

            except requests.exceptions.RequestException as exc:
                log.warning("Request error %s: %s", url, exc)
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_DELAY)

        raise RuntimeError(
            f"NSE API request failed after {_MAX_RETRIES} attempts: {url}"
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_session = _NseSession()


def nse_get(url: str, **kwargs) -> requests.Response:
    """Public API — use this everywhere instead of requests.get() for NSE URLs."""
    return _session.get(url, **kwargs)


def force_refresh() -> None:
    """Force a cookie refresh (call if you start getting 403s mid-session)."""
    with _session._lock:
        _session._last_warmup = 0
