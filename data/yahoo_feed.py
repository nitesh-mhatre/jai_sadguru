"""
data/yahoo_feed.py
------------------
Yahoo Finance feed — used ONLY for:
  1. NIFTY 50 spot price  (^NSEI)
  2. India VIX            (^INDIAVIX)

Everything else (expiries, OI, option chain, charts) stays on NSE.

Install: pip install yfinance
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

_NIFTY_TICKER = "^NSEI"
_VIX_TICKER   = "^INDIAVIX"


def get_nifty_spot() -> float:
    """
    Fetch NIFTY 50 live spot price from Yahoo Finance.
    Falls back to last daily close if fast_info is unavailable.
    Returns 0.0 on complete failure.
    """
    import yfinance as yf
    try:
        t     = yf.Ticker(_NIFTY_TICKER)
        price = t.fast_info.last_price
        if price and float(price) > 0:
            return round(float(price), 2)
        # fallback: last close
        hist = t.history(period="2d", interval="1d", auto_adjust=True)
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception as exc:
        log.warning("Yahoo spot fetch failed: %s", exc)
    return 0.0


def get_india_vix() -> float:
    """
    Fetch India VIX from Yahoo Finance (^INDIAVIX).
    Returns 0.0 on failure.
    """
    import yfinance as yf
    try:
        t     = yf.Ticker(_VIX_TICKER)
        price = t.fast_info.last_price
        if price and float(price) > 0:
            return round(float(price), 2)
        hist = t.history(period="2d", interval="1d", auto_adjust=True)
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception as exc:
        log.warning("Yahoo VIX fetch failed: %s", exc)
    return 0.0


def get_spot_and_vix() -> dict:
    """
    Convenience: fetch both NIFTY spot and India VIX in one call.
    Returns {"spot": float, "vix": float, "fetched_at": str}
    """
    return {
        "spot":       get_nifty_spot(),
        "vix":        get_india_vix(),
        "fetched_at": datetime.now().strftime("%H:%M:%S"),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = get_spot_and_vix()
    print(f"NIFTY spot : {data['spot']}")
    print(f"India VIX  : {data['vix']}")
    print(f"Fetched at : {data['fetched_at']}")
