"""
data/groww_feed.py
------------------
Groww feed for NIFTY expiries, option chain and option premium candles.

VERIFIED against the live site (Sep 2026):
  • Expiries + full chain arrive embedded in the HTML of
    https://groww.in/options/nifty inside <script id="__NEXT_DATA__">.
    Path: props.pageProps.data.optionChain
      ├─ optionContracts[] : {strikePrice (paisa → ÷100), ce{}, pe{}}
      │    ce/pe : { growwContractId, displayName, token, marketLot,
      │              liveData{ ltp, close, dayChange, oi, prevOI },
      │              greeks{ iv, delta, theta, pop } }
      └─ aggregatedDetails{ currentExpiry, expiryDates[], lotSize, maxOI }
  • JSON endpoints under groww.in/v1/api/.../option-chain returned 502
    during testing — the HTML route is the reliable one.
  • Premium candle history for individual contracts: Groww renders charts
    client-side; no stable public JSON was verifiable. fetch_option_candles()
    therefore returns empty and callers use their fallback premium model.
    (Groww Trading API — api.groww.in — supports candles but needs auth.)

All functions are resilient: failures return empty structures, never raise.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────

GROWW_OPTIONS_PAGE = "https://groww.in/options/nifty"
GROWW_CHAIN_API    = ("https://groww.in/v1/api/stocks_data/derivatives/v1/"
                      "option-chain?underlying=NIFTY&expiry={expiry}")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept":     "text/html,application/json, */*",
    "Referer":    "https://groww.in/options/nifty",
    "Accept-Language": "en-US,en;q=0.9",
}

_TIMEOUT = 15

_NEXT_DATA_RE = re.compile(
    r'id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)


def _get(url: str, as_json: bool = False) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200:
            log.warning("Groww GET %s → HTTP %d", url.split("?")[0], r.status_code)
            return None
        return r
    except Exception as exc:
        log.warning("Groww GET failed: %s", exc)
        return None


# ── Page scraper (primary verified route) ─────────────────────────────────────

def _fetch_next_data() -> Optional[dict]:
    """Fetch the options page and parse the embedded __NEXT_DATA__ JSON."""
    r = _get(GROWW_OPTIONS_PAGE)
    if r is None:
        return None
    m = _NEXT_DATA_RE.search(r.text)
    if not m:
        log.warning("Groww page: __NEXT_DATA__ not found")
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        log.warning("Groww page: NEXT_DATA JSON parse failed: %s", exc)
        return None


def _option_chain_from_next(data: dict) -> Optional[dict]:
    """Extract + normalize the optionChain block from __NEXT_DATA__."""
    try:
        oc = data["props"]["pageProps"]["data"]["optionChain"]
    except (KeyError, TypeError):
        return None

    agg = oc.get("aggregatedDetails", {}) or {}
    out: dict = {
        "current_expiry": agg.get("currentExpiry", ""),
        "expiry_dates":   list(agg.get("expiryDates", []) or []),
        "lot_size":       agg.get("lotSize", 75),
        "rows": [],
    }

    spot = 0.0
    for c in oc.get("optionContracts", []) or []:
        try:
            strike = round(float(c.get("strikePrice", 0)) / 100.0)
        except (TypeError, ValueError):
            continue
        row = {"strike": strike, "ce": {}, "pe": {}}
        for side in ("ce", "pe"):
            leg = c.get(side) or {}
            live = leg.get("liveData", {}) or {}
            greeks = leg.get("greeks", {}) or {}
            row[side] = {
                "contract_id": leg.get("growwContractId", ""),
                "display":     leg.get("displayName", ""),
                "token":       leg.get("token", ""),
                "ltp":         _num(live.get("ltp")),
                "close":       _num(live.get("close")),
                "change":      _num(live.get("dayChange")),
                "oi":          _num(live.get("oi")),
                "prev_oi":     _num(live.get("prevOI")),
                "iv":          _num(greeks.get("iv")),
                "delta":       _num(greeks.get("delta")),
                "theta":       _num(greeks.get("theta")),
                "pop":         _num(greeks.get("pop")),
            }
        # Groww doesn't expose spot on this page — infer from ATM
        ce, pe = row["ce"], row["pe"]
        if ce.get("ltp") and pe.get("ltp"):
            # spot where |CE-PE| is smallest approximates ATM
            row["_abs_diff"] = abs(ce["ltp"] - pe["ltp"])
        out["rows"].append(row)

    rows = out["rows"]
    if rows:
        # Convert OI numbers into walls / PCR / max pain
        ce_oi = sum(r["ce"]["oi"] for r in rows)
        pe_oi = sum(r["pe"]["oi"] for r in rows)
        if ce_oi:
            out["pcr"] = round(pe_oi / ce_oi, 2)
        ce_top = max(rows, key=lambda r: r["ce"]["oi"])
        pe_top = max(rows, key=lambda r: r["pe"]["oi"])
        out["ce_wall"] = ce_top["strike"]
        out["pe_wall"] = pe_top["strike"]

        # Max pain (canonical): the settlement strike where option BUYERS
        # collectively receive the LEAST intrinsic value:
        #   pain(s) = Σ max(0, s−K)·CE_OI + Σ max(0, K−s)·PE_OI   → argmin
        strikes = sorted(r["strike"] for r in rows)
        oi_ce = {r["strike"]: r["ce"]["oi"] for r in rows}
        oi_pe = {r["strike"]: r["pe"]["oi"] for r in rows}
        pain = {}
        for s in strikes:
            loss = (sum(max(0, s - k) * v for k, v in oi_ce.items())
                    + sum(max(0, k - s) * v for k, v in oi_pe.items()))
            pain[s] = loss
        out["max_pain"] = int(min(pain, key=pain.get))

        # ATM estimate: smallest |CE ltp − PE ltp|
        atm_rows = [r for r in rows if "_abs_diff" in r]
        if atm_rows:
            atm = min(atm_rows, key=lambda r: r["_abs_diff"])["strike"]
            out["atm"] = atm
            out["spot"] = float(atm)

    out["spot"] = spot
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def _norm_expiry(e: str) -> str:
    """
    Normalize an expiry to DD-Mon-YYYY (the format NSE tools expect).
    Groww returns YYYY-MM-DD; NSE already returns DD-Mon-YYYY.
    """
    e = str(e or "").strip()
    if not e:
        return ""
    if "-" in e and len(e) == 10:
        parts = e.split("-")
        if len(parts) == 3 and len(parts[0]) == 4:   # 2026-09-22
            try:
                return datetime.strptime(e, "%Y-%m-%d").strftime("%d-%b-%Y")
            except ValueError:
                return e
    try:
        datetime.strptime(e, "%d-%b-%Y")             # already 22-Sep-2026
        return e
    except ValueError:
        return e


def get_expiries() -> list[str]:
    """
    NIFTY expiry dates, nearest first, normalized to DD-Mon-YYYY
    (Groww returns YYYY-MM-DD; NSE returns DD-Mon-YYYY).
    """
    data = _fetch_next_data()
    if data:
        oc = _option_chain_from_next(data)
        if oc and oc.get("expiry_dates"):
            dates = [_norm_expiry(e) for e in oc["expiry_dates"]]
            dates = [d for d in dates if d]
            if dates:
                return (_norm_expiry(oc.get("current_expiry", "")) or dates[0], dates)

    # ── Fallback: NSE ─────────────────────────────────────────────────────
    try:
        from data.nifty_option_chain import get_expiry_dates
        nse = get_expiry_dates() or []
        return (nse[0] if nse else ""), nse
    except Exception as exc:
        log.warning("NSE expiry fallback failed: %s", exc)
        return "", []


def get_option_chain_snapshot(expiry: str = "") -> dict:
    """
    Compact chain snapshot for the UI flow:
      spot, pcr, max_pain, ce_wall, pe_wall, atm, expiry, source.
    Groww first; NSE fallback fills whatever is missing.
    """
    out: dict = {"source": "groww", "expiry": expiry}

    data = _fetch_next_data()
    oc = _option_chain_from_next(data) if data else None

    if oc:
        out.update({
            "spot":       oc.get("spot", 0),
            "atm":        oc.get("atm", 0),
            "pcr":        oc.get("pcr", 0),
            "max_pain":   oc.get("max_pain", 0),
            "ce_wall":    oc.get("ce_wall", 0),
            "pe_wall":    oc.get("pe_wall", 0),
            "expiries":   oc.get("expiry_dates", []),
            "groww_expiry": oc.get("current_expiry", ""),
        })
    else:
        out["source"] = "nse-fallback"
        try:
            from data.nifty_option_chain import get_nifty_option_chain
            df, spot_nse = get_nifty_option_chain(expiry=expiry or None)
            if spot_nse:
                out["spot"] = spot_nse
            if df is not None and not df.empty:
                num = df.select_dtypes(include="number").columns
                df[num] = df[num].replace([float("inf"), float("-inf")], 0).fillna(0)
                strikes = sorted(df["Strike"].unique().tolist())
                if strikes:
                    atm = min(strikes, key=lambda x: abs(x - (out.get("spot") or 0)))
                    out["atm"] = atm
                    pain = {}
                    for s in strikes:
                        loss = (
                            ((df["Strike"] - s).clip(lower=0) * df["CE_OI"]).sum()
                            + ((s - df["Strike"]).clip(lower=0) * df["PE_OI"]).sum()
                        )
                        pain[s] = float(loss)
                    out["max_pain"] = int(min(pain, key=pain.get))
                    ce_top = df.nlargest(1, "CE_OI")
                    pe_top = df.nlargest(1, "PE_OI")
                    out["ce_wall"] = int(ce_top["Strike"].iloc[0]) if not ce_top.empty else 0
                    out["pe_wall"] = int(pe_top["Strike"].iloc[0]) if not pe_top.empty else 0
                    ce_oi = float(df["CE_OI"].sum())
                    pe_oi = float(df["PE_OI"].sum())
                    if ce_oi:
                        out["pcr"] = round(pe_oi / ce_oi, 2)
        except Exception as exc:
            log.warning("NSE chain fallback failed: %s", exc)

    # Last-resort spot from Yahoo
    if not out.get("spot"):
        try:
            from data.yahoo_feed import get_nifty_spot
            out["spot"] = get_nifty_spot()
        except Exception:
            pass

    out["spot"] = out.get("spot") or 0
    return out


def get_chain_table(expiry: str = "") -> list[dict]:
    """
    Full chain rows for the selected expiry:
      [{strike, ce:{ltp,oi,iv,oi_change,...}, pe:{...}}, ...] sorted by strike.
    """
    data = _fetch_next_data()
    if not data:
        return []
    oc = _option_chain_from_next(data)
    if not oc:
        return []
    rows = oc.get("rows", [])
    for r in rows:
        for side in ("ce", "pe"):
            r[side]["oi_change"] = r[side]["oi"] - r[side]["prev_oi"]
    return sorted(rows, key=lambda r: r["strike"])


# ── Option premium candles ────────────────────────────────────────────────────

def fetch_option_candles(expiry: str, strike: int, side: str,
                         interval: str = "5m", limit: int = 600) -> pd.DataFrame:
    """
    Historical premium candles for one option contract from Groww.

    NOTE: Groww builds its option charts client-side and no stable public
    JSON candle endpoint was verifiable (Trading API api.groww.in supports
    candles but requires auth tokens). This returns an empty DataFrame and
    callers fall back to their premium model. Kept for future wiring.
    """
    log.info("Groww premium candles unavailable without auth — "
             "caller fallback applies (%s %s%s)", strike, side, expiry)
    return pd.DataFrame()


# ── Index candles (for Kronos + momentum) ─────────────────────────────────────

def fetch_index_candles(days: int = 5, interval: str = "5m") -> pd.DataFrame:
    """NIFTY index OHLCV — Yahoo, same shape sim_mode expects."""
    try:
        import yfinance as yf
        hist = yf.Ticker("^NSEI").history(period=f"{days}d", interval=interval,
                                          auto_adjust=True)
        if hist is None or hist.empty:
            return pd.DataFrame()
        hist = hist.reset_index()
        hist.columns = [str(c).lower() for c in hist.columns]
        return hist
    except Exception as exc:
        log.warning("Index candle fetch failed: %s", exc)
        return pd.DataFrame()


def _num(v) -> float:
    try:
        f = float(v)
        return f if f == f and f not in (float("inf"), float("-inf")) else 0.0
    except (TypeError, ValueError):
        return 0.0


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=logging.INFO)
    cur, expiries = get_expiries()
    print("current expiry:", cur)
    print("expiries:", expiries)
    snap = get_option_chain_snapshot(cur)
    print("snapshot:", {k: snap.get(k) for k in
                        ("source", "spot", "atm", "pcr", "max_pain", "ce_wall", "pe_wall")})
