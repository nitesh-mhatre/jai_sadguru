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
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

# ── Endpoints ─────────────────────────────────────────────────────────────────

GROWW_OPTIONS_PAGE = "https://groww.in/options/nifty"
GROWW_CHAIN_API    = ("https://groww.in/v1/api/stocks_data/derivatives/v1/"
                      "option-chain?underlying=NIFTY&expiry={expiry}")

# Groww delayed charting service. Verified live for NSE equities
# (segment/CASH returns real OHLCV). F&O instruments are addressed by the
# contract id carried in the option chain (growwContractId).
GROWW_CHART_API = ("https://groww.in/v1/api/charting_service/v2/chart/delayed/"
                   "exchange/NSE/segment/{segment}/{instrument}")

# interval label → minutes (None = unsupported on this route)
_INTERVAL_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "10m": 10, "15m": 15,
    "30m": 30, "60m": 60, "1h": 60, "1d": None,
}

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
            atm_row = min(atm_rows, key=lambda r: r["_abs_diff"])
            atm     = atm_row["strike"]
            out["atm"] = atm
            ce_ltp = float(atm_row["ce"].get("ltp", 0.0) or 0.0)
            pe_ltp = float(atm_row["pe"].get("ltp", 0.0) or 0.0)
            # Put-call parity at the money: C − P ≈ S − K  →  S ≈ K + (C − P).
            # Groww's page omits spot, so this estimate is the closest proxy;
            # callers prefer a live index quote when one is available.
            if ce_ltp > 0 and pe_ltp > 0:
                out["spot_est"] = round(atm + (ce_ltp - pe_ltp), 2)
                out["atm_straddle"] = round(ce_ltp + pe_ltp, 2)
            else:
                out["spot_est"] = float(atm)

    out["spot"] = out.get("spot_est") or 0.0
    return out


# ── Chain cache ───────────────────────────────────────────────────────────────
# The live trader asks for spot/OI/LTP every few seconds. Scraping the page on
# every request would hammer Groww and slow the loop, so the normalized chain is
# cached briefly — a chain a few seconds old is still accurate enough to price
# an open position.

_CHAIN_TTL   = 8.0                                  # seconds
_CHAIN_CACHE: dict = {"ts": 0.0, "data": None}


def get_chain(force: bool = False) -> Optional[dict]:
    """
    Normalized option-chain block for the expiry Groww is currently serving.
    Cached for _CHAIN_TTL seconds; pass force=True to bypass the cache.
    Returns None when the page cannot be fetched or parsed.
    """
    now = time.time()
    cached = _CHAIN_CACHE.get("data")
    if cached is not None and not force and (now - _CHAIN_CACHE["ts"]) < _CHAIN_TTL:
        return cached

    data = _fetch_next_data()
    oc   = _option_chain_from_next(data) if data else None
    if oc:
        _CHAIN_CACHE["data"] = oc
        _CHAIN_CACHE["ts"]   = now
    return oc


def chain_age_seconds() -> float:
    """Age of the cached chain in seconds (−1.0 when nothing is cached yet)."""
    if not _CHAIN_CACHE.get("data"):
        return -1.0
    return round(time.time() - _CHAIN_CACHE["ts"], 1)


def get_quote(expiry: str, strike: int, side: str) -> dict:
    """
    Live quote for one option contract straight from the Groww chain.

    Returns a dict with ltp / close / change / oi / oi_change / iv / delta /
    theta / pop plus `chain_age_s`, or {} when the strike/side is not in the
    chain. Never raises.
    """
    side = (side or "").upper()
    if side not in ("CE", "PE"):
        return {}
    try:
        oc = get_chain()
    except Exception as exc:
        log.warning("Groww quote fetch failed: %s", exc)
        return {}
    if not oc:
        return {}
    try:
        target = int(round(float(strike)))
    except (TypeError, ValueError):
        return {}

    for r in oc.get("rows", []):
        if int(r.get("strike", 0)) != target:
            continue
        leg  = r.get(side.lower(), {}) or {}
        oi   = _num(leg.get("oi"))
        prev = _num(leg.get("prev_oi"))
        return {
            "strike":      target,
            "side":        side,
            "ltp":         _num(leg.get("ltp")),
            "close":       _num(leg.get("close")),
            "change":      _num(leg.get("change")),
            "oi":          int(oi),
            "prev_oi":     int(prev),
            "oi_change":   int(oi - prev),
            "iv":          _num(leg.get("iv")),
            "delta":       _num(leg.get("delta")),
            "theta":       _num(leg.get("theta")),
            "pop":         _num(leg.get("pop")),
            "expiry":      _norm_expiry(oc.get("current_expiry", "")) or expiry,
            "source":      "groww",
            "chain_age_s": chain_age_seconds(),
        }
    return {}


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
    oc = get_chain()
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

    oc = get_chain()

    if oc:
        out.update({
            "spot":       oc.get("spot", 0),
            "atm":        oc.get("atm", 0),
            "pcr":        oc.get("pcr", 0),
            "max_pain":   oc.get("max_pain", 0),
            "ce_wall":    oc.get("ce_wall", 0),
            "pe_wall":    oc.get("pe_wall", 0),
            "expiries":   oc.get("expiry_dates", []),
            "groww_expiry": _norm_expiry(oc.get("current_expiry", "")),
            "atm_straddle": oc.get("atm_straddle", 0),
            "chain_age_s":  chain_age_seconds(),
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


def get_levels_snapshot(expiry: str, levels: list | None = None) -> dict:
    """
    Current market strip for the UI: the live NIFTY spot block plus a live quote
    for every tracked level.

    Returns
    -------
    {
      "snapshot": {spot, atm, pcr, max_pain, ce_wall, pe_wall, atm_straddle, ...},
      "levels":   [{strike, side, ltp, oi, oi_change, iv, delta, theta, ...}, ...],
    }
    A level with no chain quote is returned with ltp=0 so the caller can show it
    as '—' instead of dropping the row silently.
    """
    snap   = get_option_chain_snapshot(expiry)
    quotes: list[dict] = []
    for lv in levels or []:
        try:
            strike = int(lv.get("strike"))
            side   = str(lv.get("side", "")).upper()
        except (TypeError, ValueError):
            continue
        q = get_quote(expiry, strike, side)
        quotes.append(q or {"strike": strike, "side": side, "ltp": 0.0,
                            "oi": 0, "oi_change": 0, "iv": 0.0,
                            "delta": 0.0, "theta": 0.0, "source": "none"})
    return {"snapshot": snap, "levels": quotes}


def get_chain_table(expiry: str = "") -> list[dict]:
    """
    Full chain rows for the selected expiry:
      [{strike, ce:{ltp,oi,iv,oi_change,...}, pe:{...}}, ...] sorted by strike.
    """
    oc = get_chain()
    if not oc:
        return []
    rows = oc.get("rows", [])
    for r in rows:
        for side in ("ce", "pe"):
            r[side]["oi_change"] = r[side]["oi"] - r[side]["prev_oi"]
    return sorted(rows, key=lambda r: r["strike"])


# ── Option premium candles ────────────────────────────────────────────────────

def get_contract_id(expiry: str, strike: int, side: str) -> str:
    """
    Groww contract id for one option leg, taken from the cached chain row.
    Empty string when the chain (or that strike/expiry) is unavailable.
    """
    side = (side or "").upper()
    oc = get_chain()
    if not oc:
        return ""
    try:
        target = int(round(float(strike)))
    except (TypeError, ValueError):
        return ""
    for r in oc.get("rows", []):
        if int(r.get("strike", 0)) != target:
            continue
        leg = r.get(side.lower(), {}) or {}
        return str(leg.get("contract_id") or leg.get("token") or "")
    return ""


def _candles_from_groww_chart(instrument: str, interval: str, limit: int,
                              days: int) -> pd.DataFrame:
    """Premium/history candles from the Groww delayed charting service."""
    mins = _INTERVAL_MINUTES.get(interval, 5)
    if mins is None:
        return pd.DataFrame()

    end_ms   = int(time.time() * 1000)
    start_ms = end_ms - max(days, 1) * 24 * 60 * 60 * 1000
    params   = {
        "intervalInMinutes": mins,
        "startTimeInMillis": start_ms,
        "endTimeInMillis":   end_ms,
    }

    for segment in ("FNO", "CASH"):
        url = GROWW_CHART_API.format(segment=segment, instrument=instrument)
        try:
            r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT, params=params)
            if r.status_code != 200:
                log.debug("Groww chart %s → HTTP %d", segment, r.status_code)
                continue
            rows = (r.json() or {}).get("candles") or []
            if not rows:
                continue
            # Groww rows: [epoch_seconds, open, high, low, close, volume]
            df = pd.DataFrame(rows)
            if df.shape[1] < 5:
                continue
            df = df.iloc[:, :6]
            df.columns = ["ts", "open", "high", "low", "close", "volume"][:df.shape[1]]
            df["timestamp"] = (pd.to_datetime(df["ts"], unit="s", utc=True)
                                 .dt.tz_convert("Asia/Kolkata")
                                 .dt.tz_localize(None))
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            return df.tail(limit).reset_index(drop=True)
        except Exception as exc:
            log.debug("Groww chart fetch failed (%s): %s", instrument, exc)
    return pd.DataFrame()


def _candles_from_nse(expiry: str, strike: int, side: str,
                      interval: str = "5m") -> pd.DataFrame:
    """
    Real NSE intraday premium candles (current session only) via the cookie
    session, resampled to the requested candle size.
    """
    try:
        from data.nifty_chart import build_identifier, fetch_chart_data
    except Exception:
        return pd.DataFrame()
    try:
        ident = build_identifier(_norm_expiry(expiry), float(strike), side)
        raw   = fetch_chart_data(ident)
    except Exception as exc:
        log.debug("NSE option chart failed for %s%s: %s", strike, side, exc)
        return pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame()

    try:
        df = raw.rename(columns={"price": "close"}).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        mins = _INTERVAL_MINUTES.get(interval, 5) or 5
        ohlc = df["close"].resample(f"{mins}min").ohlc()
        ohlc["volume"] = df.get("volume", pd.Series(0, index=df.index)) \
                           .resample(f"{mins}min").sum()
        return ohlc.dropna(subset=["close"]).reset_index()
    except Exception as exc:
        log.debug("NSE candle resample failed: %s", exc)
        return pd.DataFrame()


def fetch_option_candles(expiry: str, strike: int, side: str,
                        interval: str = "5m", limit: int = 600,
                        days: int = 5) -> pd.DataFrame:
    """
    Historical premium candles for one option contract.

    Sources, first that yields data wins:
      1. Groww charting service using the chain's own contract id (real Groww data)
      2. NSE intraday chart via the cookie session (real, current session)

    Returns a DataFrame with timestamp / open / high / low / close / volume / source,
    sorted oldest-first, or an empty DataFrame when neither source has data
    (the caller then builds an index-aligned model series).
    """
    instrument = get_contract_id(expiry, strike, side)
    if instrument:
        df = _candles_from_groww_chart(instrument, interval, limit, days)
        if not df.empty:
            df["source"] = "groww"
            log.info("Groww option candles %s%s: %d rows", strike, side, len(df))
            return df

    df = _candles_from_nse(expiry, strike, side, interval)
    if not df.empty:
        df["source"] = "nse"
        log.info("NSE option candles %s%s: %d rows", strike, side, len(df))
        return df.tail(limit).reset_index(drop=True)

    log.info("No live premium history for %s%s%s — caller will model it",
             strike, side, expiry)
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
