"""
tools/implementations.py
------------------------
All tool functions callable by the Claude agent.

Data sources:
  NSE India  → expiries, option chain OI, chart data   (nifty_option_chain + nifty_chart)
  Yahoo Finance (^NSEI, ^INDIAVIX) → spot price, India VIX only
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime

import pandas as pd

# NSE — option chain, expiries, OI, charts
from data.nifty_option_chain import (
    get_expiry_dates,
    get_nifty_option_chain,
    filter_atm,
)
from data.nifty_chart import get_option_chart, get_both_charts

# Yahoo — spot price + VIX only
from data.yahoo_feed import get_nifty_spot, get_india_vix, get_spot_and_vix


# ── JSON helpers ───────────────────────────────────────────────────────────────

class _SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        try:
            if pd.isna(obj): return None
        except Exception:
            pass
        return super().default(obj)


def _dumps(obj, **kw) -> str:
    # No indent — saves ~40% tokens vs indent=2
    # indent=2 turns {"a":1} into 3 lines; compact keeps it 1
    kw["separators"] = (",", ":")
    kw["cls"]        = _SafeEncoder
    return json.dumps(obj, **kw)


def _err(msg: str) -> str:
    return _dumps({"error": msg})


def _max_pain(df: pd.DataFrame) -> int:
    strikes = sorted(df["Strike"].unique())
    best, pain = strikes[0], float("inf")
    for s in strikes:
        loss = (
            ((df["Strike"] - s).clip(lower=0) * df["CE_OI"]).sum() +
            ((s - df["Strike"]).clip(lower=0) * df["PE_OI"]).sum()
        )
        if loss < pain:
            pain, best = loss, s
    return int(best)


# ── Tool: list_expiries ────────────────────────────────────────────────────────

def tool_list_expiries(_params: dict) -> str:
    """List all available NIFTY expiry dates from NSE."""
    try:
        dates = get_expiry_dates()
        return _dumps({
            "expiries": dates,
            "count":    len(dates),
            "nearest":  dates[0] if dates else None,
            "source":   "NSE India",
        })
    except Exception as exc:
        return _err(f"Failed to fetch expiry dates: {exc}")


# ── Tool: get_spot_price ───────────────────────────────────────────────────────

def tool_get_spot_price(_params: dict) -> str:
    """
    Fetch current NIFTY 50 spot price and India VIX from Yahoo Finance (^NSEI, ^INDIAVIX).
    Also returns nearest expiry from NSE.
    """
    try:
        yf_data  = get_spot_and_vix()
        spot     = yf_data["spot"]
        vix      = yf_data["vix"]

        # Nearest expiry still from NSE
        try:
            dates  = get_expiry_dates()
            expiry = dates[0] if dates else "N/A"
        except Exception:
            expiry = "N/A"

        return _dumps({
            "spot":            spot,
            "india_vix":       vix,
            "nearest_expiry":  expiry,
            "fetched_at":      yf_data["fetched_at"],
            "source":          "Yahoo Finance (^NSEI, ^INDIAVIX)",
        })
    except Exception as exc:
        return _err(f"Spot price fetch failed: {exc}")


# ── Tool: get_option_chain ─────────────────────────────────────────────────────

def tool_get_option_chain(params: dict) -> str:
    """
    Fetch NIFTY option chain from NSE India.
    Spot price overridden with Yahoo Finance live price.

    Params:
        expiry    (str, optional)  "DD-Mon-YYYY". Default: nearest.
        atm_range (int, optional)  ±N strikes around ATM. Default 10.
    """
    expiry    = params.get("expiry")
    atm_range = int(params.get("atm_range", 5))  # was 10 — 5 = 11 strikes, enough for context

    try:
        df, spot_nse = get_nifty_option_chain(expiry=expiry, atm_range=atm_range)
        # Prefer Yahoo spot; fall back to NSE if Yahoo fails
        spot = get_nifty_spot() or spot_nse
    except Exception as exc:
        return _err(f"Option chain fetch failed: {exc}\n{traceback.format_exc()}")

    if df is None or df.empty:
        return _err("No data returned — check expiry format (DD-Mon-YYYY).")

    # ── Defensive column normalization ────────────────────────────────────────
    col_map = {"strikePrice": "Strike", "strikeprice": "Strike", "strike": "Strike"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ["Strike","CE_OI","CE_Chng_OI","CE_IV","CE_LTP",
                "PE_OI","PE_Chng_OI","PE_IV","PE_LTP","CE_Volume","PE_Volume"]:
        if col not in df.columns:
            df[col] = 0
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].replace([float("inf"), float("-inf")], 0).fillna(0)
    df["Strike"] = df["Strike"].astype(int)

    expiry_label  = str(df["Expiry"].iloc[0])
    strikes       = sorted(df["Strike"].unique())
    atm           = min(strikes, key=lambda x: abs(x - spot))
    ce_oi_total   = int(df["CE_OI"].sum())
    pe_oi_total   = int(df["PE_OI"].sum())
    pcr           = round(pe_oi_total / ce_oi_total, 2) if ce_oi_total else 0
    max_pain      = _max_pain(df)
    max_ce_strike = int(df.loc[df["CE_OI"].idxmax(), "Strike"])
    max_pe_strike = int(df.loc[df["PE_OI"].idxmax(), "Strike"])

    chain = [
        {
            "strike":     int(row["Strike"]),
            "atm":        row["Strike"] == atm,
            "ce_ltp":     float(row["CE_LTP"]),
            "ce_iv":      float(row["CE_IV"]),
            "ce_oi":      int(row["CE_OI"]),
            "ce_chng_oi": int(row["CE_Chng_OI"]),
            "ce_volume":  int(row["CE_Volume"]),
            "pe_ltp":     float(row["PE_LTP"]),
            "pe_iv":      float(row["PE_IV"]),
            "pe_oi":      int(row["PE_OI"]),
            "pe_chng_oi": int(row["PE_Chng_OI"]),
            "pe_volume":  int(row["PE_Volume"]),
        }
        for _, row in df.iterrows()
    ]

    return _dumps({
        "source":           "NSE India (OI/chain)  +  Yahoo Finance (spot)",
        "expiry":           expiry_label,
        "spot":             round(spot, 2),
        "atm_strike":       int(atm),
        "pcr":              pcr,
        "max_pain":         max_pain,
        "max_ce_oi_strike": max_ce_strike,
        "max_pe_oi_strike": max_pe_strike,
        "ce_oi_total":      ce_oi_total,
        "pe_oi_total":      pe_oi_total,
        "strikes_shown":    len(df),
        "chain":            chain,
        "fetched_at":       datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
    })


# ── Tool: get_oi_analysis ──────────────────────────────────────────────────────

def tool_get_oi_analysis(params: dict) -> str:
    """
    Deep OI analysis from NSE: PCR, max pain, key support/resistance,
    OI buildup/unwinding, IV skew.
    VIX fetched from Yahoo Finance.

    Params:
        expiry (str, optional): specific expiry. Default: nearest.
    """
    expiry = params.get("expiry")

    try:
        df, spot_nse = get_nifty_option_chain(expiry=expiry)
        spot = get_nifty_spot() or spot_nse
        vix  = get_india_vix()
    except Exception as exc:
        return _err(f"OI analysis fetch failed: {exc}")

    if df is None or df.empty:
        return _err("No data returned.")

    # ── Defensive column normalization (same as scanner) ──────────────────────
    col_map = {"strikePrice": "Strike", "strikeprice": "Strike", "strike": "Strike"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ["Strike","CE_OI","CE_Chng_OI","CE_IV","CE_LTP",
                "PE_OI","PE_Chng_OI","PE_IV","PE_LTP","CE_Volume","PE_Volume"]:
        if col not in df.columns:
            df[col] = 0
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].replace([float("inf"), float("-inf")], 0).fillna(0)
    df["Strike"] = df["Strike"].astype(int)

    strikes     = sorted(df["Strike"].unique())
    atm         = min(strikes, key=lambda x: abs(x - spot))
    ce_oi_total = int(df["CE_OI"].sum())
    pe_oi_total = int(df["PE_OI"].sum())
    pcr         = round(pe_oi_total / ce_oi_total, 2) if ce_oi_total else 0
    max_pain    = _max_pain(df)

    if   pcr >= 1.3: sentiment = "BULLISH — strong PE writing, market well supported"
    elif pcr >= 0.9: sentiment = "NEUTRAL-BULLISH — balanced with slight bullish lean"
    elif pcr >= 0.7: sentiment = "NEUTRAL-BEARISH — mild bearish pressure"
    else:            sentiment = "BEARISH — heavy CE writing, resistance overhead"

    top_ce       = df.nlargest(5, "CE_OI")[["Strike","CE_OI","CE_Chng_OI","CE_IV","CE_LTP"]]
    top_pe       = df.nlargest(5, "PE_OI")[["Strike","PE_OI","PE_Chng_OI","PE_IV","PE_LTP"]]
    ce_building  = df[df["CE_Chng_OI"] > 0].nlargest(3,  "CE_Chng_OI")[["Strike","CE_Chng_OI"]]
    pe_building  = df[df["PE_Chng_OI"] > 0].nlargest(3,  "PE_Chng_OI")[["Strike","PE_Chng_OI"]]
    ce_unwinding = df[df["CE_Chng_OI"] < 0].nsmallest(3, "CE_Chng_OI")[["Strike","CE_Chng_OI"]]
    pe_unwinding = df[df["PE_Chng_OI"] < 0].nsmallest(3, "PE_Chng_OI")[["Strike","PE_Chng_OI"]]

    atm_idx    = strikes.index(atm)
    skew_range = strikes[max(0, atm_idx - 3): atm_idx + 4]
    iv_skew    = df[df["Strike"].isin(skew_range)][["Strike","CE_IV","PE_IV"]]

    return _dumps({
        "source":           "NSE India (OI/chain)  +  Yahoo Finance (spot, VIX)",
        "expiry":           str(df["Expiry"].iloc[0]),
        "spot":             round(spot, 2),
        "atm":              int(atm),
        "pcr":              pcr,
        "sentiment":        sentiment,
        "max_pain":         max_pain,
        "india_vix":        vix,
        "ce_oi_total":      ce_oi_total,
        "pe_oi_total":      pe_oi_total,
        "key_resistance": [
            {"strike": int(r.Strike), "ce_oi": int(r.CE_OI),
             "ce_chng_oi": int(r.CE_Chng_OI), "ce_iv": float(r.CE_IV), "ce_ltp": float(r.CE_LTP)}
            for r in top_ce.itertuples()
        ],
        "key_support": [
            {"strike": int(r.Strike), "pe_oi": int(r.PE_OI),
             "pe_chng_oi": int(r.PE_Chng_OI), "pe_iv": float(r.PE_IV), "pe_ltp": float(r.PE_LTP)}
            for r in top_pe.itertuples()
        ],
        "fresh_ce_writing": [
            {"strike": int(r.Strike), "added_oi": int(r.CE_Chng_OI)}
            for r in ce_building.itertuples()
        ],
        "fresh_pe_writing": [
            {"strike": int(r.Strike), "added_oi": int(r.PE_Chng_OI)}
            for r in pe_building.itertuples()
        ],
        "ce_unwinding": [
            {"strike": int(r.Strike), "reduced_oi": int(r.CE_Chng_OI)}
            for r in ce_unwinding.itertuples()
        ],
        "pe_unwinding": [
            {"strike": int(r.Strike), "reduced_oi": int(r.PE_Chng_OI)}
            for r in pe_unwinding.itertuples()
        ],
        "iv_skew": [
            {"strike": int(r.Strike), "ce_iv": float(r.CE_IV), "pe_iv": float(r.PE_IV)}
            for r in iv_skew.itertuples()
        ],
        "fetched_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
    })


# ── Tool: get_chart_data ───────────────────────────────────────────────────────

def tool_get_chart_data(params: dict) -> str:
    """
    Fetch intraday price/volume data for a specific NIFTY option strike from NSE.

    Params:
        expiry      (str)   "DD-Mon-YYYY"
        strike      (float) e.g. 24500
        option_type (str)   "CE" | "PE" | "BOTH"  (default "CE")
    """
    expiry      = params.get("expiry")
    strike      = params.get("strike")
    option_type = str(params.get("option_type", "CE")).upper()

    if not expiry or strike is None:
        return _err("Both 'expiry' and 'strike' are required.")
    try:
        strike = float(strike)
    except (TypeError, ValueError):
        return _err(f"Invalid strike: {strike!r}")

    def _summarise(df: pd.DataFrame, otype: str) -> dict:
        if df.empty:
            return {"option_type": otype, "error": "No data returned"}
        ltp   = float(df["price"].iloc[-1])
        open_ = float(df["price"].iloc[0])
        return {
            "option_type":  otype,
            "data_points":  len(df),
            "from":         str(df["timestamp"].iloc[0]),
            "to":           str(df["timestamp"].iloc[-1]),
            "open":         round(open_, 2),
            "high":         round(float(df["price"].max()), 2),
            "low":          round(float(df["price"].min()), 2),
            "ltp":          round(ltp, 2),
            "prev_close":   round(float(df["close_price"].iloc[0]), 2)
                            if df["close_price"].iloc[0] else None,
            "total_volume": int(df["volume"].sum()),
            "price_change": round(ltp - open_, 2),
            "pct_change":   round((ltp - open_) / open_ * 100, 2) if open_ else 0,
            "recent_ticks": [
                {"time": str(r.timestamp), "price": round(float(r.price), 2),
                 "volume": int(r.volume)}
                for r in df.tail(5).itertuples()
            ],
        }

    try:
        if option_type == "BOTH":
            df_ce, df_pe = get_both_charts(expiry, strike)
            return _dumps({
                "source":     "NSE India",
                "expiry":     expiry,
                "strike":     strike,
                "CE":         _summarise(df_ce, "CE"),
                "PE":         _summarise(df_pe, "PE"),
                "fetched_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
            })
        else:
            df = get_option_chart(expiry, strike, option_type)
            return _dumps({
                "source":     "NSE India",
                "expiry":     expiry,
                "strike":     strike,
                **_summarise(df, option_type),
                "fetched_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
            })
    except Exception as exc:
        return _err(f"Chart data fetch failed: {exc}\n{traceback.format_exc()}")


# ── Registry ───────────────────────────────────────────────────────────────────

TOOL_FUNCTIONS: dict[str, callable] = {
    "list_expiries":    tool_list_expiries,
    "get_spot_price":   tool_get_spot_price,
    "get_option_chain": tool_get_option_chain,
    "get_oi_analysis":  tool_get_oi_analysis,
    "get_chart_data":   tool_get_chart_data,
}


def dispatch(name: str, params: dict) -> str:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return _err(f"Unknown tool '{name}'. Available: {', '.join(TOOL_FUNCTIONS)}")
    try:
        return fn(params)
    except Exception as exc:
        return _err(f"Tool '{name}' crashed: {exc}\n{traceback.format_exc()}")


# ── Tool: get_market_news ──────────────────────────────────────────────────────

def tool_get_market_news(params: dict) -> str:
    """
    Fetch latest market-moving news via Google News (gnews).
    Covers NIFTY, RBI, FII flows, geopolitical events.
    Params:
        days     (int, optional): lookback days. Default 1.
        category (str, optional): market|geopolitical|rbi_sebi|fii|all. Default all.
    """
    try:
        from data.news_feed import (
            fetch_market_news, fetch_geopolitical_news,
            fetch_rbi_sebi_news, fetch_fii_news, fetch_all_news,
        )
        days     = int(params.get("days", 1))
        category = str(params.get("category", "all")).lower()

        fn_map = {
            "market":      fetch_market_news,
            "geopolitical": fetch_geopolitical_news,
            "rbi_sebi":    fetch_rbi_sebi_news,
            "fii":         fetch_fii_news,
            "all":         fetch_all_news,
        }
        articles = fn_map.get(category, fetch_all_news)(days=days)

        if not articles:
            return _dumps({"news": [], "count": 0, "message": "No news found — gnews may need pip install gnews"})

        return _dumps({
            "source":   "Google News (gnews)",
            "category": category,
            "days":     days,
            "count":    len(articles),
            "news": [
                {
                    "title":     a.title,
                    "source":    a.source,
                    "published": a.published,
                    "sentiment": a.sentiment,
                    "impact":    a.impact,
                    "url":       a.url,
                }
                for a in articles[:12]
            ],
            "sentiment_summary": {
                "bullish": sum(1 for a in articles if a.sentiment == "BULLISH"),
                "bearish": sum(1 for a in articles if a.sentiment == "BEARISH"),
                "neutral": sum(1 for a in articles if a.sentiment == "NEUTRAL"),
            },
            "fetched_at": datetime.now().strftime("%d-%b-%Y %H:%M:%S"),
        })
    except Exception as exc:
        return _err(f"News fetch failed: {exc}")


# ── Tool: get_fii_dii_data ─────────────────────────────────────────────────────

def tool_get_fii_dii_data(_params: dict) -> str:
    """
    Fetch FII and DII cash market activity from NSE.
    FII net buying = bullish signal. FII net selling = bearish.
    """
    try:
        from data.market_extra import get_fii_dii_data
        return _dumps(get_fii_dii_data())
    except Exception as exc:
        return _err(f"FII/DII fetch failed: {exc}")


# ── Tool: get_global_market_cues ──────────────────────────────────────────────

def tool_get_global_market_cues(_params: dict) -> str:
    """
    Fetch global indices: S&P500, Nasdaq, Dow, Hang Seng, Nikkei, Crude Oil,
    Gold, Dollar Index, US 10Y yield.
    Global cues directly influence NIFTY's opening direction.
    """
    try:
        from data.market_extra import get_global_indices
        data = get_global_indices()
        return _dumps({
            "source": "Yahoo Finance (global indices)",
            **data,
        })
    except Exception as exc:
        return _err(f"Global cues fetch failed: {exc}")


# ── Tool: get_nifty_technicals ─────────────────────────────────────────────────

def tool_get_nifty_technicals(_params: dict) -> str:
    """
    Compute NIFTY intraday technicals from 5-min candles:
    EMA9, EMA21, VWAP, day high/low, trend direction, tech bias.
    Use this alongside OI analysis for high-confidence trade setups.
    """
    try:
        from data.market_extra import get_nifty_technicals
        return _dumps(get_nifty_technicals())
    except Exception as exc:
        return _err(f"Technicals fetch failed: {exc}")


# ── Tool: get_vix_analysis ────────────────────────────────────────────────────

def tool_get_vix_analysis(_params: dict) -> str:
    """
    India VIX current level + 5-day trend + regime classification.
    VIX regime determines strategy type:
      LOW_FEAR → sell premium | ELEVATED → buy options | EXTREME → no trades
    """
    try:
        from data.market_extra import get_india_vix_history
        return _dumps(get_india_vix_history(days=5))
    except Exception as exc:
        return _err(f"VIX analysis failed: {exc}")


# ── Update registry ───────────────────────────────────────────────────────────

TOOL_FUNCTIONS.update({
    "get_market_news":        tool_get_market_news,
    "get_fii_dii_data":       tool_get_fii_dii_data,
    "get_global_market_cues": tool_get_global_market_cues,
    "get_nifty_technicals":   tool_get_nifty_technicals,
    "get_vix_analysis":       tool_get_vix_analysis,
})
