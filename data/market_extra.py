"""
data/market_extra.py
--------------------
Extra market data tools for higher accuracy trading decisions.

  get_fii_dii_data()      — FII/DII cash + derivatives activity (NSE)
  get_india_vix_history() — VIX trend over last 5 days (Yahoo)
  get_premarket_data()    — GIFT Nifty + SGX futures gap estimate
  get_global_indices()    — S&P500, Dow, Nasdaq, Hang Seng, Nikkei
  get_nifty_technicals()  — EMA9/21, VWAP, day high/low, intraday trend
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pytz

log = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


# ── FII / DII activity ─────────────────────────────────────────────────────────

def get_fii_dii_data() -> dict:
    """
    Fetch FII + DII cash market and derivatives activity from NSE.
    Returns net buy/sell with directional interpretation.
    """
    try:
        from data.nse_session import nse_get
        r    = nse_get("https://www.nseindia.com/api/fiidiiTradeReact")
        data = r.json()

        result = {"fetched_at": datetime.now().strftime("%H:%M:%S"), "source": "NSE India"}

        for entry in data:
            category = str(entry.get("category", "")).upper()
            if "FII" in category or "FPI" in category:
                result["fii_cash_buy"]   = float(entry.get("buyValue", 0))
                result["fii_cash_sell"]  = float(entry.get("sellValue", 0))
                result["fii_cash_net"]   = float(entry.get("netValue", 0))
            elif "DII" in category:
                result["dii_cash_buy"]   = float(entry.get("buyValue", 0))
                result["dii_cash_sell"]  = float(entry.get("sellValue", 0))
                result["dii_cash_net"]   = float(entry.get("netValue", 0))

        # Interpretation
        fii_net = result.get("fii_cash_net", 0)
        dii_net = result.get("dii_cash_net", 0)

        if fii_net > 500:        result["fii_signal"] = "STRONG_BUYING"
        elif fii_net > 0:        result["fii_signal"] = "BUYING"
        elif fii_net > -500:     result["fii_signal"] = "SELLING"
        else:                    result["fii_signal"] = "HEAVY_SELLING"

        if dii_net > 0:          result["dii_signal"] = "BUYING"
        else:                    result["dii_signal"] = "SELLING"

        # Combined
        combined = fii_net + dii_net
        result["combined_net"]    = round(combined, 2)
        result["market_signal"]   = (
            "BULLISH" if combined > 0 else "BEARISH"
        )
        return result

    except Exception as exc:
        log.warning("FII/DII fetch failed: %s", exc)
        return {
            "error": str(exc),
            "fetched_at": datetime.now().strftime("%H:%M:%S"),
            "source": "NSE India (failed)",
        }


# ── India VIX history ─────────────────────────────────────────────────────────

def get_india_vix_history(days: int = 5) -> dict:
    """
    VIX trend over last N days.
    Rising VIX = fear increasing = sell options less safe, buy options more valuable.
    """
    try:
        import yfinance as yf
        t    = yf.Ticker("^INDIAVIX")
        hist = t.history(period=f"{days + 2}d", interval="1d", auto_adjust=True)
        hist = hist.dropna()

        if hist.empty or len(hist) < 2:
            return {"error": "Insufficient VIX history", "current": 0.0}

        closes   = hist["Close"].tolist()[-days:]
        current  = round(float(closes[-1]), 2)
        prev     = round(float(closes[-2]), 2)
        avg5     = round(sum(closes) / len(closes), 2)
        trend    = "RISING" if closes[-1] > closes[0] else "FALLING"
        change   = round(current - prev, 2)

        # Regime
        if   current < 13:  regime = "LOW_FEAR"
        elif current < 18:  regime = "NORMAL"
        elif current < 25:  regime = "ELEVATED"
        else:               regime = "EXTREME_FEAR"

        return {
            "current":     current,
            "prev_close":  prev,
            "change":      change,
            "avg_5d":      avg5,
            "trend":       trend,
            "regime":      regime,
            "history":     [round(c, 2) for c in closes],
            "strategy_note": {
                "LOW_FEAR":    "VIX low → sell options (strangles/condors). Premiums underpriced for buyers.",
                "NORMAL":      "VIX normal → directional trades viable. Balance buy/sell.",
                "ELEVATED":    "VIX elevated → buy options only. Sellers exposed to big moves.",
                "EXTREME_FEAR": "VIX extreme → no new trades. Close positions.",
            }.get(regime, ""),
            "fetched_at":  datetime.now().strftime("%H:%M:%S"),
        }

    except Exception as exc:
        log.warning("VIX history failed: %s", exc)
        return {"error": str(exc), "current": 0.0}


# ── Global indices ─────────────────────────────────────────────────────────────

def get_global_indices() -> dict:
    """
    Snapshot of major global indices.
    Global cues directly influence NIFTY opening gap.
    """
    tickers = {
        "sp500":       "^GSPC",
        "nasdaq":      "^IXIC",
        "dow":         "^DJI",
        "hangseng":    "^HSI",
        "nikkei":      "^N225",
        "ftse":        "^FTSE",
        "dax":         "^GDAXI",
        "us10y_yield": "^TNX",
        "dollar_index": "DX-Y.NYB",
        "crude_oil":   "CL=F",
        "gold":        "GC=F",
    }

    result: dict = {"fetched_at": datetime.now().strftime("%H:%M:%S")}
    cues: list[str] = []

    try:
        import yfinance as yf
        for key, ticker in tickers.items():
            try:
                t     = yf.Ticker(ticker)
                info  = t.fast_info
                price = float(info.last_price or 0)
                prev  = float(info.previous_close or price)
                chng  = round((price - prev) / prev * 100, 2) if prev else 0
                result[key] = {"price": round(price, 2), "change_pct": chng}

                # Generate cues for trading context
                if key in ("sp500", "nasdaq") and abs(chng) > 0.5:
                    dir_  = "up" if chng > 0 else "down"
                    cues.append(f"US markets {dir_} {abs(chng):.1f}% → NIFTY gap {dir_} likely")
                if key == "crude_oil" and abs(chng) > 1.5:
                    dir_  = "higher" if chng > 0 else "lower"
                    cues.append(f"Crude oil {dir_} {abs(chng):.1f}% → inflation/current-account pressure")
                if key == "dollar_index" and abs(chng) > 0.3:
                    dir_  = "stronger" if chng > 0 else "weaker"
                    cues.append(f"Dollar {dir_} → FII flows {'out of' if chng > 0 else 'into'} India likely")
            except Exception:
                result[key] = {"price": 0, "change_pct": 0}

    except ImportError:
        return {"error": "yfinance not installed"}

    result["trading_cues"] = cues
    result["global_bias"]  = (
        "BULLISH" if sum(
            1 for k in ("sp500","nasdaq","nikkei","hangseng")
            if result.get(k, {}).get("change_pct", 0) > 0
        ) >= 3 else
        "BEARISH" if sum(
            1 for k in ("sp500","nasdaq","nikkei","hangseng")
            if result.get(k, {}).get("change_pct", 0) < 0
        ) >= 3 else "MIXED"
    )
    return result


# ── NIFTY intraday technicals ──────────────────────────────────────────────────

def get_nifty_technicals() -> dict:
    """
    EMA9, EMA21, VWAP, intraday trend, day high/low from Yahoo 5-min candles.
    """
    try:
        import yfinance as yf
        import pandas as pd

        t    = yf.Ticker("^NSEI")
        hist = t.history(period="2d", interval="5m", auto_adjust=True)
        hist = hist.dropna()

        if hist.empty or len(hist) < 20:
            return {"error": "Insufficient intraday data"}

        # Today only
        today = datetime.now(IST).date()
        hist.index = hist.index.tz_convert(IST)
        today_df = hist[hist.index.date == today]

        if today_df.empty:
            today_df = hist.tail(40)

        close  = today_df["Close"]
        high   = today_df["High"]
        low    = today_df["Low"]
        vol    = today_df["Volume"]

        ema9   = round(float(close.ewm(span=9,  adjust=False).mean().iloc[-1]), 2)
        ema21  = round(float(close.ewm(span=21, adjust=False).mean().iloc[-1]), 2)
        ltp    = round(float(close.iloc[-1]), 2)
        day_h  = round(float(high.max()), 2)
        day_l  = round(float(low.min()), 2)

        # VWAP = sum(typical_price × volume) / sum(volume)
        typical = (high + low + close) / 3
        vwap    = round(float((typical * vol).sum() / vol.sum()), 2) if vol.sum() > 0 else ltp

        # Trend
        if   ltp > ema9 > ema21 and ltp > vwap:  trend = "STRONG_UPTREND"
        elif ltp > ema9 and ltp > vwap:           trend = "UPTREND"
        elif ltp < ema9 < ema21 and ltp < vwap:  trend = "STRONG_DOWNTREND"
        elif ltp < ema9 and ltp < vwap:           trend = "DOWNTREND"
        else:                                      trend = "SIDEWAYS"

        # Trade bias from technicals alone
        if   "UPTREND"   in trend: tech_bias = "BULLISH"
        elif "DOWNTREND" in trend: tech_bias = "BEARISH"
        else:                       tech_bias = "NEUTRAL"

        return {
            "ltp":       ltp,
            "ema9":      ema9,
            "ema21":     ema21,
            "vwap":      vwap,
            "day_high":  day_h,
            "day_low":   day_l,
            "trend":     trend,
            "tech_bias": tech_bias,
            "candles":   len(today_df),
            "trade_note": {
                "STRONG_UPTREND":   "Price>EMA9>EMA21 and above VWAP. Strong bullish. Buy CE at VWAP retest.",
                "UPTREND":          "Price above EMA9 and VWAP. Bullish bias. Buy CE on dips.",
                "STRONG_DOWNTREND": "Price<EMA9<EMA21 and below VWAP. Strong bearish. Buy PE at VWAP retest.",
                "DOWNTREND":        "Price below EMA9 and VWAP. Bearish bias. Buy PE on bounces.",
                "SIDEWAYS":         "Mixed signals. Prefer range strategies (sell OTM strangle).",
            }.get(trend, ""),
            "fetched_at": datetime.now().strftime("%H:%M:%S"),
        }

    except Exception as exc:
        log.warning("NIFTY technicals failed: %s", exc)
        return {"error": str(exc)}


# ── Pre-market data ────────────────────────────────────────────────────────────

def get_premarket_data() -> dict:
    """
    Pre-market context: GIFT Nifty gap + SGX + global overnight moves.
    Best called between 08:00–09:15.
    """
    try:
        import yfinance as yf

        nifty = yf.Ticker("^NSEI")
        info  = nifty.fast_info

        spot      = float(info.last_price      or 0)
        prev_close = float(info.previous_close or spot)

        gap        = round(spot - prev_close, 2)
        gap_pct    = round(gap / prev_close * 100, 2) if prev_close else 0
        gap_dir    = "GAP_UP" if gap > 0 else "GAP_DOWN" if gap < 0 else "FLAT"

        # Expected open range (±0.3% around gap estimate)
        open_low   = round(spot * 0.997, 0)
        open_high  = round(spot * 1.003, 0)

        return {
            "prev_close":  round(prev_close, 2),
            "gift_nifty":  round(spot, 2),
            "gap_points":  gap,
            "gap_pct":     gap_pct,
            "gap_direction": gap_dir,
            "expected_open_range": f"{open_low:.0f}–{open_high:.0f}",
            "trade_note": (
                f"Expected {gap_dir} of {abs(gap_pct):.2f}% ({abs(gap):.0f} pts). "
                + ("Watch for gap fill OR continuation past first 5-min candle." if abs(gap_pct) > 0.3
                   else "Flat open — wait for 09:30 before taking direction.")
            ),
            "fetched_at": datetime.now().strftime("%H:%M:%S"),
        }

    except Exception as exc:
        log.warning("Pre-market data failed: %s", exc)
        return {"error": str(exc)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import json
    print("\n=== NIFTY Technicals ===")
    print(json.dumps(get_nifty_technicals(), indent=2))
    print("\n=== Global Indices ===")
    g = get_global_indices()
    for cue in g.get("trading_cues", []):
        print(" •", cue)
    print("Global bias:", g.get("global_bias"))
