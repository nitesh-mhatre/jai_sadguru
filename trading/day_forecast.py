"""
trading/day_forecast.py
-----------------------
AI-powered intraday forecast for NIFTY.

Produces:
  1. 30-minute candle trend table for remaining session
     (expected direction, key level, bias, action for each slot)

  2. Next immediate move prediction
     (what happens in next 15-30 min based on current setup)

  3. Day summary
     (overall range estimate, max pain target, key pivot levels)

Data fed to AI (pre-fetched, zero AI tool calls needed):
  - Live option chain: spot, ATM, PCR, OI walls, fresh writing, IV skew
  - VIX level + 3-day trend
  - NIFTY technicals: EMA9, EMA21, VWAP, intraday candles
  - Global indices: S&P500, Nikkei, crude, DXY
  - FII/DII net flows
  - Max pain level
  - /next predictor score (12-signal weighted score already computed)
  - Market time context

AI is told:
  - Return ONLY structured JSON (no prose)
  - Produce a 30-min slot table from current time to 15:30
  - Each slot: time, direction, expected_range, key_level, bias, action_note
  - Next move: what happens in next 15-30 min
  - Day range: estimated low / high for today
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pytz

log = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


# ── Forecast result dataclasses ───────────────────────────────────────────────

@dataclass
class CandleSlot:
    time:           str    # "09:45" → "10:15"
    direction:      str    # BULLISH | BEARISH | SIDEWAYS | VOLATILE
    expected_high:  float  # estimated upper bound for this candle
    expected_low:   float  # estimated lower bound
    key_level:      str    # "24300 CE wall" | "MaxPain 24200" | "VWAP 24150"
    bias:           str    # SHORT | LONG | NEUTRAL | AVOID
    action_note:    str    # "Hold CE if above 24200" | "Book if 24350 breaks"


@dataclass
class DayForecast:
    generated_at:    str
    spot:            float
    session_phase:   str
    next_move:       str       # immediate next 15–30 min prediction
    next_move_level: str       # key level for next move
    next_move_trade: str       # suggested action for next move
    day_high_est:    float     # estimated day high
    day_low_est:     float     # estimated day low
    day_bias:        str       # overall day bias
    key_pivots:      list[str] # ordered list of key price levels
    candle_slots:    list[CandleSlot] = field(default_factory=list)
    confidence:      str = "MEDIUM"
    ai_notes:        str = ""  # any additional AI commentary


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_forecast_prompt(
    market_data: dict,
    prediction_score: float,
    prediction_regime: str,
) -> str:
    """Build the compact AI prompt with all pre-fetched data."""

    oc   = market_data.get("oc", {})
    vix  = market_data.get("vix", {})
    tech = market_data.get("tech", {})
    fii  = market_data.get("fii", {})
    gd   = market_data.get("global", {})

    now_ist  = datetime.now(IST)
    now_str  = now_ist.strftime("%H:%M")

    # Build 30-min slots from now to 15:30
    slots = _generate_time_slots(now_ist)
    slots_str = ", ".join(f'"{s}"' for s in slots)

    # Compact market brief
    data_block = f"""
MARKET DATA (live, fetched now):
  spot={oc.get('spot',0):.2f}  ATM={oc.get('atm',0)}  expiry={oc.get('expiry','N/A')}
  PCR={oc.get('pcr',0)}  MaxPain={oc.get('max_pain',0)}
  CE_wall={oc.get('ce_wall',0)}  PE_wall={oc.get('pe_wall',0)}
  fresh_CE_OI={oc.get('fresh_ce',0):,}  fresh_PE_OI={oc.get('fresh_pe',0):,}
  ATM_CE_LTP=₹{oc.get('atm_ce_ltp',0):.1f}  ATM_PE_LTP=₹{oc.get('atm_pe_ltp',0):.1f}
  CE_IV={oc.get('avg_ce_iv',0):.1f}%  PE_IV={oc.get('avg_pe_iv',0):.1f}%

VIX: current={vix.get('current',0)}  trend={vix.get('trend','?')}  regime={vix.get('regime','?')}

TECHNICALS:
  trend={tech.get('trend','?')}  EMA9={tech.get('ema9',0):.2f}  EMA21={tech.get('ema21',0):.2f}
  VWAP={tech.get('vwap',0):.2f}  day_high={tech.get('day_high',0):.2f}  day_low={tech.get('day_low',0):.2f}

FII/DII: fii_net=₹{fii.get('fii_cash_net',0):,.0f}Cr  dii_net=₹{fii.get('dii_cash_net',0):,.0f}Cr  signal={fii.get('market_signal','?')}

GLOBAL:
  S&P500={gd.get('sp500',{}).get('change_pct',0):+.1f}%
  Nikkei={gd.get('nikkei',{}).get('change_pct',0):+.1f}%
  Crude={gd.get('crude_oil',{}).get('change_pct',0):+.1f}%
  DXY={gd.get('dollar_index',{}).get('change_pct',0):+.1f}%
  global_bias={gd.get('global_bias','?')}

CALCULATION ENGINE: score={prediction_score:.1f}/100  regime={prediction_regime}
CURRENT TIME (IST): {now_str}
"""

    prompt = f"""You are Jai Sadguru — a 20+ year NIFTY F&O expert.

{data_block}

TASK: Produce an intraday forecast for today using the data above.

Generate:
1. A 30-minute candle forecast table from {now_str} to 15:30
2. Next immediate move prediction (next 15-30 min)
3. Day range estimate (high/low for today)

TIME SLOTS TO FORECAST: [{slots_str}]

RULES:
- Base every prediction on the DATA ABOVE only
- expected_high and expected_low must be realistic NIFTY price levels (e.g. 24150, 24200)
- direction must be one of: BULLISH, BEARISH, SIDEWAYS, VOLATILE
- bias must be one of: LONG, SHORT, NEUTRAL, AVOID
- key_level must reference a specific price (OI wall, VWAP, EMA, MaxPain, etc.)
- action_note max 60 chars — what a trader should DO in that slot
- next_move_trade must be a specific actionable suggestion (e.g. "BUY 24200CE if spot holds 24150")
- day_high_est and day_low_est: estimate today's full range
- confidence: HIGH if 4+ signals agree, MEDIUM if 3, LOW if mixed

RESPOND WITH ONLY THIS JSON (no text before or after, no markdown prose):
```json
{{
  "day_bias": "BULLISH|BEARISH|SIDEWAYS",
  "confidence": "HIGH|MEDIUM|LOW",
  "day_high_est": 24400.0,
  "day_low_est": 24050.0,
  "key_pivots": ["24100 PE wall (support)", "24300 CE wall (resistance)", "24200 MaxPain", "24180 VWAP"],
  "next_move": "Brief description of what happens in next 15-30 min",
  "next_move_level": "24200",
  "next_move_trade": "Specific trade action",
  "ai_notes": "One sentence overall observation",
  "candle_slots": [
    {{
      "time": "09:45–10:15",
      "direction": "BULLISH",
      "expected_high": 24250.0,
      "expected_low": 24150.0,
      "key_level": "24200 MaxPain",
      "bias": "LONG",
      "action_note": "Buy CE if sustains above VWAP 24180"
    }}
  ]
}}
```"""

    return prompt


def _generate_time_slots(now_ist: datetime) -> list[str]:
    """Generate 30-min slot strings from current time to 15:30."""
    from datetime import time as _t

    # Round up to next 30-min boundary
    minute   = now_ist.minute
    if minute < 30:
        slot_start = now_ist.replace(minute=15, second=0, microsecond=0)
        if now_ist.time() >= _t(9, 30):
            slot_start = now_ist.replace(
                minute=30 if minute >= 0 else 0, second=0, microsecond=0
            )
    else:
        slot_start = now_ist.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    # Always start at or after 09:15
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    if slot_start < market_open:
        slot_start = market_open

    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)

    slots  = []
    cursor = slot_start
    while cursor < market_close:
        end    = cursor + timedelta(minutes=30)
        end    = min(end, market_close)
        slots.append(f"{cursor.strftime('%H:%M')}–{end.strftime('%H:%M')}")
        cursor = end
        if len(slots) >= 12:  # cap at 12 slots (6 hours)
            break

    return slots if slots else ["09:15–09:45", "09:45–10:15", "10:15–10:45"]


# ── AI call ───────────────────────────────────────────────────────────────────

def generate_day_forecast(
    agent,
    market_data: dict,
    prediction_score: float,
    prediction_regime: str,
    kronos_forecast=None,
) -> DayForecast:
    """
    Call AI with pre-fetched market data.
    Uses FORECAST_SYSTEM_PROMPT for the candle prediction JSON format.
    Uses SYSTEM_PROMPT personality for the day bias + next move analysis.
    Zero tool calls — data is already in the prompt.
    """
    from agent import Session, FinalAnswerEvent, ErrorEvent
    from trading.market_regime import REGIME_TRADING_RULES
    from config import TRADING_SYSTEM_ADDENDUM, FORECAST_SYSTEM_PROMPT

    prompt   = _build_forecast_prompt(market_data, prediction_score, prediction_regime)
    if kronos_forecast is not None:
        prompt += (
            "\n\n" + kronos_forecast.to_prompt_block()
            + "\nUse the KRONOS FORECAST block above as one quantitative input "
              "when building the candle table — cross-check it against OI/PCR/VIX."
        )
    temp     = Session()
    raw_text = ""

    # Use FORECAST_SYSTEM_PROMPT as suffix so AI knows to return structured JSON
    # while still having the Jai Sadguru trader personality from SYSTEM_PROMPT
    suffix = (
        TRADING_SYSTEM_ADDENDUM
        + REGIME_TRADING_RULES
        + "\n\n"
        + FORECAST_SYSTEM_PROMPT
    )

    try:
        for event in agent.run(prompt, temp, system_suffix=suffix):
            if isinstance(event, FinalAnswerEvent):
                raw_text = event.text
                break
    except Exception as exc:
        log.error("Day forecast AI call failed: %s", exc)
        raise RuntimeError(f"AI call failed: {exc}")

    return _parse_forecast(raw_text, market_data)


# ── Response parser ───────────────────────────────────────────────────────────

def _parse_forecast(text: str, market_data: dict) -> DayForecast:
    """Extract JSON from AI response and build DayForecast."""
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if not match:
        match = re.search(r'(\{"day_bias".*?\})\s*$', text, re.DOTALL)
    if not match:
        match = re.search(r'(\{.*"candle_slots".*\})', text, re.DOTALL)

    if not match:
        log.warning("No JSON found in forecast response")
        return _fallback_forecast(market_data)

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        log.warning("Forecast JSON parse error: %s", exc)
        return _fallback_forecast(market_data)

    # Build candle slots
    slots = []
    for slot in data.get("candle_slots", []):
        try:
            slots.append(CandleSlot(
                time          = str(slot.get("time", "?")),
                direction     = str(slot.get("direction", "SIDEWAYS")),
                expected_high = float(slot.get("expected_high", 0)),
                expected_low  = float(slot.get("expected_low", 0)),
                key_level     = str(slot.get("key_level", "")),
                bias          = str(slot.get("bias", "NEUTRAL")),
                action_note   = str(slot.get("action_note", ""))[:70],
            ))
        except Exception:
            continue

    oc = market_data.get("oc", {})
    spot = oc.get("spot", 0)

    return DayForecast(
        generated_at    = datetime.now(IST).strftime("%H:%M:%S IST"),
        spot            = spot,
        session_phase   = _get_session_phase(),
        next_move       = str(data.get("next_move", "Insufficient data")),
        next_move_level = str(data.get("next_move_level", "")),
        next_move_trade = str(data.get("next_move_trade", "")),
        day_high_est    = float(data.get("day_high_est", spot * 1.01)),
        day_low_est     = float(data.get("day_low_est",  spot * 0.99)),
        day_bias        = str(data.get("day_bias", "SIDEWAYS")),
        key_pivots      = list(data.get("key_pivots", [])),
        candle_slots    = slots,
        confidence      = str(data.get("confidence", "MEDIUM")),
        ai_notes        = str(data.get("ai_notes", ""))[:200],
    )


def _fallback_forecast(market_data: dict) -> DayForecast:
    oc   = market_data.get("oc", {})
    spot = oc.get("spot", 0)
    return DayForecast(
        generated_at    = datetime.now(IST).strftime("%H:%M:%S IST"),
        spot            = spot,
        session_phase   = _get_session_phase(),
        next_move       = "AI response could not be parsed. Use /next for calculation-based prediction.",
        next_move_level = str(oc.get("max_pain", "")),
        next_move_trade = "Run /next for trade setup",
        day_high_est    = spot * 1.008,
        day_low_est     = spot * 0.992,
        day_bias        = "UNKNOWN",
        key_pivots      = [],
        candle_slots    = [],
        confidence      = "LOW",
        ai_notes        = "Fallback — AI parsing failed",
    )


def _get_session_phase() -> str:
    from trading.market_time import get_market_status
    try:
        ms = get_market_status()
        return ms.phase
    except Exception:
        return "UNKNOWN"


# ── Data fetcher (reuses next_predictor fetch functions) ──────────────────────

def fetch_forecast_data() -> dict:
    """Fetch all data needed for forecast in parallel. Returns market_data dict."""
    import concurrent.futures
    from trading.next_predictor import _fetch_option_chain, _fetch_vix_tech_fii, _fetch_global

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_oc  = pool.submit(_fetch_option_chain)
        f_vtf = pool.submit(_fetch_vix_tech_fii)
        f_g   = pool.submit(_fetch_global)
        oc    = f_oc.result(timeout=25)
        vtf   = f_vtf.result(timeout=25)
        gd    = f_g.result(timeout=25)

    return {
        "oc":     oc,
        "vix":    vtf.get("vix", {}),
        "tech":   vtf.get("tech", {}),
        "fii":    vtf.get("fii", {}),
        "global": gd,
    }
