"""
trading/r1_scanner.py
----------------------
R1 — Behavioural Pattern Scanner & Psychological Trade Engine

NOT a calculator. A pattern recogniser.

Architecture:
  1. CandleScan     — builds multi-timeframe candle objects (3m, 15m, 30m, daily)
  2. BehaviourScan  — reads NOW (last 6 candles), opening range, phase
  3. PatternLibrary — detects: opening fake, stop hunt, liquidity sweep,
                      accumulation/distribution, time-of-day traps
  4. ScanPacket     — full data object sent to AI with zero pre-interpretation
  5. r1_analyse()   — fetches all data, builds ScanPacket, calls AI with R1 prompts

The AI receives raw behavioural data and applies the psychology logic.
The scanner provides the evidence. The AI does the reasoning.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pytz

log = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


# ── Candle dataclass ───────────────────────────────────────────────────────────

@dataclass
class Candle:
    time:    str
    open:    float
    high:    float
    low:     float
    close:   float
    volume:  int

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def wick_up(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def wick_down(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def total_range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def close_vs_range(self) -> float:
        """0.0=closed at low, 1.0=closed at high, 0.5=middle."""
        if self.total_range == 0:
            return 0.5
        return (self.close - self.low) / self.total_range

    @property
    def shape(self) -> str:
        """Classify candle shape by psychology."""
        if self.total_range == 0:
            return "DOJI"
        wr = self.wick_up / self.total_range
        wl = self.wick_down / self.total_range
        br = self.body / self.total_range
        if br > 0.7:
            return "MARUBOZU" if self.is_bullish else "BEARISH_MARUBOZU"
        if wr > 0.5 and br < 0.3:
            return "SHOOTING_STAR"      # rejection of highs — trapped buyers
        if wl > 0.5 and br < 0.3:
            return "HAMMER"             # rejection of lows — trapped sellers
        if wr > 0.3 and wl > 0.3:
            return "DOJI"               # indecision
        if wr > 0.4 and self.is_bullish:
            return "BULL_LONG_UPPER"    # buying but capped at top
        if wl > 0.4 and not self.is_bullish:
            return "BEAR_LONG_LOWER"    # selling but supported at bottom
        return "BULLISH" if self.is_bullish else "BEARISH"


# ── Opening Range ──────────────────────────────────────────────────────────────

@dataclass
class OpeningRange:
    high:             float
    low:              float
    formed_at:        str
    current_spot:     float
    spot_vs_or:       str    # "ABOVE" | "BELOW" | "INSIDE"
    or_width:         float  # high - low
    # Stop hunt signals
    spike_above_or:   bool   # did price spike above or_high then come back?
    spike_below_or:   bool   # did price spike below or_low then come back?
    upper_wick_after: float  # wick above or_high on latest candle (stop hunt)
    lower_wick_after: float  # wick below or_low on latest candle (stop hunt)


# ── NOW snapshot (last 6 candles) ─────────────────────────────────────────────

@dataclass
class NowSnapshot:
    candles:         list[Candle]   # last 6 candles, oldest first
    net_move:        float          # close of last - close of first
    net_direction:   str            # "UP" | "DOWN" | "FLAT"
    vol_now:         float          # avg volume of last 3 candles
    vol_before:      float          # avg volume of first 3 candles
    vol_trend:       str            # "GROWING" | "DYING" | "STABLE"
    wick_up_last:    float          # upper wick of most recent candle
    wick_down_last:  float          # lower wick of most recent candle
    body_last:       float          # body of most recent candle
    close_vs_range:  float          # 0.0-1.0 position of close in 6-candle range
    last_3_direction: str           # direction of last 3 candles
    first_3_direction: str          # direction of first 3 candles
    reversal_forming: bool          # last 3 oppose first 3
    stop_hunt_signal: str           # "UPPER" | "LOWER" | "NONE"


# ── Pattern detections ────────────────────────────────────────────────────────

@dataclass
class PatternDetection:
    name:        str    # e.g. "OPENING_FAKE" | "STOP_HUNT_SWEEP" | "TIME_OF_DAY_TRAP"
    confidence:  str    # "HIGH" | "MEDIUM" | "LOW"
    description: str    # human-readable explanation
    evidence:    list[str]   # specific data points that triggered this
    implication: str    # what the pattern suggests for next move
    time_window: str    # "NOW" | "TODAY" | "1_WEEK" | "1_MONTH"


# ── Historical behaviour map ───────────────────────────────────────────────────

@dataclass
class HistoricalBehaviour:
    """What the market has been doing repeatedly over last week / month."""
    # Recurring stop-hunt levels (where wicks repeatedly appear)
    stop_hunt_levels_week:  list[float]   # e.g. [23600, 24000]
    stop_hunt_levels_month: list[float]

    # Time-of-day behaviour
    tod_traps: list[str]   # e.g. ["09:45-10:15: fake breakdown, reversal up"]

    # Consecutive day patterns
    consecutive_patterns: list[str]

    # Swing structure
    swing_highs: list[float]
    swing_lows:  list[float]

    # Recent distribution / accumulation zones
    distribution_zones: list[str]
    accumulation_zones: list[str]

    # How many days out of last 5 showed the opening fake pattern
    opening_fake_frequency: int   # 0-5

    # Summary sentence
    summary: str


# ── Full scan packet ───────────────────────────────────────────────────────────

@dataclass
class ScanPacket:
    """
    Everything the R1 AI needs in one object.
    The scanner provides raw observations. The AI does the interpretation.
    """
    scanned_at:     str
    phase:          str     # opening | discovery | midday | closing

    # Core market data
    spot:           float
    vix:            float
    vix_trend:      str
    pcr:            float
    max_pain:       float
    ce_wall:        float
    pe_wall:        float
    expiry:         str

    # Behavioural reads
    now:            NowSnapshot
    opening_range:  OpeningRange
    patterns:       list[PatternDetection]
    history:        HistoricalBehaviour

    # Raw candle data (multiple timeframes)
    candles_3m:     list[Candle]    # last 20 × 3-min candles
    candles_15m:    list[Candle]    # last 20 × 15-min candles
    candles_30m:    list[Candle]    # last 10 × 30-min candles
    candles_daily:  list[Candle]    # last 20 daily candles

    # Global + institutional
    fii_net:        float
    dii_net:        float
    global_bias:    str
    global_detail:  str
    news_headlines: list[str]       # top 5 impactful headlines

    # OI fresh writing
    fresh_ce_strikes: list[dict]
    fresh_pe_strikes: list[dict]
    oi_unwinding:     list[dict]

    def to_prompt_block(self) -> str:
        """Compact text block sent to R1 AI. Every number, no interpretation."""
        c = self.now.candles
        candle_lines = "\n".join(
            f"  [{i+1}] {cl.time}  O:{cl.open:.0f} H:{cl.high:.0f} "
            f"L:{cl.low:.0f} C:{cl.close:.0f}  V:{cl.volume:,}  "
            f"shape={cl.shape}  wick↑{cl.wick_up:.0f} wick↓{cl.wick_down:.0f}"
            for i, cl in enumerate(c)
        )

        pattern_lines = "\n".join(
            f"  [{p.confidence}] {p.name}: {p.description}"
            for p in self.patterns
        )

        hist = self.history
        lines = [
            f"=== R1 SCAN  {self.scanned_at} ===",
            f"PHASE: {self.phase}",
            f"",
            f"SPOT: {self.spot:.2f}  VIX: {self.vix} ({self.vix_trend})",
            f"PCR: {self.pcr}  MaxPain: {self.max_pain:.0f}  "
            f"CE_wall: {self.ce_wall:.0f}  PE_wall: {self.pe_wall:.0f}",
            f"Expiry: {self.expiry}",
            f"",
            f"NOW (last 6 candles):",
            candle_lines,
            f"  net_move={self.now.net_move:+.0f}  direction={self.now.net_direction}",
            f"  vol_now={self.now.vol_now:,.0f}  vol_before={self.now.vol_before:,.0f}  vol_trend={self.now.vol_trend}",
            f"  wick_up_last={self.now.wick_up_last:.1f}  wick_down_last={self.now.wick_down_last:.1f}  body_last={self.now.body_last:.1f}",
            f"  close_vs_range={self.now.close_vs_range:.2f}  (0=bottom 1=top)",
            f"  first_3={self.now.first_3_direction}  last_3={self.now.last_3_direction}  reversal_forming={self.now.reversal_forming}",
            f"  stop_hunt_signal={self.now.stop_hunt_signal}",
            f"",
            f"OPENING RANGE:",
            f"  or_high={self.opening_range.high:.0f}  or_low={self.opening_range.low:.0f}  "
            f"or_width={self.opening_range.or_width:.0f}pts",
            f"  spot_vs_or={self.opening_range.spot_vs_or}  "
            f"spike_above={self.opening_range.spike_above_or}  spike_below={self.opening_range.spike_below_or}",
            f"  upper_wick_after_or={self.opening_range.upper_wick_after:.1f}  "
            f"lower_wick_after_or={self.opening_range.lower_wick_after:.1f}",
            f"",
            f"PATTERNS DETECTED:",
            pattern_lines if pattern_lines else "  None detected yet",
            f"",
            f"HISTORICAL BEHAVIOUR (last 1 week):",
            f"  stop_hunt_levels: {self.history.stop_hunt_levels_week}",
            f"  time_of_day_traps: {self.history.tod_traps}",
            f"  consecutive_patterns: {self.history.consecutive_patterns}",
            f"  opening_fake_frequency: {self.history.opening_fake_frequency}/5 days",
            f"  swing_highs: {self.history.swing_highs[-3:]}  swing_lows: {self.history.swing_lows[-3:]}",
            f"  distribution_zones: {self.history.distribution_zones}",
            f"  accumulation_zones: {self.history.accumulation_zones}",
            f"  summary: {self.history.summary}",
            f"",
            f"INSTITUTIONAL / GLOBAL:",
            f"  FII net: ₹{self.fii_net:+,.0f}Cr  DII net: ₹{self.dii_net:+,.0f}Cr",
            f"  Global: {self.global_bias}  ({self.global_detail})",
            f"  fresh_CE_writing: {self.fresh_ce_strikes[:3]}",
            f"  fresh_PE_writing: {self.fresh_pe_strikes[:3]}",
            f"  oi_unwinding: {self.oi_unwinding[:3]}",
            f"",
            f"NEWS (top headlines — read last):",
        ]
        for i, h in enumerate(self.news_headlines[:5], 1):
            lines.append(f"  [{i}] {h}")

        lines.append("=== END SCAN ===")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# SCANNER — fetches and builds all objects
# ══════════════════════════════════════════════════════════════════════════════

def _get_phase(now_ist: datetime) -> str:
    from datetime import time as _t
    t = now_ist.time()
    if   _t(9, 15)  <= t < _t(9, 45):  return "opening"
    elif _t(9, 45)  <= t < _t(10, 30): return "discovery"
    elif _t(10, 30) <= t < _t(13, 0):  return "midday"
    elif _t(13, 0)  <= t <= _t(15, 30): return "closing"
    else:                               return "after_hours"


def _parse_candles(hist_df, tz=IST) -> list[Candle]:
    """Convert yfinance history DataFrame to list of Candle."""
    if hist_df is None or hist_df.empty:
        return []
    candles = []
    for ts, row in hist_df.iterrows():
        try:
            if hasattr(ts, 'tz') and ts.tz is not None:
                ts = ts.tz_convert(tz).tz_localize(None)
            candles.append(Candle(
                time   = str(ts)[-8:-3] if len(str(ts)) > 8 else str(ts),
                open   = round(float(row.get("Open", row.get("open", 0))), 2),
                high   = round(float(row.get("High", row.get("high", 0))), 2),
                low    = round(float(row.get("Low",  row.get("low",  0))), 2),
                close  = round(float(row.get("Close",row.get("close",0))), 2),
                volume = int(row.get("Volume", row.get("volume", 0))),
            ))
        except Exception:
            continue
    return candles


def _fetch_candles_yf(ticker: str, period: str, interval: str) -> list[Candle]:
    """Fetch OHLCV from Yahoo Finance for any ticker."""
    try:
        import yfinance as yf
        t    = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, auto_adjust=True)
        return _parse_candles(hist)
    except Exception as exc:
        log.warning("Candle fetch %s %s %s: %s", ticker, period, interval, exc)
        return []


def _build_now_snapshot(candles: list[Candle]) -> NowSnapshot:
    """Build NOW snapshot from last 6 candles."""
    if len(candles) < 2:
        return NowSnapshot(
            candles=candles, net_move=0, net_direction="FLAT",
            vol_now=0, vol_before=0, vol_trend="STABLE",
            wick_up_last=0, wick_down_last=0, body_last=0,
            close_vs_range=0.5, last_3_direction="FLAT",
            first_3_direction="FLAT", reversal_forming=False,
            stop_hunt_signal="NONE",
        )

    last6 = candles[-6:] if len(candles) >= 6 else candles
    last  = last6[-1]
    first = last6[0]

    net_move = last.close - first.close
    net_dir  = "UP" if net_move > 5 else ("DOWN" if net_move < -5 else "FLAT")

    # Volume trend
    mid     = max(1, len(last6) // 2)
    vol_b   = sum(c.volume for c in last6[:mid]) / mid
    vol_n   = sum(c.volume for c in last6[mid:]) / max(1, len(last6) - mid)
    vol_trend = "GROWING" if vol_n > vol_b * 1.2 else ("DYING" if vol_n < vol_b * 0.8 else "STABLE")

    # Close vs 6-candle range
    hi6 = max(c.high  for c in last6)
    lo6 = min(c.low   for c in last6)
    cvr = (last.close - lo6) / (hi6 - lo6) if (hi6 - lo6) > 0 else 0.5

    # Direction of first 3 vs last 3
    def _dir(group):
        if not group: return "FLAT"
        net = group[-1].close - group[0].close
        return "UP" if net > 3 else ("DOWN" if net < -3 else "FLAT")

    f3 = last6[:3]
    l3 = last6[-3:]
    f3_dir = _dir(f3)
    l3_dir = _dir(l3)
    reversal = (f3_dir == "UP" and l3_dir == "DOWN") or (f3_dir == "DOWN" and l3_dir == "UP")

    # Stop hunt signal
    sh = "NONE"
    if last.wick_up > last.body * 2 and last.wick_up > 15:
        sh = "UPPER"   # upper wick = stop hunt above = bearish
    elif last.wick_down > last.body * 2 and last.wick_down > 15:
        sh = "LOWER"   # lower wick = stop hunt below = bullish

    return NowSnapshot(
        candles=last6,
        net_move=round(net_move, 1),
        net_direction=net_dir,
        vol_now=round(vol_n),
        vol_before=round(vol_b),
        vol_trend=vol_trend,
        wick_up_last=round(last.wick_up, 1),
        wick_down_last=round(last.wick_down, 1),
        body_last=round(last.body, 1),
        close_vs_range=round(cvr, 2),
        last_3_direction=l3_dir,
        first_3_direction=f3_dir,
        reversal_forming=reversal,
        stop_hunt_signal=sh,
    )


def _build_opening_range(candles_3m: list[Candle], spot: float) -> OpeningRange:
    """Build opening range from first 5 × 3-min candles (09:15-09:30)."""
    if not candles_3m:
        return OpeningRange(
            high=spot*1.005, low=spot*0.995,
            formed_at="09:30", current_spot=spot,
            spot_vs_or="INSIDE", or_width=spot*0.01,
            spike_above_or=False, spike_below_or=False,
            upper_wick_after=0, lower_wick_after=0,
        )

    # First 5 candles = opening range
    or_candles = candles_3m[:5]
    or_high = max(c.high  for c in or_candles)
    or_low  = min(c.low   for c in or_candles)

    # Subsequent candles for spike detection
    later = candles_3m[5:]
    spike_above = any(c.high > or_high and c.close < or_high for c in later)
    spike_below = any(c.low  < or_low  and c.close > or_low  for c in later)

    # Latest candle wicks vs OR
    last = candles_3m[-1]
    upper_wick = max(0, last.high - or_high) if last.high > or_high else 0
    lower_wick = max(0, or_low - last.low)   if last.low  < or_low  else 0

    if spot > or_high:       spot_vs_or = "ABOVE"
    elif spot < or_low:      spot_vs_or = "BELOW"
    else:                    spot_vs_or = "INSIDE"

    return OpeningRange(
        high=round(or_high, 2), low=round(or_low, 2),
        formed_at="09:30", current_spot=round(spot, 2),
        spot_vs_or=spot_vs_or, or_width=round(or_high - or_low, 1),
        spike_above_or=spike_above, spike_below_or=spike_below,
        upper_wick_after=round(upper_wick, 1),
        lower_wick_after=round(lower_wick, 1),
    )


def _detect_patterns(
    now: NowSnapshot, opening_range: OpeningRange,
    candles_3m: list[Candle], phase: str
) -> list[PatternDetection]:
    """Detect behavioural patterns from the data."""
    patterns = []

    # ── Opening fake ──────────────────────────────────────────────────────────
    if phase in ("opening", "discovery"):
        if (opening_range.spike_above_or or opening_range.spike_below_or) \
           and now.reversal_forming:
            direction = "DOWN" if opening_range.spike_above_or else "UP"
            patterns.append(PatternDetection(
                name="OPENING_FAKE",
                confidence="HIGH" if (
                    (opening_range.spike_above_or and now.stop_hunt_signal == "UPPER") or
                    (opening_range.spike_below_or and now.stop_hunt_signal == "LOWER")
                ) else "MEDIUM",
                description=(
                    f"Spike {'above OR_HIGH '+str(opening_range.high) if opening_range.spike_above_or else 'below OR_LOW '+str(opening_range.low)}"
                    f" then reversed. Classic retail stop hunt."
                ),
                evidence=[
                    f"spike_above_or={opening_range.spike_above_or}",
                    f"spike_below_or={opening_range.spike_below_or}",
                    f"reversal_forming={now.reversal_forming}",
                    f"stop_hunt_signal={now.stop_hunt_signal}",
                ],
                implication=f"Real move likely {direction} after sweep completes.",
                time_window="NOW",
            ))

    # ── Stop hunt sweep ───────────────────────────────────────────────────────
    if now.stop_hunt_signal != "NONE":
        implications = {
            "UPPER": "Upper wick = trapped buyers above. Expect move DOWN next.",
            "LOWER": "Lower wick = trapped sellers below. Expect move UP next.",
        }
        patterns.append(PatternDetection(
            name=f"STOP_HUNT_{now.stop_hunt_signal}",
            confidence="HIGH" if (
                now.wick_up_last > now.body_last * 3 or
                now.wick_down_last > now.body_last * 3
            ) else "MEDIUM",
            description=f"Last candle shows {now.stop_hunt_signal} wick > 2× body.",
            evidence=[
                f"wick_up_last={now.wick_up_last}",
                f"wick_down_last={now.wick_down_last}",
                f"body_last={now.body_last}",
            ],
            implication=implications[now.stop_hunt_signal],
            time_window="NOW",
        ))

    # ── Reversal with dying volume ─────────────────────────────────────────────
    if now.reversal_forming and now.vol_trend == "DYING":
        patterns.append(PatternDetection(
            name="REVERSAL_DYING_VOLUME",
            confidence="MEDIUM",
            description=(
                f"Direction changed ({now.first_3_direction}→{now.last_3_direction}) "
                f"but volume is dying ({now.vol_before:,.0f}→{now.vol_now:,.0f}). "
                "Reversal lacks conviction."
            ),
            evidence=[
                f"first_3={now.first_3_direction}  last_3={now.last_3_direction}",
                f"vol_before={now.vol_before:,.0f}  vol_now={now.vol_now:,.0f}",
                f"vol_trend=DYING",
            ],
            implication="Fake reversal. Wait for volume to confirm before entering.",
            time_window="NOW",
        ))

    # ── Clean move with growing volume ────────────────────────────────────────
    if not now.reversal_forming and now.vol_trend == "GROWING" \
       and now.net_direction != "FLAT":
        patterns.append(PatternDetection(
            name=f"MOMENTUM_{now.net_direction}",
            confidence="MEDIUM",
            description=(
                f"Consistent {now.net_direction} move ({now.net_move:+.0f}pts) "
                f"with growing volume. Momentum trade possible."
            ),
            evidence=[
                f"net_move={now.net_move}",
                f"vol_trend=GROWING ({now.vol_before:,.0f}→{now.vol_now:,.0f})",
                f"close_vs_range={now.close_vs_range:.2f}",
            ],
            implication=f"Follow direction if close_vs_range > 0.7 ({now.net_direction}=UP) "
                        f"or < 0.3 (DOWN). High close = strong buyers, low close = strong sellers.",
            time_window="NOW",
        ))

    # ── Indecision at key level ────────────────────────────────────────────────
    if 0.35 < now.close_vs_range < 0.65 and now.vol_trend != "GROWING":
        patterns.append(PatternDetection(
            name="INDECISION_AT_RANGE",
            confidence="LOW",
            description=f"close_vs_range={now.close_vs_range:.2f} — neither buyers nor sellers winning.",
            evidence=[
                f"close_vs_range={now.close_vs_range:.2f}",
                f"vol_trend={now.vol_trend}",
            ],
            implication="Wait for a decisive candle before entering.",
            time_window="NOW",
        ))

    return patterns


def _build_historical_behaviour(
    candles_30m: list[Candle],
    candles_daily: list[Candle],
    spot: float,
) -> HistoricalBehaviour:
    """Analyse historical candles for recurring patterns."""

    # ── Find stop-hunt levels (where long wicks appear repeatedly) ────────────
    def _wicked_levels(candles, threshold_wick=20):
        """Find price levels where long wicks appear. Round to nearest 50."""
        levels: dict[float, int] = {}
        for c in candles:
            if c.wick_up > threshold_wick:
                lvl = round(c.high / 50) * 50
                levels[lvl] = levels.get(lvl, 0) + 1
            if c.wick_down > threshold_wick:
                lvl = round(c.low / 50) * 50
                levels[lvl] = levels.get(lvl, 0) + 1
        # Return levels seen 2+ times
        return sorted([lvl for lvl, cnt in levels.items() if cnt >= 2], reverse=True)

    stop_hunt_w = _wicked_levels(candles_30m[-20:] if candles_30m else [], threshold_wick=25)
    stop_hunt_m = _wicked_levels(candles_daily[-20:] if candles_daily else [], threshold_wick=30)

    # ── Swing highs and lows ──────────────────────────────────────────────────
    def _swing_highs(candles, n=3):
        highs = []
        for i in range(n, len(candles) - n):
            c = candles[i]
            if all(c.high >= candles[i-j].high for j in range(1, n+1)) and \
               all(c.high >= candles[i+j].high for j in range(1, n+1)):
                highs.append(round(c.high, 0))
        return highs[-5:]

    def _swing_lows(candles, n=3):
        lows = []
        for i in range(n, len(candles) - n):
            c = candles[i]
            if all(c.low <= candles[i-j].low for j in range(1, n+1)) and \
               all(c.low <= candles[i+j].low for j in range(1, n+1)):
                lows.append(round(c.low, 0))
        return lows[-5:]

    s_highs = _swing_highs(candles_daily[-20:]) if len(candles_daily) >= 7 else []
    s_lows  = _swing_lows(candles_daily[-20:])  if len(candles_daily) >= 7 else []

    # ── Detect time-of-day traps from 30m candles ─────────────────────────────
    tod_traps = []
    if candles_30m:
        # Check if 09:30-10:00 candles frequently have long wicks
        opening_candles = [c for c in candles_30m if "09:3" in c.time or "09:4" in c.time]
        if opening_candles:
            long_wick_count = sum(
                1 for c in opening_candles
                if max(c.wick_up, c.wick_down) > c.body * 2
            )
            if long_wick_count >= 3:
                tod_traps.append(
                    f"09:30-10:00: {long_wick_count}/{len(opening_candles)} candles had "
                    f"stop-hunt wicks — opening fake pattern active this week"
                )

    # ── Consecutive day patterns ──────────────────────────────────────────────
    consecutive = []
    if candles_daily and len(candles_daily) >= 3:
        last3 = candles_daily[-3:]
        directions = ["UP" if c.is_bullish else "DOWN" for c in last3]
        if len(set(directions)) == 1:
            consecutive.append(f"3 consecutive {directions[0]} daily candles")
        # Check for reversals at same levels
        if len(candles_daily) >= 5:
            closes = [c.close for c in candles_daily[-5:]]
            if closes[-1] > closes[-2] > closes[-3] < closes[-4]:
                consecutive.append("V-shape recovery in last 3 days")
            elif closes[-1] < closes[-2] < closes[-3] > closes[-4]:
                consecutive.append("Inverted V distribution in last 3 days")

    # ── Distribution / accumulation zones ────────────────────────────────────
    distribution = []
    accumulation = []
    if candles_30m:
        # High volume candles with long upper wicks = distribution
        avg_vol = sum(c.volume for c in candles_30m) / len(candles_30m)
        for c in candles_30m[-20:]:
            if c.volume > avg_vol * 1.5:
                if c.wick_up > c.body:
                    distribution.append(f"~{round(c.high/50)*50:.0f} (high vol upper wick)")
                elif c.wick_down > c.body:
                    accumulation.append(f"~{round(c.low/50)*50:.0f} (high vol lower wick)")

    # ── Opening fake frequency ────────────────────────────────────────────────
    fake_count = 0
    if candles_daily:
        for c in candles_daily[-5:]:
            # Candle opens near middle but wicks on both sides = fake-out day
            if c.total_range > 0 and c.body / c.total_range < 0.35:
                fake_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    trend = "recovering" if (candles_daily and candles_daily[-1].close > candles_daily[-3].close) \
            else "distributing"
    summary = (
        f"Market is {trend}. "
        + (f"Stop-hunt zones at {stop_hunt_w[:2]}. " if stop_hunt_w else "")
        + (f"Swing highs: {s_highs[-2:]}. " if s_highs else "")
        + (f"Opening fake pattern seen {fake_count}/5 days last week." if fake_count >= 2 else "")
    )

    return HistoricalBehaviour(
        stop_hunt_levels_week  = stop_hunt_w[:5],
        stop_hunt_levels_month = stop_hunt_m[:5],
        tod_traps              = tod_traps,
        consecutive_patterns   = consecutive,
        swing_highs            = s_highs,
        swing_lows             = s_lows,
        distribution_zones     = list(set(distribution))[:3],
        accumulation_zones     = list(set(accumulation))[:3],
        opening_fake_frequency = fake_count,
        summary                = summary,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def build_scan_packet() -> ScanPacket:
    """
    Fetch all data and build ScanPacket for R1 AI analysis.
    Runs in < 20 seconds.
    """
    import concurrent.futures
    now_ist = datetime.now(IST)

    def _fetch_nifty_candles():
        return {
            "3m":    _fetch_candles_yf("^NSEI", "2d",  "3m"),
            "15m":   _fetch_candles_yf("^NSEI", "5d",  "15m"),
            "30m":   _fetch_candles_yf("^NSEI", "1mo", "30m"),
            "daily": _fetch_candles_yf("^NSEI", "3mo", "1d"),
        }

    def _fetch_market_data():
        result = {"spot": 0, "vix": 0, "vix_trend": "?", "pcr": 0,
                  "max_pain": 0, "ce_wall": 0, "pe_wall": 0, "expiry": "N/A",
                  "fii_net": 0, "dii_net": 0, "global_bias": "?", "global_detail": "",
                  "fresh_ce": [], "fresh_pe": [], "oi_unwind": []}
        try:
            from data.yahoo_feed import get_nifty_spot, get_india_vix
            from data.market_extra import get_india_vix_history, get_global_indices, get_fii_dii_data
            from data.nifty_option_chain import get_nifty_option_chain, get_expiry_dates
            import numpy as np

            result["spot"] = get_nifty_spot()
            vix_data       = get_india_vix_history(days=3)
            result["vix"]      = vix_data.get("current", 0)
            result["vix_trend"] = vix_data.get("trend", "?")

            dates  = get_expiry_dates()
            expiry = dates[0] if dates else None
            result["expiry"] = expiry or "N/A"

            df, spot_nse = get_nifty_option_chain(expiry=expiry)
            if result["spot"] <= 0:
                result["spot"] = spot_nse

            # Sanitize
            num = df.select_dtypes(include="number").columns
            df[num] = df[num].replace([np.inf, -np.inf], 0).fillna(0)
            df["Strike"] = df["Strike"].astype(int)

            ce_oi = int(df["CE_OI"].sum())
            pe_oi = int(df["PE_OI"].sum())
            result["pcr"] = round(pe_oi / ce_oi, 2) if ce_oi else 0

            top_ce = df.nlargest(1, "CE_OI")
            top_pe = df.nlargest(1, "PE_OI")
            result["ce_wall"] = int(top_ce["Strike"].iloc[0]) if not top_ce.empty else 0
            result["pe_wall"] = int(top_pe["Strike"].iloc[0]) if not top_pe.empty else 0

            # Max pain
            strikes = sorted(df["Strike"].unique())
            pain = {}
            for s in strikes:
                loss = (((df["Strike"]-s).clip(lower=0) * df["CE_OI"]).sum() +
                        ((s-df["Strike"]).clip(lower=0) * df["PE_OI"]).sum())
                pain[s] = float(loss)
            result["max_pain"] = float(min(pain, key=pain.get)) if pain else result["spot"]

            # Fresh writing
            fc = df[df["CE_Chng_OI"]>0].nlargest(3,"CE_Chng_OI")[["Strike","CE_Chng_OI"]]
            fp = df[df["PE_Chng_OI"]>0].nlargest(3,"PE_Chng_OI")[["Strike","PE_Chng_OI"]]
            uw = df[df["CE_Chng_OI"]<0].nsmallest(3,"CE_Chng_OI")[["Strike","CE_Chng_OI"]]
            result["fresh_ce"] = fc.to_dict("records")
            result["fresh_pe"] = fp.to_dict("records")
            result["oi_unwind"] = uw.to_dict("records")

            # Global + FII
            gd = get_global_indices()
            result["global_bias"]   = gd.get("global_bias", "?")
            result["global_detail"] = (
                f"S&P{gd.get('sp500',{}).get('change_pct',0):+.1f}% "
                f"Nikkei{gd.get('nikkei',{}).get('change_pct',0):+.1f}% "
                f"Crude{gd.get('crude_oil',{}).get('change_pct',0):+.1f}% "
                f"DXY{gd.get('dollar_index',{}).get('change_pct',0):+.1f}%"
            )
            fii = get_fii_dii_data()
            result["fii_net"] = fii.get("fii_cash_net", 0) or 0
            result["dii_net"] = fii.get("dii_cash_net", 0) or 0
        except Exception as exc:
            log.warning("Market data fetch: %s", exc)
        return result

    def _fetch_news():
        try:
            from data.news_feed import fetch_all_news
            articles = fetch_all_news(days=1)
            return [
                f"[{a.sentiment}/{a.impact}] {a.title} — {a.source}"
                for a in articles[:5]
            ]
        except Exception:
            return []

    # Parallel fetch
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_candles = pool.submit(_fetch_nifty_candles)
        f_market  = pool.submit(_fetch_market_data)
        f_news    = pool.submit(_fetch_news)
        candles   = f_candles.result(timeout=25)
        market    = f_market.result(timeout=25)
        headlines = f_news.result(timeout=15)

    c3m  = candles.get("3m",    [])
    c15m = candles.get("15m",   [])
    c30m = candles.get("30m",   [])
    cdly = candles.get("daily", [])

    spot = market["spot"]
    now  = _build_now_snapshot(c3m)
    opr  = _build_opening_range(c3m, spot)
    pats = _detect_patterns(now, opr, c3m, _get_phase(now_ist))
    hist = _build_historical_behaviour(c30m, cdly, spot)

    return ScanPacket(
        scanned_at     = now_ist.strftime("%d-%b-%Y %H:%M:%S IST"),
        phase          = _get_phase(now_ist),
        spot           = spot,
        vix            = market["vix"],
        vix_trend      = market["vix_trend"],
        pcr            = market["pcr"],
        max_pain       = market["max_pain"],
        ce_wall        = market["ce_wall"],
        pe_wall        = market["pe_wall"],
        expiry         = market["expiry"],
        now            = now,
        opening_range  = opr,
        patterns       = pats,
        history        = hist,
        candles_3m     = c3m[-20:],
        candles_15m    = c15m[-20:],
        candles_30m    = c30m[-10:],
        candles_daily  = cdly[-20:],
        fii_net        = market["fii_net"],
        dii_net        = market["dii_net"],
        global_bias    = market["global_bias"],
        global_detail  = market["global_detail"],
        news_headlines = headlines,
        fresh_ce_strikes = [{"strike": int(r.get("Strike",0)), "added_oi": int(r.get("CE_Chng_OI",0))} for r in market["fresh_ce"]],
        fresh_pe_strikes = [{"strike": int(r.get("Strike",0)), "added_oi": int(r.get("PE_Chng_OI",0))} for r in market["fresh_pe"]],
        oi_unwinding     = [{"strike": int(r.get("Strike",0)), "reduced_oi": int(r.get("CE_Chng_OI",0))} for r in market["oi_unwind"]],
    )
