"""
trading/market_regime.py
-------------------------
Market regime classifier and sideways-specific strategy selector.

REGIME DETECTION (4 independent signals, need 3+ to classify):
  Signal 1 — PCR range     : MILDLY_BULLISH or MILDLY_BEARISH (not strongly directional)
  Signal 2 — OI walls tight: strong CE wall and PE wall within 200pts of each other
  Signal 3 — VIX low/flat  : VIX < 16 and not rising sharply (calm market)
  Signal 4 — Technicals    : EMA9 and EMA21 close together (< 0.3% apart), price near VWAP

REGIMES:
  STRONGLY_DIRECTIONAL  → 4-5 signals aligned one way  → buy ATM option
  MILDLY_DIRECTIONAL    → 3 signals aligned             → buy OTM option, tight SL
  SIDEWAYS              → 3+ sideways signals           → sell premium (strangle/iron condor)
  VOLATILE              → VIX > 20 + large candles      → buy options, wide SL
  UNCLEAR               → conflicting signals           → NO_ACTION

SIDEWAYS STRATEGIES:
  Short Strangle   : sell OTM CE + sell OTM PE (both 1-2 strikes OTM)
  Iron Condor      : sell OTM CE + buy further OTM CE hedge
                     sell OTM PE + buy further OTM PE hedge
  Short Straddle   : sell ATM CE + sell ATM PE (higher risk, higher premium)
  Calendar Spread  : sell near expiry, buy far expiry at same strike (time decay play)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

NIFTY_LOT_SIZE = 75


# ── Regime dataclass ───────────────────────────────────────────────────────────

@dataclass
class MarketRegime:
    regime:           str        # SIDEWAYS | MILDLY_DIRECTIONAL | STRONGLY_DIRECTIONAL | VOLATILE | UNCLEAR
    confidence:       int        # 0–4, how many signals agree
    direction:        str        # BULLISH | BEARISH | NEUTRAL
    range_low:        float      # lower bound of detected range (PE OI wall)
    range_high:       float      # upper bound (CE OI wall)
    range_width_pts:  float      # range_high - range_low
    vix:              float
    pcr:              float
    tech_trend:       str        # from get_nifty_technicals
    signals:          list[str] = field(default_factory=list)  # which signals fired
    strategy:         str = ""   # recommended strategy
    strategy_detail:  str = ""   # exact strikes, actions, rationale
    no_trade_reason:  str = ""   # populated if regime = UNCLEAR

    @property
    def is_sideways(self) -> bool:
        return self.regime == "SIDEWAYS"

    @property
    def is_directional(self) -> bool:
        return "DIRECTIONAL" in self.regime

    @property
    def is_volatile(self) -> bool:
        return self.regime == "VOLATILE"

    def prompt_block(self) -> str:
        """Compact text block injected into every AI decision prompt."""
        lines = [
            f"═══ MARKET REGIME ═══",
            f"Regime     : {self.regime}  (confidence {self.confidence}/4)",
            f"Direction  : {self.direction}",
            f"Range      : {self.range_low:.0f} – {self.range_high:.0f}  ({self.range_width_pts:.0f} pts)",
            f"VIX        : {self.vix}",
            f"PCR        : {self.pcr}",
            f"Tech trend : {self.tech_trend}",
        ]
        if self.signals:
            lines.append(f"Signals    : {', '.join(self.signals)}")
        if self.strategy:
            lines.append(f"Strategy   : {self.strategy}")
        if self.strategy_detail:
            lines.append(f"Detail     : {self.strategy_detail}")
        if self.no_trade_reason:
            lines.append(f"No trade   : {self.no_trade_reason}")
        lines.append("═════════════════════")
        return "\n".join(lines)


# ── Regime detector ────────────────────────────────────────────────────────────

def detect_regime(brief, vix: float, tech: dict) -> MarketRegime:
    """
    Classify current market regime from MarketBrief + VIX + technicals.

    brief : MarketBrief from scanner
    vix   : India VIX current value
    tech  : dict from get_nifty_technicals()
    """
    signals: list[str] = []
    sideways_signals = 0
    directional_signals = 0

    pcr       = brief.pcr
    sentiment = brief.sentiment
    spot      = brief.spot

    # ── Signal 1: PCR range ────────────────────────────────────────────────────
    if sentiment in ("MILDLY_BULLISH", "MILDLY_BEARISH"):
        signals.append("PCR_RANGE_BOUND")
        sideways_signals += 1
    elif sentiment in ("BULLISH", "STRONGLY_BULLISH"):
        signals.append("PCR_BULLISH")
        directional_signals += 1
    elif sentiment in ("BEARISH", "STRONGLY_BEARISH"):
        signals.append("PCR_BEARISH")
        directional_signals += 1

    # ── Signal 2: OI walls tight ───────────────────────────────────────────────
    range_high = brief.resistance[0]["strike"] if brief.resistance else spot * 1.01
    range_low  = brief.support[0]["strike"]    if brief.support    else spot * 0.99
    range_width = range_high - range_low

    if range_width <= 250:
        signals.append(f"TIGHT_OI_RANGE_{range_low:.0f}–{range_high:.0f}")
        sideways_signals += 1
    elif range_width <= 400:
        signals.append(f"MODERATE_RANGE_{range_low:.0f}–{range_high:.0f}")
        # counts as mild directional weakness
    else:
        signals.append(f"WIDE_RANGE_{range_width:.0f}pts")
        directional_signals += 1   # wide range = breakout possible

    # ── Signal 3: VIX low/flat ─────────────────────────────────────────────────
    if vix <= 0:
        pass   # unknown — no signal
    elif vix < 14:
        signals.append(f"VIX_LOW_{vix}")
        sideways_signals += 1
    elif vix < 18:
        signals.append(f"VIX_NORMAL_{vix}")
        # neutral — could go either way
    elif vix < 25:
        signals.append(f"VIX_ELEVATED_{vix}")
        directional_signals += 1   # elevated VIX = expect a move
    else:
        signals.append(f"VIX_EXTREME_{vix}")
        directional_signals += 2   # extreme vol = big move expected

    # ── Signal 4: Technicals ──────────────────────────────────────────────────
    tech_trend = tech.get("trend", "UNKNOWN") if tech else "UNKNOWN"
    ema9  = tech.get("ema9",  spot) if tech else spot
    ema21 = tech.get("ema21", spot) if tech else spot
    vwap  = tech.get("vwap",  spot) if tech else spot

    ema_diff_pct = abs(ema9 - ema21) / ema21 * 100 if ema21 else 0
    price_vs_vwap = abs(spot - vwap) / vwap * 100 if vwap else 0

    if tech_trend == "SIDEWAYS" or (ema_diff_pct < 0.3 and price_vs_vwap < 0.2):
        signals.append("TECH_SIDEWAYS")
        sideways_signals += 1
    elif "UPTREND" in tech_trend:
        signals.append(f"TECH_{tech_trend}")
        directional_signals += 1
    elif "DOWNTREND" in tech_trend:
        signals.append(f"TECH_{tech_trend}")
        directional_signals += 1

    # ── Classify regime ────────────────────────────────────────────────────────
    total_signals = sideways_signals + directional_signals

    if vix > 25:
        regime     = "VOLATILE"
        confidence = min(4, directional_signals)
    elif sideways_signals >= 3:
        regime     = "SIDEWAYS"
        confidence = sideways_signals
    elif directional_signals >= 3:
        if directional_signals >= 4:
            regime = "STRONGLY_DIRECTIONAL"
        else:
            regime = "MILDLY_DIRECTIONAL"
        confidence = directional_signals
    else:
        regime     = "UNCLEAR"
        confidence = 0

    # ── Direction ──────────────────────────────────────────────────────────────
    if sentiment in ("STRONGLY_BULLISH", "BULLISH", "MILDLY_BULLISH"):
        direction = "BULLISH"
    elif sentiment in ("STRONGLY_BEARISH", "BEARISH", "MILDLY_BEARISH"):
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"

    # ── Strategy ───────────────────────────────────────────────────────────────
    strategy, strategy_detail, no_trade_reason = _pick_strategy(
        regime, direction, spot, range_low, range_high, range_width, vix, brief
    )

    return MarketRegime(
        regime           = regime,
        confidence       = confidence,
        direction        = direction,
        range_low        = range_low,
        range_high       = range_high,
        range_width_pts  = range_width,
        vix              = vix,
        pcr              = pcr,
        tech_trend       = tech_trend,
        signals          = signals,
        strategy         = strategy,
        strategy_detail  = strategy_detail,
        no_trade_reason  = no_trade_reason,
    )


# ── Strategy picker ────────────────────────────────────────────────────────────

def _pick_strategy(
    regime: str, direction: str, spot: float,
    range_low: float, range_high: float, range_width: float,
    vix: float, brief
) -> tuple[str, str, str]:
    """Returns (strategy_name, detail_str, no_trade_reason)."""

    # ── VOLATILE ──────────────────────────────────────────────────────────────
    if regime == "VOLATILE":
        if direction == "BULLISH":
            return (
                "BUY_ATM_CE",
                f"VIX {vix} high + bullish direction → Buy ATM CE. Wide SL 40% of premium. "
                f"Target = first CE OI wall {range_high:.0f}.",
                "",
            )
        elif direction == "BEARISH":
            return (
                "BUY_ATM_PE",
                f"VIX {vix} high + bearish direction → Buy ATM PE. Wide SL 40% of premium. "
                f"Target = first PE OI wall {range_low:.0f}.",
                "",
            )
        else:
            return (
                "BUY_STRANGLE",
                f"VIX {vix} extreme + no clear direction → Buy OTM strangle: "
                f"buy CE above range + buy PE below range. Profit if large move either way.",
                "",
            )

    # ── SIDEWAYS ──────────────────────────────────────────────────────────────
    if regime == "SIDEWAYS":
        if range_width <= 200:
            # Tight range → Short Strangle
            return (
                "SHORT_STRANGLE",
                f"Range {range_low:.0f}–{range_high:.0f} ({range_width:.0f}pts). "
                f"SELL OTM CE near {range_high:.0f} + SELL OTM PE near {range_low:.0f}. "
                f"Collect theta decay. SL: if spot breaks range by 30pts. "
                f"Target: 50% of premium received. Exit before expiry last 2 days.",
                "",
            )
        elif range_width <= 350:
            # Moderate range → Iron Condor
            ce_sell = range_high
            ce_buy  = range_high + 100
            pe_sell = range_low
            pe_buy  = range_low - 100
            return (
                "IRON_CONDOR",
                f"Range {range_low:.0f}–{range_high:.0f} ({range_width:.0f}pts). "
                f"IRON CONDOR: "
                f"Sell {ce_sell:.0f}CE + Buy {ce_buy:.0f}CE (hedge). "
                f"Sell {pe_sell:.0f}PE + Buy {pe_buy:.0f}PE (hedge). "
                f"Max profit if NIFTY stays inside range at expiry. "
                f"Max loss capped by hedge. SL: 2× premium received.",
                "",
            )
        else:
            # Wide sideways range → Short Straddle (higher risk)
            atm = brief.atm if brief else int(round(spot / 50) * 50)
            return (
                "SHORT_STRADDLE",
                f"Wide range {range_low:.0f}–{range_high:.0f} but sideways signals. "
                f"SELL ATM {atm}CE + SELL ATM {atm}PE. "
                f"High premium collected but high risk. "
                f"SL: if either leg doubles from entry. "
                f"Delta-hedge if spot moves > 100pts from entry.",
                "",
            )

    # ── STRONGLY_DIRECTIONAL ──────────────────────────────────────────────────
    if regime == "STRONGLY_DIRECTIONAL":
        if direction == "BULLISH":
            return (
                "BUY_ATM_CE_AGGRESSIVE",
                f"4+ signals bullish. Buy ATM CE + OTM CE for leverage. "
                f"SL below {range_low:.0f} PE wall. Target: {range_high:.0f}. "
                f"Consider 2 lots if budget allows.",
                "",
            )
        else:
            return (
                "BUY_ATM_PE_AGGRESSIVE",
                f"4+ signals bearish. Buy ATM PE + OTM PE for leverage. "
                f"SL above {range_high:.0f} CE wall. Target: {range_low:.0f}. "
                f"Consider 2 lots if budget allows.",
                "",
            )

    # ── MILDLY_DIRECTIONAL ────────────────────────────────────────────────────
    if regime == "MILDLY_DIRECTIONAL":
        if direction == "BULLISH":
            return (
                "BUY_OTM_CE",
                f"3 signals bullish but mild. Buy 1 strike OTM CE. "
                f"SL 30% of premium. Target: {range_high:.0f}. "
                f"1 lot only — confirmation pending.",
                "",
            )
        else:
            return (
                "BUY_OTM_PE",
                f"3 signals bearish but mild. Buy 1 strike OTM PE. "
                f"SL 30% of premium. Target: {range_low:.0f}. "
                f"1 lot only — confirmation pending.",
                "",
            )

    # ── UNCLEAR ───────────────────────────────────────────────────────────────
    return (
        "",
        "",
        f"Mixed signals — sideways={len([s for s in ['PCR_RANGE_BOUND','TECH_SIDEWAYS','VIX_LOW'] if 'RANGE' in regime or 'SIDE' in regime])}, "
        f"directional signals conflicting. Wait for clearer setup. "
        f"PCR={brief.pcr if brief else '?'} VIX={vix}.",
    )


# ── System prompt block for sideways-aware trading ────────────────────────────

REGIME_TRADING_RULES = """
═══ MARKET REGIME RULES ═══

SIDEWAYS MARKET (most common — often missed by traders):
  Detection: PCR 0.85–1.15 + OI walls within 250pts + VIX < 16 + EMA9≈EMA21
  NEVER buy directional options in a sideways market — theta destroys buyers.
  
  CORRECT strategies:
    SHORT STRANGLE : SELL OTM CE + SELL OTM PE (2 separate trades)
                     Entry: when spot is near midpoint of range
                     SL: if spot breaks range by 30pts
                     Target: collect 50% of premium
    
    IRON CONDOR    : Sell closer strikes + buy farther strikes as hedge
                     Capped max loss, good for range 200–400pts wide
    
    SHORT STRADDLE : SELL ATM CE + SELL ATM PE (only if VIX < 13)
                     High premium but requires active delta management

  ACTION FORMAT for selling:
    "action": "SELL"  (not BUY)
    SL for SELL = price at which you BUY BACK (when premium doubles)
    Target for SELL = 50% of entry premium (when you BUY BACK at half price)

VOLATILE MARKET (VIX > 20):
  Buy options — DO NOT sell premium. Sellers lose in high-VIX markets.
  Use wider SLs (40% of premium) and bigger targets (100%+ of premium).

DIRECTIONAL MARKET:
  Use OI walls for SL/Target, not arbitrary percentages.
  Strong signal (4+ confirms) → ATM option, 2 lots
  Mild signal (3 confirms)    → OTM option, 1 lot
  Weak (<3 confirms)          → NO_ACTION

═══════════════════════════
"""
