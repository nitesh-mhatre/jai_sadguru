"""
trading/rules_fallback.py
-------------------------

Instant rule-based fallback — used when the LLM times out, returns 500, or is
otherwise unavailable.  Never blocks or retries the LLM; falls back immediately
to standard algorithmic logic so order submission continues without delay.

Decoupled from the LLM loop: the main trade execution path reads a stored
`action_bias` state variable that is updated asynchronously by a background
thread every 30–60 seconds (if the LLM is reachable) or recomputed instantly
from market data when the LLM is down.

Core algorithmic checks (executed in priority order):

  1. Spot-to-strike distances
       - How far is spot from each candidate strike?
       - ITM vs OTM premium estimates
       - Intrinsic value floor for any option

  2. PCR (Put/Call Ratio) direction bias
       - PCR >= 1.2 -> bullish bias
       - PCR <= 0.8 -> bearish bias
       - PCR 0.8-1.2 -> neutral / sideways -> prefer SELL strategies

  3. Delta neutrality checks
       - For paired trades (straddle/strangle): net delta near zero
       - For directional: delta aligned with bias

  4. OI wall proximity
       - Where are the nearest CE/PE resistance & support walls?
       - Use those for SL and target placement

  5. VIX regime filter
       - VIX > 25 -> NO new trades (too volatile)
       - VIX < 13 -> prefer SELL premium
       - VIX 13-25 -> directional buys OK

Returns a FallbackDecision with the same shape as an LLM action block so the
execution path is identical regardless of whether the source is LLM or rules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# --- Decision envelope (same shape as LLM action block) ---

@dataclass
class FallbackAction:
    """One rule-generated trade action — mirrors the LLM PLACE_TRADE schema."""
    type:             str = "PLACE_TRADE"
    expiry:           str = ""
    strike:           float = 0.0
    option_type:      str = "CE"
    action:           str = "BUY"
    qty:              int = 1
    entry_price:      float = 0.0
    sl:               float = 0.0
    target:           float = 0.0
    rationale:        str = ""          # rule-based reason, tagged [RULES]


@dataclass
class FallbackDecision:
    """Complete rule-based decision — mirrors LLM {"actions": [...]} shape."""
    actions:      list[FallbackAction] = field(default_factory=list)
    source:       str = "RULES_FALLBACK"   # distinguishes from "AI" or "/next"
    confidence:   float = 0.0              # 0-1; lower = weaker signal
    regime:       str = "UNKNOWN"          # SIDEWAYS | DIRECTIONAL | VOLATILE
    pcr:          float = 0.0
    vix:          float = 0.0
    spot:         float = 0.0
    timestamp:    str = ""
    no_action_reason: str = ""            # filled when actions is empty

    @property
    def as_dict(self) -> dict:
        """JSON-serialisable dict matching the LLM action-block schema."""
        return {
            "actions": [
                {
                    "type":         a.type,
                    "expiry":       a.expiry,
                    "strike":       a.strike,
                    "option_type":  a.option_type,
                    "action":       a.action,
                    "qty":          a.qty,
                    "entry_price":  a.entry_price,
                    "sl":           a.sl,
                    "target":       a.target,
                    "rationale":    a.rationale,
                }
                for a in self.actions
            ],
        }


# --- Pure helper: intrinsic + time-value premium estimate ---

def intrinsic_value(strike: float, spot: float, option_type: str) -> float:
    """Intrinsic value of an option — the floor below which premium cannot go."""
    if option_type.upper() == "CE":
        return max(0.0, spot - strike)
    if option_type.upper() == "PE":
        return max(0.0, strike - spot)
    return 0.0


def estimated_premium(
    strike: float,
    spot: float,
    option_type: str,
    minutes_to_expiry: int = 375,
    iv: float = 15.0,
) -> float:
    """
    Rough premium estimate when no live quote is available.
    Intrinsic + time value (Black-Scholes-ish approximation for ATM-ish).
    Used ONLY as a fallback — always prefer live LTP from Groww chain.
    """
    intrinsic = intrinsic_value(strike, spot, option_type)
    if intrinsic <= 0 and strike != spot:
        # OTM: time value only, decaying with sqrt of time remaining
        days = max(1, minutes_to_expiry / 1440.0)
        time_val = (spot * iv / 100.0) * (days ** 0.5) * 0.4   # rough ATM premium
        # Discount for distance from ATM
        moneyness = abs(strike - spot) / spot
        discount = max(0.05, 1.0 - moneyness * 1.5)
        return round(time_val * discount, 2)
    # ITM: intrinsic + small time value
    days = max(1, minutes_to_expiry / 1440.0)
    time_val = (spot * iv / 100.0) * (days ** 0.5) * 0.2
    return round(intrinsic + time_val, 2)


# --- Spot-to-strike analysis ---

@dataclass
class StrikeAnalysis:
    strike:        float
    option_type:   str
    spot_distance: float           # |spot - strike| in points
    moneyness:     str             # ITM | ATM | OTM
    intrinsic:     float
    est_premium:   float           # fallback estimate if no live quote

    @classmethod
    def from_spot(cls, strike: float, spot: float, option_type: str,
                  iv: float = 15.0, minutes_to_expiry: int = 375) -> "StrikeAnalysis":
        dist = abs(spot - strike)
        if option_type.upper() == "CE":
            moneyness = "ITM" if strike < spot else ("ATM" if dist < 25 else "OTM")
        else:
            moneyness = "ITM" if strike > spot else ("ATM" if dist < 25 else "OTM")
        intrinsic = intrinsic_value(strike, spot, option_type)
        est = estimated_premium(strike, spot, option_type, minutes_to_expiry, iv)
        return cls(
            strike=strike,
            option_type=option_type,
            spot_distance=dist,
            moneyness=moneyness,
            intrinsic=intrinsic,
            est_premium=est,
        )


# --- PCR-based direction bias ---

def pcr_bias(pcr: float) -> str:
    """Return market bias string from PCR value."""
    if pcr >= 1.20:
        return "BULLISH"
    if pcr <= 0.80:
        return "BEARISH"
    return "NEUTRAL"


def pcr_confidence(pcr: float) -> float:
    """How strong is the PCR signal? 0.0-1.0."""
    if pcr >= 1.50 or pcr <= 0.65:
        return 0.9
    if pcr >= 1.20 or pcr <= 0.80:
        return 0.7
    return 0.4   # neutral zone — weak signal


# --- VIX regime ---

def vix_regime(vix: float) -> str:
    """VIX-based regime classification."""
    if vix > 25:
        return "NO_TRADE"
    if vix < 13:
        return "SELL_PREMIUM"
    if vix < 18:
        return "BALANCED"
    return "BUY_OPTIONS"


def vix_allowed(vix: float, for_action: str) -> bool:
    """Should we place a new trade given VIX and intended action type?"""
    regime = vix_regime(vix)
    if regime == "NO_TRADE":
        return False
    if regime == "SELL_PREMIUM" and for_action == "BUY":
        return False   # prefer selling in low VIX; buying is risky
    return True


# --- Delta neutrality check ---

def delta_neutral(actions: list[FallbackAction]) -> bool:
    """
    For SELL strangles / iron condors: net delta should be near zero.
    Delta sign: CE BUY = +, CE SELL = -, PE BUY = -, PE SELL = +.
    Rough delta per lot: ATM ~0.5, OTM ~0.3, ITM ~0.7 (per 75 lots).
    """
    net = 0.0
    for a in actions:
        delta_per_lot = _rough_delta(a.strike, a.option_type, a.action)
        net += delta_per_lot * a.qty * 75   # 75 = NIFTY lot size
    return abs(net) < 500                   # within +/-500 delta points = neutral


def _rough_delta(strike: float, option_type: str, action: str,
                 spot: float = 0.0) -> float:
    """Approximate delta per single option contract (not per lot)."""
    # Positive delta for long CE / short PE; negative for long PE / short CE
    sign = 1.0 if (option_type == "CE" and action == "BUY") or \
                  (option_type == "PE" and action == "SELL") else -1.0
    # ATM delta ~0.5; OTM/ITM scale toward 0.3 / 0.7
    moneyness = abs(strike - spot) if spot else 100.0
    if moneyness < 25:
        base = 0.5
    elif moneyness < 100:
        base = 0.4
    else:
        base = 0.3
    return sign * base


# --- Main fallback engine ---

class FallbackEngine:
    """
    Instant rule-based fallback — no LLM, no network, no blocking.
    Computes a FallbackDecision from raw market data in < 5ms.

    Usage:
        fb = FallbackEngine()
        decision = fb.decide(
            spot=24150, pcr=1.3, vix=14.5, expiry="29-Sep-2026",
            ce_walls=[24200, 24300], pe_walls=[24100, 24000],
            atm=24150, budget=50000, direction="BOTH",
        )
    """

    def __init__(self, iv: float = 15.0, minutes_to_expiry: int = 375):
        self.iv = iv
        self.minutes_to_expiry = minutes_to_expiry

    def decide(
        self,
        *,
        spot: float,
        pcr: float,
        vix: float,
        expiry: str,
        atm: float,
        ce_walls: list[float] | None = None,
        pe_walls: list[float] | None = None,
        budget: float = 100_000,
        direction: str = "BOTH",
        max_loss_pct: float = 20.0,
        open_positions: int = 0,
    ) -> FallbackDecision:
        """
        Return a FallbackDecision with rule-generated actions (or NO_ACTION).
        """
        ce_walls = ce_walls or []
        pe_walls = pe_walls or []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- VIX gate ---
        if not vix_allowed(vix, "BUY"):
            return FallbackDecision(
                source="RULES_FALLBACK",
                regime="NO_TRADE" if vix > 25 else "SELL_PREMIUM",
                pcr=pcr, vix=vix, spot=spot, timestamp=now,
                no_action_reason=(
                    f"VIX={vix:.1f} — "
                    f"{'too volatile for new trades' if vix > 25
                     else 'low VIX: prefer selling premium, not buying'}"
                ),
            )

        bias = pcr_bias(pcr)
        conf = pcr_confidence(pcr)
        regime = "SIDEWAYS" if bias == "NEUTRAL" else "DIRECTIONAL"

        # --- Direction filter ---
        if direction == "BUY" and bias == "BEARISH":
            bias = "NEUTRAL"   # can't short if direction=BUY only
        if direction == "SELL" and bias == "BULLISH":
            bias = "NEUTRAL"

        max_loss_rs = budget * max_loss_pct / 100.0
        actions: list[FallbackAction] = []

        # --- SIDEWAYS regime: sell premium ---
        if bias == "NEUTRAL" or vix < 13:
            # Sell OTM strangle: CE above resistance + PE below support
            if ce_walls and pe_walls and open_positions < 3:
                ce_strike = max(ce_walls) + 50
                pe_strike = min(pe_walls) - 50

                ce_premium = estimated_premium(ce_strike, spot, "CE",
                                                self.minutes_to_expiry, self.iv)
                pe_premium = estimated_premium(pe_strike, spot, "PE",
                                                self.minutes_to_expiry, self.iv)

                cost = (ce_premium + pe_premium) * 75   # selling -> premium received
                if cost <= budget * 0.3 and vix >= 13:
                    actions.append(FallbackAction(
                        type="PLACE_TRADE",
                        expiry=expiry,
                        strike=ce_strike,
                        option_type="CE",
                        action="SELL",
                        qty=1,
                        entry_price=round(ce_premium, 2),
                        sl=round(ce_premium * 2, 2),        # buy back if premium doubles
                        target=round(ce_premium * 0.5, 2),  # collect 50%
                        rationale=(
                            f"[RULES] SIDEWAYS/VIX={vix:.1f}: selling OTM CE at "
                            f"{ce_strike} (above {max(ce_walls)} CE wall). "
                            f"PCR={pcr:.2f} neutral -> premium sell."
                        ),
                    ))
                    actions.append(FallbackAction(
                        type="PLACE_TRADE",
                        expiry=expiry,
                        strike=pe_strike,
                        option_type="PE",
                        action="SELL",
                        qty=1,
                        entry_price=round(pe_premium, 2),
                        sl=round(pe_premium * 2, 2),
                        target=round(pe_premium * 0.5, 2),
                        rationale=(
                            f"[RULES] SIDEWAYS/VIX={vix:.1f}: selling OTM PE at "
                            f"{pe_strike} (below {min(pe_walls)} PE wall). "
                            f"PCR={pcr:.2f} neutral -> premium sell."
                        ),
                    ))

        # --- DIRECTIONAL BULLISH: buy CE ---
        elif bias == "BULLISH":
            # Pick strike near ATM or slightly OTM
            buy_strike = atm + 50 if atm else spot + 50
            premium = estimated_premium(buy_strike, spot, "CE",
                                         self.minutes_to_expiry, self.iv)
            sl = round(premium * 0.65, 2)          # 35% stop on premium
            target = round(premium * 1.5, 2)       # 1:1.5 RR minimum
            cost = premium * 75

            if cost <= budget and cost <= max_loss_rs * 2 and open_positions < 3:
                actions.append(FallbackAction(
                    type="PLACE_TRADE",
                    expiry=expiry,
                    strike=buy_strike,
                    option_type="CE",
                    action="BUY",
                    qty=1,
                    entry_price=round(premium, 2),
                    sl=sl,
                    target=target,
                    rationale=(
                        f"[RULES] PCR={pcr:.2f} BULLISH -> buying CE at "
                        f"{buy_strike} (spot={spot:.0f}, {buy_strike-spot:+.0f}pts "
                        f"from spot). SL at {sl:.0f} (35% premium). "
                        f"PCR>1.2 supports bullish bias."
                    ),
                ))

        # --- DIRECTIONAL BEARISH: buy PE ---
        elif bias == "BEARISH":
            buy_strike = atm - 50 if atm else spot - 50
            premium = estimated_premium(buy_strike, spot, "PE",
                                         self.minutes_to_expiry, self.iv)
            sl = round(premium * 0.65, 2)
            target = round(premium * 1.5, 2)
            cost = premium * 75

            if cost <= budget and cost <= max_loss_rs * 2 and open_positions < 3:
                actions.append(FallbackAction(
                    type="PLACE_TRADE",
                    expiry=expiry,
                    strike=buy_strike,
                    option_type="PE",
                    action="BUY",
                    qty=1,
                    entry_price=round(premium, 2),
                    sl=sl,
                    target=target,
                    rationale=(
                        f"[RULES] PCR={pcr:.2f} BEARISH -> buying PE at "
                        f"{buy_strike} (spot={spot:.0f}, {spot-buy_strike:+.0f}pts "
                        f"from spot). SL at {sl:.0f} (35% premium). "
                        f"PCR<0.8 supports bearish bias."
                    ),
                ))

        # --- No actionable signal ---
        if not actions:
            return FallbackDecision(
                source="RULES_FALLBACK",
                regime=regime,
                pcr=pcr, vix=vix, spot=spot, timestamp=now,
                confidence=conf,
                no_action_reason=(
                    f"PCR={pcr:.2f} ({bias}), VIX={vix:.1f}, "
                    f"spot={spot:.0f} — no rule-based setup matched. "
                    f"Budget={budget:,.0f}, open={open_positions}."
                ),
            )

        # --- Delta neutrality validation for multi-leg trades ---
        if len(actions) > 1 and not delta_neutral(actions):
            log.warning("Rule fallback: multi-leg trade not delta neutral — "
                        "trimming to single leg")
            actions = [actions[0]]

        return FallbackDecision(
            actions=actions,
            source="RULES_FALLBACK",
            regime=regime,
            pcr=pcr, vix=vix, spot=spot, timestamp=now,
            confidence=conf,
        )


# --- Singleton convenience ---

_fallback_engine = FallbackEngine()

def fallback_decide(**kwargs) -> FallbackDecision:
    """Stateless convenience wrapper — same signature as FallbackEngine.decide()."""
    return _fallback_engine.decide(**kwargs)
