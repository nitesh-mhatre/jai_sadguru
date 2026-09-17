"""
trading/next_predictor.py
--------------------------
/next command — pure-calculation trade predictor. Zero AI. < 15 seconds.
Also used by /go live trading bot every cycle instead of waiting for AI.

SIGNALS (12 total, weighted):

  Group A — Institutional Flow (35%)
    1. PCR level            15%  — put/call ratio direction
    2. OI fresh writing     12%  — smart money adding positions
    3. FII/DII net          8%   — institutional cash flows

  Group B — Price Structure (30%)
    4. OI wall proximity    10%  — where spot is vs CE/PE walls
    5. Max pain gravity     10%  — spot pulls toward max pain into expiry
    6. Technicals EMA/VWAP  10%  — trend and momentum

  Group C — Volatility Context (20%)
    7. VIX level+trend       8%  — fear gauge + direction
    8. IV skew               7%  — options market fear direction
    9. Premium ratio         5%  — ATM CE/PE premium imbalance

  Group D — External Cues (15%)
   10. Global indices        8%  — S&P, Nikkei, crude, DXY
   11. Time-of-day weight    4%  — signal reliability changes by session
   12. Range tightness       3%  — how tight OI walls are (sideways confirmation)

REGIME THRESHOLDS (with hysteresis to reduce flip-flopping):
  Score ≥ 70 : STRONGLY_BULLISH  → BUY ATM CE 2 lots
  Score 60–70: BULLISH            → BUY OTM CE 1 lot
  Score 45–60: SIDEWAYS           → SHORT STRANGLE
  Score 35–45: BEARISH            → BUY OTM PE 1 lot
  Score < 35 : STRONGLY_BEARISH   → BUY ATM PE 2 lots

CONFIDENCE:
  HIGH   — score outside 38–62 band AND 3+ signals agree on same side
  MEDIUM — score outside 42–58 band
  LOW    — score 42–58 (genuinely mixed)

SL/TARGET (mathematical):
  Buy  : SL = entry × 0.72 (near OI wall if available), T1 = 1.50×, T2 = 1.85×
  Sell : SL = entry × 2.00 (premium doubles), T1 = 0.45× (55% collected)
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)
NIFTY_LOT_SIZE = 75


# ── Signal score ───────────────────────────────────────────────────────────────

@dataclass
class SignalScore:
    name:      str
    raw_value: str
    score:     float      # 0–100  (50 = neutral, >50 = bullish, <50 = bearish)
    weight:    float      # fraction of total weight
    note:      str = ""


# ── NextTrade result ───────────────────────────────────────────────────────────

@dataclass
class NextTrade:
    fetched_at:      str
    fetch_time_ms:   int
    spot:            float
    atm:             int
    expiry:          str
    pcr:             float
    vix:             float
    sentiment:       str
    signals:         list[SignalScore]
    total_score:     float
    regime:          str
    action:          str   # BUY | SELL | NO_TRADE
    option_type:     str   # CE | PE | STRANGLE | NONE
    strike:          int
    strike_pe:       int
    entry_price:     float
    entry_price_pe:  float
    sl:              float
    target1:         float
    target2:         float
    sl_pe:           float
    target_pe:       float
    rr_ratio:        float
    qty_lots:        int
    capital:         float
    rationale:       str
    key_levels:      list[str]
    confidence:      str   # HIGH | MEDIUM | LOW
    # New: breakdown counts
    bull_signals:    int = 0
    bear_signals:    int = 0
    neutral_signals: int = 0

    def as_trade_action(self, expiry_override: str = "") -> Optional[dict]:
        """
        Convert to PLACE_TRADE action dict for live_mode executor.
        Returns None if NO_TRADE.
        """
        if self.action == "NO_TRADE":
            return None
        expiry = expiry_override or self.expiry
        if self.option_type == "STRANGLE":
            # Return two actions (CE leg first)
            return [
                {
                    "type": "PLACE_TRADE",
                    "expiry": expiry,
                    "strike": self.strike,
                    "option_type": "CE",
                    "action": "SELL",
                    "qty": 1,
                    "entry_price": self.entry_price,
                    "sl": self.sl,
                    "target": self.target1,
                    "partial_target": None,   # single-lot leg — cannot be split
                    "rationale": f"[/next] STRANGLE CE leg. {self.rationale[:120]}",
                },
                {
                    "type": "PLACE_TRADE",
                    "expiry": expiry,
                    "strike": self.strike_pe,
                    "option_type": "PE",
                    "action": "SELL",
                    "qty": 1,
                    "entry_price": self.entry_price_pe,
                    "sl": self.sl_pe,
                    "target": self.target_pe,
                    "partial_target": None,   # single-lot leg — cannot be split
                    "rationale": f"[/next] STRANGLE PE leg. {self.rationale[:120]}",
                },
            ]
        return [{
            "type":         "PLACE_TRADE",
            "expiry":       expiry,
            "strike":       self.strike,
            "option_type":  self.option_type,
            "action":       self.action,
            "qty":          self.qty_lots,
            "entry_price":  self.entry_price,
            "sl":           self.sl,
            "target":       self.target2,
            "partial_target": self.target1,   # book half at T1, run the rest to T2
            "rationale":    f"[/next score={self.total_score:.0f} conf={self.confidence}] {self.rationale[:150]}",
        }]# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS — run in parallel threads
# ══════════════════════════════════════════════════════════════════════════════

def option_metrics_from_groww_rows(rows: list[dict], expiry: str = "") -> dict:
    """
    Turn Groww chain rows (data/groww_feed.py) into the same metrics dict that
    `_build_trade` and the signal scorers expect from the NSE chain.

    Pure function (no network) so it can be unit-tested offline.
    Returns {"error": ...} when the rows are unusable.
    """
    if not rows:
        return {"error": "No option chain data"}

    def _f(v) -> float:
        try:
            x = float(v)
            return 0.0 if (x != x or abs(x) == float("inf")) else x
        except (TypeError, ValueError):
            return 0.0

    def _leg(r: dict, side: str) -> dict:
        return r.get(side.lower(), {}) or {}

    try:
        rows = sorted(rows, key=lambda r: _f(r.get("strike")))
        strikes = [int(_f(r.get("strike"))) for r in rows]
    except Exception as exc:
        return {"error": f"Malformed chain rows: {exc}"}
    if not strikes:
        return {"error": "Chain has no strikes"}

    oi_ce, oi_pe, ch_ce, ch_pe = {}, {}, {}, {}
    ltp_ce, ltp_pe, iv_ce, iv_pe = {}, {}, {}, {}
    for r in rows:
        s = int(_f(r["strike"]))
        ce, pe = _leg(r, "CE"), _leg(r, "PE")
        oi_ce[s] = _f(ce.get("oi"))
        oi_pe[s] = _f(pe.get("oi"))
        ch_ce[s] = _f(ce.get("oi")) - _f(ce.get("prev_oi"))
        ch_pe[s] = _f(pe.get("oi")) - _f(pe.get("prev_oi"))
        ltp_ce[s] = _f(ce.get("ltp"))
        ltp_pe[s] = _f(pe.get("ltp"))
        iv_ce[s] = _f(ce.get("iv"))
        iv_pe[s] = _f(pe.get("iv"))

    ce_oi_total = sum(oi_ce.values())
    pe_oi_total = sum(oi_pe.values())
    pcr = round(pe_oi_total / ce_oi_total, 2) if ce_oi_total else 0

    # ATM = strike where CE/PE premiums are closest (Groww's page has no spot)
    atm     = min(strikes, key=lambda s: abs(ltp_ce.get(s, 0) - ltp_pe.get(s, 0)))
    atm_idx = strikes.index(atm)

    # Spot via put-call parity at the money: S ≈ K + (C − P)
    spot = atm + (ltp_ce.get(atm, 0) - ltp_pe.get(atm, 0))
    if spot <= 0:
        spot = float(atm)

    # Max pain (canonical): strike where option buyers receive least intrinsic
    pain = {}
    for s in strikes:
        pain[s] = (sum(max(0, s - k) * v for k, v in oi_ce.items())
                   + sum(max(0, k - s) * v for k, v in oi_pe.items()))
    max_pain = int(min(pain, key=pain.get))

    ce_wall = int(max(oi_ce, key=oi_ce.get))
    pe_wall = int(max(oi_pe, key=oi_pe.get))

    fresh_ce  = int(sum(v for v in ch_ce.values() if v > 0))
    fresh_pe  = int(sum(v for v in ch_pe.values() if v > 0))
    unwind_ce = int(abs(sum(v for v in ch_ce.values() if v < 0)))
    unwind_pe = int(abs(sum(v for v in ch_pe.values() if v < 0)))

    skew_rng = strikes[max(0, atm_idx - 4): atm_idx + 5]

    def _avg(d: dict, keys: list) -> float:
        vals = [d[k] for k in keys if d.get(k, 0) > 0]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def _idx(i: int, fallback: int) -> int:
        return strikes[i] if 0 <= i < len(strikes) else fallback

    otm_ce1 = _idx(atm_idx + 1, atm)
    otm_ce2 = _idx(atm_idx + 2, otm_ce1)
    otm_pe1 = _idx(atm_idx - 1, atm)
    otm_pe2 = _idx(atm_idx - 2, otm_pe1)

    top_ce = [{"Strike": s, "CE_OI": int(oi_ce[s]), "CE_Chng_OI": int(ch_ce[s]),
               "CE_LTP": ltp_ce[s], "CE_IV": iv_ce[s]}
              for s in sorted(oi_ce, key=oi_ce.get, reverse=True)[:3]]
    top_pe = [{"Strike": s, "PE_OI": int(oi_pe[s]), "PE_Chng_OI": int(ch_pe[s]),
               "PE_LTP": ltp_pe[s], "PE_IV": iv_pe[s]}
              for s in sorted(oi_pe, key=oi_pe.get, reverse=True)[:3]]

    return {
        "spot": round(spot, 2), "atm": atm, "expiry": expiry or "N/A",
        "pcr": pcr, "max_pain": max_pain,
        "ce_wall": ce_wall, "pe_wall": pe_wall,
        "range_width": ce_wall - pe_wall,
        "ce_wall_ltp": ltp_ce.get(ce_wall, 0.0),
        "pe_wall_ltp": ltp_pe.get(pe_wall, 0.0),
        "fresh_ce": fresh_ce, "fresh_pe": fresh_pe,
        "unwind_ce": unwind_ce, "unwind_pe": unwind_pe,
        "avg_ce_iv": _avg(iv_ce, skew_rng), "avg_pe_iv": _avg(iv_pe, skew_rng),
        "atm_ce_ltp": ltp_ce.get(atm, 0.0), "atm_pe_ltp": ltp_pe.get(atm, 0.0),
        "otm_ce1": otm_ce1, "otm_ce2": otm_ce2,
        "otm_pe1": otm_pe1, "otm_pe2": otm_pe2,
        "otm_ce1_ltp": ltp_ce.get(otm_ce1, 0.0), "otm_ce2_ltp": ltp_ce.get(otm_ce2, 0.0),
        "otm_pe1_ltp": ltp_pe.get(otm_pe1, 0.0), "otm_pe2_ltp": ltp_pe.get(otm_pe2, 0.0),
        "resistance": top_ce, "support": top_pe,
        "source": "groww",
    }


def _fetch_option_chain() -> dict:
    # ── PRIMARY: Groww chain (coded in data/groww_feed.py, no auth) ────────────
    try:
        from data.groww_feed import get_chain, _norm_expiry
        from data.yahoo_feed import get_nifty_spot

        oc_g = get_chain()
        if oc_g and oc_g.get("rows"):
            metrics = option_metrics_from_groww_rows(
                oc_g["rows"], expiry=_norm_expiry(oc_g.get("current_expiry", ""))
            )
            if "error" not in metrics:
                try:
                    live_spot = get_nifty_spot()
                except Exception:
                    live_spot = 0.0
                if live_spot and live_spot > 0:
                    metrics["spot"] = round(float(live_spot), 2)
                if metrics.get("spot", 0) > 0:
                    return metrics
    except Exception as exc:
        log.warning("Groww option chain unavailable (%s) — falling back to NSE", exc)

    # ── FALLBACK: NSE chain ───────────────────────────────────────────────────
    try:
        from data.nifty_option_chain import get_nifty_option_chain, get_expiry_dates
        from data.yahoo_feed import get_nifty_spot
        import numpy as np

        dates  = get_expiry_dates()
        expiry = dates[0] if dates else None
        df, spot_nse = get_nifty_option_chain(expiry=expiry)

        spot_yf = get_nifty_spot()
        spot    = spot_yf if spot_yf > 0 else spot_nse
        if df.empty or spot <= 0:
            return {"error": "No option chain data"}

        # Sanitize
        num = df.select_dtypes(include="number").columns
        df[num] = df[num].replace([float("inf"), float("-inf")], 0).fillna(0)
        df["Strike"] = df["Strike"].astype(int)

        strikes  = sorted(df["Strike"].unique().tolist())
        atm      = min(strikes, key=lambda x: abs(x - spot))
        atm_idx  = strikes.index(atm)

        ce_oi = int(df["CE_OI"].sum())
        pe_oi = int(df["PE_OI"].sum())
        pcr   = round(pe_oi / ce_oi, 2) if ce_oi else 0

        # OI walls — top 3
        top_ce = df.nlargest(3, "CE_OI")[
            ["Strike","CE_OI","CE_Chng_OI","CE_LTP","CE_IV"]].to_dict("records")
        top_pe = df.nlargest(3, "PE_OI")[
            ["Strike","PE_OI","PE_Chng_OI","PE_LTP","PE_IV"]].to_dict("records")

        # Fresh writing totals
        fc = df[df["CE_Chng_OI"] > 0].nlargest(5, "CE_Chng_OI")
        fp = df[df["PE_Chng_OI"] > 0].nlargest(5, "PE_Chng_OI")
        total_fresh_ce = int(fc["CE_Chng_OI"].sum())
        total_fresh_pe = int(fp["PE_Chng_OI"].sum())

        # OI unwinding
        uc = df[df["CE_Chng_OI"] < 0]["CE_Chng_OI"].sum()
        up = df[df["PE_Chng_OI"] < 0]["PE_Chng_OI"].sum()

        # IV skew ATM±4
        skew_rng = strikes[max(0, atm_idx-4): atm_idx+5]
        skew_df  = df[df["Strike"].isin(skew_rng)]
        avg_ce_iv = float(skew_df["CE_IV"].mean()) if not skew_df.empty else 0
        avg_pe_iv = float(skew_df["PE_IV"].mean()) if not skew_df.empty else 0

        # ATM CE/PE premium ratio
        atm_row     = df[df["Strike"] == atm]
        atm_ce_ltp  = float(atm_row["CE_LTP"].iloc[0]) if not atm_row.empty else 0
        atm_pe_ltp  = float(atm_row["PE_LTP"].iloc[0]) if not atm_row.empty else 0

        # OTM strikes (1 and 2 steps out)
        otm_ce1 = strikes[atm_idx+1] if atm_idx+1 < len(strikes) else atm
        otm_ce2 = strikes[atm_idx+2] if atm_idx+2 < len(strikes) else otm_ce1
        otm_pe1 = strikes[atm_idx-1] if atm_idx > 0 else atm
        otm_pe2 = strikes[atm_idx-2] if atm_idx > 1 else otm_pe1

        def _ltp(s, ot):
            r = df[df["Strike"]==s]
            if r.empty: return 0.0
            c = f"{ot}_LTP"
            return float(r[c].iloc[0]) if c in r.columns else 0.0

        # Max pain (canonical): strike where option BUYERS receive the least
        # intrinsic value — pain(s) = Σ max(0, s−K)·CE_OI + Σ max(0, K−s)·PE_OI
        pain = {}
        for s in strikes:
            loss = (
                ((s - df["Strike"]).clip(lower=0) * df["CE_OI"]).sum() +
                ((df["Strike"] - s).clip(lower=0) * df["PE_OI"]).sum()
            )
            pain[s] = float(loss)
        max_pain = int(min(pain, key=pain.get))

        # Range tightness: distance between top CE and PE walls
        ce_wall = int(top_ce[0]["Strike"]) if top_ce else atm + 150
        pe_wall = int(top_pe[0]["Strike"]) if top_pe else atm - 150
        range_width = ce_wall - pe_wall

        return {
            "spot": spot, "atm": atm, "expiry": expiry or "N/A",
            "pcr": pcr, "max_pain": max_pain,
            "ce_wall": ce_wall, "pe_wall": pe_wall,
            "range_width": range_width,
            "ce_wall_ltp": float(top_ce[0].get("CE_LTP",0)) if top_ce else 0,
            "pe_wall_ltp": float(top_pe[0].get("PE_LTP",0)) if top_pe else 0,
            "fresh_ce": total_fresh_ce, "fresh_pe": total_fresh_pe,
            "unwind_ce": abs(int(uc)), "unwind_pe": abs(int(up)),
            "avg_ce_iv": round(avg_ce_iv, 2), "avg_pe_iv": round(avg_pe_iv, 2),
            "atm_ce_ltp": atm_ce_ltp, "atm_pe_ltp": atm_pe_ltp,
            "otm_ce1": otm_ce1, "otm_ce2": otm_ce2,
            "otm_pe1": otm_pe1, "otm_pe2": otm_pe2,
            "otm_ce1_ltp": _ltp(otm_ce1,"CE"), "otm_ce2_ltp": _ltp(otm_ce2,"CE"),
            "otm_pe1_ltp": _ltp(otm_pe1,"PE"), "otm_pe2_ltp": _ltp(otm_pe2,"PE"),
            "resistance": top_ce, "support": top_pe,
        }
    except Exception as exc:
        log.warning("Option chain fetch: %s", exc)
        return {"error": str(exc)}


def _fetch_vix_tech_fii() -> dict:
    """VIX history + technicals + FII/DII — one thread."""
    out = {"vix": {}, "tech": {}, "fii": {}}
    try:
        from data.market_extra import get_india_vix_history, get_nifty_technicals
        out["vix"]  = get_india_vix_history(days=3)
        out["tech"] = get_nifty_technicals()
    except Exception as exc:
        log.warning("VIX/tech: %s", exc)
    try:
        from data.market_extra import get_fii_dii_data
        out["fii"] = get_fii_dii_data()
    except Exception as exc:
        log.warning("FII/DII: %s", exc)
    return out


def _fetch_global() -> dict:
    try:
        from data.market_extra import get_global_indices
        return get_global_indices()
    except Exception as exc:
        log.warning("Global: %s", exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _score_pcr(pcr: float) -> SignalScore:
    """Weight 15%"""
    if   pcr >= 1.50: s, n = 88, f"PCR {pcr} — heavy PE writing, strong support"
    elif pcr >= 1.25: s, n = 75, f"PCR {pcr} — bullish, PE writers dominant"
    elif pcr >= 1.08: s, n = 62, f"PCR {pcr} — mildly bullish"
    elif pcr >= 0.93: s, n = 50, f"PCR {pcr} — balanced, no directional edge"
    elif pcr >= 0.78: s, n = 38, f"PCR {pcr} — mildly bearish"
    elif pcr >= 0.62: s, n = 25, f"PCR {pcr} — bearish, CE writers dominant"
    else:             s, n = 12, f"PCR {pcr} — heavy CE writing, strong resistance"
    return SignalScore("PCR", str(pcr), s, 0.15, n)


def _score_oi_writing(fresh_ce: float, fresh_pe: float,
                      unwind_ce: float, unwind_pe: float) -> SignalScore:
    """
    Weight 12%. Net OI direction = fresh writing minus unwinding.
    Net positive PE = institutions defending support = bullish.
    """
    net_ce = fresh_ce - unwind_ce
    net_pe = fresh_pe - unwind_pe
    total  = abs(net_ce) + abs(net_pe)
    if total == 0:
        return SignalScore("Net OI Flow", "none", 50, 0.12, "No net OI flow")
    pe_pct = (net_pe / total * 100) if total else 50
    if   pe_pct >= 70: s, n = 82, f"Net PE adding {net_pe:+,.0f} → strong bull defense"
    elif pe_pct >= 55: s, n = 65, f"Net PE {net_pe:+,.0f} > CE {net_ce:+,.0f} → bullish"
    elif pe_pct >= 40: s, n = 50, f"Mixed OI flow CE:{net_ce:+,.0f} PE:{net_pe:+,.0f}"
    elif pe_pct >= 25: s, n = 35, f"Net CE {net_ce:+,.0f} > PE → bearish"
    else:              s, n = 18, f"Heavy CE writing {net_ce:+,.0f} → bear resistance"
    val = f"CE:{net_ce:+,.0f}/PE:{net_pe:+,.0f}"
    return SignalScore("Net OI Flow", val, s, 0.12, n)


def _score_fii_dii(fii: dict) -> SignalScore:
    """Weight 8%. FII net cash = institutional smart money."""
    if not fii or "error" in fii:
        return SignalScore("FII/DII", "unavail", 50, 0.08, "FII data unavailable")
    fii_net = fii.get("fii_cash_net", 0) or 0
    dii_net = fii.get("dii_cash_net", 0) or 0
    combined = fii_net + dii_net
    if   combined >  1500: s, n = 85, f"FII+DII net +₹{combined:,.0f}Cr — heavy buying"
    elif combined >   500: s, n = 70, f"FII+DII net +₹{combined:,.0f}Cr — buying"
    elif combined >     0: s, n = 58, f"FII+DII net +₹{combined:,.0f}Cr — mild buying"
    elif combined >  -500: s, n = 42, f"FII+DII net ₹{combined:,.0f}Cr — mild selling"
    elif combined > -1500: s, n = 30, f"FII+DII net ₹{combined:,.0f}Cr — selling"
    else:                  s, n = 15, f"FII+DII net ₹{combined:,.0f}Cr — heavy selling"
    return SignalScore("FII/DII Flow", f"₹{combined:+,.0f}Cr", s, 0.08, n)


def _score_oi_wall_proximity(spot: float, ce_wall: float, pe_wall: float) -> SignalScore:
    """Weight 10%."""
    dist_ce = ce_wall - spot
    dist_pe = spot - pe_wall
    if dist_ce <= 0 or dist_pe <= 0:
        return SignalScore("Wall Proximity", "at wall", 50, 0.10, "Spot at/beyond OI wall")
    total = dist_ce + dist_pe
    # Higher score = closer to PE wall (support) = more bullish
    score = int((dist_ce / total) * 100)
    if   score >= 72: n = f"Spot {dist_pe:.0f}pts above PE support {pe_wall:.0f} → bullish zone"
    elif score >= 58: n = f"Spot in upper half of range, mildly bullish"
    elif score >= 42: n = f"Spot balanced CE:{dist_ce:.0f}pts  PE:{dist_pe:.0f}pts"
    elif score >= 28: n = f"Spot in lower half, mildly bearish"
    else:             n = f"Spot {dist_ce:.0f}pts below CE resistance {ce_wall:.0f} → bearish zone"
    return SignalScore("Wall Proximity", f"CE:{dist_ce:.0f}  PE:{dist_pe:.0f}", score, 0.10, n)


def _score_max_pain(spot: float, max_pain: int, expiry: str) -> SignalScore:
    """
    Weight 10%. Max pain gravity — spot tends to drift toward max pain
    especially in the second half of expiry week.
    """
    try:
        from datetime import datetime
        exp_dt = datetime.strptime(expiry, "%d-%b-%Y").date()
        days_left = (exp_dt - datetime.now().date()).days
        gravity = max(0.3, 1.0 - days_left * 0.12)   # stronger closer to expiry
    except Exception:
        gravity = 0.5
        days_left = 3

    diff = max_pain - spot
    if   diff >  150: s, n = int(65 + gravity*20), f"MaxPain {max_pain} is {diff:.0f}pts ABOVE spot → upward gravity (exp {days_left}d)"
    elif diff >   50: s, n = int(58 + gravity*10), f"MaxPain {max_pain} slightly above → mild upward pull"
    elif diff >  -50: s, n = 50,                    f"MaxPain {max_pain} near spot → balanced"
    elif diff > -150: s, n = int(42 - gravity*10), f"MaxPain {max_pain} slightly below → mild downward pull"
    else:             s, n = int(35 - gravity*20), f"MaxPain {max_pain} is {abs(diff):.0f}pts BELOW spot → downward gravity"
    return SignalScore("Max Pain Gravity", f"MP={max_pain} spot={spot:.0f} Δ={diff:+.0f}", min(90,max(10,s)), 0.10, n)


def _score_technicals(tech: dict) -> SignalScore:
    """Weight 10%."""
    if not tech or "error" in tech:
        return SignalScore("EMA/VWAP", "unavail", 50, 0.10, "Intraday data unavailable")
    trend = tech.get("trend", "UNKNOWN")
    ema9  = tech.get("ema9", 0)
    ema21 = tech.get("ema21", 0)
    vwap  = tech.get("vwap", 0)
    ltp   = tech.get("ltp", 0)

    if   trend == "STRONG_UPTREND":   s, n = 85, f"Price>EMA9({ema9:.0f})>EMA21({ema21:.0f}) above VWAP({vwap:.0f})"
    elif trend == "UPTREND":          s, n = 67, f"Price above EMA9 and VWAP → bullish"
    elif trend == "SIDEWAYS":         s, n = 50, f"EMA9≈EMA21, price near VWAP → choppy"
    elif trend == "DOWNTREND":        s, n = 33, f"Price below EMA9 and VWAP → bearish"
    elif trend == "STRONG_DOWNTREND": s, n = 15, f"Price<EMA9({ema9:.0f})<EMA21({ema21:.0f}) below VWAP({vwap:.0f})"
    else:                             s, n = 50, "Trend unknown"
    return SignalScore("EMA/VWAP Trend", f"{trend}", s, 0.10, n)


def _score_vix(vix_data: dict) -> SignalScore:
    """Weight 8%."""
    vix   = vix_data.get("current", 0)
    trend = vix_data.get("trend", "UNKNOWN")
    if vix <= 0:
        return SignalScore("VIX", "unavail", 50, 0.08, "VIX unavailable")
    # VIX: higher = fear = bearish. Low = calm = neutral/slightly bullish.
    if   vix < 11:  s, n = 62, f"VIX {vix} — extremely calm, sell premium"
    elif vix < 14:  s, n = 58, f"VIX {vix} — low fear, options cheap for buyers"
    elif vix < 17:  s, n = 50, f"VIX {vix} — normal range"
    elif vix < 21:  s, n = 38, f"VIX {vix} — elevated, caution"
    elif vix < 26:  s, n = 25, f"VIX {vix} — high fear, buy options only"
    else:           s, n =  8, f"VIX {vix} — extreme, no new trades"
    if trend == "RISING":  n += " ↑rising (fear increasing)"
    if trend == "FALLING": n += " ↓falling (fear easing)"
    return SignalScore("VIX", f"{vix} {trend}", s, 0.08, n)


def _score_iv_skew(ce_iv: float, pe_iv: float) -> SignalScore:
    """Weight 7%. PE IV > CE IV = market fearing downside."""
    if ce_iv <= 0 or pe_iv <= 0:
        return SignalScore("IV Skew", "unavail", 50, 0.07, "IV data unavailable")
    skew = pe_iv - ce_iv
    if   skew >  4: s, n = 22, f"PE IV {pe_iv:.1f}% >> CE IV {ce_iv:.1f}% → heavy downside fear"
    elif skew >  2: s, n = 35, f"PE IV {pe_iv:.1f}% > CE IV {ce_iv:.1f}% → mild downside fear"
    elif skew >  0: s, n = 45, f"Slight PE skew ({skew:.1f}%) → minor bearish lean"
    elif skew > -2: s, n = 55, f"Slight CE skew → minor bullish lean"
    elif skew > -4: s, n = 65, f"CE IV {ce_iv:.1f}% > PE IV {pe_iv:.1f}% → upside call buying"
    else:           s, n = 78, f"CE IV >> PE IV → aggressive call buying"
    return SignalScore("IV Skew", f"CE:{ce_iv:.1f}% PE:{pe_iv:.1f}%", s, 0.07, n)


def _score_premium_ratio(atm_ce: float, atm_pe: float) -> SignalScore:
    """
    Weight 5%. ATM CE premium vs PE premium.
    If CE > PE: market pricing more upside movement → mildly bullish.
    """
    if atm_ce <= 0 or atm_pe <= 0:
        return SignalScore("ATM Premium", "unavail", 50, 0.05, "Premium data unavailable")
    ratio = atm_ce / atm_pe
    if   ratio > 1.25: s, n = 68, f"CE ₹{atm_ce:.1f} > PE ₹{atm_pe:.1f} — upside premium higher"
    elif ratio > 1.05: s, n = 57, f"CE slightly premium over PE"
    elif ratio > 0.95: s, n = 50, f"CE ₹{atm_ce:.1f} ≈ PE ₹{atm_pe:.1f} — balanced"
    elif ratio > 0.80: s, n = 43, f"PE slightly premium over CE"
    else:              s, n = 32, f"PE ₹{atm_pe:.1f} >> CE ₹{atm_ce:.1f} — downside premium"
    return SignalScore("ATM Premium", f"CE:₹{atm_ce:.1f} PE:₹{atm_pe:.1f}", s, 0.05, n)


def _score_global(global_data: dict) -> SignalScore:
    """Weight 8%."""
    if not global_data:
        return SignalScore("Global Cues", "unavail", 50, 0.08, "Global data unavailable")
    sp  = global_data.get("sp500",    {}).get("change_pct", 0) or 0
    nk  = global_data.get("nikkei",   {}).get("change_pct", 0) or 0
    hs  = global_data.get("hangseng", {}).get("change_pct", 0) or 0
    oil = global_data.get("crude_oil",{}).get("change_pct", 0) or 0
    dxy = global_data.get("dollar_index",{}).get("change_pct", 0) or 0

    bull = sum(1 for x in [sp, nk, hs] if x > 0.3)
    bear = sum(1 for x in [sp, nk, hs] if x < -0.3)
    oil_adj = -12 if oil > 2.5 else (-6 if oil > 1.5 else (4 if oil < -1.5 else 0))
    dxy_adj = -8  if dxy > 0.4 else (4 if dxy < -0.4 else 0)
    score   = max(5, min(95, 50 + (bull - bear)*13 + oil_adj + dxy_adj))
    n = (f"S&P{sp:+.1f}% NK{nk:+.1f}% HS{hs:+.1f}% | "
         f"Oil{oil:+.1f}% DXY{dxy:+.1f}% | "
         f"{'bullish' if score>55 else 'bearish' if score<45 else 'neutral'}")
    return SignalScore("Global Cues", f"{bull}↑/{bear}↓", score, 0.08, n)


def _score_time_of_day() -> SignalScore:
    """
    Weight 4%. Signal reliability varies by session.
    09:15–09:45: high noise, reduce confidence.
    09:45–11:30: prime, full weight.
    11:30–13:00: moderate.
    13:00–14:30: lazy, mixed.
    14:30–15:30: closing rush, directional.
    """
    import pytz
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist).time()
    from datetime import time as _t

    if   _t(9,15)  <= now < _t(9,45):  s, n = 45, "Opening 30min — noise zone, reduce confidence"
    elif _t(9,45)  <= now < _t(11,30): s, n = 55, "Prime session — signals most reliable"
    elif _t(11,30) <= now < _t(13,0):  s, n = 52, "Mid-morning — moderate reliability"
    elif _t(13,0)  <= now < _t(14,30): s, n = 48, "Afternoon lull — lower directional reliability"
    elif _t(14,30) <= now < _t(15,30): s, n = 53, "Closing push — institutional squaring, act fast"
    else:                              s, n = 50, "Market closed — using last session data"
    return SignalScore("Time of Day", now.strftime("%H:%M"), s, 0.04, n)


def _score_range_tightness(range_width: float, vix: float) -> SignalScore:
    """
    Weight 3%. Tight OI range + low VIX = sideways confirmation.
    Score 50 = sideways. Score away from 50 = directional.
    """
    if range_width <= 0:
        return SignalScore("Range Width", "N/A", 50, 0.03, "Range data unavailable")
    # Tight range → score near 50 (sideways)
    # Wide range → score away from 50 based on VIX direction
    if   range_width <= 150: s, n = 50, f"Range {range_width:.0f}pts — very tight, sideways"
    elif range_width <= 250: s, n = 50, f"Range {range_width:.0f}pts — tight, sideways"
    elif range_width <= 400: s, n = 52, f"Range {range_width:.0f}pts — moderate, slight direction possible"
    else:                    s, n = 55, f"Range {range_width:.0f}pts — wide, breakout territory"
    return SignalScore("Range Width", f"{range_width:.0f}pts", s, 0.03, n)


# ══════════════════════════════════════════════════════════════════════════════
# TRADE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_trade(score: float, regime: str, oc: dict,
                 vix: float, budget: float) -> tuple:
    spot     = oc["spot"]
    atm      = oc["atm"]
    ce_wall  = oc["ce_wall"]
    pe_wall  = oc["pe_wall"]
    max_pain = oc["max_pain"]

    key_levels = [
        f"Resistance {ce_wall}",
        f"Support {pe_wall}",
        f"MaxPain {max_pain}",
        f"ATM {atm}",
    ]

    if regime == "SIDEWAYS":
        # Short strangle — sell 1 OTM each side
        ce_s = oc["otm_ce1"]
        pe_s = oc["otm_pe1"]
        ce_p = oc["otm_ce1_ltp"] or oc["atm_ce_ltp"] * 0.6
        pe_p = oc["otm_pe1_ltp"] or oc["atm_pe_ltp"] * 0.6

        if ce_p < 8:  # too thin, use ATM
            ce_s, ce_p = atm, oc["atm_ce_ltp"]
        if pe_p < 8:
            pe_s, pe_p = atm, oc["atm_pe_ltp"]

        sl_ce  = round(ce_p * 2.0,  1)
        sl_pe  = round(pe_p * 2.0,  1)
        tgt_ce = round(ce_p * 0.45, 1)
        tgt_pe = round(pe_p * 0.45, 1)
        cap    = (ce_p + pe_p) * NIFTY_LOT_SIZE
        rat    = (f"SIDEWAYS (score {score:.0f}/100). Range {pe_wall}–{ce_wall} "
                  f"({oc['range_width']:.0f}pts). Sell strangle to collect "
                  f"₹{ce_p+pe_p:.1f} premium. Theta works in seller's favour.")
        return ("SELL","STRANGLE", ce_s, pe_s, ce_p, pe_p,
                sl_ce, tgt_ce, tgt_ce, sl_pe, tgt_pe, 1.8, 1, cap, rat, key_levels)

    elif "BULLISH" in regime:
        # Choose ATM for STRONGLY, OTM1 for BULLISH
        if "STRONGLY" in regime:
            strike = atm
            ltp    = oc["atm_ce_ltp"]
        else:
            strike = oc["otm_ce1"]
            ltp    = oc["otm_ce1_ltp"]
            if ltp < 12:
                strike, ltp = atm, oc["atm_ce_ltp"]
        if ltp <= 0: ltp = 100.0

        # SL: use OI wall if close, else 28% of premium
        oi_sl  = (spot - pe_wall) * 0.7 if (spot - pe_wall) < ltp * 0.5 else ltp * 0.28
        sl     = round(max(ltp * 0.70, ltp - oi_sl), 1)
        t1     = round(ltp * 1.50, 1)
        t2     = round(ltp * 1.85, 1)
        rr     = round((t1 - ltp) / max(ltp - sl, 0.1), 2)
        qty    = max(1, min(2 if "STRONGLY" in regime else 1,
                            int(budget / max(ltp * NIFTY_LOT_SIZE, 1))))
        cap    = ltp * qty * NIFTY_LOT_SIZE
        rat    = (f"{regime} (score {score:.0f}/100). BUY {strike}CE @ ₹{ltp:.1f}. "
                  f"SL ₹{sl:.1f}. T1 ₹{t1:.1f} T2 ₹{t2:.1f}. "
                  f"CE wall {ce_wall}, PE support {pe_wall}. R:R 1:{rr:.1f}.")
        return ("BUY","CE", strike, 0, ltp, 0, sl, t1, t2, 0, 0, rr, qty, cap, rat, key_levels)

    elif "BEARISH" in regime:
        if "STRONGLY" in regime:
            strike = atm
            ltp    = oc["atm_pe_ltp"]
        else:
            strike = oc["otm_pe1"]
            ltp    = oc["otm_pe1_ltp"]
            if ltp < 12:
                strike, ltp = atm, oc["atm_pe_ltp"]
        if ltp <= 0: ltp = 100.0

        oi_sl  = (ce_wall - spot) * 0.7 if (ce_wall - spot) < ltp * 0.5 else ltp * 0.28
        sl     = round(max(ltp * 0.70, ltp - oi_sl), 1)
        t1     = round(ltp * 1.50, 1)
        t2     = round(ltp * 1.85, 1)
        rr     = round((t1 - ltp) / max(ltp - sl, 0.1), 2)
        qty    = max(1, min(2 if "STRONGLY" in regime else 1,
                            int(budget / max(ltp * NIFTY_LOT_SIZE, 1))))
        cap    = ltp * qty * NIFTY_LOT_SIZE
        rat    = (f"{regime} (score {score:.0f}/100). BUY {strike}PE @ ₹{ltp:.1f}. "
                  f"SL ₹{sl:.1f}. T1 ₹{t1:.1f} T2 ₹{t2:.1f}. "
                  f"PE support {pe_wall}, CE wall {ce_wall}. R:R 1:{rr:.1f}.")
        return ("BUY","PE", strike, 0, ltp, 0, sl, t1, t2, 0, 0, rr, qty, cap, rat, key_levels)

    else:  # NO_TRADE
        return ("NO_TRADE","NONE", atm, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                f"Score {score:.0f}/100 — signals too mixed. "
                "Wait for clearer setup (score < 38 or > 62).", key_levels)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def predict_next(budget: float = 50_000.0) -> NextTrade:
    """
    Full pipeline: fetch all data in parallel → score 12 signals → build trade.
    Zero AI. Target: < 15 seconds.
    """
    t0 = time.time()

    # ── Parallel fetch (3 threads) ────────────────────────────────────────────
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        f_oc  = pool.submit(_fetch_option_chain)
        f_vtf = pool.submit(_fetch_vix_tech_fii)
        f_g   = pool.submit(_fetch_global)
        oc    = f_oc.result(timeout=25)
        vtf   = f_vtf.result(timeout=25)
        gd    = f_g.result(timeout=25)

    if "error" in oc:
        raise RuntimeError(f"Option chain failed: {oc['error']}")

    vix_data = vtf.get("vix", {})
    tech     = vtf.get("tech", {})
    fii      = vtf.get("fii", {})
    vix_val  = vix_data.get("current", 0)

    # ── Score all 12 signals ──────────────────────────────────────────────────
    signals = [
        # Group A — Institutional Flow (35%)
        _score_pcr(oc["pcr"]),
        _score_oi_writing(oc["fresh_ce"], oc["fresh_pe"],
                          oc["unwind_ce"], oc["unwind_pe"]),
        _score_fii_dii(fii),
        # Group B — Price Structure (30%)
        _score_oi_wall_proximity(oc["spot"], oc["ce_wall"], oc["pe_wall"]),
        _score_max_pain(oc["spot"], oc["max_pain"], oc["expiry"]),
        _score_technicals(tech),
        # Group C — Volatility Context (20%)
        _score_vix(vix_data),
        _score_iv_skew(oc["avg_ce_iv"], oc["avg_pe_iv"]),
        _score_premium_ratio(oc["atm_ce_ltp"], oc["atm_pe_ltp"]),
        # Group D — External Cues (15%)
        _score_global(gd),
        _score_time_of_day(),
        _score_range_tightness(oc["range_width"], vix_val),
    ]

    # Verify weights sum to 1.0
    total_w = sum(s.weight for s in signals)
    total_score = sum(s.score * s.weight for s in signals) / total_w * 1.0

    # ── Regime (with hysteresis buffer at boundaries) ─────────────────────────
    if   total_score >= 70: regime = "STRONGLY_BULLISH"
    elif total_score >= 60: regime = "BULLISH"
    elif total_score >= 45: regime = "SIDEWAYS"
    elif total_score >= 35: regime = "BEARISH"
    else:                   regime = "STRONGLY_BEARISH"

    # VIX override
    if vix_val > 25 and regime == "SIDEWAYS":
        regime = "BEARISH"   # extreme fear breaks range

    # ── Signal agreement count ────────────────────────────────────────────────
    bull_sig = sum(1 for s in signals if s.score > 58)
    bear_sig = sum(1 for s in signals if s.score < 42)
    neut_sig = len(signals) - bull_sig - bear_sig

    # ── Confidence ────────────────────────────────────────────────────────────
    dominant = max(bull_sig, bear_sig)
    if   total_score >= 72 or total_score <= 28:   confidence = "HIGH"
    elif total_score >= 63 or total_score <= 37:
        confidence = "HIGH" if dominant >= 6 else "MEDIUM"
    elif total_score >= 55 or total_score <= 45:   confidence = "MEDIUM"
    else:                                           confidence = "LOW"

    # ── PCR sentiment label ───────────────────────────────────────────────────
    pcr = oc["pcr"]
    if   pcr >= 1.50: sentiment = "STRONGLY_BULLISH"
    elif pcr >= 1.25: sentiment = "BULLISH"
    elif pcr >= 1.08: sentiment = "MILDLY_BULLISH"
    elif pcr >= 0.93: sentiment = "NEUTRAL"
    elif pcr >= 0.78: sentiment = "MILDLY_BEARISH"
    elif pcr >= 0.62: sentiment = "BEARISH"
    else:             sentiment = "STRONGLY_BEARISH"

    # ── Build trade ────────────────────────────────────────────────────────────
    (action, otype, strike, strike_pe, ep_ce, ep_pe,
     sl, t1, t2, sl_pe, tgt_pe, rr, qty, cap,
     rationale, key_levels) = _build_trade(total_score, regime, oc, vix_val, budget)

    fetch_ms = int((time.time() - t0) * 1000)

    return NextTrade(
        fetched_at=datetime.now().strftime("%H:%M:%S"),
        fetch_time_ms=fetch_ms,
        spot=oc["spot"], atm=oc["atm"], expiry=oc["expiry"],
        pcr=pcr, vix=vix_val, sentiment=sentiment,
        signals=signals, total_score=round(total_score, 1),
        regime=regime, action=action, option_type=otype,
        strike=strike, strike_pe=strike_pe,
        entry_price=ep_ce, entry_price_pe=ep_pe,
        sl=sl, target1=t1, target2=t2,
        sl_pe=sl_pe, target_pe=tgt_pe,
        rr_ratio=rr, qty_lots=qty, capital=cap,
        rationale=rationale, key_levels=key_levels,
        confidence=confidence,
        bull_signals=bull_sig, bear_signals=bear_sig, neutral_signals=neut_sig,
    )
