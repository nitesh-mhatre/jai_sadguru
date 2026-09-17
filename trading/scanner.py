"""
trading/scanner.py
------------------
MarketScanner — the new data layer between raw NSE APIs and the AI.

YOUR PRIMARY APPROACH (from your instructions):
  Step 1: Pick targeted expiry   (nearest weekly by default)
  Step 2: Get OI analysis        (spot, ATM, PCR, key levels)
  Step 3: Select targeted strikes (ATM ± 1 step, based on direction + PCR)
  Step 4: Fetch chart ONLY for those 2-3 selected strikes
  Step 5: Return compact market brief (~300 tokens, not 5000)

This replaces the old "AI calls tools one by one" approach.
AI now receives one pre-digested MarketBrief and returns one JSON block.
No tool calls needed → 1 AI round-trip per cycle instead of 4-5.
Token usage drops ~85%.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

# NIFTY strike interval
STRIKE_STEP = 50


def _safe_num(v) -> float:
    """float() that never raises and maps NaN/±Inf to 0.0."""
    try:
        f = float(v)
        return 0.0 if (f != f or abs(f) == float("inf")) else f
    except (TypeError, ValueError):
        return 0.0


# ── MarketBrief ───────────────────────────────────────────────────────────────

@dataclass
class StrikeBrief:
    """Compact data for one targeted strike."""
    strike:      int
    option_type: str          # CE or PE
    ltp:         float        # current premium
    prev_close:  float
    high:        float
    low:         float
    oi:          int
    chng_oi:     int          # +ve = fresh writing, -ve = unwinding
    iv:          float
    volume:      int
    price_chng_pct: float
    oi_buildup:  str          # "BUILDING" | "UNWINDING" | "STABLE"
    trend:       str          # "UP" | "DOWN" | "FLAT"
    delta:       float = 0.0  # option greeks (Groww chain only)
    theta:       float = 0.0


@dataclass
class MarketBrief:
    """
    Everything the AI needs in one compact object.
    Total size when serialised ≈ 400-600 chars (vs 5000+ with old approach).
    """
    fetched_at:  str
    expiry:      str
    spot:        float
    atm:         int
    pcr:         float
    sentiment:   str          # "BULLISH" | "NEUTRAL" | "BEARISH"
    max_pain:    int
    vix:         float        # 0.0 if unavailable
    # Key OI walls
    resistance:  list[dict]   # top 2 CE OI strikes
    support:     list[dict]   # top 2 PE OI strikes
    # Fresh writing (smart money signal)
    fresh_ce_writing: list[dict]
    fresh_pe_writing: list[dict]
    # Targeted strikes with chart data
    targeted:    list[StrikeBrief] = field(default_factory=list)
    # Scan metadata
    scan_duration_ms: int = 0
    error:       Optional[str] = None
    source:      str = "nse"     # "groww" | "nse" — where the option data came from
    chain_age_s: float = -1.0    # age of the Groww chain snapshot, if used

    def to_ai_prompt(self, direction: str, budget: float, max_loss_pct: float) -> str:
        """
        Render a compact, dense prompt block for the AI.
        No fluff. Every word is signal.
        """
        lines = [
            f"=== MARKET BRIEF {self.fetched_at}  [source={self.source}] ===",
            f"NIFTY spot={self.spot}  ATM={self.atm}  expiry={self.expiry}",
            f"PCR={self.pcr}  sentiment={self.sentiment}  MaxPain={self.max_pain}  VIX={self.vix}",
        ]

        # Quick sideways hint based on PCR alone
        if self.pcr:
            range_wall_diff = 0
            if self.resistance and self.support:
                range_wall_diff = self.resistance[0]["strike"] - self.support[0]["strike"]
            if 0.85 <= self.pcr <= 1.15 and self.vix < 17:
                side_note = (
                    f"⚠ PCR={self.pcr} in range-bound zone + VIX={self.vix} low → "
                    f"LIKELY SIDEWAYS. Consider selling premium (strangle/condor), NOT buying."
                )
                if range_wall_diff and range_wall_diff < 300:
                    side_note += f" Range width only {range_wall_diff:.0f}pts."
                lines.append(side_note)

        lines += [
            "",
            "OI WALLS (resistance=CE, support=PE):",
        ]
        for r in self.resistance:
            lines.append(f"  CE wall @ {r['strike']}  OI={r['ce_oi']:,}  Δ={r['chng_oi']:+,}  LTP=₹{r['ltp']}")
        for s in self.support:
            lines.append(f"  PE wall @ {s['strike']}  OI={s['pe_oi']:,}  Δ={s['chng_oi']:+,}  LTP=₹{s['ltp']}")

        if self.fresh_ce_writing:
            lines.append(f"Fresh CE writing (bearish signal): " +
                         ", ".join(f"{x['strike']}(+{x['added_oi']:,})" for x in self.fresh_ce_writing))
        if self.fresh_pe_writing:
            lines.append(f"Fresh PE writing (bullish signal): " +
                         ", ".join(f"{x['strike']}(+{x['added_oi']:,})" for x in self.fresh_pe_writing))

        lines.append("")
        lines.append("TARGETED STRIKE CHARTS:")
        for t in self.targeted:
            oi_tag = {"BUILDING": "↑OI", "UNWINDING": "↓OI", "STABLE": "=OI"}.get(t.oi_buildup, "")
            greek = f"  Δ={t.delta:+.2f} Θ={t.theta:+.2f}" if (t.delta or t.theta) else ""
            lines.append(
                f"  {t.strike}{t.option_type}  LTP=₹{t.ltp}  prev_close=₹{t.prev_close}"
                f"  H={t.high} L={t.low}  chng={t.price_chng_pct:+.1f}%"
                f"  IV={t.iv}{greek}  OI={t.oi:,}  {oi_tag}  trend={t.trend}"
            )

        lines += [
            "",
            f"SESSION CONSTRAINTS:",
            f"  direction_allowed={direction}  budget=₹{budget:,.0f}  max_loss={max_loss_pct}%",
            f"  lot_size=75  capital_per_lot=entry_price×75",
        ]

        return "\n".join(lines)


# ── MarketScanner ─────────────────────────────────────────────────────────────

class MarketScanner:
    """
    Orchestrates targeted data collection with smart caching.

    Per-cycle decision logic:
      Step 2 (option chain)  → always fresh — spot/OI changes every cycle
      Step 3 (strike select) → reuse previous targets if spot moved < 50pts
      Step 4 (chart fetch)   → skip if same strike fetched < CHART_CACHE_TTL seconds ago,
                               EXCEPT for open position strikes (always refresh those)

    Typical savings after cycle 1:
      Range-bound market  → Step 3 skipped ~70% of cycles
      Chart cache hits    → Step 4 skipped ~50% of time per strike
    """

    SPOT_MOVE_THRESHOLD = 50      # points — reselect strikes if spot moves this much
    CHART_CACHE_TTL     = 4 * 60  # seconds — skip chart re-fetch within this window

    def __init__(self, direction: str = "BOTH"):
        self.direction = direction

        # ── Cache state (persists across cycles) ──────────────────────────────
        self._last_spot:    float = 0.0
        self._last_targets: list[tuple[int, str]] = []
        self._chart_cache:  dict[tuple, dict] = {}       # (expiry,strike,otype) → {brief, fetched_at}
        self._last_expiry:  str = ""
        self._cycle:        int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        expiry: Optional[str] = None,
        open_strikes: Optional[list[tuple[int, str]]] = None,  # open position strikes
    ) -> MarketBrief:
        """
        Smart scan:
          - Always fetches fresh option chain (Step 2)
          - Reuses strike selection if spot hasn't moved enough (Step 3)
          - Reuses chart data from cache if fresh enough (Step 4)
          - Always refreshes chart for open positions

        open_strikes: list of (strike, option_type) for currently open trades
                      so their CMPs are always updated regardless of cache.
        """
        self._cycle += 1
        start = datetime.now()

        try:
            brief = self._scan_internal(expiry, open_strikes or [])
        except Exception as exc:
            log.error("MarketScanner.scan failed: %s", exc, exc_info=True)
            brief = MarketBrief(
                fetched_at=datetime.now().strftime("%H:%M:%S"),
                expiry=expiry or "N/A",
                spot=0.0, atm=0, pcr=0.0, sentiment="UNKNOWN",
                max_pain=0, vix=0.0,
                resistance=[], support=[],
                fresh_ce_writing=[], fresh_pe_writing=[],
                error=str(exc),
            )

        brief.scan_duration_ms = int((datetime.now() - start).total_seconds() * 1000)
        return brief

    # ── Internals ─────────────────────────────────────────────────────────────

    def _scan_internal(
        self, expiry: Optional[str], open_strikes: list[tuple[int, str]]
    ) -> MarketBrief:
        from data.nifty_option_chain import (
            get_nifty_option_chain,
            get_expiry_dates,
        )
        from data.yahoo_feed import get_nifty_spot, get_india_vix

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1 — Groww option chain (PRIMARY source)
        # data/groww_feed.py serves live LTP / OI / IV / delta / theta per strike
        # with no auth. Used whenever Groww is serving the expiry we asked for;
        # the NSE path below stays as the fallback.
        # ══════════════════════════════════════════════════════════════════════
        try:
            groww_brief = self._scan_from_groww(expiry, open_strikes)
            if groww_brief is not None:
                return groww_brief
        except Exception as exc:
            log.warning("Groww scan failed (%s) — using NSE path", exc)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2 — Option chain fetch   ← ALWAYS runs every cycle, no cache
        # Source: NSE India (OI, strikes, expiry) — fallback when Groww has none
        # Spot  : Yahoo Finance ^NSEI (overrides NSE underlyingValue)
        # VIX   : Yahoo Finance ^INDIAVIX
        # ══════════════════════════════════════════════════════════════════════
        if expiry is None:
            dates  = get_expiry_dates()
            expiry = dates[0] if dates else None

        log.info("Cycle %d: Fetching option chain expiry=%s", self._cycle, expiry)
        df, spot_nse = get_nifty_option_chain(expiry=expiry)

        # ── Guard: must have data before touching any column ──────────────────
        if df is None or df.empty:
            raise ValueError(
                "Empty option chain from NSE — market may be closed or session needs refresh."
            )

        # ── Normalize column names defensively ────────────────────────────────
        # NSE sometimes returns strikePrice instead of Strike
        col_map = {"strikePrice": "Strike", "strikeprice": "Strike", "strike": "Strike"}
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Ensure all expected columns exist
        for col in ["Strike","Expiry","CE_OI","CE_Chng_OI","CE_Volume","CE_IV","CE_LTP",
                    "PE_OI","PE_Chng_OI","PE_Volume","PE_IV","PE_LTP"]:
            if col not in df.columns:
                log.warning("Column %s missing — filling with 0", col)
                df[col] = 0

        if "Strike" not in df.columns or df["Strike"].isna().all():
            raise ValueError(
                f"Strike column missing. Available: {list(df.columns)}"
            )

        # ── Sanitize NaN / ±Inf ───────────────────────────────────────────────
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = (
            df[numeric_cols]
            .replace([float("inf"), float("-inf")], 0)
            .fillna(0)
        )
        df["Strike"] = df["Strike"].astype(int)

        # Prefer Yahoo spot; fall back to NSE if Yahoo returns 0
        spot_yf = get_nifty_spot()
        spot    = spot_yf if spot_yf > 0 else spot_nse

        if spot <= 0:
            raise ValueError(
                f"Invalid spot ({spot}) — check NSE + Yahoo connectivity."
            )

        # ── ATM from ACTUAL available strikes (not assumed 50-pt grid) ────────
        # Bug was: _round_to_strike(24039.75)=24050 but 24050 may not exist
        # Fix: pick closest real strike from the chain
        strikes = sorted(df["Strike"].astype(int).unique().tolist())
        atm     = min(strikes, key=lambda x: abs(x - spot))

        log.info(
            "Cycle %d: spot=%.2f  spot_nse=%.2f  atm=%d  strikes_in_chain=%d",
            self._cycle, spot, spot_nse, atm, len(strikes),
        )

        ce_oi_total = int(df["CE_OI"].sum())
        pe_oi_total = int(df["PE_OI"].sum())
        pcr         = round(pe_oi_total / ce_oi_total, 2) if ce_oi_total else 0

        sentiment  = self._pcr_to_sentiment(pcr)
        max_pain   = self._calc_max_pain(df)
        expiry_str = str(df["Expiry"].iloc[0]) if not df.empty else (expiry or "N/A")
        vix        = self._get_vix()

        # ── Invalidate chart cache if expiry rolled over ───────────────────────
        if self._last_expiry and self._last_expiry != expiry_str:
            log.info(
                "Expiry changed %s → %s — clearing chart cache",
                self._last_expiry, expiry_str,
            )
            self._chart_cache.clear()
            self._last_targets = []   # force strike reselect on new expiry
        self._last_expiry = expiry_str

        # Safe int conversion helper — guards against any residual inf/nan
        def _safe_int(v) -> int:
            try:
                f = float(v)
                if f != f or abs(f) == float("inf"):   # nan or inf check
                    return 0
                return int(f)
            except (TypeError, ValueError):
                return 0

        def _safe_float(v) -> float:
            try:
                f = float(v)
                return 0.0 if (f != f or abs(f) == float("inf")) else round(f, 2)
            except (TypeError, ValueError):
                return 0.0

        top_ce = df.nlargest(2, "CE_OI")[["Strike","CE_OI","CE_Chng_OI","CE_LTP"]].to_dict("records")
        top_pe = df.nlargest(2, "PE_OI")[["Strike","PE_OI","PE_Chng_OI","PE_LTP"]].to_dict("records")

        resistance = [{"strike": _safe_int(r["Strike"]), "ce_oi": _safe_int(r["CE_OI"]),
                       "chng_oi": _safe_int(r["CE_Chng_OI"]), "ltp": _safe_float(r["CE_LTP"])}
                      for r in top_ce]
        support    = [{"strike": _safe_int(r["Strike"]), "pe_oi": _safe_int(r["PE_OI"]),
                       "chng_oi": _safe_int(r["PE_Chng_OI"]), "ltp": _safe_float(r["PE_LTP"])}
                      for r in top_pe]

        fresh_ce = (df[df["CE_Chng_OI"] > 0]
                    .nlargest(2, "CE_Chng_OI")[["Strike","CE_Chng_OI"]]
                    .to_dict("records"))
        fresh_pe = (df[df["PE_Chng_OI"] > 0]
                    .nlargest(2, "PE_Chng_OI")[["Strike","PE_Chng_OI"]]
                    .to_dict("records"))

        fresh_ce_writing = [{"strike": _safe_int(r["Strike"]), "added_oi": _safe_int(r["CE_Chng_OI"])} for r in fresh_ce]
        fresh_pe_writing = [{"strike": _safe_int(r["Strike"]), "added_oi": _safe_int(r["PE_Chng_OI"])} for r in fresh_pe]

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3 — Strike selection  ← CONDITIONAL, uses REAL strikes from chain
        # ══════════════════════════════════════════════════════════════════════
        spot_moved = abs(spot - self._last_spot) >= self.SPOT_MOVE_THRESHOLD
        is_first   = self._cycle == 1

        if is_first or spot_moved or not self._last_targets:
            targets = self._select_target_strikes(spot, atm, strikes, sentiment, df)
            self._last_targets = targets
            self._last_spot    = spot
            log.info("Cycle %d: Strike reselect → %s  (spot=%.0f moved=%.0f pts)",
                     self._cycle, targets, spot, abs(spot - self._last_spot))
        else:
            targets = self._last_targets
            log.info("Cycle %d: Reusing strikes %s (spot Δ=%.0f < %d threshold)",
                     self._cycle, targets, abs(spot - self._last_spot), self.SPOT_MOVE_THRESHOLD)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 4 — Chart fetch  ← cache keyed on (expiry, strike, otype)
        # ══════════════════════════════════════════════════════════════════════
        open_strike_set = set(open_strikes)
        targeted_briefs = []
        now_ts          = datetime.now().timestamp()

        for (strike, otype) in targets:
            cache_key   = (expiry_str, strike, otype)   # expiry-aware — prevents stale chart after rollover
            is_open_pos = (strike, otype) in open_strike_set
            cached      = self._chart_cache.get(cache_key)
            cache_age   = now_ts - cached["fetched_at"] if cached else float("inf")
            cache_fresh = cache_age < self.CHART_CACHE_TTL

            if is_open_pos or not cache_fresh:
                reason = "open pos" if is_open_pos else (
                    "first" if is_first else f"expired {int(cache_age)}s")
                log.info("Cycle %d: Fetching chart %s%s@%s (%s)",
                         self._cycle, strike, otype, expiry_str, reason)
                sb = self._get_strike_brief(expiry_str, strike, otype, df)
                if sb:
                    self._chart_cache[cache_key] = {"brief": sb, "fetched_at": now_ts}
                    targeted_briefs.append(sb)
                    log.info("Cycle %d: Chart OK %s%s ltp=%.2f trend=%s oi_buildup=%s",
                             self._cycle, strike, otype, sb.ltp, sb.trend, sb.oi_buildup)
                else:
                    log.warning("Cycle %d: No chart data for %s%s", self._cycle, strike, otype)
            else:
                sb = cached["brief"]
                # Refresh LTP from already-fetched option chain (free — no API call)
                row = df[df["Strike"] == strike]
                if not row.empty:
                    ltp_col = f"{otype}_LTP"
                    if ltp_col in row.columns:
                        new_ltp = _safe_float(row[ltp_col].iloc[0])
                        if new_ltp > 0:
                            sb.ltp = new_ltp
                log.info("Cycle %d: Cache hit %s%s (age=%ds ltp=%.2f)",
                         self._cycle, strike, otype, int(cache_age), sb.ltp)
                targeted_briefs.append(sb)

        # Open-position strikes not in targets — always refresh
        for (strike, otype) in open_strikes:
            if (strike, otype) not in set(targets):
                cache_key = (expiry_str, strike, otype)
                sb = self._get_strike_brief(expiry_str, strike, otype, df)
                if sb:
                    self._chart_cache[cache_key] = {"brief": sb, "fetched_at": now_ts}
                    targeted_briefs.append(sb)

        log.info("Cycle %d DONE: %d briefs — %s",
                 self._cycle,
                 len(targeted_briefs),
                 [(b.strike, b.option_type, b.ltp) for b in targeted_briefs])

        return MarketBrief(
            fetched_at       = datetime.now().strftime("%H:%M:%S"),
            expiry           = expiry_str,
            spot             = round(spot, 2),
            atm              = atm,
            pcr              = pcr,
            sentiment        = sentiment,
            max_pain         = max_pain,
            vix              = vix,
            resistance       = resistance,
            support          = support,
            fresh_ce_writing = fresh_ce_writing,
            fresh_pe_writing = fresh_pe_writing,
            targeted         = targeted_briefs,
        )

    # ── Groww path ────────────────────────────────────────────────────────────

    def _scan_from_groww(
        self, expiry: Optional[str], open_strikes: list[tuple[int, str]]
    ) -> Optional[MarketBrief]:
        """
        Build a MarketBrief straight from the Groww chain (live LTP/OI/IV/greeks).

        Returns None (so the caller falls back to NSE) when:
          • Groww has no chain right now, or
          • the user selected an expiry different from the one Groww serves.
        """
        from data.groww_feed import get_chain, chain_age_seconds, _norm_expiry
        from data.yahoo_feed import get_nifty_spot

        oc = get_chain()
        if not oc or not oc.get("rows"):
            return None

        groww_expiry = _norm_expiry(oc.get("current_expiry", ""))
        if expiry and groww_expiry and _norm_expiry(expiry) != groww_expiry:
            log.info(
                "Groww chain serves %s but %s was selected — NSE path",
                groww_expiry, expiry,
            )
            return None
        expiry_str = groww_expiry or (expiry or "N/A")

        rows    = sorted(oc["rows"], key=lambda r: int(r["strike"]))
        strikes = [int(r["strike"]) for r in rows]

        # Spot: live index quote preferred, else Groww's parity estimate
        spot = 0.0
        try:
            spot = float(get_nifty_spot() or 0.0)
        except Exception:
            spot = 0.0
        if spot <= 0:
            spot = float(oc.get("spot") or 0.0)
        if spot <= 0:
            spot = float(oc.get("atm") or (strikes[len(strikes) // 2] if strikes else 0))
            if spot <= 0:
                return None

        atm = int(oc.get("atm") or min(strikes, key=lambda x: abs(x - spot)))
        pcr = float(oc.get("pcr") or 0.0)
        sentiment = self._pcr_to_sentiment(pcr)

        def _leg(r, side):
            return r.get(side.lower(), {}) or {}

        def _oi(r, side):
            return _safe_num(_leg(r, side).get("oi"))

        # OI walls
        top_ce = sorted(rows, key=lambda r: _oi(r, "CE"), reverse=True)[:2]
        top_pe = sorted(rows, key=lambda r: _oi(r, "PE"), reverse=True)[:2]

        def _wall(r, side):
            leg = _leg(r, side)
            return {
                "strike":  int(r["strike"]),
                f"{side.lower()}_oi": int(_safe_num(leg.get("oi"))),
                "chng_oi": int(_safe_num(leg.get("oi")) - _safe_num(leg.get("prev_oi"))),
                "ltp":     round(_safe_num(leg.get("ltp")), 2),
            }

        resistance = [_wall(r, "CE") for r in top_ce]
        support    = [_wall(r, "PE") for r in top_pe]

        def _fresh(side):
            pos = []
            for r in rows:
                leg = _leg(r, side)
                add = _safe_num(leg.get("oi")) - _safe_num(leg.get("prev_oi"))
                if add > 0:
                    pos.append({"strike": int(r["strike"]), "added_oi": int(add)})
            return sorted(pos, key=lambda x: x["added_oi"], reverse=True)[:2]

        targets = self._select_target_strikes(spot, atm, strikes, sentiment, None)
        wanted  = list(dict.fromkeys(list(targets) + list(open_strikes)))

        targeted: list[StrikeBrief] = []
        for strike, otype in wanted:
            row = next((r for r in rows if int(r["strike"]) == int(strike)), None)
            if row is None:
                continue
            targeted.append(self._strike_brief_groww(row, int(strike), otype))

        log.info(
            "Groww scan OK: spot=%.2f atm=%d pcr=%.2f strikes=%d targets=%s age=%ss",
            spot, atm, pcr, len(strikes), wanted, chain_age_seconds(),
        )

        return MarketBrief(
            fetched_at       = datetime.now().strftime("%H:%M:%S"),
            expiry           = expiry_str,
            spot             = round(spot, 2),
            atm              = atm,
            pcr              = pcr,
            sentiment        = sentiment,
            max_pain         = int(oc.get("max_pain") or 0),
            vix              = self._get_vix(),
            resistance       = resistance,
            support          = support,
            fresh_ce_writing = _fresh("CE"),
            fresh_pe_writing = _fresh("PE"),
            targeted         = targeted,
            source           = "groww",
            chain_age_s      = chain_age_seconds(),
        )

    def _strike_brief_groww(self, row: dict, strike: int, otype: str) -> StrikeBrief:
        """StrikeBrief from one Groww chain row (no chart API call needed)."""
        leg      = row.get(otype.lower(), {}) or {}
        ltp      = _safe_num(leg.get("ltp"))
        close    = _safe_num(leg.get("close"))
        oi       = _safe_num(leg.get("oi"))
        prev_oi  = _safe_num(leg.get("prev_oi"))
        change   = _safe_num(leg.get("change"))
        chng_oi  = oi - prev_oi
        pct      = round(change / close * 100, 2) if close else 0.0

        base = max(abs(prev_oi), 1)
        if   chng_oi >  base * 0.05: oi_buildup = "BUILDING"
        elif chng_oi < -base * 0.05: oi_buildup = "UNWINDING"
        else:                        oi_buildup = "STABLE"

        trend = "UP" if change > 0 else ("DOWN" if change < 0 else "FLAT")
        hi = max(ltp, close) if (ltp and close) else ltp
        lo = min(ltp, close) if (ltp and close) else ltp

        return StrikeBrief(
            strike         = strike,
            option_type    = otype,
            ltp            = round(ltp, 2),
            prev_close     = round(close, 2),
            high           = round(hi, 2),
            low            = round(lo, 2),
            oi             = int(oi),
            chng_oi        = int(chng_oi),
            iv             = round(_safe_num(leg.get("iv")), 2),
            volume         = 0,
            price_chng_pct = pct,
            oi_buildup     = oi_buildup,
            trend          = trend,
            delta          = round(_safe_num(leg.get("delta")), 3),
            theta          = round(_safe_num(leg.get("theta")), 2),
        )

    def _select_target_strikes(
        self, spot: float, atm: int, strikes: list,
        sentiment: str, df
    ) -> list[tuple[int, str]]:
        """
        Pick 2-3 specific strikes from REAL available strikes in the chain.
        Uses nearest() helper so we never miss because of assumed 50-pt grid.
        """
        # Helper: nearest real strike above/below ATM
        strikes_above = [s for s in strikes if s >= atm]
        strikes_below = [s for s in strikes if s <= atm]

        def nearest_above(n: int = 1) -> int | None:
            """n-th strike above ATM (n=1 → first OTM CE)"""
            return strikes_above[n] if len(strikes_above) > n else None

        def nearest_below(n: int = 1) -> int | None:
            """n-th strike below ATM (n=1 → first OTM PE)"""
            idx = len(strikes_below) - 1 - n
            return strikes_below[idx] if idx >= 0 else None

        targets = []

        bullish  = "BULLISH" in sentiment
        bearish  = "BEARISH" in sentiment
        sideways = sentiment in ("MILDLY_BULLISH", "MILDLY_BEARISH")

        # ── Sideways: fetch both OTM sides for strangle/condor analysis ───────
        if sideways:
            ce_s = nearest_above(1)
            pe_s = nearest_below(1)
            if ce_s and ce_s in strikes: targets.append((ce_s, "CE"))
            if pe_s and pe_s in strikes: targets.append((pe_s, "PE"))
            log.info("Sideways regime: strangle strikes CE=%s PE=%s", ce_s, pe_s)

        elif self.direction == "BUY":
            if bullish:
                for s in [atm, nearest_above(1)]:
                    if s and s in strikes: targets.append((s, "CE"))
            else:
                for s in [atm, nearest_below(1)]:
                    if s and s in strikes: targets.append((s, "PE"))

        elif self.direction == "SELL":
            if bullish:
                for s in [nearest_below(1), nearest_below(2)]:
                    if s and s in strikes: targets.append((s, "PE"))
            else:
                for s in [nearest_above(1), nearest_above(2)]:
                    if s and s in strikes: targets.append((s, "CE"))

        else:  # BOTH
            if bullish:
                if atm in strikes: targets.append((atm, "CE"))
                pe_s = nearest_below(1)
                if pe_s and pe_s in strikes: targets.append((pe_s, "PE"))
            else:
                if atm in strikes: targets.append((atm, "PE"))
                ce_s = nearest_above(1)
                if ce_s and ce_s in strikes: targets.append((ce_s, "CE"))

        # Fallback — always get at least one strike
        if not targets:
            otype = "CE" if "BULLISH" in sentiment else "PE"
            if atm in strikes:
                targets.append((atm, otype))
            elif strikes:
                targets.append((strikes[len(strikes)//2], otype))

        # Deduplicate, cap at 3
        seen, unique = set(), []
        for t in targets:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        log.info("_select_target_strikes: atm=%d sentiment=%s direction=%s → %s",
                 atm, sentiment, self.direction, unique[:3])
        return unique[:3]

    def _get_strike_brief(
        self, expiry: str, strike: int, otype: str, df
    ) -> Optional[StrikeBrief]:
        """Fetch chart for one specific strike and return StrikeBrief."""
        try:
            from data.nifty_chart import get_option_chart

            # Get OI data from option chain (already fetched, no API call)
            row = df[df["Strike"] == strike]
            if row.empty:
                oi, chng_oi, iv, ltp_oc = 0, 0, 0.0, 0.0
            else:
                col_oi  = f"{otype}_OI"
                col_coi = f"{otype}_Chng_OI"
                col_iv  = f"{otype}_IV"
                col_ltp = f"{otype}_LTP"
                oi      = int(row[col_oi].iloc[0])   if col_oi  in row.columns else 0
                chng_oi = int(row[col_coi].iloc[0])  if col_coi in row.columns else 0
                iv      = float(row[col_iv].iloc[0]) if col_iv  in row.columns else 0.0
                ltp_oc  = float(row[col_ltp].iloc[0]) if col_ltp in row.columns else 0.0

            # OI trend classification
            if   chng_oi > oi * 0.05:   oi_buildup = "BUILDING"
            elif chng_oi < -oi * 0.05:  oi_buildup = "UNWINDING"
            else:                        oi_buildup = "STABLE"

            # Chart data for this specific strike
            chart_df = get_option_chart(expiry, strike, otype)

            if chart_df is None or chart_df.empty:
                # Use option chain LTP if chart unavailable
                return StrikeBrief(
                    strike=strike, option_type=otype,
                    ltp=ltp_oc, prev_close=0.0,
                    high=0.0, low=0.0,
                    oi=oi, chng_oi=chng_oi, iv=iv, volume=0,
                    price_chng_pct=0.0,
                    oi_buildup=oi_buildup, trend="UNKNOWN",
                )

            ltp        = float(chart_df["price"].iloc[-1])
            open_price = float(chart_df["price"].iloc[0])
            high       = float(chart_df["price"].max())
            low        = float(chart_df["price"].min())
            volume     = int(chart_df["volume"].sum())
            prev_close = float(chart_df["close_price"].iloc[0]) if "close_price" in chart_df.columns else 0.0
            pct_chng   = round((ltp - open_price) / open_price * 100, 2) if open_price else 0.0

            # Simple trend from first/last 20% of candles
            n = len(chart_df)
            if n >= 5:
                early_avg = chart_df["price"].iloc[:n//5].mean()
                late_avg  = chart_df["price"].iloc[-n//5:].mean()
                if   late_avg > early_avg * 1.005:  trend = "UP"
                elif late_avg < early_avg * 0.995:  trend = "DOWN"
                else:                               trend = "FLAT"
            else:
                trend = "FLAT"

            return StrikeBrief(
                strike=strike, option_type=otype,
                ltp=round(ltp, 2), prev_close=round(prev_close, 2),
                high=round(high, 2), low=round(low, 2),
                oi=oi, chng_oi=chng_oi, iv=round(iv, 2), volume=volume,
                price_chng_pct=pct_chng,
                oi_buildup=oi_buildup, trend=trend,
            )

        except Exception as exc:
            log.warning("get_strike_brief failed for %s%s: %s", strike, otype, exc)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _round_to_strike(spot: float) -> int:
        """Round spot to nearest NIFTY strike (multiples of 50)."""
        return int(round(spot / STRIKE_STEP) * STRIKE_STEP)

    @staticmethod
    def _pcr_to_sentiment(pcr: float) -> str:
        """Granular mapping — eliminates the double-NEUTRAL dead zone."""
        if   pcr >= 1.50: return "STRONGLY_BULLISH"
        elif pcr >= 1.20: return "BULLISH"
        elif pcr >= 1.00: return "MILDLY_BULLISH"
        elif pcr >= 0.85: return "MILDLY_BEARISH"
        elif pcr >= 0.65: return "BEARISH"
        else:             return "STRONGLY_BEARISH"

    @staticmethod
    def _get_vix() -> float:
        """Fetch India VIX from Yahoo Finance (^INDIAVIX)."""
        try:
            from data.yahoo_feed import get_india_vix
            return get_india_vix()
        except Exception:
            return 0.0

    @staticmethod
    def _calc_max_pain(df) -> int:
        """Standard max pain calculation from OI data."""
        try:
            strikes = sorted(df["Strike"].unique())
            losses  = {}
            for s in strikes:
                ce_loss = df[df["Strike"] < s]["CE_OI"].multiply(
                    df[df["Strike"] < s]["Strike"].apply(lambda k: s - k)
                ).sum()
                pe_loss = df[df["Strike"] > s]["PE_OI"].multiply(
                    df[df["Strike"] > s]["Strike"].apply(lambda k: k - s)
                ).sum()
                losses[s] = ce_loss + pe_loss
            return int(min(losses, key=losses.get))
        except Exception:
            return 0
