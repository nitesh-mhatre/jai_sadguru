"""
trading/opening_momentum.py
---------------------------
Opening Momentum Strategy — 09:15 to 09:30 window

This is the highest-velocity 15 minutes of the trading day.
The strategy captures the GAP + momentum burst that happens at open.

Why this window is special:
  • All overnight news, FII activity, global cues get priced in instantly
  • Option premiums are at their highest IV of the day (IV crush follows)
  • Large institutional orders hit the book simultaneously → clean directional move
  • The FIRST 5-min candle of NIFTY sets the tone for the full day 80%+ of the time

Strategy logic:
  1. PRE-OPEN (09:00–09:15): scan GIFT Nifty + prev close to estimate gap
  2. At 09:15: wait exactly 2 minutes for the opening candle to form
  3. At 09:17: read direction from opening candle + OI wall context
  4. 09:17–09:25: enter ATM option in gap direction (tight SL, quick target)
  5. 09:30: mandatory exit regardless of P&L — window closes

Risk rules specific to this window:
  • Max 1 lot — size down because premium is highest of day
  • SL = 30% of premium (options can halve in 5 min)
  • Target = 50% of premium (achievable in a strong gap)
  • Time-based exit at 09:30 NO MATTER WHAT
  • Gap < 0.2% → skip (too small to trade cleanly)
  • Gap > 2.0% → skip (gapped too far, premiums reflect it already)

Usage:
    from trading.opening_momentum import OpeningMomentumEngine
    engine = OpeningMomentumEngine(session, dataset_logger)
    engine.run()   # blocking call, returns after 09:30
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

import pytz

from trading.engine import Trade, TradingSession, NIFTY_LOT_SIZE

log = logging.getLogger(__name__)
IST = pytz.timezone("Asia/Kolkata")


# ── Parameters ─────────────────────────────────────────────────────────────────

@dataclass
class OpeningMomentumConfig:
    min_gap_pct:     float = 0.20    # skip if gap < 0.2%
    max_gap_pct:     float = 2.00    # skip if gap > 2.0% (premium already priced)
    sl_pct:          float = 30.0    # SL = 30% of entry premium
    target_pct:      float = 50.0    # Target = 50% of entry premium
    max_lots:        int   = 1       # size down in this window
    wait_candles_sec: int  = 120     # wait 2 min (09:17) before entry
    hard_exit_time:  str  = "09:30"  # mandatory exit regardless of P&L


# ── Gap Analysis ───────────────────────────────────────────────────────────────

@dataclass
class GapAnalysis:
    prev_close:       float
    gift_nifty:       float       # 0.0 if unavailable
    estimated_open:   float       # best estimate of where NIFTY opens
    gap_points:       float
    gap_pct:          float
    direction:        str         # "UP" | "DOWN" | "FLAT"
    tradeable:        bool
    skip_reason:      str         # why skipping if not tradeable
    option_to_buy:    str         # "CE" | "PE"
    strategy_note:    str


def analyse_gap(config: OpeningMomentumConfig) -> GapAnalysis:
    """
    Estimate opening gap from GIFT Nifty and previous close.
    Called during pre-open (09:00–09:15).
    """
    prev_close = _get_prev_close()
    gift       = _get_gift_nifty()

    # Best estimate of open
    if gift > 0 and prev_close > 0:
        estimated_open = gift
        gap_points     = round(gift - prev_close, 2)
        gap_pct        = round(gap_points / prev_close * 100, 2)
    elif prev_close > 0:
        estimated_open = prev_close
        gap_points     = 0.0
        gap_pct        = 0.0
    else:
        return GapAnalysis(
            prev_close=0, gift_nifty=gift, estimated_open=0,
            gap_points=0, gap_pct=0, direction="FLAT",
            tradeable=False, skip_reason="Cannot determine previous close",
            option_to_buy="CE", strategy_note="No data available",
        )

    direction = "UP" if gap_pct > 0.05 else "DOWN" if gap_pct < -0.05 else "FLAT"

    # Tradeable checks
    abs_gap = abs(gap_pct)
    if abs_gap < config.min_gap_pct:
        return GapAnalysis(
            prev_close=prev_close, gift_nifty=gift,
            estimated_open=estimated_open,
            gap_points=gap_points, gap_pct=gap_pct, direction=direction,
            tradeable=False,
            skip_reason=f"Gap too small ({gap_pct:+.2f}%) — need ≥{config.min_gap_pct}%",
            option_to_buy="CE",
            strategy_note="Flat open expected. Wait for direction after 09:30.",
        )

    if abs_gap > config.max_gap_pct:
        return GapAnalysis(
            prev_close=prev_close, gift_nifty=gift,
            estimated_open=estimated_open,
            gap_points=gap_points, gap_pct=gap_pct, direction=direction,
            tradeable=False,
            skip_reason=f"Gap too large ({gap_pct:+.2f}%) — premium already priced in",
            option_to_buy="PE" if direction == "UP" else "CE",
            strategy_note=(
                f"Extreme gap {direction}. ATM premiums will open inflated. "
                f"Better to SELL {'PE' if direction == 'UP' else 'CE'} on gap fill "
                f"OR wait for confirmation after 09:30."
            ),
        )

    # Tradeable
    option_to_buy = "CE" if direction == "UP" else "PE"
    note = (
        f"Gap {direction} {gap_pct:+.2f}% ({gap_points:+.0f} pts). "
        f"Plan: BUY ATM {option_to_buy} at 09:17 after opening candle confirms. "
        f"SL={config.sl_pct:.0f}% of premium. Target={config.target_pct:.0f}% of premium. "
        f"HARD EXIT at {config.hard_exit_time}."
    )

    return GapAnalysis(
        prev_close=prev_close, gift_nifty=gift,
        estimated_open=estimated_open,
        gap_points=gap_points, gap_pct=gap_pct, direction=direction,
        tradeable=True, skip_reason="",
        option_to_buy=option_to_buy,
        strategy_note=note,
    )


# ── Opening Candle Reader ──────────────────────────────────────────────────────

@dataclass
class OpeningCandle:
    """First 2-minute candle of NIFTY spot (09:15–09:17)."""
    open:      float
    high:      float
    low:       float
    close:     float
    direction: str    # "BULLISH" | "BEARISH" | "DOJI"
    body_pct:  float  # candle body as % of range — doji if < 30%
    confirms:  str    # "CE" | "PE" | "NONE"


def read_opening_candle() -> Optional[OpeningCandle]:
    """
    Read the first 2-min candle of NIFTY after market opens.
    Returns None if data unavailable.
    """
    try:
        import yfinance as yf
        t    = yf.Ticker("^NSEI")
        hist = t.history(period="1d", interval="2m", auto_adjust=True)
        if hist.empty or len(hist) < 1:
            return None

        c1    = hist.iloc[0]
        open_ = float(c1["Open"])
        high  = float(c1["High"])
        low   = float(c1["Low"])
        close = float(c1["Close"])
        rng   = high - low
        body  = abs(close - open_)
        body_pct = (body / rng * 100) if rng > 0 else 0

        if body_pct < 30:
            direction = "DOJI"
            confirms  = "NONE"
        elif close > open_:
            direction = "BULLISH"
            confirms  = "CE"
        else:
            direction = "BEARISH"
            confirms  = "PE"

        return OpeningCandle(
            open=open_, high=high, low=low, close=close,
            direction=direction, body_pct=round(body_pct, 1),
            confirms=confirms,
        )
    except Exception as exc:
        log.warning("Opening candle read failed: %s", exc)
        return None


# ── Opening Momentum Engine ────────────────────────────────────────────────────

class OpeningMomentumEngine:
    """
    Runs the full 09:15–09:30 opening momentum strategy.

    Lifecycle:
      pre_scan()    → call during 09:00–09:15 to prepare gap analysis
      should_run()  → True if today is a trading day and gap is tradeable
      run()         → blocking: waits for candle, enters trade, manages it, exits by 09:30
    """

    def __init__(
        self,
        session:    TradingSession,
        dataset=None,       # DatasetLogger (optional)
        config:     OpeningMomentumConfig = None,
        log_fn=None,        # LiveTrader._log callback
    ):
        self.session = session
        self.dataset = dataset
        self.cfg     = config or OpeningMomentumConfig()
        self._log    = log_fn or (lambda msg, lvl="INFO": print(f"[{lvl}] {msg}"))
        self.gap     = Optional[GapAnalysis]
        self._trade: Optional[Trade] = None

    def pre_scan(self) -> GapAnalysis:
        """Call at 09:00–09:15. Returns gap analysis for display/logging."""
        self.gap = analyse_gap(self.cfg)
        self._log(
            f"🌅 Opening gap analysis: {self.gap.gap_pct:+.2f}%  "
            f"direction={self.gap.direction}  "
            f"tradeable={self.gap.tradeable}  "
            + (self.gap.skip_reason if not self.gap.tradeable else self.gap.strategy_note),
            "OK" if self.gap.tradeable else "WARN",
        )
        return self.gap

    def should_run(self) -> bool:
        """True if conditions met to run the opening momentum play."""
        if self.gap is None:
            self.gap = analyse_gap(self.cfg)
        return self.gap.tradeable

    def run(self, brief=None) -> None:
        """
        Full opening momentum run. Blocking — returns after 09:30 or when stopped.
        brief: latest MarketBrief from scanner (for ATM strike + expiry).
        """
        if not self.should_run():
            self._log(
                f"⏭  Opening momentum skipped: {self.gap.skip_reason}",
                "INFO",
            )
            return

        self._log("⚡ OPENING MOMENTUM: waiting 2 min for opening candle (09:17)…", "OK")
        time.sleep(self.cfg.wait_candles_sec)

        # Read opening candle
        candle = read_opening_candle()
        if candle is None:
            self._log("Opening candle unavailable — skipping momentum play", "WARN")
            return

        self._log(
            f"🕯  Opening candle: O={candle.open:.0f} H={candle.high:.0f} "
            f"L={candle.low:.0f} C={candle.close:.0f}  "
            f"{candle.direction}  body={candle.body_pct:.0f}%  confirms={candle.confirms}",
            "OK",
        )

        # Candle must confirm gap direction
        gap_side = self.gap.option_to_buy
        if candle.confirms == "NONE":
            self._log("Opening candle is DOJI — no momentum confirmed, skipping", "WARN")
            return
        if candle.confirms != gap_side:
            self._log(
                f"Candle ({candle.direction}) conflicts with gap direction ({gap_side}) — skipping",
                "WARN",
            )
            return

        # Get ATM and expiry from brief or live fetch
        expiry, atm, strike = self._resolve_strike(brief, candle)
        if not expiry or not strike:
            self._log("Cannot resolve strike/expiry — skipping", "WARN")
            return

        # Get current premium
        premium = self._get_ltp(expiry, strike, gap_side)
        if premium <= 0:
            self._log(f"Invalid premium ₹{premium} — skipping", "WARN")
            return

        sl     = round(premium * (1 - self.cfg.sl_pct     / 100), 1)
        target = round(premium * (1 + self.cfg.target_pct / 100), 1)
        qty    = min(self.cfg.max_lots, max(1, int(
            self.session.available_budget / (premium * NIFTY_LOT_SIZE)
        )))

        if qty < 1:
            self._log(
                f"Insufficient budget ₹{self.session.available_budget:,.0f} "
                f"for 1 lot at ₹{premium} — skipping",
                "WARN",
            )
            return

        # Place the trade
        trade_id = f"OM{len(self.session.trades) + 1:03d}"
        trade = Trade(
            id           = trade_id,
            expiry       = expiry,
            strike       = float(strike),
            option_type  = gap_side,
            action       = "BUY",
            qty          = qty,
            entry_price  = premium,
            current_price = premium,
            sl           = sl,
            target       = target,
            rationale    = (
                f"Opening momentum: gap {self.gap.direction} {self.gap.gap_pct:+.2f}% "
                f"confirmed by {candle.direction} opening candle "
                f"(body {candle.body_pct:.0f}%). "
                f"Hard exit 09:30. RR={self.cfg.target_pct/self.cfg.sl_pct:.1f}:1"
            ),
        )

        ok, msg = self.session.add_trade(trade)
        if not ok:
            self._log(f"Opening momentum trade rejected: {msg}", "WARN")
            return

        self._trade = trade
        self._log(
            f"⚡ OM TRADE PLACED: {trade_id} {strike}{gap_side} BUY {qty}L "
            f"@ ₹{premium}  SL=₹{sl}  TGT=₹{target}  exit by 09:30",
            "TRADE",
        )

        if self.dataset:
            self.dataset.on_trade_placed(trade, brief, cycle_num=0)

        # Monitor until 09:30 hard exit
        self._monitor_until_exit(trade, expiry, strike, gap_side, brief)

    def _monitor_until_exit(
        self, trade: Trade, expiry: str, strike: int,
        otype: str, brief
    ) -> None:
        """Poll price every 30 seconds until 09:30 or SL/Target hit."""
        hard_exit_time = self._parse_time(self.cfg.hard_exit_time)

        while True:
            now_ist = datetime.now(IST)
            if now_ist.time() >= hard_exit_time:
                # HARD EXIT
                ltp = self._get_ltp(expiry, strike, otype)
                if ltp > 0:
                    trade._close("CLOSED", ltp)
                self._log(
                    f"⏰ HARD EXIT {trade.id} @ ₹{trade.exit_price or ltp:.1f}  "
                    f"P&L: {trade.pnl:+.0f}  (09:30 time exit)",
                    "TRADE",
                )
                if self.dataset:
                    self.dataset.on_trade_closed(trade, "MANUAL", brief)
                break

            # Price check
            ltp = self._get_ltp(expiry, strike, otype)
            if ltp > 0:
                auto_closed = trade.update_price(ltp)
                self._log(
                    f"OM monitor {trade.id}: LTP=₹{ltp:.1f}  "
                    f"P&L={trade.pnl:+.0f}",
                    "OK",
                )
                if auto_closed:
                    icon = "🎯" if trade.status == "TARGET_HIT" else "🛑"
                    self._log(
                        f"{icon} {trade.status} {trade.id} @ ₹{ltp:.1f}  "
                        f"P&L={trade.pnl:+.0f}",
                        "TRADE",
                    )
                    if self.dataset:
                        self.dataset.on_trade_closed(trade, trade.status, brief)
                    break

            time.sleep(30)   # check every 30 seconds

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_strike(self, brief, candle) -> tuple:
        """Get expiry + ATM strike. Use brief if available, else fetch live."""
        try:
            if brief and brief.expiry and brief.atm:
                return brief.expiry, brief.atm, brief.atm

            from data.nifty_option_chain import get_expiry_dates, get_nifty_option_chain
            from data.yahoo_feed import get_nifty_spot
            dates    = get_expiry_dates()
            expiry   = dates[0] if dates else None
            spot     = get_nifty_spot() or candle.close
            df, _    = get_nifty_option_chain(expiry=expiry)
            strikes  = sorted(df["Strike"].astype(int).unique())
            atm      = min(strikes, key=lambda x: abs(x - spot))
            return expiry, atm, atm
        except Exception as exc:
            log.warning("_resolve_strike failed: %s", exc)
            return None, None, None

    def _get_ltp(self, expiry: str, strike: int, otype: str) -> float:
        """Get current LTP for the option from NSE chart."""
        try:
            from data.nifty_chart import get_option_chart
            df = get_option_chart(expiry, strike, otype)
            if df is not None and not df.empty:
                return round(float(df["price"].iloc[-1]), 2)
        except Exception:
            pass
        # Fallback: option chain LTP
        try:
            from data.nifty_option_chain import get_nifty_option_chain
            df, _ = get_nifty_option_chain(expiry=expiry)
            row   = df[df["Strike"] == strike]
            if not row.empty:
                return round(float(row[f"{otype}_LTP"].iloc[0]), 2)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _parse_time(t_str: str):
        """'09:30' → datetime.time(9, 30)"""
        from datetime import time as dtime
        h, m = map(int, t_str.split(":"))
        return dtime(h, m)


# ── Standalone helpers ────────────────────────────────────────────────────────

def _get_prev_close() -> float:
    """Get NIFTY previous session close from Yahoo."""
    try:
        import yfinance as yf
        t    = yf.Ticker("^NSEI")
        info = t.fast_info
        pc   = float(info.previous_close or 0)
        return round(pc, 2)
    except Exception:
        return 0.0


def _get_gift_nifty() -> float:
    """Get GIFT Nifty / NIFTY spot from Yahoo as gap proxy."""
    try:
        import yfinance as yf
        t    = yf.Ticker("^NSEI")
        info = t.fast_info
        return round(float(info.last_price or 0), 2)
    except Exception:
        return 0.0


# ── System prompt addition for opening window ─────────────────────────────────

OPENING_MOMENTUM_PROMPT = """
═══ OPENING MOMENTUM WINDOW (09:15–09:30) ═══
This is the highest-velocity 15-minute window of the trading day.

WHAT HAPPENS:
  • Overnight news, FII flows, global markets all get priced in simultaneously
  • Option premiums are at their peak IV — decay begins immediately after
  • The first 5-min candle direction holds for 60–80% of the session

STRATEGY:
  1. Pre-open (09:00–09:15): Identify gap direction from GIFT Nifty
  2. Wait for 09:17 (first 2-min candle to complete)
  3. If opening candle CONFIRMS gap direction → BUY ATM option
  4. SL = 30% of premium paid | Target = 50% of premium
  5. HARD EXIT at 09:30 regardless of P&L — window closes

SKIP IF:
  • Gap < 0.2% (no momentum)
  • Gap > 2.0% (premium already inflated)
  • Opening candle is a DOJI (no confirmation)
  • Opening candle direction CONFLICTS with gap direction
  • Budget insufficient for 1 lot at ATM premium

RISK RULES:
  • Max 1 lot only — premiums are highest, risk is real
  • This is a scalp — 15-minute trade, not an intraday position
  • Never hold past 09:30 waiting for recovery
═══════════════════════════════════════════
"""
