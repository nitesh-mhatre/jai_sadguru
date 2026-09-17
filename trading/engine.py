"""
trading/engine.py
-----------------
Core data structures for live (paper) trading.

Tracks: open/closed trades, budget, P&L, SL/Target auto-hits,
loss-limit enforcement, and recovery mode.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

try:  # config is standalone — safe to import from the trading package
    from config import (BREAKEVEN_TRIGGER_PCT, TRAIL_PCT,
                        PARTIAL_BOOK_PCT, PARTIAL_BOOK_TRIGGER,
                        MAX_HOLD_MINUTES)
    _DEFAULT_BREAKEVEN_TRIGGER = BREAKEVEN_TRIGGER_PCT
    _DEFAULT_TRAIL_PCT         = TRAIL_PCT
    _DEFAULT_PARTIAL_PCT       = PARTIAL_BOOK_PCT
    _DEFAULT_PARTIAL_TRIGGER   = PARTIAL_BOOK_TRIGGER
    _DEFAULT_MAX_HOLD_MIN      = MAX_HOLD_MINUTES
except Exception:                     # pragma: no cover — defensive
    _DEFAULT_BREAKEVEN_TRIGGER = 0.50
    _DEFAULT_TRAIL_PCT         = 0.25
    _DEFAULT_PARTIAL_PCT       = 0.50
    _DEFAULT_PARTIAL_TRIGGER   = 0.50
    _DEFAULT_MAX_HOLD_MIN      = 90

# NIFTY current lot size (changed to 75 in 2024)
NIFTY_LOT_SIZE = 75

# SL stage labels (also shown in the order book)
SL_INITIAL    = "INITIAL"      # original stop-loss
SL_BREAKEVEN  = "BREAKEVEN"    # moved to entry — trade can no longer lose
SL_TRAILING   = "TRAILING"     # trailing behind the best premium — profit locked


def _hhmm_to_minutes(hhmm: str) -> int:
    """'15:15' → 915. Returns -1 when the string is not a valid HH:MM."""
    try:
        h, m = str(hhmm).strip().split(":")
        h, m = int(h), int(m)
        if 0 <= h < 24 and 0 <= m < 60:
            return h * 60 + m
    except Exception:
        pass
    return -1


# ── Trade ─────────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    id:            str
    expiry:        str
    strike:        float
    option_type:   str   # "CE" or "PE"
    action:        str   # "BUY" or "SELL"
    qty:           int   # number of lots
    entry_price:   float
    current_price: float
    sl:            float  # stop-loss trigger price
    target:        float  # profit target price
    rationale:     str = ""
    entry_time:    datetime = field(default_factory=datetime.now)
    exit_price:    Optional[float] = None
    exit_time:     Optional[datetime] = None
    status:        str = "OPEN"  # OPEN | CLOSED | SL_HIT | TARGET_HIT
    close_reason:  str = ""      # why the order was closed (shown in the order book)
    source:        str = ""      # where the order came from: /next | AI | PASSIVE

    # ── Order management: breakeven + trailing stop-loss ──────────────────────
    breakeven_trigger: float = _DEFAULT_BREAKEVEN_TRIGGER  # fraction of move to target
    trail_pct:         float = _DEFAULT_TRAIL_PCT          # trail behind best premium
    initial_sl:        Optional[float] = None  # original SL (for display/diagnostics)
    mfe_price:         Optional[float] = None  # most favourable premium reached
    sl_stage:          str = SL_INITIAL        # INITIAL | BREAKEVEN | TRAILING

    # ── Partial profit booking (T1) ───────────────────────────────────────────
    partial_pct:      float = _DEFAULT_PARTIAL_PCT      # fraction of lots booked at T1
    partial_trigger:  float = _DEFAULT_PARTIAL_TRIGGER  # fraction of entry→target move
    partial_target:   Optional[float] = None            # explicit T1 price (else derived)
    partial_booked_lots: int = 0
    partial_pnl:      float = 0.0            # realised P&L from the booked portion

    # ── Time-based exit ───────────────────────────────────────────────────────
    opened_at_ts:     float = field(default_factory=time.time)  # epoch when placed
    max_hold_minutes: Optional[int] = _DEFAULT_MAX_HOLD_MIN

    def __post_init__(self) -> None:
        if self.initial_sl is None:
            self.initial_sl = self.sl
        if self.mfe_price is None:
            self.mfe_price = self.entry_price

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def total_units(self) -> int:
        """Total option units = lots × lot size."""
        return self.qty * NIFTY_LOT_SIZE

    @property
    def remaining_lots(self) -> int:
        """Lots still open after any partial profit booking."""
        return max(0, self.qty - self.partial_booked_lots)

    @property
    def remaining_units(self) -> int:
        """Option units still open (the 'runner' after a partial book)."""
        return self.remaining_lots * NIFTY_LOT_SIZE

    @property
    def cost_basis(self) -> float:
        """Capital deployed = entry_price × total_units."""
        return self.entry_price * self.total_units

    @property
    def remaining_cost_basis(self) -> float:
        """Capital still at risk after a partial profit booking."""
        return self.entry_price * self.remaining_units

    @property
    def pnl(self) -> float:
        """P&L in ₹ — realised partial bookings + the open runner's mark."""
        price = self.exit_price if self.exit_price is not None else self.current_price
        direction   = 1.0 if self.action == "BUY" else -1.0
        unrealised  = direction * (price - self.entry_price) * self.remaining_units
        return round(self.partial_pnl + unrealised, 2)

    @property
    def pnl_pct(self) -> float:
        """P&L as % of cost basis."""
        return round(self.pnl / self.cost_basis * 100, 2) if self.cost_basis else 0.0

    @property
    def is_closed(self) -> bool:
        return self.status in ("CLOSED", "SL_HIT", "TARGET_HIT")

    # ── Price update with auto-close ──────────────────────────────────────────

    def update_price(self, new_price: float) -> bool:
        """
        Update current_price, manage breakeven/trailing stop-loss, then
        auto-close on SL or target hit. Returns True if auto-closed.
        """
        if self.is_closed:
            return False
        self.current_price = new_price

        # Track the most favourable premium reached, then protect profit
        if self.mfe_price is None:
            self.mfe_price = self.entry_price
        if self.action == "BUY":
            if new_price > self.mfe_price:
                self.mfe_price = new_price
        else:
            if new_price < self.mfe_price:
                self.mfe_price = new_price
        self._manage_stop_loss()

        # 1) Stop-loss first — a losing order must never book partial profit
        if self.action == "BUY":
            if new_price <= self.sl:
                self._close("SL_HIT", new_price, self._sl_reason())
                return True
        else:  # SELL
            if new_price >= self.sl:
                self._close("SL_HIT", new_price, self._sl_reason())
                return True

        # 2) Book partial profit at the first target (T1)
        self._maybe_book_partial(new_price)

        # 3) The final target closes the runner
        if self.action == "BUY":
            if new_price >= self.target:
                self._close("TARGET_HIT", new_price, "target hit")
                return True
        else:  # SELL
            if new_price <= self.target:
                self._close("TARGET_HIT", new_price, "target hit")
                return True
        return False

    # ── Breakeven + trailing stop management ──────────────────────────────────

    @property
    def profit_progress(self) -> float:
        """
        How far the trade has travelled from entry towards target:
        0.0 = at entry, 1.0 = at target. 0.0 when target is on the wrong side.
        """
        best = self.mfe_price if self.mfe_price is not None else self.entry_price
        if self.action == "BUY":
            denom = self.target - self.entry_price
            return max(0.0, (best - self.entry_price) / denom) if denom > 0 else 0.0
        denom = self.entry_price - self.target
        return max(0.0, (self.entry_price - best) / denom) if denom > 0 else 0.0

    @property
    def sl_moved(self) -> bool:
        """True once the stop-loss has been moved off its original level."""
        return self.initial_sl is not None and abs(self.sl - self.initial_sl) > 1e-9

    def _manage_stop_loss(self) -> None:
        """
        Two-stage capital protection, applied on every price tick:

          1. BREAKEVEN — once the trade has covered `breakeven_trigger` of the
             entry→target move, the SL is moved to entry (never backwards).
          2. TRAILING  — afterwards the SL trails `trail_pct` behind the most
             favourable premium reached, so locked-in profit can only grow.

        The SL is monotonic: for BUY it only moves up, for SELL only down.
        """
        best = self.mfe_price if self.mfe_price is not None else self.entry_price

        # 1) Move to breakeven once enough of the move to target is captured
        if self.breakeven_trigger > 0 and self.profit_progress >= self.breakeven_trigger:
            if self.action == "BUY":
                if self.entry_price > self.sl:
                    self.sl = self.entry_price
            else:
                if self.entry_price < self.sl:
                    self.sl = self.entry_price
            if self.sl_stage == SL_INITIAL:
                self.sl_stage = SL_BREAKEVEN

        # 2) Trail behind the best price (only after breakeven is active)
        if self.sl_stage in (SL_BREAKEVEN, SL_TRAILING):
            if self.action == "BUY":
                trail = best * (1.0 - self.trail_pct)
                if trail > self.sl:
                    self.sl = round(trail, 2)
                    self.sl_stage = SL_TRAILING
            else:
                trail = best * (1.0 + self.trail_pct)
                if trail < self.sl:
                    self.sl = round(trail, 2)
                    self.sl_stage = SL_TRAILING

    def _sl_reason(self) -> str:
        return {
            SL_TRAILING:  "trailing stop hit (profit locked)",
            SL_BREAKEVEN: "breakeven stop hit",
        }.get(self.sl_stage, "stop-loss hit")

    # ── Partial profit booking (T1) ───────────────────────────────────────────

    @property
    def planned_partial_lots(self) -> int:
        """
        Lots to book at T1. 0 when partial booking is off, or when the position
        is a single lot — a lot cannot be split, so 2+ lots are required.
        """
        if self.partial_pct <= 0 or self.qty < 2:
            return 0
        return max(1, min(int(math.ceil(self.qty * self.partial_pct)), self.qty - 1))

    @property
    def partial_target_price(self) -> Optional[float]:
        """T1 price: explicit `partial_target`, else derived from the trigger."""
        if self.partial_target is not None:
            return self.partial_target
        if self.action == "BUY":
            span = self.target - self.entry_price
            return round(self.entry_price + span * self.partial_trigger, 2) if span > 0 else None
        span = self.entry_price - self.target
        return round(self.entry_price - span * self.partial_trigger, 2) if span > 0 else None

    @property
    def partial_done(self) -> bool:
        return self.partial_booked_lots > 0 and self.partial_booked_lots >= self.planned_partial_lots

    def book_partial(self, price: float) -> tuple[int, float]:
        """
        Book the planned portion of the position at `price`.
        Returns (lots_booked, realised P&L from this booking).
        """
        planned = self.planned_partial_lots
        if self.is_closed or planned <= 0 or self.partial_booked_lots >= planned:
            return 0, 0.0
        lots      = planned - self.partial_booked_lots
        units     = lots * NIFTY_LOT_SIZE
        direction = 1.0 if self.action == "BUY" else -1.0
        booked    = direction * (price - self.entry_price) * units
        self.partial_booked_lots += lots
        self.partial_pnl = round(self.partial_pnl + booked, 2)
        return lots, round(booked, 2)

    def _maybe_book_partial(self, price: float) -> None:
        """Auto-book the planned partial when price reaches T1."""
        if self.partial_done:
            return
        trigger = self.partial_target_price
        if trigger is None or self.planned_partial_lots <= 0:
            return
        if self.action == "BUY" and price >= trigger:
            self.book_partial(price)
        elif self.action == "SELL" and price <= trigger:
            self.book_partial(price)

    # ── Time-based exit ───────────────────────────────────────────────────────

    def held_minutes(self, now_ts: float) -> float:
        """Minutes the order has been open (tz-free — epoch seconds)."""
        if not self.opened_at_ts:
            return 0.0
        return max(0.0, (now_ts - self.opened_at_ts) / 60.0)

    def should_time_exit(self, now_ts: float, now_hhmm: str,
                         hard_exit_hhmm: str = "") -> str:
        """
        Time-based square-off check. Returns a reason string when the order
        must be closed on time, else "".

          • hard exit — `now_hhmm` is at/after `hard_exit_hhmm` (IST, '15:15')
          • max hold  — the order has been open longer than max_hold_minutes

        Both clocks are supplied by the caller so this stays pure and testable.
        """
        if self.is_closed:
            return ""
        hard = _hhmm_to_minutes(hard_exit_hhmm)
        now  = _hhmm_to_minutes(now_hhmm)
        if hard >= 0 and now >= hard:
            return f"hard time exit {hard_exit_hhmm} IST"
        if self.max_hold_minutes and self.held_minutes(now_ts) >= self.max_hold_minutes:
            return f"max hold {self.max_hold_minutes}m reached"
        return ""

    def _close(self, status: str, price: float, reason: str = "") -> None:
        self.status      = status
        self.exit_price  = price
        self.exit_time   = datetime.now()
        if reason:
            self.close_reason = reason


# ── TradingSession ────────────────────────────────────────────────────────────

@dataclass
class TradingSession:
    """
    Holds the state for a single /go live-trading run.
    Thread-safe via caller-side locking (LiveTrader holds the lock).
    """
    budget:        float           # total capital allocated
    max_loss_pct:  float           # e.g. 10.0 → stop if 10% of budget lost
    direction:     str             # "BUY" | "SELL" | "BOTH"
    go_message:    str = ""        # original /go text for display
    trades:        list = field(default_factory=list)
    is_active:     bool  = True
    recovery_mode: bool  = False   # True → no new trades, only close open ones
    start_time:    datetime = field(default_factory=datetime.now)
    last_decision_time: Optional[datetime] = None
    decision_count: int = 0

    # ── Aggregates ────────────────────────────────────────────────────────────

    @property
    def open_trades(self) -> list:
        return [t for t in self.trades if not t.is_closed]

    @property
    def closed_trades(self) -> list:
        return [t for t in self.trades if t.is_closed]

    @property
    def realized_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def unrealized_pnl(self) -> float:
        return sum(t.pnl for t in self.open_trades)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def max_loss_amount(self) -> float:
        return self.budget * self.max_loss_pct / 100.0

    @property
    def is_loss_limit_breached(self) -> bool:
        return self.total_pnl <= -self.max_loss_amount

    @property
    def capital_deployed(self) -> float:
        # Only capital still at risk — a booked partial frees the rest up
        return sum(t.remaining_cost_basis for t in self.open_trades)

    @property
    def available_budget(self) -> float:
        # budget + any realised gains/losses − currently deployed
        effective_budget = self.budget + self.realized_pnl
        return max(0.0, effective_budget - self.capital_deployed)

    @property
    def win_trades(self) -> int:
        return sum(1 for t in self.closed_trades if t.pnl > 0)

    @property
    def loss_trades(self) -> int:
        return sum(1 for t in self.closed_trades if t.pnl <= 0)

    @property
    def win_rate(self) -> float:
        total = len(self.closed_trades)
        return round(self.win_trades / total * 100, 1) if total else 0.0

    # ── Trade lifecycle ───────────────────────────────────────────────────────

    def can_place_new_trade(self, action: str, cost: float) -> tuple[bool, str]:
        """Returns (ok, reason)."""
        # Check recovery / loss limit first
        if self.recovery_mode or self.is_loss_limit_breached:
            self.recovery_mode = True
            return False, (
                f"🔴 Recovery mode — max loss {self.max_loss_pct}% breached. "
                "Close active trades to recover capital."
            )
        # Direction constraint
        if self.direction != "BOTH" and action != self.direction:
            return False, f"Only {self.direction} trades allowed in this session."
        # Budget check
        if cost > self.available_budget:
            return False, (
                f"Insufficient budget: need ₹{cost:,.0f}, "
                f"available ₹{self.available_budget:,.0f}"
            )
        return True, ""

    def add_trade(self, trade: "Trade") -> tuple[bool, str]:
        ok, reason = self.can_place_new_trade(trade.action, trade.cost_basis)
        if not ok:
            return False, reason
        self.trades.append(trade)
        return True, f"Trade {trade.id} placed — {trade.action} {trade.qty}L {trade.strike}{trade.option_type} @ ₹{trade.entry_price}"

    def close_trade(self, trade_id: str, exit_price: float) -> tuple[bool, str]:
        """Close a specific trade. trade_id="ALL" closes everything open."""
        closed_any = False
        for t in self.open_trades:
            if trade_id.upper() == "ALL" or t.id == trade_id.upper():
                t._close("CLOSED", exit_price, "manual close")
                closed_any = True
                if trade_id.upper() != "ALL":
                    return True, f"Trade {t.id} manually closed at ₹{exit_price}"
        if closed_any:
            return True, f"All trades manually closed at ₹{exit_price}"
        return False, f"Trade '{trade_id}' not found or already closed"

    def check_and_set_recovery_mode(self) -> bool:
        """Returns True if recovery mode was just triggered."""
        if not self.recovery_mode and self.is_loss_limit_breached:
            self.recovery_mode = True
            return True
        return False
