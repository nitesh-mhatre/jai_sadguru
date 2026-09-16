"""
trading/engine.py
-----------------
Core data structures for live (paper) trading.

Tracks: open/closed trades, budget, P&L, SL/Target auto-hits,
loss-limit enforcement, and recovery mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# NIFTY current lot size (changed to 75 in 2024)
NIFTY_LOT_SIZE = 75


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

    # ── Derived ───────────────────────────────────────────────────────────────

    @property
    def total_units(self) -> int:
        """Total option units = lots × lot size."""
        return self.qty * NIFTY_LOT_SIZE

    @property
    def cost_basis(self) -> float:
        """Capital deployed = entry_price × total_units."""
        return self.entry_price * self.total_units

    @property
    def pnl(self) -> float:
        """Realised or unrealised P&L in ₹."""
        price = self.exit_price if self.exit_price is not None else self.current_price
        if self.action == "BUY":
            return (price - self.entry_price) * self.total_units
        else:
            return (self.entry_price - price) * self.total_units

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
        Update current_price and auto-close on SL/Target hit.
        Returns True if trade was auto-closed.
        """
        if self.is_closed:
            return False
        self.current_price = new_price

        if self.action == "BUY":
            if new_price <= self.sl:
                self._close("SL_HIT", new_price)
                return True
            if new_price >= self.target:
                self._close("TARGET_HIT", new_price)
                return True
        else:  # SELL
            if new_price >= self.sl:
                self._close("SL_HIT", new_price)
                return True
            if new_price <= self.target:
                self._close("TARGET_HIT", new_price)
                return True
        return False

    def _close(self, status: str, price: float) -> None:
        self.status      = status
        self.exit_price  = price
        self.exit_time   = datetime.now()


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
        return sum(t.cost_basis for t in self.open_trades)

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
                t._close("CLOSED", exit_price)
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
