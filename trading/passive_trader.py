"""
trading/passive_trader.py
-------------------------
Passive Trader — executes PassivePlan, monitors trades, refreshes plan async.

Architecture:
  Thread 1 — main loop
    ├── Check entry conditions for pending planned trades
    ├── Monitor open trades (price refresh every 30s)
    ├── SL / Target auto-close
    └── Market hours gate

  Thread 2 — async plan refresher (never blocks Thread 1)
    ├── Triggered by: time interval OR price move OR trade event
    ├── Calls PassivePlanner.refresh() with fresh LLM analysis
    ├── Updates self._plan atomically (thread-safe swap)
    └── Pending unexecuted trades updated; open trades NEVER touched

  Main thread — Rich Live dashboard
    └── Shows: plan version, all planned trades status, P&L, logs

Key design: TRADES NEVER STOP during a plan refresh.
The refresh only affects which NEW trades get entered next.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pytz
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trading.engine import Trade, TradingSession, NIFTY_LOT_SIZE
from trading.passive_planner import PassivePlan, PlannedTrade, PassivePlanner

IST = pytz.timezone("Asia/Kolkata")

# Status for each planned trade
STATUS_PENDING   = "PENDING"    # waiting for condition
STATUS_ENTERED   = "ENTERED"    # trade placed in session
STATUS_SKIPPED   = "SKIPPED"    # condition never met / budget
STATUS_CANCELLED = "CANCELLED"  # removed by plan refresh


@dataclass
class PlannedTradeState:
    planned:    PlannedTrade
    status:     str = STATUS_PENDING
    trade_id:   str = ""          # session Trade.id once entered
    entered_at: str = ""
    ltp:        float = 0.0       # current LTP if entered


class PassiveTrader:
    """
    Runs the passive trading engine.

    Usage:
        pt = PassiveTrader(plan, planner, session, dataset, interval_seconds)
        pt.start()
        # Rich Live dashboard loop
        pt.stop()
    """

    PRICE_POLL_SEC   = 30     # seconds between price refreshes
    PLAN_REFRESH_SEC = 900    # 15 minutes between async plan refreshes

    def __init__(
        self,
        plan:      PassivePlan,
        planner:   PassivePlanner,
        session:   TradingSession,
        dataset=None,
        interval:  int = 30,
    ):
        self._plan         = plan
        self._planner      = planner
        self.session       = session
        self._dataset      = dataset
        self.interval      = interval

        # Thread safety
        self._lock         = threading.Lock()
        self._stop         = threading.Event()

        # Trade states — one per planned trade
        self._states: dict[str, PlannedTradeState] = {
            pt.id: PlannedTradeState(planned=pt)
            for pt in plan.trades
        }

        # Logs
        self._log_buf: list[str] = []

        # Threads
        self._main_thread:    Optional[threading.Thread] = None
        self._refresh_thread: Optional[threading.Thread] = None

        # Tracking
        self._cycle           = 0
        self._last_spot       = plan.spot
        self._last_brief      = None
        self._last_refresh_ts = time.time()
        self._phase           = "STARTING"

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop.clear()
        self._main_thread = threading.Thread(
            target=self._main_loop, daemon=True, name="passive-main"
        )
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name="passive-refresh"
        )
        self._main_thread.start()
        self._refresh_thread.start()
        self._phase = "RUNNING"

    def stop(self) -> None:
        self._stop.set()
        for t in [self._main_thread, self._refresh_thread]:
            if t:
                t.join(timeout=15)
        self._phase = "STOPPED"
        if self._dataset:
            try:
                self._dataset.save_session_summary(self.session)
            except Exception:
                pass

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO") -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
        icon = {"INFO":"·","OK":"✓","WARN":"⚠","ERROR":"✗","TRADE":"◆","PLAN":"📋"}.get(level,"·")
        with self._lock:
            self._log_buf.append(f"[{ts}] {icon} {msg}")
            if len(self._log_buf) > 40:
                self._log_buf.pop(0)

    def get_logs(self) -> list[str]:
        with self._lock:
            return list(self._log_buf)

    # ── Main loop (Thread 1) ──────────────────────────────────────────────────

    def _main_loop(self) -> None:
        self._log("⚡ Passive trader started", "OK")

        while not self._stop.is_set():
            self._cycle += 1
            try:
                # Market hours check
                from trading.market_time import get_market_status
                ms = get_market_status()
                if not ms.is_open:
                    self._log(
                        f"⏰ Market {ms.phase} — next open "
                        f"{ms.next_trading_day.strftime('%d-%b')} "
                        f"in {ms.minutes_to_open}m",
                        "WARN",
                    )
                    self._wait(self.interval)
                    continue

                # Refresh prices for open positions
                self._update_open_positions()

                # Check entry conditions for pending planned trades
                self._check_entry_conditions()

                # Check loss limit
                if self.session.check_and_set_recovery_mode():
                    self._log(
                        f"🔴 Max loss breached — recovery mode, no new entries",
                        "WARN",
                    )

            except Exception as exc:
                self._log(f"Main loop error: {exc}", "ERROR")

            self._wait(self.interval)

        self._phase = "STOPPED"
        self._log("🛑 Passive trader stopped", "OK")

    # ── Async refresh loop (Thread 2) ─────────────────────────────────────────

    def _refresh_loop(self) -> None:
        """
        Runs independently. Never blocks the main trading loop.
        Refreshes the plan every PLAN_REFRESH_SEC or when triggered.
        Open trades are NEVER touched by a refresh.
        """
        self._log("🔄 Async plan refresher started", "OK")

        while not self._stop.is_set():
            # Wait for refresh interval
            for _ in range(self.PLAN_REFRESH_SEC):
                if self._stop.is_set():
                    return
                time.sleep(1)

            try:
                self._log("🔄 Async plan refresh triggered (interval)", "PLAN")
                self._do_refresh()
            except Exception as exc:
                self._log(f"Plan refresh error: {exc}", "ERROR")

    def trigger_refresh(self, reason: str = "") -> None:
        """
        Called from main loop when a refresh trigger fires.
        Spawns a one-shot refresh thread so main loop is not blocked.
        """
        def _run():
            self._log(f"🔄 Async plan refresh: {reason}", "PLAN")
            try:
                self._do_refresh()
            except Exception as exc:
                self._log(f"Triggered refresh error: {exc}", "ERROR")

        t = threading.Thread(target=_run, daemon=True, name="passive-refresh-trigger")
        t.start()

    def _do_refresh(self) -> None:
        """
        Core refresh: call LLM for new plan, atomic swap, update PENDING states.
        Open/Entered trades are NEVER modified.
        """
        with self._lock:
            open_ids = [
                st.trade_id
                for st in self._states.values()
                if st.status == STATUS_ENTERED and st.trade_id
            ]
            old_version = self._plan.version

        new_plan = self._planner.refresh(
            existing_plan  = self._plan,
            open_trade_ids = open_ids,
            log_fn         = self._log,
        )

        with self._lock:
            self._plan = new_plan

            # Rebuild states for NEW planned trades
            new_ids = {pt.id for pt in new_plan.trades}
            old_ids = set(self._states.keys())

            # Mark removed trades as CANCELLED (only if PENDING)
            for old_id in old_ids - new_ids:
                if self._states[old_id].status == STATUS_PENDING:
                    self._states[old_id].status = STATUS_CANCELLED
                    self._log(
                        f"Plan refresh: PT{old_id} cancelled (removed from new plan)",
                        "PLAN",
                    )

            # Add brand-new planned trades
            for pt in new_plan.trades:
                if pt.id not in old_ids:
                    self._states[pt.id] = PlannedTradeState(planned=pt)
                    self._log(
                        f"Plan refresh: PT{pt.id} added — {pt.display()}",
                        "PLAN",
                    )
                else:
                    # Update plan details for PENDING trades
                    if self._states[pt.id].status == STATUS_PENDING:
                        self._states[pt.id].planned = pt

        self._log(
            f"✓ Plan refreshed v{old_version} → v{new_plan.version}  "
            f"bias={new_plan.overall_bias}  "
            f"trades={len(new_plan.trades)}",
            "OK",
        )

    # ── Price updater ──────────────────────────────────────────────────────────

    def _update_open_positions(self) -> None:
        """Refresh LTP for all entered (open) trades."""
        entered_ids = {
            st.trade_id
            for st in self._states.values()
            if st.status == STATUS_ENTERED and st.trade_id
        }
        if not entered_ids:
            return

        from data.nifty_chart import get_option_chart

        for trade in self.session.open_trades:
            if trade.id not in entered_ids:
                continue
            try:
                df = get_option_chart(trade.expiry, trade.strike, trade.option_type)
                if df is not None and not df.empty:
                    ltp = float(df["price"].iloc[-1])

                    # Update state LTP
                    for st in self._states.values():
                        if st.trade_id == trade.id:
                            st.ltp = ltp

                    auto_closed = trade.update_price(ltp)
                    if auto_closed:
                        icon = "🎯" if trade.status == "TARGET_HIT" else "🛑"
                        self._log(
                            f"{icon} {trade.status}: {trade.id} "
                            f"@ ₹{ltp:.1f}  P&L={trade.pnl:+.0f}",
                            "TRADE",
                        )
                        if self._dataset:
                            self._dataset.on_trade_closed(trade, trade.status, None)
                        # Trigger plan refresh on any close event
                        self.trigger_refresh(f"{trade.id} {trade.status}")
                    else:
                        self._log(
                            f"Price {trade.id}: ₹{ltp:.1f}  P&L={trade.pnl:+.0f}",
                            "OK",
                        )
                        if self._dataset:
                            self._dataset.on_price_update(trade, ltp, trade.pnl)
            except Exception as exc:
                self._log(f"Price update failed {trade.id}: {exc}", "ERROR")

    # ── Entry condition checker ────────────────────────────────────────────────

    def _check_entry_conditions(self) -> None:
        """
        For each PENDING planned trade, evaluate its condition.
        Uses live spot + OI to decide if condition is met.
        Enters the trade if condition is satisfied and budget allows.
        """
        with self._lock:
            pending = [
                st for st in self._states.values()
                if st.status == STATUS_PENDING
            ]
            plan = self._plan
            recovery = self.session.recovery_mode

        if not pending or recovery:
            return

        # Get live spot
        try:
            from data.yahoo_feed import get_nifty_spot
            spot = get_nifty_spot()
        except Exception:
            spot = plan.spot

        for state in pending:
            pt = state.planned

            # Direction filter
            if self.session.direction != "BOTH" and pt.action != self.session.direction:
                continue

            # Check if spot satisfies condition
            if not self._eval_condition(pt.condition, spot, plan):
                continue

            # Check budget
            cost = pt.entry_ideal * pt.qty * NIFTY_LOT_SIZE
            if cost > self.session.available_budget:
                self._log(
                    f"PT{pt.id} condition met but budget short "
                    f"(need ₹{cost:,.0f}, avail ₹{self.session.available_budget:,.0f})",
                    "WARN",
                )
                continue

            # Check max open positions
            if len(self.session.open_trades) >= 5:
                self._log("Max 5 open positions reached — waiting", "WARN")
                break

            # ENTER TRADE
            self._enter_planned_trade(state, spot)

    def _eval_condition(self, condition: str, spot: float, plan: PassivePlan) -> bool:
        """
        Evaluate a condition string like:
          "Enter if spot > 24500 after 09:30"
          "Enter when spot breaks above 24550"
          "Enter at market"
          "Enter if spot < 24200"
        Returns True if condition is currently met.
        """
        cond = condition.lower()

        # Always enter
        if any(kw in cond for kw in ["at market", "immediately", "always", "any time"]):
            return True

        # Time constraint — check current IST time
        import re as _re
        time_match = _re.search(r'after (\d{2}):(\d{2})', cond)
        if time_match:
            h, m     = int(time_match.group(1)), int(time_match.group(2))
            now_time = datetime.now(IST).time()
            from datetime import time as _time
            if now_time < _time(h, m):
                return False   # time not yet reached

        # Spot conditions
        if "spot >" in cond or "above" in cond or "breaks above" in cond:
            nums = _re.findall(r'\d{4,6}', cond)
            if nums and spot > float(nums[0]):
                return True
            elif nums:
                return False

        if "spot <" in cond or "below" in cond or "breaks below" in cond:
            nums = _re.findall(r'\d{4,6}', cond)
            if nums and spot < float(nums[0]):
                return True
            elif nums:
                return False

        # Default — enter if condition has no parseable constraint
        return True

    def _enter_planned_trade(self, state: PlannedTradeState, spot: float) -> None:
        """Place the planned trade into the live session."""
        pt = state.planned

        # Get current LTP for entry
        ltp = self._get_current_ltp(pt)
        if ltp <= 0:
            ltp = pt.entry_ideal

        # Check entry range
        if not (pt.entry_min <= ltp <= pt.entry_max):
            self._log(
                f"PT{pt.id} condition met but LTP ₹{ltp:.1f} outside "
                f"entry range ₹{pt.entry_min:.0f}–{pt.entry_max:.0f}",
                "WARN",
            )
            return

        trade_id = f"P{len(self.session.trades) + 1:03d}"
        trade = Trade(
            id            = trade_id,
            expiry        = pt.expiry,
            strike        = pt.strike,
            option_type   = pt.option_type,
            action        = pt.action,
            qty           = pt.qty,
            entry_price   = ltp,
            current_price = ltp,
            sl            = pt.sl,
            target        = pt.target2,   # use T2 as main target
            rationale     = f"[PASSIVE PT{pt.id}] {pt.rationale}",
        )

        ok, msg = self.session.add_trade(trade)
        if ok:
            with self._lock:
                state.status     = STATUS_ENTERED
                state.trade_id   = trade_id
                state.entered_at = datetime.now().strftime("%H:%M:%S")
                state.ltp        = ltp

            self._log(
                f"◆ ENTERED PT{pt.id} → {trade_id}  "
                f"{int(pt.strike)}{pt.option_type} {pt.action} {pt.qty}L "
                f"@ ₹{ltp:.1f}  SL=₹{pt.sl:.1f}  T1=₹{pt.target1:.1f}  T2=₹{pt.target2:.1f}",
                "TRADE",
            )
            if self._dataset:
                self._dataset.on_trade_placed(trade, None, self._cycle)
        else:
            self._log(f"PT{pt.id} entry failed: {msg}", "WARN")

    def _get_current_ltp(self, pt: PlannedTrade) -> float:
        """Fetch live LTP for a planned trade."""
        try:
            from data.nifty_chart import get_option_chart
            df = get_option_chart(pt.expiry, pt.strike, pt.option_type)
            if df is not None and not df.empty:
                return round(float(df["price"].iloc[-1]), 2)
        except Exception:
            pass
        return 0.0

    def _wait(self, seconds: int) -> None:
        for _ in range(seconds):
            if self._stop.is_set():
                break
            time.sleep(1)

    # ── Dashboard renderer ─────────────────────────────────────────────────────

    def render_dashboard(self) -> Group:
        with self._lock:
            plan   = self._plan
            states = dict(self._states)
        s = self.session

        pnl_sign  = "+" if s.total_pnl >= 0 else ""
        pnl_color = "green" if s.total_pnl >= 0 else "red"
        pnl_pct   = s.total_pnl / s.budget * 100 if s.budget else 0

        # Market status
        try:
            from trading.market_time import get_market_status
            ms         = get_market_status()
            mkt_label  = "🟢 OPEN" if ms.is_open else f"🔴 {ms.phase}"
            mkt_style  = "green" if ms.is_open else "red"
        except Exception:
            mkt_label, mkt_style = "NSE", "dim"

        # Header
        header = Table.grid(expand=True, padding=(0, 2))
        header.add_column(ratio=2)
        header.add_column(ratio=2)
        header.add_column(ratio=2)
        header.add_column(ratio=2)
        header.add_row(
            Text(f"⚡ PASSIVE TRADER  Plan v{plan.version}", style="bold color(208)"),
            Text(f"Budget: ₹{s.budget:,.0f}", style="bold white"),
            Text(f"Bias: {plan.overall_bias}  PCR={plan.pcr}", style="cyan"),
            Text(mkt_label, style=mkt_style),
        )

        # P&L row
        pnl_row = Table.grid(expand=True, padding=(0, 3))
        pnl_row.add_column(ratio=3)
        pnl_row.add_column(ratio=3)
        pnl_row.add_column(ratio=4)
        pnl_row.add_row(
            Text(f"Total P&L: {pnl_sign}₹{s.total_pnl:,.0f} ({pnl_sign}{pnl_pct:.2f}%)",
                 style=f"bold {pnl_color}"),
            Text(f"Win {s.win_trades} / Loss {s.loss_trades}  Win-rate {s.win_rate}%",
                 style="dim"),
            Text(f"Plan: {plan.plan_note[:60]}", style="dim"),
        )

        # ── Plan table — all planned trades with status ────────────────────────
        plan_tbl = Table(
            title="[bold cyan]📋 Passive Trade Plan[/bold cyan]",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            expand=True,
        )
        for col, w in [
            ("Pri",4),("ID",6),("Strike",8),("Opt",5),("Act",5),
            ("L",4),("Entry Range",15),("SL",8),("T1",8),("T2",8),
            ("Status",10),("P&L",12),("Scenario",10),
        ]:
            plan_tbl.add_column(col, width=w)

        for pt in sorted(plan.trades, key=lambda x: x.priority):
            state  = states.get(pt.id, PlannedTradeState(planned=pt))
            status = state.status

            # Find live trade P&L if entered
            pnl_text = "—"
            if status == STATUS_ENTERED and state.trade_id:
                lt = next((t for t in s.trades if t.id == state.trade_id), None)
                if lt:
                    ps = "+" if lt.pnl >= 0 else ""
                    pc = "green" if lt.pnl >= 0 else "red"
                    pnl_text = f"[{pc}]{ps}₹{lt.pnl:,.0f}[/{pc}]"

            status_style = {
                STATUS_PENDING:   "dim",
                STATUS_ENTERED:   "bold green",
                STATUS_SKIPPED:   "dim red",
                STATUS_CANCELLED: "dim",
            }.get(status, "dim")

            plan_tbl.add_row(
                str(pt.priority),
                f"PT{pt.id}",
                str(int(pt.strike)),
                pt.option_type,
                pt.action,
                str(pt.qty),
                f"₹{pt.entry_min:.0f}–{pt.entry_max:.0f}",
                f"₹{pt.sl:.0f}",
                f"₹{pt.target1:.0f}",
                f"₹{pt.target2:.0f}",
                Text(status, style=status_style),
                pnl_text,
                pt.scenario[:10],
            )

        # Scenarios panel
        scen_lines = []
        for k, v in plan.scenarios.items():
            color = {"bull":"green","bear":"red","sideways":"yellow"}.get(k.lower(),"dim")
            scen_lines.append(f"[{color}]{k.upper()}[/{color}]: {v}")
        scen_panel = Panel(
            "\n".join(scen_lines) if scen_lines else "[dim]No scenarios[/dim]",
            title="[dim]Scenarios[/dim]",
            border_style="dim",
            padding=(0, 1),
        )

        # Logs
        logs      = self.get_logs()[-12:]
        log_panel = Panel(
            "\n".join(logs) if logs else "[dim]Starting…[/dim]",
            title="[dim]Activity Log[/dim]",
            border_style="dim",
            padding=(0, 1),
        )

        return Group(
            Panel(
                header,
                border_style="color(208)",
                title="[bold color(208)]🙏 Jai Sadguru — Passive Mode[/bold color(208)]",
                subtitle=f"[dim]Plan created {plan.created_at}  ·  "
                         f"Refresh every 15m  ·  Ctrl+C to exit[/dim]",
            ),
            Panel(pnl_row, border_style=pnl_color, padding=(0,1)),
            plan_tbl,
            scen_panel,
            log_panel,
        )
