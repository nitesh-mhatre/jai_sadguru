"""
trading/live_mode.py
--------------------
LiveTrader orchestrates the autonomous trading loop:

  Thread 1 (background)
    ├── update_prices()         – refresh open position CMPs from NSE
    ├── check_loss_limit()      – trigger recovery mode if needed
    └── ask_agent_for_decision() – agent analyses market, returns JSON actions

  Main thread
    └── render_dashboard()      – Rich renderable; call inside Rich Live

The agent receives a structured prompt that includes:
  • current portfolio state (JSON)
  • constraint summary (direction, budget remaining, recovery mode)
  • instruction to respond with a JSON action block

JSON action block format:
{
  "actions": [
    {
      "type": "PLACE_TRADE",
      "expiry": "DD-Mon-YYYY",
      "strike": 24500,
      "option_type": "CE",       // CE or PE
      "action": "BUY",           // BUY or SELL
      "qty": 1,                  // lots
      "entry_price": 125.5,
      "sl": 100.0,
      "target": 160.0,
      "rationale": "..."
    },
    {
      "type": "CLOSE_TRADE",
      "trade_id": "T001",
      "exit_price": 155.0,
      "reason": "Target approached"
    },
    {
      "type": "NO_ACTION",
      "reason": "Waiting for clearer signal"
    }
  ]
}
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime, date
from typing import Optional

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import ui
from .engine import Trade, TradingSession

log = logging.getLogger(__name__)

try:
    from config import HARD_EXIT_TIME_IST as HARD_EXIT_TIME
except Exception:                     # pragma: no cover — defensive
    HARD_EXIT_TIME = "15:15"


def _now_ist_hhmm() -> str:
    """Current wall clock as 'HH:MM' in IST (falls back to local time)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%H:%M")
    except Exception:
        return datetime.now().strftime("%H:%M")


def _opt_float(v) -> Optional[float]:
    """Optional float from a trade action — None for missing/zero/invalid."""
    try:
        f = float(v)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None


# ── Loss → rule.md section mapping ────────────────────────────────────────────

def loss_section(regime: str) -> str:
    """Map a detected market regime onto the matching rule.md section."""
    r = (regime or "").upper()
    if "SIDEWAY" in r or "RANGE" in r:
        return "REGIME SIDEWAYS"
    if "VOLATIL" in r:
        return "REGIME VOLATILE"
    if any(k in r for k in ("DIRECTION", "TREND", "BULL", "BEAR", "MOMENTUM")):
        return "REGIME DIRECTIONAL"
    return "GLOBAL"


# ── System prompt injection for live mode ─────────────────────────────────────

LIVE_TRADING_SUFFIX = """
═══════════════════════════════════════════════════════════════
LIVE TRADING MODE — AUTOMATED EXECUTION
═══════════════════════════════════════════════════════════════
You are now executing trades autonomously. After fetching live
market data with your tools, respond with a JSON action block:

```json
{
  "actions": [
    {
      "type": "PLACE_TRADE",
      "expiry": "DD-Mon-YYYY",
      "strike": 24500,
      "option_type": "CE",
      "action": "BUY",
      "qty": 1,
      "entry_price": 125.5,
      "sl": 100.0,
      "target": 160.0,
      "rationale": "brief reason"
    }
  ]
}
```
Action types: PLACE_TRADE | CLOSE_TRADE (with trade_id, exit_price, reason) | NO_ACTION (with reason)
Each PLACE_TRADE must have: expiry, strike, option_type, action, qty, entry_price, sl, target.
Risk:reward must be at least 1:1.5. Prefer 1:2 or better.
Lot size for NIFTY = 75 units. Capital cost = entry_price × qty × 75.
If market is choppy or no clean setup, use NO_ACTION — don't force trades.
═══════════════════════════════════════════════════════════════
"""


# ── LiveTrader ────────────────────────────────────────────────────────────────

class LiveTrader:
    """
    Manages the autonomous paper-trading loop for one TradingSession.

    Usage:
        trader = LiveTrader(session, agent, config)
        trader.start(interval_seconds=120)
        # ... display loop ...
        trader.stop()
    """

    # Open orders are re-priced this often between decision cycles so the
    # breakeven/trailing stop-loss tracks the market, not the decision cadence.
    PRICE_REFRESH_SEC = 5

    def __init__(self, session: TradingSession, agent, config, log_fn=None,
                 expiry: str = "", levels: list[dict] | None = None):
        self.session  = session
        self.agent    = agent
        self.config   = config
        self._stop    = threading.Event()
        self._lock    = threading.Lock()
        self._log_buf: list[str] = []
        self.interval = 120
        self._thread: Optional[threading.Thread] = None
        self._phase   = "STARTING"
        self._last_brief = None
        self._recent_decisions: list[str] = []   # last 5 decisions for context
        self._no_action_count: int = 0           # consecutive NO_ACTION counter

        # ── Async LLM decoupling: bias state written by bg thread, read instantly ──
        # The main trade loop reads self._action_bias INSTANTLY — it is updated
        # by _llm_bias_loop() every LLM_BIAS_UPDATE_SEC (30-60s) in a separate
        # thread. When the LLM is down the bias is computed from PCR/VIX rules.
        self._action_bias: str = "NEUTRAL"       # BULLISH | BEARISH | NEUTRAL | UNKNOWN
        self._bias_source: str = "NONE"         # LLM | RULES | NONE
        self._bias_last_update: float = 0.0
        self._llm_bias_thread: Optional[threading.Thread] = None

        # Backend diagnostics shown on the dashboard
        self._last_source  = ""        # "groww" | "nse" — last option-data source
        self._last_regime  = ""        # last detected market regime
        self._quote_note   = ""        # live-quote freshness note for the order book
        self._rules_written = 0        # lessons written to rule.md this session
        self._groww_greeks: dict = {}  # last IV/delta/theta seen from the Groww chain
        # Live market strip: current NIFTY value + current premium of each level
        self._market_snap:   dict = {}
        self._market_levels: list = []
        # User-selected expiry + levels (from the startup flow)
        self.expiry = expiry
        self.levels = levels or []               # [{strike, side}]
        # Persistent scanner — cache survives across cycles
        from trading.scanner import MarketScanner
        self._scanner = MarketScanner(direction=session.direction)
        # Dataset logger — records every trade as a case study
        from trading.dataset import DatasetLogger
        import uuid
        session_id    = datetime.now().strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4].upper()
        self._dataset = DatasetLogger(
            session_id  = session_id,
            go_command  = session.go_message,
        )
        self._log(f"📁 Trade dataset: {self._dataset.dataset_path}", "OK")

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self, interval_seconds: int = 120) -> None:
        self.interval = interval_seconds
        self._stop.clear()
        # Populate the market strip before the dashboard first paints, so the
        # NIFTY value and level premiums are on screen from second one.
        self._refresh_market_strip()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="live-trader")
        self._llm_bias_thread = threading.Thread(
            target=self._llm_bias_loop, daemon=True, name="live-llm-bias"
        )
        self._llm_bias_thread.start()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        for t in [self._thread, self._llm_bias_thread]:
            if t:
                t.join(timeout=15)
        self._phase = "STOPPED"
        # Save full session case study dataset on exit
        try:
            path = self._dataset.save_session_summary(self.session)
            self._log(f"📁 Session dataset saved → {path}", "OK")
        except Exception as exc:
            self._log(f"Dataset save failed: {exc}", "ERROR")

    # ── Logging ───────────────────────────────────────────────────────────────

    _ICONS = {"INFO": "·", "OK": "✓", "WARN": "⚠", "ERROR": "✗", "TRADE": "◆",
              "PLAN": "📋", "DATA": "🗄", "AI": "🤖", "RULE": "📖"}

    def _log(self, msg: str, level: str = "INFO") -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
        icon = self._ICONS.get(level, "·")
        line = f"[{ts}] {icon} {msg}"
        with self._lock:
            self._log_buf.append(line)
            if len(self._log_buf) > 250:
                self._log_buf.pop(0)

    def get_logs(self) -> list[str]:
        with self._lock:
            return list(self._log_buf)

    # ── Price updater ─────────────────────────────────────────────────────────

    def _live_premium(self, trade: Trade) -> tuple[float, str]:
        """
        Latest premium for one contract.

        Groww option chain first — data/groww_feed.get_quote covers every strike
        from one cached chain fetch (the Groww route already coded in this repo).
        NSE option chart is the fallback when Groww has no chain for that expiry.
        """
        try:
            from data.groww_feed import get_quote
            q   = get_quote(trade.expiry, int(trade.strike), trade.option_type)
            ltp = float(q.get("ltp") or 0.0)
            if ltp > 0:
                self._groww_greeks = {
                    "iv": q.get("iv", 0.0), "delta": q.get("delta", 0.0),
                    "theta": q.get("theta", 0.0), "oi_change": q.get("oi_change", 0),
                }
                return ltp, f"Groww(age={q.get('chain_age_s', '?')}s)"
        except Exception as exc:
            log.debug("Groww quote failed for %s: %s", trade.id, exc)

        try:
            from data.nifty_chart import get_option_chart
            df = get_option_chart(trade.expiry, trade.strike, trade.option_type)
            if df is not None and not df.empty:
                return float(df["price"].iloc[-1]), "NSE-chart"
        except Exception as exc:
            log.debug("NSE chart failed for %s: %s", trade.id, exc)

        return 0.0, "no-data"

    def _update_prices(self, quiet: bool = False) -> None:
        """
        Refresh CMP + P&L for every open order, manage the stop-loss
        (breakeven / trailing) and auto-close on SL/target hit.

        quiet=True is used by the fast price ticker: closes and SL moves are
        still logged, only the per-tick summary line is suppressed.
        """
        open_trades = self.session.open_trades
        if not open_trades:
            return

        refreshed, failed = 0, 0
        sources: set[str] = set()

        for trade in open_trades:
            ltp, src = self._live_premium(trade)
            if ltp <= 0:
                failed += 1
                self._log(
                    f"No live quote for {trade.id} {int(trade.strike)}{trade.option_type} "
                    f"— CMP held at ₹{trade.current_price:.1f}",
                    "WARN",
                )
                continue

            refreshed += 1
            sources.add(src)
            prev_sl     = trade.sl
            prev_booked = trade.partial_booked_lots
            auto_closed = trade.update_price(ltp)

            # Partial profit booked at T1 (price-driven, inside update_price)
            if trade.partial_booked_lots > prev_booked:
                lots = trade.partial_booked_lots - prev_booked
                self._log(
                    f"💰 PARTIAL BOOK {trade.id} {int(trade.strike)}{trade.option_type}: "
                    f"{lots}L booked @ ₹{ltp:.1f} — T1 hit · realised "
                    f"₹{trade.partial_pnl:,.0f} · {trade.remaining_lots}L running "
                    f"(SL ₹{trade.sl:.1f} {trade.sl_stage})",
                    "TRADE",
                )

            if auto_closed:
                icon = "🎯 TARGET" if trade.status == "TARGET_HIT" else "🛑 SL"
                self._log(
                    f"{icon} {trade.id} {int(trade.strike)}{trade.option_type} "
                    f"closed @ ₹{ltp:.1f} — {trade.close_reason or trade.status} · "
                    f"P&L {'+' if trade.pnl >= 0 else ''}₹{trade.pnl:,.0f} "
                    f"({trade.pnl_pct:+.1f}%)  [{src}]",
                    "TRADE",
                )
                self._dataset.on_trade_closed(trade, trade.status, self._last_brief)
                # Loss → write a lesson into rule.md so it is not repeated
                if trade.pnl < 0:
                    self._record_loss_lesson(trade, trade.close_reason or trade.status)
            else:
                self._dataset.on_price_update(trade, ltp, trade.pnl, self._last_brief)
                # Log a stop-loss move (breakeven / trail) so the backend is visible
                if abs(trade.sl - prev_sl) > 1e-9:
                    self._log(
                        f"SL moved {trade.id} {int(trade.strike)}{trade.option_type}: "
                        f"₹{prev_sl:.1f} → ₹{trade.sl:.1f}  [{trade.sl_stage}] "
                        f"(best ₹{trade.mfe_price:.1f}, progress {trade.profit_progress*100:.0f}%)",
                        "OK",
                    )

        # Keep the live NIFTY value + level premiums fresh on every tick
        self._refresh_market_strip()

        src_note = ", ".join(sorted(sources)) if sources else "no data"
        self._quote_note = (
            f"Quotes: {refreshed}/{len(open_trades)} open orders refreshed · {src_note}"
        )
        if not quiet:
            self._log(
                f"Quotes refreshed {refreshed}/{len(open_trades)} open orders · {src_note}"
                + (f" · {failed} failed" if failed else ""),
                "DATA",
            )

    def _refresh_market_strip(self) -> None:
        """
        Track the CURRENT NIFTY value and the current premium/OI/IV/greeks of
        every tracked level (Groww chain, TTL-cached so this is cheap).
        """
        try:
            from data.groww_feed import get_levels_snapshot
            data = get_levels_snapshot(self.expiry, self.levels)
            self._market_snap   = data.get("snapshot", {}) or {}
            self._market_levels = data.get("levels", []) or []
        except Exception as exc:
            log.debug("Market strip refresh failed: %s", exc)

    # ── Async LLM bias loop (background thread, decoupled from trade execution) ──

    LLM_BIAS_UPDATE_SEC = 30   # update bias every 30s (30–60s target range)

    def _llm_bias_loop(self) -> None:
        """
        Background thread: updates action_bias state every LLM_BIAS_UPDATE_SEC.

        DECOUPLED from trade execution. The main loop reads self._action_bias
        INSTANTLY — no network calls, no blocking on NVIDIA NIM.

        Strategy:
          1. Try to compute bias from live PCR/VIX/OI data (fast, local).
          2. If the LLM is reachable and responsive, optionally ask it for a
             sentiment opinion — but NEVER block the main loop on this.
          3. When the LLM is down (timeout/500): use rule-based bias instantly.
        """
        _log = self._log

        while not self._stop.is_set():
            for _ in range(self.LLM_BIAS_UPDATE_SEC):
                if self._stop.is_set():
                    return
                time.sleep(1)

            bias     = "UNKNOWN"
            source   = "NONE"
            new_pcr  = 0.0
            new_vix  = 0.0

            try:
                # ── Instant rule-based bias from market data ────────────────────
                from trading.rules_fallback import pcr_bias, vix_regime
                from data.yahoo_feed import get_spot_and_vix

                spot_data = get_spot_and_vix()
                spot_val  = spot_data.get("spot", 0.0)
                vix_val   = spot_data.get("vix", 0.0)

                # Get PCR from option chain
                try:
                    from data.nifty_option_chain import get_nifty_option_chain
                    df, _ = get_nifty_option_chain(expiry=None)
                    ce_oi = int(df["CE_OI"].sum())
                    pe_oi = int(df["PE_OI"].sum())
                    new_pcr = round(pe_oi / ce_oi, 2) if ce_oi else 0.0
                except Exception:
                    new_pcr = 0.0

                new_vix = vix_val

                if vix_val > 25:
                    bias = "NO_TRADE"
                    source = "RULES (VIX)"
                elif vix_val < 13:
                    bias = "SELL_PREMIUM"
                    source = "RULES (VIX)"
                elif new_pcr >= 1.2:
                    bias = "BULLISH"
                    source = "RULES (PCR)"
                elif new_pcr <= 0.8:
                    bias = "BEARISH"
                    source = "RULES (PCR)"
                else:
                    bias = "NEUTRAL"
                    source = "RULES (PCR)"

            except Exception as exc:
                _log(f"bias loop market data error: {exc} — keeping last bias", "WARN")

            # Atomic write — main loop reads this INSTANTLY
            with self._lock:
                self._action_bias   = bias
                self._bias_source   = source
                self._bias_last_update = time.time()
                self._last_regime   = (
                    "VOLATILE" if new_vix > 25 else
                    "SIDEWAYS"  if new_vix < 13 else
                    "DIRECTIONAL" if new_pcr >= 1.2 or new_pcr <= 0.8 else "SIDEWAYS"
                )

            _log(f"bias updated: {bias} [{source}] PCR={new_pcr:.2f} VIX={new_vix:.1f}", "OK")

    def _manage_time_exits(self, now_hhmm: str = "") -> None:
        """
        Square off open option orders on time:
          • orders held longer than `max_hold_minutes`
          • everything at the hard exit time (default 15:15 IST)
        Runs on the fast ticker as well as each decision cycle.
        """
        if not self.session.open_trades:
            return
        now_hhmm = now_hhmm or _now_ist_hhmm()
        now_ts   = time.time()

        for trade in list(self.session.open_trades):
            reason = trade.should_time_exit(now_ts, now_hhmm, HARD_EXIT_TIME)
            if not reason:
                continue
            ltp, src   = self._live_premium(trade)
            exit_price = ltp if ltp > 0 else trade.current_price
            ok, msg    = self.session.close_trade(trade.id, exit_price)
            if not ok:
                self._log(f"Time exit failed for {trade.id}: {msg}", "WARN")
                continue
            trade.close_reason = reason
            self._log(
                f"⏱ TIME EXIT {trade.id} {int(trade.strike)}{trade.option_type} "
                f"@ ₹{exit_price:.1f} — {reason} · "
                f"P&L {'+' if trade.pnl >= 0 else ''}₹{trade.pnl:,.0f} "
                f"({trade.pnl_pct:+.1f}%)  [{src}]",
                "TRADE",
            )
            self._dataset.on_trade_closed(trade, reason, self._last_brief)
            if trade.pnl < 0:
                self._record_loss_lesson(trade, reason)

    def _record_loss_lesson(self, trade: Trade, close_reason: str) -> None:
        """
        Persist a loss as a lesson in rule.md so the same mistake is not made
        again. rule.md is injected into every AI prompt (agent.py), so the next
        live AND simulation cycle sees it.
        """
        try:
            from trading.rules import record_lessons
        except Exception as exc:
            log.warning("rules module unavailable: %s", exc)
            return

        brief  = self._last_brief
        regime = self._last_regime or "UNKNOWN"
        pcr    = getattr(brief, "pcr", 0.0) if brief else 0.0
        vix    = getattr(brief, "vix", 0.0) if brief else 0.0
        entry  = trade.entry_price
        exit_p = trade.exit_price if trade.exit_price is not None else trade.current_price

        lesson = (
            f"[LIVE LOSS] {int(trade.strike)}{trade.option_type} {trade.action} in "
            f"{regime} regime (PCR={pcr}, VIX={vix}): entry ₹{entry:.1f} → "
            f"exit ₹{exit_p:.1f} ({trade.pnl_pct:+.0f}%, {close_reason}). "
            f"Avoid this setup while the same regime/PCR/VIX conditions are present."
        )
        section = loss_section(regime)
        record_lessons([lesson], section=section,
                       source_note=f"live {trade.id} loss · {close_reason}")
        self._rules_written += 1
        self._log(
            f"Loss recorded in rule.md → [{section}] ({trade.id}); "
            f"the AI will see it in every future prompt",
            "RULE",
        )

    # ── Agent decision cycle ──────────────────────────────────────────────────

    # ── Hybrid decision: /next first, AI as fallback ──────────────────────────

    def _decide_with_next_or_ai(self) -> None:
        """
        Decision flow:
          Step 1 — Run /next predictor (pure calc, < 15s, zero AI cost)
          Step 2 — If HIGH or MEDIUM confidence → execute trades directly
          Step 3 — If LOW confidence            → run AI for deeper analysis
          Step 4 — If NO_TRADE from /next       → run AI for confirmation

        This gives the best of both worlds:
          Fast path  : /next fires in < 15s when market signal is clear
          Slow path  : AI kicks in only when signals are genuinely mixed
        """
        from trading.next_predictor import predict_next

        session = self.session
        self._log("⚡ Running /next predictor…", "INFO")

        try:
            prediction = predict_next(budget=session.available_budget)
        except Exception as exc:
            self._log(f"⚡ /next failed ({exc}) — falling back to AI", "WARN")
            self._ask_agent_for_decision()
            return

        self._last_regime = prediction.regime
        self._log(
            f"⚡ /next predictor: score={prediction.total_score:.1f}/100  "
            f"regime={prediction.regime}  conf={prediction.confidence}  "
            f"bull={prediction.bull_signals}/bear={prediction.bear_signals}  "
            f"action={prediction.action} {prediction.option_type}",
            "AI",
        )

        # ── Direction filter ──────────────────────────────────────────────────
        if session.direction != "BOTH":
            if prediction.action == "BUY" and prediction.option_type == "CE" \
               and session.direction == "SELL":
                self._log("⚡ /next CE buy blocked — direction=SELL only", "WARN")
                prediction = None   # force AI
            elif prediction.action == "BUY" and prediction.option_type == "PE" \
                 and session.direction == "BUY":
                self._log("⚡ /next PE buy blocked — direction=BUY only", "WARN")
                prediction = None

        # ── High/Medium confidence: execute /next result directly ─────────────
        if prediction and prediction.confidence in ("HIGH", "MEDIUM") \
           and prediction.action != "NO_TRADE":

            actions = prediction.as_trade_action(expiry_override=prediction.expiry)
            if actions:
                for act in actions:
                    self._execute_action(act, source="/next")

            # Update decision history
            with self._lock:
                session.decision_count      += 1
                session.last_decision_time   = datetime.now()
                self._no_action_count        = 0
                self._recent_decisions.append(
                    f"/next:{prediction.regime}:{prediction.action}"
                    f"{prediction.option_type}@{prediction.entry_price:.0f}"
                )
                if len(self._recent_decisions) > 5:
                    self._recent_decisions.pop(0)
            return

        # ── NO_TRADE or LOW confidence: run AI for deeper analysis ────────────
        reason = (
            f"LOW confidence ({prediction.total_score:.0f}/100)"
            if prediction and prediction.confidence == "LOW"
            else "NO_TRADE signal"
        )
        self._log(
            f"⚡ /next says {reason} — sending to AI for deeper analysis",
            "INFO",
        )
        self._ask_agent_for_decision()

    def _execute_action(self, act: dict, source: str = "") -> None:
        """
        Execute a single PLACE_TRADE action dict directly
        (bypassing the AI JSON parser).
        """
        try:
            trade_id = f"T{len(self.session.trades) + 1:03d}"
            from trading.engine import Trade
            trade = Trade(
                id            = trade_id,
                expiry        = str(act["expiry"]),
                strike        = float(act["strike"]),
                option_type   = str(act.get("option_type","CE")).upper(),
                action        = str(act.get("action","BUY")).upper(),
                qty           = max(1, int(act.get("qty", 1))),
                entry_price   = float(act["entry_price"]),
                current_price = float(act["entry_price"]),
                sl            = float(act["sl"]),
                target        = float(act["target"]),
                partial_target = _opt_float(act.get("partial_target")),
                rationale     = str(act.get("rationale","")),
                source        = source or "/next",
            )
        except (KeyError, ValueError, TypeError) as exc:
            self._log(f"Invalid trade params from {source}: {exc}", "ERROR")
            return

        with self._lock:
            ok, msg = self.session.add_trade(trade)

        level = "TRADE" if ok else "WARN"
        self._log(
            f"{source or 'signal'} → {trade_id} "
            f"{int(trade.strike)}{trade.option_type} {trade.action} "
            f"{trade.qty}L @ ₹{trade.entry_price}  "
            f"SL ₹{trade.sl}  TGT ₹{trade.target}"
            + ("" if ok else f"  — rejected: {msg}"),
            level,
        )

        if ok and self._dataset:
            self._dataset.on_trade_placed(trade, self._last_brief,
                                          self.session.decision_count)
            with self._lock:
                self._no_action_count = 0

    # ── Agent decision (AI path — used for LOW confidence / recovery) ─────────

    def _ask_agent_for_decision(self) -> None:
        """
        Targeted scan + single AI call with INSTANT rule-based fallback.

        DECOUPLED from trade execution: this method is called from the main
        decision loop but if the LLM times out or returns 500, we fall back
        INSTANTLY to rule-based logic — no blocking, no retry.

        The async _llm_bias_loop() updates self._action_bias in the background
        every 30s so the main loop always has a fresh bias without waiting.
        """
        from agent import Session as AgentSession, FinalAnswerEvent, ErrorEvent
        from trading.rules_fallback import FallbackEngine, fallback_decide
        from config import FAST_MODEL_TIMEOUT
        import time as _time_mod

        session = self.session

        # ── Step 1: Python-side market scan (no AI tokens) ────────────────────
        self._log("📡 Scanning market (targeted strikes only)…")
        open_strikes = [(int(t.strike), t.option_type) for t in session.open_trades]
        brief = self._scanner.scan(open_strikes=open_strikes)
        self._last_brief = brief   # store for dataset hooks

        if brief.error:
            self._log(f"Scan error: {brief.error}", "ERROR")
            return

        self._last_source = brief.source
        self._log(
            f"Market scan [{brief.source}] in {brief.scan_duration_ms}ms  "
            f"spot={brief.spot}  ATM={brief.atm}  PCR={brief.pcr}  "
            f"sentiment={brief.sentiment}  "
            f"strikes: {[f'{t.strike}{t.option_type}' for t in brief.targeted]}",
            "DATA",
        )

        # ── Step 2: Build compact portfolio state ─────────────────────────────
        open_pos_summary = ""
        if session.open_trades:
            lines = []
            for t in session.open_trades:
                booked = (f" T1_booked={t.partial_booked_lots}L/{t.qty}L"
                          if t.partial_booked_lots else "")
                lines.append(
                    f"  {t.id} {int(t.strike)}{t.option_type} {t.action} {t.qty}L "
                    f"entry=₹{t.entry_price} CMP=₹{t.current_price} "
                    f"PnL={t.pnl:+.0f} SL=₹{t.sl:.1f}({t.sl_stage}) TGT=₹{t.target}"
                    f"{booked}"
                )
            open_pos_summary = "OPEN POSITIONS:\n" + "\n".join(lines)
        else:
            open_pos_summary = "OPEN POSITIONS: none"

        recovery_note = (
            "⚠️ RECOVERY MODE — max loss breached. ONLY close/manage open positions. "
            "DO NOT place new trades.\n"
            if session.recovery_mode else ""
        )

        # ── Step 3: Build the compact AI prompt ───────────────────────────────
        market_text = brief.to_ai_prompt(
            direction    = session.direction,
            budget       = session.available_budget,
            max_loss_pct = session.max_loss_pct,
        )

        # Focus levels block (user-selected at startup, optional)
        levels_block = ""
        if self.levels:
            lv = ", ".join(f"{l['strike']}{l['side']}" for l in self.levels)
            levels_block = (
                f"\nFOCUS LEVELS (user-selected — prioritise these strikes for "
                f"entries/exits; expiry: {self.expiry or brief.expiry}):\n  {lv}\n"
            )

        # News brief (fast, non-blocking)
        news_text = ""
        try:
            from data.news_feed import news_brief
            news_text = news_brief(days=1)
        except Exception:
            news_text = "News unavailable"

        # Technicals (EMA/VWAP) — quick call
        tech_text = ""
        try:
            from data.market_extra import get_nifty_technicals
            tech = get_nifty_technicals()
            if "error" not in tech:
                tech_text = (
                    f"TECHNICALS: trend={tech['trend']}  "
                    f"EMA9={tech['ema9']}  EMA21={tech['ema21']}  "
                    f"VWAP={tech['vwap']}  bias={tech['tech_bias']}  "
                    f"note={tech.get('trade_note','')}"
                )
        except Exception:
            tech_text = "Technicals unavailable"
        vix = brief.vix
        if vix <= 0:      vix_note = "VIX unavailable"
        elif vix < 13:    vix_note = f"VIX={vix} LOW → SELL premium strategy preferred"
        elif vix < 18:    vix_note = f"VIX={vix} NORMAL → directional buy or premium sell both valid"
        elif vix < 25:    vix_note = f"VIX={vix} HIGH → BUY options only, sellers at risk"
        else:             vix_note = f"VIX={vix} EXTREME → NO NEW TRADES, close open positions only"

        # ── Market regime detection (sideways / directional / volatile) ───────
        regime_block = ""
        try:
            from trading.market_regime import detect_regime
            tech_raw = {}
            try:
                from data.market_extra import get_nifty_technicals
                tech_raw = get_nifty_technicals()
            except Exception:
                pass
            regime = detect_regime(brief, vix, tech_raw)
            regime_block = "\n" + regime.prompt_block()
            self._last_regime = regime.regime
            self._log(
                f"📊 Regime: {regime.regime} ({regime.confidence}/4)  "
                f"Range: {regime.range_low:.0f}–{regime.range_high:.0f}  "
                f"Strategy: {regime.strategy or 'unclear'}",
                "OK",
            )
        except Exception as exc:
            log.warning("Regime detection failed: %s", exc)

        # ── Kronos candle forecast (multi-model layer: quantitative vote) ────
        kronos_block = ""
        try:
            from trading.kronos_forecast import forecast_nifty_live
            kf = forecast_nifty_live(pred_len=3, interval="5m")
            if kf:
                kronos_block = "\n" + kf.to_prompt_block() + "\n"
                self._log(
                    f"📈 Kronos: {kf.direction} {kf.move_points:+.0f}pts "
                    f"over {kf.pred_len}×{kf.interval}",
                    "OK",
                )
            else:
                self._log("📈 Kronos unavailable this cycle — LLM-only vote", "INFO")
        except Exception as exc:
            self._log(f"Kronos forecast failed: {exc}", "WARN")

        # OI velocity (fresh writing signals interpreted)
        oi_signals = []
        for fw in brief.fresh_ce_writing:
            oi_signals.append(f"Fresh CE writing at {fw['strike']} (+{fw['added_oi']:,}) → BEARISH pressure building")
        for fw in brief.fresh_pe_writing:
            oi_signals.append(f"Fresh PE writing at {fw['strike']} (+{fw['added_oi']:,}) → BULLISH support building")
        oi_velocity = "\n".join(oi_signals) if oi_signals else "No significant fresh OI writing detected"

        # Recent decision history (so model knows it has been inactive)
        if self._recent_decisions:
            history_note = (
                f"YOUR LAST {len(self._recent_decisions)} DECISIONS: "
                + " | ".join(self._recent_decisions[-5:])
            )
            if self._no_action_count >= 3:
                history_note += (
                    f"\n⚠️  {self._no_action_count} consecutive NO_ACTIONs. "
                    "If market is SIDEWAYS, that is CORRECT — use a premium-selling strategy instead of NO_ACTION. "
                    "If market has clear direction, find the setup and trade it."
                )
        else:
            history_note = "First decision cycle."

        prompt = f"""LIVE TRADING CYCLE #{session.decision_count + 1}  {brief.fetched_at}
{recovery_note}
{market_text}
{levels_block}
{regime_block}
VIX REGIME: {vix_note}

OI VELOCITY (smart money signals):
{oi_velocity}

{tech_text}

{news_text}
{kronos_block}
{open_pos_summary}
total_pnl=₹{session.total_pnl:+,.0f}  realised=₹{session.realized_pnl:+,.0f}
open_positions={len(session.open_trades)}/3  available_budget=₹{session.available_budget:,.0f}

{history_note}

DECISION CHECKLIST:
1. What is the REGIME? (SIDEWAYS / DIRECTIONAL / VOLATILE)
2. If SIDEWAYS  → use SHORT STRANGLE or IRON CONDOR (SELL both sides)
3. If DIRECTIONAL → count confirms (need 3+): PCR direction, OI writing, tech trend, VIX
4. If VOLATILE  → buy options wide SL, not sell
5. R:R ≥ 1:1.5 for buys. For sells: target = collect 50% premium, SL = premium doubles.
6. If a KRONOS FORECAST block is present, treat it as ONE vote — agree = confidence up, disagree = explain why you overrule it. Never trade on Kronos alone.
7. Open orders manage themselves: the stop-loss moves to BREAKEVEN (entry) once 50% of the entry→target move is captured, then TRAILS 25% behind the best premium; at T1 (50% of the move) half the lots are booked and the rest runs; orders are squared off automatically after the max hold time or at 15:15 IST. Do NOT micromanage SL/partials — only CLOSE_TRADE when the thesis is invalidated.

{"RECOVERY MODE — close only." if session.recovery_mode else ""}

RESPOND ONLY WITH JSON:
```json
{{"actions": [{{"type": "PLACE_TRADE", "expiry": "{brief.expiry}", "strike": 0, "option_type": "CE", "action": "BUY", "qty": 1, "entry_price": 0.0, "sl": 0.0, "target": 0.0, "rationale": "REGIME=X, X confirms: [list]. SL at OI wall. RR=1:X"}}]}}
```
For SELL trades use "action": "SELL" and TWO actions (CE + PE for strangle).
For NO_ACTION: {{"actions": [{{"type": "NO_ACTION", "reason": "REGIME=X because [signals]. No suitable setup."}}]}}
"""
        # ── Step 4: Multi-model async LLM with vote aggregation + fallback ──
        # Fire llama-vision + mistral + nemo-light in PARALLEL.
        # The fastest model's response is used; if all fail → rules fallback.
        # This is NON-BLOCKING for the trade loop — the bias thread keeps running.
        try:
            from trading.async_llm import LLMPool, sync_query_with_fallback, sync_vote_all
            from config import TRADING_SYSTEM_ADDENDUM, REGIME_TRADING_RULES

            # Build the full prompt with system suffix for the LLM pool
            full_prompt = prompt
            system_suffix = TRADING_SYSTEM_ADDENDUM + REGIME_TRADING_RULES

            # ── Option A: query with fallback chain (fastest first) ───────────
            # This fires llama-vision first (fastest), then falls back to
            # mistral, then nemo-light if needed — all async.
            llm_result = sync_query_with_fallback(
                pool=None,  # will create fresh pool for this call
                prompt=full_prompt,
                system_suffix=system_suffix,
            )

            if llm_result.success and llm_result.response_text.strip():
                final_text = llm_result.response_text
                self._log(
                    f"🤖 {llm_result.model_name} responded in {llm_result.elapsed:.1f}s "
                    f"({len(final_text)} chars)",
                    "OK",
                )
                self._parse_and_execute(final_text)
            else:
                raise Exception(f"{llm_result.model_name} failed: {llm_result.error}")

        except Exception as exc:
            msg = str(exc)
            self._log(f"Multi-model LLM call failed: {msg[:150]}", "ERROR")

            # ── INSTANT RULE-BASED FALLBACK — no blocking, no retry ──────────
            self._log(
                "⚠ LLM unavailable — falling back to rules INSTANTLY",
                "WARN",
            )

            fb = FallbackEngine(
                iv=brief.vix if brief.vix > 0 else 15.0,
                minutes_to_expiry=375,
            )
            decision = fb.decide(
                spot=brief.spot,
                pcr=brief.pcr,
                vix=brief.vix,
                expiry=brief.expiry,
                atm=brief.atm,
                ce_walls=[fw['strike'] for fw in brief.fresh_ce_writing] or [],
                pe_walls=[fw['strike'] for fw in brief.fresh_pe_writing] or [],
                budget=session.available_budget,
                direction=session.direction,
                max_loss_pct=session.max_loss_pct,
                open_positions=len(session.open_trades),
            )

            if decision.actions:
                self._log(
                    f"Rules fallback: {len(decision.actions)} action(s) [src={decision.source}]",
                    "OK",
                )
                for act in decision.actions:
                    self._execute_action(
                        {
                            "type":          act.type,
                            "expiry":        act.expiry,
                            "strike":        act.strike,
                            "option_type":   act.option_type,
                            "action":        act.action,
                            "qty":           act.qty,
                            "entry_price":   act.entry_price,
                            "sl":            act.sl,
                            "target":        act.target,
                            "rationale":     act.rationale,
                        },
                        source="RULES",
                    )
            else:
                self._log(
                    f"Rules fallback: NO_ACTION — {decision.no_action_reason}",
                    "INFO",
                )
                with self._lock:
                    self._no_action_count += 1
                    self._recent_decisions.append(
                        f"NO_ACTION(rules:{decision.no_action_reason[:40]})"
                    )
                    if len(self._recent_decisions) > 5:
                        self._recent_decisions.pop(0)

        with self._lock:
            session.decision_count       += 1
            session.last_decision_time    = datetime.now()

    def _parse_and_execute(self, agent_text: str) -> None:
        """Extract JSON action block from agent response and execute actions."""
        # Try fenced block first, then raw JSON
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', agent_text, re.DOTALL)
        if not match:
            match = re.search(r'\{\s*"actions"\s*:.*\}', agent_text, re.DOTALL)
        if not match:
            self._log("Agent returned no JSON action block — no action taken", "WARN")
            return

        raw = match.group(1) if match.lastindex else match.group(0)
        try:
            data    = json.loads(raw)
            actions = data.get("actions", [])
        except json.JSONDecodeError as exc:
            self._log(f"JSON parse error: {exc}", "ERROR")
            return

        for act in actions:
            atype = str(act.get("type", "")).upper()

            if atype == "PLACE_TRADE":
                trade_id = f"T{len(self.session.trades) + 1:03d}"
                try:
                    trade = Trade(
                        id            = trade_id,
                        expiry        = str(act["expiry"]),
                        strike        = float(act["strike"]),
                        option_type   = str(act.get("option_type", "CE")).upper(),
                        action        = str(act.get("action", "BUY")).upper(),
                        qty           = int(act.get("qty", 1)),
                        entry_price   = float(act["entry_price"]),
                        current_price = float(act["entry_price"]),
                        sl            = float(act["sl"]),
                        target        = float(act["target"]),
                        partial_target = _opt_float(act.get("partial_target")),
                        rationale     = str(act.get("rationale", "")),
                        source        = "AI",
                    )
                except (KeyError, ValueError, TypeError) as exc:
                    self._log(f"Invalid PLACE_TRADE params: {exc}", "ERROR")
                    continue

                with self._lock:
                    ok, msg = self.session.add_trade(trade)
                level = "TRADE" if ok else "WARN"
                self._log(
                    f"PLACE {trade_id} {int(trade.strike)}{trade.option_type} "
                    f"{trade.action} {trade.qty}L @ ₹{trade.entry_price}  "
                    f"SL ₹{trade.sl}  TGT ₹{trade.target}"
                    + ("" if ok else f"  — rejected: {msg}"),
                    level,
                )
                # Dataset: record entry with market context
                if ok:
                    self._dataset.on_trade_placed(
                        trade, self._last_brief, self.session.decision_count
                    )
                    with self._lock:
                        self._no_action_count = 0   # reset on successful trade
                        self._recent_decisions.append(
                            f"PLACED({trade_id} {int(trade.strike)}{trade.option_type} "
                            f"@ ₹{trade.entry_price})"
                        )
                        if len(self._recent_decisions) > 5:
                            self._recent_decisions.pop(0)

            elif atype == "CLOSE_TRADE":
                trade_id   = str(act.get("trade_id", ""))
                try:
                    exit_price = float(act.get("exit_price", 0))
                except (TypeError, ValueError):
                    exit_price = 0.0
                reason = act.get("reason", "MANUAL")

                # Find the trade before closing so we can log it
                closing_trade = next(
                    (t for t in self.session.open_trades if t.id == trade_id.upper()),
                    None
                )
                with self._lock:
                    ok, msg = self.session.close_trade(trade_id, exit_price)
                self._log(
                    f"CLOSE {trade_id} @ ₹{exit_price}  [{reason}]"
                    + ("" if ok else f"  — {msg}"),
                    "TRADE" if ok else "WARN",
                )
                # Dataset: record manual close + learn from a losing exit
                if ok and closing_trade:
                    closing_trade.close_reason = reason or "manual close"
                    self._dataset.on_trade_closed(
                        closing_trade, "MANUAL", self._last_brief
                    )
                    if closing_trade.pnl < 0:
                        self._record_loss_lesson(
                            closing_trade, closing_trade.close_reason
                        )

            elif atype == "NO_ACTION":
                reason = act.get('reason', 'No signal')
                self._log(f"NO ACTION: {reason}", "INFO")
                with self._lock:
                    self._no_action_count += 1
                    self._recent_decisions.append(f"NO_ACTION({reason[:40]})")
                    if len(self._recent_decisions) > 5:
                        self._recent_decisions.pop(0)

            else:
                self._log(f"Unknown action type: {atype}", "WARN")

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        self._phase = "RUNNING"
        self._log("🚀 Live trading started", "OK")
        _om_ran_today: Optional[date] = None   # track if OM already ran today

        while not self._stop.is_set():
            try:
                from trading.market_time import get_market_status
                from datetime import date as _date
                ms  = get_market_status()
                now = ms.now_ist

                # ── Market closed — skip and wait ─────────────────────────────
                if not ms.is_open and ms.phase not in ("PRE_OPEN",):
                    reason = ms.close_reason
                    if ms.holiday_name:
                        reason += f" — {ms.holiday_name}"
                    self._log(
                        f"⏰ Market {ms.phase} ({reason})  "
                        f"Next: {ms.next_trading_day.strftime('%d-%b %a')}"
                        + (f" in {ms.minutes_to_open}m" if ms.minutes_to_open > 0 else ""),
                        "WARN",
                    )
                    if ms.gift_nifty > 0:
                        self._log(
                            f"📡 GIFT Nifty: {ms.gift_nifty:.1f}  "
                            f"chng={ms.gift_nifty_chng:+.1f}  → {ms.gap_direction}",
                            "INFO",
                        )
                    for _ in range(self.interval):
                        if self._stop.is_set():
                            break
                        time.sleep(1)
                    continue

                # ── PRE-OPEN: gap scan (09:00–09:15) ─────────────────────────
                if ms.phase == "PRE_OPEN":
                    from trading.opening_momentum import OpeningMomentumEngine
                    om = OpeningMomentumEngine(
                        session = self.session,
                        dataset = self._dataset,
                        log_fn  = self._log,
                    )
                    self._log("🌅 PRE-OPEN: scanning gap for opening momentum…", "INFO")
                    gap = om.pre_scan()
                    self._log(
                        f"Gap: {gap.gap_pct:+.2f}% {gap.direction}  "
                        + (f"→ {gap.strategy_note}" if gap.tradeable
                           else f"Skip: {gap.skip_reason}"),
                        "OK" if gap.tradeable else "INFO",
                    )
                    # Wait until 09:15
                    for _ in range(self.interval):
                        if self._stop.is_set():
                            break
                        time.sleep(1)
                    continue

                # ── OPENING MOMENTUM (09:15–09:30) ────────────────────────────
                today = now.date()
                in_om_window = (
                    now.hour == 9 and now.minute < 30   # 09:15–09:29
                )
                if in_om_window and _om_ran_today != today:
                    _om_ran_today = today
                    from trading.opening_momentum import OpeningMomentumEngine
                    om = OpeningMomentumEngine(
                        session = self.session,
                        dataset = self._dataset,
                        log_fn  = self._log,
                    )
                    self._log(
                        "⚡ OPENING MOMENTUM WINDOW — running gap play…", "OK"
                    )
                    om.run(brief=self._last_brief)
                    # After OM finishes (09:30), fall through to normal cycle
                    continue

                # ── NORMAL SESSION: time exits + price refresh + decisions ─
                self._manage_time_exits(now.strftime("%H:%M"))
                if self.session.open_trades:
                    self._update_prices()
                else:
                    self._log("No open positions — skipping price refresh")

                just_triggered = self.session.check_and_set_recovery_mode()
                if just_triggered:
                    self._phase = "RECOVERY"
                    self._log(
                        f"🔴 MAX LOSS {self.session.max_loss_pct}% BREACHED — "
                        "RECOVERY MODE: no new trades",
                        "WARN",
                    )

                # ── /next predictor first (< 15s, zero AI tokens) ────────────
                # HIGH/MEDIUM confidence → execute directly (fast path)
                # LOW confidence         → fall back to full AI analysis
                if not self.session.recovery_mode:
                    self._decide_with_next_or_ai()
                else:
                    # Recovery: only use AI to manage/close open positions
                    self._ask_agent_for_decision()

            except Exception as exc:
                self._log(f"Loop error: {exc}", "ERROR")

            # ── Wait out the decision interval, but keep pricing open orders ──
            # The stop-loss (breakeven/trailing) must react faster than the
            # decision cadence, so positions are re-priced every
            # PRICE_REFRESH_SEC while we wait — quietly, to keep logs readable.
            waited = 0
            while waited < self.interval and not self._stop.is_set():
                time.sleep(1)
                waited += 1
                if waited % self.PRICE_REFRESH_SEC == 0:
                    if self.session.open_trades:
                        self._update_prices(quiet=True)
                    self._manage_time_exits()

        self._phase = "STOPPED"
        self._log("🛑 Live trading loop ended", "OK")

    # ── Dashboard renderer ────────────────────────────────────────────────────

    def render_dashboard(self) -> Group:
        """Return a Rich Group renderable — plug into a Rich Live context."""
        s = self.session

        pnl_sign  = "+" if s.total_pnl >= 0 else ""
        pnl_color = "green" if s.total_pnl >= 0 else "red"
        pnl_pct   = s.total_pnl / s.budget * 100 if s.budget else 0

        # ── Header row ────────────────────────────────────────────────────────
        # ── Market status ──────────────────────────────────────────────────────
        try:
            from trading.market_time import get_market_status
            ms         = get_market_status()
            mkt_label  = "🟢 OPEN" if ms.is_open else f"🔴 {ms.phase}"
            mkt_style  = "green" if ms.is_open else "red"
            mkt_detail = (
                f"closes in {ms.minutes_to_close}m" if ms.is_open
                else f"opens {ms.next_trading_day.strftime('%d-%b')} in {ms.minutes_to_open}m"
            )
            gift_line  = (
                f"  GIFT:{ms.gift_nifty:.0f}({ms.gift_nifty_chng:+.0f}) {ms.gap_direction}"
                if ms.gift_nifty > 0 else ""
            )
        except Exception:
            mkt_label, mkt_style, mkt_detail, gift_line = "NSE", "dim", "", ""

        mode_label = "🔴 RECOVERY MODE" if s.recovery_mode else "🟢 LIVE TRADING"
        phase_style = "bold red" if s.recovery_mode else "bold green"

        header = Table.grid(expand=True, padding=(0, 2))
        for _ in range(4):
            header.add_column(ratio=2)
        header.add_row(
            Text(mode_label, style=phase_style),
            Text(f"Budget: ₹{s.budget:,.0f}", style="bold white"),
            Text(f"Max Loss: {s.max_loss_pct}%  (₹{s.max_loss_amount:,.0f})", style="yellow"),
            Text(f"NSE {mkt_label}  {mkt_detail}{gift_line}", style=mkt_style),
        )

        # ── P&L summary ───────────────────────────────────────────────────────
        pnl_row = Table.grid(expand=True, padding=(0, 3))
        for _ in range(4):
            pnl_row.add_column(ratio=3)
        unreal_sign = "+" if s.unrealized_pnl >= 0 else ""
        real_sign   = "+" if s.realized_pnl   >= 0 else ""
        pnl_row.add_row(
            Text(f"Total P&L: {pnl_sign}₹{s.total_pnl:,.0f}  ({pnl_sign}{pnl_pct:.2f}%)",
                 style=f"bold {pnl_color}"),
            Text(f"Unrealised: {unreal_sign}₹{s.unrealized_pnl:,.0f}", style="dim"),
            Text(f"Realised: {real_sign}₹{s.realized_pnl:,.0f}", style="dim"),
            Text(
                f"Orders: {len(s.trades)}  |  {len(s.open_trades)} open  |  "
                f"{len(s.closed_trades)} closed  |  Win-rate {s.win_rate}%",
                style="dim",
            ),
        )

        # ── Order book — one table, OPEN orders first then CLOSED, live P&L ────
        order_book = ui.render_order_book(s, quote_note=self._quote_note)

        # ── Backend activity log ──────────────────────────────────────────────
        if s.last_decision_time:
            elapsed   = int((datetime.now() - s.last_decision_time).total_seconds())
            remaining = max(0, self.interval - elapsed)
            next_note = (
                f"next decision in {remaining}s · cycle #{s.decision_count} · "
                f"data source: {self._last_source or 'groww'} · "
                f"regime: {self._last_regime or '—'} · "
                f"rule.md lessons written: {self._rules_written}"
            )
        else:
            next_note = "first decision pending · option data: Groww chain"

        log_panel = ui.render_log_panel(
            self.get_logs()[-16:],
            title="Backend Activity — what the engine is doing",
            border_style="dim",
            subtitle=f"[dim]{next_note}[/dim]",
        )

        # ── Assemble ──────────────────────────────────────────────────────────
        return Group(
            Panel(header,  border_style="color(208)", padding=(0, 1),
                  title="[bold color(208)]🙏 Jai Sadguru — Live Option Trader[/bold color(208)]",
                  subtitle=f"[dim]Started {s.start_time.strftime('%H:%M:%S')}  ·  "
                           f"NIFTY F&O · option data via Groww chain  ·  "
                           f"Ctrl+C to exit  ·  "
                           f"Dataset: {self._dataset.dataset_path}[/dim]"),
            Panel(pnl_row, border_style=pnl_color, padding=(0, 1)),
            ui.render_market_strip(self._market_snap, self._market_levels,
                                   title="NIFTY now + tracked levels"),
            order_book,
            log_panel,
        )
