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
import re
import threading
import time
from datetime import datetime, date
from typing import Optional

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import Trade, TradingSession


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
        self._thread = threading.Thread(target=self._loop, daemon=True, name="live-trader")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)
        self._phase = "STOPPED"
        # Save full session case study dataset on exit
        try:
            path = self._dataset.save_session_summary(self.session)
            self._log(f"📁 Session dataset saved → {path}", "OK")
        except Exception as exc:
            self._log(f"Dataset save failed: {exc}", "ERROR")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, msg: str, level: str = "INFO") -> None:
        ts   = datetime.now().strftime("%H:%M:%S")
        icon = {"INFO": "·", "OK": "✓", "WARN": "⚠", "ERROR": "✗", "TRADE": "◆"}.get(level, "·")
        line = f"[{ts}] {icon} {msg}"
        with self._lock:
            self._log_buf.append(line)
            if len(self._log_buf) > 25:
                self._log_buf.pop(0)

    def get_logs(self) -> list[str]:
        with self._lock:
            return list(self._log_buf)

    # ── Price updater ─────────────────────────────────────────────────────────

    def _update_prices(self) -> None:
        """Refresh LTP for all open positions from NSE live data."""
        open_trades = self.session.open_trades
        if not open_trades:
            return

        try:
            from data.nifty_chart import get_option_chart
        except ImportError:
            self._log("Chart module unavailable — skipping price update", "WARN")
            return

        for trade in open_trades:
            try:
                df = get_option_chart(trade.expiry, trade.strike, trade.option_type)
                if df is not None and not df.empty:
                    ltp = float(df["price"].iloc[-1])
                    auto_closed = trade.update_price(ltp)
                    if auto_closed:
                        icon = "🎯 TARGET" if trade.status == "TARGET_HIT" else "🛑 SL"
                        self._log(
                            f"{icon} hit — {trade.id} {int(trade.strike)}{trade.option_type} "
                            f"@ ₹{ltp:.1f}  P&L: {'+' if trade.pnl >= 0 else ''}₹{trade.pnl:,.0f}",
                            "TRADE",
                        )
                        # Dataset: record auto-close
                        self._dataset.on_trade_closed(
                            trade, trade.status, self._last_brief
                        )
                    else:
                        self._log(
                            f"Price update {trade.id}: ₹{ltp:.1f}  "
                            f"P&L: {'+' if trade.pnl >= 0 else ''}₹{trade.pnl:,.0f}",
                            "OK",
                        )
                        # Dataset: record price journey (throttled inside on_price_update)
                        self._dataset.on_price_update(
                            trade, ltp, trade.pnl, self._last_brief
                        )
            except Exception as exc:
                self._log(f"Price fetch failed for {trade.id}: {exc}", "ERROR")

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

        self._log(
            f"⚡ /next: score={prediction.total_score:.1f}/100  "
            f"regime={prediction.regime}  conf={prediction.confidence}  "
            f"bull={prediction.bull_signals}/bear={prediction.bear_signals}  "
            f"action={prediction.action} {prediction.option_type}",
            "OK",
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
                rationale     = str(act.get("rationale","")),
            )
        except (KeyError, ValueError, TypeError) as exc:
            self._log(f"Invalid trade params from {source}: {exc}", "ERROR")
            return

        with self._lock:
            ok, msg = self.session.add_trade(trade)

        level = "TRADE" if ok else "WARN"
        self._log(
            f"{source} → {trade_id} "
            f"{int(trade.strike)}{trade.option_type} {trade.action} "
            f"{trade.qty}L @ ₹{trade.entry_price}  "
            f"SL ₹{trade.sl}  TGT ₹{trade.target} "
            + ("✓" if ok else f"✗ {msg}"),
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
        NEW APPROACH — targeted scan + single AI call.

        Old approach (broken):
          AI calls list_expiries → get_spot_price → get_oi_analysis → get_chart_data
          = 4-5 LLM round-trips, 5000+ tokens, slow, spot=0.0 bug

        New approach:
          Python fetches everything first (MarketScanner, no tokens used)
          AI receives compact pre-digested MarketBrief (~400 chars)
          AI returns ONE JSON action block — no tool calls needed
          = 1 LLM round-trip, ~800 tokens total, fast, accurate
        """
        from agent import Session as AgentSession, FinalAnswerEvent, ErrorEvent

        session = self.session

        # ── Step 1: Python-side market scan (no AI tokens) ────────────────────
        self._log("📡 Scanning market (targeted strikes only)…")
        open_strikes = [(int(t.strike), t.option_type) for t in session.open_trades]
        brief = self._scanner.scan(open_strikes=open_strikes)
        self._last_brief = brief   # store for dataset hooks

        if brief.error:
            self._log(f"Scan error: {brief.error}", "ERROR")
            return

        self._log(
            f"✓ Scan done in {brief.scan_duration_ms}ms  "
            f"spot={brief.spot}  ATM={brief.atm}  PCR={brief.pcr}  "
            f"sentiment={brief.sentiment}  "
            f"strikes scanned: {[f'{t.strike}{t.option_type}' for t in brief.targeted]}",
            "OK",
        )

        # ── Step 2: Build compact portfolio state ─────────────────────────────
        open_pos_summary = ""
        if session.open_trades:
            lines = []
            for t in session.open_trades:
                lines.append(
                    f"  {t.id} {int(t.strike)}{t.option_type} {t.action} {t.qty}L "
                    f"entry=₹{t.entry_price} CMP=₹{t.current_price} "
                    f"PnL={t.pnl:+.0f} SL=₹{t.sl} TGT=₹{t.target}"
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

{"RECOVERY MODE — close only." if session.recovery_mode else ""}

RESPOND ONLY WITH JSON:
```json
{{"actions": [{{"type": "PLACE_TRADE", "expiry": "{brief.expiry}", "strike": 0, "option_type": "CE", "action": "BUY", "qty": 1, "entry_price": 0.0, "sl": 0.0, "target": 0.0, "rationale": "REGIME=X, X confirms: [list]. SL at OI wall. RR=1:X"}}]}}
```
For SELL trades use "action": "SELL" and TWO actions (CE + PE for strangle).
For NO_ACTION: {{"actions": [{{"type": "NO_ACTION", "reason": "REGIME=X because [signals]. No suitable setup."}}]}}
"""
        # ── Step 4: Single AI call — NO tool calls ────────────────────────────
        temp_session = AgentSession()
        final_text   = ""

        try:
            from config import TRADING_SYSTEM_ADDENDUM
            from trading.market_regime import REGIME_TRADING_RULES
            # temp_session is fresh every cycle — no stale history accumulates
            for event in self.agent.run(
                prompt, temp_session,
                system_suffix = TRADING_SYSTEM_ADDENDUM + REGIME_TRADING_RULES,
            ):
                if isinstance(event, FinalAnswerEvent):
                    final_text = event.text
                elif isinstance(event, ErrorEvent):
                    self._log(f"Agent error: {event.message}", "ERROR")
                    return
        except Exception as exc:
            self._log(f"Agent call failed: {exc}", "ERROR")
            return

        self._log(
            f"🤖 AI response received ({len(final_text)} chars)",
            "OK",
        )

        if final_text:
            self._parse_and_execute(final_text)

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
                        rationale     = str(act.get("rationale", "")),
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
                    f"SL ₹{trade.sl}  TGT ₹{trade.target} — "
                    + ("✓" if ok else f"✗ {msg}"),
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
                    f"CLOSE {trade_id} @ ₹{exit_price}  [{reason}] — "
                    + ("✓" if ok else f"✗ {msg}"),
                    "TRADE" if ok else "WARN",
                )
                # Dataset: record manual close
                if ok and closing_trade:
                    self._dataset.on_trade_closed(
                        closing_trade, "MANUAL", self._last_brief
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

                # ── NORMAL SESSION: price refresh + decisions ──────────────
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

            for _ in range(self.interval):
                if self._stop.is_set():
                    break
                time.sleep(1)

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

        # Last /next prediction info
        last_next = ""
        for d in reversed(self._recent_decisions):
            if d.startswith("/next:"):
                parts = d.split(":")
                last_next = f"  ⚡ last: {parts[1] if len(parts)>1 else '?'}"
                break

        header = Table.grid(expand=True, padding=(0, 2))
        header.add_column(ratio=2)
        header.add_column(ratio=2)
        header.add_column(ratio=2)
        header.add_column(ratio=2)
        header.add_row(
            Text(mode_label + last_next,  style=phase_style),
            Text(f"Budget: ₹{s.budget:,.0f}", style="bold white"),
            Text(f"Max Loss: {s.max_loss_pct}%  (₹{s.max_loss_amount:,.0f})", style="yellow"),
            Text(f"NSE {mkt_label}  {mkt_detail}{gift_line}", style=mkt_style),
        )

        # ── P&L summary ───────────────────────────────────────────────────────
        pnl_row = Table.grid(expand=True, padding=(0, 3))
        pnl_row.add_column(ratio=3)
        pnl_row.add_column(ratio=2)
        pnl_row.add_column(ratio=2)
        pnl_row.add_column(ratio=3)
        unreal_sign = "+" if s.unrealized_pnl >= 0 else ""
        real_sign   = "+" if s.realized_pnl   >= 0 else ""
        pnl_row.add_row(
            Text(f"Total P&L: {pnl_sign}₹{s.total_pnl:,.0f}  ({pnl_sign}{pnl_pct:.2f}%)",
                 style=f"bold {pnl_color}"),
            Text(f"Unreal: {unreal_sign}₹{s.unrealized_pnl:,.0f}", style="dim"),
            Text(f"Realised: {real_sign}₹{s.realized_pnl:,.0f}", style="dim"),
            Text(
                f"Trades: {len(s.trades)} total  |  "
                f"{len(s.open_trades)} open  |  "
                f"{len(s.closed_trades)} closed  |  "
                f"Win-rate: {s.win_rate}%",
                style="dim",
            ),
        )

        # ── Active trades table ───────────────────────────────────────────────
        active = Table(
            title="[bold cyan]📊 Active Trades[/bold cyan]",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            border_style="color(208)",
            expand=True,
            show_lines=False,
        )
        for col, w in [
            ("ID",6), ("Expiry",12), ("Strike",8), ("Opt",5),
            ("Act",6), ("Lots",5), ("Entry",9), ("CMP",9),
            ("P&L (₹)",14), ("SL",8), ("Target",8),
        ]:
            active.add_column(col, width=w)

        for t in s.open_trades:
            ps = "+" if t.pnl >= 0 else ""
            pc = "green" if t.pnl >= 0 else "red"
            active.add_row(
                t.id,
                t.expiry,
                str(int(t.strike)),
                t.option_type,
                Text(t.action, style="green" if t.action == "BUY" else "red"),
                str(t.qty),
                f"₹{t.entry_price:.1f}",
                f"₹{t.current_price:.1f}",
                Text(f"{ps}₹{t.pnl:,.0f} ({ps}{t.pnl_pct:.1f}%)", style=f"bold {pc}"),
                f"₹{t.sl:.1f}",
                f"₹{t.target:.1f}",
            )

        if not s.open_trades:
            active.add_row("[dim]—[/dim]", "", "", "", "", "", "", "", "[dim]No active trades[/dim]", "", "")

        # ── Closed trades table ───────────────────────────────────────────────
        closed = Table(
            title="[bold]✅ Closed Trades (last 10)[/bold]",
            box=box.SIMPLE,
            header_style="bold dim",
            expand=True,
        )
        for col, w in [
            ("ID",6), ("Strike",8), ("Opt",5), ("Act",6),
            ("Entry",9), ("Exit",9), ("P&L (₹)",14), ("Status",12),
        ]:
            closed.add_column(col, width=w)

        closed_sorted = sorted(
            s.closed_trades,
            key=lambda t: t.exit_time or datetime.now(),
            reverse=True,
        )[:10]

        for t in closed_sorted:
            ps  = "+" if t.pnl >= 0 else ""
            pc  = "green" if t.pnl >= 0 else "red"
            sc  = {"TARGET_HIT": "green", "SL_HIT": "red"}.get(t.status, "dim")
            closed.add_row(
                t.id,
                str(int(t.strike)),
                t.option_type,
                t.action,
                f"₹{t.entry_price:.1f}",
                f"₹{(t.exit_price or 0):.1f}",
                Text(f"{ps}₹{t.pnl:,.0f} ({ps}{t.pnl_pct:.1f}%)", style=f"bold {pc}"),
                Text(t.status, style=sc),
            )

        if not closed_sorted:
            closed.add_row("[dim]—[/dim]", "", "", "", "", "", "[dim]No closed trades yet[/dim]", "")

        # ── Activity log ──────────────────────────────────────────────────────
        logs      = self.get_logs()[-10:]
        log_lines = "\n".join(logs) if logs else "[dim]Waiting for first decision…[/dim]"

        # Time to next decision
        if s.last_decision_time:
            elapsed   = int((datetime.now() - s.last_decision_time).total_seconds())
            remaining = max(0, self.interval - elapsed)
            next_note = f"  [dim cyan]Next decision in {remaining}s  |  Cycle #{s.decision_count}[/dim cyan]"
        else:
            next_note = "  [dim]First decision pending…[/dim]"

        log_panel = Panel(
            log_lines + "\n" + next_note,
            title="[dim]Activity Log[/dim]",
            border_style="dim",
            padding=(0, 1),
        )

        # ── Assemble ──────────────────────────────────────────────────────────
        return Group(
            Panel(header,  border_style="color(208)", padding=(0, 1),
                  title=f"[bold color(208)]🙏 Jai Sadguru — Live Trading Mode[/bold color(208)]",
                  subtitle=f"[dim]Started {s.start_time.strftime('%H:%M:%S')}  ·  "
                           f"Press Ctrl+C to exit live mode  ·  "
                           f"Dataset: {self._dataset.dataset_path}[/dim]"),
            Panel(pnl_row, border_style=pnl_color, padding=(0, 1)),
            active,
            closed,
            log_panel,
        )
