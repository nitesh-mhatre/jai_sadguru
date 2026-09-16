"""
trading/sim_mode.py
-------------------
Simulation mode — replay past NIFTY data through the full trading pipeline.

Pipeline per cycle (identical code path to live mode):
  1. Feed window of candles up to "now" (replay clock)   → preprocessing
  2. Kronos forecasts the next candles                   → quantitative vote
  3. LLM (NVIDIA NIM) reads brief + Kronos block         → reasoning vote
  4. Voting merges votes → decision (hold / place / close orders)
  5. Orders flow to the market layer (TradingSession = order table)

Data:
  • Index candles from Yahoo (data/groww_feed.fetch_index_candles)
  • PE/CE premium candles for the USER-SELECTED levels from Groww
    (falls back to an intrinsic+time-value model when unavailable)
  • The last candle of each replay window is the "live" candle: only its
    OPEN is final. It is used to validate the previous prediction and is
    never fed to the forecaster.

At session end the core distils results into rule.md lessons.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .engine import Trade, TradingSession

log = logging.getLogger(__name__)


# ── Data download ─────────────────────────────────────────────────────────────

def download_history(days: int = 7, interval: str = "5m") -> pd.DataFrame:
    """Download past NIFTY index candles (Yahoo via groww_feed)."""
    from data.groww_feed import fetch_index_candles
    hist = fetch_index_candles(days=days, interval=interval)
    if hist is None or hist.empty:
        raise RuntimeError(
            "No historical NIFTY data returned — check network / try again"
        )
    return hist


def pick_strikes(spot: float, n: int = 4, step: int = 50) -> list[int]:
    """Default strikes around spot when the user didn't select levels."""
    base = int(round(spot / step) * step)
    half = max(2, n // 2)
    strikes = sorted({base - step * (half - i) for i in range(half)}
                     | {base + step * (i + 1) for i in range(n - half)})
    return sorted(strikes)[:n]


# ── Fallback premium model (only when Groww premium candles unavailable) ──────

def _fallback_premium(strike: int, spot: float, side: str, minutes_left: int,
                      base_vol_pts: float = 55.0) -> float:
    intrinsic = max(0.0, spot - strike) if side == "CE" else max(0.0, strike - spot)
    tv = base_vol_pts * (max(minutes_left, 0) / 375.0) ** 0.5
    return round(intrinsic + tv, 2)


# ── Result structures ─────────────────────────────────────────────────────────

@dataclass
class SimCycleRecord:
    cycle:        int
    clock:        datetime
    spot:         float
    kronos_dir:   str
    llm_dir:      str
    decision:     str
    realized_pts: float = 0.0     # actual move seen by next cycle (filled later)
    correct:      bool = False


@dataclass
class SimResult:
    started_at:   datetime
    ended_at:     datetime
    days:         int
    interval:     str
    expiry:       str
    levels:       list
    cycles:       int
    kronos_agree: int
    kronos_total: int
    llm_correct:  int
    llm_total:    int
    total_pnl:    float
    win_rate:     float
    trades:       int
    regime:       str
    data_source:  str = "groww"
    lessons:      list[str] = field(default_factory=list)


# ── The simulation engine ─────────────────────────────────────────────────────

class SimTrader:
    """
    Candle-by-candle replay: preprocess → Kronos → LLM → voting → orders.

    Usage:
        sim    = SimTrader(session, agent, config, log_fn=print,
                           expiry="26-Jun-2025", levels=[...])
        result = sim.run(days=3, interval="5m")
    """

    def __init__(self, session: TradingSession, agent, config,
                 log_fn=None, speed: float | None = None,
                 expiry: str = "", levels: list[dict] | None = None):
        self.session = session
        self.agent   = agent
        self.config  = config
        self._log_fn = log_fn or (lambda msg, level="INFO": None)
        self.speed   = SIM_SPEED if speed is None else speed
        self._stop   = False
        self.expiry  = expiry
        self.levels  = levels or []          # [{strike:int, side:'CE'|'PE'}]
        self._llm_fail_streak = 0            # bypass LLM when backend is down
        self._resolved_levels: list[tuple[int, str]] = []

        self.records: list[SimCycleRecord] = []
        self._kronos_agree = 0
        self._kronos_total = 0
        self._llm_correct  = 0
        self._llm_total    = 0
        self._prev_prediction: dict | None = None
        self.open_orders: dict[str, dict] = {}   # "24500CE" → order info

    # ── Control ───────────────────────────────────────────────────────────────

    def stop(self):
        self._stop = True

    def _log(self, msg: str, level: str = "INFO"):
        self._log_fn(msg, level)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, days: int = 3, interval: str = "5m",
            pred_len: int = 3, max_cycles: int = 60) -> SimResult:
        started = datetime.now()

        # ── 1. Download index candles ─────────────────────────────────────
        self._log(f"⬇  Downloading {days}d of NIFTY {interval} index candles…")
        hist = download_history(days, interval)
        if hist.empty:
            raise RuntimeError("Historical download empty")
        self._log(f"✓ {len(hist)} index candles downloaded", "OK")

        # ── 2. Resolve levels (user-selected or auto) ─────────────────────
        first_spot = float(hist["close"].iloc[0])
        if self.levels:
            resolved = [(int(l["strike"]), str(l["side"]).upper()) for l in self.levels]
        else:
            strikes = pick_strikes(first_spot, n=4)
            resolved = [(strikes[0], "PE"), (strikes[1], "PE"),
                        (strikes[-2], "CE"), (strikes[-1], "CE")]
        self._resolved_levels = resolved
        self._log(f"✓ Tracking levels: " + ", ".join(f"{s}{sd}" for s, sd in resolved), "OK")

        # ── 3. Download PE/CE premium candles for each level ──────────────
        premium_data = self._download_premiums(resolved, interval)
        data_source  = "groww" if premium_data else "model-fallback"

        # ── 4. Replay ─────────────────────────────────────────────────────
        window   = 60
        step_min = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}.get(interval, 5)
        cycle    = 0

        for i in range(window, len(hist) - 1):
            if self._stop or cycle >= max_cycles:
                break
            cycle += 1

            # Replay window: closed candles + the "live" open candle
            chunk       = hist.iloc[i - window:i + 1]
            past        = chunk.iloc[:-1]
            live_candle = chunk.iloc[-1]

            ts_idx = past.index[-1]
            now_clock = (pd.Timestamp(ts_idx).to_pydatetime()
                         if not isinstance(ts_idx, pd.Timestamp)
                         else ts_idx.to_pydatetime())

            spot        = float(live_candle["open"])   # open-candle rule
            closed_spot = float(past["close"].iloc[-1])
            minutes_left = 375 - _minutes_since_open(now_clock)

            # ── Validate previous prediction against this open ────────────
            self._validate_previous(spot)

            # ── Preprocessing ─────────────────────────────────────────────
            momentum = _window_momentum(past)

            # ── Kronos vote on CLOSED candles only ────────────────────────
            from .kronos_forecast import forecast_candles, kronos_vote
            ts_col = "date" if "date" in past.columns else ""
            kf = forecast_candles(past, pred_len=pred_len, interval=interval,
                                  timestamp_col=ts_col)
            k_vote, k_reason = kronos_vote(kf)

            # ── LLM vote (bypassed after 3 straight failures) ─────────────
            llm_dir, llm_reason = self._llm_vote_safe(
                spot, momentum, kf, now_clock, resolved,
                premium_data, closed_spot, minutes_left)

            # ── Voting → decision ─────────────────────────────────────────
            decision = _vote_decision(k_vote, llm_dir)

            # ── Market layer: execute on the order table ──────────────────
            self._execute_orders(decision, closed_spot, minutes_left, resolved,
                                 premium_data, now_clock)

            rec = SimCycleRecord(
                cycle=cycle, clock=now_clock, spot=spot,
                kronos_dir=kf.direction if kf else "N/A",
                llm_dir=llm_dir, decision=decision,
            )
            self.records.append(rec)
            if self._prev_prediction:
                self._prev_prediction["record"] = rec
            self._prev_prediction = {
                "spot": spot, "direction": llm_dir, "record": rec,
            }

            self._log(
                f"[{now_clock:%d-%b %H:%M}] cycle {cycle}/{max_cycles}  spot={spot:.0f}  "
                f"Kronos={rec.kronos_dir}  LLM={llm_dir} ({llm_reason[:35]})  → {decision}",
                "OK",
            )

            if self.speed > 0:
                time.sleep(self.speed)

        # ── Validate the final prediction with the last close ─────────────
        if self._prev_prediction and len(hist):
            actual = float(hist["close"].iloc[-1]) - self._prev_prediction["spot"]
            rec    = self._prev_prediction.get("record")
            exp    = self._prev_prediction["direction"]
            if rec is not None:
                rec.realized_pts = round(actual, 2)
                rec.correct = ((exp == "BULLISH" and actual > 0)
                               or (exp == "BEARISH" and actual < 0)
                               or (exp == "SIDEWAYS" and abs(actual) < 15))
                if rec.correct:
                    self._llm_correct += 1
                self._llm_total += 1

        ended  = datetime.now()
        result = self._build_result(started, ended, days, interval, data_source)
        self._learn_rules(result)
        return result

    # ── Premium data download ─────────────────────────────────────────────────

    def _download_premiums(self, resolved: list[tuple[int, str]],
                           interval: str) -> dict[tuple[int, str], pd.DataFrame]:
        """Download PE/CE premium candles per level from Groww."""
        from data.groww_feed import fetch_option_candles
        out: dict[tuple[int, str], pd.DataFrame] = {}
        for strike, side in resolved:
            if self._stop:
                break
            with ui_status(f"Downloading {strike}{side} premiums…"):
                df = fetch_option_candles(self.expiry, strike, side,
                                          interval=interval, limit=800)
            if df is not None and not df.empty:
                out[(strike, side)] = df
                self._log(f"✓ {strike}{side}: {len(df)} premium candles", "OK")
            else:
                self._log(f"⚠ {strike}{side}: no premium data — using model fallback", "WARN")
        return out

    def _premium_at(self, strike: int, side: str, when: datetime,
                    premium_data: dict, spot: float, minutes_left: int) -> float:
        """Real premium at `when` if available, else model fallback."""
        df = premium_data.get((strike, side))
        if df is not None and not df.empty and "timestamp" in df.columns:
            mask = df["timestamp"] <= pd.Timestamp(when)
            if mask.any():
                return float(df.loc[mask, "close"].iloc[-1])
            return float(df["close"].iloc[0])
        return _fallback_premium(strike, spot, side, minutes_left)

    # ── Validation (live-candle rule) ─────────────────────────────────────────

    def _validate_previous(self, spot_now: float) -> None:
        prev = self._prev_prediction
        if not prev or prev.get("validated"):
            return
        actual = spot_now - prev["spot"]
        rec    = prev.get("record")
        exp    = prev["direction"]
        if rec is not None:
            rec.realized_pts = round(actual, 2)
            rec.correct = ((exp == "BULLISH" and actual > 0)
                           or (exp == "BEARISH" and actual < 0)
                           or (exp == "SIDEWAYS" and abs(actual) < 15))
            if rec.correct:
                self._llm_correct += 1
            self._llm_total += 1
            # Track Kronos agreement with realized direction each cycle
            if rec.kronos_dir in ("BULLISH", "BEARISH"):
                self._kronos_total += 1
                if ((rec.kronos_dir == "BULLISH" and actual > 0)
                        or (rec.kronos_dir == "BEARISH" and actual < 0)):
                    self._kronos_agree += 1
        prev["validated"] = True

    # ── LLM vote with fail-streak bypass ──────────────────────────────────────

    def _llm_vote_safe(self, spot, momentum, kf, now_clock, resolved,
                       premium_data, closed_spot, minutes_left) -> tuple[str, str]:
        if self._llm_fail_streak >= 3:
            return self._fallback_direction(momentum), "LLM bypassed (failing) — momentum-based"

        levels_txt = ", ".join(f"{s}{sd}₹{self._premium_at(s, sd, now_clock, premium_data, closed_spot, minutes_left):.1f}"
                               for s, sd in resolved)
        from .kronos_forecast import KronosForecast  # noqa: F401  (type hint only)
        kf_block = kf.to_prompt_block() if kf else "Kronos unavailable this cycle."
        prompt = f"""SIMULATION CYCLE — NIFTY replay {now_clock:%d-%b-%Y %H:%M}

spot (live candle open — only confirmed value): {spot:.2f}
tracked levels: {levels_txt}

recent momentum (closed candles): {momentum}

{kf_block}

Vote on the NEXT cycle direction. Reply ONLY with JSON:
{{"direction": "BULLISH|BEARISH|SIDEWAYS", "reason": "≤12 words citing data above"}}"""

        from agent import Session, FinalAnswerEvent, ErrorEvent, extract_json
        temp = Session()
        try:
            for ev in self.agent.run(prompt, temp,
                                     system_suffix="Reply only with the JSON vote."):
                if isinstance(ev, FinalAnswerEvent):
                    data = extract_json(ev.text) or {}
                    d = str(data.get("direction", "")).upper()
                    if d in ("BULLISH", "BEARISH", "SIDEWAYS"):
                        self._llm_fail_streak = 0
                        return d, str(data.get("reason", ""))[:60]
                    self._llm_fail_streak += 1
                    return self._fallback_direction(momentum), "unparseable LLM vote — momentum fallback"
                elif isinstance(ev, ErrorEvent):
                    self._llm_fail_streak += 1
                    if self._llm_fail_streak == 3:
                        self._log("⚠ LLM failing repeatedly — switching to momentum-only votes", "WARN")
                    return self._fallback_direction(momentum), f"LLM error: {ev.message[:40]}"
        except Exception as exc:
            self._llm_fail_streak += 1
            log.warning("LLM vote failed: %s", exc)
            return self._fallback_direction(momentum), "LLM exception"
        return self._fallback_direction(momentum), "no LLM response"

    def _fallback_direction(self, momentum: str) -> str:
        m = momentum.lower()
        if "change -" in m or "change −" in m:
            return "BEARISH"
        if "change +" in m:
            return "BULLISH"
        return "SIDEWAYS"

    # ── Order execution on the market layer ───────────────────────────────────

    def _execute_orders(self, decision: str, spot: float, minutes_left: int,
                        resolved: list, premium_data: dict, now_clock) -> None:
        side_needed = {"BULLISH": "CE", "BEARISH": "PE"}.get(decision)

        # Close orders that no longer match the voted direction
        for key in list(self.open_orders):
            o = self.open_orders[key]
            if side_needed is None or o["side"] != side_needed:
                exit_price = self._premium_at(o["strike"], o["side"], now_clock,
                                              premium_data, spot, minutes_left)
                ok, msg = self.session.close_trade(o["trade_id"], exit_price)
                self._log(f"   close {o['trade_id']} {o['strike']}{o['side']} @ ₹{exit_price:.1f}"
                          f" — {'✓' if ok else msg}", "TRADE")
                del self.open_orders[key]

        if side_needed is None:
            return

        # Open up to 2 orders in the voted direction
        want = [(s, sd) for (s, sd) in resolved if sd == side_needed][:2]
        for strike, side in want:
            key = f"{strike}{side}"
            if key in self.open_orders:
                continue
            entry = self._premium_at(strike, side, now_clock, premium_data,
                                     spot, minutes_left)
            trade = Trade(
                id            = f"S{len(self.session.trades)+1:03d}",
                expiry        = self.expiry or "SIM",
                strike        = float(strike),
                option_type   = side,
                action        = "BUY",
                qty           = 1,
                entry_price   = entry,
                current_price = entry,
                sl            = round(entry * 0.75, 2),
                target        = round(entry * 1.5, 2),
                rationale     = f"sim vote {decision} @ spot {spot:.0f}",
            )
            ok, msg = self.session.add_trade(trade)
            if ok:
                self.open_orders[key] = {"trade_id": trade.id, "strike": strike, "side": side}
                self._log(f"   open {trade.id} {strike}{side} @ ₹{entry:.1f}", "TRADE")

    # ── Result + learning ─────────────────────────────────────────────────────

    def _build_result(self, started, ended, days, interval, data_source) -> SimResult:
        s = self.session
        bull = sum(1 for r in self.records if r.llm_dir == "BULLISH")
        bear = sum(1 for r in self.records if r.llm_dir == "BEARISH")
        n    = max(1, len(self.records))
        if bull > n * 0.6:
            regime = "DIRECTIONAL_BULL"
        elif bear > n * 0.6:
            regime = "DIRECTIONAL_BEAR"
        elif bull + bear < n * 0.4:
            regime = "SIDEWAYS"
        else:
            regime = "MIXED"
        return SimResult(
            started_at=started, ended_at=ended, days=days, interval=interval,
            expiry=self.expiry or "N/A",
            levels=[f"{s}{sd}" for s, sd in getattr(self, "_resolved_levels", [])] or
                    [f"{l['strike']}{l['side']}" for l in self.levels] or ["auto"],
            cycles=len(self.records),
            kronos_agree=self._kronos_agree, kronos_total=self._kronos_total,
            llm_correct=self._llm_correct, llm_total=self._llm_total,
            total_pnl=s.total_pnl, win_rate=s.win_rate,
            trades=len(s.trades), regime=regime, data_source=data_source,
        )

    def _learn_rules(self, result: SimResult) -> None:
        from .rules import record_session_outcome
        lessons: list[str] = []
        wrong = [r for r in self.records if not r.correct and r.realized_pts]
        if wrong:
            w = wrong[0]
            lessons.append(
                f"Replay {w.clock:%d-%b %H:%M}: voted {w.llm_dir} but moved {w.realized_pts:+.0f}pts "
                f"(Kronos said {w.kronos_dir}) — check OI wall conflict before trusting momentum votes"
            )
        if result.kronos_total:
            ka = result.kronos_agree / result.kronos_total * 100
            if ka >= 60:
                lessons.append(f"Kronos agreed {ka:.0f}% of cycles — weight its vote higher in regime {result.regime}")
            elif ka <= 35:
                lessons.append(f"Kronos agreed only {ka:.0f}% of cycles — discount it in regime {result.regime}")
        result.lessons = lessons[:3]
        try:
            record_session_outcome(
                regime=result.regime,
                kronos_agree=result.kronos_agree,
                kronos_disagree=result.kronos_total - result.kronos_agree,
                total_pnl=result.total_pnl,
                trade_count=result.trades,
                win_rate=result.win_rate,
                ai_summary="; ".join(lessons[:3]),
            )
        except Exception as exc:
            log.warning("Rule recording failed: %s", exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

SIM_SPEED = 0.0


def _minutes_since_open(dt: datetime) -> int:
    minutes = dt.hour * 60 + dt.minute
    return max(0, minutes - (9 * 60 + 15))


def _window_momentum(past: pd.DataFrame) -> str:
    """Simple preprocessing summary of the closed-candle window."""
    c = past["close"].tail(12)
    if len(c) < 6:
        return "insufficient data"
    change = float(c.iloc[-1] - c.iloc[0])
    hi = float(past["high"].tail(12).max())
    lo = float(past["low"].tail(12).min())
    rng = hi - lo
    pos = (float(c.iloc[-1]) - lo) / rng if rng else 0.5
    return (f"12-candle change {change:+.1f}pts, "
            f"close at {pos*100:.0f}% of range (H {hi:.0f}/L {lo:.0f})")


def _vote_decision(k_vote: int, llm_dir: str) -> str:
    """
    Voting layer: Kronos (+1/-1/0) + LLM direction.
    HOLD when they conflict; follow LLM when Kronos neutral.
    """
    llm_vote = {"BULLISH": 1, "BEARISH": -1}.get(llm_dir, 0)
    total = k_vote + llm_vote
    if total >= 1:
        return "BULLISH"
    if total <= -1:
        return "BEARISH"
    if llm_vote != 0 and k_vote == 0:
        return llm_dir
    return "HOLD"


def ui_status(text: str):
    """Context-manager status line; safe when stdout is redirected."""
    try:
        import ui as _ui
        return _ui.console.status(text, spinner="dots2")
    except Exception:
        import contextlib
        return contextlib.nullcontext()
