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
  • PE/CE premium series for the USER-SELECTED levels, built on the SAME time
    frame and candle size as the index. Free public sources do not serve
    per-contract option history (verified: Groww's delayed chart route returns
    candles only for the FNO/ CASH underlyings, and the NSE option-chart route
    is cookie-walled), so each level is anchored to its REAL Groww chain quote
    (ltp / delta / theta) and moved along the index grid by that contract's own
    delta, then decayed by its own theta.
  • The last candle of each replay window is the "live" candle: only its
    OPEN is final. It is used to validate the previous prediction and is
    never fed to the forecaster.

At session end the core distils results into rule.md lessons.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd

from .engine import NIFTY_LOT_SIZE, Trade, TradingSession

log = logging.getLogger(__name__)


# ── Data download ─────────────────────────────────────────────────────────────

_OHLCV = ["open", "high", "low", "close", "volume"]


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a tz-naive IST DatetimeIndex plus lower-case OHLCV columns.

    yfinance hands the frame back with a RangeIndex after reset_index(), which
    silently made the replay clock read 1970-01-01 (logged as '01-Jan 00:00').
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for c in _OHLCV:
        if c not in out.columns:
            out[c] = 0.0

    ts_col = next((c for c in ("datetime", "date", "timestamp", "index")
                   if c in out.columns), None)
    if ts_col:
        ts = pd.to_datetime(out[ts_col], errors="coerce")
        try:
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
        except Exception:
            pass
        out = out.assign(_ts=ts).dropna(subset=["_ts"])
        out = out.set_index("_ts")
    elif not isinstance(out.index, pd.DatetimeIndex):
        raise RuntimeError(
            "Historical NIFTY data has no recognisable datetime column "
            f"(got {list(df.columns)})"
        )

    return out.sort_index()[_OHLCV].astype(float)


def download_history(days: int = 7, interval: str = "5m") -> pd.DataFrame:
    """Download past NIFTY index candles (Yahoo via groww_feed)."""
    from data.groww_feed import fetch_index_candles
    hist = fetch_index_candles(days=days, interval=interval)
    if hist is None or hist.empty:
        raise RuntimeError(
            "No historical NIFTY data returned — check network / try again"
        )
    return _normalise_index(hist)


def _align_premium(df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """
    Put a premium series on the SAME timestamp grid as the NIFTY candles, so
    both are analysed on one time frame and candle size. Missing candles are
    forward-filled (the option simply did not trade that minute).
    """
    if df is None or df.empty or hist is None or hist.empty:
        return pd.DataFrame()
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
    lo, hi = hist.index[0], hist.index[-1]
    d = d[(d["timestamp"] >= lo) & (d["timestamp"] <= hi)]
    if d.empty:
        return pd.DataFrame()
    src = str(d["source"].iloc[0]) if "source" in d.columns else "live"
    d = d.set_index("timestamp")
    d = d[~d.index.duplicated(keep="last")]
    aligned = d.reindex(hist.index, method="ffill")
    aligned["close"] = aligned["close"].bfill()
    aligned = aligned.dropna(subset=["close"])
    if aligned.empty:
        return pd.DataFrame()
    aligned["source"] = src
    return aligned.reset_index()


def _intrinsic(strike: float, spot: float, side: str) -> float:
    return max(0.0, spot - strike) if side == "CE" else max(0.0, strike - spot)


def _anchored_premium_series(strike: int, side: str, hist: pd.DataFrame,
                             leg: dict) -> pd.DataFrame:
    """
    Index-aligned premium series anchored to the REAL Groww chain quote for the
    selected level. The premium travels with the NIFTY candle grid through the
    contract's own delta and decays with its own theta, so the replay prices the
    leg the way the market actually prices it instead of an invented flat vol.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()

    p0    = float(leg.get("ltp") or 0.0)
    delta = float(leg.get("delta") or 0.0)
    theta = float(leg.get("theta") or 0.0)
    if not delta:                       # no greeks → ATM-ish sensitivity
        delta = 0.5 if side == "CE" else -0.5

    step_min = 0.0
    if len(hist) > 1:
        try:
            step_min = (hist.index[1] - hist.index[0]).total_seconds() / 60.0
        except Exception:
            step_min = 0.0

    s0 = float(hist["close"].iloc[0])
    rows = []
    for i, (ts, row) in enumerate(hist.iterrows()):
        spot = float(row["close"])
        elapsed = i * step_min                      # trading minutes into replay
        p = p0 + delta * (spot - s0) + theta * (elapsed / 375.0)
        p = max(p, _intrinsic(strike, spot, side) + 0.05)
        rows.append({"timestamp": ts, "open": p, "high": p, "low": p,
                     "close": p, "volume": 0, "source": "groww-anchored"})
    return pd.DataFrame(rows)


def _model_premium_series(strike: int, side: str, hist: pd.DataFrame,
                          chain_leg: dict | None = None,
                          base_vol_pts: float = 55.0) -> pd.DataFrame:
    """
    Index-aligned premium series on the SAME time frame + candle size as NIFTY.

    Preferred: anchored to the live Groww chain quote (real ltp / delta / theta).
    Last resort: the BLACK fallback model when the chain has no quote for the leg.
    """
    if chain_leg and float(chain_leg.get("ltp") or 0.0) > 0:
        anchored = _anchored_premium_series(strike, side, hist, chain_leg)
        if not anchored.empty:
            return anchored

    rows = []
    for ts, row in hist.iterrows():
        spot = float(row["close"])
        minutes_left = max(1, 375 - _minutes_since_open(ts))
        p = _fallback_premium(strike, spot, side, minutes_left, base_vol_pts)
        rows.append({"timestamp": ts, "open": p, "high": p, "low": p,
                     "close": p, "volume": 0, "source": "model"})
    return pd.DataFrame(rows)


def plan_replay(n: int, window: int | None = None,
                max_cycles: int | None = None) -> tuple[int, int]:
    """
    Replay geometry for a series of `n` candles → (window_bars, decisions).

    The window slides ONE candle per decision, so `decisions = n - window - 1`:
    the model sees bars 0…window, decides, then bars 1…window+1, decides, and so
    on until the data ends. A 500-candle download with a 200-bar window gives
    299 decisions. `max_cycles=None` means "replay everything".
    """
    n   = max(3, int(n))
    win = window or min(SIM_WINDOW_BARS,
                        max(SIM_MIN_WINDOW_BARS, n - SIM_MIN_DECISIONS))
    win = max(2, min(int(win), n - 2))
    total = max(1, n - win - 1)
    cycles = total if max_cycles is None else max(1, min(int(max_cycles), total))
    return win, cycles


def _chain_snapshot_safe(expiry: str = "") -> dict:
    """Groww chain snapshot (spot / PCR / walls). Never raises."""
    try:
        from data.groww_feed import get_option_chain_snapshot
        return get_option_chain_snapshot(expiry) or {}
    except Exception as exc:
        log.warning("Chain snapshot failed: %s", exc)
        return {}


def _premium_source(df: pd.DataFrame) -> str:
    try:
        if df is not None and not df.empty and "source" in df.columns:
            return str(df["source"].iloc[0])
    except Exception:
        pass
    return "model"


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
    llm_ok:       bool = True     # False when the vote came from a fallback


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
        """See run() for the window/candle-count semantics of the replay."""
        self.session = session
        self.agent   = agent
        self.config  = config
        self._log_fn = log_fn                # None → logs only buffered for the dashboard
        self.speed   = SIM_SPEED if speed is None else speed
        self._stop   = False
        self.expiry  = expiry
        self.levels  = levels or []          # [{strike:int, side:'CE'|'PE'}]
        self._llm_fail_streak = 0            # bypass LLM when backend is down
        self._resolved_levels: list[tuple[int, str]] = []

        # Dashboard / progress state
        self._lock      = threading.Lock()
        self._log_buf: list[str] = []
        self._phase     = "STARTING"
        self._days      = 0
        self._interval  = "5m"
        self._max_cycles = 60
        self._data_source = ""
        self._window      = SIM_WINDOW_BARS
        self._premium_sources: dict[str, str] = {}
        self._level_quotes: dict[tuple[int, str], dict] = {}
        self._chain_spot = 0.0
        self._chain_snap: dict = {}       # live chain snapshot for the market strip

        self.records: list[SimCycleRecord] = []
        self._kronos_agree = 0
        self._kronos_total = 0
        self._llm_correct  = 0
        self._llm_total    = 0
        self._prev_prediction: dict | None = None
        self.open_orders: dict[str, dict] = {}   # "24500CE" → order info

    # ── Control ───────────────────────────────────────────────────────────────

    def stop(self):
        self._stop  = True
        self._phase = "STOPPING"

    _ICONS = {"INFO": "·", "OK": "✓", "WARN": "⚠", "ERROR": "✗", "TRADE": "◆",
              "PLAN": "📋", "DATA": "🗄", "AI": "🤖", "RULE": "📖"}

    def _log(self, msg: str, level: str = "INFO"):
        icon = self._ICONS.get(level, "·")
        line = f"[{datetime.now():%H:%M:%S}] {icon} {msg}"
        with self._lock:
            self._log_buf.append(line)
            if len(self._log_buf) > 300:
                self._log_buf.pop(0)
        if self._log_fn:
            self._log_fn(msg, level)

    def get_logs(self) -> list[str]:
        with self._lock:
            return list(self._log_buf)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, days: int = 3, interval: str = "5m",
            pred_len: int = 3, max_cycles: int | None = None,
            window: int | None = None) -> SimResult:
        """
        Slide a `window`-candle view across the whole downloaded series, one
        candle at a time, deciding at every step:

            bars 0 … window        → decision 1
            bars 1 … window + 1    → decision 2
            bars 2 … window + 2    → decision 3   … until the data runs out

        `max_cycles=None` (default) replays the ENTIRE series — the old
        hard-coded 60 cut the run short no matter how much data was downloaded.
        """
        started = datetime.now()
        self._phase        = "RUNNING"
        self._days         = days
        self._interval     = interval

        # ── 1. Index candles + Groww chain, fetched IN PARALLEL ───────────
        #    The index fixes the reference time frame / candle size; the chain
        #    supplies each level's real quote. Neither depends on the other, so
        #    both start together instead of serialising two network round-trips.
        self._log(
            f"Downloading {days}d of NIFTY {interval} index candles + Groww "
            f"option chain in parallel…",
            "DATA",
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            hist_fut   = pool.submit(download_history, days, interval)
            snap_fut   = pool.submit(_chain_snapshot_safe, self.expiry)
            hist       = hist_fut.result()
            chain_snap = snap_fut.result()
        if hist.empty:
            raise RuntimeError("Historical download empty")
        self._chain_snap = chain_snap or {}
        self._chain_spot = float(self._chain_snap.get("spot") or 0.0)
        self._log(
            f"{len(hist)} NIFTY candles {hist.index[0]:%d-%b %H:%M} → "
            f"{hist.index[-1]:%d-%b %H:%M} ({interval}) · "
            f"chain spot ≈ {self._chain_spot:,.0f} · PCR {chain_snap.get('pcr', '—')}",
            "OK",
        )

        # ── 2. Resolve levels (user-selected or auto) ─────────────────────
        first_spot = float(hist["close"].iloc[0])
        if self.levels:
            resolved = [(int(l["strike"]), str(l["side"]).upper()) for l in self.levels]
        else:
            strikes = pick_strikes(first_spot, n=4)
            resolved = [(strikes[0], "PE"), (strikes[1], "PE"),
                        (strikes[-2], "CE"), (strikes[-1], "CE")]
        self._resolved_levels = resolved
        self._log("Tracking levels: " + ", ".join(f"{s}{sd}" for s, sd in resolved), "OK")

        # ── 3. Premium candles for each level, on the SAME grid as NIFTY ──
        premium_data = self._download_premiums(resolved, interval, days, hist,
                                               chain_snap=chain_snap)
        srcs    = [_premium_source(df) for df in premium_data.values()]
        live    = sum(1 for s in srcs if s in ("groww", "nse"))
        anchored = sum(1 for s in srcs if s == "groww-anchored")
        modeled = len(srcs) - live - anchored
        data_source = "live" if live else ("groww-anchored" if anchored else "model-fallback")
        self._data_source = data_source
        self._log(
            f"Data ready: {len(hist)} NIFTY candles ({interval}) + "
            f"{len(premium_data)} level series — {live} live, {anchored} Groww-anchored, "
            f"{modeled} model · all on the same time frame",
            "DATA",
        )

        # ── 4. Replay: slide the window over the whole series, 1 candle/step ──
        n  = len(hist)
        win, max_cycles = plan_replay(n, window, max_cycles)
        self._window     = win
        self._max_cycles = max_cycles

        self._log(
            f"Replay plan: {n} candles · sliding window {win} bars · step 1 "
            f"candle → {max_cycles} decisions "
            f"(bars {win}→{win + max_cycles - 1} of {n - 1})",
            "OK",
        )

        cycle = 0
        for i in range(win, n - 1):
            if self._stop or cycle >= max_cycles:
                break
            cycle += 1

            # Replay window: closed candles + the "live" open candle
            chunk       = hist.iloc[i - win:i + 1]
            past        = chunk.iloc[:-1]
            live_candle = chunk.iloc[-1]

            now_clock = _to_pydatetime(past.index[-1])

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
            llm_dir, llm_reason, llm_ok = self._llm_vote_safe(
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
                llm_dir=llm_dir, decision=decision, llm_ok=llm_ok,
            )
            self.records.append(rec)
            if self._prev_prediction:
                self._prev_prediction["record"] = rec
            self._prev_prediction = {
                "spot": spot, "direction": llm_dir, "record": rec,
            }

            # Make a fallback vote unmistakable: ✗BEARISH* = LLM failed, the
            # direction came from price momentum — NOT from the model.
            llm_label = llm_dir if llm_ok else f"✗{llm_dir}*"
            if not llm_ok and self._llm_fail_streak == 1:
                self._log(
                    "LLM unavailable — votes fall back to price momentum "
                    "(shown with *). Decisions still run, but treat them as "
                    "momentum-only, not model-backed.",
                    "WARN",
                )
            self._log(
                f"cycle {cycle}/{max_cycles} [{now_clock:%d-%b %H:%M}] spot={spot:.0f}  "
                f"Kronos={rec.kronos_dir}  LLM={llm_label}  → {decision}  "
                f"({llm_reason[:40]})",
                "OK" if llm_ok else "WARN",
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
        self._phase = "DONE"
        return result

    # ── Premium data download ─────────────────────────────────────────────────

    def _download_premiums(self, resolved: list[tuple[int, str]], interval: str,
                           days: int, hist: pd.DataFrame,
                           chain_snap: dict | None = None
                           ) -> dict[tuple[int, str], pd.DataFrame]:
        """
        Fetch premium candles for every level IN PARALLEL, then align each series
        to the NIFTY candle grid (identical time frame + candle size). When a
        contract has no live history, an index-aligned model series is built so
        the level still takes part in the analysis instead of silently dropping.
        """
        from data.groww_feed import fetch_option_candles
        out: dict[tuple[int, str], pd.DataFrame] = {}
        if not resolved:
            return out

        # ── Real Groww chain quotes for the selected levels ───────────────
        # These give the true premium scale (ltp) plus the contract's own
        # delta/theta/IV/OI — the series every level is anchored to.
        try:
            from data.groww_feed import get_quote
            snap = chain_snap or _chain_snapshot_safe(self.expiry)
            self._chain_spot = float(snap.get("spot") or 0.0)
            for s, sd in resolved:
                q = get_quote(self.expiry, s, sd)
                if q and float(q.get("ltp") or 0.0) > 0:
                    self._level_quotes[(s, sd)] = q
            if self._level_quotes:
                self._log(
                    f"Groww chain: spot ≈ {self._chain_spot:,.0f} · "
                    f"{len(self._level_quotes)}/{len(resolved)} level quotes live "
                    f"(ltp/OI/IV/delta/theta)",
                    "DATA",
                )
            else:
                self._log(
                    "Groww chain returned no quote for the selected levels "
                    "(expiry may differ from the live chain) — using model grid",
                    "WARN",
                )
        except Exception as exc:
            self._log(f"Groww chain quotes unavailable — {exc}", "WARN")

        self._log(
            f"Downloading {len(resolved)} level series in parallel "
            f"(NIFTY {len(hist)} × {interval} as the reference grid)…",
            "DATA",
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(resolved)) as pool:
            futures = {
                pool.submit(fetch_option_candles, self.expiry, s, sd,
                            interval, 800, days): (s, sd)
                for s, sd in resolved
            }
            for fut in concurrent.futures.as_completed(futures):
                strike, side = futures[fut]
                label = f"{strike}{side}"
                try:
                    df = fut.result()
                except Exception as exc:
                    df = None
                    self._log(f"{label}: download failed — {exc}", "WARN")

                if df is not None and not df.empty:
                    aligned = _align_premium(df, hist)
                    if not aligned.empty:
                        out[(strike, side)] = aligned
                        src = _premium_source(aligned)
                        self._premium_sources[label] = src
                        self._log(
                            f"{label}: {len(aligned)} premium candles aligned to the "
                            f"NIFTY {interval} grid (source={src})",
                            "OK",
                        )
                        continue

                # No free public premium history for the leg → build a series on
                # the SAME grid, anchored to the real Groww chain quote.
                leg    = self._level_quotes.get((strike, side))
                series = _model_premium_series(strike, side, hist, chain_leg=leg)
                out[(strike, side)] = series
                src = _premium_source(series)
                self._premium_sources[label] = src
                if src == "groww-anchored":
                    self._log(
                        f"{label}: {len(series)} candles on the NIFTY {interval} "
                        f"grid anchored to live Groww quote — ₹{leg.get('ltp'):.1f} "
                        f"· IV {leg.get('iv', 0):.2f} · delta {leg.get('delta', 0):+.2f} "
                        f"· OI {int(leg.get('oi') or 0):,}",
                        "OK",
                    )
                else:
                    self._log(
                        f"{label}: no live quote/history — built {len(series)}-candle "
                        f"model series on the NIFTY {interval} grid",
                        "WARN",
                    )
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
                       premium_data, closed_spot, minutes_left) -> tuple[str, str, bool]:
        """
        Returns (direction, reason, from_llm).
        from_llm=False means the direction did NOT come from the model — it was
        derived from price momentum because the LLM was unavailable/unparseable.

        Uses FAST_MODEL_TIMEOUT (2 min) hard cutoff. On timeout or 500/502/503
        the connection is cut immediately and the rule-based fallback is used.
        """
        import time as _time_mod
        from config import FAST_MODEL_TIMEOUT

        if self._llm_fail_streak >= 3:
            return (self._fallback_direction(momentum),
                    "LLM bypassed (failing) — momentum-based", False)

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
        _start = _time_mod.monotonic()
        try:
            for ev in self.agent.run(prompt, temp,
                                     system_suffix="Reply only with the JSON vote."):
                if isinstance(ev, FinalAnswerEvent):
                    data = extract_json(ev.text) or {}
                    d = str(data.get("direction", "")).upper()
                    if d in ("BULLISH", "BEARISH", "SIDEWAYS"):
                        self._llm_fail_streak = 0
                        return d, str(data.get("reason", ""))[:60], True
                    self._llm_fail_streak += 1
                    return (self._fallback_direction(momentum),
                            "unparseable LLM vote — momentum fallback", False)
                elif isinstance(ev, ErrorEvent):
                    self._llm_fail_streak += 1
                    if self._llm_fail_streak == 3:
                        self._log("LLM failing repeatedly — switching to momentum-only votes", "WARN")
                    return (self._fallback_direction(momentum),
                            f"LLM FAILED ({ev.message[:40]}) — momentum fallback", False)
                # Hard timeout cut — do NOT wait for the LLM beyond 2 min
                if _time_mod.monotonic() - _start > FAST_MODEL_TIMEOUT:
                    self._llm_fail_streak += 1
                    self._log("LLM vote timed out (%ss) — momentum fallback", "WARN",
                              FAST_MODEL_TIMEOUT)
                    return (self._fallback_direction(momentum),
                            "LLM timeout — momentum fallback", False)
        except Exception as exc:
            self._llm_fail_streak += 1
            log.warning("LLM vote failed: %s", exc)
            return self._fallback_direction(momentum), f"LLM exception: {exc}", False
        return self._fallback_direction(momentum), "no LLM response", False

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

            # Size the position to the session budget — a NIFTY lot at a real
            # premium (₹100–₹250) costs ₹7.5k–₹19k, so a fixed 1 lot silently
            # blew past a small budget and the order was rejected with no log.
            lot_cost = entry * NIFTY_LOT_SIZE
            avail    = self.session.available_budget
            qty      = int(avail // lot_cost) if lot_cost > 0 else 0
            if qty <= 0:
                self._log(
                    f"   skip {strike}{side} @ ₹{entry:.1f}: 1 lot = "
                    f"₹{lot_cost:,.0f} but only ₹{avail:,.0f} available "
                    f"(raise the /sim budget to trade this level)",
                    "WARN",
                )
                break
            qty = min(qty, 4)                     # cap position size

            trade = Trade(
                id            = f"S{len(self.session.trades)+1:03d}",
                expiry        = self.expiry or "SIM",
                strike        = float(strike),
                option_type   = side,
                action        = "BUY",
                qty           = qty,
                entry_price   = entry,
                current_price = entry,
                sl            = round(entry * 0.75, 2),
                target        = round(entry * 1.5, 2),
                rationale     = f"sim vote {decision} @ spot {spot:.0f}",
            )
            ok, msg = self.session.add_trade(trade)
            if ok:
                self.open_orders[key] = {"trade_id": trade.id, "strike": strike, "side": side}
                self._log(f"   open {trade.id} {qty}L {strike}{side} @ ₹{entry:.1f}", "TRADE")
            else:
                self._log(f"   order rejected {strike}{side}: {msg}", "WARN")

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

    # ── Dashboard (Rich Live) ─────────────────────────────────────────────

    def render_dashboard(self):
        """
        Continuously-updating simulation view: replay state, the order book
        (OPEN + CLOSED), per-cycle votes and the backend activity log.
        """
        from rich import box as _box
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        import ui

        s    = self.session
        pnl  = s.total_pnl
        pc   = "green" if pnl >= 0 else "red"
        last = self.records[-1] if self.records else None

        header = Table.grid(expand=True, padding=(0, 2))
        for _ in range(4):
            header.add_column(ratio=2)
        header.add_row(
            Text("🧪 SIMULATION", style="bold color(208)"),
            Text(f"Phase: {self._phase}", style="bold white"),
            Text(f"Cycle: {len(self.records)}/{self._max_cycles}", style="cyan"),
            Text(f"Data: {self._data_source or '—'}", style="dim"),
        )
        header.add_row(
            Text(f"Clock: {last.clock:%d-%b %H:%M}" if last else "Clock: —", style="dim"),
            Text(f"Spot: {last.spot:,.0f}" if last else "Spot: —", style="dim"),
            Text(
                (f"Votes: Kronos={last.kronos_dir} LLM={last.llm_dir}"
                 + ("" if last.llm_ok else " (fallback)")) if last else "Votes: —",
                style="dim" if (last and last.llm_ok) else "yellow",
            ),
            Text(f"P&L: {'+' if pnl >= 0 else ''}₹{pnl:,.0f}", style=f"bold {pc}"),
        )

        order_book = ui.render_order_book(
            s,
            quote_note=(f"{len(self._premium_sources)} level series · "
                        f"interval {self._interval} · expiry {self.expiry or '—'}"),
        )

        # Live reference strip: where NIFTY is NOW and what the tracked levels
        # cost NOW — alongside the replayed book so the two can be compared.
        live_levels = [
            self._level_quotes.get((st, sd),
                                   {"strike": st, "side": sd, "ltp": 0.0})
            for st, sd in getattr(self, "_resolved_levels", [])
        ]
        market_strip = ui.render_market_strip(
            self._chain_snap or {"spot": self._chain_spot},
            live_levels,
            title="NIFTY now + tracked levels (live, not replayed)",
        )

        votes = Table(
            title="[bold cyan]🗳 Cycle Votes (last 10)[/bold cyan]",
            box=_box.SIMPLE, header_style="bold dim", expand=True,
        )
        for col, w in [("Cycle", 6), ("Clock", 14), ("Spot", 9), ("Kronos", 9),
                       ("LLM", 12), ("Decision", 10), ("Moved", 9), ("OK", 4)]:
            votes.add_column(col, width=w)
        for r in self.records[-10:]:
            lstyle = "dim" if r.llm_ok else "yellow"
            dstyle = {"BULLISH": "green", "BEARISH": "red"}.get(r.decision, "yellow")
            votes.add_row(
                str(r.cycle),
                f"{r.clock:%d-%b %H:%M}",
                f"{r.spot:,.0f}",
                r.kronos_dir,
                Text(r.llm_dir + ("" if r.llm_ok else " *"), style=lstyle),
                Text(r.decision, style=dstyle),
                f"{r.realized_pts:+.0f}" if r.realized_pts else "—",
                "✓" if r.correct else ("✗" if r.realized_pts else ""),
            )
        if not self.records:
            votes.add_row("—", "—", "—", "—", "[dim]waiting for first cycle[/dim]", "", "", "")

        log_panel = ui.render_log_panel(
            self.get_logs()[-14:],
            title="Simulation Activity — data + votes + orders",
            border_style="dim",
            subtitle=f"[dim]expiry {self.expiry or '—'} · {self._days}d replay · Ctrl+C to stop[/dim]",
        )

        return Group(
            Panel(
                header,
                border_style="color(208)",
                title="[bold color(208)]🙏 Jai Sadguru — Simulation[/bold color(208)]",
                subtitle=f"[dim]{self._days}d · {self._interval} candles · "
                         f"levels: {', '.join(k for k in self._premium_sources)}"
                         + (f" · started {self.records[0].clock:%d-%b %H:%M}"
                            if self.records else "") + "[/dim]",
            ),
            market_strip,
            order_book,
            votes,
            log_panel,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

SIM_SPEED = 0.0

# Replay geometry: how many candles the model sees before each decision.
# The window slides ONE candle per decision, so a 500-candle download with a
# 200-bar window yields 299 decisions (bars 200→499 and onwards).
SIM_WINDOW_BARS     = 200
SIM_MIN_WINDOW_BARS = 30
SIM_MIN_DECISIONS   = 60     # keep at least this many steps on short downloads


def _to_pydatetime(ts) -> datetime:
    """
    Convert any pandas index value to datetime WITHOUT the noisy
    "Discarding nonzero nanoseconds in conversion" UserWarning.
    """
    stamp = pd.Timestamp(ts)
    try:
        return stamp.to_pydatetime(warn=False)
    except TypeError:      # older pandas without the warn kwarg
        return stamp.to_pydatetime()


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
