"""
trading/kronos_forecast.py
--------------------------
Kronos K-line foundation model integration.
https://github.com/shiyu-coder/Kronos   (weights: NeoQuasar/* on HuggingFace)

Kronos is a decoder-only foundation model pre-trained on candlestick (K-line)
sequences from 45+ global exchanges. A specialized tokenizer quantizes OHLCV
into discrete tokens, then an autoregressive Transformer predicts future candles.

Flow used here:
  raw candles → preprocessing (OHLCV DataFrame + timestamps)
              → KronosPredictor.predict()  → future candles DataFrame
              → analysis (direction, projected high/low, targets)
              → prompt block for the LLM voting layer

The model weights (~25MB small / ~100MB base) download from HuggingFace on
first use and cache locally. Everything is lazy-loaded so the CLI starts
instantly and other commands work even without torch installed.
If Kronos is unavailable (no torch / no weights / no network), every entry
point degrades gracefully to None and the pipeline continues with LLM-only
analysis.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config import (
    KRONOS_TOKENIZER_ID,
    KRONOS_MODEL_ID,
    KRONOS_MAX_CONTEXT,
    KRONOS_LOOKBACK,
    KRONOS_T,
    KRONOS_TOP_P,
    KRONOS_SAMPLE_COUNT,
    KRONOS_DEVICE,
)

log = logging.getLogger(__name__)


# ── Lazy singleton model holder ───────────────────────────────────────────────

class _KronosHolder:
    """Loads tokenizer + model once, on first use, in a thread-safe way."""

    def __init__(self):
        self._lock = threading.Lock()
        self._predictor = None
        self._failed    = False   # sticky failure — don't retry every cycle

    def get(self):
        if self._predictor is not None or self._failed:
            return self._predictor
        with self._lock:
            if self._predictor is not None or self._failed:
                return self._predictor
            try:
                from model import Kronos, KronosTokenizer, KronosPredictor  # Kronos repo on sys.path
            except ImportError:
                try:   # fallback: pip-installed package name
                    from kronos.model import Kronos, KronosTokenizer, KronosPredictor
                except ImportError:
                    log.warning("Kronos repo not importable — forecast disabled")
                    self._failed = True
                    return None
            try:
                import torch
                if KRONOS_DEVICE == "auto":
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                else:
                    device = KRONOS_DEVICE
                log.info("Loading Kronos %s …", KRONOS_MODEL_ID)
                tokenizer = KronosTokenizer.from_pretrained(KRONOS_TOKENIZER_ID)
                model     = Kronos.from_pretrained(KRONOS_MODEL_ID)
                self._predictor = KronosPredictor(model, tokenizer, device=device,
                                                  max_context=KRONOS_MAX_CONTEXT)
                log.info("Kronos loaded on %s", device)
            except Exception as exc:
                log.warning("Kronos load failed: %s — forecast disabled", exc)
                self._failed = True
                self._predictor = None
            return self._predictor


_holder = _KronosHolder()


def kronos_available() -> bool:
    """True if the Kronos model is loaded (or loads successfully)."""
    return _holder.get() is not None


# ── Forecast result ───────────────────────────────────────────────────────────

@dataclass
class KronosForecast:
    generated_at:    str
    interval:        str            # "5m", "15m", "30m", "1d"
    last_close:      float
    pred_len:        int
    candles:         pd.DataFrame   # forecast OHLCV indexed by future timestamps
    direction:       str            # BULLISH | BEARISH | SIDEWAYS
    move_points:     float          # projected close - last close
    move_pct:        float
    projected_high:  float
    projected_low:   float
    confidence_note: str = ""
    # Path the model drew from last_close → final close (percentile band)
    path:            list[float] = field(default_factory=list)

    def summary_line(self) -> str:
        arrow = {"BULLISH": "▲", "BEARISH": "▼"}.get(self.direction, "►")
        return (
            f"Kronos {arrow} {self.direction}  "
            f"{self.move_points:+.1f}pts ({self.move_pct:+.2f}%)  "
            f"H≈{self.projected_high:.0f}  L≈{self.projected_low:.0f}  "
            f"over {self.pred_len}×{self.interval}"
        )

    def to_prompt_block(self) -> str:
        """
        Compact block for the LLM voting layer.
        Lists each predicted candle so the AI can reason over levels.
        """
        lines = [
            "═══ KRONOS FORECAST (K-line foundation model) ═══",
            f"model      : {KRONOS_MODEL_ID}  interval={self.interval}  pred_len={self.pred_len}",
            f"last_close : {self.last_close:.2f}   at {self.generated_at}",
            f"direction  : {self.direction}  ({self.move_points:+.1f} pts, {self.move_pct:+.2f}%)",
            f"proj high  : {self.projected_high:.1f}    proj low: {self.projected_low:.1f}",
            "predicted candles:",
        ]
        for ts, row in self.candles.iterrows():
            try:
                t = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)
            except Exception:
                t = str(ts)
            lines.append(
                f"  {t}  O:{row['open']:.1f} H:{row['high']:.1f} "
                f"L:{row['low']:.1f} C:{row['close']:.1f} V:{int(row.get('volume', 0)):,}"
            )
        if self.confidence_note:
            lines.append(f"note: {self.confidence_note}")
        lines.append(
            "RULES: Kronos is ONE vote among several. Cross-check against OI walls, "
            "PCR and VIX before acting. Never trade on Kronos alone."
        )
        lines.append("═══════════════════════════════════════")
        return "\n".join(lines)


# ── Core forecast function ────────────────────────────────────────────────────

def forecast_candles(
    df: pd.DataFrame,
    pred_len: int = 12,
    interval: str = "5m",
    timestamp_col: str = "",
) -> Optional[KronosForecast]:
    """
    Forecast the next `pred_len` candles from historical OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns ['open','high','low','close'] and optional
         'volume' / 'amount'. Case-insensitive column match. Oldest first.
         May carry timestamps in `timestamp_col` or in a DatetimeIndex.
    pred_len : number of future candles to predict
    interval : label for reporting ("5m", "15m", "30m", "1d")

    Returns KronosForecast, or None when Kronos is unavailable.
    """
    predictor = _holder.get()
    if predictor is None or df is None or len(df) < 30:
        return None

    # ── Normalize column names ────────────────────────────────────────────────
    colmap = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("open", "high", "low", "close", "volume", "amount"):
            colmap[c] = cl
    work = df.rename(columns=colmap)
    for req in ("open", "high", "low", "close"):
        if req not in work.columns:
            log.warning("Kronos forecast: missing column %s", req)
            return None
    if "volume" not in work.columns:
        work["volume"] = 0
    if "amount" not in work.columns:
        work["amount"] = 0

    # ── Timestamps: future slot times must exist even without full history ───
    ts = None
    if timestamp_col and timestamp_col in work.columns:
        ts = pd.to_datetime(work[timestamp_col])
    elif isinstance(work.index, pd.DatetimeIndex):
        ts = pd.Series(work.index)

    if ts is None or len(ts) == 0:
        # Synthesize timestamps ending "now" at the right interval
        step = _interval_minutes(interval)
        now  = datetime.now().replace(second=0, microsecond=0)
        ts   = pd.Series([now - timedelta(minutes=step * (len(work) - i))
                          for i in range(len(work))])

    x_df        = work[["open", "high", "low", "close", "volume", "amount"]].tail(KRONOS_LOOKBACK).reset_index(drop=True)
    x_timestamp = ts.tail(len(x_df)).reset_index(drop=True)

    # Future timestamps: continue x_timestamp at the interval step
    step_min   = _interval_minutes(interval)
    last_t     = pd.Timestamp(x_timestamp.iloc[-1])
    y_timestamp = pd.Series([last_t + timedelta(minutes=step_min * (i + 1))
                             for i in range(pred_len)])

    try:
        pred_df = predictor.predict(
            df          = x_df,
            x_timestamp = x_timestamp,
            y_timestamp = y_timestamp,
            pred_len    = pred_len,
            T           = KRONOS_T,
            top_p       = KRONOS_TOP_P,
            sample_count= KRONOS_SAMPLE_COUNT,
        )
    except Exception as exc:
        log.warning("Kronos predict failed: %s", exc)
        return None

    if pred_df is None or len(pred_df) == 0:
        return None

    last_close = float(x_df["close"].iloc[-1])
    final_close = float(pred_df["close"].iloc[-1])
    move        = final_close - last_close
    move_pct    = move / last_close * 100 if last_close else 0.0

    # Direction classification — small moves are noise
    if   move_pct >  0.10: direction = "BULLISH"
    elif move_pct < -0.10: direction = "BEARISH"
    else:                  direction = "SIDEWAYS"

    return KronosForecast(
        generated_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        interval       = interval,
        last_close     = last_close,
        pred_len       = len(pred_df),
        candles        = pred_df,
        direction      = direction,
        move_points    = round(move, 2),
        move_pct       = round(move_pct, 2),
        projected_high = round(float(pred_df["high"].max()), 2),
        projected_low  = round(float(pred_df["low"].min()), 2),
        path           = [round(float(v), 2) for v in pred_df["close"].tolist()],
        confidence_note = (
            f"sample_count={KRONOS_SAMPLE_COUNT} T={KRONOS_T} top_p={KRONOS_TOP_P}; "
            "probabilistic model — levels are estimates"
        ),
    )


def _interval_minutes(interval: str) -> int:
    return {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "1h": 60, "1d": 1440}.get(
        interval, 5
    )


# ── Live NIFTY forecast (fetches its own data) ────────────────────────────────

def forecast_nifty_live(
    pred_len: int = 12,
    interval: str = "5m",
    lookback_days: int = 5,
) -> Optional[KronosForecast]:
    """
    Fetch recent NIFTY candles from Yahoo and forecast the next ones.
    Used by the /next command and by live-mode cycles.
    """
    try:
        import yfinance as yf
        t    = yf.Ticker("^NSEI")
        hist = t.history(period=f"{lookback_days}d", interval=interval, auto_adjust=True)
        if hist is None or hist.empty:
            return None
        return forecast_candles(hist.reset_index(), pred_len=pred_len,
                                interval=interval, timestamp_col="Datetime")
    except Exception as exc:
        log.warning("forecast_nifty_live failed: %s", exc)
        return None


# ── LLM vote aggregation helper ───────────────────────────────────────────────

def kronos_vote(forecast: Optional[KronosForecast]) -> tuple[int, str]:
    """
    Convert a forecast into (vote, reason) for the voting layer.
    vote: +1 bullish, -1 bearish, 0 neutral / unavailable
    """
    if forecast is None:
        return 0, "Kronos unavailable — no model vote"
    if forecast.direction == "BULLISH":
        return 1, f"Kronos projects +{forecast.move_points:.0f}pts over {forecast.pred_len}×{forecast.interval}"
    if forecast.direction == "BEARISH":
        return -1, f"Kronos projects {forecast.move_points:.0f}pts over {forecast.pred_len}×{forecast.interval}"
    return 0, "Kronos projects sideways move"


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=logging.INFO)
    f = forecast_nifty_live(pred_len=8, interval="5m")
    if f:
        print(f.summary_line())
        print(f.to_prompt_block())
    else:
        print("Kronos unavailable — install torch + clone the Kronos repo to enable.")
