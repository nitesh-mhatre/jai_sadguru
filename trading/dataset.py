"""
trading/dataset.py
------------------
Trade Case Study Dataset Logger

Every trade lifecycle event is recorded as a structured JSON case study:
  - Market conditions at entry (spot, PCR, OI walls, VIX, sentiment)
  - Trade parameters (strike, expiry, qty, entry price, SL, target, rationale)
  - Every price update during the trade
  - Exit conditions (SL hit / target hit / manual close)
  - Final P&L with % return

Files saved to: ~/jai_sadguru_trades/
  YYYY-MM-DD_SESSION.jsonl      ← append-only session log (one JSON per line)
  YYYY-MM-DD_T001_summary.json  ← full case study per trade

The JSONL file is the primary dataset for future analysis.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Storage location ───────────────────────────────────────────────────────────
DATASET_DIR = Path.home() / "jai_sadguru_trades"


def _ensure_dir() -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    return DATASET_DIR


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── Case study record ─────────────────────────────────────────────────────────

class TradeCaseStudy:
    """
    Full lifecycle record for one trade.
    Saved to disk at entry and updated at every significant event.
    """

    def __init__(
        self,
        trade_id:        str,
        session_id:      str,
        go_command:      str,
        # Entry market conditions
        market_at_entry: dict,
        # Trade params
        expiry:          str,
        strike:          float,
        option_type:     str,
        action:          str,
        qty:             int,
        entry_price:     float,
        sl:              float,
        target:          float,
        rationale:       str,
        lot_size:        int = 75,
    ):
        self.trade_id    = trade_id
        self.session_id  = session_id
        self.go_command  = go_command
        self.lot_size    = lot_size

        self.entry_time  = _ts()
        self.expiry      = expiry
        self.strike      = strike
        self.option_type = option_type
        self.action      = action
        self.qty         = qty
        self.entry_price = entry_price
        self.sl          = sl
        self.target      = target
        self.rationale   = rationale
        self.total_units = qty * lot_size
        self.capital_deployed = round(entry_price * qty * lot_size, 2)

        # Market context at entry
        self.market_at_entry = market_at_entry

        # Price journey
        self.price_updates: list[dict] = []

        # Exit info
        self.exit_time:   Optional[str]   = None
        self.exit_price:  Optional[float] = None
        self.exit_reason: Optional[str]   = None   # SL_HIT | TARGET_HIT | MANUAL | RECOVERY
        self.final_pnl:   Optional[float] = None
        self.final_pnl_pct: Optional[float] = None

        # AI decision context
        self.ai_cycle_number: int = 0
        self.market_at_exit:  Optional[dict] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def record_price_update(self, cmp: float, pnl: float, market_snapshot: Optional[dict] = None):
        self.price_updates.append({
            "time":     _ts(),
            "cmp":      round(cmp, 2),
            "pnl":      round(pnl, 2),
            "market":   market_snapshot,   # optional compact snapshot
        })

    def record_exit(
        self,
        exit_price:   float,
        exit_reason:  str,
        final_pnl:    float,
        market_at_exit: Optional[dict] = None,
    ):
        self.exit_time      = _ts()
        self.exit_price     = round(exit_price, 2)
        self.exit_reason    = exit_reason
        self.final_pnl      = round(final_pnl, 2)
        self.final_pnl_pct  = round(
            final_pnl / self.capital_deployed * 100, 2
        ) if self.capital_deployed else 0.0
        self.market_at_exit = market_at_exit

    def to_dict(self) -> dict:
        return {
            # Identity
            "trade_id":          self.trade_id,
            "session_id":        self.session_id,
            "go_command":        self.go_command,
            # Trade parameters
            "expiry":            self.expiry,
            "strike":            self.strike,
            "option_type":       self.option_type,
            "action":            self.action,
            "qty_lots":          self.qty,
            "lot_size":          self.lot_size,
            "total_units":       self.total_units,
            "entry_price":       self.entry_price,
            "sl":                self.sl,
            "target":            self.target,
            "capital_deployed":  self.capital_deployed,
            "rationale":         self.rationale,
            "ai_cycle_number":   self.ai_cycle_number,
            # Timestamps
            "entry_time":        self.entry_time,
            "exit_time":         self.exit_time,
            # Market conditions
            "market_at_entry":   self.market_at_entry,
            "market_at_exit":    self.market_at_exit,
            # Journey
            "price_updates":     self.price_updates,
            # Outcome
            "exit_price":        self.exit_price,
            "exit_reason":       self.exit_reason,
            "final_pnl":         self.final_pnl,
            "final_pnl_pct":     self.final_pnl_pct,
            "outcome":           self._outcome(),
            # Metadata
            "recorded_at":       _ts(),
        }

    def _outcome(self) -> str:
        if self.final_pnl is None:
            return "OPEN"
        if self.exit_reason == "TARGET_HIT":
            return "WIN"
        if self.exit_reason == "SL_HIT":
            return "LOSS"
        if self.exit_reason in ("MANUAL", "RECOVERY"):
            return "WIN" if (self.final_pnl or 0) > 0 else "LOSS"
        return "UNKNOWN"

    def save(self, session_jsonl: Path):
        """
        Save / update this case study.
        1. Individual JSON file: ~/jai_sadguru_trades/YYYY-MM-DD_T001_summary.json
        2. Append one line to session JSONL: ~/jai_sadguru_trades/YYYY-MM-DD_SESSION.jsonl
        """
        d = DATASET_DIR

        # Individual summary file (overwrite to reflect latest state)
        summary_path = d / f"{_today()}_{self.trade_id}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        log.debug("Case study saved: %s", summary_path)

        # Append to session JSONL
        with open(session_jsonl, "a") as f:
            f.write(json.dumps(self.to_dict(), default=str) + "\n")


# ── DatasetLogger ─────────────────────────────────────────────────────────────

class DatasetLogger:
    """
    Attached to LiveTrader. Hooks into every trade event.

    Usage in live_mode.py:
        self._dataset = DatasetLogger(session_id, go_command)
        self._dataset.on_trade_placed(trade, market_brief, cycle_num)
        self._dataset.on_price_update(trade, cmp, pnl, brief)
        self._dataset.on_trade_closed(trade, exit_reason, market_brief)
        self._dataset.save_session_summary(session)
    """

    def __init__(self, session_id: str, go_command: str):
        self.session_id   = session_id
        self.go_command   = go_command
        self._studies:    dict[str, TradeCaseStudy] = {}
        self._dir         = _ensure_dir()
        self._jsonl_path  = self._dir / f"{_today()}_{session_id}.jsonl"
        log.info("DatasetLogger: saving to %s", self._dir)

    # ── Event hooks ───────────────────────────────────────────────────────────

    def on_trade_placed(
        self,
        trade,
        market_brief,           # MarketBrief or None
        cycle_num: int = 0,
    ):
        """Called immediately after a trade is successfully added to the session."""
        market_snapshot = self._brief_to_dict(market_brief) if market_brief else {}

        study = TradeCaseStudy(
            trade_id         = trade.id,
            session_id       = self.session_id,
            go_command       = self.go_command,
            market_at_entry  = market_snapshot,
            expiry           = trade.expiry,
            strike           = trade.strike,
            option_type      = trade.option_type,
            action           = trade.action,
            qty              = trade.qty,
            entry_price      = trade.entry_price,
            sl               = trade.sl,
            target           = trade.target,
            rationale        = trade.rationale,
        )
        study.ai_cycle_number = cycle_num
        self._studies[trade.id] = study
        study.save(self._jsonl_path)
        log.info("Dataset: recorded entry %s @ ₹%.2f", trade.id, trade.entry_price)

    def on_price_update(
        self,
        trade,
        cmp: float,
        pnl: float,
        market_brief=None,
    ):
        """Called each time open position CMP is refreshed."""
        study = self._studies.get(trade.id)
        if not study:
            return
        # Only record every 5th update to keep file size manageable
        if len(study.price_updates) % 5 == 0:
            compact = {
                "spot": getattr(market_brief, "spot", None),
                "pcr":  getattr(market_brief, "pcr", None),
            } if market_brief else None
            study.record_price_update(cmp, pnl, compact)
            study.save(self._jsonl_path)

    def on_trade_closed(
        self,
        trade,
        exit_reason: str,
        market_brief=None,
    ):
        """Called when a trade is closed (SL, target, manual, recovery)."""
        study = self._studies.get(trade.id)
        if not study:
            # Trade placed before DatasetLogger started — create stub
            study = TradeCaseStudy(
                trade_id=trade.id, session_id=self.session_id,
                go_command=self.go_command, market_at_entry={},
                expiry=trade.expiry, strike=trade.strike,
                option_type=trade.option_type, action=trade.action,
                qty=trade.qty, entry_price=trade.entry_price,
                sl=trade.sl, target=trade.target, rationale=trade.rationale,
            )
            self._studies[trade.id] = study

        study.record_exit(
            exit_price    = trade.exit_price or trade.current_price,
            exit_reason   = exit_reason,
            final_pnl     = trade.pnl,
            market_at_exit = self._brief_to_dict(market_brief) if market_brief else None,
        )
        study.save(self._jsonl_path)
        log.info(
            "Dataset: recorded exit %s  reason=%s  P&L=₹%.0f (%+.1f%%)",
            trade.id, exit_reason, trade.pnl, study.final_pnl_pct or 0,
        )

    def save_session_summary(self, session):
        """Save full session-level summary at end of /go mode."""
        summary = {
            "session_id":      self.session_id,
            "go_command":      self.go_command,
            "date":            _today(),
            "start_time":      str(session.start_time),
            "end_time":        _ts(),
            "budget":          session.budget,
            "max_loss_pct":    session.max_loss_pct,
            "direction":       session.direction,
            "total_trades":    len(session.trades),
            "open_at_close":   len(session.open_trades),
            "closed_trades":   len(session.closed_trades),
            "win_trades":      session.win_trades,
            "loss_trades":     session.loss_trades,
            "win_rate_pct":    session.win_rate,
            "realized_pnl":    round(session.realized_pnl, 2),
            "unrealized_pnl":  round(session.unrealized_pnl, 2),
            "total_pnl":       round(session.total_pnl, 2),
            "total_pnl_pct":   round(session.total_pnl / session.budget * 100, 2)
                               if session.budget else 0,
            "recovery_mode_hit": session.recovery_mode,
            "decision_cycles": session.decision_count,
            "trade_ids":       [t.id for t in session.trades],
            "dataset_dir":     str(self._dir),
        }

        path = self._dir / f"{_today()}_{self.session_id}_SESSION.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        log.info("Session summary saved: %s", path)
        return path

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _brief_to_dict(brief) -> dict:
        """Convert MarketBrief to a compact dict for storage."""
        if brief is None:
            return {}
        return {
            "fetched_at":  brief.fetched_at,
            "expiry":      brief.expiry,
            "spot":        brief.spot,
            "atm":         brief.atm,
            "pcr":         brief.pcr,
            "sentiment":   brief.sentiment,
            "max_pain":    brief.max_pain,
            "vix":         brief.vix,
            "resistance":  brief.resistance[:2],
            "support":     brief.support[:2],
            "fresh_ce":    brief.fresh_ce_writing[:2],
            "fresh_pe":    brief.fresh_pe_writing[:2],
            "targeted": [
                {
                    "strike":      t.strike,
                    "option_type": t.option_type,
                    "ltp":         t.ltp,
                    "oi":          t.oi,
                    "oi_buildup":  t.oi_buildup,
                    "trend":       t.trend,
                    "iv":          t.iv,
                }
                for t in brief.targeted
            ],
        }

    @property
    def dataset_path(self) -> str:
        return str(self._dir)
