"""
trading/passive_planner.py
--------------------------
Passive Planner — LLM + full market data → detailed multi-trade plan

The planner runs BEFORE any trade is placed.
It fetches the complete option chain, builds a rich market context,
then asks the LLM to produce a full structured trade plan covering:

  • Multiple simultaneous trade setups (CE + PE + hedges)
  • For each trade: exact strike, expiry, action, qty, entry range,
    SL level with reason, target level with reason, scenario narrative
  • Scenario matrix: bull / bear / sideways — what happens in each
  • Priority order (which trade to enter first if budget is tight)
  • Update triggers: what price/OI event should prompt a plan refresh

Output: PassivePlan dataclass
  plan.trades      → list[PlannedTrade]   ready to execute
  plan.scenarios   → dict with bull/bear/sideways narrative
  plan.refresh_at  → conditions that trigger async re-plan

This plan is handed to PassiveTrader for execution.
PassiveUpdater then refreshes it asynchronously without stopping trades.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

NIFTY_LOT_SIZE = 75


# ── Planned Trade ──────────────────────────────────────────────────────────────

@dataclass
class PlannedTrade:
    id:           str          # PT001, PT002 …
    expiry:       str
    strike:       float
    option_type:  str          # CE | PE
    action:       str          # BUY | SELL
    qty:          int          # lots
    entry_min:    float        # entry range low
    entry_max:    float        # entry range high
    entry_ideal:  float        # ideal entry price
    sl:           float
    sl_reason:    str          # "below 23900 PE OI wall"
    target1:      float        # first target (partial exit)
    target2:      float        # second target (full exit)
    target_reason: str
    rationale:    str          # full reasoning
    priority:     int          # 1 = enter first
    scenario:     str          # which scenario this fits: BULL | BEAR | SIDEWAYS | ALL
    condition:    str          # "enter if spot > 24100 in first 15 min"
    capital:      float = 0.0  # computed: entry_ideal × qty × lot_size

    def __post_init__(self):
        self.capital = round(self.entry_ideal * self.qty * NIFTY_LOT_SIZE, 2)

    def display(self) -> str:
        return (
            f"PT{self.id:>3}  {self.expiry}  {int(self.strike)}{self.option_type}  "
            f"{self.action}  {self.qty}L  "
            f"Entry ₹{self.entry_min:.0f}–{self.entry_max:.0f} "
            f"(ideal ₹{self.entry_ideal:.0f})  "
            f"SL ₹{self.sl:.0f}  T1 ₹{self.target1:.0f}  T2 ₹{self.target2:.0f}  "
            f"Capital ₹{self.capital:,.0f}  [{self.scenario}]"
        )


# ── Passive Plan ───────────────────────────────────────────────────────────────

@dataclass
class PassivePlan:
    created_at:    str
    expiry:        str
    spot:          float
    atm:           int
    pcr:           float
    vix:           float
    sentiment:     str
    max_pain:      int
    trades:        list[PlannedTrade] = field(default_factory=list)
    scenarios:     dict = field(default_factory=dict)  # bull/bear/sideways narrative
    refresh_triggers: list[str] = field(default_factory=list)
    overall_bias:  str = "NEUTRAL"
    plan_note:     str = ""
    version:       int = 1   # increments on async refresh

    @property
    def total_capital_required(self) -> float:
        return sum(t.capital for t in self.trades)

    def summary(self) -> str:
        lines = [
            f"═══ PASSIVE PLAN v{self.version} ═══ {self.created_at}",
            f"Expiry: {self.expiry}  Spot: {self.spot}  ATM: {self.atm}  "
            f"PCR: {self.pcr}  VIX: {self.vix}  Bias: {self.overall_bias}",
            f"Trades planned: {len(self.trades)}  "
            f"Total capital: ₹{self.total_capital_required:,.0f}",
            "",
        ]
        for t in sorted(self.trades, key=lambda x: x.priority):
            lines.append(f"  [{t.priority}] {t.display()}")
            lines.append(f"       SL reason: {t.sl_reason}")
            lines.append(f"       Target reason: {t.target_reason}")
            lines.append(f"       Condition: {t.condition}")
            lines.append(f"       Rationale: {t.rationale}")
            lines.append("")
        if self.scenarios:
            lines.append("SCENARIOS:")
            for s, desc in self.scenarios.items():
                lines.append(f"  {s.upper()}: {desc}")
            lines.append("")
        if self.refresh_triggers:
            lines.append("REFRESH TRIGGERS:")
            for r in self.refresh_triggers:
                lines.append(f"  • {r}")
        return "\n".join(lines)


# ── LLM Prompt ────────────────────────────────────────────────────────────────

PLANNER_PROMPT_TEMPLATE = """
You are Jai Sadguru — a senior NIFTY F&O trader with 20+ years experience.

MARKET DATA (fresh, fetched right now):
{market_data}

CONSTRAINTS:
  Budget available : ₹{budget:,.0f}
  Direction allowed: {direction}
  Max loss         : {max_loss_pct}%  (₹{max_loss_rs:,.0f})
  NIFTY lot size   : 75 units
  Capital per lot  : entry_price × 75

YOUR TASK — Generate a complete passive trade plan:

1. Analyse the full option chain above (OI walls, PCR, IV skew, fresh writing).
2. Design 3–6 specific trades covering different scenarios (bull, bear, sideways).
3. For each trade provide EXACT numbers — no ranges for SL/target, give the price.
4. Consider: ATM, OTM (1 strike out), OTM (2 strikes out), spreads if budget allows.
5. Order by PRIORITY — which to enter first when market confirms direction.
6. State the CONDITION for each trade — what must happen before entering.
7. Give SCENARIO narratives — what the full portfolio looks like under bull/bear/sideways.
8. Give REFRESH TRIGGERS — what events should cause the plan to be re-analysed.

RESPOND WITH ONLY THIS JSON (no text before or after):
```json
{{
  "overall_bias": "BULLISH|NEUTRAL|BEARISH",
  "plan_note": "one-line market summary",
  "trades": [
    {{
      "id": "PT001",
      "expiry": "DD-Mon-YYYY",
      "strike": 24500,
      "option_type": "CE",
      "action": "BUY",
      "qty": 1,
      "entry_min": 120.0,
      "entry_max": 135.0,
      "entry_ideal": 127.0,
      "sl": 90.0,
      "sl_reason": "below 24400 CE OI wall support; stop if premium loses 30%",
      "target1": 165.0,
      "target2": 195.0,
      "target_reason": "24600 resistance wall; book partial at T1, full at T2",
      "rationale": "PCR 1.31 bullish, fresh PE writing at 24400, ATM CE has rising OI buildup",
      "priority": 1,
      "scenario": "BULL",
      "condition": "Enter if spot holds above 24450 after 09:30 candle close"
    }}
  ],
  "scenarios": {{
    "bull": "If NIFTY sustains above 24500: T001 CE hits T2, T002 PE expires worthless (loss). Net: +₹X",
    "bear": "If NIFTY breaks below 24300: T001 CE SL hit (loss), T003 PE hits T1. Net: +₹Y",
    "sideways": "If NIFTY stays 24300–24550: T001 CE partial exit, T002 short CE gains theta. Net: +₹Z"
  }},
  "refresh_triggers": [
    "Spot moves > 100 pts from current level",
    "PCR changes by > 0.3",
    "Any trade hits SL or target",
    "After 12:00 — reassess for second half"
  ]
}}
```
"""


# ── Market Data Builder ────────────────────────────────────────────────────────

def build_market_data(budget: float) -> tuple[str, dict]:
    """
    Fetch ALL market data needed for planning.
    Returns (formatted_string, raw_dict)
    """
    raw: dict = {}
    lines     = []

    # Spot + VIX
    try:
        from data.yahoo_feed import get_spot_and_vix
        yf = get_spot_and_vix()
        raw["spot"] = yf["spot"]
        raw["vix"]  = yf["vix"]
        lines.append(f"NIFTY spot={yf['spot']}  India VIX={yf['vix']}")
    except Exception as e:
        raw["spot"] = 0.0
        raw["vix"]  = 0.0
        lines.append(f"Spot/VIX unavailable: {e}")

    # Expiries
    try:
        from data.nifty_option_chain import get_expiry_dates
        expiries = get_expiry_dates()
        raw["expiries"] = expiries[:4]
        raw["nearest"]  = expiries[0] if expiries else None
        lines.append(f"Expiries: {expiries[:4]}")
    except Exception as e:
        raw["nearest"]  = None
        raw["expiries"] = []
        lines.append(f"Expiries unavailable: {e}")

    # Full option chain — ATM ±10 for complete picture
    try:
        from data.nifty_option_chain import get_nifty_option_chain, filter_atm
        df, spot_nse = get_nifty_option_chain(expiry=raw.get("nearest"))
        if raw["spot"] <= 0:
            raw["spot"] = spot_nse

        spot    = raw["spot"]
        strikes = sorted(df["Strike"].astype(int).unique())
        atm     = min(strikes, key=lambda x: abs(x - spot))
        raw["atm"]     = atm
        raw["strikes"] = strikes

        # Sanitize
        import pandas as pd
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].replace([float("inf"), float("-inf")], 0).fillna(0)
        df["Strike"]  = df["Strike"].astype(int)

        # PCR
        ce_total = int(df["CE_OI"].sum())
        pe_total = int(df["PE_OI"].sum())
        pcr      = round(pe_total / ce_total, 2) if ce_total else 0
        raw["pcr"] = pcr

        # Max pain
        pain = {}
        for s in strikes:
            loss = (
                ((df["Strike"] - s).clip(lower=0) * df["CE_OI"]).sum() +
                ((s - df["Strike"]).clip(lower=0) * df["PE_OI"]).sum()
            )
            pain[s] = loss
        max_pain   = int(min(pain, key=pain.get))
        raw["max_pain"] = max_pain

        # Sentiment
        if   pcr >= 1.3: sentiment = "BULLISH"
        elif pcr >= 0.9: sentiment = "NEUTRAL-BULLISH"
        elif pcr >= 0.7: sentiment = "NEUTRAL-BEARISH"
        else:            sentiment = "BEARISH"
        raw["sentiment"] = sentiment

        lines.append(
            f"ATM={atm}  PCR={pcr}  sentiment={sentiment}  MaxPain={max_pain}"
        )

        # OI walls (top 5 each side for full picture)
        top_ce = df.nlargest(5, "CE_OI")[["Strike","CE_OI","CE_Chng_OI","CE_IV","CE_LTP"]]
        top_pe = df.nlargest(5, "PE_OI")[["Strike","PE_OI","PE_Chng_OI","PE_IV","PE_LTP"]]

        lines.append("KEY RESISTANCE (CE OI walls):")
        for _, r in top_ce.iterrows():
            lines.append(
                f"  {int(r.Strike)}CE  OI={int(r.CE_OI):,}  "
                f"ΔOI={int(r.CE_Chng_OI):+,}  IV={r.CE_IV:.1f}%  LTP=₹{r.CE_LTP:.1f}"
            )

        lines.append("KEY SUPPORT (PE OI walls):")
        for _, r in top_pe.iterrows():
            lines.append(
                f"  {int(r.Strike)}PE  OI={int(r.PE_OI):,}  "
                f"ΔOI={int(r.PE_Chng_OI):+,}  IV={r.PE_IV:.1f}%  LTP=₹{r.PE_LTP:.1f}"
            )

        # Fresh writing signals
        fc = df[df["CE_Chng_OI"] > 0].nlargest(3, "CE_Chng_OI")
        fp = df[df["PE_Chng_OI"] > 0].nlargest(3, "PE_Chng_OI")
        if not fc.empty:
            lines.append("FRESH CE WRITING (bearish signal): " +
                "  ".join(f"{int(r.Strike)}(+{int(r.CE_Chng_OI):,})"
                           for _, r in fc.iterrows()))
        if not fp.empty:
            lines.append("FRESH PE WRITING (bullish signal): " +
                "  ".join(f"{int(r.Strike)}(+{int(r.PE_Chng_OI):,})"
                           for _, r in fp.iterrows()))

        # IV Skew (ATM ±4)
        atm_idx    = strikes.index(atm)
        skew_range = strikes[max(0, atm_idx-4): atm_idx+5]
        skew_df    = df[df["Strike"].isin(skew_range)]
        lines.append("IV SKEW (ATM ±4 strikes):")
        lines.append("Strike   CE_OI    CE_LTP  CE_IV  PE_LTP  PE_OI    PE_IV")
        for _, row in skew_df.iterrows():
            m = "<<ATM" if row["Strike"] == atm else "     "
            lines.append(
                f"{int(row.Strike)}{m}  "
                f"{int(row.CE_OI):>8,}  {row.CE_LTP:>6.1f}  {row.CE_IV:>5.1f}  "
                f"{row.PE_LTP:>6.1f}  {int(row.PE_OI):>8,}  {row.PE_IV:>5.1f}"
            )

        raw["expiry"] = raw.get("nearest", "N/A")
        raw["df"]     = df

    except Exception as e:
        lines.append(f"Option chain error: {e}")
        raw.setdefault("atm", 0)
        raw.setdefault("pcr", 0)
        raw.setdefault("sentiment", "UNKNOWN")
        raw.setdefault("max_pain", 0)
        raw.setdefault("expiry", "N/A")

    return "\n".join(lines), raw


# ── PassivePlanner ─────────────────────────────────────────────────────────────

class PassivePlanner:
    """
    Generates a PassivePlan using LLM + full market data.
    Call plan() once before trading starts.
    Call refresh() asynchronously to update plan without stopping trades.
    """

    def __init__(self, agent, budget: float, direction: str, max_loss_pct: float):
        self.agent        = agent
        self.budget       = budget
        self.direction    = direction
        self.max_loss_pct = max_loss_pct

    def plan(self, log_fn=None) -> PassivePlan:
        """Full plan from scratch. Blocking ~30 sec."""
        _log = log_fn or (lambda m, l="INFO": print(f"[{l}] {m}"))
        _log("📊 Fetching full market data for passive plan…", "INFO")

        market_str, raw = build_market_data(self.budget)

        _log(
            f"Market: spot={raw.get('spot',0)}  ATM={raw.get('atm',0)}  "
            f"PCR={raw.get('pcr',0)}  sentiment={raw.get('sentiment','?')}",
            "OK",
        )

        prompt = PLANNER_PROMPT_TEMPLATE.format(
            market_data  = market_str,
            budget       = self.budget,
            direction    = self.direction,
            max_loss_pct = self.max_loss_pct,
            max_loss_rs  = self.budget * self.max_loss_pct / 100,
        )

        _log("🧠 LLM building trade plan…", "INFO")
        response = self._call_llm(prompt)
        plan     = self._parse_response(response, raw)

        _log(
            f"✓ Plan v{plan.version}: {len(plan.trades)} trades  "
            f"bias={plan.overall_bias}  "
            f"capital required ₹{plan.total_capital_required:,.0f}",
            "OK",
        )
        return plan

    def refresh(self, existing_plan: PassivePlan,
                open_trade_ids: list[str],
                log_fn=None) -> PassivePlan:
        """
        Async refresh — re-plans with fresh data.
        Preserves open trades (they are NOT cancelled — only future entries updated).
        """
        _log = log_fn or (lambda m, l="INFO": print(f"[{l}] {m}"))
        _log("🔄 Async plan refresh — fetching fresh market data…", "INFO")

        market_str, raw = build_market_data(self.budget)

        open_note = ""
        if open_trade_ids:
            open_note = (
                f"\nNOTE: The following trades are ALREADY OPEN and must NOT be cancelled: "
                f"{', '.join(open_trade_ids)}. "
                f"Only plan NEW entries around them."
            )

        prompt = PLANNER_PROMPT_TEMPLATE.format(
            market_data  = market_str + open_note,
            budget       = self.budget,
            direction    = self.direction,
            max_loss_pct = self.max_loss_pct,
            max_loss_rs  = self.budget * self.max_loss_pct / 100,
        )

        response  = self._call_llm(prompt)
        new_plan  = self._parse_response(response, raw)
        new_plan.version = existing_plan.version + 1

        _log(
            f"✓ Plan refreshed to v{new_plan.version}: "
            f"{len(new_plan.trades)} trades  bias={new_plan.overall_bias}",
            "OK",
        )
        return new_plan

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Single LLM call — returns raw text response."""
        from agent import Session
        from agent import FinalAnswerEvent, ErrorEvent
        temp = Session()
        text = ""
        try:
            for event in self.agent.run(prompt, temp):
                if isinstance(event, FinalAnswerEvent):
                    text = event.text
                    break
        except Exception as exc:
            log.error("Planner LLM call failed: %s", exc)
        return text

    # ── Response parser ────────────────────────────────────────────────────────

    def _parse_response(self, text: str, raw: dict) -> PassivePlan:
        """Extract JSON from LLM response and build PassivePlan."""
        # Extract JSON block
        match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if not match:
            match = re.search(r'(\{"overall_bias".*\})', text, re.DOTALL)

        trades_data = []
        scenarios   = {}
        triggers    = []
        bias        = "NEUTRAL"
        note        = ""

        if match:
            try:
                data       = json.loads(match.group(1))
                trades_data = data.get("trades", [])
                scenarios   = data.get("scenarios", {})
                triggers    = data.get("refresh_triggers", [])
                bias        = data.get("overall_bias", "NEUTRAL")
                note        = data.get("plan_note", "")
            except json.JSONDecodeError as e:
                log.error("Plan JSON parse error: %s", e)
        else:
            log.warning("No JSON block found in planner LLM response")

        # Build PlannedTrade list
        planned = []
        for i, td in enumerate(trades_data, 1):
            try:
                pt = PlannedTrade(
                    id            = str(td.get("id", f"PT{i:03d}")).replace("PT",""),
                    expiry        = str(td.get("expiry", raw.get("expiry", "N/A"))),
                    strike        = float(td.get("strike", raw.get("atm", 0))),
                    option_type   = str(td.get("option_type", "CE")).upper(),
                    action        = str(td.get("action", "BUY")).upper(),
                    qty           = max(1, int(td.get("qty", 1))),
                    entry_min     = float(td.get("entry_min", td.get("entry_ideal", 0))),
                    entry_max     = float(td.get("entry_max", td.get("entry_ideal", 0))),
                    entry_ideal   = float(td.get("entry_ideal", 0)),
                    sl            = float(td.get("sl", 0)),
                    sl_reason     = str(td.get("sl_reason", "")),
                    target1       = float(td.get("target1", 0)),
                    target2       = float(td.get("target2", 0)),
                    target_reason = str(td.get("target_reason", "")),
                    rationale     = str(td.get("rationale", "")),
                    priority      = int(td.get("priority", i)),
                    scenario      = str(td.get("scenario", "ALL")).upper(),
                    condition     = str(td.get("condition", "Enter at market")),
                )
                planned.append(pt)
            except Exception as e:
                log.warning("Could not parse planned trade %d: %s", i, e)

        return PassivePlan(
            created_at        = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            expiry            = raw.get("expiry", "N/A"),
            spot              = raw.get("spot", 0.0),
            atm               = raw.get("atm", 0),
            pcr               = raw.get("pcr", 0.0),
            vix               = raw.get("vix", 0.0),
            sentiment         = raw.get("sentiment", "UNKNOWN"),
            max_pain          = raw.get("max_pain", 0),
            trades            = planned,
            scenarios         = scenarios,
            refresh_triggers  = triggers,
            overall_bias      = bias,
            plan_note         = note,
        )
