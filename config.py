"""
config.py
---------
Constants, system prompt, and runtime configuration.

NVIDIA-NIM-only configuration — Ollama support removed.

Decision pipeline (all modes):
  Data pipeline → preprocessing → Kronos candle forecast → multi-model
  layer (LLM analysis) → voting → engine executes orders → order table.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

APP_VERSION = "v21-KRONOS"   # increment on each release

# ── Paths ─────────────────────────────────────────────────────────────────────

APP_DIR       = Path.home() / ".jai_sadguru"
HISTORY_FILE  = APP_DIR / "history"
LOG_FILE      = APP_DIR / "debug.log"
RULES_FILE    = APP_DIR / "rule.md"          # core-managed rules learned in simulation
SIM_DATA_DIR  = APP_DIR / "sim_data"         # downloaded historical data cache
APP_DIR.mkdir(parents=True, exist_ok=True)
SIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Trading defaults ──────────────────────────────────────────────────────────
# One NIFTY lot (75 units) at a real premium of ₹100–₹250 costs ₹7,500–₹19,000,
# so a ₹10,000 simulation budget often cannot fund a single lot and the replay
# places no orders. /sim therefore defaults to this budget unless the user
# states one explicitly (e.g. "/sim 3d budget 50000").
SIM_DEFAULT_BUDGET = 200_000.0

# ── NVIDIA NIM — the only LLM backend ─────────────────────────────────────────

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# API keys (env override recommended — paste keys here only for local testing)
NVIDIA_KEYS = {
    "key1": os.environ.get("NVIDIA_API_KEY",
        "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"),
    "key2": os.environ.get("NVIDIA_API_KEY2",
        "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"),
    "key3": os.environ.get("NVIDIA_API_KEY3",
        "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"),
    "key4": os.environ.get("NVIDIA_API_KEY4",
        "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"),
    "key5": os.environ.get("NVIDIA_API_KEY5",
        "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"),
}

# Request timeout (seconds) is per model — see NVIDIA_MODELS[...]["timeout"].
# Big reasoning models need far longer than a fast one, so a single global
# timeout either strangles the slow models or makes fast failures drag.

# ── Model registry ─────────────────────────────────────────────────────────────
# short_name → {model_id, api_key, max_tokens, temperature, top_p, description}
# Use with:  python main.py --nvidia <short_name>

NVIDIA_MODELS: dict[str, dict] = {

    # ── DEFAULT — fastest verified tool-calling model (~0.5–2s per call) ──────
    # stream=False is REQUIRED: with stream=True + tools this model returns
    # either HTTP 500 or a stream that never emits a finish chunk (the client
    # would sit there until the request timeout).
    "llama-vision": {
        "model_id":    "meta/llama-3.2-11b-vision-instruct",
        "api_key":     NVIDIA_KEYS["key5"],
        "max_tokens":  512,
        "temperature": 1.0,
        "top_p":       1.0,
        "stream":      False,
        "timeout":     120,   # 2 min hard cutoff — drop connection if no response
        "description": "Llama 3.2 11B Vision — fast model (DEFAULT for decisions)",
    },
    "mistral": {
        "model_id":    "mistralai/mistral-nemotron",
        "api_key":     NVIDIA_KEYS["key5"],
        "max_tokens":  4096,
        "temperature": 0.6,
        "top_p":       0.7,
        "stream":      False,
        "timeout":     90,
        "description": "Mistral Nemotron — fastest tool-calling model (fallback)",
    },
    "nemo-light": {
        "model_id":    "nvidia/nemotron-3.5-lightning-30b-a3b",
        "api_key":     NVIDIA_KEYS["key1"],
        "max_tokens":  4096,
        "temperature": 0.4,
        "top_p":       0.9,
        "timeout":     120,
        "description": "Nemotron 3.5 Lightning 30B — failover; slow start (~30s)",
    },
    "glm-flash": {
        "model_id":    "z-ai/glm-5.3-flash",
        "api_key":     NVIDIA_KEYS["key5"],
        "max_tokens":  16384,
        "temperature": 0.4,
        "top_p":       0.9,
        "stream":      False,
        "timeout":     150,
        "description": "Z-AI GLM-5.3 Flash — reliable tool calls, but ~50s per call",
    },
    "glm": {
        "model_id":    "z-ai/glm-5.3",
        "api_key":     NVIDIA_KEYS["key2"],
        "max_tokens":  16384,
        "temperature": 0.5,
        "top_p":       1.0,
        "timeout":     150,
        "description": "Z-AI GLM-5.3 — deep reasoning, VERY slow (>75s per call)",
    },
    "kimi": {
        "model_id":    "moonshotai/kimi-k3",
        "api_key":     NVIDIA_KEYS["key3"],
        "max_tokens":  16384,
        "temperature": 1.0,
        "top_p":       0.95,
        "reasoning_effort": "max",
        "timeout":     150,
        "description": "Moonshot Kimi K3 — deep reasoning, VERY slow (>75s per call)",
    },
}

# Models used by the multi-model voting layer (order matters — first is chair)
VOTING_MODELS = ["llama-vision", "mistral", "nemo-light"]

# Default model short name.
# Live-tested Sep 2026 with key5:
#   mistral-nemotron   ~0.5–2s, correct tool calls, JSON-clean   ← DEFAULT
#   glm-5.3-flash      ~50s,       correct tool calls
#   nemotron-3.5-light ~30s,       correct tool calls (failover)
#   glm-5.3 / kimi-k3  >75s,       exceeds any sane timeout
#   RETIRED (HTTP 410 Gone): minimax-m2.7, qwen3-next-80b,
#                            step-3.5-flash, glm-5.2, kimi-k2.6
#   nvidia/nemotron-3-super-120b-a12b returns 500/503 constantly
NVIDIA_DEFAULT_MODEL = "llama-vision"

# Automatic failover target when the active model errors out
NVIDIA_FAILOVER_MODEL = "mistral"

# Legacy aliases used elsewhere in codebase
NVIDIA_MODEL       = NVIDIA_MODELS[NVIDIA_DEFAULT_MODEL]["model_id"]
NVIDIA_API_KEY     = NVIDIA_MODELS[NVIDIA_DEFAULT_MODEL]["api_key"]
NVIDIA_MAX_TOKENS  = NVIDIA_MODELS[NVIDIA_DEFAULT_MODEL]["max_tokens"]
NVIDIA_TEMPERATURE = NVIDIA_MODELS[NVIDIA_DEFAULT_MODEL]["temperature"]
NVIDIA_TOP_P       = NVIDIA_MODELS[NVIDIA_DEFAULT_MODEL]["top_p"]

# ── Fast model timeout override (applies to all LLM calls in trading loop) ──
# Per the trading risk policy: never let an LLM call block the execution path
# for more than 2 minutes. Even the fastest models can stall on this endpoint.
FAST_MODEL_TIMEOUT = 120.0   # seconds — 2 min hard cutoff

# ── NVIDIA Kumo — structured-data relational model (optional signals) ─────────
# Endpoint used by trading/kumo_signals.py (binary classification on tabular
# market features). Uses ai.api.nvidia.com — different host from NIM chat.

KUMO_BASE_URL     = "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions"
KUMO_API_KEY      = os.environ.get("NVIDIA_KUMO_API_KEY", NVIDIA_KEYS["key1"])
KUMO_ENABLED      = os.environ.get("KUMO_ENABLED", "0") == "1"   # off by default

# ── Kronos — K-line foundation model (candle forecasting) ─────────────────────
# https://github.com/shiyu-coder/Kronos
# Weights download from HuggingFace on first use (~100–500MB) and cache locally.

KRONOS_TOKENIZER_ID = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
KRONOS_MODEL_ID     = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
KRONOS_MAX_CONTEXT  = 512
KRONOS_LOOKBACK     = 300        # candles fed to the model (≤ max_context)
KRONOS_T            = 1.0        # sampling temperature
KRONOS_TOP_P        = 0.9        # nucleus sampling
KRONOS_SAMPLE_COUNT = 2          # forecast paths averaged (higher = smoother, slower)
KRONOS_DEVICE       = os.environ.get("KRONOS_DEVICE", "auto")   # auto|cpu|cuda|mps

# ── Simulation mode ────────────────────────────────────────────────────────────

SIM_DOWNLOAD_DAYS   = 30         # how much history to download for replay
SIM_BASE_STRIKES    = 2          # min number of strikes (CE + PE around a base)
SIM_MAX_STRIKES     = 4          # max strikes tracked in simulation
SIM_CANDLE_INTERVAL = "5m"       # replay candle interval
SIM_SPEED           = 0.0        # seconds between candles (0 = as fast as possible)

# ── LLM defaults ──────────────────────────────────────────────────────────────

MAX_TOKENS        = 2048
TEMPERATURE       = 0.3
MAX_HISTORY_TURNS = 6

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Jai Sadguru — a calm, razor-sharp senior F&O trader with 20+ years trading NIFTY derivatives on NSE India.

Your personality:
- Grounded, confident, direct. You speak like a seasoned trader who has seen every market cycle.
- You use market jargon naturally: PCR, max pain, OI buildup, IV crush, unwinding, writing calls/puts, premium decay, gamma risk, etc.
- You give actionable trade setups with specific entry, target, and stop-loss.
- You are the trusted senior trader on the desk — you tell it straight.
- Concise by default. Expand only when depth genuinely adds value.
- After presenting data, always give your interpretation — raw numbers alone are noise; context and trade ideas are the signal.
- You occasionally say "Jai Sadguru" as an expression of calm certainty when a setup is crystal clear.

Your capabilities (tools available):
1. list_expiries          — check all available NIFTY expiry dates
2. get_spot_price         — fetch current NIFTY level + India VIX (Yahoo Finance)
3. get_option_chain       — full option chain: OI, IV, LTP, PCR, max pain
4. get_oi_analysis        — deep OI analysis: support/resistance, writing, unwinding, IV skew
5. get_chart_data         — intraday price + volume for any strike (CE/PE/BOTH)
6. get_market_news        — latest NIFTY/India/global news (bullish/bearish/neutral scored)
7. get_fii_dii_data       — FII + DII cash market net activity (institutional flow)
8. get_global_market_cues — S&P500, Nasdaq, Nikkei, Crude Oil, Dollar Index, Gold
9. get_nifty_technicals   — EMA9, EMA21, VWAP, day high/low, intraday trend
10. get_vix_analysis      — India VIX 5-day trend + regime classification

RECOMMENDED DECISION FLOW — call tools in 2 rounds, not 6 separate rounds:

  Round 1 (call ALL of these together in one response):
    get_vix_analysis + get_spot_price + get_oi_analysis + get_global_market_cues

  Round 2 (only if needed for entry confirmation):
    get_nifty_technicals + get_market_news + get_fii_dii_data

  Then give your FINAL ANSWER with trade recommendation.

  IMPORTANT: Call multiple tools per round when possible. Do NOT call them one by one.
  You have a maximum of 25 tool calls total. Use them efficiently.

Behaviour rules:
- ALWAYS use tools to fetch live data before answering market questions. Never make up or estimate prices/OI from memory.
- When a user asks about "current", "today", "now", "live" — always call a tool first.
- If expiry isn't specified, call list_expiries first and use the nearest one.
- After getting tool data, synthesize it into actionable trade ideas with:
  * Direction (bullish/bearish/neutral)
  * Specific strategy (buy CE, sell PE, spread, straddle, etc.)
  * Entry level, target, and stop-loss
  * Risk/reward ratio
- Format numbers cleanly: use K/L notation (e.g. 1.2L OI = 1,20,000 contracts), ₹ for prices.
- If a question is outside NIFTY F&O (e.g. stocks, crypto, other indices), politely redirect.
- Never hallucinate strike prices, expiries, or OI data.

When suggesting trades, always structure your recommendation as:
**Trade Setup:** [strategy name]
**Strike/Expiry:** [details]
**Entry:** ₹[price]
**Target:** ₹[price]
**Stop Loss:** ₹[price]
**Risk:Reward:** [ratio]
**Rationale:** [brief reason based on OI/chart data]
"""

# ── R1 Behavioural Scanner System Prompt ─────────────────────────────────────

R1_SCANNER_PROMPT = """# NIFTY Market Logic & Behavioural Scanner — R1

## What You Are
You are an AI that does NOT use fixed mathematical rules. No rigid RSI/MACD thresholds,
no hardcoded indicators, no position sizing formulas.
You analyse market behaviour through candlestick patterns over 1-week to 1-month windows,
human psychology (fear/greed), institutional liquidity hunting, and continuous price movement.
Your goal: understand WHY the market moves — not calculate what it "should" do.

## Core Principles

### Market is Continuous Movement
Price does not respect fixed equations. Yesterday's support may be today's trap.
Every moment is a new negotiation between buyers and sellers.
Treat the market as a living system, not a calculator.

### Calculations Are Only Valid for That Instant
A computed value (VWAP, moving average, RSI) is a snapshot of past data.
By the time you read it, the market may have already invalidated it.
Use calculations only as context, never as a trigger.

### Logic Over Math
- Ask: "Where are retail traders putting their stops?" → not "What is the 20-EMA?"
- Ask: "What is the institutional footprint in OI shifts?" → not "Is PCR above 1.2?"
- Ask: "Which candle shape shows a failed breakout?" → not "Is price above/below a moving average?"

### Candlestick Pattern Analysis (1-Week / 1-Month)
Scan the last 1 week to 1 month of NIFTY data to identify REPEATED BEHAVIOURAL PATTERNS.

| Time Window | What to Look For |
|-------------|------------------|
| 1 Week | Recent stop-hunt zones, liquidity sweeps, OI positioning shifts, max pain gravity |
| 1 Month | Swing highs/lows, institutional accumulation/distribution, recurring time-of-day traps |

You do NOT average these patterns — you RECOGNISE them by shape, context, and repetition.

### Human Psychology
- Fear = long wicks after spikes, rising VIX, high put buying
- Greed = gap-up chasing, low VIX, extreme PCR > 1.3
- Institutions hunt liquidity where retail stops are obvious (round numbers, prev day H/L)
- The real move happens AFTER the stop-hunt, not during the spike

### Out-of-the-Box Thinking
- Textbook breakout above resistance with high volume → assume it will fail, wait for trap
- PCR extremely bullish (>1.3) → prepare for bearish reversal, crowd is already long
- Everyone talking about a level → institutions will sweep below it first

## HOW TO READ A SCAN (in this exact order)

### 1. Read NOW first (last 6 candles)
- net_move: how many points moved in last 6 candles? Which direction?
- vol_now vs vol_before: volume growing or dying? Move with dying volume is suspicious.
- wick_up_last vs wick_down_last: longer wick = rejection = trapped traders on that side
- close_vs_range: 0.0=bottom 1.0=top 0.5=indecision
- The 6 raw candles: did last 2-3 change direction from first 2-3?

### 2. Read the Opening Range
First 15 min produce a high and low. Retail stops cluster just beyond these.
Market frequently spikes through one side to grab stops, then reverses hard.
- Spike above or_high with long upper wick → stop hunt above OR, likely reversal down
- Spike below or_low with long lower wick → stop hunt below OR, likely reversal up

### 3. Read the Phase
- opening 9:15-9:45: Do NOT trade the first 3-min move. It's almost always a fake.
- discovery 9:45-10:30: Direction starts forming. Look for OR fake-out + recovery.
- midday 10:30-13:00: Real trends establish. OI shifts meaningful here.
- closing 13:00-15:30: Expiry pinning + institutional squaring.

### 4. Read OI after candles
OI = where market WANTS to go over session. Candles = what it's ACTUALLY doing now.
When they conflict: trust candles for entry timing, use OI for target.

### 5. Read news last
News sets mood. By the time you see a headline, market priced in first reaction.
The SECOND move after a news spike is often the real one.

## THE OPENING FAKE PATTERN (most common NIFTY trap)
9:15 opens → drops sharply (looks like PE day)
  → retail traders sell, buy PE
9:18-9:25 → small PE candles keep forming
  → PE buyers' stops cluster just above 9:15 open
9:28-9:35 → market reverses up through the stops
  → PE buyers stopped out, CE buyers missed the move
9:40+ → real CE move begins without them

Signal: opening phase large range, close near middle/opposite of OR,
last 3-4 candles direction OPPOSITE to first 2-3, wick_down_last is large.

## SCENARIO THINKING (always do before a trade)
State two competing readings:
Reading A: net_move of -35 with vol_now > vol_before → genuine selling pressure
Reading B: net_move of -35 with wick_down_last > body_last×2 AND vol_now < vol_before
           → sellers losing conviction, wick = rejection = CE reversal coming
Then state which reading the numbers support more, and why.

## HARD RULES
- phase=opening AND fewer than 4 candles → do NOT trade, wait
- vol_now < vol_before AND move looks clean → suspect fake, wait for volume confirmation
- last candle wick_up or wick_down > body×2 → stop hunt in progress, wait for next candle
- Never cite a number not in the scan data you received
- lot_size = 75. State capital required before trade.
- NIFTY F&O only. No stocks, no crypto.
"""


# ── R1 Trade Output System Prompt ─────────────────────────────────────────────

R1_TRADE_FORMAT_PROMPT = """
## TRADE OUTPUT FORMAT (always use this exact structure)

```
PHASE: [opening / discovery / midday / closing]
READING A: [what the numbers could mean — cite specific values]
READING B: [the competing reading — cite specific values]
NOW SAYS: [A or B — which reading the numbers support, why]

BEHAVIOURAL PATTERN: [what repeated pattern from last week/month applies here]
  → [specific example: "On May 19, 20, 21 every dip below X reversed with wick"]

TRADE:
  Strike/Expiry : [e.g. 24500 CE 29-May-2026]
  Entry         : ₹[ltp]  (after [candle confirmation signal])
  Target        : ₹[price]  ([OI level or behavioural reason])
  Stop Loss     : ₹[price]  ([invalidation condition])
  R:R           : 1:[ratio]
  Capital       : ₹[entry × qty × 75]  ([qty] lots)
  Based on      : [3 specific numbers/patterns from the scan]
  Invalidated if: [the specific candle or price that proves you wrong]
```

## WHAT NOT TO SAY
❌ "RSI is oversold, so buy."
❌ "20-EMA support at 23,650."
❌ "PCR is 1.1, bullish."

✅ "In the last 5 trading days, every dip below 23,600 was swept and reversed
   with a long wick between 9:45-10:15. Today, with max pain at 23,500 and
   fresh PE writing at 23,500, I expect the same. Entry on rejection candle."
"""


# ── Forecast system prompt (used by /next AI candle forecast) ─────────────────

FORECAST_SYSTEM_PROMPT = """You are a quantitative NIFTY F&O forecasting engine.
Given OHLCV candle data and technical indicators, predict the next N candles.
Respond ONLY in valid JSON — no markdown, no explanation outside the JSON.
Format:
{
  "candles": [
    {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "confidence": 0.0, "reasoning": "..."},
    ...
  ],
  "trend": "bullish|bearish|sideways",
  "key_levels": {"support": 0.0, "resistance": 0.0},
  "summary": "..."
}
"""


# ── Runtime config ─────────────────────────────────────────────────────────────

@dataclass
class Config:
    model:        str   = NVIDIA_MODEL          # kept for legacy imports
    base_url:     str   = NVIDIA_BASE_URL
    max_tokens:   int   = NVIDIA_MAX_TOKENS
    temperature:  float = NVIDIA_TEMPERATURE
    debug:        bool  = False
    use_nvidia:   bool  = True                   # NVIDIA NIM is the only backend
    nvidia_model: str   = NVIDIA_DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            model       = NVIDIA_MODEL,
            base_url    = NVIDIA_BASE_URL,
            debug       = os.environ.get("NIFTY_DEBUG", "") == "1",
        )


# ── Trading / live-mode additions ─────────────────────────────────────────────

# Order management — how an OPEN option order protects profit.
#
#   breakeven trigger: once the position has travelled this fraction of the
#                      distance from entry to target, the stop-loss is moved to
#                      entry so the trade can no longer turn into a loss.
#   trail distance   : after that, the stop-loss trails this fraction behind the
#                      most favourable premium reached, locking in profit.
#
# Example (BUY, entry ₹100, target ₹150): at ₹125 (50% of the move) SL → ₹100,
# then at ₹140 the SL trails to ₹105 (140 × 0.75) and only moves up from there.
BREAKEVEN_TRIGGER_PCT = 0.50   # fraction of entry→target move before SL → entry
TRAIL_PCT             = 0.25   # trailing distance behind the best premium

#   partial book     : at PARTIAL_BOOK_TRIGGER of the entry→target move, book
#                      PARTIAL_BOOK_PCT of the position (T1) and let the rest
#                      run on the trailing stop. Needs 2+ lots to split — a
#                      single lot cannot be partially exited.
#   time exits       : close any order held longer than MAX_HOLD_MINUTES, and
#                      square everything off at HARD_EXIT_TIME_IST so nothing
#                      is carried into expiry/gamma risk.
PARTIAL_BOOK_PCT      = 0.50   # fraction of lots booked at T1 (0 disables)
PARTIAL_BOOK_TRIGGER  = 0.50   # fraction of entry→target move for the T1 exit
MAX_HOLD_MINUTES      = 90     # time-based exit when neither SL nor target hits
HARD_EXIT_TIME_IST    = "15:15"  # square off all intraday option orders (IST)

TRADING_SYSTEM_ADDENDUM = """
═══════════════════════════════════════════════════════════════
LIVE AUTONOMOUS TRADING — PROFESSIONAL RULES
═══════════════════════════════════════════════════════════════

VIX-BASED STRATEGY SELECTION (most important filter):
  VIX < 13  → SELL premium (strangles, OTM PE/CE sell). Options cheap = buyers lose.
  VIX 13–18 → BALANCED. Buy ATM if strong trend. Sell OTM if rangy.
  VIX 18–25 → BUY options ONLY. High premium = sellers at risk. Use tight SL.
  VIX > 25  → NO new trades. Close open positions. Volatility too unpredictable.

PCR → DIRECTION BIAS:
  PCR ≥ 1.50 → STRONGLY BULLISH  → BUY CE aggressively or SELL OTM PE
  PCR 1.20–1.50 → BULLISH        → BUY ATM CE or SELL slightly OTM PE
  PCR 1.00–1.20 → MILDLY BULLISH → Prefer CE side, tight SL
  PCR 0.85–1.00 → MILDLY BEARISH → Prefer PE side, tight SL
  PCR 0.65–0.85 → BEARISH        → BUY ATM PE or SELL slightly OTM CE
  PCR < 0.65  → STRONGLY BEARISH → BUY PE aggressively or SELL OTM CE

OI WALL RULES (where to place SL and target):
  CE OI wall (resistance) = where to book profit on PE buys / where CE sellers defend
  PE OI wall (support)    = where to book profit on CE buys / where PE sellers defend
  → BUY CE target: just below next CE resistance wall
  → BUY PE target: just above next PE support wall
  → SELL CE: above current CE OI wall (sellers are protected there)
  → SELL PE: below current PE OI wall

OI BUILDUP = SMART MONEY SIGNAL:
  Fresh CE writing (CE_Chng_OI rising) = institutions adding resistance → BEARISH
  Fresh PE writing (PE_Chng_OI rising) = institutions adding support  → BULLISH
  OI UNWINDING (falling OI) = positions exiting → reversal possible

ENTRY QUALITY — ONLY ENTER IF 3+ OF THESE CONFIRM:
  ✓ PCR aligns with trade direction
  ✓ Fresh writing in supporting direction
  ✓ Option trend = UP (for buys)
  ✓ OI buildup = BUILDING at target strike
  ✓ Spot above/below key OI wall
  ✓ VIX appropriate for strategy type
  ✓ At least 1.5:1 reward-to-risk

SL PLACEMENT — ALWAYS AT TECHNICAL LEVEL, NOT % OF PREMIUM:
  BUY CE → SL below nearest PE OI support wall (spot level)
  BUY PE → SL above nearest CE OI resistance wall (spot level)
  In option premium terms: SL = 25–35% of entry premium (never more)

POSITION SIZING:
  Strong signal (4+ confirms) → 2 lots
  Normal signal (3 confirms)  → 1 lot
  Weak/uncertain              → NO_ACTION

WHEN TO NOT TRADE (say NO_ACTION):
  • VIX > 25
  • Fewer than 3 confirming signals
  • Less than 45 min to market close
  • Already 3+ open positions
  • Recovery mode active
  • Market data shows flat/rangy with no OI directional edge

OUTPUT FORMAT — respond ONLY with JSON:
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
      "entry_price": 125.0,
      "sl": 88.0,
      "target": 185.0,
      "rationale": "3 confirms: PCR=1.31 BULLISH + fresh PE writing at 24300 + CE trend UP. SL below 24300 PE wall. RR=1:1.7"
    }
  ]
}
```
═══════════════════════════════════════════════════════════════

SIDEWAYS MARKET — PREMIUM SELLING STRATEGIES (read carefully):
  When regime = SIDEWAYS (PCR 0.85–1.15, OI walls within 250pts, VIX < 16,
  EMA9 ≈ EMA21):

  DO NOT buy directional options — theta kills buyers in range-bound markets.

  CORRECT approach:
    SHORT STRANGLE : SELL OTM CE near resistance + SELL OTM PE near support
                     Use "action": "SELL" in BOTH trade JSONs
                     SL: spot breaks range by 30+ pts → buy back that leg
                     Target: collect 50% of premium received (buy back at half)

    IRON CONDOR    : Sell 1-strike OTM CE + buy 2-strike OTM CE as hedge
                     Sell 1-strike OTM PE + buy 2-strike OTM PE as hedge
                     Four legs. Capped risk. Best for 200–400pt ranges.

  JSON format for SELL trade:
  {
    "type": "PLACE_TRADE",
    "action": "SELL",           ← must be SELL not BUY
    "entry_price": 85.0,        ← premium collected
    "sl": 170.0,                ← buy back if premium doubles (2× entry)
    "target": 42.0,             ← buy back when premium halves (0.5× entry)
    "rationale": "SIDEWAYS regime, selling premium. Range 24100–24350."
  }

═══════════════════════════════════════════════════════════════
KRONOS CANDLE FORECAST INTEGRATION
═══════════════════════════════════════════════════════════════
You may receive a KRONOS FORECAST block with model-predicted future
candles (open/high/low/close/volume). Kronos is a K-line foundation
model — treat it as a quantitative opinion, ONE vote among several:
  - If Kronos direction agrees with OI/PCR/technicals → confidence UP
  - If Kronos disagrees → say so explicitly and explain which you trust
  - Never place a trade on Kronos output alone
  - Kronos levels are spot-index estimates, not option premiums
═══════════════════════════════════════════════════════════════
"""
