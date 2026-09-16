# 🙏 Jai Sadguru — NIFTY F&O Expert

A calm, sharp NIFTY F&O trading assistant powered by **NVIDIA NIM** models
(z-ai/glm-5.3 default) with a **Kronos** K-line foundation model added to the
decision pipeline. Ollama support has been removed.

On startup the tool asks which mode you want:

| Mode | What it does |
|------|--------------|
| 📡 **LIVE** | Trades the live market. Every cycle fetches fresh data → preprocessing → Kronos forecast → multi-model LLM layer → voting → orders → order table on screen. The last candle is the *live* candle — only its **open** is final, so it is used to validate the previous prediction, never as a prediction input. |
| 🧪 **SIMULATION** | Downloads past NIFTY data + 2–4 strikes (e.g. 24000 PE / 26000 CE side) and replays the **same pipeline** candle-by-candle. Because the core controls the data, it scores every prediction against what actually happened and **learns rules into `rule.md`** to minimise mistakes in the next run. |
| 💬 **CHAT** | Ask anything about NIFTY F&O — option chain, OI analysis, trade ideas. |

## Setup

```bash
bash setup.sh
source venv/bin/activate
```

Set your NVIDIA API key (or edit `NVIDIA_KEYS` in `config.py`):
```bash
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
```

Optional — enable Kronos candle forecasting:
```bash
pip install torch transformers huggingface_hub
git clone https://github.com/shiyu-coder/Kronos
export PYTHONPATH="$PWD/Kronos:$PYTHONPATH"
```
Weights auto-download from HuggingFace (`NeoQuasar/Kronos-small`) on first use.
Without Kronos the app still works — the LLM just votes alone.

## Run

```bash
python main.py                  # interactive mode picker (live / simulation / chat)
python main.py --mode simulation
python main.py --nvidia kimi    # any short name from /models
python main.py --debug
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/spot` | Live NIFTY spot price |
| `/expiries` | List expiry dates |
| `/trade` | Best trade right now |
| `/next` | Calc predictor + Kronos forecast + AI 30-min candle table |
| `/r1` | Behavioural pattern scanner |
| `/models` | List NVIDIA NIM models |
| `/rules` | View/consolidate learned rules (`rule.md`) |
| `/sim 7d` | 🧪 Simulation replay — 7 days of NIFTY candles |
| `/go ...` | 📡 Live autonomous trading |
| `/passive ...` | Pre-planned passive trading |
| `/clear` | Clear chat history |
| `/exit` | Quit |

## 🚀 /go — Live Autonomous Trading

```
/go only buy budget is 5000 rs only
/go budget is 100000 and loss taking capacity is 10 percent only
/go only sell budget 50000 loss 5 percent every 3 minutes
```

Every cycle:
1. Python fetches ALL market data (no AI tokens) — option chain, VIX, tech, news, regime
2. **Kronos** forecasts the next candles (quantitative vote)
3. The LLM reads the compact brief **+ Kronos block** and returns one JSON action block
4. Votes are merged (Kronos = 1 vote, never a trade on its own) → orders execute
5. The order table updates on screen (open/closed trades, P&L, SL/target auto-hits)

Press **Ctrl+C** to exit and see the session summary. Recovery mode triggers
automatically past your max-loss %.

## 🧪 /sim — Simulation Replay

```
/sim          # last 3 trading days, 5-minute candles
/sim 7d       # last 7 days
```

- Replays history candle-by-candle through the identical pipeline
- Tracks 2–4 strikes around the session's opening spot
- Scores each cycle's prediction against the next candle's open (live-candle rule)
- Reports: LLM direction accuracy, Kronos agreement, paper P&L, win rate
- **Writes lessons into `~/.jai_sadguru/rule.md`** — these are injected into
  every future prompt (live AND simulation), so each run starts smarter

## Models (`/models`)

| Short name | Model ID | Notes |
|------------|----------|-------|
| `glm` | z-ai/glm-5.3 | **default** — reasoning + tool-calling |
| `kimi` | moonshotai/kimi-k3 | deep reasoning |
| `nemo-light` | nvidia/nemotron-3.5-lightning-30b-a3b | fast cycles |
| `qwen` | qwen/qwen3-next-80b-a3b-instruct | structured JSON |
| `minimax` | minimaxai/minimax-m2.7 | long-context write-ups |
| `nemotron` | nvidia/nemotron-3-ultra-550b-a55b | structured analysis |
| `stepfun` | stepfun-ai/step-3.5-flash | fast reasoning |
| `glm52` | z-ai/glm-5.2 | alternate reasoning |

## How Trading Works

This bot uses **paper trading** — it tracks positions against real NSE prices
but does NOT connect to a broker. To add real execution, integrate a broker
SDK (Zerodha Kite, Angel One, etc.) in `trading/engine.py`.

## Architecture

```
main.py                 ← CLI + mode picker + REPL
agent.py                ← Agentic loop (NVIDIA NIM only) + JSON extractor
config.py               ← Model registry, keys, prompts, Kronos/sim settings
ui.py                   ← Rich terminal UI + trading panels
trading/
  engine.py             ← Trade / TradingSession (order table)
  parser.py             ← /go natural-language parser
  live_mode.py          ← Live loop: fetch → Kronos → LLM → vote → orders
  sim_mode.py           ← Simulation replay + prediction scoring
  kronos_forecast.py    ← Kronos K-line foundation model integration
  rules.py              ← rule.md experience file (learned rules)
  market_regime.py      ← Regime detection (sideways/directional/volatile)
  next_predictor.py     ← Pure-calc 12-signal predictor
  day_forecast.py       ← AI 30-min candle forecast (+ Kronos input)
  r1_scanner.py / r1_agent.py  ← Behavioural scanner
tools/                  ← Tool specs + NSE implementations
data/                   ← NSE session, option chain, charts, news, Yahoo feed
```
