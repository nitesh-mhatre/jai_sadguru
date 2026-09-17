#!/usr/bin/env python3
"""
main.py
-------
Jai Sadguru — NIFTY F&O Expert powered by NVIDIA NIM + Kronos.

On startup the tool asks which mode to run:
  1. LIVE        — trade the live market (data fetched fresh every cycle;
                   last candle is the live candle — only its OPEN is final,
                   used for validation, never for prediction)
  2. SIMULATION  — download past NIFTY data + 2–4 strikes and replay the
                   same pipeline candle-by-candle; core learns rules into
                   rule.md between runs

Pipeline (both modes):
  data → preprocessing → Kronos forecast → multi-model LLM layer → voting
  → orders → market layer (order table on screen)

/go command — live autonomous trading:
    /go only buy budget is 5000 rs only
    /go budget is 100000 and loss taking capacity is 10 percent only
    /go only sell budget 50000 max loss 5%
    /go budget 200000 loss 15 percent every 3 minutes

/sim command — simulation replay:
    /sim                  → last 7 days, 5-minute candles, 200-bar sliding window
    /sim 10d              → last 10 days
    /sim 5d 2024-06-14    → 5 days ending on a specific date
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def _check_deps():
    missing = []
    for pkg in ["rich", "prompt_toolkit", "requests", "pandas", "openai"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}\n")
        sys.exit(1)

_check_deps()

from prompt_toolkit import PromptSession as _PTSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner

sys.path.insert(0, str(Path(__file__).parent))

from config import Config, HISTORY_FILE, APP_VERSION
from agent import Agent, Session, FinalAnswerEvent, ToolCallEvent, ToolResultEvent, ErrorEvent
import ui


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        filename=str(Path.home() / ".jai_sadguru" / "debug.log"),
        level=level,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )


_PT_STYLE = PTStyle.from_dict({
    "prompt":  "bold ansiyellow",
    "rprompt": "ansidarkgray",
})
def _build_prompt_html() -> HTML:
    """Rebuild prompt each turn with current version + time."""
    ts = datetime.now().strftime("%H:%M:%S")
    return HTML(
        f"<ansibrightblack>{APP_VERSION}  {ts}</ansibrightblack>  "
        "<ansiyellow><b>you ❯ </b></ansiyellow>"
    )


class _SpinnerPreview:
    """
    Spinner text that updates live with model output.
    Rich Live re-renders in a background thread, so mutating the Text object
    from the on_token callback is enough to show streaming progress.
    """

    def __init__(self, label: str = "Thinking…"):
        from rich.text import Text as _Text
        self._Text = _Text
        self.label  = label
        self.buffer = ""
        self.text   = _Text(f"  {label}", style="js.muted")

    def on_token(self, delta: str) -> None:
        self.buffer = (self.buffer + delta)[-200:]
        tail = " ".join(self.buffer.split())[-100:]
        self.text.plain = f"  {self.label}  {tail}"

    def reset(self, label: str) -> None:
        self.label  = label
        self.buffer = ""
        self.text.plain = f"  {label}"


_SLASH_TO_QUERY: dict[str, str] = {
    "/expiries":   "List all upcoming NIFTY expiry dates",
    "/expiry":     "List all upcoming NIFTY expiry dates",
    "/spot":       "What is the current NIFTY spot price?",
    "/price":      "What is the current NIFTY spot price?",
    "/trade":      "What is the best trade setup right now based on current OI and price data?",
    "/analysis":   "Give me a full OI analysis and trade recommendation for today",
}


# ── Mode picker (shown at startup) ────────────────────────────────────────────

def _choose_mode() -> str:
    """Ask the user which mode to start in. Returns 'live' | 'simulation' | 'chat'."""
    ui.console.print()
    ui.console.print(Panel(
        "[bold]1[/bold]  📡 [bold color(208)]LIVE[/bold color(208)]        — trade the live market now\n"
        "[bold]2[/bold]  🧪 [bold color(208)]SIMULATION[/bold color(208)]  — replay past NIFTY data, learn rules\n"
        "[bold]3[/bold]  💬 [bold color(208)]CHAT[/bold color(208)]        — ask anything about NIFTY F&O",
        title=f"[bold color(208)]🙏 Jai Sadguru — select mode[/bold color(208)]",
        border_style="color(208)",
        padding=(0, 2),
    ))
    try:
        raw = input("  mode ❯ ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "chat"
    if raw in ("1", "live", "l", "go"):
        return "live"
    if raw in ("2", "sim", "simulation", "s"):
        return "simulation"
    return "chat"


# ── Expiry + level selection flow (after mode selection) ───────────────────

def _choose_expiry() -> str:
    """Show available NIFTY expiries (Groww → NSE fallback) and let the user pick."""
    from data.groww_feed import get_expiries
    current, expiries = "", []
    with ui.console.status("[dim]Fetching expiries (Groww)…[/dim]", spinner="dots2"):
        try:
            current, expiries = get_expiries()
        except Exception:
            pass
    expiries = expiries or []
    if not expiries:
        ui.console.print("  [yellow]⚠ Could not fetch expiries — nearest expiry will be used[/yellow]")
        return ""
    ui.console.print()
    ui.console.print("  [bold color(208)]📅 Available expiries[/bold color(208)]")
    for i, e in enumerate(expiries[:8], 1):
        marker = "  [dim](nearest)[/dim]" if e == current else ""
        ui.console.print(f"    [bold]{i}[/bold]  {e}{marker}")
    try:
        raw = input("  select expiry ❯ ").strip()
    except (KeyboardInterrupt, EOFError):
        return current or expiries[0]
    if raw.isdigit() and 1 <= int(raw) <= min(8, len(expiries)):
        return expiries[int(raw) - 1]
    return raw if raw else (current or expiries[0])


def _choose_direction() -> str:
    """Ask the user whether to allow BUY only, SELL only, or both sides."""
    ui.console.print()
    ui.console.print("  [bold color(208)]📊 Trading direction[/bold color(208)]  "
                     "[dim](press Enter for both sides)[/dim]")
    ui.console.print("    [bold]1[/bold]  ⚡ [dim]Both BUY & SELL (default)[/dim]")
    ui.console.print("    [bold]2[/bold]  🟢 [dim]BUY only (long positions)[/dim]")
    ui.console.print("    [bold]3[/bold]  🔴 [dim]SELL only (short positions)[/dim]")
    try:
        raw = input("  direction ❯ ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "BOTH"
    if raw in ("2", "buy", "b"):
        return "BUY"
    if raw in ("3", "sell", "s"):
        return "SELL"
    return "BOTH"


def _choose_levels(spot: float) -> list[dict]:
    """
    Ask the user to select 2–4 strike levels to track.
    Suggests defaults around spot; accepts input like '1 3' or '24100CE,24000PE'.
    Returns [{strike:int, side:'CE'|'PE'}, ...]
    """
    step = 50
    base = int(round(spot / step) * step) if spot else 0
    suggestions = [
        {"strike": base - 100, "side": "PE"}  if base else None,
        {"strike": base - 50,  "side": "PE"}  if base else None,
        {"strike": base + 50,  "side": "CE"}  if base else None,
        {"strike": base + 100, "side": "CE"}  if base else None,
    ]
    suggestions = [s for s in suggestions if s]

    ui.console.print()
    ui.console.print(f"  [bold color(208)]🎯 Select levels to track[/bold color(208)]  "
                     f"[dim](spot ≈ {spot:,.0f} — press Enter for defaults)[/dim]")
    for i, s in enumerate(suggestions, 1):
        ui.console.print(f"    [bold]{i}[/bold]  {s['strike']} {s['side']}")
    ui.console.print("    [dim]…or type strikes like[/dim] [js.accent]24100CE, 24000PE[/js.accent]")
    try:
        raw = input("  select levels ❯ ").strip()
    except (KeyboardInterrupt, EOFError):
        return suggestions[:2]

    # Custom entry: "24100CE, 24000PE"
    if raw and not raw.replace(" ", "").replace(",", "").isdigit():
        levels = []
        import re
        for m in re.finditer(r"(\d{4,6})\s*(CE|PE)", raw.upper()):
            levels.append({"strike": int(m.group(1)), "side": m.group(2)})
        if levels:
            return levels[:4]
        ui.console.print("  [yellow]⚠ Could not parse — using defaults[/yellow]")
        return suggestions[:2]

    # Numeric picks: "1 3" or "1,3"
    picks = [int(p) for p in raw.replace(",", " ").split() if p.isdigit()] if raw else list(range(1, len(suggestions) + 1))
    levels = [suggestions[p - 1] for p in picks if 1 <= p <= len(suggestions)]
    return (levels or suggestions)[:4]


# ── Slash commands ────────────────────────────────────────────────────────────

def _handle_slash(raw: str, agent: Agent, session: Session, config: Config) -> tuple[bool, str | None]:
    parts = raw.strip().split()
    cmd   = parts[0].lower() if parts else ""

    if cmd in ("/exit", "/quit"):
        ui.console.print("\n[js.muted]  Jai Sadguru. Trade smart. 🙏[/js.muted]\n")
        sys.exit(0)

    elif cmd == "/help":
        ui.show_help()
        return True, None

    elif cmd == "/r1":
        _run_r1(session, agent, config)
        return True, None

    elif cmd == "/next":
        _run_next_predictor(session, config)
        return True, None

    elif cmd == "/models":
        _show_models(config)
        return True, None

    elif cmd == "/rules":
        _show_rules(agent)
        return True, None

    elif cmd == "/clear":
        session.clear()
        ui.show_cleared()
        return True, None

    elif cmd == "/model":
        from config import NVIDIA_MODELS
        entry = NVIDIA_MODELS.get(config.nvidia_model, {})
        ui.console.print(
            f"  [js.muted]Current model:[/js.muted] [js.accent]{entry.get('model_id', config.nvidia_model)}[/js.accent]  "
            f"[js.muted]·  Backend:[/js.muted] [js.accent]NVIDIA NIM[/js.accent]\n"
        )
        return True, None

    elif cmd in _SLASH_TO_QUERY:
        return True, _SLASH_TO_QUERY[cmd]

    else:
        ui.console.print(
            f"  [js.warn]Unknown command:[/js.warn] [js.accent]{raw}[/js.accent]  "
            "[js.muted](type[/js.muted] [js.accent]/help[/js.accent] [js.muted]for commands)[/js.muted]"
        )
        return True, None


# ── /go — live trading mode ───────────────────────────────────────────────────

def _run_live_mode(go_text: str, agent: Agent, config: Config,
                   expiry: str = "", levels: list[dict] | None = None) -> None:
    """Parse /go parameters, start LiveTrader, run Rich Live dashboard."""
    from trading.parser import parse_go_command
    from trading.engine import TradingSession
    from trading.live_mode import LiveTrader

    params  = parse_go_command(go_text)
    session = TradingSession(
        budget       = params["budget"],
        max_loss_pct = params["max_loss_pct"],
        direction    = params["direction"],
        go_message   = go_text,
    )

    ui.show_go_startup(params, go_text)

    trader = LiveTrader(session, agent, config, expiry=expiry, levels=levels)
    trader.start(interval_seconds=params["interval_seconds"])

    ui.console.print(
        "\n  [js.muted]Live option trader active. Option data: [/js.muted]"
        "[js.accent]Groww chain[/js.accent]  [js.muted]·  Order book refreshes every second. "
        "Press [/js.muted][js.accent]Ctrl+C[/js.accent]"
        "[js.muted] to stop and return to normal mode.[/js.muted]\n"
    )

    try:
        with Live(
            trader.render_dashboard(),
            console=ui.console,
            refresh_per_second=2,
            screen=False,
            transient=False,
        ) as live:
            while True:
                time.sleep(1)
                live.update(trader.render_dashboard())

    except KeyboardInterrupt:
        pass

    finally:
        trader.stop()
        ui.console.print(
            "\n[js.muted]  Exiting live mode. Returning to normal conversation.[/js.muted]\n"
        )
        ui.show_trade_summary(session)


# ── /sim — simulation mode ────────────────────────────────────────────────────

def _run_sim_mode(sim_text: str, agent: Agent, config: Config,
                  expiry: str = "", levels: list[dict] | None = None) -> None:
    """
    /sim [days] — replay past data through the pipeline.

    Flow: download index + selected PE/CE level candles (Groww, NSE fallback)
    → preprocessing → Kronos → multi-model LLM → voting → orders → results.
    """
    from trading.parser import parse_go_command
    from trading.engine import TradingSession
    from trading.sim_mode import SimTrader

    parts = sim_text.split() if sim_text else []
    # 7 calendar days ≈ 375 five-minute NIFTY candles, which is enough for the
    # 200-bar sliding window plus a long run of decisions. Override with
    # "/sim 10d", "/sim 1m", etc.
    days  = 7
    for p in parts:
        if p.lower().endswith("d") and p[:-1].isdigit():
            days = int(p[:-1])

    params = parse_go_command(sim_text)

    # A single NIFTY lot at a real premium costs ~₹7.5k–₹19k, so the parser's
    # ₹10,000 default can fund no trade at all. Use a realistic simulation
    # budget unless the user stated one explicitly.
    from config import SIM_DEFAULT_BUDGET
    if not params.get("budget_explicit"):
        params["budget"] = SIM_DEFAULT_BUDGET

    session = TradingSession(
        budget       = params["budget"],
        max_loss_pct = params["max_loss_pct"],
        direction    = params["direction"],
        go_message   = f"/sim {sim_text}".strip(),
    )

    ui.console.print()
    ui.console.print(Panel(
        f"[bold color(208)]🧪 SIMULATION MODE[/bold color(208)]\n"
        f"[js.muted]Replaying[/js.muted] [js.accent]{days}d[/js.accent] "
        f"[js.muted]of NIFTY data through the pipeline…[/js.muted]\n"
        f"[dim]Budget ₹{params['budget']:,.0f} · Max loss {params['max_loss_pct']}%"
        + (f" · Expiry {expiry}" if expiry else "") + "[/dim]",
        border_style="color(208)",
        title="[bold color(208)]🙏 Jai Sadguru — Simulation[/bold color(208)]",
    ))

    from trading.kronos_forecast import kronos_available
    if kronos_available():
        ui.console.print("  [green]✓[/green] [dim]Kronos loaded — candle forecasts active[/dim]")
    else:
        ui.console.print(
            "  [yellow]⚠ Kronos unavailable — LLM-only votes "
            "(pip install torch + clone github.com/shiyu-coder/Kronos to enable)[/yellow]"
        )

    # Logs go to the dashboard panel (log_fn=None) so the Rich Live view stays clean
    sim = SimTrader(session, agent, config, log_fn=None,
                    expiry=expiry, levels=levels)
    ui.console.print(
        "\n  [js.muted]Simulation running — live order book, votes and data log below. "
        "Press [/js.muted][js.accent]Ctrl+C[/js.accent]"
        "[js.muted] to stop early.[/js.muted]\n"
    )

    outcome: dict = {}

    def _sim_worker() -> None:
        try:
            outcome["result"] = sim.run(days=days)
        except Exception as exc:            # surfaced after the dashboard closes
            outcome["error"] = exc

    worker = threading.Thread(target=_sim_worker, daemon=True, name="sim-run")
    worker.start()

    try:
        with Live(
            sim.render_dashboard(),
            console=ui.console,
            refresh_per_second=2,
            screen=False,
            transient=False,
        ) as live:
            while worker.is_alive():
                live.update(sim.render_dashboard())
                time.sleep(0.5)
            live.update(sim.render_dashboard())
    except KeyboardInterrupt:
        sim.stop()
        worker.join(timeout=10)
        ui.console.print("\n[js.muted]  Simulation stopped early.[/js.muted]\n")

    if "error" in outcome:
        ui.console.print(f"\n  [red]✗ Simulation failed: {outcome['error']}[/red]\n")
        return

    result = outcome.get("result")
    if result:
        _show_sim_summary(result, session)


def _sim_log(msg: str, level: str = "INFO") -> None:
    """Simulation log bridge → shared, colourised backend log renderer."""
    ui.log_line(msg, level)


def _show_sim_summary(result, session) -> None:
    from rich.table import Table
    t = Table(title="Simulation Summary", box=box_simple(), header_style="bold color(208)")
    t.add_column("Metric", style="bold")
    t.add_column("Value")
    t.add_row("Window", f"{result.days}d · {result.interval} candles")
    t.add_row("Levels tracked", ", ".join(str(s) for s in result.levels) or "auto")
    t.add_row("Data source", result.data_source)
    t.add_row("Cycles", str(result.cycles))
    t.add_row("LLM direction accuracy", f"{result.llm_correct}/{result.llm_total}")
    t.add_row("Kronos agreement", f"{result.kronos_agree}/{result.kronos_total}")
    t.add_row("Trades", str(result.trades))
    t.add_row("Win rate", f"{result.win_rate}%")
    t.add_row("P&L", f"₹{result.total_pnl:+,.0f}")
    t.add_row("Regime", result.regime)
    ui.console.print()
    ui.console.print(t)
    if result.lessons:
        ui.console.print("\n  [bold]Lessons learned (also written to rule.md):[/bold]")
        for l in result.lessons:
            ui.console.print(f"  • {l}")
    ui.show_trade_summary(session)


def box_simple():
    from rich import box as _box
    return _box.SIMPLE


# ── /rules — view learned rules ───────────────────────────────────────────────

def _show_rules(agent) -> None:
    from trading.rules import load_rules_block, consolidate_rules
    block = load_rules_block()
    if not block:
        ui.console.print(
            "\n  [js.muted]No learned rules yet. Run a simulation "
            "(/sim 3d) — the core writes lessons to rule.md after each session.[/js.muted]\n"
        )
        return
    ui.console.print()
    ui.console.print(Panel(block, title="[bold color(208)]📖 rule.md[/bold color(208)]",
                           border_style="color(208)"))
    ui.console.print()
    with ui.console.status("[dim]Consolidating…[/dim]", spinner="dots2"):
        status = consolidate_rules(agent)
    ui.console.print(f"  [dim]{status}[/dim]\n")


def _show_models(config) -> None:
    """Print all available NVIDIA NIM models with short names."""
    from config import NVIDIA_MODELS, NVIDIA_DEFAULT_MODEL, NVIDIA_KEYS
    from rich.table import Table

    t = Table(title="Available NVIDIA NIM Models", box=None,
              header_style="bold color(208)", show_lines=False)
    t.add_column("Short Name",   style="bold cyan",  no_wrap=True)
    t.add_column("Model ID",     style="js.accent",  no_wrap=True)
    t.add_column("Tokens",       style="dim",         no_wrap=True)
    t.add_column("Key",          style="dim",         no_wrap=True)
    t.add_column("Description",  style="js.muted")

    active = config.nvidia_model
    for short, entry in NVIDIA_MODELS.items():
        key_num = next((k for k, v in NVIDIA_KEYS.items() if v == entry["api_key"]), "?")
        marker = " ◀ active" if short == active else ""
        t.add_row(
            short + marker,
            entry["model_id"],
            str(entry["max_tokens"]),
            key_num,
            entry["description"],
        )

    ui.console.print()
    ui.console.print(t)
    ui.console.print(
        "\n  [js.muted]Switch model:[/js.muted] "
        "[js.accent]python main.py --nvidia kimi[/js.accent]  "
        "[js.muted]or[/js.muted]  [js.accent]python main.py --nvidia nemo-light[/js.accent]\n"
    )


# ── /next — calculation + AI day forecast ─────────────────────────────────────

def _run_next_predictor(session, config) -> None:
    """
    /next command:
      Step 1 — Fetch all market data in parallel (option chain, VIX, tech, global, FII)
      Step 2 — /next pure-calculation prediction (< 15s, no AI)
      Step 3 — AI day forecast: 30-min candle table + next move
      NEW   — Kronos quantitative candle forecast runs alongside the AI one
    """
    from trading.market_time import get_market_status
    from trading.next_predictor import predict_next
    from trading.day_forecast import fetch_forecast_data, generate_day_forecast
    from trading.kronos_forecast import forecast_nifty_live

    ms = get_market_status()

    ui.console.print(
        f"\n  [bold color(208)]⚡ /next[/bold color(208)]  "
        f"[dim]Fetching data + running calc predictor + Kronos + AI forecast…[/dim]"
    )

    if not ms.is_open:
        ui.console.print(
            f"  [yellow]⏰ Market {ms.phase} — showing analysis on last session data. "
            f"Next open: {ms.next_trading_day.strftime('%d-%b %a')}[/yellow]"
        )

    t0 = datetime.now()

    ui.console.print("  [dim]📡 Step 1/3: Fetching market data + calculation engine…[/dim]")
    with ui.console.status("[dim]Fetching…[/dim]", spinner="dots2"):
        try:
            market_data = fetch_forecast_data()
            oc_data = market_data.get("oc", {})
            if "error" in oc_data:
                ui.console.print(f"\n  [red]✗ Data fetch failed: {oc_data['error']}[/red]\n")
                return
            calc_result = predict_next(budget=50_000)
        except Exception as exc:
            ui.console.print(f"\n  [red]✗ /next failed: {exc}[/red]\n")
            return

    calc_elapsed = (datetime.now() - t0).total_seconds()
    ui.console.print(
        f"  [green]✓[/green] [dim]Calculation done in {calc_elapsed:.1f}s  "
        f"score={calc_result.total_score:.1f}/100  "
        f"regime={calc_result.regime}  conf={calc_result.confidence}[/dim]"
    )

    # ── Step 2: Kronos quantitative candle forecast ───────────────────────────
    ui.console.print("  [dim]📈 Step 2/3: Kronos K-line model forecasting…[/dim]")
    kronos_fc = None
    with ui.console.status("[dim]Kronos thinking…[/dim]", spinner="dots2"):
        kronos_fc = forecast_nifty_live(pred_len=6, interval="5m")
    if kronos_fc:
        ui.console.print(f"  [green]✓[/green] [dim]{kronos_fc.summary_line()}[/dim]")
    else:
        ui.console.print("  [yellow]⚠[/yellow] [dim]Kronos unavailable — skipping model forecast[/dim]")

    # ── Step 3: AI forecast (pre-fetched data + Kronos block) ─────────────────
    ui.console.print("  [dim]🧠 Step 3/3: AI generating 30-min candle forecast…[/dim]")
    ai_forecast = None
    with ui.console.status("[dim]AI thinking…[/dim]", spinner="dots2"):
        try:
            ai_forecast = generate_day_forecast(
                agent            = Agent(config),
                market_data      = market_data,
                prediction_score = calc_result.total_score,
                prediction_regime = calc_result.regime,
                kronos_forecast  = kronos_fc,
            )
        except Exception as exc:
            ui.console.print(f"  [yellow]⚠ AI forecast failed: {exc} — showing calc only[/yellow]")

    total_elapsed = (datetime.now() - t0).total_seconds()

    ui.show_next_prediction(calc_result, calc_elapsed)

    if kronos_fc:
        ui.show_kronos_forecast(kronos_fc)

    if ai_forecast:
        ui.show_day_forecast(ai_forecast, total_elapsed)


# ── /r1 — behavioural scanner ─────────────────────────────────────────────────

def _run_r1(session, agent, config) -> None:
    """
    /r1 command:
      1. Build ScanPacket (multi-TF candles + OI + VIX + FII + news)
      2. AI reads scan with R1 behavioural prompts — NO tool calls
      3. Output: scenario A/B + pattern detection + trade with entry/SL/target
    """
    from trading.r1_scanner import build_scan_packet
    from trading.r1_agent import run_r1

    ui.console.print(
        f"\n  [bold color(208)]🧠 R1[/bold color(208)]  "
        f"[dim]Behavioural Pattern Scanner — reading 3m / 15m / 30m / daily candles…[/dim]"
    )

    t0 = datetime.now()

    with ui.console.status("[dim]📡 Fetching multi-timeframe candles + OI + FII + news…[/dim]",
                            spinner="dots2"):
        try:
            scan = build_scan_packet()
        except Exception as exc:
            ui.console.print(f"\n  [red]✗ R1 scan failed: {exc}[/red]\n")
            return

    scan_elapsed = (datetime.now() - t0).total_seconds()
    ui.console.print(
        f"  [green]✓[/green] [dim]Scan built in {scan_elapsed:.1f}s  "
        f"phase={scan.phase}  patterns={len(scan.patterns)}  "
        f"spot={scan.spot:.2f}  pcr={scan.pcr}[/dim]"
    )

    ui.console.print("  [dim]🧠 R1 AI reading behavioural patterns…[/dim]")
    with ui.console.status("[dim]AI applying psychology logic…[/dim]", spinner="dots2"):
        try:
            analysis = run_r1(Agent(config), scan)
        except Exception as exc:
            ui.console.print(f"\n  [red]✗ R1 AI failed: {exc}[/red]\n")
            return

    total_elapsed = (datetime.now() - t0).total_seconds()
    ui.show_r1_analysis(scan, analysis, total_elapsed)


# ── REPL ──────────────────────────────────────────────────────────────────────

def run(config: Config, start_mode: str = "chat") -> None:
    agent   = Agent(config)
    session = Session()

    pt = _PTSession(
        history      = FileHistory(str(HISTORY_FILE)),
        auto_suggest = AutoSuggestFromHistory(),
        style        = _PT_STYLE,
    )

    def _sigint(_sig, _frame):
        ui.console.print("\n[js.muted]  Interrupted. Type /exit to quit.[/js.muted]\n")
    signal.signal(signal.SIGINT, _sigint)

    from config import NVIDIA_MODELS
    entry = NVIDIA_MODELS.get(config.nvidia_model, {})
    ui.banner(f"{config.nvidia_model} ({entry.get('model_id', config.nvidia_model)})", "NVIDIA NIM")

    ok, msg = agent.check_health()
    ui.show_health(ok, msg)
    if not ok:
        ui.console.print(
            "  [js.warn]⚠  Starting anyway — backend must be available for queries.[/js.warn]\n"
        )

    # ── Startup mode ────────────────────────────────────────────────────────
    # Flow: pick mode → show Groww expiries → pick levels → download PE/CE
    # data for those levels → run prediction pipeline → show results.
    mode   = start_mode or _choose_mode()
    expiry = ""
    levels: list[dict] = []
    direction = "BOTH"
    if mode in ("live", "simulation"):
        expiry = _choose_expiry()
        from data.groww_feed import get_option_chain_snapshot
        snap = {}
        with ui.console.status("[dim]Fetching option chain snapshot…[/dim]", spinner="dots2"):
            snap = get_option_chain_snapshot(expiry)
        levels = _choose_levels(float(snap.get("spot") or 0))
        direction = _choose_direction()
        # Show the CURRENT NIFTY value + the current premium/OI/IV of every
        # chosen level before the mode starts, so the user can see what they
        # are about to trade (refresh any time with /levels).
        ui.show_market_now(expiry, levels, title="Your tracked levels — current values")
        if mode == "live":
            _run_live_mode(
                f"{'only buy ' if direction == 'BUY' else 'only sell ' if direction == 'SELL' else ''}budget is 100000 and loss taking capacity is 10 percent only",
                agent, config, expiry=expiry, levels=levels
            )
        else:
            _run_sim_mode(
                f"{'only buy ' if direction == 'BUY' else 'only sell ' if direction == 'SELL' else ''}7d",
                agent, config, expiry=expiry, levels=levels
            )

    while True:
        try:
            send_time = datetime.now()
            raw = pt.prompt(_build_prompt_html()).strip()
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\n[js.muted]  Jai Sadguru. Trade smart. 🙏[/js.muted]\n")
            break

        if not raw:
            continue

        # ── /sim — simulation mode ─────────────────────────────────────────────
        if raw.lower().startswith("/sim"):
            _run_sim_mode(raw[4:].strip(), agent, config)
            continue

        # ── /levels — refresh current NIFTY value + level premiums ────────────
        if raw.lower().startswith("/levels") or raw.lower() in ("/nifty", "/spot"):
            ui.show_market_now(expiry, levels,
                               title="Your tracked levels — current values")
            continue

        # ── /passive — passive pre-planned trading mode ────────────────────────
        if raw.lower().startswith("/passive"):
            passive_text = raw[8:].strip()
            _run_passive_mode(passive_text, agent, config)
            continue

        # ── /go — live trading mode ────────────────────────────────────────────
        if raw.lower().startswith("/go"):
            go_text = raw[3:].strip()
            _run_live_mode(go_text, agent, config)
            continue

        # ── Other slash commands ───────────────────────────────────────────────
        if raw.startswith("/"):
            handled, query = _handle_slash(raw, agent, session, config)
            if handled:
                if query is None:
                    continue
                user_message = query
            else:
                user_message = raw
        else:
            user_message = raw

        # ── Agent loop ─────────────────────────────────────────────────────────
        ui.console.print()
        spinner_ctx = None

        try:
            spinner_ctx = Live(
                Spinner("dots2", text="[js.muted]Thinking…[/js.muted]"),
                transient=True,
                console=ui.console,
            )
            spinner_ctx.start()

            for event in agent.run(user_message, session):

                if isinstance(event, ToolCallEvent):
                    spinner_ctx.stop()
                    spinner_ctx = None
                    ui.show_tool_call(event.name, event.params)

                elif isinstance(event, ToolResultEvent):
                    ui.show_tool_result(event.name, event.result, event.is_error)
                    spinner_ctx = Live(
                        Spinner("dots2", text="[js.muted]Analysing…[/js.muted]"),
                        transient=True,
                        console=ui.console,
                    )
                    spinner_ctx.start()

                elif isinstance(event, FinalAnswerEvent):
                    if spinner_ctx:
                        spinner_ctx.stop()
                        spinner_ctx = None
                    ui.show_answer(event.text, send_time=send_time)

                elif isinstance(event, ErrorEvent):
                    if spinner_ctx:
                        spinner_ctx.stop()
                        spinner_ctx = None
                    ui.show_error(event.message)

        except Exception as exc:
            if spinner_ctx:
                try:
                    spinner_ctx.stop()
                except Exception:
                    pass
            ui.show_error(f"Unexpected error: {exc}")
        finally:
            if spinner_ctx:
                try:
                    spinner_ctx.stop()
                except Exception:
                    pass


# ── /passive — passive pre-planned trading mode ───────────────────────────────

def _run_passive_mode(passive_text: str, agent, config) -> None:
    """
    /passive command flow:
      1. Parse budget / direction / loss% from natural language (same as /go)
      2. Show startup banner
      3. PassivePlanner fetches ALL market data + runs LLM → full trade plan
      4. Display plan for user confirmation
      5. PassiveTrader starts: executes conditions, monitors, async plan refresh
      6. Rich Live dashboard until Ctrl+C
      7. Session summary + dataset saved
    """
    from trading.parser import parse_go_command
    from trading.engine import TradingSession
    from trading.passive_planner import PassivePlanner
    from trading.passive_trader import PassiveTrader
    from trading.dataset import DatasetLogger
    from rich.live import Live

    params  = parse_go_command(passive_text)
    session = TradingSession(
        budget       = params["budget"],
        max_loss_pct = params["max_loss_pct"],
        direction    = params["direction"],
        go_message   = f"/passive {passive_text}",
    )

    ui.console.print()
    ui.console.print(
        Panel(
            f"[bold color(208)]⚡ PASSIVE MODE[/bold color(208)]\n"
            f"[js.muted]Budget:[/js.muted] [js.accent]₹{params['budget']:,.0f}[/js.accent]  "
            f"[js.muted]Direction:[/js.muted] [js.accent]{params['direction']}[/js.accent]  "
            f"[js.muted]Max Loss:[/js.muted] [js.accent]{params['max_loss_pct']}%[/js.accent]\n\n"
            f"[dim]Step 1/3: Fetching complete market data…[/dim]",
            border_style="color(208)",
            title="[bold color(208)]🙏 Jai Sadguru — Passive Trader[/bold color(208)]",
        )
    )

    planner = PassivePlanner(
        agent        = agent,
        budget       = params["budget"],
        direction    = params["direction"],
        max_loss_pct = params["max_loss_pct"],
    )

    logs: list[str] = []

    def _log(msg: str, level: str = "INFO") -> None:
        icon = {"INFO":"·","OK":"✓","WARN":"⚠","ERROR":"✗","PLAN":"📋"}.get(level,"·")
        ts   = datetime.now().strftime("%H:%M:%S")
        logs.append(f"[{ts}] {icon} {msg}")
        ui.console.print(f"  [dim]{ts}[/dim] {icon} {msg}")

    plan = planner.plan(log_fn=_log)

    ui.console.print()
    ui.console.print(plan.summary())
    ui.console.print()
    ui.console.print(
        "  [js.muted]Plan ready. Press [/js.muted][js.accent]Enter[/js.accent]"
        "[js.muted] to start passive trading, or [/js.muted][js.accent]Ctrl+C[/js.accent]"
        "[js.muted] to cancel.[/js.muted]"
    )

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        ui.console.print("\n[js.muted]  Passive mode cancelled.[/js.muted]\n")
        return

    import uuid
    session_id = datetime.now().strftime("%H%M%S") + "_" + uuid.uuid4().hex[:4].upper()
    dataset    = DatasetLogger(session_id=session_id, go_command=f"/passive {passive_text}")

    trader = PassiveTrader(
        plan     = plan,
        planner  = planner,
        session  = session,
        dataset  = dataset,
        interval = params["interval_seconds"],
    )
    trader.start()

    ui.console.print(
        "\n  [js.muted]Passive mode active. Press [/js.muted][js.accent]Ctrl+C[/js.accent]"
        "[js.muted] to stop.[/js.muted]\n"
    )

    try:
        with Live(
            trader.render_dashboard(),
            console=ui.console,
            refresh_per_second=2,
            screen=False,
            transient=False,
        ) as live:
            while True:
                time.sleep(1)
                live.update(trader.render_dashboard())
    except KeyboardInterrupt:
        pass
    finally:
        trader.stop()
        ui.console.print("\n[js.muted]  Exiting passive mode.[/js.muted]\n")
        ui.show_trade_summary(session)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jai Sadguru — NIFTY F&O Expert (NVIDIA NIM + Kronos)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  interactive startup asks: LIVE / SIMULATION / CHAT

/go examples (live trading):
  /go only buy budget is 5000 rs only
  /go budget is 100000 and loss taking capacity is 10 percent only
  /go only sell budget 50000 loss 5 percent every 3 minutes

/sim examples (simulation replay):
  /sim           → last 7 days, sliding 200-bar window, learns into rule.md
  /sim 15d       → last 15 days (longest runs, most decisions)

NVIDIA NIM models (--nvidia <short> or /models in chat):
  glm        ← default (z-ai/glm-5.3)
  kimi       ← moonshotai/kimi-k3 (deep reasoning)
  nemo-light ← fast cycles
  qwen, minimax, nemotron, stepfun, glm52
        """,
    )
    parser.add_argument("--nvidia", "-n", default=None, metavar="SHORT_NAME",
        help="NVIDIA NIM model short name (e.g. glm, kimi, nemo-light). "
             "Run /models in chat to see all options.")
    parser.add_argument("--mode", default=None, choices=["live", "simulation", "chat"],
        help="Skip the startup mode picker")
    parser.add_argument("--debug",  "-d", action="store_true")
    args = parser.parse_args()

    _setup_logging(args.debug)
    config = Config.from_env()

    if args.nvidia:
        from config import NVIDIA_MODELS
        short = args.nvidia.lower()
        if short not in NVIDIA_MODELS:
            valid = ", ".join(NVIDIA_MODELS.keys())
            print(f"\n[ERROR] Unknown NVIDIA model '{short}'.")
            print(f"  Valid short names: {valid}\n")
            sys.exit(1)
        config.nvidia_model = short
    else:
        from config import NVIDIA_DEFAULT_MODEL
        config.nvidia_model = NVIDIA_DEFAULT_MODEL

    if args.debug:
        config.debug = True

    run(config, start_mode=args.mode)


if __name__ == "__main__":
    main()
