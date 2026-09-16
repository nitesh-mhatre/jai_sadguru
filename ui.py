"""
ui.py
-----
Terminal rendering layer using Rich — Jai Sadguru edition.
"""

from __future__ import annotations

import json
from datetime import datetime

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme


# ── Theme ─────────────────────────────────────────────────────────────────────

_THEME = Theme({
    "js.title":   "bold white",
    "js.tool":    "bold cyan",
    "js.param":   "dim cyan",
    "js.ok":      "bold green",
    "js.err":     "bold red",
    "js.warn":    "bold yellow",
    "js.muted":   "dim white",
    "js.accent":  "bold color(208)",
    "js.border":  "color(208)",
    "js.spot":    "bold bright_yellow",
    "js.bull":    "bold green",
    "js.bear":    "bold red",
    "js.neutral": "bold yellow",
    "js.header":  "bold bright_cyan",
})

console = Console(theme=_THEME, highlight=True)


# ── Banner ─────────────────────────────────────────────────────────────────────

def banner(model: str, base_url: str) -> None:
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("  JAI SADGURU  ", "bold white on color(130)"),
                ("  ·  ", "dim white"),
                ("NIFTY F&O Expert", "bold white"),
                ("  ·  ", "dim white"),
                ("Powered by NVIDIA NIM + Kronos", "dim color(208)"),
            ),
            border_style="color(208)",
            padding=(0, 1),
        )
    )
    console.print(
        f"  [js.muted]Model:[/js.muted] [js.accent]{model}[/js.accent]  "
        f"[js.muted]·  Endpoint:[/js.muted] [js.accent]{base_url}[/js.accent]\n"
        "  [js.muted]Ask me anything about NIFTY F&O: option chain, OI analysis, trade ideas.[/js.muted]\n"
        "  [js.muted]Type[/js.muted] [js.accent]/help[/js.accent] [js.muted]for commands  ·[/js.muted] "
        "[js.accent]/exit[/js.accent] [js.muted]to quit[/js.muted]\n"
    )


def show_health(ok: bool, message: str) -> None:
    if ok:
        console.print(f"  [js.ok]✓[/js.ok] [js.muted]{message}[/js.muted]\n")
    else:
        console.print(
            Panel(
                f"[js.err]{escape(message)}[/js.err]\n\n"
                "[js.muted]Fix: set [/js.muted][js.accent]NVIDIA_API_KEY[/js.accent]"
                "[js.muted] env var or update [/js.muted][js.accent]NVIDIA_KEYS[/js.accent]"
                "[js.muted] in [/js.muted][js.accent]config.py[/js.accent]\n"
                "[js.muted]Get a key at:  [/js.muted][js.accent]https://build.nvidia.com[/js.accent]",
                border_style="yellow",
                title="[js.warn]⚠  NVIDIA NIM not available[/js.warn]",
                padding=(0, 2),
            )
        )


# ── Tool call display ─────────────────────────────────────────────────────────

_TOOL_ICONS = {
    "list_expiries":          "📅",
    "get_spot_price":         "📈",
    "get_option_chain":       "📊",
    "get_oi_analysis":        "🔍",
    "get_chart_data":         "📉",
    "get_market_news":        "📰",
    "get_fii_dii_data":       "🏦",
    "get_global_market_cues": "🌐",
    "get_nifty_technicals":   "📐",
    "get_vix_analysis":       "🌡",
}

def show_tool_call(name: str, params: dict) -> None:
    icon  = _TOOL_ICONS.get(name, "🔧")
    ts    = _ts()
    p_str = "  ".join(
        f"[js.param]{k}=[/js.param][js.accent]{escape(str(v))}[/js.accent]"
        for k, v in params.items()
    )
    console.print(
        f"  [dim]{ts}[/dim]  {icon}  [js.tool]{name}[/js.tool]"
        + (f"  {p_str}" if p_str else "")
    )


def show_tool_result(name: str, result: str, is_error: bool) -> None:
    ts = _ts()
    if is_error:
        try:
            msg = json.loads(result).get("error", result)
        except Exception:
            msg = result
        console.print(f"  [dim]{ts}[/dim]  [js.err]✗ {escape(msg[:200])}[/js.err]")
    else:
        try:
            data = json.loads(result)
            _show_data_summary(name, data, ts)
        except Exception:
            console.print(f"  [dim]{ts}[/dim]  [js.ok]✓[/js.ok] [js.muted]{escape(result[:120])}[/js.muted]")


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _show_data_summary(name: str, data: dict, ts: str = "") -> None:
    ts = ts or _ts()
    pre = f"  [dim]{ts}[/dim]  [js.ok]✓[/js.ok]"

    if name == "list_expiries":
        console.print(
            f"{pre} [js.muted]{len(data.get('expiries', []))} expiries  ·  nearest:[/js.muted] "
            f"[js.accent]{data.get('nearest', '')}[/js.accent]"
        )
    elif name == "get_spot_price":
        vix = data.get("india_vix", 0)
        vc  = "green" if vix < 15 else "yellow" if vix < 20 else "red"
        console.print(
            f"{pre} [js.muted]Spot:[/js.muted] "
            f"[js.spot]₹{data.get('spot', 0):,.2f}[/js.spot]  "
            f"[js.muted]VIX:[/js.muted] [{vc}]{vix}[/{vc}]  "
            f"[js.muted]Expiry:[/js.muted] [js.accent]{data.get('nearest_expiry', '')}[/js.accent]"
        )
    elif name == "get_option_chain":
        pcr = data.get("pcr", 0)
        col = "js.bull" if pcr > 1.0 else ("js.bear" if pcr < 0.8 else "js.neutral")
        console.print(
            f"{pre} [js.muted]Spot:[/js.muted] [js.spot]₹{data.get('spot', 0):,.2f}[/js.spot]  "
            f"[js.muted]ATM:[/js.muted] [js.accent]{data.get('atm_strike', 0):,}[/js.accent]  "
            f"[js.muted]PCR:[/js.muted] [{col}]{pcr}[/{col}]  "
            f"[js.muted]MaxPain:[/js.muted] [js.accent]{data.get('max_pain', 0):,}[/js.accent]"
        )
    elif name == "get_oi_analysis":
        sentiment = data.get("sentiment", "")
        col = "js.bull" if "Bullish" in sentiment else ("js.bear" if "Bearish" in sentiment else "js.neutral")
        vix = data.get("india_vix", 0)
        vc  = "green" if vix < 15 else "yellow" if vix < 20 else "red"
        console.print(
            f"{pre} [js.muted]PCR:[/js.muted] [js.accent]{data.get('pcr', 0)}[/js.accent]  "
            f"[{col}]{sentiment[:30]}[/{col}]  "
            f"[js.muted]MaxPain:[/js.muted] [js.accent]{data.get('max_pain', 0):,}[/js.accent]  "
            f"[js.muted]VIX:[/js.muted] [{vc}]{vix}[/{vc}]"
        )
    elif name == "get_chart_data":
        if "CE" in data and "PE" in data:
            ce = data["CE"]
            pe = data["PE"]
            ce_col = "green" if (ce.get("pct_change", 0) or 0) >= 0 else "red"
            pe_col = "green" if (pe.get("pct_change", 0) or 0) >= 0 else "red"
            console.print(
                f"{pre} [js.muted]CE:[/js.muted] ₹{ce.get('ltp', 0)} "
                f"([{ce_col}]{(ce.get('pct_change') or 0):+.1f}%[/{ce_col}])  "
                f"[js.muted]PE:[/js.muted] ₹{pe.get('ltp', 0)} "
                f"([{pe_col}]{(pe.get('pct_change') or 0):+.1f}%[/{pe_col}])"
            )
        else:
            pct = data.get("pct_change", 0) or 0
            col = "green" if pct >= 0 else "red"
            console.print(
                f"{pre} [js.muted]LTP:[/js.muted] ₹{data.get('ltp', 0)}  "
                f"[js.muted]Change:[/js.muted] [{col}]{pct:+.1f}%[/{col}]  "
                f"[js.muted]Volume:[/js.muted] {data.get('total_volume', 0):,}"
            )
    elif name == "get_market_news":
        s = data.get("sentiment_summary", {})
        console.print(
            f"{pre} [js.muted]News:[/js.muted] {data.get('count', 0)} articles  "
            f"[green]🟢 {s.get('bullish', 0)}[/green]  "
            f"[red]🔴 {s.get('bearish', 0)}[/red]  "
            f"[dim]⚪ {s.get('neutral', 0)}[/dim]"
        )
    elif name == "get_fii_dii_data":
        fii = data.get("fii_cash_net", 0)
        dii = data.get("dii_cash_net", 0)
        fc  = "green" if fii > 0 else "red"
        dc  = "green" if dii > 0 else "red"
        console.print(
            f"{pre} [js.muted]FII net:[/js.muted] [{fc}]₹{fii:,.0f}Cr[/{fc}]  "
            f"[js.muted]DII net:[/js.muted] [{dc}]₹{dii:,.0f}Cr[/{dc}]  "
            f"[js.muted]Signal:[/js.muted] [js.accent]{data.get('market_signal', '?')}[/js.accent]"
        )
    elif name == "get_global_market_cues":
        bias = data.get("global_bias", "?")
        bc   = "green" if bias == "BULLISH" else "red" if bias == "BEARISH" else "yellow"
        sp   = data.get("sp500",    {}).get("change_pct", 0)
        nk   = data.get("nikkei",   {}).get("change_pct", 0)
        oil  = data.get("crude_oil",{}).get("change_pct", 0)
        sc   = "green" if sp  > 0 else "red"
        nc   = "green" if nk  > 0 else "red"
        oc   = "red"   if oil > 0 else "green"   # oil up = bad for India
        console.print(
            f"{pre} [{bc}]Global {bias}[/{bc}]  "
            f"[js.muted]S&P:[/js.muted] [{sc}]{sp:+.1f}%[/{sc}]  "
            f"[js.muted]Nikkei:[/js.muted] [{nc}]{nk:+.1f}%[/{nc}]  "
            f"[js.muted]Crude:[/js.muted] [{oc}]{oil:+.1f}%[/{oc}]"
        )
    elif name == "get_nifty_technicals":
        trend = data.get("trend", "?")
        tc    = "green" if "UPTREND" in trend else "red" if "DOWNTREND" in trend else "yellow"
        console.print(
            f"{pre} [{tc}]{trend}[/{tc}]  "
            f"[js.muted]EMA9:[/js.muted] [js.accent]{data.get('ema9', 0)}[/js.accent]  "
            f"[js.muted]EMA21:[/js.muted] [js.accent]{data.get('ema21', 0)}[/js.accent]  "
            f"[js.muted]VWAP:[/js.muted] [js.accent]{data.get('vwap', 0)}[/js.accent]"
        )
    elif name == "get_vix_analysis":
        regime = data.get("regime", "?")
        rc     = {"LOW_FEAR": "green", "NORMAL": "cyan", "ELEVATED": "yellow", "EXTREME_FEAR": "red"}.get(regime, "dim")
        trend  = data.get("trend", "?")
        tc     = "red" if trend == "RISING" else "green"
        console.print(
            f"{pre} [js.muted]VIX:[/js.muted] [{rc}]{data.get('current', 0)} {regime}[/{rc}]  "
            f"[js.muted]Trend:[/js.muted] [{tc}]{trend}[/{tc}]  "
            f"[js.muted]5d avg:[/js.muted] {data.get('avg_5d', 0)}"
        )
    else:
        console.print(f"{pre} [js.muted]{name} data received[/js.muted]")


# ── Answer display ─────────────────────────────────────────────────────────────

def show_answer(text: str, send_time: datetime = None, version: str = "") -> None:
    from config import APP_VERSION
    ver      = version or APP_VERSION
    now      = datetime.now()
    ts_str   = now.strftime("%H:%M:%S")

    if send_time:
        elapsed  = (now - send_time).total_seconds()
        time_str = f"[dim]{ts_str}  ·  ⏱ {elapsed:.1f}s[/dim]"
    else:
        time_str = f"[dim]{ts_str}[/dim]"

    console.print()
    console.print(
        Panel(
            Markdown(text),
            border_style="color(208)",
            padding=(0, 2),
            title=f"[bold color(208)]🙏 Jai Sadguru[/bold color(208)]  [dim color(208)]{ver}[/dim color(208)]",
            subtitle=time_str,
            title_align="left",
            subtitle_align="right",
        )
    )
    console.print()


# ── Error display ─────────────────────────────────────────────────────────────

def show_error(message: str) -> None:
    ts = _ts()
    console.print()
    console.print(
        Panel(
            f"[js.err]{escape(message)}[/js.err]",
            border_style="red",
            title="[js.err]Error[/js.err]",
            subtitle=f"[dim]{ts}[/dim]",
            padding=(0, 2),
        )
    )
    console.print()


# ── Help panel ─────────────────────────────────────────────────────────────────

def show_help() -> None:
    console.print()
    t = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="js.header",
        border_style="color(208)",
        padding=(0, 2),
        title="[bold color(208)]🙏 Jai Sadguru — Help[/bold color(208)]",
    )
    t.add_column("Command / Example",           style="js.accent bold", no_wrap=True)
    t.add_column("What it does",                style="js.muted")

    commands = [
        ("/help",                               "Show this help panel"),
        ("/r1",                                 "🧠 R1 behavioural scanner — psychology + candle patterns"),
        ("/next",                               "⚡ Instant prediction — calc + Kronos + AI forecast"),
        ("/models",                             "List all available NVIDIA NIM models"),
        ("/rules",                              "📖 View/consolidate learned rules (rule.md)"),
        ("/clear",                              "Clear conversation history"),
        ("/expiries",                           "List all upcoming NIFTY expiry dates"),
        ("/spot",                               "Quick NIFTY spot price"),
        ("/model",                              "Show current model"),
        ("/exit  or  /quit",                    "Exit the bot"),
        ("─── Live Trading ───",               ""),
        ("/go only buy budget is 5000 rs",      "Live mode — agent decides trades each cycle"),
        ("/passive budget 50000 loss 10%",      "Passive mode — full pre-planned multi-trade"),
        ("/go budget 100000 loss 10 percent",   "Both sides, ₹1L, 10% max loss"),
        ("/go only sell budget 50000 loss 5%",  "Live mode, SELL only, ₹50k, 5% stop"),
        ("─── Simulation ───",                  ""),
        ("/sim",                                "🧪 Replay last 3 days, learn into rule.md"),
        ("/sim 7d",                             "Replay 7 days of NIFTY candles"),
        ("─── Example Queries ───",             ""),
        ("Where is Nifty right now?",           "Live spot price"),
        ("Show me the option chain",            "Full chain for nearest expiry"),
        ("OI analysis for this week",           "PCR, max pain, support/resistance"),
        ("What's the best trade right now?",    "AI-driven trade recommendation"),
        ("How is the 24500 CE doing?",          "Intraday chart data for a strike"),
        ("What's the max pain today?",          "Max pain calculation"),
        ("Where's the heaviest CE writing?",    "Key resistance from OI"),
        ("Suggest a bullish trade",             "Bullish strategy with entry/target/SL"),
        ("Should I buy or sell options?",       "Directional view with data"),
        ("Compare 24000 CE and PE",             "Both legs chart data"),
    ]

    for cmd, desc in commands:
        if cmd.startswith("───"):
            t.add_row(f"[dim]{cmd}[/dim]", f"[dim]{desc}[/dim]")
        else:
            t.add_row(cmd, desc)

    console.print(t)
    console.print()


# ── Kronos forecast display ──────────────────────────────────────────

def show_kronos_forecast(fc) -> None:
    """Display the Kronos K-line model forecast alongside AI output."""
    dc   = {"BULLISH": "green", "BEARISH": "red"}.get(fc.direction, "yellow")
    arrow = {"BULLISH": "▲", "BEARISH": "▼"}.get(fc.direction, "►")

    body = Table.grid(padding=(0, 2), expand=True)
    body.add_column(ratio=2)
    body.add_column(ratio=2)
    body.add_column(ratio=2)
    body.add_row(
        Text(f"{arrow} {fc.direction}", style=f"bold {dc}"),
        Text(f"Move: {fc.move_points:+.1f}pts ({fc.move_pct:+.2f}%)", style=dc),
        Text(f"H≈ {fc.projected_high:,.0f}   L≈ {fc.projected_low:,.0f}", style="dim"),
    )

    # Mini candle table
    tbl = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", expand=True)
    tbl.add_column("Time", style="dim", width=10)
    tbl.add_column("Open", width=10)
    tbl.add_column("High", width=10)
    tbl.add_column("Low",  width=10)
    tbl.add_column("Close", width=10)
    for ts, row in list(fc.candles.iterrows())[:6]:
        try:
            t = ts.strftime("%H:%M") if hasattr(ts, "strftime") else str(ts)[:5]
        except Exception:
            t = str(ts)[:5]
        c_col = "green" if row["close"] >= row["open"] else "red"
        tbl.add_row(
            t,
            f"{row['open']:,.1f}",
            f"{row['high']:,.1f}",
            f"{row['low']:,.1f}",
            Text(f"{row['close']:,.1f}", style=c_col),
        )

    console.print()
    console.print(
        Panel(
            Group(body, tbl),
            border_style="color(208)",
            title="[bold color(208)]📈 Kronos K-line Forecast[/bold color(208)]",
            subtitle=(
                f"[dim]foundation model  ·  {fc.interval} candles  ·  "
                f"pred_len={fc.pred_len}  ·  ONE vote in the ensemble[/dim]"
            ),
        )
    )
    console.print()


def show_cleared() -> None:
    console.print("[js.muted]  ✓ Conversation cleared.[/js.muted]")
    console.print()


def separator() -> None:
    console.print(Rule(style="dim color(208)"))


# ═══════════════════════════════════════════════════════════════════════════════
# Live trading UI additions
# ═══════════════════════════════════════════════════════════════════════════════

def show_go_startup(params: dict, go_text: str) -> None:
    """Show a startup banner when /go is entered."""
    direction_icon = {"BUY": "🟢", "SELL": "🔴", "BOTH": "⚡"}.get(params["direction"], "⚡")
    interval_min = params["interval_seconds"] // 60
    interval_sec = params["interval_seconds"] % 60

    t = Table.grid(padding=(0, 2))
    t.add_column(style="js.muted",   no_wrap=True)
    t.add_column(style="js.accent",  no_wrap=True)
    t.add_row("Command :",    go_text or "(default settings)")
    t.add_row("Direction :",  f"{direction_icon}  {params['direction']}")
    t.add_row("Budget :",     f"₹{params['budget']:,.0f}")
    t.add_row("Max Loss :",   f"{params['max_loss_pct']}%  (₹{params['budget'] * params['max_loss_pct'] / 100:,.0f})")
    t.add_row("Interval :",   f"{interval_min}m {interval_sec}s between decisions")

    console.print()
    console.print(
        Panel(
            t,
            border_style="color(208)",
            title="[bold color(208)]🚀 Starting Live Trading Mode[/bold color(208)]",
            padding=(0, 2),
        )
    )


def show_trade_summary(session) -> None:
    """Show final session P&L summary after live mode ends."""
    pnl_color = "green" if session.total_pnl >= 0 else "red"
    pnl_sign  = "+" if session.total_pnl >= 0 else ""
    pnl_pct   = session.total_pnl / session.budget * 100 if session.budget else 0
    duration  = str(
        __import__("datetime").datetime.now() - session.start_time
    ).split(".")[0]

    t = Table.grid(padding=(0, 2))
    t.add_column(style="js.muted",  no_wrap=True)
    t.add_column(no_wrap=True)
    t.add_row("Session Duration :",
              f"[js.accent]{duration}[/js.accent]")
    t.add_row("Total Trades :",
              f"[js.accent]{len(session.trades)}[/js.accent]  "
              f"[js.muted]({len(session.open_trades)} open, {len(session.closed_trades)} closed)[/js.muted]")
    t.add_row("Win / Loss :",
              f"[js.ok]{session.win_trades} wins[/js.ok]  [js.muted]/[/js.muted]  "
              f"[js.err]{session.loss_trades} losses[/js.err]  "
              f"[js.muted](Win-rate {session.win_rate}%)[/js.muted]")
    t.add_row("Realised P&L :",
              f"[{'green' if session.realized_pnl >= 0 else 'red'}]"
              f"{'+' if session.realized_pnl >= 0 else ''}₹{session.realized_pnl:,.0f}[/]")
    t.add_row("Unrealised P&L :",
              f"[dim]{'+' if session.unrealized_pnl >= 0 else ''}₹{session.unrealized_pnl:,.0f}[/dim]")
    t.add_row("Total P&L :",
              f"[bold {pnl_color}]{pnl_sign}₹{session.total_pnl:,.0f}  ({pnl_sign}{pnl_pct:.2f}%)[/bold {pnl_color}]")
    if session.recovery_mode:
        t.add_row("Note :", "[js.warn]Entered recovery mode during session[/js.warn]")

    console.print()
    console.print(
        Panel(
            t,
            border_style=pnl_color,
            title=f"[bold {pnl_color}]📋 Session Summary[/bold {pnl_color}]",
            padding=(0, 2),
        )
    )
    console.print()


# ── /next prediction display ──────────────────────────────────────────────────

def show_next_prediction(result, elapsed: float) -> None:
    """Rich display for /next pure-calculation trade prediction."""
    from rich.table import Table

    score = result.total_score
    if   score >= 68: regime_color = "bold green"
    elif score >= 58: regime_color = "green"
    elif score >= 42: regime_color = "yellow"
    elif score >= 32: regime_color = "red"
    else:             regime_color = "bold red"

    conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "dim"}.get(result.confidence, "dim")

    # ── Signal table ──────────────────────────────────────────────────────────
    sig_tbl = Table(box=None, show_header=True, header_style="bold dim", expand=True)
    sig_tbl.add_column("Signal",      style="dim",    width=22)
    sig_tbl.add_column("Value",       style="dim",    width=24)
    sig_tbl.add_column("Score",       width=8)
    sig_tbl.add_column("",            width=22)  # bar
    sig_tbl.add_column("Note",        style="dim")

    for s in result.signals:
        sc = s.score
        if   sc >= 65: sc_color = "green"
        elif sc >= 55: sc_color = "cyan"
        elif sc >= 45: sc_color = "yellow"
        elif sc >= 35: sc_color = "orange3"
        else:          sc_color = "red"
        bar_filled = int(sc / 100 * 18)
        bar = f"[{sc_color}]{'█' * bar_filled}[/{sc_color}][dim]{'░' * (18 - bar_filled)}[/dim]"
        sig_tbl.add_row(
            s.name,
            s.raw_value[:22],
            f"[{sc_color}]{sc:.0f}[/{sc_color}]",
            bar,
            s.note[:60],
        )

    # ── Trade panel ───────────────────────────────────────────────────────────
    if result.action == "NO_TRADE":
        trade_line = f"[yellow]❌ NO TRADE — {result.rationale}[/yellow]"
    elif result.option_type == "STRANGLE":
        trade_line = (
            f"[bold]📊 SHORT STRANGLE[/bold]   "
            f"[red]SELL {result.strike}CE @ ₹{result.entry_price:.1f}[/red]   "
            f"[green]SELL {result.strike_pe}PE @ ₹{result.entry_price_pe:.1f}[/green]   "
            f"Capital: ₹{result.capital:,.0f}   "
            f"[dim]SL_CE ₹{result.sl:.1f}  SL_PE ₹{result.sl_pe:.1f}[/dim]"
        )
    else:
        color = "green" if result.option_type == "CE" else "red"
        trade_line = (
            f"[bold {color}]{result.action} {result.strike}{result.option_type}[/bold {color}]   "
            f"₹{result.entry_price:.1f}   "
            f"[dim]SL ₹{result.sl:.1f}[/dim]   "
            f"T1 ₹{result.target1:.1f}   "
            f"T2 ₹{result.target2:.1f}   "
            f"R:R 1:{result.rr_ratio:.1f}   "
            f"{result.qty_lots}L   ₹{result.capital:,.0f}"
        )

    body = Table.grid(padding=(0, 1), expand=True)
    body.add_column()
    body.add_row(sig_tbl)
    body.add_row(Text(""))
    body.add_row(Text(
        f"Spot: ₹{result.spot:,.2f}   ATM: {result.atm}   "
        f"PCR: {result.pcr}   VIX: {result.vix}   "
        f"Expiry: {result.expiry}",
        style="dim",
    ))
    body.add_row(Text(
        f"Regime: {result.regime}   Score: {result.total_score:.1f}/100   "
        f"Confidence: {result.confidence}",
        style=regime_color,
    ))
    body.add_row(Text(""))
    body.add_row(Text(trade_line))
    body.add_row(Text(""))
    if result.key_levels:
        body.add_row(Text("Key levels: " + "  |  ".join(result.key_levels), style="dim"))
    body.add_row(Text(f"Rationale: {result.rationale}", style="dim"))

    console.print()
    console.print(
        Panel(
            body,
            border_style="color(208)",
            title=f"[bold color(208)]⚡ /next — Pure Calculation Prediction[/bold color(208)]",
            subtitle=(
                f"[dim]Fetched in {result.fetch_time_ms}ms  ·  "
                f"Total {elapsed:.1f}s  ·  No AI used  ·  {result.fetched_at}[/dim]"
            ),
        )
    )
    console.print()


# ── Day forecast display ──────────────────────────────────────────────────────

def show_day_forecast(forecast, elapsed: float) -> None:
    """Rich display for AI-generated 30-min candle forecast table."""
    from rich.table import Table

    bias = forecast.day_bias
    bc   = {"BULLISH":"green","BEARISH":"red","SIDEWAYS":"yellow","VOLATILE":"orange3"}.get(bias,"dim")
    cc   = {"HIGH":"green","MEDIUM":"yellow","LOW":"dim"}.get(forecast.confidence,"dim")

    # ── Candle table ──────────────────────────────────────────────────────────
    tbl = Table(
        title=f"[bold]📊 30-Min Candle Forecast — {forecast.generated_at}[/bold]",
        box=box.SIMPLE_HEAVY,
        header_style="bold color(208)",
        show_lines=False,
        expand=True,
    )
    tbl.add_column("Time",       width=13, style="dim")
    tbl.add_column("Direction",  width=12)
    tbl.add_column("Range",      width=16, style="dim")
    tbl.add_column("Key Level",  width=22, style="cyan")
    tbl.add_column("Bias",       width=9)
    tbl.add_column("Action",     style="dim")

    dir_icons = {
        "BULLISH":   ("🟢 BULLISH",  "green"),
        "BEARISH":   ("🔴 BEARISH",  "red"),
        "SIDEWAYS":  ("⚪ SIDEWAYS", "yellow"),
        "VOLATILE":  ("⚡ VOLATILE", "orange3"),
    }
    bias_icons = {
        "LONG":    ("📈 LONG",    "green"),
        "SHORT":   ("📉 SHORT",   "red"),
        "NEUTRAL": ("⚖  NEUTRAL", "dim"),
        "AVOID":   ("🚫 AVOID",   "dim"),
    }

    for slot in forecast.candle_slots:
        d_label, d_color = dir_icons.get(slot.direction, (slot.direction, "dim"))
        b_label, b_color = bias_icons.get(slot.bias,      (slot.bias,      "dim"))
        rng = (
            f"₹{slot.expected_low:,.0f}–{slot.expected_high:,.0f}"
            if slot.expected_high > 0 and slot.expected_low > 0
            else "—"
        )
        tbl.add_row(
            slot.time,
            Text(d_label, style=d_color),
            rng,
            slot.key_level[:22],
            Text(b_label, style=b_color),
            slot.action_note[:55],
        )

    # ── Next move panel ───────────────────────────────────────────────────────
    next_body = Table.grid(padding=(0, 2), expand=True)
    next_body.add_column(ratio=3)
    next_body.add_column(ratio=1)
    next_body.add_row(
        Text(f"📍 {forecast.next_move}", style="bold white"),
        Text(f"Level: {forecast.next_move_level}", style="cyan"),
    )
    if forecast.next_move_trade:
        next_body.add_row(
            Text(f"⚡ Trade: {forecast.next_move_trade}", style="bold color(208)"),
            Text(""),
        )

    # ── Day summary ───────────────────────────────────────────────────────────
    summary_body = Table.grid(padding=(0, 2), expand=True)
    summary_body.add_column(ratio=2)
    summary_body.add_column(ratio=2)
    summary_body.add_column(ratio=2)
    summary_body.add_row(
        Text(f"Day Bias: {bias}", style=f"bold {bc}"),
        Text(f"High est: ₹{forecast.day_high_est:,.2f}", style="green"),
        Text(f"Low est:  ₹{forecast.day_low_est:,.2f}",  style="red"),
    )
    if forecast.key_pivots:
        summary_body.add_row(
            Text("Pivots: " + "  |  ".join(forecast.key_pivots[:4]), style="dim"),
            Text(""),
            Text(""),
        )
    if forecast.ai_notes:
        summary_body.add_row(
            Text(f"💡 {forecast.ai_notes[:120]}", style="dim"),
            Text(""),
            Text(""),
        )

    console.print()
    console.print(
        Panel(
            Group(
                Panel(summary_body, border_style=bc, title=f"[{bc}]Day Summary[/{bc}]",
                      padding=(0,1)),
                Panel(next_body, border_style="color(208)",
                      title="[bold color(208)]Next Move (15–30 min)[/bold color(208)]",
                      padding=(0,1)),
                tbl,
            ),
            border_style="color(208)",
            title="[bold color(208)]🧠 AI Intraday Forecast[/bold color(208)]",
            subtitle=(
                f"[dim]AI generated  ·  Total {elapsed:.1f}s  ·  "
                f"Confidence [{cc}]{forecast.confidence}[/{cc}]  ·  "
                f"{forecast.generated_at}[/dim]"
            ),
        )
    )
    console.print()


# ── R1 analysis display ───────────────────────────────────────────────────────

def show_r1_analysis(scan, analysis, elapsed: float) -> None:
    """Rich display for R1 behavioural trade analysis."""

    conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "dim"}.get(
        analysis.confidence, "dim"
    )
    phase_color = {
        "opening": "yellow", "discovery": "cyan",
        "midday": "green", "closing": "orange3",
    }.get(scan.phase, "dim")

    # ── Pattern detections ────────────────────────────────────────────────────
    pat_tbl = Table(box=None, show_header=False, expand=True, padding=(0,1))
    pat_tbl.add_column(width=8)
    pat_tbl.add_column(width=28)
    pat_tbl.add_column()
    for p in scan.patterns:
        pc = {"HIGH":"red","MEDIUM":"yellow","LOW":"dim"}.get(p.confidence,"dim")
        pat_tbl.add_row(
            Text(f"[{p.confidence}]", style=pc),
            Text(p.name, style="bold"),
            Text(p.description[:80], style="dim"),
        )
    if not scan.patterns:
        pat_tbl.add_row("", Text("No patterns detected", style="dim"), "")

    # ── NOW snapshot bar ──────────────────────────────────────────────────────
    n    = scan.now
    cvr  = n.close_vs_range
    bar  = "█" * int(cvr * 20) + "░" * (20 - int(cvr * 20))
    dir_c = {"UP":"green","DOWN":"red","FLAT":"yellow"}.get(n.net_direction,"dim")
    vol_c = {"GROWING":"green","DYING":"red","STABLE":"dim"}.get(n.vol_trend,"dim")
    sh_c  = {"UPPER":"red","LOWER":"green","NONE":"dim"}.get(n.stop_hunt_signal,"dim")

    now_grid = Table.grid(padding=(0,2), expand=True)
    now_grid.add_column(ratio=2)
    now_grid.add_column(ratio=2)
    now_grid.add_column(ratio=2)
    now_grid.add_column(ratio=3)
    now_grid.add_row(
        Text(f"net_move {n.net_move:+.0f}pts [{n.net_direction}]", style=dir_c),
        Text(f"vol {n.vol_trend}", style=vol_c),
        Text(f"stop_hunt {n.stop_hunt_signal}", style=sh_c),
        Text(f"close_vs_range [{bar}] {cvr:.2f}", style="dim"),
    )
    now_grid.add_row(
        Text(f"wick↑{n.wick_up_last:.0f}  wick↓{n.wick_down_last:.0f}  body{n.body_last:.0f}", style="dim"),
        Text(f"first3={n.first_3_direction}  last3={n.last_3_direction}", style="dim"),
        Text(f"reversal={n.reversal_forming}", style="yellow" if n.reversal_forming else "dim"),
        Text(f"OR: {scan.opening_range.low:.0f}–{scan.opening_range.high:.0f}  "
             f"({scan.opening_range.spot_vs_or})", style="dim"),
    )

    # ── Scenario thinking ─────────────────────────────────────────────────────
    scenario_body = Table.grid(padding=(0,1), expand=True)
    scenario_body.add_column(ratio=1)
    scenario_body.add_column(ratio=1)
    scenario_body.add_row(
        Panel(
            Text(analysis.reading_a, style="dim"),
            title="[green]Reading A[/green]", border_style="dim", padding=(0,1),
        ),
        Panel(
            Text(analysis.reading_b, style="dim"),
            title="[red]Reading B[/red]", border_style="dim", padding=(0,1),
        ),
    )
    scenario_body.add_row(
        Panel(
            Text(analysis.now_says, style="bold white"),
            title=f"[{conf_color}]NOW SAYS[/{conf_color}]",
            border_style=conf_color, padding=(0,1),
        ),
        Panel(
            Text(analysis.behavioural_pattern or "—", style="dim"),
            title="[cyan]Behavioural Pattern[/cyan]",
            border_style="cyan", padding=(0,1),
        ),
    )

    # ── Trade panel ───────────────────────────────────────────────────────────
    trade_grid = Table.grid(padding=(0,2), expand=True)
    trade_grid.add_column(ratio=1)
    trade_grid.add_column(ratio=1)
    trade_grid.add_column(ratio=1)
    trade_grid.add_row(
        Text(f"Strike/Expiry\n{analysis.trade_strike}", style="bold color(208)"),
        Text(f"Entry\n{analysis.trade_entry}", style="bold green"),
        Text(f"Target\n{analysis.trade_target}", style="green"),
    )
    trade_grid.add_row(
        Text(f"Stop Loss\n{analysis.trade_sl}", style="bold red"),
        Text(f"R:R\n{analysis.trade_rr}", style="bold"),
        Text(f"Capital\n{analysis.trade_capital}", style="dim"),
    )
    trade_grid.add_row(
        Text(f"Based on\n{analysis.based_on[:60]}", style="dim"),
        Text(f"Invalidated if\n{analysis.invalidated_if[:60]}", style="dim"),
        Text(""),
    )

    # ── History summary ───────────────────────────────────────────────────────
    hist = scan.history
    hist_lines = []
    if hist.stop_hunt_levels_week:
        hist_lines.append(f"Stop-hunt zones (week): {hist.stop_hunt_levels_week[:3]}")
    if hist.tod_traps:
        for t in hist.tod_traps:
            hist_lines.append(f"Time trap: {t}")
    if hist.consecutive_patterns:
        hist_lines.append(f"Consecutive: {', '.join(hist.consecutive_patterns)}")
    hist_lines.append(f"Opening fake frequency: {hist.opening_fake_frequency}/5 days")
    if hist.summary:
        hist_lines.append(f"Summary: {hist.summary}")

    console.print()
    console.print(
        Panel(
            Group(
                # Header row
                Panel(
                    now_grid,
                    title=f"[bold]NOW  [{phase_color}]{scan.phase.upper()}[/{phase_color}]  "
                          f"Spot: {scan.spot:.2f}  VIX: {scan.vix}  PCR: {scan.pcr}[/bold]",
                    border_style=phase_color, padding=(0,1),
                ),
                # Patterns
                Panel(
                    pat_tbl,
                    title="[bold]Patterns Detected[/bold]",
                    border_style="dim", padding=(0,1),
                ),
                # Scenario
                Panel(
                    scenario_body,
                    title="[bold]Scenario Thinking[/bold]",
                    border_style="color(208)", padding=(0,1),
                ),
                # Trade
                Panel(
                    trade_grid,
                    title="[bold color(208)]Trade Setup[/bold color(208)]",
                    border_style="color(208)", padding=(0,1),
                ),
                # History
                Panel(
                    Text("\n".join(hist_lines), style="dim"),
                    title="[dim]Historical Behaviour (last week)[/dim]",
                    border_style="dim", padding=(0,1),
                ),
            ),
            border_style="color(208)",
            title=(
                f"[bold color(208)]🧠 R1 — Behavioural Pattern Scanner[/bold color(208)]  "
                f"[dim]{scan.scanned_at}[/dim]"
            ),
            subtitle=(
                f"[dim]Total {elapsed:.1f}s  ·  "
                f"Confidence [{conf_color}]{analysis.confidence}[/{conf_color}]  ·  "
                f"Patterns: {len(scan.patterns)}[/dim]"
            ),
        )
    )
    console.print()
