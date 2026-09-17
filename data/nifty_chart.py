"""
Nifty Option Chart Data Fetcher & Plotter
------------------------------------------
Fetches intraday price/OI time-series for a specific NIFTY strike+expiry
and renders an interactive HTML chart (Plotly).

Identifier format used by NSE:
    OPTIDXNIFTY{DD-MM-YYYY}{CE|PE}{STRIKE}.00
    e.g.  OPTIDXNIFTY05-05-2026CE24000.00

Usage:
    python nifty_chart.py --expiry 05-May-2026 --strike 24000
    python nifty_chart.py --expiry 05-May-2026 --strike 24000 --type CE
    python nifty_chart.py --expiry 05-May-2026 --strike 24000 --both     # CE + PE overlay
    python nifty_chart.py --expiry 05-May-2026 --strike 24000 --both --save-csv

Requirements:
    pip install requests pandas plotly
"""

import requests
import pandas as pd
import argparse
import sys
import json
import logging
from datetime import datetime

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

CHART_URL = "https://www.nseindia.com/api/chart-databyindex?index={identifier}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_identifier(expiry: str, strike: float, option_type: str) -> str:
    """
    Build the NSE chart identifier string.

    Args:
        expiry      : '05-May-2026'  (DD-Mon-YYYY)
        strike      : 24000
        option_type : 'CE' or 'PE'

    Returns:
        'OPTIDXNIFTY05-05-2026CE24000.00'
    """
    # Convert 05-May-2026 → 05-05-2026
    dt = datetime.strptime(expiry, "%d-%b-%Y")
    expiry_fmt = dt.strftime("%d-%m-%Y")
    strike_fmt = f"{float(strike):.2f}"
    return f"OPTIDXNIFTY{expiry_fmt}{option_type.upper()}{strike_fmt}"


def _nse_json(url: str) -> dict:
    """
    GET an NSE JSON endpoint through the cookie-managed session.
    NSE returns 403 to plain requests without the browser cookies that
    data/nse_session.py establishes, so this is the fix for "no chart data".
    """
    try:
        from data.nse_session import nse_get
        resp = nse_get(url, headers=HEADERS)
    except Exception as exc:
        log.debug("NSE session fetch failed (%s) — falling back to requests", exc)
        resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_chart_data(identifier: str) -> pd.DataFrame:
    """
    Fetch time-series chart data for a given NSE option identifier.

    Returns a DataFrame with columns: timestamp, price, volume
    """
    url = CHART_URL.format(identifier=identifier)

    raw = _nse_json(url)

    # NSE returns: {"grapData": [[timestamp_ms, price], ...], "volume": [[timestamp_ms, vol], ...]}
    graph_data = raw.get("grapthData", [])  # NSE typo: "grapth" not "graph"
    volume_data = raw.get("volume", [])

    if not graph_data:
        log.warning("No chart data returned for %s", identifier)
        return pd.DataFrame()

    df_price = pd.DataFrame(graph_data, columns=["timestamp_ms", "price"])
    df_price["timestamp"] = pd.to_datetime(df_price["timestamp_ms"], unit="ms", utc=True) \
                              .dt.tz_convert("Asia/Kolkata") \
                              .dt.tz_localize(None)

    if volume_data:
        df_vol = pd.DataFrame(volume_data, columns=["timestamp_ms", "volume"])
        df_price = df_price.merge(df_vol[["timestamp_ms", "volume"]], on="timestamp_ms", how="left")
    else:
        df_price["volume"] = 0

    df_price["close_price"] = raw.get("closePrice", None)
    df_price = df_price[["timestamp", "price", "volume", "close_price"]].sort_values("timestamp").reset_index(drop=True)
    return df_price


def fetch_both(expiry: str, strike: float) -> tuple:
    """Fetch CE and PE chart data for the same strike. Returns (df_ce, df_pe)."""
    id_ce = build_identifier(expiry, strike, "CE")
    id_pe = build_identifier(expiry, strike, "PE")
    print(f"\n[*] Fetching CE data: {id_ce}")
    df_ce = fetch_chart_data(id_ce)
    print(f"[*] Fetching PE data: {id_pe}")
    df_pe = fetch_chart_data(id_pe)
    return df_ce, df_pe


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_single(df: pd.DataFrame, identifier: str, expiry: str,
                strike: float, option_type: str) -> str:
    """Plot price + volume for a single CE or PE. Returns output HTML filename."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[Error] plotly not installed. Run: pip install plotly")
        sys.exit(1)

    color = "#2196F3" if option_type == "CE" else "#E91E63"
    title = f"NIFTY {expiry} | Strike {strike:.0f} | {option_type}"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Price (LTP)", "Volume")
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["price"],
        mode="lines",
        name=f"{option_type} Price",
        line=dict(color=color, width=2),
        hovertemplate="Time: %{x}<br>Price: ₹%{y:.2f}<extra></extra>"
    ), row=1, col=1)

    # Close price reference line
    close = df["close_price"].iloc[0] if "close_price" in df.columns and df["close_price"].iloc[0] else None
    if close:
        fig.add_hline(y=close, line_dash="dash", line_color="orange",
                      annotation_text=f"Prev Close ₹{close:.2f}",
                      annotation_position="top right", row=1, col=1)

    # Volume bars
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["volume"],
        name="Volume",
        marker_color=color,
        opacity=0.5,
        hovertemplate="Time: %{x}<br>Volume: %{y:,.0f}<extra></extra>"
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis2_title="Time",
        yaxis_title="Price (₹)",
        yaxis2_title="Volume",
        hovermode="x unified",
        template="plotly_dark",
        height=600,
        showlegend=True,
        legend=dict(orientation="h", y=1.08)
    )

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_chart_{strike:.0f}_{option_type}_{ts}.html"
    fig.write_html(filename)
    print(f"\n[Chart saved] {filename}")
    return filename


def plot_both(df_ce: pd.DataFrame, df_pe: pd.DataFrame,
              expiry: str, strike: float) -> str:
    """Plot CE and PE price + volume overlaid. Returns output HTML filename."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("[Error] plotly not installed. Run: pip install plotly")
        sys.exit(1)

    title = f"NIFTY {expiry} | Strike {strike:.0f} | CE vs PE"

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.05,
        subplot_titles=("Price (LTP) — CE vs PE", "Volume — CE vs PE")
    )

    # CE price
    if not df_ce.empty:
        fig.add_trace(go.Scatter(
            x=df_ce["timestamp"], y=df_ce["price"],
            mode="lines", name="CE Price",
            line=dict(color="#2196F3", width=2),
            hovertemplate="CE Price: ₹%{y:.2f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=df_ce["timestamp"], y=df_ce["volume"],
            name="CE Volume", marker_color="#2196F3", opacity=0.4,
            hovertemplate="CE Volume: %{y:,.0f}<extra></extra>"
        ), row=2, col=1)

    # PE price
    if not df_pe.empty:
        fig.add_trace(go.Scatter(
            x=df_pe["timestamp"], y=df_pe["price"],
            mode="lines", name="PE Price",
            line=dict(color="#E91E63", width=2),
            hovertemplate="PE Price: ₹%{y:.2f}<extra></extra>"
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=df_pe["timestamp"], y=df_pe["volume"],
            name="PE Volume", marker_color="#E91E63", opacity=0.4,
            hovertemplate="PE Volume: %{y:,.0f}<extra></extra>"
        ), row=2, col=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis2_title="Time",
        yaxis_title="Price (₹)",
        yaxis2_title="Volume",
        hovermode="x unified",
        template="plotly_dark",
        height=650,
        barmode="group",
        legend=dict(orientation="h", y=1.08)
    )

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"nifty_chart_{strike:.0f}_CE_PE_{ts}.html"
    fig.write_html(filename)
    print(f"\n[Chart saved] {filename}")
    return filename


# ── Module API ────────────────────────────────────────────────────────────────

def get_option_chart(expiry: str, strike: float, option_type: str = "CE") -> pd.DataFrame:
    """
    Fetch chart data as a DataFrame. Use when importing as a module.

    Example:
        from nifty_chart import get_option_chart
        df = get_option_chart('05-May-2026', 24000, 'CE')
        print(df.head())
    """
    identifier = build_identifier(expiry, strike, option_type)
    return fetch_chart_data(identifier)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch & plot intraday chart data for a NIFTY option strike.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nifty_chart.py --expiry 05-May-2026 --strike 24000
  python nifty_chart.py --expiry 05-May-2026 --strike 24000 --type PE
  python nifty_chart.py --expiry 05-May-2026 --strike 24000 --both
  python nifty_chart.py --expiry 05-May-2026 --strike 24000 --both --save-csv
        """
    )
    parser.add_argument("--expiry",   "-e", required=True,
                        help="Expiry date e.g. '05-May-2026'")
    parser.add_argument("--strike",   "-k", required=True, type=float,
                        help="Strike price e.g. 24000")
    parser.add_argument("--type",     "-t", default="CE", choices=["CE", "PE"],
                        help="Option type: CE (default) or PE")
    parser.add_argument("--both",     "-b", action="store_true",
                        help="Fetch and overlay both CE and PE on the same chart")
    parser.add_argument("--save-csv", "-s", action="store_true",
                        help="Save raw time-series data to CSV")
    args = parser.parse_args()

    if args.both:
        df_ce, df_pe = fetch_both(args.expiry, args.strike)

        if args.save_csv:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not df_ce.empty:
                f = f"chart_{args.strike:.0f}_CE_{ts}.csv"
                df_ce.to_csv(f, index=False)
                print(f"[Saved] {f}")
            if not df_pe.empty:
                f = f"chart_{args.strike:.0f}_PE_{ts}.csv"
                df_pe.to_csv(f, index=False)
                print(f"[Saved] {f}")

        plot_both(df_ce, df_pe, args.expiry, args.strike)

    else:
        identifier = build_identifier(args.expiry, args.strike, args.type)
        print(f"\n[*] Fetching chart data: {identifier}")
        df = fetch_chart_data(identifier)

        if df.empty:
            print("[Error] No data returned.")
            sys.exit(1)

        print(f"\n  Rows     : {len(df)}")
        print(f"  From     : {df['timestamp'].iloc[0]}")
        print(f"  To       : {df['timestamp'].iloc[-1]}")
        print(f"  Price    : min={df['price'].min():.2f}  max={df['price'].max():.2f}  last={df['price'].iloc[-1]:.2f}")
        print(f"\n{df.tail(10).to_string(index=False)}")

        if args.save_csv:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            f  = f"chart_{args.strike:.0f}_{args.type}_{ts}.csv"
            df.to_csv(f, index=False)
            print(f"\n[Saved] {f}")

        plot_single(df, identifier, args.expiry, args.strike, args.type)

    print("\n[Done] Open the .html file in your browser to view the interactive chart.")


if __name__ == "__main__":
    main()

# ── Module API: both legs ─────────────────────────────────────────────────────

def get_both_charts(expiry: str, strike: float):
    """
    Fetch CE and PE chart data for the same strike.
    Used by tools/implementations.py.

    Returns:
        (df_ce, df_pe)  — each is a DataFrame with columns: timestamp, price, volume, close_price
    """
    return fetch_both(expiry, strike)
