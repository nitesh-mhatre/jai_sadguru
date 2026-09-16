"""
Nifty Option Chain Fetcher
--------------------------
Usage:
    python nifty_option_chain.py                        # nearest expiry
    python nifty_option_chain.py --expiry 05-May-2026   # specific expiry
    python nifty_option_chain.py --atm 10               # ±10 strikes around ATM
    python nifty_option_chain.py --save                 # save to CSV
    python nifty_option_chain.py --list-expiries        # show all expiry dates

Requirements:
    pip install requests pandas tabulate
"""

import requests
import pandas as pd
import argparse
import sys
from datetime import datetime

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/91.0.4472.124 Safari/537.36"
    )
}

BASE_URL         = "https://www.nseindia.com/api/option-chain-v3?type=Indices&symbol=NIFTY"
CONTRACT_INFO_URL = "https://www.nseindia.com/api/option-chain-contract-info?symbol=NIFTY"


def fetch_option_chain(expiry: str = None) -> dict:
    """Fetch raw option chain JSON from NSE v3 API."""
    url = BASE_URL
    if expiry:
        url += f"&expiry={expiry}"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return response.json()


def get_expiry_dates() -> list:
    """
    Fetch available expiry dates for NIFTY.
    Tries the dedicated contract-info endpoint first;
    falls back to parsing dates from the option chain data.
    """
    try:
        response = requests.get(CONTRACT_INFO_URL, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            dates = response.json().get("expiryDates", [])
            if dates:
                return dates
    except Exception:
        pass

    # Fallback: extract unique expiry dates from the option chain itself
    response = requests.get(BASE_URL, headers=HEADERS, timeout=15)
    response.raise_for_status()
    data = response.json()
    expiries = []
    seen = set()
    for row in data.get("records", {}).get("data", []):
        for side in ("CE", "PE"):
            exp = (row.get(side) or {}).get("expiryDate", "")
            if exp and exp not in seen:
                seen.add(exp)
                expiries.append(exp)
    return expiries


def parse_option_chain(data: dict):
    """Parse NSE JSON into a clean DataFrame. Returns (df, spot_price)."""
    records    = data.get("records", {})
    spot_price = records.get("underlyingValue", 0)
    rows = []

    for item in records.get("data", []):
        ce = item.get("CE") or {}
        pe = item.get("PE") or {}
        rows.append({
            "Strike":     item.get("strikePrice"),
            "Expiry":     ce.get("expiryDate") or pe.get("expiryDate", ""),
            # ── CALL ────────────────────────────────────────
            "CE_OI":      ce.get("openInterest",         0),
            "CE_Chng_OI": ce.get("changeinOpenInterest", 0),
            "CE_Volume":  ce.get("totalTradedVolume",    0),
            "CE_IV":      ce.get("impliedVolatility",    0),
            "CE_LTP":     ce.get("lastPrice",            0),
            "CE_Chng":    ce.get("change",               0),
            "CE_Bid":     ce.get("buyPrice1",            0),
            "CE_Ask":     ce.get("sellPrice1",           0),
            # ── PUT ─────────────────────────────────────────
            "PE_Bid":     pe.get("buyPrice1",            0),
            "PE_Ask":     pe.get("sellPrice1",           0),
            "PE_LTP":     pe.get("lastPrice",            0),
            "PE_Chng":    pe.get("change",               0),
            "PE_IV":      pe.get("impliedVolatility",    0),
            "PE_Volume":  pe.get("totalTradedVolume",    0),
            "PE_Chng_OI": pe.get("changeinOpenInterest", 0),
            "PE_OI":      pe.get("openInterest",         0),
        })

    df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
    return df, spot_price


def filter_atm(df: pd.DataFrame, spot: float, n: int) -> pd.DataFrame:
    """Keep only ±n strikes around the ATM strike."""
    strikes = sorted(df["Strike"].unique())
    atm     = min(strikes, key=lambda x: abs(x - spot))
    idx     = strikes.index(atm)
    keep    = strikes[max(0, idx - n): idx + n + 1]
    return df[df["Strike"].isin(keep)].reset_index(drop=True)


def print_chain(df: pd.DataFrame, spot: float):
    """Pretty-print the option chain table."""
    try:
        from tabulate import tabulate
        use_tabulate = True
    except ImportError:
        use_tabulate = False

    cols = ["CE_OI", "CE_Chng_OI", "CE_Volume", "CE_IV", "CE_LTP",
            "Strike",
            "PE_LTP", "PE_IV", "PE_Volume", "PE_Chng_OI", "PE_OI"]
    view = df[[c for c in cols if c in df.columns]].copy()

    atm = min(df["Strike"].unique(), key=lambda x: abs(x - spot))
    view.insert(0, " ", df["Strike"].apply(lambda x: "<<ATM" if x == atm else ""))

    expiry_label = df["Expiry"].iloc[0] if not df.empty else ""
    print(f"\n{'='*105}")
    print(f"  NIFTY Option Chain  |  Expiry: {expiry_label}  |  Spot: Rs.{spot:,.2f}  |  {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}")
    print(f"{'='*105}")

    if use_tabulate:
        print(tabulate(view, headers="keys", tablefmt="rounded_outline",
                       floatfmt=".2f", showindex=False))
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(view.to_string(index=False))

    ce_oi = df["CE_OI"].sum()
    pe_oi = df["PE_OI"].sum()
    print(f"\n  CE OI: {ce_oi:,.0f}  |  PE OI: {pe_oi:,.0f}  |  PCR: {pe_oi/ce_oi:.2f}  |  Strikes: {len(df)}")


def save_to_csv(df: pd.DataFrame, spot: float) -> str:
    filename = f"nifty_option_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"\n[Saved] {filename}  (spot=Rs.{spot:.2f}, rows={len(df)})")
    return filename


# ── Module API ────────────────────────────────────────────────────────────────

def get_nifty_option_chain(expiry: str = None, atm_range: int = None):
    """
    Use this when importing as a module.

    Example:
        from nifty_option_chain import get_nifty_option_chain
        df, spot = get_nifty_option_chain(expiry='05-May-2026', atm_range=10)
    """
    data     = fetch_option_chain(expiry)
    df, spot = parse_option_chain(data)
    if atm_range:
        df = filter_atm(df, spot, atm_range)
    return df, spot


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch NIFTY option chain from NSE India.")
    parser.add_argument("--expiry", "-e", default=None,
                        help="Expiry date e.g. '05-May-2026'. Default: nearest expiry.")
    parser.add_argument("--atm", "-a", type=int, default=None, metavar="N",
                        help="Show only ±N strikes around ATM.")
    parser.add_argument("--save", "-s", action="store_true",
                        help="Save to timestamped CSV.")
    parser.add_argument("--list-expiries", "-l", action="store_true",
                        help="List all available expiry dates and exit.")
    args = parser.parse_args()

    print("[*] Fetching data from NSE...")

    if args.list_expiries:
        expiries = get_expiry_dates()
        print(f"\nAvailable NIFTY expiry dates ({len(expiries)}):")
        for i, e in enumerate(expiries, 1):
            print(f"  {i:3d}.  {e}")
        return

    data     = fetch_option_chain(expiry=args.expiry)
    df, spot = parse_option_chain(data)

    if df.empty:
        print("[Error] No data returned. Check the expiry date format e.g. 05-May-2026")
        sys.exit(1)

    if args.atm is not None:
        df = filter_atm(df, spot, args.atm)

    print_chain(df, spot)

    if args.save:
        save_to_csv(df, spot)

    return df, spot


if __name__ == "__main__":
    main()