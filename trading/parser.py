"""
trading/parser.py
-----------------
Natural-language parser for the /go command.

Extracts: direction, budget (₹), max_loss_pct, interval_seconds.

Example inputs:
    /go only buy budget is 5000 rs only
    /go budget is 100000 and loss taking capacity is 10 percent only
    /go only sell budget 25000 max loss 5% interval 3 minutes
    /go both sides budget 200000 loss 15 percent
"""

from __future__ import annotations

import re


def parse_go_command(message: str) -> dict:
    """
    Parse the text that follows '/go' and return a config dict.

    Returns
    -------
    {
        "direction":        "BUY" | "SELL" | "BOTH",
        "budget":           float,   # ₹ amount
        "max_loss_pct":     float,   # e.g. 10.0 for 10%
        "interval_seconds": int,     # seconds between agent decisions
    }
    """
    msg = message.lower().strip()

    result: dict = {
        "direction":        "BOTH",
        "budget":           10_000.0,
        "max_loss_pct":     20.0,   # conservative default
        "interval_seconds": 120,    # 2 minutes
    }

    # ── Direction ─────────────────────────────────────────────────────────────
    if re.search(r'\bonly\s+buy\b|\bbuy\s+only\b|\bbuy\s+side\b', msg):
        result["direction"] = "BUY"
    elif re.search(r'\bonly\s+sell\b|\bsell\s+only\b|\bsell\s+side\b', msg):
        result["direction"] = "SELL"
    # else BOTH

    # ── Budget ────────────────────────────────────────────────────────────────
    # "budget is 5000", "budget 5000", "capital 100000", "5000 rs", "₹50000"
    budget_patterns = [
        r'(?:budget|capital|amount)\s+(?:is\s+|of\s+)?(?:rs\.?\s*|₹\s*)?(\d[\d,]*(?:\.\d+)?)',
        r'(?:rs\.?|₹)\s*(\d[\d,]*(?:\.\d+)?)\s*(?:budget|capital|only)?',
        r'(\d[\d,]*(?:\.\d+)?)\s*(?:rs|rupees|inr)(?!\s*percent|\s*%)',
        r'(\d[\d,]+)\s*(?:rs\.?)?(?=\s+(?:and|only|loss|stop|max|\Z))',
    ]
    for pat in budget_patterns:
        m = re.search(pat, msg)
        if m:
            val = m.group(1).replace(',', '')
            result["budget"] = float(val)
            break

    # ── Max-loss percentage ───────────────────────────────────────────────────
    loss_patterns = [
        r'(?:loss\s+taking\s+capacity|max\s+loss|loss\s+limit|drawdown|stop\s+loss\s+capital)\s+'
        r'(?:is\s+)?(\d+(?:\.\d+)?)\s*(?:percent|%|p\.?c\.?)',
        r'(?:loss|drawdown)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*(?:percent|%)',
        r'(\d+(?:\.\d+)?)\s*(?:percent|%)\s+(?:loss|drawdown|max|limit)',
        r'(?:if\s+capital\s+lost\s+by\s+|capital\s+loss\s+)(\d+(?:\.\d+)?)\s*(?:percent|%)',
        r'(\d+(?:\.\d+)?)\s*%\s*(?:stop|limit)',
    ]
    for pat in loss_patterns:
        m = re.search(pat, msg)
        if m:
            result["max_loss_pct"] = float(m.group(1))
            break

    # ── Interval ──────────────────────────────────────────────────────────────
    interval_patterns = [
        r'(?:every|interval)\s+(\d+)\s*min(?:ute)?s?',
        r'(\d+)\s*min(?:ute)?s?\s+(?:interval|cycle|refresh)',
        r'(?:every|each)\s+(\d+)\s*(?:mins?|minutes?)',
    ]
    for pat in interval_patterns:
        m = re.search(pat, msg)
        if m:
            result["interval_seconds"] = int(m.group(1)) * 60
            break

    return result


def summarise_params(params: dict) -> str:
    """Human-readable summary of parsed /go params."""
    direction_label = {
        "BUY":  "🟢 BUY only",
        "SELL": "🔴 SELL only",
        "BOTH": "⚡ Both BUY & SELL",
    }.get(params["direction"], params["direction"])

    return (
        f"{direction_label}  |  "
        f"Budget ₹{params['budget']:,.0f}  |  "
        f"Max Loss {params['max_loss_pct']}%  |  "
        f"Decision interval {params['interval_seconds']//60}m {params['interval_seconds']%60}s"
    )
