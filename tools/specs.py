"""
tools/specs.py
--------------
Tool definitions in Ollama's function-calling format (OpenAI-compatible).
These are passed as `tools` in each /api/chat request.
"""

from __future__ import annotations

TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name":        "list_expiries",
            "description": (
                "Fetch all available NIFTY option expiry dates from NSE India. "
                "Use this to find valid expiry date strings before calling other tools. "
                "Always call this if the user hasn't specified an expiry."
            ),
            "parameters": {
                "type":       "object",
                "properties": {},
                "required":   [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_spot_price",
            "description": (
                "Get the current NIFTY spot (index) price and nearest expiry date. "
                "Use this for quick spot price checks or when the user asks 'where is nifty' "
                "or 'what is the nifty level'."
            ),
            "parameters": {
                "type":       "object",
                "properties": {},
                "required":   [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_option_chain",
            "description": (
                "Fetch the NIFTY option chain for a given expiry. Returns spot price, "
                "PCR (Put-Call Ratio), ATM strike, max pain level, top OI strikes, "
                "and a full chain table with CE/PE OI, IV, LTP for strikes around ATM. "
                "Use this for option chain analysis, identifying support/resistance via OI, "
                "or checking current premiums."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "expiry": {
                        "type":        "string",
                        "description": "Expiry date in DD-Mon-YYYY format, e.g. '05-May-2026'. "
                                       "If omitted, uses the nearest expiry.",
                    },
                    "atm_range": {
                        "type":        "integer",
                        "description": "Number of strikes above and below ATM to include (default 10, max 25).",
                        "default":     10,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_oi_analysis",
            "description": (
                "Perform a deep open-interest analysis for NIFTY. Returns: "
                "PCR with sentiment interpretation, max pain level, top 5 CE resistance strikes, "
                "top 5 PE support strikes, fresh OI writing (new positions being added), "
                "OI unwinding (positions being closed), and IV skew around ATM. "
                "Use this when the user asks about OI buildup, unwinding, sentiment, "
                "support/resistance levels, or overall market positioning. "
                "Also use this as the primary input for trade recommendations."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "expiry": {
                        "type":        "string",
                        "description": "Expiry date in DD-Mon-YYYY format. Defaults to nearest expiry.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_chart_data",
            "description": (
                "Fetch intraday price and volume time-series for a specific NIFTY option strike. "
                "Returns OHLC summary (open, high, low, LTP), price change %, total volume, "
                "previous close, and the last 5 data ticks. "
                "Use this when the user asks about the price movement of a specific strike, "
                "'how is the 24000 CE doing today', trend of a particular option, etc. "
                "Set option_type to 'BOTH' to compare CE and PE of the same strike."
            ),
            "parameters": {
                "type":       "object",
                "properties": {
                    "expiry": {
                        "type":        "string",
                        "description": "Expiry date in DD-Mon-YYYY format, e.g. '05-May-2026'.",
                    },
                    "strike": {
                        "type":        "number",
                        "description": "Strike price, e.g. 24000 or 24500.",
                    },
                    "option_type": {
                        "type":        "string",
                        "enum":        ["CE", "PE", "BOTH"],
                        "description": "CE for call, PE for put, BOTH for both on same strike. Default CE.",
                        "default":     "CE",
                    },
                },
                "required": ["expiry", "strike"],
            },
        },
    },
{
        "type": "function",
        "function": {
            "name":        "get_market_news",
            "description": (
                "Fetch latest market-moving news via Google News. Covers NIFTY, BSE, "
                "RBI policy, SEBI, FII flows, geopolitical events (crude oil, US Fed, "
                "sanctions). Returns sentiment (BULLISH/BEARISH/NEUTRAL) and impact "
                "(HIGH/MEDIUM) per article. Use before any trade decision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days":     {"type": "integer", "description": "Lookback days (default 1)", "default": 1},
                    "category": {"type": "string",  "enum": ["all","market","geopolitical","rbi_sebi","fii"],
                                 "description": "News category. Default: all", "default": "all"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_fii_dii_data",
            "description": (
                "Fetch FII and DII cash market activity from NSE India. "
                "FII net buying > ₹500Cr = strong bullish institutional signal. "
                "FII net selling = bearish pressure. Use this to confirm directional bias."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_global_market_cues",
            "description": (
                "Fetch global market snapshots: S&P500, Nasdaq, Dow, Hang Seng, "
                "Nikkei, Crude Oil, Gold, Dollar Index, US 10Y yield. "
                "Global cues directly determine NIFTY opening gap direction. "
                "Always call this for pre-market or overnight trade setups."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_nifty_technicals",
            "description": (
                "Compute NIFTY intraday technicals from live 5-min candles: "
                "EMA9, EMA21, VWAP, day high/low, trend (UPTREND/DOWNTREND/SIDEWAYS). "
                "Use to confirm OI-based bias: OI says bullish + EMA9>EMA21 = high confidence. "
                "Disagreement between OI and technicals = skip the trade."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "get_vix_analysis",
            "description": (
                "India VIX current level + 5-day trend + regime (LOW_FEAR/NORMAL/ELEVATED/EXTREME). "
                "VIX regime is the most important strategy selector: "
                "LOW → sell premium, ELEVATED → buy options, EXTREME → no new trades. "
                "Always check VIX before deciding buy vs sell strategy."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
