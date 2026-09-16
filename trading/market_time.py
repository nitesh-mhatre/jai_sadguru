"""
trading/market_time.py
----------------------
Market time awareness for NSE India.

Answers:
  - Is market open RIGHT NOW?
  - What session phase is it? (pre-open / normal / closing / closed)
  - Why is it closed? (night / weekend / NSE holiday)
  - What is the next trading day?
  - Gap indicators for next open (GIFT Nifty / SGX)
  - What time-based trade rules apply?

Used in two places:
  1. agent.py  → injected into every system prompt as a time context block
  2. scanner.py → gates live-mode decisions (no new trades outside market hours)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Optional
import pytz

log = logging.getLogger(__name__)

# ── Timezone ───────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")


# ── NSE Trading Sessions ───────────────────────────────────────────────────────
# All times in IST
SESSION_PRE_OPEN_START  = time(9,  0)   # 09:00
SESSION_PRE_OPEN_END    = time(9, 15)   # 09:15
SESSION_NORMAL_START    = time(9, 15)   # 09:15
SESSION_NORMAL_END      = time(15, 30)  # 15:30
SESSION_CLOSING_START   = time(15, 30)  # 15:30
SESSION_CLOSING_END     = time(16,  0)  # 16:00


# ── NSE Holidays 2025–2026 ─────────────────────────────────────────────────────
# Source: NSE India official holiday list
# Format: date(YYYY, MM, DD): "Holiday name"
NSE_HOLIDAYS: dict[date, str] = {
    # 2025
    date(2025,  1, 26): "Republic Day",
    date(2025,  2, 26): "Mahashivratri",
    date(2025,  3, 14): "Holi",
    date(2025,  3, 31): "Id-Ul-Fitr (Ramadan Eid)",
    date(2025,  4, 10): "Shri Ram Navami",
    date(2025,  4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    date(2025,  4, 18): "Good Friday",
    date(2025,  5,  1): "Maharashtra Day",
    date(2025,  8, 15): "Independence Day",
    date(2025,  8, 27): "Ganesh Chaturthi",
    date(2025, 10,  2): "Mahatma Gandhi Jayanti",
    date(2025, 10,  2): "Dussehra",
    date(2025, 10, 20): "Diwali Laxmi Puja (Muhurat Trading)",
    date(2025, 10, 21): "Diwali Balipratipada",
    date(2025, 11,  5): "Prakash Gurpurb Sri Guru Nanak Dev Ji",
    date(2025, 12, 25): "Christmas",
    # 2026
    date(2026,  1, 26): "Republic Day",
    date(2026,  3, 20): "Holi",
    date(2026,  4,  3): "Good Friday",
    date(2026,  4, 14): "Dr. Baba Saheb Ambedkar Jayanti",
    date(2026,  4, 15): "Ram Navami",
    date(2026,  5,  1): "Maharashtra Day",
    date(2026,  8, 15): "Independence Day",
    date(2026,  8, 25): "Ganesh Chaturthi",
    date(2026, 10,  2): "Mahatma Gandhi Jayanti",
    date(2026, 10, 29): "Diwali Laxmi Puja",
    date(2026, 11, 11): "Gurunanak Jayanti",
    date(2026, 12, 25): "Christmas",
}


# ── Market Status Dataclass ────────────────────────────────────────────────────

@dataclass
class MarketStatus:
    now_ist:          datetime
    is_open:          bool
    phase:            str        # PRE_OPEN | NORMAL | CLOSING | CLOSED
    close_reason:     str        # "" | NIGHT | WEEKEND | HOLIDAY | EARLY_CLOSE
    holiday_name:     str        # name if today is a holiday
    next_trading_day: date
    next_open_ist:    datetime
    minutes_to_open:  int        # <0 if market is open
    minutes_to_close: int        # <0 if market is closed
    weekday_name:     str        # Monday…Sunday
    # Trade rules for this time
    trade_rules:      list[str]
    # Gap indicator (from GIFT Nifty if available)
    gift_nifty:       float      # 0.0 if unavailable
    gift_nifty_chng:  float      # change vs prev close
    gap_direction:    str        # "GAP_UP" | "GAP_DOWN" | "FLAT" | "UNKNOWN"

    def context_block(self) -> str:
        """
        Compact text injected into every AI system prompt.
        Tells the model exactly what time it is and what rules apply.
        """
        lines = [
            f"═══ MARKET TIME CONTEXT ═══",
            f"Current IST    : {self.now_ist.strftime('%A %d-%b-%Y %H:%M:%S IST')}",
            f"Market status  : {'🟢 OPEN' if self.is_open else '🔴 CLOSED'} — {self.phase}",
        ]

        if not self.is_open:
            lines.append(f"Closed because : {self.close_reason}"
                         + (f" ({self.holiday_name})" if self.holiday_name else ""))
            lines.append(f"Next trading   : {self.next_trading_day.strftime('%A %d-%b-%Y')}"
                         + (f" (opens in {self.minutes_to_open} min)"
                            if self.minutes_to_open > 0 else ""))

        if self.gift_nifty > 0:
            lines.append(
                f"GIFT Nifty     : {self.gift_nifty:.2f}  "
                f"chng={self.gift_nifty_chng:+.2f}  "
                f"→ {self.gap_direction}"
            )

        if self.trade_rules:
            lines.append("Trade rules    :")
            for r in self.trade_rules:
                lines.append(f"  • {r}")

        lines.append("═══════════════════════")
        return "\n".join(lines)


# ── Core function ──────────────────────────────────────────────────────────────

def get_market_status() -> MarketStatus:
    """Return a fully populated MarketStatus for the current moment."""
    now      = datetime.now(IST)
    today    = now.date()
    now_time = now.time()
    weekday  = now.weekday()   # 0=Mon … 6=Sun

    # ── Determine phase ─────────────────────────────────────────────────────
    holiday_name = _holiday_for(today)
    is_weekend   = weekday >= 5   # Sat/Sun
    is_holiday   = bool(holiday_name)

    if is_weekend or is_holiday:
        phase        = "CLOSED"
        is_open      = False
        close_reason = "WEEKEND" if is_weekend else "HOLIDAY"
    elif now_time < SESSION_PRE_OPEN_START:
        phase        = "CLOSED"
        is_open      = False
        close_reason = "NIGHT"
    elif now_time < SESSION_PRE_OPEN_END:
        phase        = "PRE_OPEN"
        is_open      = False          # can't trade in pre-open
        close_reason = "PRE_OPEN"
    elif now_time <= SESSION_NORMAL_END:
        phase        = "NORMAL"
        is_open      = True
        close_reason = ""
    elif now_time <= SESSION_CLOSING_END:
        phase        = "CLOSING"
        is_open      = False          # closing session — no new positions
        close_reason = "CLOSING_SESSION"
    else:
        phase        = "CLOSED"
        is_open      = False
        close_reason = "NIGHT"

    # ── Next trading day ──────────────────────────────────────────────────────
    next_td   = _next_trading_day(today)
    next_open = IST.localize(
        datetime.combine(next_td, SESSION_NORMAL_START)
    )
    now_aware = now if now.tzinfo else IST.localize(now)
    mins_to_open  = int((next_open - now_aware).total_seconds() / 60)
    if is_open:
        mins_to_open = -1

    next_close    = IST.localize(
        datetime.combine(today, SESSION_NORMAL_END)
    )
    mins_to_close = int((next_close - now_aware).total_seconds() / 60)
    if not is_open:
        mins_to_close = -1

    # ── Trade rules based on time ─────────────────────────────────────────────
    rules = _build_trade_rules(now, is_open, phase, next_td, today)

    # ── GIFT Nifty (gap indicator) ────────────────────────────────────────────
    gift, gift_chng, gap_dir = _get_gift_nifty()

    return MarketStatus(
        now_ist          = now,
        is_open          = is_open,
        phase            = phase,
        close_reason     = close_reason,
        holiday_name     = holiday_name,
        next_trading_day = next_td,
        next_open_ist    = next_open,
        minutes_to_open  = max(0, mins_to_open),
        minutes_to_close = max(0, mins_to_close),
        weekday_name     = now.strftime("%A"),
        trade_rules      = rules,
        gift_nifty       = gift,
        gift_nifty_chng  = gift_chng,
        gap_direction    = gap_dir,
    )


def _next_trading_day(from_date: date) -> date:
    """Return the next NSE trading day after from_date (or same day if open today)."""
    d = from_date
    now_time = datetime.now(IST).time()
    if (d.weekday() < 5 and not _holiday_for(d)
            and now_time < SESSION_NORMAL_END):
        return d
    d += timedelta(days=1)
    while d.weekday() >= 5 or _holiday_for(d):
        d += timedelta(days=1)
    return d


def _build_trade_rules(
    now: datetime, is_open: bool, phase: str,
    next_td: date, today: date,
) -> list[str]:
    rules = []
    now_time = now.time()

    if not is_open:
        # ── Market closed rules ───────────────────────────────────────────────
        rules.append(
            "Market is CLOSED — any LTP shown is the LAST TRADED PRICE from the previous session, "
            "NOT the current price. Do NOT treat it as a live price."
        )
        rules.append(
            "If suggesting a trade, present it as a 'watch for tomorrow' setup. "
            "State expected entry range at open, not current LTP."
        )
        # Weekend
        if now.weekday() >= 5:
            days_to_mon = 7 - now.weekday()
            rules.append(
                f"Weekend — next trading day is {next_td.strftime('%A %d-%b-%Y')}. "
                f"Factor in 2+ days of news flow, global markets, and SGX/GIFT Nifty gap at open."
            )
        # Pre-open
        if phase == "PRE_OPEN":
            rules.append(
                "PRE-OPEN session (09:00–09:15). No execution possible yet. "
                "Prices discovered via auction. Watch GIFT Nifty for gap direction."
            )
        # Holiday
        holiday_name = _holiday_for(today)
        if holiday_name:
            rules.append(
                f"Today is a NSE holiday ({holiday_name}). "
                f"Next trading day: {next_td.strftime('%A %d-%b-%Y')}."
            )
        # Check if next day is also a holiday
        next_day = today + timedelta(days=1)
        next_holiday = _holiday_for(next_day)
        if next_holiday:
            rules.append(
                f"Note: tomorrow ({next_day.strftime('%d-%b')}) is also a holiday ({next_holiday}). "
                f"Extended non-trading gap — factor in higher uncertainty at open."
            )

    else:
        # ── Market open rules ─────────────────────────────────────────────────
        if time(9, 15) <= now_time <= time(9, 30):
            rules.append(
                "⚡ OPENING MOMENTUM WINDOW (09:15–09:30): This is the highest-velocity "
                "15 minutes of the day. A gap open followed by a confirming candle is a "
                "strong scalp opportunity. BUY ATM option in gap direction. "
                "SL=30% of premium, Target=50%, HARD EXIT at 09:30 — no exceptions. "
                "Max 1 lot only. If candle is DOJI or conflicts gap → skip entirely."
            )
        elif time(9, 30) <= now_time <= time(9, 45):
            rules.append(
                "POST-OPENING (09:30–09:45): Opening momentum window just closed. "
                "Close any OM scalp trades now if not already exited. "
                "Wait for 09:45 before starting normal intraday analysis."
            )
        elif time(9, 45) <= now_time <= time(11, 30):
            rules.append(
                "MORNING SESSION (09:45–11:30): Prime trading window. "
                "OI buildup is most reliable. Institutional flows visible. Good for directional trades."
            )
        elif time(11, 30) <= now_time <= time(14, 0):
            rules.append(
                "MID-SESSION (11:30–14:00): Often range-bound. "
                "Prefer selling premium (strangles/condors) if IV is elevated. "
                "Be cautious with directional bets unless clear trend."
            )
        elif time(14, 0) <= now_time <= time(14, 30):
            rules.append(
                "PRE-EXPIRY CAUTION ZONE (14:00–14:30 on expiry day or near expiry): "
                "Gamma risk spikes. Option premiums decay aggressively. "
                "Tighten stop-losses on all open positions."
            )
        elif time(14, 30) <= now_time <= time(15, 0):
            rules.append(
                "CLOSING APPROACH (14:30–15:00): Institutional squaring of positions. "
                "Avoid new entries. Focus on managing and closing existing positions."
            )
        elif now_time >= time(15, 0):
            rules.append(
                "LAST 30 MIN (15:00–15:30): High volatility, low liquidity on OTM options. "
                "Close all intraday positions. Do NOT open new trades."
            )

        # Expiry day check
        try:
            from data.nifty_option_chain import get_expiry_dates
            expiries = get_expiry_dates()
            if expiries and expiries[0] == today.strftime("%d-%b-%Y"):
                rules.append(
                    f"TODAY IS EXPIRY DAY ({expiries[0]}). "
                    "Theta decay is maximum — option buyers lose value rapidly. "
                    "Sellers have edge but gamma risk is extreme near ATM. "
                    "No new BUY positions after 13:00. Close all positions before 15:15."
                )
        except Exception:
            pass

        # Minutes to close
        mins_left = int(
            (IST.localize(datetime.combine(today, SESSION_NORMAL_END)) - now
             ).total_seconds() / 60
        )
        if mins_left < 45:
            rules.append(
                f"Only {mins_left} minutes until market close (15:30). "
                "Do not open new intraday trades. Focus on exits."
            )

    return rules


# ── GIFT Nifty cache ─────────────────────────────────────────────────────────
# yfinance fast_info is slow (5–30s) and sometimes hangs; get_market_status()
# runs on EVERY agent turn, so this must be cached. 10-min TTL is plenty for a
# gap indicator. A stale value is far better than a frozen CLI.

_GIFT_CACHE: dict = {"ts": 0.0, "value": (0.0, 0.0, "UNKNOWN")}
_GIFT_TTL      = 600      # seconds
_GIFT_TIMEOUT  = 8        # hard cap for the fetch itself


def _get_gift_nifty() -> tuple[float, float, str]:
    """
    Fetch GIFT Nifty (NSE IFSC) as gap indicator — cached (10 min TTL).
    Falls back to 0.0 if unavailable. Never blocks for long.
    Returns (gift_price, change, direction)
    """
    import time as _time
    now = _time.time()
    if now - _GIFT_CACHE["ts"] < _GIFT_TTL:
        return _GIFT_CACHE["value"]

    value = (0.0, 0.0, "UNKNOWN")
    try:
        import yfinance as yf
        t    = yf.Ticker("^NSEI")
        info = t.fast_info
        prev = float(info.previous_close or 0)
        curr = float(info.last_price     or 0)
        if curr > 0 and prev > 0:
            chng = curr - prev
            pct  = chng / prev * 100
            if   pct >  0.3: direction = "GAP_UP"
            elif pct < -0.3: direction = "GAP_DOWN"
            else:             direction = "FLAT"
            value = (round(curr, 2), round(chng, 2), direction)
    except Exception:
        pass

    _GIFT_CACHE["ts"]    = now
    _GIFT_CACHE["value"] = value
    return value


def _holiday_for(d: date) -> str:
    """NSE holiday name for a date, with graceful behaviour beyond the table."""
    if d in NSE_HOLIDAYS:
        return NSE_HOLIDAYS[d]
    if d > max(NSE_HOLIDAYS):
        # Beyond known table — assume weekday rule; user can extend NSE_HOLIDAYS.
        return ""
    # Fixed-date holidays that repeat every year (approximate — adjust yearly)
    if (d.month, d.day) in {(1, 26), (8, 15), (10, 2), (12, 25), (5, 1)}:
        return "NSE Holiday (fixed-date)"
    return ""
