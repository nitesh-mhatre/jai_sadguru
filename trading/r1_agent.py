"""
trading/r1_agent.py
--------------------
R1 Agent — calls AI with ScanPacket and R1 behavioural prompts.
Returns structured trade analysis with scenario thinking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class R1Analysis:
    phase:            str
    reading_a:        str
    reading_b:        str
    now_says:         str
    behavioural_pattern: str
    trade_strike:     str
    trade_expiry:     str
    trade_entry:      str
    trade_target:     str
    trade_sl:         str
    trade_rr:         str
    trade_capital:    str
    based_on:         str
    invalidated_if:   str
    raw_response:     str
    scan_at:          str
    confidence:       str = "MEDIUM"


def run_r1(agent, scan_packet) -> R1Analysis:
    """
    Build the R1 prompt from scan packet and call AI.
    AI receives raw behavioural data + R1 scanner + trade format prompts.
    Zero tool calls — all data is pre-fetched.
    """
    from agent import Session, FinalAnswerEvent, ErrorEvent
    from config import R1_SCANNER_PROMPT, R1_TRADE_FORMAT_PROMPT

    prompt = (
        "You are now in R1 behavioural analysis mode.\n\n"
        + scan_packet.to_prompt_block()
        + "\n\nApply the behavioural scanner logic above. "
        + "Read the scan in order: NOW → Opening Range → Phase → OI → News. "
        + "Produce scenario A/B thinking then give the trade.\n"
        + "IMPORTANT: Respond ONLY using the TRADE OUTPUT FORMAT. No extra prose."
    )

    suffix = R1_SCANNER_PROMPT + "\n\n" + R1_TRADE_FORMAT_PROMPT
    temp   = Session()
    text   = ""

    try:
        for event in agent.run(prompt, temp, system_suffix=suffix):
            if isinstance(event, FinalAnswerEvent):
                text = event.text
                break
            elif isinstance(event, ErrorEvent):
                log.error("R1 AI error: %s", event.message)
                break
    except Exception as exc:
        log.error("R1 agent run failed: %s", exc)
        raise

    return _parse_r1_response(text, scan_packet)


def _parse_r1_response(text: str, scan) -> R1Analysis:
    """Extract structured fields from R1 AI response."""
    import re

    def _get(label: str, default: str = "") -> str:
        patterns = [
            rf"{label}\s*:\s*(.+?)(?=\n[A-Z]|\n```|$)",
            rf"{label}\s*[:\-]\s*(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip().split("\n")[0].strip()
        return default

    phase   = _get("PHASE", scan.phase)
    r_a     = _get("READING A", "See full response")
    r_b     = _get("READING B", "See full response")
    now_s   = _get("NOW SAYS", "See full response")
    beh_p   = _get("BEHAVIOURAL PATTERN", "")
    strike  = _get("Strike/Expiry", "")
    entry   = _get("Entry", "")
    target  = _get("Target", "")
    sl      = _get("Stop Loss", "")
    rr      = _get("R:R", "")
    capital = _get("Capital", "")
    based   = _get("Based on", "")
    inv     = _get("Invalidated if", "")

    # Confidence from patterns
    if scan.patterns:
        highs = sum(1 for p in scan.patterns if p.confidence == "HIGH")
        confidence = "HIGH" if highs >= 2 else ("MEDIUM" if highs >= 1 else "LOW")
    else:
        confidence = "LOW"

    return R1Analysis(
        phase            = phase,
        reading_a        = r_a,
        reading_b        = r_b,
        now_says         = now_s,
        behavioural_pattern = beh_p,
        trade_strike     = strike,
        trade_expiry     = "",
        trade_entry      = entry,
        trade_target     = target,
        trade_sl         = sl,
        trade_rr         = rr,
        trade_capital    = capital,
        based_on         = based,
        invalidated_if   = inv,
        raw_response     = text,
        scan_at          = scan.scanned_at,
        confidence       = confidence,
    )
