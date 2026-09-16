"""
trading/rules.py
----------------
Core-managed market rules — the experience file.

In SIMULATION mode the core controls the data feed, so after every session it
knows exactly which predictions were right and wrong. It distils those
outcomes into compact rules (what worked, what failed, in which regime) and
writes them to rule.md. Every later run — simulation AND live — injects the
current rules into the AI system prompt so past mistakes are not repeated.

File: ~/.jai_sadguru/rule.md
Structure: markdown with one lesson per bullet under fixed sections so
load_rules_block() can parse it deterministically.
"""

from __future__ import annotations

import logging
import re
import threading
from datetime import datetime
from pathlib import Path

from config import RULES_FILE

log = logging.getLogger(__name__)

_MAX_LESSONS_PER_SECTION = 12
_MAX_LESSON_LEN          = 220
_MAX_CONSOLIDATE_LESSONS = 24   # when a section exceeds this, merge with AI

_lock = threading.Lock()

_TEMPLATE = """# NIFTY Trading Rules — Learned Experience
<!-- Managed by the simulation core. Injected into every AI decision prompt. -->
<!-- Format is machine-parsed: keep one lesson per bullet, no sub-bullets. -->

## GLOBAL
<!-- Rules that always apply -->

## REGIME SIDEWAYS
<!-- Rules for range-bound / sideways markets -->

## REGIME DIRECTIONAL
<!-- Rules for trending markets -->

## REGIME VOLATILE
<!-- Rules for high-VIX / fast markets -->

## KRONOS
<!-- Lessons about when the Kronos candle forecast agrees/disagrees with price -->

## SOURCES
<!-- Meta notes: which session/run a batch of lessons came from -->
"""


# ── Load / save ───────────────────────────────────────────────────────────────

def _ensure_file() -> Path:
    p = Path(RULES_FILE)
    if not p.exists():
        p.write_text(_TEMPLATE, encoding="utf-8")
    return p


def load_rules_block(max_chars: int = 2500) -> str:
    """
    Return the current rules as a prompt block for the AI system prompt.
    Lessons from regime sections carry a [SECTION] prefix so the AI knows
    when each rule applies; SOURCES meta-notes are excluded.
    Empty string when no lessons yet.
    """
    try:
        text = _ensure_file().read_text(encoding="utf-8")
    except Exception as exc:
        log.warning("Cannot read rules file: %s", exc)
        return ""

    sections = _parse_sections(text)
    lines: list[str] = []
    for name, lessons in sections.items():
        if name == "SOURCES":
            continue
        prefix = ""
        if name.startswith("REGIME "):
            prefix = f"[{name.split(' ', 1)[1]}] "
        elif name == "KRONOS":
            prefix = "[KRONOS] "
        lines.extend(f"{prefix}{lesson}" for lesson in lessons)

    body = "\n".join(lines).strip()
    if not body:
        return ""
    if len(body) > max_chars:
        body = body[:max_chars] + "\n… (truncated)"
    return "═══ LEARNED RULES (from past simulation sessions) ═══\n" + body


# ── Parsing ───────────────────────────────────────────────────────────────────

def _parse_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("## "):
            current = s[3:].strip().upper()
            sections.setdefault(current, [])
        elif current and s and not s.startswith("<!--") and not s.startswith("#"):
            sections[current].append(s.lstrip("-• "))   # store WITHOUT bullet
    return sections


def _render(sections: dict[str, list[str]]) -> str:
    out = ["# NIFTY Trading Rules — Learned Experience",
           "<!-- Managed by the simulation core. Injected into every AI decision prompt. -->",
           "<!-- Format is machine-parsed: keep one lesson per bullet, no sub-bullets. -->",
           ""]
    for name, lessons in sections.items():
        out.append(f"## {name}")
        seen: set[str] = set()
        for lesson in lessons:
            key = lesson.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(f"- {lesson}")
        out.append("")
    return "\n".join(out)


# ── Record lessons ────────────────────────────────────────────────────────────

def record_lessons(lessons: list[str], section: str = "GLOBAL",
                   source_note: str = "") -> None:
    """
    Append lessons (dedup, cap length) to a section of rule.md.
    Safe to call from any thread; never raises.
    """
    if not lessons:
        return
    section = section.strip().upper() or "GLOBAL"
    with _lock:
        try:
            text = _ensure_file().read_text(encoding="utf-8")
            sections = _parse_sections(text)
            bucket = sections.setdefault(section, [])
            added = 0
            for lesson in lessons:
                lesson = " ".join(str(lesson).split())[:_MAX_LESSON_LEN]
                if not lesson or any(lesson.lower()[:60] == e.lower()[:60] for e in bucket):
                    continue
                bucket.append(lesson)
                added += 1
            if source_note:
                sections.setdefault("SOURCES", []).append(
                    f"{datetime.now().strftime('%d-%b %H:%M')} — {source_note} (+{added} lessons)"
                )
            # Cap section sizes, keeping the newest
            for name in list(sections):
                if len(sections[name]) > _MAX_LESSONS_PER_SECTION:
                    sections[name] = sections[name][-_MAX_LESSONS_PER_SECTION:]
            _ensure_file().write_text(_render(sections), encoding="utf-8")
        except Exception as exc:
            log.warning("record_lessons failed: %s", exc)


def record_session_outcome(
    regime: str,
    kronos_agree: int,
    kronos_disagree: int,
    total_pnl: float,
    trade_count: int,
    win_rate: float,
    ai_summary: str = "",
) -> None:
    """
    Called at the end of a simulation session. Records headline stats as
    lessons under the right sections. `ai_summary` is an optional 2-3 sentence
    AI-written reflection (already generated by the session-end LLM call).
    """
    lessons: list[str] = []
    section = "GLOBAL"
    regime_u = (regime or "GLOBAL").upper()
    if "SIDEWAYS" in regime_u:
        section = "REGIME SIDEWAYS"
    elif "DIRECTIONAL" in regime_u or "TREND" in regime_u:
        section = "REGIME DIRECTIONAL"
    elif "VOLATILE" in regime_u:
        section = "REGIME VOLATILE"

    outcome = "profitable" if total_pnl > 0 else "loss-making"
    lessons.append(
        f"[{datetime.now().strftime('%d-%b-%Y')}] {outcome} session: {trade_count} trades, "
        f"win-rate {win_rate:.0f}%, P&L {total_pnl:+,.0f} — regime {regime_u}"
    )
    if ai_summary:
        for sentence in re.split(r"(?<=[.!?]) +", ai_summary.strip())[:3]:
            sentence = " ".join(sentence.split())
            if sentence:
                lessons.append(sentence[:_MAX_LESSON_LEN])

    record_lessons(lessons, section=section,
                   source_note=f"sim session regime={regime_u}")

    if kronos_disagree + kronos_agree > 0:
        kronos_note = (
            f"[{datetime.now().strftime('%d-%b-%Y')}] Kronos agreed with price action "
            f"{kronos_agree}/{kronos_agree + kronos_disagree} times this session"
        )
        record_lessons([kronos_note], section="KRONOS",
                       source_note="sim session kronos tracking")


# ── Consolidation (AI-aided, optional) ────────────────────────────────────────

def consolidate_rules(agent=None) -> str:
    """
    Merge near-duplicate lessons with AI help. Called rarely (when a section
    exceeds _MAX_CONSOLIDATE_LESSONS). Returns a status string.
    """
    text = _ensure_file().read_text(encoding="utf-8")
    sections = _parse_sections(text)

    oversized = [n for n, ls in sections.items() if len(ls) > _MAX_CONSOLIDATE_LESSONS]
    if not oversized:
        return "Rules file healthy — no consolidation needed."

    if agent is None:
        return f"Sections {oversized} need consolidation — run with an agent."

    merged_any = False
    for name in oversized:
        lessons = sections[name]
        prompt = (
            "You maintain a trading rules file. Merge the lessons below into at most "
            f"{_MAX_LESSONS_PER_SECTION} non-redundant, specific rules. "
            "Keep concrete numbers/levels when present. Reply with ONE lesson per line, "
            "no numbering, no extra text.\n\n" + "\n".join(lessons)
        )
        from agent import Session, FinalAnswerEvent, ErrorEvent
        temp = Session()
        reply_lines: list[str] = []
        try:
            for ev in agent.run(prompt, temp, system_suffix="Respond only with the merged rule lines."):
                if isinstance(ev, FinalAnswerEvent):
                    reply_lines = [ln.strip("-• ").strip() for ln in ev.text.splitlines()
                                   if ln.strip() and not ln.strip().startswith("#")]
                    break
                elif isinstance(ev, ErrorEvent):
                    return f"Consolidation failed: {ev.message}"
        except Exception as exc:
            return f"Consolidation failed: {exc}"
        if reply_lines:
            sections[name] = [ln for ln in reply_lines if ln][:_MAX_LESSONS_PER_SECTION]
            merged_any = True

    if merged_any:
        _ensure_file().write_text(_render(sections), encoding="utf-8")
        return "Rules consolidated."
    return "No changes made."
