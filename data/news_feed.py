"""
data/news_feed.py
-----------------
News feed for NIFTY trading context.

Sources:
  1. gnews (Google News RSS — no API key)  → market, geopolitical, economy news
  2. NSE announcements                     → corporate actions, circuit filters
  3. RBI / SEBI headlines                  → policy-sensitive events

Install: pip install gnews

News categories:
  fetch_market_news()       — NIFTY, BSE, markets, economy
  fetch_geopolitical_news() — sanctions, war, oil, global tensions
  fetch_rbi_sebi_news()     — RBI policy, SEBI regulations
  fetch_fii_news()          — FII flows, foreign investment
  fetch_all_news()          — all categories, deduplicated, scored for impact
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)


# ── News article dataclass ─────────────────────────────────────────────────────

@dataclass
class NewsArticle:
    title:       str
    source:      str
    published:   str
    url:         str
    category:    str
    sentiment:   str   # "BULLISH" | "BEARISH" | "NEUTRAL"
    impact:      str   # "HIGH" | "MEDIUM" | "LOW"
    summary:     str = ""

    def one_line(self) -> str:
        icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(self.sentiment, "·")
        imp  = {"HIGH": "🔥", "MEDIUM": "⚡", "LOW": "·"}.get(self.impact, "·")
        return f"{icon}{imp} [{self.source}] {self.title}"


# ── Keyword sets ───────────────────────────────────────────────────────────────

_MARKET_KW = (
    '"NIFTY" OR "NSE" OR "BSE" OR "Sensex" OR "Indian stock market" OR '
    '"FII" OR "DII" OR "India GDP" OR "RBI" OR "SEBI" OR "budget India"'
)

_GEO_KW = (
    '"geopolitics" OR "sanctions" OR "oil price" OR "crude oil" OR '
    '"US Fed" OR "Federal Reserve" OR "China economy" OR '
    '"Russia Ukraine" OR "Middle East" OR "OPEC" OR "dollar index"'
)

_RBI_SEBI_KW = (
    '"RBI policy" OR "repo rate" OR "monetary policy" OR '
    '"SEBI" OR "circuit breaker" OR "FPI limit" OR "derivatives regulation"'
)

_FII_KW = (
    '"FII buying" OR "FII selling" OR "foreign portfolio" OR '
    '"FPI flows" OR "DII buying" OR "institutional investors India"'
)

# ── Sentiment keywords ────────────────────────────────────────────────────────

_BULLISH_WORDS = {
    "rally", "surge", "jump", "gain", "rise", "bullish", "positive",
    "recovery", "rebound", "upside", "growth", "rate cut", "stimulus",
    "buying", "inflow", "invest", "strong", "record high", "breakout",
}
_BEARISH_WORDS = {
    "fall", "crash", "drop", "decline", "sell", "bearish", "negative",
    "recession", "inflation", "outflow", "withdraw", "weak", "concern",
    "war", "sanction", "rate hike", "selloff", "correction", "breakdown",
}
_HIGH_IMPACT = {
    "RBI", "repo rate", "Fed", "Federal Reserve", "GDP", "inflation",
    "war", "sanction", "circuit breaker", "SEBI ban", "budget",
    "crude oil", "dollar index", "FII", "FPI",
}


def _score_article(title: str, category: str) -> tuple[str, str]:
    """Return (sentiment, impact) for a headline."""
    low = title.lower()
    bull = sum(1 for w in _BULLISH_WORDS if w in low)
    bear = sum(1 for w in _BEARISH_WORDS if w in low)

    if   bull > bear: sentiment = "BULLISH"
    elif bear > bull: sentiment = "BEARISH"
    else:             sentiment = "NEUTRAL"

    impact = "HIGH" if any(w.lower() in low for w in _HIGH_IMPACT) else "MEDIUM"
    if category == "geopolitical" and sentiment == "BEARISH":
        impact = "HIGH"

    return sentiment, impact


# ── Core fetch function ───────────────────────────────────────────────────────

def _fetch(keywords: str, category: str, days: int = 1,
           max_results: int = 8) -> list[NewsArticle]:
    """Generic gnews fetch."""
    try:
        from gnews import GNews
        gn = GNews(language="en", country="IN", max_results=max_results)
        gn.start_date = datetime.now() - timedelta(days=days)
        gn.end_date   = datetime.now()
        articles_raw  = gn.get_news(keywords)
    except ImportError:
        log.warning("gnews not installed. Run: pip install gnews")
        return []
    except Exception as exc:
        log.warning("gnews fetch failed [%s]: %s", category, exc)
        return []

    out = []
    seen_titles: set[str] = set()

    for a in articles_raw or []:
        title = a.get("title", "").strip()
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)

        sentiment, impact = _score_article(title, category)
        out.append(NewsArticle(
            title     = title,
            source    = a.get("publisher", {}).get("title", "Unknown"),
            published = a.get("published date", a.get("published_date", "")),
            url       = a.get("url", ""),
            category  = category,
            sentiment = sentiment,
            impact    = impact,
        ))

    return out


# ── Public functions ──────────────────────────────────────────────────────────

def fetch_market_news(days: int = 1) -> list[NewsArticle]:
    return _fetch(_MARKET_KW, "market", days=days, max_results=8)


def fetch_geopolitical_news(days: int = 2) -> list[NewsArticle]:
    return _fetch(_GEO_KW, "geopolitical", days=days, max_results=8)


def fetch_rbi_sebi_news(days: int = 3) -> list[NewsArticle]:
    return _fetch(_RBI_SEBI_KW, "rbi_sebi", days=days, max_results=6)


def fetch_fii_news(days: int = 1) -> list[NewsArticle]:
    return _fetch(_FII_KW, "fii_flow", days=days, max_results=6)


def fetch_all_news(days: int = 2) -> list[NewsArticle]:
    """All categories, deduplicated, HIGH impact first."""
    all_articles: list[NewsArticle] = []
    seen: set[str] = set()

    for fn in [fetch_market_news, fetch_fii_news, fetch_rbi_sebi_news, fetch_geopolitical_news]:
        for a in fn(days=days):
            if a.title not in seen:
                seen.add(a.title)
                all_articles.append(a)

    # Sort: HIGH first, then BULLISH/BEARISH before NEUTRAL
    def _sort_key(a: NewsArticle) -> tuple:
        imp = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(a.impact, 2)
        sen = {"BEARISH": 0, "BULLISH": 0, "NEUTRAL": 1}.get(a.sentiment, 1)
        return (imp, sen)

    return sorted(all_articles, key=_sort_key)


def news_brief(days: int = 1) -> str:
    """
    Compact news summary for injection into AI trading prompt.
    Returns ~300 chars covering most market-moving headlines.
    """
    articles = fetch_all_news(days=days)
    if not articles:
        return "No market news available."

    high   = [a for a in articles if a.impact == "HIGH"][:4]
    medium = [a for a in articles if a.impact == "MEDIUM"][:3]

    lines = ["=== MARKET NEWS ==="]
    for a in high + medium:
        lines.append(a.one_line())

    # Aggregate sentiment
    bulls = sum(1 for a in articles if a.sentiment == "BULLISH")
    bears = sum(1 for a in articles if a.sentiment == "BEARISH")
    total = len(articles)
    if total:
        if bulls > bears * 1.5:  overall = "NEWS SENTIMENT: 🟢 BULLISH"
        elif bears > bulls * 1.5: overall = "NEWS SENTIMENT: 🔴 BEARISH"
        else:                      overall = "NEWS SENTIMENT: ⚪ MIXED"
        lines.append(overall + f" ({bulls} bullish / {bears} bearish of {total} articles)")

    lines.append("=== END NEWS ===")
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(news_brief())
