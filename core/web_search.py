"""Web search module for Discord Chatbot using DuckDuckGo.

Provides keyword detection to decide when a query needs live internet results,
a search function that returns summarized results, metadata enrichment for
author/date extraction, and a Discord embed builder for source citations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import aiohttp
import discord
from bs4 import BeautifulSoup

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single web search result with optional metadata."""
    title: str
    snippet: str
    url: str
    author: str = ""
    pub_date: str = ""


# ---------------------------------------------------------------------------
# Keyword detection -- decide if a prompt needs a web search
# ---------------------------------------------------------------------------

# Patterns that strongly suggest the user wants current/factual info.
# Grouped by category for readability. Order doesn't matter -- any match
# triggers a search (unless an exclusion fires first).
_SEARCH_PATTERNS: list[re.Pattern] = [

    # ── Direct search intent ─────────────────────────────────────────
    re.compile(r"\b(search|google|look\s*up|find\s*(out|me)?|bing)\b", re.I),
    re.compile(r"\b(check\s+(if|whether|that|the|this))\b", re.I),

    # ── Current events / live data ───────────────────────────────────
    re.compile(r"\b(latest|current(ly)?|today|tonight|yesterday|this\s+(week|month|year))\b", re.I),
    re.compile(r"\b(recent(ly)?|right\s*now|breaking|news|headlines|trending)\b", re.I),
    re.compile(r"\b(as\s+of\s+\d|in\s+202\d|this\s+season)\b", re.I),

    # ── Factual / knowledge queries ──────────────────────────────────
    re.compile(r"\b(who\s+(is|was|are|were|plays?|owns?|won|leads?|runs?|founded|created|invented|discovered|killed|married))\b", re.I),
    re.compile(r"\b(what\s+(is|are|was|were|does|did|happened|caused))\b", re.I),
    re.compile(r"\b(when\s+(is|was|did|does|will))\b", re.I),
    re.compile(r"\b(where\s+(is|was|are|does|did|do|can))\b", re.I),
    re.compile(r"\b(how\s+(much|many|old|tall|far|long|fast|big|deep|heavy|large))\b", re.I),
    re.compile(r"\b(which\s+(country|state|city|team|company|player|president|party))\b", re.I),
    re.compile(r"\b(is\s+it\s+true\s+that|did\s+.+\s+really|has\s+.+\s+ever)\b", re.I),

    # ── People, places, organizations ────────────────────────────────
    re.compile(r"\b(president|prime\s*minister|ceo|governor|mayor|senator|congressman)\b", re.I),
    re.compile(r"\b(population|gdp|capital\s+of|located\s+in|headquartered)\b", re.I),

    # ── Sports ───────────────────────────────────────────────────────
    re.compile(r"\b(score|scoreboard|stats?|record|standings|rankings?|seedings?)\b", re.I),
    re.compile(r"\b(roster|lineup|playoffs|championship|finals|super\s*bowl|world\s*(cup|series))\b", re.I),
    re.compile(r"\b(traded|signed|drafted|waived|released|free\s*agent|contract|extension)\b", re.I),
    re.compile(r"\b(mvp|all[- ]?star|rookie|injured|injury|suspension|suspended)\b", re.I),
    re.compile(r"\b(nba|nfl|mlb|nhl|mls|ufc|wwe|fifa|premier\s*league|la\s*liga|serie\s*a|bundesliga)\b", re.I),
    re.compile(r"\b(lakers|celtics|warriors|knicks|bulls|nets|heat|mavs|mavericks|cowboys|eagles|chiefs|49ers|patriots)\b", re.I),

    # ── Scores, weather, prices, stocks ──────────────────────────────
    re.compile(r"\b(weather|forecast|temperature|rain|snow|humidity)\b", re.I),
    re.compile(r"\b(price|stock|market|crypto|bitcoin|ethereum|shares|ticker)\b", re.I),
    re.compile(r"\b(worth|cost|salary|net\s*worth|valuation|revenue|earnings)\b", re.I),

    # ── Tech / products / releases ───────────────────────────────────
    re.compile(r"\b(release\s*date|come\s*out|launch(ed|ing)?|announced|announcement|unveiled)\b", re.I),
    re.compile(r"\b(specs?|features?|benchmark|review|comparison)\b", re.I),
    re.compile(r"\b(available|discontinued|recalled|banned|approved|legalized)\b", re.I),
    re.compile(r"\b(update|patch|version|changelog)\b", re.I),

    # ── Comparisons that need data ───────────────────────────────────
    re.compile(r"\b(vs\.?|versus|compared?\s*to|better\s+than|worse\s+than|difference\s+between)\b", re.I),

    # ── Definitions / explanations ───────────────────────────────────
    re.compile(r"\b(define|definition|meaning\s+of|explain\s+what|what\s+does\s+.+\s+mean)\b", re.I),
    re.compile(r"\b(symptoms?\s+of|side\s+effects?\s+of|caused?\s+by)\b", re.I),

    # ── Politics / elections / law ───────────────────────────────────
    re.compile(r"\b(elected|election|vote[ds]?|ballot|poll(s|ing)?|inaugurated|impeach)\b", re.I),
    re.compile(r"\b(passed|signed\s+into\s+law|executive\s+order|supreme\s+court|ruling|verdict)\b", re.I),
    re.compile(r"\b(democrat|republican|gop|congress|senate|bill\s+\d|legislation)\b", re.I),

    # ── Entertainment / pop culture ──────────────────────────────────
    re.compile(r"\b(oscar|grammy|emmy|tony|golden\s*globe|award|nominated|winner)\b", re.I),
    re.compile(r"\b(box\s*office|ratings?|premiere|cancelled|renewed|streaming)\b", re.I),
    re.compile(r"\b(album|single|tour|concert|festival|sold\s+out)\b", re.I),

    # ── Life events / major changes (verify before denying) ──────────
    re.compile(r"\b(died|dead|death|passed\s+away|born|birthday|married|divorced|pregnant|engaged)\b", re.I),
    re.compile(r"\b(fired|hired|resigned|quit|retired|stepped\s+down|appointed|promoted)\b", re.I),
    re.compile(r"\b(arrested|charged|convicted|sentenced|acquitted|indicted|sued|lawsuit)\b", re.I),
    re.compile(r"\b(bankrupt|shutdown|closed|merged|acquired|ipo|went\s+public)\b", re.I),

    # ── User correcting the bot / asserting facts ────────────────────
    re.compile(r"\b(that'?s\s*(not|wrong|incorrect|false|old|outdated|inaccurate))\b", re.I),
    re.compile(r"\b(you'?re\s*(wrong|incorrect|lying|making\s+that\s+up))\b", re.I),
    re.compile(r"\b(stop\s*(gaslighting|lying|making\s+stuff\s+up|spreading))\b", re.I),
    re.compile(r"\b(no\s+(he|she|they|it)\s+(was|were|is|are|did|got|has|have)n'?t?)\b", re.I),
    re.compile(r"\b(actually|fact\s*check|source\??|proof|prove\s+it|look\s+it\s+up)\b", re.I),
    re.compile(r"\b(use\s*(the\s*)?(current|latest|new|updated|real)\s*(info|information|data|news)?)\b", re.I),

    # ── Numeric / data requests ──────────────────────────────────────
    re.compile(r"\b(percentage|ratio|average|median|total|statistic)\b", re.I),
    re.compile(r"\b(how\s+(to|do\s+(you|i))\s+(get|make|cook|fix|build|install|setup|configure))\b", re.I),
]

# Patterns that should NOT trigger a search (casual chat, opinions, greetings,
# banter, creative requests). Checked BEFORE search patterns.
_NO_SEARCH_PATTERNS: list[re.Pattern] = [
    # Greetings / farewells
    re.compile(r"^(hey|hi|hello|sup|yo|what'?s\s*up|gm|gn|good\s*(morning|night|evening)|bye|later|peace)\b", re.I),
    # Opinions / personal takes
    re.compile(r"\b(opinion|think|feel|believe|reckon|would\s+you|should\s+i|prefer|favorite|fav)\b", re.I),
    # Creative / roleplay / fun requests
    re.compile(r"\b(roast|rate|judge|describe|imagine|pretend|roleplay|write\s+(me\s+)?a\s+(poem|song|story|rap|joke))\b", re.I),
    re.compile(r"\b(tell\s+(me\s+)?a\s+(joke|story|fun\s+fact)|make\s+(me\s+)?laugh|be\s+funny)\b", re.I),
    # Self-referential questions about the bot
    re.compile(r"\b(are\s+you|what\s+(personality|model|mode|version|persona))\b", re.I),
    re.compile(r"\b(how\s+are\s+you|what\s+do\s+you\s+think|do\s+you\s+(like|love|hate|want|remember))\b", re.I),
    # Pure insults / reactions with no factual content
    re.compile(r"^(stfu|shut\s*up|dumbass|idiot|bruh|lol|lmao|bro|damn|wtf|smh|nah|cap|fr|ong|deadass|ratio|W|L)\s*$", re.I),
    # Hypotheticals
    re.compile(r"\b(what\s+if|hypothetically|in\s+theory|let'?s\s+say|would\s+you\s+rather)\b", re.I),
    # Coding / technical help (LLM can handle without search)
    re.compile(r"\b(debug|error\s+message|stack\s*trace|syntax|function|variable|code|program(ming)?|script)\b", re.I),
    re.compile(r"\b(python|javascript|java|c\+\+|rust|go|typescript|html|css|sql|react|node)\b", re.I),
    # Emotional / venting
    re.compile(r"\b(i'?m\s+(sad|happy|angry|bored|tired|lonely|depressed|stressed))\b", re.I),
    re.compile(r"\b(vent|rant|need\s+to\s+talk|just\s+saying|no\s+offense)\b", re.I),
]


# Patterns that indicate the user is giving a directive that refers back to
# the ongoing conversation ("look it up", "try again", "answer him", "search
# for that").  These are NOT searchable queries on their own — the real topic
# must be extracted from recent channel context.
_DIRECTIVE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(look\s*(it|that|this)\s*up|search\s*(for\s*)?(it|that|this))\s*[.!?]*$", re.I),
    re.compile(r"^(try\s*again|do\s*it\s*again|one\s*more\s*time)\s*[.!?]*$", re.I),
    re.compile(r"^(answer\s*(him|her|them|that|this|the\s*question))\s*[.!?]*$", re.I),
    re.compile(r"^(check\s*(it|that|this)|verify\s*(it|that|this))\s*[.!?]*$", re.I),
    re.compile(r"^(fact\s*check\s*(it|that|this|yourself)?)\s*[.!?]*$", re.I),
    re.compile(r"^(prove\s*it|source\??|citation\s*needed)\s*[.!?]*$", re.I),
]


def _is_directive(prompt: str) -> bool:
    """Return True if the prompt is a short directive referencing prior context."""
    cleaned = prompt.strip()
    if len(cleaned) > 60:
        return False
    for pattern in _DIRECTIVE_PATTERNS:
        if pattern.search(cleaned):
            return True
    return False


def needs_search(prompt: str, channel_context: str = "") -> bool:
    """Return True if the prompt looks like it needs live internet data.

    Decision logic:
    1. Directive prompts ("look it up", "try again", "answer him") always
       trigger search when channel context is available — query is built
       from context, not from the raw directive.
    2. Exclusion patterns fire early — casual chat, pure banter, insults,
       opinions, greetings, code help.  These never trigger search regardless
       of context.
    3. Too-short messages (< 8 chars) that survived exclusion still don't
       search — they're almost always reactions.
    4. If any search pattern matches, search is triggered.
    5. Default: no search.
    """
    cleaned = prompt.strip()

    # Directive prompts always trigger search (query is built from context)
    if _is_directive(cleaned) and channel_context:
        return True

    # Check exclusions BEFORE anything else — banter is banter, even if
    # the channel is discussing factual topics
    for pattern in _NO_SEARCH_PATTERNS:
        if pattern.search(cleaned):
            return False

    # Too short for search (and not a directive)
    if len(cleaned) < 8:
        return False

    # Check search triggers — but skip if prompt is a directive without
    # context (would search for "look it up" literally, returning junk)
    if not _is_directive(cleaned):
        for pattern in _SEARCH_PATTERNS:
            if pattern.search(cleaned):
                return True

    return False


def build_search_query(prompt: str, channel_context: str = "") -> str:
    """Build an effective search query from the prompt and conversation context.

    For directive prompts ("look it up", "try again"), extracts the actual
    topic from recent channel messages instead of searching the raw directive.

    For normal prompts, returns the prompt as-is (possibly trimmed).
    """
    cleaned = prompt.strip()

    # If the prompt is a directive or too vague to search on its own,
    # extract the topic from the conversation context
    if _is_directive(cleaned) or len(cleaned) < 15:
        topic = _extract_topic_from_context(channel_context)
        if topic:
            return topic

    # Normal prompt — use it directly, trimmed to a reasonable search length
    # Strip common filler and keep the core question
    query = cleaned
    # Remove @mentions
    query = re.sub(r"<@!?\d+>", "", query).strip()
    # Truncate overly long prompts for search
    if len(query) > 200:
        query = query[:200]
    return query


def _extract_topic_from_context(context: str) -> str:
    """Extract the main factual topic from recent channel messages.

    Scans the last few messages for named entities, factual claims, or
    questions, and builds a focused search query.
    """
    if not context:
        return ""

    # Get the last ~5 messages from the context
    lines = context.strip().split("\n")
    recent = lines[-8:] if len(lines) > 8 else lines

    # Strategy 1: Find the most recent line that contains a factual claim
    # (people names, team names, events, etc.)
    # Look for proper nouns and factual patterns — scan from most recent first
    best_topic = ""
    for line in reversed(recent):
        # Strip the speaker prefix ("Username: ...")
        text = re.sub(r"^[^:]+:\s*", "", line).strip()
        if not text or len(text) < 10:
            continue

        # Skip bot meta-responses ("That's not accurate", "I've checked...")
        if re.search(r"\b(that'?s not accurate|i'?ve checked|last i checked|no such)\b", text, re.I):
            continue

        # Prefer lines with factual claims (names + verbs)
        factual_score = 0
        # Has proper-noun-like capitalized words
        if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
            factual_score += 3  # multi-word proper noun (e.g. "Luka Doncic")
        # Has factual verbs
        if re.search(r"\b(traded|signed|plays?|won|lost|is|are|was|were|joined|left|moved)\b", text, re.I):
            factual_score += 2
        # Has team/org names
        if re.search(r"\b(lakers|celtics|warriors|mavericks|nets|cowboys|eagles|chiefs)\b", text, re.I):
            factual_score += 2
        # Contains a question
        if text.endswith("?"):
            factual_score += 1

        if factual_score >= 2 and len(text) > len(best_topic):
            best_topic = text

    if best_topic:
        # Trim to a reasonable search query — take the key claim
        if len(best_topic) > 150:
            best_topic = best_topic[:150]
        return best_topic

    # Strategy 2: Fallback — just use the last substantive message as the query
    for line in reversed(recent):
        text = re.sub(r"^[^:]+:\s*", "", line).strip()
        if len(text) > 15:
            return text[:150]

    return ""


# ---------------------------------------------------------------------------
# DuckDuckGo HTML search
# ---------------------------------------------------------------------------

_DDG_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


async def search(
    query: str,
    max_results: int = 3,
    session: Optional[aiohttp.ClientSession] = None,
) -> list[dict[str, str]]:
    """Search DuckDuckGo and return a list of {title, snippet, url} dicts.

    If *session* is provided it is reused (caller owns lifecycle).
    Otherwise a throwaway session is created.
    """
    results: list[dict[str, str]] = []
    try:
        timeout = aiohttp.ClientTimeout(total=config.SEARCH_TIMEOUT)
        owns_session = session is None
        sess = session or aiohttp.ClientSession(timeout=timeout, headers=_HEADERS)
        try:
            async with sess.post(
                _DDG_URL,
                data={"q": query, "b": ""},
                headers=_HEADERS,
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    logger.warning("DuckDuckGo returned %d for query: %s", resp.status, query)
                    return results
                html = await resp.text()
        finally:
            if owns_session:
                await sess.close()

        soup = BeautifulSoup(html, "html.parser")

        for result_div in soup.select(".result__body")[:max_results]:
            title_tag = result_div.select_one(".result__a")
            snippet_tag = result_div.select_one(".result__snippet")
            url_tag = result_div.select_one(".result__url")

            title = title_tag.get_text(strip=True) if title_tag else ""
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
            url = url_tag.get_text(strip=True) if url_tag else ""

            if title or snippet:
                results.append({"title": title, "snippet": snippet, "url": url})

    except Exception as exc:
        logger.warning("Web search failed for '%s': %s", query, exc)

    return results


def filter_results(
    results: list[dict[str, str]],
    min_results: int = 2,
    max_results: int = 5,
) -> list[dict[str, str]]:
    """Score and filter search results by quality.

    Keeps between *min_results* and *max_results*.  Results with no useful
    content are dropped, but we always try to return at least *min_results*
    (even low-quality ones) so the user sees something.
    """
    if not results:
        return results

    def _score(r: dict[str, str]) -> int:
        """Higher = better quality."""
        s = 0
        if r.get("title"):
            s += 2
        if r.get("snippet") and len(r["snippet"]) > 30:
            s += 3  # meaningful snippet
        elif r.get("snippet"):
            s += 1  # short/weak snippet
        if r.get("url"):
            s += 1
        return s

    scored = [(r, _score(r)) for r in results[:max_results]]
    # Sort by score descending (stable — preserves DDG ranking for ties)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep all results that have at least a title + some snippet
    good = [(r, s) for r, s in scored if s >= 3]

    if len(good) >= min_results:
        return [r for r, _ in good[:max_results]]

    # Not enough good results — pad with lower-quality ones to hit min
    remaining = [r for r, s in scored if s < 3]
    combined = [r for r, _ in good] + remaining
    return combined[:max(min_results, len(good))]


# ---------------------------------------------------------------------------
# Metadata extraction -- fetch author & publication date from result pages
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%m/%d/%Y",
]


def _parse_date(raw: str) -> str:
    """Try to parse a date string into 'Mon DD, YYYY' format."""
    if not raw:
        return ""
    cleaned = raw.strip()[:30]
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.strftime("%b %d, %Y")
        except ValueError:
            continue
    # Try ISO-ish partial parse (just grab the date portion)
    m = re.match(r"(\d{4}-\d{2}-\d{2})", cleaned)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d")
            return dt.strftime("%b %d, %Y")
        except ValueError:
            pass
    return ""


def _extract_meta(soup: BeautifulSoup) -> dict[str, str]:
    """Extract author and publication date from a page's meta tags and JSON-LD."""
    author = ""
    pub_date = ""

    # --- Author ---
    for attr, key in [
        ("name", "author"),
        ("property", "article:author"),
        ("property", "og:article:author"),
        ("name", "dc.creator"),
    ]:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content", "").strip():
            author = tag["content"].strip()
            break

    # --- Publication date ---
    for attr, key in [
        ("property", "article:published_time"),
        ("name", "date"),
        ("property", "og:article:published_time"),
        ("name", "dc.date"),
        ("name", "publishdate"),
        ("property", "datePublished"),
    ]:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content", "").strip():
            pub_date = _parse_date(tag["content"])
            if pub_date:
                break

    # Fallback: <time datetime="...">
    if not pub_date:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            pub_date = _parse_date(time_tag["datetime"])

    # Fallback: JSON-LD
    if not author or not pub_date:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                # Could be a list
                if isinstance(data, list):
                    data = data[0] if data else {}
                if not author:
                    a = data.get("author")
                    if isinstance(a, dict):
                        author = a.get("name", "")
                    elif isinstance(a, list) and a:
                        author = a[0].get("name", "") if isinstance(a[0], dict) else str(a[0])
                    elif isinstance(a, str):
                        author = a
                if not pub_date:
                    dp = data.get("datePublished", "")
                    if dp:
                        pub_date = _parse_date(dp)
                if author and pub_date:
                    break
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

    # Ensure values are always plain strings (BeautifulSoup or JSON-LD
    # can sometimes hand back lists, Tags, or other non-str types).
    if not isinstance(author, str):
        author = str(author) if author else ""
    if not isinstance(pub_date, str):
        pub_date = str(pub_date) if pub_date else ""
    return {"author": author, "pub_date": pub_date}


async def _fetch_metadata(
    url: str,
    session: aiohttp.ClientSession,
) -> dict[str, str]:
    """Fetch a page and extract author + publication date from meta tags."""
    if not url:
        return {"author": "", "pub_date": ""}

    # Ensure URL has a scheme
    fetch_url = url if url.startswith("http") else f"https://{url}"

    try:
        timeout = aiohttp.ClientTimeout(total=config.SEARCH_META_TIMEOUT)
        async with session.get(
            fetch_url,
            headers=_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return {"author": "", "pub_date": ""}
            # Only parse HTML pages, skip PDFs/images/etc.
            ct = resp.headers.get("Content-Type", "")
            if "html" not in ct.lower():
                return {"author": "", "pub_date": ""}
            # Limit read to first 100KB to avoid huge pages
            raw = await resp.content.read(102400)
            html = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("Metadata fetch failed for %s: %s", url, exc)
        return {"author": "", "pub_date": ""}

    soup = BeautifulSoup(html, "html.parser")
    return _extract_meta(soup)


async def enrich_results(
    raw_results: list[dict[str, str]],
    session: aiohttp.ClientSession,
) -> list[SearchResult]:
    """Fetch metadata for each search result concurrently and return SearchResult objects."""
    if not raw_results:
        return []

    tasks = [_fetch_metadata(r.get("url", ""), session) for r in raw_results]
    meta_list = await asyncio.gather(*tasks, return_exceptions=True)

    enriched: list[SearchResult] = []
    for raw, meta in zip(raw_results, meta_list):
        if isinstance(meta, Exception):
            meta = {"author": "", "pub_date": ""}
        enriched.append(SearchResult(
            title=raw.get("title", ""),
            snippet=raw.get("snippet", ""),
            url=raw.get("url", ""),
            author=meta.get("author", ""),
            pub_date=meta.get("pub_date", ""),
        ))

    return enriched


# ---------------------------------------------------------------------------
# Formatting -- prompt injection + Discord embed
# ---------------------------------------------------------------------------

def format_results_raw(results: list[dict[str, str]]) -> str:
    """Format raw search dicts into a text block for LLM prompt injection.

    This is the fast path used before metadata enrichment completes.
    """
    if not results:
        return ""

    lines = ["[Web search results]"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}")
        snippet = r.get("snippet", "")
        if snippet:
            lines.append(f"   {snippet}")
        url = r.get("url", "")
        if url:
            lines.append(f"   Source: {url}")
    lines.append("[End of search results]")
    return "\n".join(lines)


def format_results(results: list[SearchResult]) -> str:
    """Format enriched search results into a text block for injection into the LLM prompt."""
    if not results:
        return ""

    lines = ["[Web search results]"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        meta_parts = []
        if r.author:
            meta_parts.append(f"By {r.author}")
        if r.pub_date:
            meta_parts.append(r.pub_date)
        if meta_parts:
            lines.append(f"   {' | '.join(meta_parts)}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
        if r.url:
            lines.append(f"   Source: {r.url}")
    lines.append("[End of search results]")
    return "\n".join(lines)


def build_sources_embed(results: list[SearchResult]) -> discord.Embed:
    """Build a compact Discord embed displaying cited sources with clickable links.

    Layout per source:
      [Title](url)
      domain · author · date
    """
    embed = discord.Embed(color=config.COLOR_SOURCES)

    desc_lines: list[str] = []
    for r in results:
        url = r.url if r.url.startswith("http") else f"https://{r.url}" if r.url else ""
        # Extract clean domain for display
        domain = ""
        if url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(url).netloc.removeprefix("www.")
            except Exception:
                pass

        # Title as clickable link
        display_title = r.title or domain or "Link"
        if url:
            line = f"[{display_title}]({url})"
        else:
            line = display_title

        # Meta line: domain · author · date
        meta_parts: list[str] = []
        if domain:
            meta_parts.append(domain)
        if r.author:
            meta_parts.append(str(r.author) if not isinstance(r.author, str) else r.author)
        if r.pub_date:
            meta_parts.append(str(r.pub_date) if not isinstance(r.pub_date, str) else r.pub_date)
        if meta_parts:
            line += f"\n{' · '.join(meta_parts)}"

        desc_lines.append(line)

    embed.description = "\n\n".join(desc_lines)
    embed.set_footer(text=f"DuckDuckGo · {len(results)} source{'s' if len(results) != 1 else ''}")
    return embed
