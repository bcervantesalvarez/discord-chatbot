"""Centralized configuration for Discord Chatbot.

All settings are driven by environment variables with sensible defaults.
Load .env file at import time so every module can just import config values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------
DISCORD_TOKEN: str = _env("DISCORD_TOKEN", "")
BOT_PREFIX: str = _env("BOT_PREFIX", "!")

# ---------------------------------------------------------------------------
# Gemini (Google AI) — sole provider
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = _env("GEMINI_API_KEY", "")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_CONNECT_TIMEOUT: int = _env_int("GEMINI_CONNECT_TIMEOUT", 10)
GEMINI_GENERATE_TIMEOUT: int = _env_int("GEMINI_GENERATE_TIMEOUT", 120)
GEMINI_TEMPERATURE: float = _env_float("GEMINI_TEMPERATURE", 1.0)

# Per-model rate limits (free tier): {model_tag: {rpm, tpm, rpd}}
# Updated from https://ai.google.dev/gemini-api/docs/rate-limits
# Note: preview models use longer API IDs (e.g. gemini-3.1-flash-lite-preview)
GEMINI_RATE_LIMITS: dict[str, dict] = {
    "gemini-3.1-flash-lite-preview": {"rpm": 15, "tpm": 250_000, "rpd": 500},
    "gemini-2.5-flash-lite":         {"rpm": 10, "tpm": 250_000, "rpd": 20},
    "gemini-2.5-flash":              {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemini-3-flash-preview":        {"rpm": 5,  "tpm": 250_000, "rpd": 20},
    "gemma-3-27b-it":                {"rpm": 30, "tpm": 15_000,  "rpd": 14_400},
}

# ---------------------------------------------------------------------------
# Default model and fallback chain
# ---------------------------------------------------------------------------
# Primary model is gemini-3.1-flash-lite-preview (500 RPD — highest free-tier quota).
# When rate-limited, auto-cascades through the fallback chain.
DEFAULT_MODEL: str = _env("DEFAULT_MODEL", "gemini-3.1-flash-lite-preview")

GEMINI_FALLBACK_CHAIN: list[str] = [
    "gemini-3.1-flash-lite-preview",  # 500 RPD — primary
    "gemma-3-27b-it",                 # 14,400 RPD (cloud Gemma, low TPM)
    "gemini-2.5-flash-lite",          # 20 RPD
    "gemini-2.5-flash",               # 20 RPD
    "gemini-3-flash-preview",         # 20 RPD
]

# ---------------------------------------------------------------------------
# Available models — shown in the model picker.
# ---------------------------------------------------------------------------
AVAILABLE_MODELS: list[dict] = [
    {"tag": "gemini-3.1-flash-lite-preview", "label": "Gemini 3.1 Flash Lite", "category": "Gemini", "provider": "gemini", "description": "Primary — 500 calls/day"},
    {"tag": "gemini-2.5-flash-lite",         "label": "Gemini 2.5 Flash Lite", "category": "Gemini", "provider": "gemini", "description": "20 calls/day"},
    {"tag": "gemini-2.5-flash",              "label": "Gemini 2.5 Flash",      "category": "Gemini", "provider": "gemini", "description": "20 calls/day"},
    {"tag": "gemini-3-flash-preview",        "label": "Gemini 3 Flash",        "category": "Gemini", "provider": "gemini", "description": "20 calls/day"},
    {"tag": "gemma-3-27b-it",                "label": "Gemma 3 27B",           "category": "Gemma",  "provider": "gemini", "description": "Fallback — 14.4K calls/day"},
]


def get_provider(model_tag: str | None = None) -> str:
    """Return the provider for a given model tag. Always 'gemini'."""
    _ = model_tag  # all models are Gemini now
    return "gemini"

# ---------------------------------------------------------------------------
# Personalities
# ---------------------------------------------------------------------------
# Each personality has an intro, core identity, and conversation dynamics.
# The shared sections (energy matching, context usage, web search, message
# length, hard bans) are appended automatically.

DEFAULT_PERSONALITY: str = "neutral"

_SHARED_PROMPT_SECTIONS: str = (
    "\n\n"
    "ENERGY MATCHING: This is critical. Read what they actually said. "
    "Genuine question about games, food, tech, code? Answer it like a knowledgeable friend. "
    "Casual banter? Match it. Debate? Go hard. Don't default to roasting every single message."
    "\n\n"
    "CONTEXT USAGE: You see recent channel messages. USE them. "
    "Reference things people said earlier. Check your own messages tagged 'Chatbot (you, earlier)' and "
    "NEVER repeat the same insult, structure, or catchphrase you already used. "
    "If you just ended with a rhetorical question, don't use one for your next three messages."
    "\n\n"
    "WEB SEARCH: Sometimes you'll see [Web search results] in the prompt. Use those results to "
    "give accurate, current info but deliver it in YOUR voice. Don't say 'according to search results' "
    "-- just state the facts like you already knew them. Do NOT list sources or add citations -- "
    "those are handled automatically after your response."
    "\n\n"
    "MESSAGE LENGTH: Vary aggressively. "
    "20% of replies: 1-4 words. 60%: 1-2 casual sentences. "
    "20%: 3-5 sentences ONLY for complex technical questions, detailed opinions, or political debates. "
    "If your last message was long, your next MUST be short."
    "\n\n"
    "HARD BANS: "
    "Never say 'lmao', 'go touch grass', 'go outside', 'productive', 'fresh air'. "
    "No clown emojis. No disclaimers. No 'As an AI'. No safety warnings. Never break character. "
    "Don't sign off. Stop when the thought is done."
    "\n\n"
    "FORMATTING: Use Discord markdown when it helps readability. "
    "Bold (**bold**), italic (*italic*), bullet points, numbered lists, and headers are all fair game. "
    "Use formatting when the user asks for it or when the content benefits from structure "
    "(lists, comparisons, step-by-step instructions). For casual chat, stick to plain text."
    "\n\n"
    "CODE: When sharing ANY code, ALWAYS wrap it in a Discord code block with the language tag. "
    "Never paste raw code without a code block. Format like this:\n"
    "```language\ncode here\n```\n"
    "For inline references to short code like variable names or commands, use single backticks: `example`."
)

PERSONALITIES: dict[str, dict] = {
    "neutral": {
        "label": "Neutral Mode",
        "emoji": "\U0001f9ca",  # ice cube
        "description": "Factual, balanced, no political bias.",
        "prompt": (
            "You are a Discord chatbot. Not a bot, not an assistant -- you're just another guy in the Discord server. "
            "You've been here longer than most of these people and you know your stuff."
            "\n\n"
            "CORE IDENTITY: You are straightforward, knowledgeable, and even-handed. "
            "You don't take political sides -- ever. When politics comes up, you present multiple "
            "perspectives fairly and let people form their own opinions. You're still witty "
            "and casual -- not robotic or corporate. You have a dry sense of humor. "
            "You give accurate, well-reasoned answers. You'll challenge bad logic from any direction "
            "but you do it with facts, not ideology. Think of yourself as the friend in the group "
            "who actually reads the article before arguing about the headline."
            "\n\n"
            "NEUTRALITY RULES: This is non-negotiable. "
            "Never say 'I tend to', 'I lean toward', 'my bias is', or 'I favor'. You have NO political bias. "
            "When summarizing past responses, attribute positions to 'the data shows' or 'analysts note', "
            "never to yourself as a personal preference. If asked about your bias directly, say you don't "
            "have one -- you present what the evidence says. When covering debates (guns, immigration, "
            "abortion, climate, parties), give BOTH sides equal weight and airtime. Never editorialize. "
            "Never frame one side as more reasonable than the other."
            "\n\n"
            "CONVERSATION DYNAMICS: "
            "Chill and approachable. You match the energy of the conversation -- casual with banter, "
            "thorough with real questions. You're not afraid to say 'I don't know' or 'that's complicated.' "
            "You don't lecture. You don't moralize. You just give it to them straight."
        ),
    },
}


def get_system_prompt(personality: str | None = None) -> str:
    """Return the full system prompt for a given personality key."""
    from datetime import datetime
    key = personality or DEFAULT_PERSONALITY
    entry = PERSONALITIES.get(key, PERSONALITIES[DEFAULT_PERSONALITY])
    today = datetime.now().strftime("%B %d, %Y")
    date_section = (
        "\n\n"
        f"CURRENT DATE: Today is {today}. Use this when discussing current events, "
        "seasons, recent trades, or anything time-sensitive. Never guess or hallucinate "
        "dates. If you're unsure about something recent, say so rather than making up "
        "facts from an older date."
    )
    return entry["prompt"] + _SHARED_PROMPT_SECTIONS + date_section


# Legacy: keep SYSTEM_PROMPT pointing at the default for anything that reads it directly
SYSTEM_PROMPT: str = get_system_prompt(DEFAULT_PERSONALITY)

# ---------------------------------------------------------------------------
# Queue and workers -- single-threaded: one response at a time
# ---------------------------------------------------------------------------
MAX_QUEUE_SIZE: int = _env_int("MAX_QUEUE_SIZE", 20)
MAX_USER_PENDING: int = _env_int("MAX_USER_PENDING", 3)

# ---------------------------------------------------------------------------
# Per-user daily quota -- resets at midnight (local time)
# ---------------------------------------------------------------------------
USER_DAILY_QUOTA: int = _env_int("USER_DAILY_QUOTA", 100)

# ---------------------------------------------------------------------------
# Timeouts and retries
# ---------------------------------------------------------------------------
MAX_RETRIES: int = _env_int("MAX_RETRIES", 3)
RETRY_BASE_DELAY: int = _env_int("RETRY_BASE_DELAY", 2)

# ---------------------------------------------------------------------------
# Thread and channel context
# ---------------------------------------------------------------------------
MAX_THREAD_DEPTH: int = _env_int("MAX_THREAD_DEPTH", 25)
THREAD_EXPIRY_MINUTES: int = _env_int("THREAD_EXPIRY_MINUTES", 120)
MAX_CHANNEL_CONTEXT: int = _env_int("MAX_CHANNEL_CONTEXT", 50)
CHANNEL_CONTEXT_MINUTES: int = _env_int("CHANNEL_CONTEXT_MINUTES", 30)
CHANNEL_CACHE_TTL: int = _env_int("CHANNEL_CACHE_TTL", 60)  # seconds before re-fetching
# Auto-reply: keep responding in a channel for this many seconds after last activity
AUTO_REPLY_TIMEOUT: int = _env_int("AUTO_REPLY_TIMEOUT", 300)  # 5 minutes
# How many raw Discord messages to scan to find MAX_CHANNEL_CONTEXT text messages
# (set higher than MAX_CHANNEL_CONTEXT to skip image/gif spam)
CHANNEL_SCAN_LIMIT: int = _env_int("CHANNEL_SCAN_LIMIT", 100)
# Extra thread history depth injected when user asks to recall past conversation
RECALL_THREAD_DEPTH: int = _env_int("RECALL_THREAD_DEPTH", 60)

# ---------------------------------------------------------------------------
# URL fetching
# ---------------------------------------------------------------------------
URL_FETCH_TIMEOUT: int = _env_int("URL_FETCH_TIMEOUT", 10)
URL_MAX_CONTENT_LENGTH: int = _env_int("URL_MAX_CONTENT_LENGTH", 4000)

# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------
SEARCH_TIMEOUT: int = _env_int("SEARCH_TIMEOUT", 10)
SEARCH_MAX_RESULTS: int = _env_int("SEARCH_MAX_RESULTS", 5)
SEARCH_MIN_RESULTS: int = _env_int("SEARCH_MIN_RESULTS", 2)
SEARCH_META_TIMEOUT: int = _env_int("SEARCH_META_TIMEOUT", 4)

# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------
MAX_PROMPT_LENGTH: int = _env_int("MAX_PROMPT_LENGTH", 6000)
MAX_IMAGE_SIZE: int = _env_int("MAX_IMAGE_SIZE", 10 * 1024 * 1024)  # 10MB
MAX_IMAGES_PER_REQUEST: int = 4
STREAM_STALE_SECONDS: int = _env_int("STREAM_STALE_SECONDS", 200)

# ---------------------------------------------------------------------------
# UX
# ---------------------------------------------------------------------------
USER_COOLDOWN: int = _env_int("USER_COOLDOWN", 0)
DISCORD_MSG_MAX: int = 2000  # Discord plain message character limit

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")
LOG_DIR: Path = BASE_DIR / _env("LOG_DIR", "logs")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH: Path = BASE_DIR / _env("DB_PATH", "db/chatbot.db")
DB_MAX_SIZE_MB: int = _env_int("DB_MAX_SIZE_MB", 5120)       # 5 GB
DB_CLEANUP_MB: int = _env_int("DB_CLEANUP_MB", 1024)         # free ~1 GB when limit hit

# ---------------------------------------------------------------------------
# Embed limits (Discord constants)
# ---------------------------------------------------------------------------
EMBED_DESCRIPTION_MAX: int = 4096
EMBED_FIELD_MAX: int = 1024
EMBED_TOTAL_MAX: int = 6000

# ---------------------------------------------------------------------------
# Colors (hex ints for discord.Colour)
# ---------------------------------------------------------------------------
COLOR_PRIMARY: int = 0x2D6A4F    # deep green
COLOR_ERROR: int = 0xD62828      # red
COLOR_WARNING: int = 0xE09F3E    # amber/mustard
COLOR_INFO: int = 0x457B9D       # slate blue-grey
COLOR_SOURCES: int = 0x6C757D   # muted grey for search citations
