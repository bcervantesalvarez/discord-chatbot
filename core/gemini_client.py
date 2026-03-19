"""Async Google Gemini API client with streaming, rate limiting, and system prompt support."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import aiohttp

import config
from core.models import GenerationResult, ThreadMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rate limiter -- in-memory, per-model rolling windows
# ---------------------------------------------------------------------------

@dataclass
class _RateState:
    """Tracks usage counters for a single Gemini model."""
    # Timestamps of requests in the current minute window (for RPM)
    minute_timestamps: list[float] = field(default_factory=list)
    # Timestamps of requests in the current day window (for RPD)
    day_timestamps: list[float] = field(default_factory=list)
    # (timestamp, token_count) pairs for TPM tracking
    minute_tokens: list[tuple[float, int]] = field(default_factory=list)


class GeminiRateLimiter:
    """Rate limiter for Gemini API free-tier limits.

    Tracks requests-per-minute (RPM), tokens-per-minute (TPM), and
    requests-per-day (RPD) using rolling time windows.

    Persists RPD data to SQLite so counters survive bot restarts.
    """

    def __init__(self, db: "Database | None" = None) -> None:
        self._states: dict[str, _RateState] = {}
        self._db = db  # optional Database for persistence

    def _get_state(self, model: str) -> _RateState:
        if model not in self._states:
            self._states[model] = _RateState()
        return self._states[model]

    async def load_from_db(self) -> int:
        """Load persisted usage data from the database (last 24h).

        Call this once at startup to restore rate counters.
        Returns the number of records loaded.
        """
        if not self._db:
            return 0
        rows = await self._db.load_gemini_usage(window_seconds=86400)
        now = time.time()
        count = 0
        for row in rows:
            model = row["model"]
            ts = row["timestamp"]
            tokens = row["tokens"]
            state = self._get_state(model)
            # Restore day timestamps (RPD)
            if now - ts < 86400:
                state.day_timestamps.append(ts)
            # Restore minute timestamps (RPM) — only if within last 60s
            if now - ts < 60:
                state.minute_timestamps.append(ts)
                if tokens > 0:
                    state.minute_tokens.append((ts, tokens))
            count += 1
        logger.info("Loaded %d Gemini rate-limit records from database", count)
        # Prune old records from DB while we're at it
        pruned = await self._db.prune_gemini_usage(86400)
        if pruned > 0:
            logger.info("Pruned %d expired rate-limit records from database", pruned)
        return count

    @staticmethod
    def _prune(timestamps: list[float], window_seconds: float) -> None:
        """Remove entries older than *window_seconds* in-place."""
        cutoff = time.time() - window_seconds
        while timestamps and timestamps[0] < cutoff:
            timestamps.pop(0)

    @staticmethod
    def _prune_tokens(entries: list[tuple[float, int]], window_seconds: float) -> None:
        cutoff = time.time() - window_seconds
        while entries and entries[0][0] < cutoff:
            entries.pop(0)

    def check_rate_limit(self, model: str) -> tuple[bool, int, str]:
        """Check whether a request to *model* is allowed.

        Returns (allowed, remaining_rpd, warning_message).
        *warning_message* is non-empty when remaining_rpd <= 5.
        """
        limits = config.GEMINI_RATE_LIMITS.get(model)
        if not limits:
            return True, 999, ""

        state = self._get_state(model)
        now = time.time()

        # Prune stale entries
        self._prune(state.minute_timestamps, 60)
        self._prune(state.day_timestamps, 86400)
        self._prune_tokens(state.minute_tokens, 60)

        rpm_limit = limits["rpm"]
        rpd_limit = limits["rpd"]
        tpm_limit = limits["tpm"]

        # Check RPM
        if len(state.minute_timestamps) >= rpm_limit:
            return False, rpd_limit - len(state.day_timestamps), (
                f"Gemini rate limit: {rpm_limit} requests/minute reached for {model}. "
                "Wait a moment or switch models."
            )

        # Check RPD
        remaining_rpd = rpd_limit - len(state.day_timestamps)
        if remaining_rpd <= 0:
            return False, 0, (
                f"Gemini daily limit reached for {model} ({rpd_limit} requests/day). "
                "Switch to another model."
            )

        # Check TPM
        current_tpm = sum(t for _, t in state.minute_tokens)
        if current_tpm >= tpm_limit:
            return False, remaining_rpd, (
                f"Gemini token limit: {tpm_limit:,} tokens/minute reached for {model}. "
                "Wait a moment or switch models."
            )

        # Build warning if close to daily limit
        warning = ""
        if remaining_rpd <= 5:
            warning = (
                f"\u26a0\ufe0f Gemini rate limit: {remaining_rpd} of {rpd_limit} "
                f"daily calls remaining for {model}"
            )

        return True, remaining_rpd, warning

    def record_usage(self, model: str, total_tokens: int = 0) -> None:
        """Record a completed request for rate tracking (in-memory).

        Also schedules a DB write if a database is configured.
        """
        state = self._get_state(model)
        now = time.time()
        state.minute_timestamps.append(now)
        state.day_timestamps.append(now)
        if total_tokens > 0:
            state.minute_tokens.append((now, total_tokens))
        # Persist to DB asynchronously (fire-and-forget)
        if self._db:
            asyncio.ensure_future(self._persist_usage(model, now, total_tokens))

    async def _persist_usage(self, model: str, timestamp: float, tokens: int) -> None:
        """Write a single usage record to the database."""
        try:
            if self._db:
                await self._db.record_gemini_usage(model, timestamp, tokens)
        except Exception:
            logger.warning("Failed to persist Gemini rate usage to DB", exc_info=True)

    def get_rate_warning(self, model: str) -> str:
        """Return a warning string if the model is close to its daily limit."""
        limits = config.GEMINI_RATE_LIMITS.get(model)
        if not limits:
            return ""
        state = self._get_state(model)
        self._prune(state.day_timestamps, 86400)
        remaining = limits["rpd"] - len(state.day_timestamps)
        if remaining <= 5:
            return (
                f"\u26a0\ufe0f Gemini rate limit: {remaining} of {limits['rpd']} "
                f"daily calls remaining for {model}"
            )
        return ""

    def get_status(self, model: str) -> dict:
        """Return current usage stats for a model (for /chat-status)."""
        limits = config.GEMINI_RATE_LIMITS.get(model, {})
        if not limits:
            return {}
        state = self._get_state(model)
        self._prune(state.minute_timestamps, 60)
        self._prune(state.day_timestamps, 86400)
        self._prune_tokens(state.minute_tokens, 60)
        return {
            "rpm_used": len(state.minute_timestamps),
            "rpm_limit": limits["rpm"],
            "rpd_used": len(state.day_timestamps),
            "rpd_limit": limits["rpd"],
            "tpm_used": sum(t for _, t in state.minute_tokens),
            "tpm_limit": limits["tpm"],
        }


# ---------------------------------------------------------------------------
# Gemini REST API client
# ---------------------------------------------------------------------------

class GeminiClient:
    """Async client for the Google Gemini REST API.

    Mirrors the interface used by QueueManager (chat_stream, is_healthy)
    so it can be used as the primary LLM provider.
    """

    def __init__(
        self,
        api_key: str | None = None,
        session: aiohttp.ClientSession | None = None,
        db: "Database | None" = None,
    ) -> None:
        self.api_key = api_key or config.GEMINI_API_KEY
        self.base_url = config.GEMINI_BASE_URL
        self._session: aiohttp.ClientSession | None = session
        self._owns_session = session is None
        self.rate_limiter = GeminiRateLimiter(db=db)

    async def start(self) -> None:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(
                connect=config.GEMINI_CONNECT_TIMEOUT,
                total=config.GEMINI_GENERATE_TIMEOUT,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
            self._owns_session = True
        # Restore rate-limit counters from DB so they survive restarts
        loaded = await self.rate_limiter.load_from_db()
        logger.info(
            "Gemini client started (API key: %s, restored %d rate-limit records)",
            "set" if self.api_key else "MISSING", loaded,
        )

    async def close(self) -> None:
        if self._session and self._owns_session:
            await self._session.close()
        self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("GeminiClient not started.")
        return self._session

    # -----------------------------------------------------------------------
    # Health check
    # -----------------------------------------------------------------------

    async def is_healthy(self, model: str | None = None) -> bool:
        """Check API key is configured and model isn't rate-limited."""
        if not self.api_key:
            logger.warning("Gemini API key not configured")
            return False
        if model:
            allowed, _, _ = self.rate_limiter.check_rate_limit(model)
            if not allowed:
                return False
        return True

    # -----------------------------------------------------------------------
    # Message format conversion
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_gemini_payload(
        messages: list[ThreadMessage],
        prompt: str,
        images: list[str] | None = None,
        channel_context: str = "",
        system_prompt: str = "",
        model: str | None = None,
    ) -> dict:
        """Convert inputs into a Gemini generateContent payload."""
        # System instruction
        system_parts: list[dict] = []
        sp = system_prompt or config.SYSTEM_PROMPT
        if sp:
            system_parts.append({"text": sp})
        if channel_context:
            system_parts.append({
                "text": (
                    "[Recent channel messages for context -- this is what people "
                    "have been saying in the channel recently]\n"
                    f"{channel_context}\n"
                    "[End of channel context]"
                ),
            })

        # Conversation history
        contents: list[dict] = []
        for msg in messages:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({
                "role": role,
                "parts": [{"text": msg.content}],
            })

        # Current user message
        user_parts: list[dict] = [{"text": prompt}]
        if images:
            for img_b64 in images:
                user_parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": img_b64,
                    }
                })
        contents.append({"role": "user", "parts": user_parts})

        payload: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": config.GEMINI_TEMPERATURE,
            },
        }

        # Gemma models don't support systemInstruction — inject it as a
        # leading user message instead so personality/context still works.
        use_model = model or ""
        if system_parts and use_model.startswith("gemma"):
            system_text = "\n\n".join(p["text"] for p in system_parts)
            contents.insert(0, {
                "role": "user",
                "parts": [{"text": f"[System instructions — follow these for all responses]\n{system_text}\n[End of system instructions]"}],
            })
            # Gemini API requires alternating user/model turns — add a
            # placeholder model ack so the next real user message is valid.
            contents.insert(1, {
                "role": "model",
                "parts": [{"text": "Understood."}],
            })
        elif system_parts:
            payload["systemInstruction"] = {"parts": system_parts}

        return payload

    # -----------------------------------------------------------------------
    # Streaming chat
    # -----------------------------------------------------------------------

    async def chat_stream(
        self,
        messages: list[ThreadMessage],
        prompt: str,
        images: list[str] | None = None,
        channel_context: str = "",
        model: str | None = None,
        system_prompt: str = "",
    ) -> AsyncGenerator[str, None]:
        """Stream text chunks from the Gemini generateContent API."""
        use_model = model or config.DEFAULT_MODEL
        payload = self._build_gemini_payload(
            messages, prompt, images, channel_context, system_prompt, use_model,
        )

        url = f"{self.base_url}/models/{use_model}:streamGenerateContent?alt=sse&key={self.api_key}"

        total_tokens = 0
        try:
            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Gemini API %d for %s: %s", resp.status, use_model, body[:300])
                    raise GeminiError(_friendly_api_error(resp.status, body, use_model))

                async for line in resp.content:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue

                    json_str = line[6:]  # strip "data: " prefix
                    if json_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue

                    # Extract text content
                    candidates = chunk.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text

                    # Track token usage from the final chunk
                    usage = chunk.get("usageMetadata")
                    if usage:
                        total_tokens = usage.get("totalTokenCount", 0)

        finally:
            # Always record usage even on error (the request was made)
            if total_tokens > 0:
                self.rate_limiter.record_usage(use_model, total_tokens)
            else:
                # Still count the request even if we didn't get token info
                self.rate_limiter.record_usage(use_model, 0)

    # -----------------------------------------------------------------------
    # Non-streaming chat (for completeness)
    # -----------------------------------------------------------------------

    async def chat(
        self,
        messages: list[ThreadMessage],
        prompt: str,
        images: list[str] | None = None,
        channel_context: str = "",
        model: str | None = None,
        system_prompt: str = "",
    ) -> GenerationResult:
        """Non-streaming Gemini call. Returns a complete GenerationResult."""
        use_model = model or config.DEFAULT_MODEL
        payload = self._build_gemini_payload(
            messages, prompt, images, channel_context, system_prompt, use_model,
        )

        url = f"{self.base_url}/models/{use_model}:generateContent?key={self.api_key}"

        try:
            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Gemini API %d for %s: %s", resp.status, use_model, body[:300])
                    raise GeminiError(_friendly_api_error(resp.status, body, use_model))
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            return GenerationResult(
                content="", model=use_model,
                error=f"Gemini request failed: {exc}",
            )

        # Extract response text
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        # Token usage
        usage = data.get("usageMetadata", {})
        total_tokens = usage.get("totalTokenCount", 0)
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)

        self.rate_limiter.record_usage(use_model, total_tokens)

        return GenerationResult(
            content=text,
            model=use_model,
            eval_count=completion_tokens,
            prompt_eval_count=prompt_tokens,
            rate_warning=self.rate_limiter.get_rate_warning(use_model),
        )

    def get_rate_warning(self, model: str) -> str:
        """Return warning string if close to limit, else empty."""
        return self.rate_limiter.get_rate_warning(model)


    # -----------------------------------------------------------------------
    # Image generation
    # -----------------------------------------------------------------------

    async def generate_image(self, prompt: str) -> Optional[tuple[bytes, str]]:
        """Generate an image using Gemini's image generation capability.

        Uses gemini-2.0-flash-exp with IMAGE response modality.
        Returns (image_bytes, mime_type) or None on failure.
        """
        model = "gemini-2.0-flash-exp"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
            },
        }

        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

        try:
            async with self.session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning("Image generation failed (%d): %s", resp.status, body[:300])
                    return None
                data = await resp.json()

            candidates = data.get("candidates", [])
            if not candidates:
                return None

            parts = candidates[0].get("content", {}).get("parts", [])
            for part in parts:
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    import base64
                    image_bytes = base64.b64decode(inline["data"])
                    mime_type = inline.get("mimeType", "image/png")
                    return image_bytes, mime_type

        except Exception as exc:
            logger.warning("Image generation error: %s", exc)

        return None


class GeminiError(Exception):
    """Raised when the Gemini API returns an error."""


def _friendly_api_error(status: int, body: str, model: str) -> str:
    """Convert a raw Gemini API error into a user-friendly message."""
    if status == 429:
        return (
            f"Rate limit hit for {model}. Too many requests -- "
            "try again in a minute or switch models with `/chat-model`."
        )
    if status == 401 or status == 403:
        return "API authentication error. The bot owner needs to check the API key."
    if status == 400:
        # Try to extract the actual message from JSON
        try:
            data = json.loads(body)
            msg = data.get("error", {}).get("message", "")
            if "safety" in msg.lower() or "block" in msg.lower():
                return "That request was blocked by the API's safety filters. Try rephrasing."
            if msg:
                return f"Bad request: {msg[:200]}"
        except (json.JSONDecodeError, AttributeError):
            pass
        return "Bad request -- the API rejected this prompt. Try rephrasing."
    if status == 404:
        return f"Model '{model}' not found. Try switching models with `/chat-model`."
    if status == 500 or status == 503:
        return "Gemini API is temporarily down. Try again in a moment."
    # Generic fallback -- still don't dump raw JSON
    return f"Gemini API error (HTTP {status}). Try again in a moment."
