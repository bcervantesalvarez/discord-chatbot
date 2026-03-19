"""Single-worker async queue manager for Discord Chatbot.

One worker processes requests sequentially -- only one response at a time.
SQLite handles persistence; stale pending items from a previous session are
cancelled on startup (they've lost runtime data).

Provider: Gemini (cloud) with automatic fallback chain across models.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

import config
from core.database import Database
from core.gemini_client import GeminiClient, GeminiError
from core.models import GenerationResult, QueueItem, QueueStatus, ThreadMessage

logger = logging.getLogger(__name__)


class QueueManager:
    """Manages the request queue and dispatches work to Gemini."""

    def __init__(
        self,
        db: Database,
        gemini: GeminiClient | None = None,
    ) -> None:
        self.db = db
        self.gemini = gemini
        self._queue: asyncio.Queue[QueueItem] = asyncio.Queue(maxsize=config.MAX_QUEUE_SIZE)
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False
        self._busy = False  # True while actively generating a response
        self._response_callback: Optional[Callable] = None
        self._chunk_callback: Optional[Callable] = None
        self._thread_history_fn: Optional[Callable] = None

    # -- callbacks ----------------------------------------------------------

    def set_response_callback(self, callback: Callable) -> None:
        self._response_callback = callback

    def set_chunk_callback(self, callback: Callable) -> None:
        self._chunk_callback = callback

    def set_thread_history_fn(self, fn: Callable) -> None:
        self._thread_history_fn = fn

    # -- lifecycle ----------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        self._running = True

        # Cancel stale pending items from a previous session -- they've lost
        # images, channel_context, and display_name so replaying them produces
        # garbage.  Better to fail cleanly.
        cancelled = await self.db.cancel_all_pending()
        if cancelled > 0:
            logger.info("Cancelled %d stale pending items from previous session", cancelled)

        self._worker_task = asyncio.create_task(self._worker_loop(), name="chatbot-worker")
        logger.info("Queue manager started (single worker, queue size: %d)", config.MAX_QUEUE_SIZE)

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("Queue manager stopped")

    # -- submission ---------------------------------------------------------

    async def submit(self, item: QueueItem) -> tuple[bool, str]:
        # Per-user daily quota check
        daily_used = await self.db.get_user_daily_usage(item.user_id)
        if daily_used >= config.USER_DAILY_QUOTA:
            return False, (
                f"You've hit your daily limit ({config.USER_DAILY_QUOTA} requests). "
                "Resets at midnight."
            )

        user_pending = await self.db.count_user_pending(item.user_id)
        if user_pending >= config.MAX_USER_PENDING:
            return False, f"You already have {user_pending} pending. Chill."

        if config.USER_COOLDOWN > 0:
            last_time = await self.db.get_user_last_request_time(item.user_id)
            if last_time:
                elapsed = time.time() - last_time
                if elapsed < config.USER_COOLDOWN:
                    remaining = int(config.USER_COOLDOWN - elapsed)
                    return False, f"Cooldown. Try again in {remaining}s."

        if self._queue.full():
            return False, "Queue's full. Try again in a sec."

        db_id = await self.db.enqueue(item)
        item.db_id = db_id

        await self._queue.put(item)
        position = self._queue.qsize()

        if self._busy and position > 0:
            return True, f"I'm busy rn. You're #{position} in line."
        return True, ""

    async def cancel_user_requests(self, user_id: int) -> int:
        return await self.db.cancel_user_pending(user_id)

    # -- model resolution ---------------------------------------------------

    async def _resolve_model(self, requested_model: str | None) -> tuple[str, str, str]:
        """Resolve a model request through the fallback chain.

        Returns ``(model_tag, provider, switch_notice)``.
        *switch_notice* is non-empty when we fell back from the originally
        intended model.
        """
        # User explicitly picked a model via /chat-model -- honour it, no fallback.
        if requested_model:
            if self.gemini:
                allowed, _, msg = self.gemini.rate_limiter.check_rate_limit(requested_model)
                if not allowed:
                    return requested_model, "gemini", ""  # caller handles the error
            return requested_model, "gemini", ""

        # Default path — walk the Gemini fallback chain
        if self.gemini:
            chain = config.GEMINI_FALLBACK_CHAIN
            for i, model in enumerate(chain):
                allowed, remaining_rpd, _ = self.gemini.rate_limiter.check_rate_limit(model)
                if allowed:
                    notice = ""
                    if i > 0:
                        notice = (
                            f"\u26a0\ufe0f Auto-switched from {chain[0]} to {model} "
                            f"(rate limit reached on previous models)"
                        )
                    return model, "gemini", notice

        # All Gemini models exhausted (or no Gemini client)
        return config.DEFAULT_MODEL, "gemini", (
            "\u26a0\ufe0f All models are rate-limited for today. "
            "Try again later or wait for limits to reset."
        )

    # -- worker -------------------------------------------------------------

    async def _worker_loop(self) -> None:
        logger.info("chatbot-worker started")
        while self._running:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._process_item(item)
            except Exception:
                logger.exception("Unhandled error processing queue item %s", item.db_id)
                if item.db_id:
                    await self.db.update_queue_status(item.db_id, QueueStatus.FAILED, "Internal error")
            finally:
                self._busy = False
                self._queue.task_done()

    async def _process_item(self, item: QueueItem) -> None:
        self._busy = True
        t0 = time.time()

        if item.db_id:
            await self.db.update_queue_status(item.db_id, QueueStatus.PROCESSING)

        # ---- Resolve model through fallback chain -------------------------
        use_model, provider, switch_notice = await self._resolve_model(item.model_override)

        # ---- Health / rate-limit pre-check --------------------------------
        if not self.gemini:
            error_msg = "Gemini API not configured. Set GEMINI_API_KEY in .env."
            logger.warning(error_msg)
            if item.db_id:
                await self.db.update_queue_status(item.db_id, QueueStatus.FAILED, error_msg)
            if self._response_callback:
                result = GenerationResult(content="", model=use_model, error=error_msg)
                await self._response_callback(item, result, True)
            return

        allowed, _, rate_msg = self.gemini.rate_limiter.check_rate_limit(use_model)
        if not allowed:
            logger.warning("Gemini rate limit: %s", rate_msg)
            if item.db_id:
                await self.db.update_queue_status(item.db_id, QueueStatus.FAILED, rate_msg)
            if self._response_callback:
                result = GenerationResult(content="", model=use_model, error=rate_msg)
                await self._response_callback(item, result, True)
            return

        if not await self.gemini.is_healthy(use_model):
            error_msg = f"Gemini API unavailable for model '{use_model}'."
            logger.warning(error_msg)
            if item.db_id:
                await self.db.update_queue_status(item.db_id, QueueStatus.FAILED, error_msg)
            if self._response_callback:
                result = GenerationResult(content="", model=use_model, error=error_msg)
                await self._response_callback(item, result, True)
            return

        logger.info(
            "Health check OK [gemini:%s] (%.1fs), building thread context...",
            use_model, time.time() - t0,
        )

        # Build thread context
        thread_history: list[ThreadMessage] = []
        if self._thread_history_fn:
            thread_history = await self._thread_history_fn(item)

        logger.info(
            "Thread context ready (%.1fs). Prompt: %d chars, %d thread msgs, %d chars channel ctx. Starting generation...",
            time.time() - t0, len(item.prompt), len(thread_history), len(item.channel_context),
        )

        # ---- Generate via Gemini (with retries for transient failures) -----
        start_time = time.time()
        chunks: list[str] = []
        error_msg: Optional[str] = None
        rate_warning: str = ""

        # Resolve per-user personality into a system prompt
        system_prompt = config.get_system_prompt(item.personality)

        for attempt in range(1, config.MAX_RETRIES + 1):
            chunks.clear()
            error_msg = None
            first_token_logged = False

            try:
                stream = self.gemini.chat_stream(
                    thread_history,
                    item.prompt,
                    item.images or None,
                    item.channel_context,
                    model=use_model,
                    system_prompt=system_prompt,
                )

                async for chunk in stream:
                    chunks.append(chunk)
                    if not first_token_logged:
                        logger.info("First token in %.1fs [gemini]", time.time() - start_time)
                        first_token_logged = True
                    if self._chunk_callback:
                        await self._chunk_callback(item, "".join(chunks))

                break  # success -- exit retry loop

            except GeminiError as exc:
                error_msg = str(exc)
                logger.error("Gemini error for item %s: %s", item.db_id, error_msg)
                # Don't retry API errors (rate limits, bad request, etc.)
                break
            except asyncio.TimeoutError:
                error_msg = "Generation timed out."
                logger.warning("Timeout for item %s (attempt %d/%d)", item.db_id, attempt, config.MAX_RETRIES)
                if attempt < config.MAX_RETRIES:
                    delay = config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.info("Retrying in %ds...", delay)
                    await asyncio.sleep(delay)
            except (ConnectionError, OSError) as exc:
                error_msg = str(exc)
                logger.warning("Network error for item %s (attempt %d/%d): %s", item.db_id, attempt, config.MAX_RETRIES, exc)
                if attempt < config.MAX_RETRIES:
                    delay = config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.info("Retrying in %ds...", delay)
                    await asyncio.sleep(delay)
            except Exception as exc:
                error_msg = str(exc)
                logger.exception("Error processing item %s", item.db_id)
                break  # don't retry unknown errors

        # Grab rate warning after generation (usage was recorded inside chat_stream)
        if use_model:
            rate_warning = self.gemini.get_rate_warning(use_model)

        collected_text = "".join(chunks)
        duration = time.time() - start_time
        result = GenerationResult(
            content=collected_text,
            model=use_model,
            total_duration_ns=int(duration * 1_000_000_000),
            eval_count=len(collected_text.split()),
            error=error_msg,
            rate_warning=rate_warning or None,
            switch_notice=switch_notice or None,
        )

        if self._response_callback:
            await self._response_callback(item, result, True)

        # --- Batch all post-generation DB writes into a single commit ---------
        if item.db_id:
            status = QueueStatus.FAILED if error_msg else QueueStatus.COMPLETED
            await self.db.update_queue_status(item.db_id, status, error_msg, commit=False)

        # Increment per-user daily usage counter (counts all attempts, not just successes)
        await self.db.increment_user_daily_usage(item.user_id, commit=False)

        await self.db.log_request(
            user_id=item.user_id,
            channel_id=item.channel_id,
            guild_id=item.guild_id,
            prompt_length=len(item.prompt),
            response_length=len(collected_text) if collected_text else None,
            model=use_model,
            eval_tokens=result.eval_count,
            prompt_tokens=result.prompt_eval_count,
            duration_secs=duration,
            status="completed" if not error_msg else "failed",
            commit=False,
        )

        # Save thread history using the ORIGINAL prompt (not the augmented one
        # with search results / URL content prepended).
        if not error_msg and collected_text:
            thread_key = self._get_thread_key(item)
            save_prompt = item.original_prompt or item.prompt
            await self.db.save_thread_message(
                thread_key,
                ThreadMessage(role="user", content=save_prompt, user_id=item.user_id),
                commit=False,
            )
            await self.db.save_thread_message(
                thread_key,
                ThreadMessage(role="assistant", content=collected_text),
                commit=False,
            )

            # Check if thread needs summarization (compress older messages)
            msg_count = await self.db.get_thread_message_count(thread_key)
            if msg_count > config.MAX_THREAD_DEPTH and self.gemini:
                asyncio.ensure_future(self._summarize_thread(thread_key))

        # Single commit for all post-generation writes
        await self.db.conn.commit()

        logger.info(
            "Processed item %s for user %s in %.1fs via gemini:%s (%s)",
            item.db_id, item.user_id, duration, use_model,
            "ok" if not error_msg else "error",
        )

    async def _summarize_thread(self, thread_key: str) -> None:
        """Compress older thread messages into a summary using Gemini.

        Keeps the most recent messages intact and summarizes the older ones.
        This runs as a fire-and-forget background task.
        """
        try:
            history = await self.db.get_thread_history(thread_key)
            if len(history) <= config.MAX_THREAD_DEPTH:
                return

            # Take the older messages (everything except the last 10)
            keep_recent = 10
            older = history[:-keep_recent] if len(history) > keep_recent else []
            if len(older) < 5:
                return  # not enough to summarize

            # Build a summarization prompt
            conversation_text = "\n".join(
                f"{'User' if m.role == 'user' else 'Bot'}: {m.content[:300]}"
                for m in older
            )

            summary_prompt = (
                "Summarize this conversation in 2-4 sentences. Focus on key topics discussed, "
                "decisions made, and important facts mentioned. Be concise.\n\n"
                f"{conversation_text}"
            )

            # Use a lightweight model for summarization
            result = await self.gemini.chat(
                messages=[],
                prompt=summary_prompt,
                model="gemini-2.5-flash-lite",
                system_prompt="You are a conversation summarizer. Output only the summary, nothing else.",
            )

            if result.content and not result.error:
                await self.db.save_thread_summary(
                    thread_key, result.content.strip(), len(history),
                )
                logger.info(
                    "Summarized %d older messages for thread %s",
                    len(older), thread_key,
                )

        except Exception:
            logger.warning("Thread summarization failed for %s", thread_key, exc_info=True)

    @staticmethod
    def _get_thread_key(item: QueueItem) -> str:
        return f"channel:{item.channel_id}:{item.user_id}"
