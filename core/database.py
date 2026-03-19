"""Async SQLite database layer for Discord Chatbot.

Single shared connection with WAL mode for concurrent read safety.
Handles queue persistence, thread history, and usage metrics.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

import config
from core.models import QueueItem, QueueStatus, ThreadMessage

logger = logging.getLogger(__name__)


class Database:
    """Async SQLite wrapper. Use as: db = Database(); await db.connect()."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or config.DB_PATH
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open the database connection, apply WAL mode, and ensure schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row

        # WAL mode: better concurrent read/write performance
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA busy_timeout=5000")

        await self._ensure_schema()
        logger.info("Database connected: %s", self.db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed")

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected. Call await db.connect() first.")
        return self._conn

    # -----------------------------------------------------------------------
    # Schema management
    # -----------------------------------------------------------------------

    async def _ensure_schema(self) -> None:
        """Apply schema if the database is fresh, then run any migrations."""
        async with self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
            schema_sql = schema_path.read_text(encoding="utf-8")
            await self.conn.executescript(schema_sql)
            await self.conn.commit()
            logger.info("Database schema initialized (version 2)")
        else:
            await self._apply_migrations()

    async def _apply_migrations(self) -> None:
        """Apply incremental migrations based on current schema version."""
        async with self.conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ) as cursor:
            row = await cursor.fetchone()
        current_version = row[0] if row and row[0] else 1

        if current_version < 2:
            # v2: add model_preferences table
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS model_preferences (
                    user_id     INTEGER PRIMARY KEY,
                    model       TEXT    NOT NULL,
                    updated_at  REAL    NOT NULL
                );
                INSERT INTO schema_version (version) VALUES (2);
            """)
            await self.conn.commit()
            logger.info("Applied migration v2: model_preferences table")
            current_version = 2

        if current_version < 3:
            # v3: add personality_preferences table
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS personality_preferences (
                    user_id     INTEGER PRIMARY KEY,
                    personality TEXT    NOT NULL,
                    updated_at  REAL    NOT NULL
                );
                INSERT INTO schema_version (version) VALUES (3);
            """)
            await self.conn.commit()
            logger.info("Applied migration v3: personality_preferences table")
            current_version = 3

        if current_version < 4:
            # v4: Gemini rate-limit usage tracking (persists across restarts)
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS gemini_rate_usage (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    model       TEXT    NOT NULL,
                    timestamp   REAL    NOT NULL,
                    tokens      INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_gemini_rate_model_ts
                    ON gemini_rate_usage (model, timestamp);
                INSERT INTO schema_version (version) VALUES (4);
            """)
            await self.conn.commit()
            logger.info("Applied migration v4: gemini_rate_usage table")
            current_version = 4

        if current_version < 5:
            # v5: guild_settings table for per-server configuration
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS guild_settings (
                    guild_id    INTEGER NOT NULL,
                    key         TEXT    NOT NULL,
                    value       TEXT    NOT NULL,
                    updated_at  REAL    NOT NULL,
                    PRIMARY KEY (guild_id, key)
                );
                INSERT INTO schema_version (version) VALUES (5);
            """)
            await self.conn.commit()
            logger.info("Applied migration v5: guild_settings table")
            current_version = 5

        if current_version < 6:
            # v6: thread_summaries for conversation memory compression
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS thread_summaries (
                    thread_key  TEXT PRIMARY KEY,
                    summary     TEXT    NOT NULL,
                    msg_count   INTEGER NOT NULL DEFAULT 0,
                    updated_at  REAL    NOT NULL
                );
                INSERT INTO schema_version (version) VALUES (6);
            """)
            await self.conn.commit()
            logger.info("Applied migration v6: thread_summaries table")
            current_version = 6

        if current_version < 7:
            # v7: per-user daily usage tracking with midnight reset
            await self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS daily_usage (
                    user_id       INTEGER NOT NULL,
                    date_key      TEXT    NOT NULL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (user_id, date_key)
                );
                INSERT INTO schema_version (version) VALUES (7);
            """)
            await self.conn.commit()
            logger.info("Applied migration v7: daily_usage table")

    # -----------------------------------------------------------------------
    # Queue operations
    # -----------------------------------------------------------------------

    async def enqueue(self, item: QueueItem) -> int:
        """Insert a queue item and return its database ID."""
        async with self.conn.execute(
            """INSERT INTO queue (user_id, channel_id, message_id, guild_id, prompt, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                item.user_id,
                item.channel_id,
                item.message_id,
                item.guild_id,
                item.prompt,
                item.status.value,
                item.created_at,
            ),
        ) as cursor:
            db_id = cursor.lastrowid
        await self.conn.commit()
        return db_id  # type: ignore[return-value]

    async def update_queue_status(
        self,
        db_id: int,
        status: QueueStatus,
        error_message: Optional[str] = None,
        *,
        commit: bool = True,
    ) -> None:
        """Update the status of a queue item."""
        completed_at = time.time() if status in (QueueStatus.COMPLETED, QueueStatus.FAILED) else None
        await self.conn.execute(
            "UPDATE queue SET status=?, completed_at=?, error_message=? WHERE id=?",
            (status.value, completed_at, error_message, db_id),
        )
        if commit:
            await self.conn.commit()

    async def get_pending_items(self) -> list[QueueItem]:
        """Retrieve all pending queue items ordered by creation time."""
        async with self.conn.execute(
            "SELECT * FROM queue WHERE status='pending' ORDER BY created_at ASC"
        ) as cursor:
            rows = await cursor.fetchall()

        items = []
        for row in rows:
            items.append(
                QueueItem(
                    db_id=row["id"],
                    user_id=row["user_id"],
                    channel_id=row["channel_id"],
                    message_id=row["message_id"],
                    guild_id=row["guild_id"],
                    prompt=row["prompt"],
                    status=QueueStatus(row["status"]),
                    created_at=row["created_at"],
                )
            )
        return items

    async def count_user_pending(self, user_id: int) -> int:
        """Count pending requests for a specific user."""
        async with self.conn.execute(
            "SELECT COUNT(*) FROM queue WHERE user_id=? AND status IN ('pending', 'processing')",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def count_pending(self) -> int:
        """Count total pending items in the queue."""
        async with self.conn.execute(
            "SELECT COUNT(*) FROM queue WHERE status IN ('pending', 'processing')"
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    async def cancel_user_pending(self, user_id: int) -> int:
        """Cancel all pending requests for a user. Returns count cancelled."""
        async with self.conn.execute(
            "UPDATE queue SET status='cancelled' WHERE user_id=? AND status='pending'",
            (user_id,),
        ) as cursor:
            count = cursor.rowcount
        await self.conn.commit()
        return count

    async def cancel_all_pending(self) -> int:
        """Cancel all pending/processing items (used on startup to clear stale state)."""
        async with self.conn.execute(
            "UPDATE queue SET status='cancelled', error_message='Stale: bot restarted' "
            "WHERE status IN ('pending', 'processing')",
        ) as cursor:
            count = cursor.rowcount
        await self.conn.commit()
        return count

    # -----------------------------------------------------------------------
    # Thread history
    # -----------------------------------------------------------------------

    async def save_thread_message(
        self, thread_key: str, message: ThreadMessage, *, commit: bool = True,
    ) -> None:
        """Save a message to thread history."""
        expires_at = time.time() + (config.THREAD_EXPIRY_MINUTES * 60)
        await self.conn.execute(
            """INSERT INTO thread_history (thread_key, role, content, user_id, timestamp, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (thread_key, message.role, message.content, message.user_id, message.timestamp, expires_at),
        )
        if commit:
            await self.conn.commit()

    async def get_thread_history(
        self, thread_key: str, *, depth_override: Optional[int] = None,
    ) -> list[ThreadMessage]:
        """Retrieve recent thread history with summary support.

        Returns at most *depth_override* messages (or MAX_THREAD_DEPTH when
        *depth_override* is None).  A larger depth is used for recall requests.

        If a conversation summary exists, it is prepended as the first message
        so the LLM has context from older conversation.
        """
        depth = depth_override or config.MAX_THREAD_DEPTH
        now = time.time()
        async with self.conn.execute(
            """SELECT role, content, user_id, timestamp FROM thread_history
               WHERE thread_key=? AND expires_at > ?
               ORDER BY timestamp ASC""",
            (thread_key, now),
        ) as cursor:
            rows = await cursor.fetchall()

        messages = [
            ThreadMessage(
                role=row["role"],
                content=row["content"],
                user_id=row["user_id"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

        # Prepend conversation summary if one exists
        summary = await self.get_thread_summary(thread_key)
        if summary:
            summary_msg = ThreadMessage(
                role="user",
                content=f"[Previous conversation summary]\n{summary}\n[End of summary]",
                timestamp=0.0,  # sorts first
            )
            return [summary_msg] + messages[-depth:]

        return messages[-depth:]

    async def clear_thread(self, thread_key: str) -> None:
        """Delete all history for a thread."""
        await self.conn.execute("DELETE FROM thread_history WHERE thread_key=?", (thread_key,))
        await self.conn.commit()

    async def cleanup_expired_threads(self) -> int:
        """Remove expired thread messages. Returns count deleted."""
        now = time.time()
        async with self.conn.execute(
            "DELETE FROM thread_history WHERE expires_at <= ?", (now,)
        ) as cursor:
            count = cursor.rowcount
        await self.conn.commit()
        return count

    # -----------------------------------------------------------------------
    # Thread summaries (conversation memory compression)
    # -----------------------------------------------------------------------

    async def get_thread_summary(self, thread_key: str) -> Optional[str]:
        """Get the compressed summary for a thread, if one exists."""
        async with self.conn.execute(
            "SELECT summary FROM thread_summaries WHERE thread_key=?",
            (thread_key,),
        ) as cursor:
            row = await cursor.fetchone()
        return row["summary"] if row else None

    async def save_thread_summary(
        self, thread_key: str, summary: str, msg_count: int,
    ) -> None:
        """Save or update a conversation summary for a thread."""
        await self.conn.execute(
            """INSERT INTO thread_summaries (thread_key, summary, msg_count, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(thread_key) DO UPDATE SET
                   summary=excluded.summary,
                   msg_count=excluded.msg_count,
                   updated_at=excluded.updated_at""",
            (thread_key, summary, msg_count, time.time()),
        )
        await self.conn.commit()

    async def get_thread_message_count(self, thread_key: str) -> int:
        """Count non-expired messages in a thread."""
        now = time.time()
        async with self.conn.execute(
            "SELECT COUNT(*) FROM thread_history WHERE thread_key=? AND expires_at > ?",
            (thread_key, now),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else 0

    # -----------------------------------------------------------------------
    # Model preferences
    # -----------------------------------------------------------------------

    async def get_user_model(self, user_id: int) -> Optional[str]:
        """Get a user's preferred model, or None for the default.

        Returns None (triggering the default fallback chain) if the stored
        model is no longer in AVAILABLE_MODELS — e.g. after a model rename.
        """
        async with self.conn.execute(
            "SELECT model FROM model_preferences WHERE user_id=?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        model = row["model"]
        # Validate against current available models
        valid_tags = {m["tag"] for m in config.AVAILABLE_MODELS}
        if model not in valid_tags:
            # Stale preference — clear it so the user falls back to default
            logger.info("Clearing stale model preference '%s' for user %s", model, user_id)
            await self.conn.execute("DELETE FROM model_preferences WHERE user_id=?", (user_id,))
            await self.conn.commit()
            return None
        return model

    async def set_user_model(self, user_id: int, model: str) -> None:
        """Set a user's preferred model."""
        await self.conn.execute(
            """INSERT INTO model_preferences (user_id, model, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET model=excluded.model, updated_at=excluded.updated_at""",
            (user_id, model, time.time()),
        )
        await self.conn.commit()

    async def clear_user_model(self, user_id: int) -> None:
        """Reset a user's model preference to default."""
        await self.conn.execute("DELETE FROM model_preferences WHERE user_id=?", (user_id,))
        await self.conn.commit()

    # -----------------------------------------------------------------------
    # Personality preferences
    # -----------------------------------------------------------------------

    async def get_user_personality(self, user_id: int) -> Optional[str]:
        """Get a user's personality key, or None for the default."""
        async with self.conn.execute(
            "SELECT personality FROM personality_preferences WHERE user_id=?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return row["personality"] if row else None

    async def set_user_personality(self, user_id: int, personality: str) -> None:
        """Set a user's personality preference."""
        await self.conn.execute(
            """INSERT INTO personality_preferences (user_id, personality, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET personality=excluded.personality, updated_at=excluded.updated_at""",
            (user_id, personality, time.time()),
        )
        await self.conn.commit()

    async def clear_user_personality(self, user_id: int) -> None:
        """Reset a user's personality to default."""
        await self.conn.execute("DELETE FROM personality_preferences WHERE user_id=?", (user_id,))
        await self.conn.commit()

    # -----------------------------------------------------------------------
    # Gemini rate-limit persistence
    # -----------------------------------------------------------------------

    async def record_gemini_usage(
        self, model: str, timestamp: float, tokens: int = 0, *, commit: bool = True,
    ) -> None:
        """Persist a single Gemini API call for rate-limit tracking."""
        await self.conn.execute(
            "INSERT INTO gemini_rate_usage (model, timestamp, tokens) VALUES (?, ?, ?)",
            (model, timestamp, tokens),
        )
        if commit:
            await self.conn.commit()

    async def load_gemini_usage(self, window_seconds: float = 86400) -> list[dict]:
        """Load recent Gemini API usage within *window_seconds* (default 24h).

        Returns a list of dicts: {"model": str, "timestamp": float, "tokens": int}.
        """
        cutoff = time.time() - window_seconds
        async with self.conn.execute(
            "SELECT model, timestamp, tokens FROM gemini_rate_usage WHERE timestamp > ? ORDER BY timestamp ASC",
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [{"model": row["model"], "timestamp": row["timestamp"], "tokens": row["tokens"]} for row in rows]

    async def prune_gemini_usage(self, window_seconds: float = 86400) -> int:
        """Delete Gemini usage records older than *window_seconds*."""
        cutoff = time.time() - window_seconds
        async with self.conn.execute(
            "DELETE FROM gemini_rate_usage WHERE timestamp <= ?", (cutoff,),
        ) as cursor:
            count = cursor.rowcount
        await self.conn.commit()
        return count

    # -----------------------------------------------------------------------
    # Usage metrics and request logging
    # -----------------------------------------------------------------------

    async def log_request(
        self,
        user_id: int,
        channel_id: int,
        guild_id: Optional[int],
        prompt_length: int,
        response_length: Optional[int],
        model: Optional[str],
        eval_tokens: Optional[int],
        prompt_tokens: Optional[int],
        duration_secs: Optional[float],
        status: str,
        *,
        commit: bool = True,
    ) -> None:
        """Log a completed (or failed) request for analytics.

        Pass ``commit=False`` when batching multiple writes into a single
        transaction — the caller is then responsible for committing.
        """
        await self.conn.execute(
            """INSERT INTO request_log
               (user_id, channel_id, guild_id, prompt_length, response_length,
                model, eval_tokens, prompt_tokens, duration_secs, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, channel_id, guild_id, prompt_length, response_length,
                model, eval_tokens, prompt_tokens, duration_secs, status, time.time(),
            ),
        )

        # Upsert usage_metrics
        await self.conn.execute(
            """INSERT INTO usage_metrics (user_id, request_count, total_tokens, total_duration, last_request_at)
               VALUES (?, 1, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   request_count = request_count + 1,
                   total_tokens = total_tokens + excluded.total_tokens,
                   total_duration = total_duration + excluded.total_duration,
                   last_request_at = excluded.last_request_at""",
            (user_id, eval_tokens or 0, duration_secs or 0.0, time.time()),
        )
        if commit:
            await self.conn.commit()

    async def get_avg_response_time(self, hours: int = 24) -> Optional[float]:
        """Average response time over the last N hours."""
        cutoff = time.time() - (hours * 3600)
        async with self.conn.execute(
            "SELECT AVG(duration_secs) FROM request_log WHERE status='completed' AND created_at > ?",
            (cutoff,),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row and row[0] is not None else None

    async def get_user_last_request_time(self, user_id: int) -> Optional[float]:
        """Get the timestamp of a user's last request."""
        async with self.conn.execute(
            "SELECT last_request_at FROM usage_metrics WHERE user_id=?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row[0] if row else None

    # -----------------------------------------------------------------------
    # Size-based cleanup
    # -----------------------------------------------------------------------

    def _file_size_mb(self) -> float:
        """Return the current database file size in MB."""
        try:
            return self.db_path.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    async def check_and_cleanup(self) -> bool:
        """If the DB exceeds DB_MAX_SIZE_MB, delete the oldest data to free
        approximately DB_CLEANUP_MB.  Returns True if a cleanup was performed.

        Strategy:
        1. Calculate what fraction of rows to purge (cleanup / current size).
        2. Delete that fraction of the oldest rows from `queue` (finished only)
           and `request_log` -- the two unbounded-growth tables.
        3. Delete all expired thread history.
        4. VACUUM to actually release disk space.
        """
        size_mb = self._file_size_mb()
        if size_mb < config.DB_MAX_SIZE_MB:
            return False

        logger.info(
            "Database size %.0f MB exceeds limit %d MB -- starting cleanup (target: free ~%d MB)",
            size_mb, config.DB_MAX_SIZE_MB, config.DB_CLEANUP_MB,
        )

        # Fraction of rows to remove
        fraction = min(config.DB_CLEANUP_MB / size_mb, 0.5)  # never nuke more than half

        # --- queue table (only finished items) ---
        async with self.conn.execute(
            "SELECT COUNT(*) FROM queue WHERE status IN ('completed', 'failed', 'cancelled')"
        ) as cur:
            row = await cur.fetchone()
        queue_total = row[0] if row else 0
        queue_delete = int(queue_total * fraction)

        if queue_delete > 0:
            await self.conn.execute(
                "DELETE FROM queue WHERE id IN ("
                "  SELECT id FROM queue"
                "  WHERE status IN ('completed', 'failed', 'cancelled')"
                "  ORDER BY created_at ASC LIMIT ?"
                ")",
                (queue_delete,),
            )
            logger.info("Purged %d oldest queue rows (of %d finished)", queue_delete, queue_total)

        # --- request_log table ---
        async with self.conn.execute("SELECT COUNT(*) FROM request_log") as cur:
            row = await cur.fetchone()
        log_total = row[0] if row else 0
        log_delete = int(log_total * fraction)

        if log_delete > 0:
            await self.conn.execute(
                "DELETE FROM request_log WHERE id IN ("
                "  SELECT id FROM request_log ORDER BY created_at ASC LIMIT ?"
                ")",
                (log_delete,),
            )
            logger.info("Purged %d oldest request_log rows (of %d)", log_delete, log_total)

        # --- expired threads (always safe to remove) ---
        now = time.time()
        await self.conn.execute("DELETE FROM thread_history WHERE expires_at <= ?", (now,))

        await self.conn.commit()

        # VACUUM reclaims freed pages but blocks ALL writers.  Only run if the
        # queue is idle (no pending/processing items) to avoid stalling requests.
        pending = await self.count_pending()
        if pending == 0:
            logger.info("No active requests -- running VACUUM")
            await self.conn.execute("VACUUM")
        else:
            logger.info("Skipping VACUUM (%d active requests) -- will retry next cycle", pending)

        new_size = self._file_size_mb()
        logger.info("Cleanup complete: %.0f MB -> %.0f MB (freed %.0f MB)", size_mb, new_size, size_mb - new_size)
        return True

    # -----------------------------------------------------------------------
    # Guild settings (per-server configuration)
    # -----------------------------------------------------------------------

    async def get_guild_settings(self, guild_id: int) -> dict[str, str]:
        """Get all settings for a guild as a dict."""
        async with self.conn.execute(
            "SELECT key, value FROM guild_settings WHERE guild_id=?", (guild_id,)
        ) as cursor:
            rows = await cursor.fetchall()
        return {row["key"]: row["value"] for row in rows}

    async def get_guild_setting(self, guild_id: int, key: str) -> Optional[str]:
        """Get a single guild setting value."""
        async with self.conn.execute(
            "SELECT value FROM guild_settings WHERE guild_id=? AND key=?",
            (guild_id, key),
        ) as cursor:
            row = await cursor.fetchone()
        return row["value"] if row else None

    async def set_guild_setting(self, guild_id: int, key: str, value: str) -> None:
        """Set a guild setting (upsert)."""
        await self.conn.execute(
            """INSERT INTO guild_settings (guild_id, key, value, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(guild_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
            (guild_id, key, value, time.time()),
        )
        await self.conn.commit()

    # -----------------------------------------------------------------------
    # Per-user daily quota
    # -----------------------------------------------------------------------

    @staticmethod
    def _today_key() -> str:
        """Return today's date as YYYY-MM-DD in local time."""
        from datetime import date
        return date.today().isoformat()

    async def get_user_daily_usage(self, user_id: int) -> int:
        """Return the number of requests a user has made today."""
        date_key = self._today_key()
        async with self.conn.execute(
            "SELECT request_count FROM daily_usage WHERE user_id=? AND date_key=?",
            (user_id, date_key),
        ) as cursor:
            row = await cursor.fetchone()
        return row["request_count"] if row else 0

    async def increment_user_daily_usage(self, user_id: int, *, commit: bool = True) -> int:
        """Increment a user's daily request count and return the new total."""
        date_key = self._today_key()
        await self.conn.execute(
            """INSERT INTO daily_usage (user_id, date_key, request_count)
               VALUES (?, ?, 1)
               ON CONFLICT(user_id, date_key) DO UPDATE SET
                   request_count = request_count + 1""",
            (user_id, date_key),
        )
        if commit:
            await self.conn.commit()
        # Return updated count
        async with self.conn.execute(
            "SELECT request_count FROM daily_usage WHERE user_id=? AND date_key=?",
            (user_id, date_key),
        ) as cursor:
            row = await cursor.fetchone()
        return row["request_count"] if row else 1

    async def prune_old_daily_usage(self) -> int:
        """Delete daily_usage rows from previous days (midnight reset)."""
        date_key = self._today_key()
        async with self.conn.execute(
            "DELETE FROM daily_usage WHERE date_key < ?", (date_key,),
        ) as cursor:
            count = cursor.rowcount
        if count:
            await self.conn.commit()
        return count

    async def get_all_daily_usage(self) -> list[tuple[int, int]]:
        """Return all (user_id, request_count) pairs for today."""
        date_key = self._today_key()
        async with self.conn.execute(
            "SELECT user_id, request_count FROM daily_usage WHERE date_key=? ORDER BY request_count DESC",
            (date_key,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [(row["user_id"], row["request_count"]) for row in rows]

    # -----------------------------------------------------------------------
    # Usage statistics (for /chat-stats dashboard)
    # -----------------------------------------------------------------------

    async def get_usage_stats(self, hours: int = 168) -> dict:
        """Aggregate usage statistics for the stats dashboard.

        Returns a dict with: top_users, daily_counts, avg_response_time,
        total_requests, model_usage.
        """
        cutoff = time.time() - (hours * 3600)
        stats: dict = {}

        # Top users by request count
        async with self.conn.execute(
            "SELECT user_id, COUNT(*) as cnt FROM request_log "
            "WHERE created_at > ? AND status='completed' "
            "GROUP BY user_id ORDER BY cnt DESC LIMIT 10",
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()
        stats["top_users"] = [(row["user_id"], row["cnt"]) for row in rows]

        # Daily request counts
        async with self.conn.execute(
            "SELECT date(created_at, 'unixepoch') as day, COUNT(*) as cnt "
            "FROM request_log WHERE created_at > ? "
            "GROUP BY day ORDER BY day ASC",
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()
        stats["daily_counts"] = [(row["day"], row["cnt"]) for row in rows]

        # Average response time
        async with self.conn.execute(
            "SELECT AVG(duration_secs), COUNT(*) FROM request_log "
            "WHERE status='completed' AND created_at > ?",
            (cutoff,),
        ) as cursor:
            row = await cursor.fetchone()
        stats["avg_response_time"] = row[0] if row and row[0] is not None else None
        stats["total_requests"] = row[1] if row else 0

        # Model usage breakdown
        async with self.conn.execute(
            "SELECT model, COUNT(*) as cnt FROM request_log "
            "WHERE created_at > ? AND model IS NOT NULL "
            "GROUP BY model ORDER BY cnt DESC",
            (cutoff,),
        ) as cursor:
            rows = await cursor.fetchall()
        stats["model_usage"] = [(row["model"], row["cnt"]) for row in rows]

        return stats
