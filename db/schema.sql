-- =============================================================================
-- Discord Chatbot Database Schema
-- Version: 6
-- =============================================================================

-- Track schema version for future migrations
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL,
    applied_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO schema_version (version) VALUES (7);

-- Request queue with persistence across restarts
CREATE TABLE IF NOT EXISTS queue (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    channel_id    INTEGER NOT NULL,
    message_id    INTEGER NOT NULL,
    guild_id      INTEGER,
    prompt        TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending',
    created_at    REAL    NOT NULL,
    completed_at  REAL,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_queue_status ON queue (status);
CREATE INDEX IF NOT EXISTS idx_queue_user   ON queue (user_id, status);

-- Multi-turn thread history keyed by a thread identifier
-- thread_key format: channel:{channel_id}:{user_id}
CREATE TABLE IF NOT EXISTS thread_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_key  TEXT    NOT NULL,
    role        TEXT    NOT NULL,  -- 'user' or 'assistant'
    content     TEXT    NOT NULL,
    user_id     INTEGER,
    timestamp   REAL    NOT NULL,
    expires_at  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_thread_key     ON thread_history (thread_key);
CREATE INDEX IF NOT EXISTS idx_thread_expires  ON thread_history (expires_at);

-- Aggregated usage metrics per user
CREATE TABLE IF NOT EXISTS usage_metrics (
    user_id         INTEGER PRIMARY KEY,
    request_count   INTEGER NOT NULL DEFAULT 0,
    total_tokens    INTEGER NOT NULL DEFAULT 0,
    total_duration  REAL    NOT NULL DEFAULT 0.0,
    last_request_at REAL
);

-- Per-request log for detailed analytics
CREATE TABLE IF NOT EXISTS request_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    channel_id      INTEGER NOT NULL,
    guild_id        INTEGER,
    prompt_length   INTEGER NOT NULL,
    response_length INTEGER,
    model           TEXT,
    eval_tokens     INTEGER,
    prompt_tokens   INTEGER,
    duration_secs   REAL,
    status          TEXT    NOT NULL,
    created_at      REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_request_log_user ON request_log (user_id);
CREATE INDEX IF NOT EXISTS idx_request_log_time ON request_log (created_at);

-- Per-user model preference (v2)
CREATE TABLE IF NOT EXISTS model_preferences (
    user_id     INTEGER PRIMARY KEY,
    model       TEXT    NOT NULL,
    updated_at  REAL    NOT NULL
);

-- Per-user personality preference (v3)
CREATE TABLE IF NOT EXISTS personality_preferences (
    user_id     INTEGER PRIMARY KEY,
    personality TEXT    NOT NULL,
    updated_at  REAL    NOT NULL
);

-- Gemini API rate-limit usage tracking (v4)
-- Each row = one API call. Loaded on startup to restore rate counters.
CREATE TABLE IF NOT EXISTS gemini_rate_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model       TEXT    NOT NULL,
    timestamp   REAL    NOT NULL,
    tokens      INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_gemini_rate_model_ts ON gemini_rate_usage (model, timestamp);

-- Per-server configuration (v5)
CREATE TABLE IF NOT EXISTS guild_settings (
    guild_id    INTEGER NOT NULL,
    key         TEXT    NOT NULL,
    value       TEXT    NOT NULL,
    updated_at  REAL    NOT NULL,
    PRIMARY KEY (guild_id, key)
);

-- Conversation summaries for long-term memory (v6)
CREATE TABLE IF NOT EXISTS thread_summaries (
    thread_key  TEXT PRIMARY KEY,
    summary     TEXT    NOT NULL,
    msg_count   INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL    NOT NULL
);

-- Per-user daily usage tracking (v7)
-- date_key is ISO date string (YYYY-MM-DD) in local time.
-- Resets automatically: rows from previous days are pruned on startup / cleanup.
CREATE TABLE IF NOT EXISTS daily_usage (
    user_id     INTEGER NOT NULL,
    date_key    TEXT    NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, date_key)
);
