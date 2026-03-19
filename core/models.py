"""Data models used across the Discord Chatbot."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QueueStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueItem:
    """A single request in the processing queue."""

    user_id: int
    channel_id: int
    message_id: int
    prompt: str
    status: QueueStatus = QueueStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    db_id: Optional[int] = None
    # Runtime-only fields (not persisted to queue table)
    guild_id: Optional[int] = None
    user_display_name: str = ""
    reply_message_id: Optional[int] = None
    images: list[str] = field(default_factory=list)
    channel_context: str = ""
    # The user's original prompt text (before search/URL content was prepended).
    # Saved to thread history so future turns don't reload stale search results.
    original_prompt: str = ""
    # Per-request model override (from user's model preference). None = use default.
    model_override: Optional[str] = None
    # Enriched search results to display as a sources embed after the response.
    search_results: list = field(default_factory=list)
    # When True, inject deeper thread history for recall/recap requests.
    recall_mode: bool = False
    # Per-user personality key (e.g. "neutral"). None = default.
    personality: Optional[str] = None


@dataclass
class ThreadMessage:
    """A single message in a conversation thread."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    user_id: Optional[int] = None


@dataclass
class GenerationResult:
    """Result from an LLM generation request."""

    content: str
    model: str
    total_duration_ns: int = 0
    eval_count: int = 0
    prompt_eval_count: int = 0
    error: Optional[str] = None
    rate_warning: Optional[str] = None  # Gemini rate-limit warning
    switch_notice: Optional[str] = None  # Auto-fallback notice ("Switched from X to Y")

    @property
    def duration_seconds(self) -> float:
        return self.total_duration_ns / 1_000_000_000

    @property
    def tokens_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return self.eval_count / self.duration_seconds
