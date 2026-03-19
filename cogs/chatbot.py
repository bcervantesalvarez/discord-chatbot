"""Discord Chatbot cog: channel context, vision, URL reading, web search, slash commands.

Features:
- @bot mentions, !chat prefix commands, and /chat slash command
- Auto-reply: once engaged, keeps responding for 5 min without needing @mention
- Channel message history as conversation context
- Image extraction from attachments, replies, and URLs
- URL content extraction from message text
- Automatic web search for factual queries
- Single-threaded: one response at a time, others queue
- Plain text responses for short answers, embeds for long ones
- /chat-stats usage dashboard
- Slash command autocomplete for models & personalities
- Per-server configuration (/chat-config)
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import aiohttp
import discord
from bs4 import BeautifulSoup
from discord import app_commands
from discord.ext import commands, tasks

import config
from core.database import Database
from core.models import GenerationResult, QueueItem, QueueStatus, ThreadMessage
from core.queue_manager import QueueManager
from core import web_search

logger = logging.getLogger(__name__)

# Regex for detecting image URLs in message text
IMAGE_URL_RE = re.compile(
    r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp))(?:\?\S*)?',
    re.IGNORECASE,
)

# Regex for detecting any URL in message text
URL_RE = re.compile(r'(https?://[^\s<>\"]+)', re.IGNORECASE)

# Reaction emoji for acknowledgment
THINKING_EMOJI = "\U0001f9e0"  # brain emoji

# ---------------------------------------------------------------------------
# SSRF protection -- block requests to private/internal networks
# ---------------------------------------------------------------------------
import ipaddress
from urllib.parse import urlparse


def _is_url_safe(url: str) -> bool:
    """Return True if the URL points to a public internet host.

    Blocks private IPs, loopback, link-local, and reserved ranges to
    prevent SSRF attacks where a user tricks the bot into fetching
    internal network resources.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False

        # Block obvious internal hostnames
        lower = hostname.lower()
        if lower in ("localhost", "0.0.0.0") or lower.endswith(".local"):
            return False

        # Resolve to IP and check ranges
        import socket
        for info in socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM):
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
    except (ValueError, OSError):
        return False
    return True

# Patterns for GIF / sticker / reaction-only messages (no useful text)
_GIF_URL_RE = re.compile(r'https?://(?:tenor\.com|giphy\.com|media\d?\.giphy\.com|cdn\.discordapp\.com/attachments)\S*', re.I)

# ---------------------------------------------------------------------------
# Recall detection -- keywords that mean "remind me what we talked about"
# ---------------------------------------------------------------------------
_RECALL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(what did (we|they|i|he|she|you|someone)\s+(talk|say|discuss|mention))", re.I),
    re.compile(r"\b(recap|summarize|summary)\s*(the\s+)?(chat|convo|conversation|thread)?", re.I),
    re.compile(r"\b(what was said|what happened)\s*(earlier|before|yesterday|last\s+time)?", re.I),
    re.compile(r"\b(do you remember|you remember)\b", re.I),
    re.compile(r"\b(bring me up to speed|catch me up|fill me in)\b", re.I),
    re.compile(r"\b(what('s| is| was) the context)\b", re.I),
    re.compile(r"\b(scroll back|look back|check (the )?history)\b", re.I),
]


def _wants_recall(prompt: str) -> bool:
    """Return True if the user is asking to recall past conversation."""
    for pattern in _RECALL_PATTERNS:
        if pattern.search(prompt):
            return True
    return False


# ---------------------------------------------------------------------------
# In-memory channel context cache
# ---------------------------------------------------------------------------

class _ChannelCache:
    """Caches text-only channel messages per channel with a TTL.

    Skips images, GIFs, stickers, reactions, and embed-only messages so the
    LLM context window is filled with *actual conversation*, not spam.
    """

    def __init__(self) -> None:
        # channel_id -> (timestamp, list[formatted_str])
        self._store: dict[int, tuple[float, list[str]]] = {}

    def get(self, channel_id: int) -> list[str] | None:
        """Return cached messages if still fresh, else None."""
        entry = self._store.get(channel_id)
        if entry is None:
            return None
        ts, msgs = entry
        if (time.time() - ts) > config.CHANNEL_CACHE_TTL:
            return None  # stale
        return msgs

    def put(self, channel_id: int, messages: list[str]) -> None:
        self._store[channel_id] = (time.time(), messages)

    def invalidate(self, channel_id: int) -> None:
        self._store.pop(channel_id, None)

    def evict_stale(self, max_age: float) -> int:
        """Remove entries older than *max_age* seconds. Returns count evicted."""
        now = time.time()
        stale = [cid for cid, (ts, _) in self._store.items() if (now - ts) > max_age]
        for cid in stale:
            del self._store[cid]
        return len(stale)

    def clear(self) -> None:
        self._store.clear()


# ---------------------------------------------------------------------------
# Category styling for the model picker buttons
# ---------------------------------------------------------------------------
_CATEGORY_STYLES: dict[str, discord.ButtonStyle] = {
    "Gemini": discord.ButtonStyle.primary,      # blurple for Gemini (cloud primary)
    "Gemma":  discord.ButtonStyle.secondary,    # grey for Gemma (fallback)
}


class ModelPickerView(discord.ui.View):
    """Interactive button grid for selecting a Gemini model.

    Shows buttons for all configured Gemini models when the API key is set.
    """

    def __init__(
        self,
        cog: "ChatBotCog",
        user_id: int,
        current_model: str,
        loaded_models: list[str],
    ) -> None:
        super().__init__(timeout=120)
        self.cog = cog
        self.user_id = user_id
        self.current_model = current_model
        self._build_buttons(loaded_models)

    def _build_buttons(self, loaded_models: list[str]) -> None:
        gemini_available = bool(config.GEMINI_API_KEY)
        if not gemini_available:
            return

        for entry in config.AVAILABLE_MODELS:
            tag = entry["tag"]
            is_current = (tag == self.current_model)
            style = discord.ButtonStyle.primary if is_current else _CATEGORY_STYLES.get(
                entry["category"], discord.ButtonStyle.secondary,
            )

            label = entry["label"]
            if is_current:
                label = f"\u2713 {label}"

            button = discord.ui.Button(
                label=label,
                style=style,
                custom_id=f"model_select:{tag}",
                row=self._row_for_category(entry["category"]),
            )
            button.callback = self._make_callback(tag, entry["label"])
            self.add_item(button)

        # Reset to default button
        reset_btn = discord.ui.Button(
            label="Reset to Default",
            style=discord.ButtonStyle.danger,
            custom_id="model_select:reset",
            row=3,
        )
        reset_btn.callback = self._make_reset_callback()
        self.add_item(reset_btn)

    @staticmethod
    def _row_for_category(category: str) -> int:
        return 0 if category == "Gemini" else 1

    def _make_callback(self, tag: str, label: str):
        async def callback(interaction: discord.Interaction) -> None:
            if interaction.user.id != self.user_id:
                await interaction.response.send_message("This picker isn't for you.", ephemeral=True)
                return
            await self.cog.db.set_user_model(self.user_id, tag)
            self.current_model = tag
            # Rebuild the view with updated highlight
            new_view = ModelPickerView(self.cog, self.user_id, tag, [])
            await interaction.response.edit_message(
                content=f"**Model set to `{label}`** for your requests.",
                view=new_view,
            )
        return callback

    def _make_reset_callback(self):
        async def callback(interaction: discord.Interaction) -> None:
            if interaction.user.id != self.user_id:
                await interaction.response.send_message("This picker isn't for you.", ephemeral=True)
                return
            await self.cog.db.clear_user_model(self.user_id)
            default = config.DEFAULT_MODEL
            self.current_model = default
            new_view = ModelPickerView(self.cog, self.user_id, default, [])
            await interaction.response.edit_message(
                content=f"**Reset to default model** (`{default}` with auto-fallback).",
                view=new_view,
            )
        return callback

    async def on_timeout(self) -> None:
        # Disable all buttons when the view expires
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True


class PersonalityPickerView(discord.ui.View):
    """Interactive button picker for switching personality."""

    _STYLE_MAP: dict[str, discord.ButtonStyle] = {
        "neutral": discord.ButtonStyle.secondary,  # grey
    }

    def __init__(self, cog: "ChatBotCog", user_id: int, current: str) -> None:
        super().__init__(timeout=120)
        self.cog = cog
        self.user_id = user_id
        self.current = current
        self._build_buttons()

    def _build_buttons(self) -> None:
        for key, entry in config.PERSONALITIES.items():
            is_current = (key == self.current)
            label = entry["label"]
            if is_current:
                label = f"\u2713 {label}"

            style = self._STYLE_MAP.get(key, discord.ButtonStyle.secondary)
            if is_current:
                style = discord.ButtonStyle.success  # green highlight for active

            button = discord.ui.Button(
                label=label,
                emoji=entry.get("emoji"),
                style=style,
                custom_id=f"persona_select:{key}",
                row=0,
            )
            button.callback = self._make_callback(key, entry["label"])
            self.add_item(button)

        # Reset to default
        reset_btn = discord.ui.Button(
            label="Reset to Default",
            style=discord.ButtonStyle.danger,
            custom_id="persona_select:reset",
            row=1,
        )
        reset_btn.callback = self._make_reset_callback()
        self.add_item(reset_btn)

    def _make_callback(self, key: str, label: str):
        async def callback(interaction: discord.Interaction) -> None:
            if interaction.user.id != self.user_id:
                await interaction.response.send_message("This picker isn't for you.", ephemeral=True)
                return
            await self.cog.db.set_user_personality(self.user_id, key)
            self.current = key
            new_view = PersonalityPickerView(self.cog, self.user_id, key)
            desc = config.PERSONALITIES[key]["description"]
            await interaction.response.edit_message(
                content=f"**Personality set to {label}** -- {desc}",
                view=new_view,
            )
        return callback

    def _make_reset_callback(self):
        async def callback(interaction: discord.Interaction) -> None:
            if interaction.user.id != self.user_id:
                await interaction.response.send_message("This picker isn't for you.", ephemeral=True)
                return
            await self.cog.db.clear_user_personality(self.user_id)
            default = config.DEFAULT_PERSONALITY
            self.current = default
            new_view = PersonalityPickerView(self.cog, self.user_id, default)
            label = config.PERSONALITIES[default]["label"]
            await interaction.response.edit_message(
                content=f"**Reset to default personality** ({label}).",
                view=new_view,
            )
        return callback

    async def on_timeout(self) -> None:
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True


class ChatBotCog(commands.Cog, name="Chatbot"):
    """Core cog for the Discord Chatbot."""

    def __init__(
        self,
        bot: commands.Bot,
        db: Database,
        queue_manager: QueueManager,
    ) -> None:
        self.bot = bot
        self.db = db
        self.queue_manager = queue_manager
        self._start_time = time.time()
        self._active_streams: dict[int, dict] = {}
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._channel_cache = _ChannelCache()

        # Auto-reply tracking: channel_id -> last_activity_timestamp
        # When the bot is mentioned, the channel enters "active" mode.
        # Subsequent messages in the same channel get auto-replies
        # without needing an @mention, until 5 min of silence.
        self._active_channels: dict[int, float] = {}

        # Register callbacks
        self.queue_manager.set_response_callback(self._handle_response)
        self.queue_manager.set_chunk_callback(self._handle_chunk)
        self.queue_manager.set_thread_history_fn(self._get_thread_history)

    async def cog_load(self) -> None:
        connector = aiohttp.TCPConnector(
            limit=200,            # total concurrent connections
            limit_per_host=30,    # per-host cap (prevents hammering one site)
            ttl_dns_cache=300,    # cache DNS lookups for 5 minutes
            enable_cleanup_closed=True,
        )
        self._http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=15),
            headers={"User-Agent": "DiscordChatbot/3.0"},
        )
        self._cleanup_threads.start()
        self._stream_updater.start()
        logger.info("ChatBotCog loaded")

    async def cog_unload(self) -> None:
        self._cleanup_threads.cancel()
        self._stream_updater.cancel()
        self._channel_cache.clear()
        logger.info("Channel context cache cleared")
        if self._http_session:
            await self._http_session.close()
            self._http_session = None

    @property
    def http(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            raise RuntimeError("HTTP session not initialized")
        return self._http_session

    # -----------------------------------------------------------------------
    # Auto-reply helpers
    # -----------------------------------------------------------------------

    def _mark_channel_active(self, channel_id: int) -> None:
        """Mark a channel as active (bot was mentioned or is auto-replying)."""
        self._active_channels[channel_id] = time.time()

    def _is_channel_active(self, channel_id: int) -> bool:
        """Return True if the channel has had bot activity within the timeout."""
        last = self._active_channels.get(channel_id)
        if last is None:
            return False
        if (time.time() - last) > config.AUTO_REPLY_TIMEOUT:
            # Expired -- clean up
            self._active_channels.pop(channel_id, None)
            return False
        return True

    def _deactivate_channel(self, channel_id: int) -> None:
        """Remove a channel from active tracking."""
        self._active_channels.pop(channel_id, None)

    # -----------------------------------------------------------------------
    # Event listeners
    # -----------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or not message.guild:
            return

        # Don't double-fire if this is also a valid prefix command
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return

        is_mentioned = self.bot.user in message.mentions
        is_active_channel = self._is_channel_active(message.channel.id)

        # Only respond if explicitly mentioned OR channel is in active mode
        if not is_mentioned and not is_active_channel:
            return

        prompt = message.content
        # Strip bot mention from the prompt
        if self.bot.user:
            for mention_str in [f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"]:
                prompt = prompt.replace(mention_str, "").strip()

        if not prompt and not message.attachments:
            if is_mentioned:
                await message.reply("What's up? Ask me something or send me an image.", mention_author=False)
            return

        if not prompt and message.attachments:
            prompt = "What do you see in this image?"

        # Mark/refresh the channel as active
        self._mark_channel_active(message.channel.id)

        await self._submit_request(message, prompt)

    # -----------------------------------------------------------------------
    # Prefix commands
    # -----------------------------------------------------------------------

    @commands.command(name="chat", aliases=["c"])
    async def chat_cmd(self, ctx: commands.Context, *, prompt: str = "") -> None:
        if not prompt and not ctx.message.attachments:
            await ctx.reply(f"Usage: `{config.BOT_PREFIX}chat <your question>`", mention_author=False)
            return
        if not prompt and ctx.message.attachments:
            prompt = "What do you see in this image?"
        self._mark_channel_active(ctx.channel.id)
        await self._submit_request(ctx.message, prompt)

    @commands.command(name="chat-status")
    async def chat_status(self, ctx: commands.Context) -> None:
        await self._send_status(ctx)

    @commands.command(name="chat-help")
    async def chat_help(self, ctx: commands.Context) -> None:
        await self._send_help(ctx)

    @commands.command(name="chat-clear")
    async def chat_clear(self, ctx: commands.Context) -> None:
        count = await self.queue_manager.cancel_user_requests(ctx.author.id)
        msg = f"Cancelled {count} pending request(s)." if count > 0 else "You have no pending requests."
        await ctx.reply(msg, mention_author=False)

    @commands.command(name="chat-forget")
    async def chat_forget(self, ctx: commands.Context) -> None:
        thread_key = f"channel:{ctx.channel.id}:{ctx.author.id}"
        await self.db.clear_thread(thread_key)
        await ctx.reply("Cleared your conversation history in this channel.", mention_author=False)

    @commands.command(name="chat-model")
    async def chat_model(self, ctx: commands.Context) -> None:
        """Show the model picker."""
        await self._send_model_picker(ctx.author, ctx.channel, ctx.message)

    @commands.command(name="chat-persona")
    @commands.cooldown(1, 300, commands.BucketType.user)  # 5 minute cooldown
    async def chat_persona(self, ctx: commands.Context) -> None:
        """Pick personality for your requests."""
        await self._send_persona_picker(ctx.author, ctx.channel, ctx.message)

    @commands.command(name="chat-stats")
    async def chat_stats_cmd(self, ctx: commands.Context) -> None:
        """Show usage statistics dashboard."""
        embed = await self._build_stats_embed(ctx.guild)
        await ctx.reply(embed=embed, mention_author=False)

    @commands.command(name="chat-stop")
    async def chat_stop(self, ctx: commands.Context) -> None:
        """Stop auto-replying in this channel."""
        self._deactivate_channel(ctx.channel.id)
        await ctx.reply("Stopped auto-replying in this channel.", mention_author=False)

    @commands.command(name="chat-quota")
    async def chat_quota(self, ctx: commands.Context) -> None:
        """Check your remaining daily quota."""
        used = await self.db.get_user_daily_usage(ctx.author.id)
        remaining = max(0, config.USER_DAILY_QUOTA - used)
        pct = (used / config.USER_DAILY_QUOTA) * 100 if config.USER_DAILY_QUOTA > 0 else 0
        bar_len = 20
        filled = int(bar_len * used / config.USER_DAILY_QUOTA) if config.USER_DAILY_QUOTA > 0 else 0
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        await ctx.reply(
            f"**Daily Quota:** {used}/{config.USER_DAILY_QUOTA} used ({pct:.0f}%)\n"
            f"`{bar}` {remaining} remaining\n"
            f"Resets at midnight.",
            mention_author=False,
        )

    # -----------------------------------------------------------------------
    # Slash commands (modern Discord)
    # -----------------------------------------------------------------------

    @app_commands.command(name="chat", description="Ask the chatbot a question")
    @app_commands.describe(question="What do you want to ask?")
    async def slash_chat(self, interaction: discord.Interaction, question: str) -> None:
        await interaction.response.defer(thinking=True)
        followup = await interaction.followup.send(f"**{interaction.user.display_name}** asked: {question}", wait=True)
        self._mark_channel_active(interaction.channel_id)
        await self._submit_request(followup, question, invoker=interaction.user)

    @app_commands.command(name="chat-status", description="Check the chatbot's status")
    async def slash_status(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        embed = await self._build_status_embed()
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="chat-forget", description="Clear your conversation history")
    async def slash_forget(self, interaction: discord.Interaction) -> None:
        thread_key = f"channel:{interaction.channel_id}:{interaction.user.id}"
        await self.db.clear_thread(thread_key)
        await interaction.response.send_message("Cleared your conversation history in this channel.", ephemeral=True)

    @app_commands.command(name="chat-model", description="Pick which AI model to use for your requests")
    async def slash_model(self, interaction: discord.Interaction) -> None:
        current = await self.db.get_user_model(interaction.user.id) or config.DEFAULT_MODEL
        view = ModelPickerView(self, interaction.user.id, current, [])
        await interaction.response.send_message(
            f"**Pick a model.** Currently using `{current}`.",
            view=view,
            ephemeral=True,
        )

    @app_commands.command(name="chat-persona", description="Switch personality for your requests")
    @app_commands.checks.cooldown(1, 300)  # 5 minute cooldown
    async def slash_persona(self, interaction: discord.Interaction) -> None:
        current = await self.db.get_user_personality(interaction.user.id) or config.DEFAULT_PERSONALITY
        entry = config.PERSONALITIES.get(current, config.PERSONALITIES[config.DEFAULT_PERSONALITY])
        view = PersonalityPickerView(self, interaction.user.id, current)
        await interaction.response.send_message(
            f"**Pick a personality.** Currently: {entry['emoji']} {entry['label']}",
            view=view,
            ephemeral=True,
        )

    @app_commands.command(name="chat-stats", description="View usage statistics dashboard")
    async def slash_stats(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        embed = await self._build_stats_embed(interaction.guild)
        await interaction.followup.send(embed=embed, ephemeral=True)

    @app_commands.command(name="chat-stop", description="Stop auto-replying in this channel")
    async def slash_stop(self, interaction: discord.Interaction) -> None:
        self._deactivate_channel(interaction.channel_id)
        await interaction.response.send_message("Stopped auto-replying in this channel.", ephemeral=True)

    @app_commands.command(name="chat-quota", description="Check your remaining daily request quota")
    async def slash_quota(self, interaction: discord.Interaction) -> None:
        used = await self.db.get_user_daily_usage(interaction.user.id)
        remaining = max(0, config.USER_DAILY_QUOTA - used)
        pct = (used / config.USER_DAILY_QUOTA) * 100 if config.USER_DAILY_QUOTA > 0 else 0
        bar_len = 20
        filled = int(bar_len * used / config.USER_DAILY_QUOTA) if config.USER_DAILY_QUOTA > 0 else 0
        bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
        await interaction.response.send_message(
            f"**Daily Quota:** {used}/{config.USER_DAILY_QUOTA} used ({pct:.0f}%)\n"
            f"`{bar}` {remaining} remaining\n"
            f"Resets at midnight.",
            ephemeral=True,
        )

    @app_commands.command(name="chat-config", description="Configure the chatbot for this server (admin only)")
    @app_commands.describe(
        default_personality="Default personality for this server",
        auto_reply="Enable/disable auto-reply in this server",
    )
    @app_commands.checks.has_permissions(manage_guild=True)
    async def slash_config(
        self,
        interaction: discord.Interaction,
        default_personality: Optional[str] = None,
        auto_reply: Optional[bool] = None,
    ) -> None:
        if not interaction.guild:
            await interaction.response.send_message("This command only works in servers.", ephemeral=True)
            return

        changes: list[str] = []
        guild_id = interaction.guild.id

        if default_personality is not None:
            if default_personality not in config.PERSONALITIES:
                valid = ", ".join(config.PERSONALITIES.keys())
                await interaction.response.send_message(
                    f"Invalid personality. Choose from: {valid}", ephemeral=True,
                )
                return
            await self.db.set_guild_setting(guild_id, "default_personality", default_personality)
            changes.append(f"Default personality: **{default_personality}**")

        if auto_reply is not None:
            await self.db.set_guild_setting(guild_id, "auto_reply", str(auto_reply).lower())
            changes.append(f"Auto-reply: **{'enabled' if auto_reply else 'disabled'}**")

        if not changes:
            # Show current settings
            settings = await self.db.get_guild_settings(guild_id)
            lines = [
                f"**Default personality:** {settings.get('default_personality', config.DEFAULT_PERSONALITY)}",
                f"**Auto-reply:** {settings.get('auto_reply', 'true')}",
            ]
            await interaction.response.send_message(
                "**Current server settings:**\n" + "\n".join(lines),
                ephemeral=True,
            )
        else:
            await interaction.response.send_message(
                "**Settings updated:**\n" + "\n".join(changes),
                ephemeral=True,
            )

    @slash_config.autocomplete("default_personality")
    async def _config_personality_autocomplete(
        self, interaction: discord.Interaction, current: str,
    ) -> list[app_commands.Choice[str]]:
        return [
            app_commands.Choice(name=entry["label"], value=key)
            for key, entry in config.PERSONALITIES.items()
            if current.lower() in key.lower() or current.lower() in entry["label"].lower()
        ][:25]

    @app_commands.command(name="chat-imagine", description="Generate an image from a text prompt")
    @app_commands.describe(prompt="Describe the image you want to generate")
    async def slash_imagine(self, interaction: discord.Interaction, prompt: str) -> None:
        gemini = self.queue_manager.gemini
        if not gemini:
            await interaction.response.send_message("Image generation unavailable (no API key).", ephemeral=True)
            return
        await interaction.response.defer(thinking=True)
        result = await gemini.generate_image(prompt)
        if result is None:
            await interaction.followup.send("Failed to generate image. Try a different prompt.")
            return
        image_bytes, mime_type = result
        ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
        file = discord.File(
            fp=__import__("io").BytesIO(image_bytes),
            filename=f"chatbot_gen.{ext}",
        )
        await interaction.followup.send(f"**{prompt}**", file=file)

    @commands.command(name="chat-imagine")
    async def chat_imagine_cmd(self, ctx: commands.Context, *, prompt: str = "") -> None:
        """Generate an image from a text prompt."""
        if not prompt:
            await ctx.reply(f"Usage: `{config.BOT_PREFIX}chat-imagine <description>`", mention_author=False)
            return
        gemini = self.queue_manager.gemini
        if not gemini:
            await ctx.reply("Image generation unavailable.", mention_author=False)
            return
        async with ctx.typing():
            result = await gemini.generate_image(prompt)
        if result is None:
            await ctx.reply("Failed to generate image. Try a different prompt.", mention_author=False)
            return
        image_bytes, mime_type = result
        ext = mime_type.split("/")[-1] if "/" in mime_type else "png"
        file = discord.File(
            fp=__import__("io").BytesIO(image_bytes),
            filename=f"chatbot_gen.{ext}",
        )
        await ctx.reply(f"**{prompt}**", file=file, mention_author=False)

    # -----------------------------------------------------------------------
    # Shared helpers for prefix + slash commands
    # -----------------------------------------------------------------------

    async def _build_status_embed(self) -> discord.Embed:
        queue_depth = self.queue_manager.depth
        total_pending = await self.db.count_pending()
        avg_time = await self.db.get_avg_response_time(hours=24)
        uptime_secs = time.time() - self._start_time
        hours, remainder = divmod(int(uptime_secs), 3600)
        minutes, seconds = divmod(remainder, 60)

        active_channels = sum(1 for ts in self._active_channels.values()
                              if (time.time() - ts) <= config.AUTO_REPLY_TIMEOUT)

        lines = [
            f"**Default model:** {config.DEFAULT_MODEL} (auto-fallback)",
            f"**Status:** {'Generating...' if self.queue_manager.is_busy else 'Idle'}",
            f"**Queue:** {queue_depth} waiting / {total_pending} total pending",
            f"**Avg response time (24h):** {avg_time:.1f}s" if avg_time else "**Avg response time:** No data yet",
            f"**Uptime:** {hours}h {minutes}m {seconds}s",
            f"**Active conversations:** {active_channels}",
        ]

        # Gemini fallback chain status
        gemini = self.queue_manager.gemini
        if gemini:
            lines.append("")
            lines.append("**Gemini Fallback Chain:**")
            for tag in config.GEMINI_FALLBACK_CHAIN:
                stats = gemini.rate_limiter.get_status(tag)
                if stats:
                    remaining = stats['rpd_limit'] - stats['rpd_used']
                    indicator = "\u2705" if remaining > 5 else "\u26a0\ufe0f" if remaining > 0 else "\u274c"
                    lines.append(
                        f"  {indicator} {tag}: {stats['rpd_used']}/{stats['rpd_limit']} daily, "
                        f"{stats['rpm_used']}/{stats['rpm_limit']} rpm"
                    )
                else:
                    lines.append(f"  \u2705 {tag}: 0 used")

        color = config.COLOR_PRIMARY
        return self._make_embed("Chatbot Status", "\n".join(lines), color=color)

    async def _build_stats_embed(self, guild: Optional[discord.Guild] = None) -> discord.Embed:
        """Build a usage statistics dashboard embed."""
        stats = await self.db.get_usage_stats(hours=168)  # last 7 days

        lines: list[str] = []

        # Top users (last 7 days)
        if stats.get("top_users"):
            lines.append("**Top Users (7d):**")
            for i, (user_id, count) in enumerate(stats["top_users"][:5], 1):
                # Try to resolve username
                user = self.bot.get_user(user_id)
                name = user.display_name if user else f"User {user_id}"
                lines.append(f"  {i}. {name} -- {count} requests")
            lines.append("")

        # Daily request counts (last 7 days)
        if stats.get("daily_counts"):
            lines.append("**Daily Requests (7d):**")
            for date_str, count in stats["daily_counts"][-7:]:
                bar = "\u2588" * min(count, 30)
                lines.append(f"  `{date_str}` {bar} {count}")
            lines.append("")

        # Response time stats
        avg = stats.get("avg_response_time")
        total = stats.get("total_requests", 0)
        lines.append(f"**Total requests (7d):** {total}")
        if avg is not None:
            lines.append(f"**Avg response time:** {avg:.1f}s")

        # Model usage breakdown
        if stats.get("model_usage"):
            lines.append("")
            lines.append("**Model Usage (7d):**")
            for model, count in stats["model_usage"]:
                lines.append(f"  {model}: {count} requests")

        # Rate limit status
        gemini = self.queue_manager.gemini
        if gemini:
            lines.append("")
            lines.append("**Quota Remaining:**")
            for tag in config.GEMINI_FALLBACK_CHAIN:
                s = gemini.rate_limiter.get_status(tag)
                if s:
                    remaining = s['rpd_limit'] - s['rpd_used']
                    pct = (remaining / s['rpd_limit']) * 100 if s['rpd_limit'] > 0 else 0
                    lines.append(f"  {tag}: {remaining}/{s['rpd_limit']} ({pct:.0f}%)")

        return self._make_embed("Chatbot Stats", "\n".join(lines), color=config.COLOR_INFO)

    async def _send_status(self, ctx: commands.Context) -> None:
        embed = await self._build_status_embed()
        await ctx.reply(embed=embed, mention_author=False)

    async def _send_model_picker(
        self, user: discord.User | discord.Member, channel: discord.abc.Messageable, message: discord.Message,
    ) -> None:
        current = await self.db.get_user_model(user.id) or config.DEFAULT_MODEL
        if not config.GEMINI_API_KEY:
            await self._safe_reply(message, "No API key configured. Can't show models.")
            return
        view = ModelPickerView(self, user.id, current, [])
        await channel.send(
            f"**Pick a model, {user.display_name}.** Currently using `{current}`.",
            view=view,
        )

    async def _send_persona_picker(
        self, user: discord.User | discord.Member, channel: discord.abc.Messageable, message: discord.Message,
    ) -> None:
        current = await self.db.get_user_personality(user.id) or config.DEFAULT_PERSONALITY
        entry = config.PERSONALITIES.get(current, config.PERSONALITIES[config.DEFAULT_PERSONALITY])
        view = PersonalityPickerView(self, user.id, current)
        await channel.send(
            f"**Pick a personality, {user.display_name}.** Currently: {entry['emoji']} {entry['label']}",
            view=view,
        )

    async def _send_help(self, ctx: commands.Context) -> None:
        bot_name = self.bot.user.display_name if self.bot.user else "Chatbot"
        lines = [
            f"**Ask a question:** `@{bot_name} <question>` or `{config.BOT_PREFIX}chat <question>` or `/chat`",
            "",
            "**Features:**",
            f"Reads the last {config.MAX_CHANNEL_CONTEXT} channel messages for context.",
            "Attach images or paste image URLs and I can see them.",
            "Ask factual questions and I'll search the web automatically.",
            "Paste a link and I'll read the page content.",
            f"Reply to a message and mention @{bot_name} to include that message as context.",
            f"Once engaged, I keep replying for {config.AUTO_REPLY_TIMEOUT // 60} min without needing @mention.",
            "",
            f"**Commands:** `{config.BOT_PREFIX}chat-status` | `{config.BOT_PREFIX}chat-model` | "
            f"`{config.BOT_PREFIX}chat-persona` | `{config.BOT_PREFIX}chat-stats` | "
            f"`{config.BOT_PREFIX}chat-quota` | `{config.BOT_PREFIX}chat-clear` | "
            f"`{config.BOT_PREFIX}chat-forget` | `{config.BOT_PREFIX}chat-stop`",
            "**Slash commands:** `/chat` | `/chat-model` | `/chat-persona` | "
            "`/chat-status` | `/chat-stats` | `/chat-quota` | `/chat-forget` | "
            "`/chat-stop` | `/chat-config`",
        ]
        await ctx.reply(
            embed=self._make_embed(f"{bot_name} Help", "\n".join(lines), color=config.COLOR_INFO),
            mention_author=False,
        )

    # -----------------------------------------------------------------------
    # Request submission
    # -----------------------------------------------------------------------

    async def _submit_request(
        self,
        message: discord.Message,
        prompt: str,
        invoker: Optional[discord.User | discord.Member] = None,
    ) -> None:
        """Build and submit a queue item.

        *invoker* overrides the user identity (used for slash commands where
        ``message`` is the bot's followup, not the user's original message).
        """
        user = invoker or message.author

        # Look up user's model preference
        user_model = await self.db.get_user_model(user.id)
        user_personality = await self.db.get_user_personality(user.id)

        # React with brain emoji for instant acknowledgment
        try:
            await message.add_reaction(THINKING_EMOJI)
        except discord.HTTPException:
            pass

        # --- Gather pre-generation work in parallel --------------------------
        # All network-bound tasks are fired concurrently to minimize wall-clock
        # time before generation starts.

        # Task 1: extract images from this message (+ replied-to message)
        async def _gather_images() -> list[str]:
            imgs = await self._extract_images(message)
            if message.reference and message.reference.message_id:
                try:
                    ref_msg = await message.channel.fetch_message(message.reference.message_id)
                    imgs.extend(await self._extract_images(ref_msg))
                except (discord.NotFound, discord.Forbidden):
                    pass
            # Also fetch image URLs from message text
            image_urls = IMAGE_URL_RE.findall(prompt)
            fetch_tasks = [
                self._fetch_image_url(url)
                for url in image_urls[:config.MAX_IMAGES_PER_REQUEST - len(imgs)]
            ]
            if fetch_tasks:
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, str):
                        imgs.append(r)
            return imgs

        # Task 2: fetch URL content (if any non-image URLs present)
        async def _gather_url_content() -> str:
            non_image_urls = [u for u in URL_RE.findall(prompt) if not IMAGE_URL_RE.match(u)]
            if non_image_urls:
                return await self._fetch_url_content(non_image_urls[0])
            return ""

        # Task 3: channel context -- fetched BEFORE search so the search
        # decision can use conversation context for directive detection
        # ("look it up", "try again") and topic extraction.
        channel_context = await self._build_channel_context(message)

        # Task 4: web search (DuckDuckGo only -- metadata fetched later)
        async def _gather_search() -> tuple[str, list[dict[str, str]]]:
            if not web_search.needs_search(prompt, channel_context):
                return "", []
            # Build a smart query -- directives ("look it up") get replaced
            # with the actual topic extracted from conversation context
            search_query = web_search.build_search_query(prompt, channel_context)
            logger.info("Search triggered for user %s: query=%s (prompt=%s)", user.id, search_query[:80], prompt[:40])
            raw = await web_search.search(
                search_query, max_results=config.SEARCH_MAX_RESULTS, session=self._http_session,
            )
            # Quality-filter: keep 2-5 results based on relevance
            raw = web_search.filter_results(
                raw, min_results=config.SEARCH_MIN_RESULTS, max_results=config.SEARCH_MAX_RESULTS,
            )
            # Format raw results for the LLM prompt (no metadata fetch yet)
            context = web_search.format_results_raw(raw) if raw else ""
            return context, raw

        # Fire remaining tasks concurrently (channel context already done)
        images, url_content, (search_context, raw_search_results) = (
            await asyncio.gather(
                _gather_images(),
                _gather_url_content(),
                _gather_search(),
            )
        )

        # Assemble augmented prompt (search + URL content prepended)
        original_prompt = prompt
        full_prompt = prompt
        if search_context:
            full_prompt = f"{search_context}\n\n{full_prompt}"
        if url_content:
            full_prompt = f"[Content from linked page]\n{url_content}\n[End of page content]\n\n{full_prompt}"

        # Truncate assembled prompt to avoid blowing the context window
        if len(full_prompt) > config.MAX_PROMPT_LENGTH:
            full_prompt = full_prompt[:config.MAX_PROMPT_LENGTH] + "\n[...truncated]"

        # Detect recall requests -- inject deeper thread history
        recall = _wants_recall(prompt)
        if recall:
            logger.info("Recall triggered for user %s: %s", user.id, prompt[:80])

        item = QueueItem(
            user_id=user.id,
            channel_id=message.channel.id,
            message_id=message.id,
            guild_id=message.guild.id if message.guild else None,
            prompt=full_prompt,
            user_display_name=user.display_name,
            reply_message_id=message.reference.message_id if message.reference else None,
            images=images[:config.MAX_IMAGES_PER_REQUEST],
            channel_context=channel_context,
            original_prompt=original_prompt,
            model_override=user_model,
            recall_mode=recall,
            personality=user_personality,
        )

        # Kick off metadata enrichment concurrently -- runs while item waits
        # in queue and during generation.  The result lands on the item
        # before _handle_response needs it (generation takes 5-30s, enrichment ~1-3s).
        if raw_search_results:
            async def _enrich_later() -> None:
                try:
                    item.search_results = await web_search.enrich_results(
                        raw_search_results, self._http_session,
                    )
                except Exception as exc:
                    logger.warning("Metadata enrichment failed: %s", exc)
                    # Fallback: build SearchResult objects without metadata
                    item.search_results = [
                        web_search.SearchResult(
                            title=r.get("title", ""),
                            snippet=r.get("snippet", ""),
                            url=r.get("url", ""),
                        )
                        for r in raw_search_results
                    ]
            asyncio.create_task(_enrich_later())

        success, status_msg = await self.queue_manager.submit(item)

        if not success:
            try:
                await message.remove_reaction(THINKING_EMOJI, self.bot.user)
            except discord.HTTPException:
                pass
            await self._safe_reply(message, status_msg)
            return

        # Register stream tracker for typing indicator
        self._active_streams[item.message_id] = {
            "original_msg": message,
            "finalized": False,
            "channel": message.channel,
            "started_at": time.time(),
        }

    # -----------------------------------------------------------------------
    # Channel context builder (cached, text-only)
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_text_message(msg: discord.Message, bot_id: int) -> bool:
        """Return True only if the message contains meaningful text content.

        Skips: image-only, GIF links, sticker-only, embed-only (bot status),
        and reaction-style single-emoji messages.
        """
        # Bot's embed-only messages (status, errors, sources)
        if msg.author.id == bot_id and msg.embeds and not msg.content:
            return False

        content = msg.content.strip()

        # No text at all (image / sticker / reaction only)
        if not content:
            return False

        # Single emoji (unicode or custom Discord emoji)
        if len(content) <= 50 and re.fullmatch(r'(<a?:\w+:\d+>\s*)+', content):
            return False  # custom emoji spam
        if len(content) <= 8 and all(
            ('\U0001f000' <= c <= '\U0001faff') or ('\u2600' <= c <= '\u27bf')
            or c in '\ufe0f\u200d' or c.isspace()
            for c in content
        ):
            return False  # unicode emoji only

        # GIF / tenor / giphy link with no other text
        stripped_of_urls = _GIF_URL_RE.sub('', content).strip()
        if not stripped_of_urls:
            return False

        return True

    async def _build_channel_context(self, message: discord.Message) -> str:
        """Return recent text-only channel messages, using cache when fresh."""
        channel_id = message.channel.id

        # Check cache first
        cached = self._channel_cache.get(channel_id)
        if cached is not None:
            return "\n".join(cached)

        # Cache miss -- fetch from Discord, filtering to text-only
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=config.CHANNEL_CONTEXT_MINUTES)
            text_msgs: list[str] = []
            bot_id = self.bot.user.id if self.bot.user else 0

            async for msg in message.channel.history(
                limit=config.CHANNEL_SCAN_LIMIT, before=message,
            ):
                if msg.created_at < cutoff:
                    break
                if not self._is_text_message(msg, bot_id):
                    continue

                content = msg.content.strip()
                if len(content) > 200:
                    content = content[:200] + "..."

                is_self = msg.author.id == bot_id
                if is_self:
                    text_msgs.append(f"Chatbot (you, earlier): {content}")
                else:
                    text_msgs.append(f"{msg.author.display_name}: {content}")

                if len(text_msgs) >= config.MAX_CHANNEL_CONTEXT:
                    break

            text_msgs.reverse()
            self._channel_cache.put(channel_id, text_msgs)
            return "\n".join(text_msgs)

        except (discord.Forbidden, discord.HTTPException) as exc:
            logger.warning("Failed to fetch channel history: %s", exc)
            return ""

    # -----------------------------------------------------------------------
    # Image extraction
    # -----------------------------------------------------------------------

    @staticmethod
    async def _extract_images(message: discord.Message) -> list[str]:
        images: list[str] = []
        image_mimes = ("image/png", "image/jpeg", "image/gif", "image/webp")

        for attachment in message.attachments:
            if len(images) >= config.MAX_IMAGES_PER_REQUEST:
                break
            # Skip oversized attachments
            if attachment.size and attachment.size > config.MAX_IMAGE_SIZE:
                logger.info("Skipping oversized attachment %s (%d bytes)", attachment.filename, attachment.size)
                continue
            ct = (attachment.content_type or "").split(";")[0].strip().lower()
            if ct in image_mimes or attachment.filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".gif", ".webp")
            ):
                try:
                    image_bytes = await attachment.read()
                    images.append(base64.b64encode(image_bytes).decode("utf-8"))
                    logger.info("Extracted image: %s (%d bytes)", attachment.filename, len(image_bytes))
                except (discord.HTTPException, discord.NotFound) as exc:
                    logger.warning("Failed to download attachment %s: %s", attachment.filename, exc)

        return images

    async def _fetch_image_url(self, url: str) -> Optional[str]:
        """Download an image URL and return as base64."""
        if not _is_url_safe(url):
            logger.warning("Blocked image fetch (SSRF protection): %s", url)
            return None
        try:
            async with self.http.get(url) as resp:
                if resp.status != 200:
                    return None
                ct = resp.content_type or ""
                if not ct.startswith("image/"):
                    return None
                # Check content-length header before downloading
                cl = resp.content_length
                if cl and cl > config.MAX_IMAGE_SIZE:
                    return None
                data = await resp.read()
                if len(data) > config.MAX_IMAGE_SIZE:
                    return None
                return base64.b64encode(data).decode("utf-8")
        except Exception as exc:
            logger.warning("Failed to fetch image URL %s: %s", url, exc)
            return None

    # -----------------------------------------------------------------------
    # URL content extraction
    # -----------------------------------------------------------------------

    async def _fetch_url_content(self, url: str) -> str:
        """Fetch a URL and extract readable text content."""
        if not _is_url_safe(url):
            logger.warning("Blocked URL fetch (SSRF protection): %s", url)
            return ""
        try:
            timeout = aiohttp.ClientTimeout(total=config.URL_FETCH_TIMEOUT)
            async with self._http_session.get(url, timeout=timeout) as resp:
                if resp.status != 200:
                    return ""
                ct = resp.content_type or ""
                if "html" not in ct and "text" not in ct:
                    return ""
                # Read only the first 200KB to avoid blocking on huge pages
                raw = await resp.content.read(204800)
                html = raw.decode("utf-8", errors="replace")

            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            if len(text) > config.URL_MAX_CONTENT_LENGTH:
                text = text[:config.URL_MAX_CONTENT_LENGTH] + "\n[...truncated]"

            return text

        except Exception as exc:
            logger.warning("Failed to fetch URL %s: %s", url, exc)
            return ""

    # -----------------------------------------------------------------------
    # Thread context retrieval
    # -----------------------------------------------------------------------

    async def _get_thread_history(self, item: QueueItem) -> list[ThreadMessage]:
        messages: list[ThreadMessage] = []

        thread_key = QueueManager._get_thread_key(item)
        stored = await self.db.get_thread_history(
            thread_key,
            depth_override=config.RECALL_THREAD_DEPTH if item.recall_mode else None,
        )
        messages.extend(stored)

        if item.reply_message_id:
            try:
                channel = self.bot.get_channel(item.channel_id)
                if channel is None:
                    channel = await self.bot.fetch_channel(item.channel_id)
                chain_messages = await self._walk_reply_chain(channel, item.reply_message_id)
                for cm in chain_messages:
                    if not any(m.content == cm.content and m.role == cm.role for m in messages):
                        messages.append(cm)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException) as exc:
                logger.warning("Failed to fetch reply chain: %s", exc)

        depth = config.RECALL_THREAD_DEPTH if item.recall_mode else config.MAX_THREAD_DEPTH
        return messages[-depth:]

    async def _walk_reply_chain(
        self, channel: discord.abc.Messageable, message_id: int, depth: int = 0,
    ) -> list[ThreadMessage]:
        if depth >= config.MAX_THREAD_DEPTH:
            return []

        try:
            msg = await channel.fetch_message(message_id)
        except (discord.NotFound, discord.Forbidden):
            return []

        is_bot = msg.author.id == self.bot.user.id if self.bot.user else False
        role = "assistant" if is_bot else "user"

        parent_messages: list[ThreadMessage] = []
        if msg.reference and msg.reference.message_id:
            parent_messages = await self._walk_reply_chain(channel, msg.reference.message_id, depth + 1)

        content = msg.content
        if is_bot and msg.embeds:
            content = msg.embeds[0].description or content

        current = ThreadMessage(
            role=role, content=content, user_id=msg.author.id, timestamp=msg.created_at.timestamp(),
        )

        return parent_messages + [current]

    # -----------------------------------------------------------------------
    # Response handling
    # -----------------------------------------------------------------------

    async def _handle_response(self, item: QueueItem, result: GenerationResult, is_final: bool) -> None:
        stream_data = self._active_streams.pop(item.message_id, None)
        if not stream_data:
            await self._send_fallback_response(item, result)
            return

        original_msg: discord.Message = stream_data["original_msg"]

        # Remove brain reaction
        try:
            await original_msg.remove_reaction(THINKING_EMOJI, self.bot.user)
        except discord.HTTPException:
            pass

        if result.error:
            await self._safe_reply(original_msg, f"Something went wrong: {result.error}")
            return

        content = result.content.strip()

        # Empty response -- reply in-character instead of breaking immersion
        if not content:
            await self._safe_reply(original_msg, "...")
            return

        if len(content) <= config.DISCORD_MSG_MAX:
            await self._safe_reply(original_msg, content)
        else:
            # Over 2000 chars -- split into multiple messages, embeds as last resort
            await self._send_long_response(original_msg, content, result)

        # Append a sources embed if web search was used
        await self._send_sources_embed(original_msg.channel, item)

        # Append model switch notice and/or rate-limit warning
        notices = [n for n in (result.switch_notice, result.rate_warning) if n]
        if notices:
            try:
                await original_msg.channel.send("\n".join(notices))
            except discord.HTTPException:
                pass

        # Refresh auto-reply timer after responding
        self._mark_channel_active(item.channel_id)

    async def _send_sources_embed(
        self, channel: discord.abc.Messageable, item: QueueItem,
    ) -> None:
        """Send the search sources embed if the item has enriched search results."""
        if not item.search_results:
            return
        try:
            embed = web_search.build_sources_embed(item.search_results)
            await channel.send(embed=embed)
        except discord.HTTPException as exc:
            logger.warning("Failed to send sources embed: %s", exc)

    async def _send_long_response(
        self, original_msg: discord.Message, content: str, result: GenerationResult,
    ) -> None:
        """Split long responses into multiple plain-text messages (max 2000 chars each).
        Falls back to embeds only if plain-text splitting fails."""
        chunks = self._split_text(content, config.DISCORD_MSG_MAX)
        try:
            for chunk in chunks:
                await original_msg.channel.send(chunk)
        except discord.HTTPException as exc:
            logger.warning("Failed to send chunked response, falling back to embeds: %s", exc)
            embeds = self._split_response_embeds(content, result)
            for embed in embeds:
                try:
                    await original_msg.channel.send(embed=embed)
                except discord.HTTPException:
                    break

    @staticmethod
    def _split_text(text: str, max_len: int) -> list[str]:
        """Split text into chunks at natural boundaries."""
        chunks: list[str] = []
        while text:
            if len(text) <= max_len:
                chunks.append(text)
                break
            split_at = text.rfind("\n\n", 0, max_len)
            if split_at == -1:
                split_at = text.rfind("\n", 0, max_len)
            if split_at == -1:
                split_at = text.rfind(" ", 0, max_len)
            if split_at == -1:
                split_at = max_len
            chunks.append(text[:split_at])
            text = text[split_at:].lstrip()
        return chunks

    async def _send_fallback_response(self, item: QueueItem, result: GenerationResult) -> None:
        try:
            channel = self.bot.get_channel(item.channel_id)
            if channel is None:
                channel = await self.bot.fetch_channel(item.channel_id)

            if result.error:
                await channel.send(embed=self._make_embed("Error", result.error, color=config.COLOR_ERROR))
            else:
                content = result.content.strip() or "..."
                if len(content) <= config.DISCORD_MSG_MAX:
                    await channel.send(content)
                else:
                    chunks = self._split_text(content, config.DISCORD_MSG_MAX)
                    for chunk in chunks:
                        await channel.send(chunk)

            # Append sources embed in fallback path too
            await self._send_sources_embed(channel, item)

            # Append switch notice / rate-limit warning in fallback path too
            notices = [n for n in (result.switch_notice, result.rate_warning) if n]
            if notices:
                await channel.send("\n".join(notices))
        except (discord.NotFound, discord.Forbidden, discord.HTTPException) as exc:
            logger.error("Failed to send fallback response: %s", exc)

    @staticmethod
    async def _safe_reply(message: discord.Message, content: str) -> None:
        """Reply to a message, handling the case where it was deleted."""
        try:
            await message.reply(content, mention_author=False)
        except discord.NotFound:
            # Message was deleted while we were processing
            try:
                await message.channel.send(content)
            except discord.HTTPException:
                pass
        except discord.HTTPException as exc:
            logger.warning("Failed to reply: %s", exc)

    # -----------------------------------------------------------------------
    # Chunk handler + streaming updater
    # -----------------------------------------------------------------------

    async def _handle_chunk(self, item: QueueItem, accumulated_text: str) -> None:
        # Chunks are collected by the queue manager. We only use the stream
        # tracker for typing indicators, so this is intentionally minimal.
        pass

    @tasks.loop(seconds=5.0)
    async def _stream_updater(self) -> None:
        """Fire Discord typing indicator for active streams + clean stale entries."""
        now = time.time()
        for msg_id, sd in list(self._active_streams.items()):
            if sd["finalized"]:
                self._active_streams.pop(msg_id, None)
                continue
            # Clean up stale streams (generation died mid-stream, etc.)
            if now - sd["started_at"] > config.STREAM_STALE_SECONDS:
                logger.warning("Cleaning up stale stream for message %s", msg_id)
                self._active_streams.pop(msg_id, None)
                try:
                    original_msg = sd["original_msg"]
                    await original_msg.remove_reaction(THINKING_EMOJI, self.bot.user)
                except discord.HTTPException:
                    pass
                continue
            try:
                await sd["channel"].typing()
            except (discord.HTTPException, AttributeError):
                pass

        # Clean expired auto-reply channels
        expired = [
            ch_id for ch_id, ts in self._active_channels.items()
            if (now - ts) > config.AUTO_REPLY_TIMEOUT
        ]
        for ch_id in expired:
            self._active_channels.pop(ch_id, None)

    @_stream_updater.before_loop
    async def _before_stream_updater(self) -> None:
        await self.bot.wait_until_ready()

    # -----------------------------------------------------------------------
    # Background maintenance
    # -----------------------------------------------------------------------

    @tasks.loop(minutes=10)
    async def _cleanup_threads(self) -> None:
        count = await self.db.cleanup_expired_threads()
        if count > 0:
            logger.info("Cleaned up %d expired thread messages", count)
        # Prune yesterday's daily usage rows (midnight reset)
        pruned = await self.db.prune_old_daily_usage()
        if pruned > 0:
            logger.info("Pruned %d old daily usage rows (midnight reset)", pruned)
        # Evict stale channel cache entries (no activity for 10+ min)
        self._channel_cache.evict_stale(600)
        # Size-based cleanup: if DB exceeds 5 GB, purge oldest ~1 GB
        await self.db.check_and_cleanup()

    @_cleanup_threads.before_loop
    async def _before_cleanup(self) -> None:
        await self.bot.wait_until_ready()

    # -----------------------------------------------------------------------
    # Embed helpers
    # -----------------------------------------------------------------------

    def _split_response_embeds(self, content: str, result: GenerationResult) -> list[discord.Embed]:
        max_len = config.EMBED_DESCRIPTION_MAX - 100
        chunks = self._split_text(content, max_len)

        embeds: list[discord.Embed] = []
        for i, chunk in enumerate(chunks):
            title = "Chatbot" if i == 0 else f"Chatbot ({i + 1}/{len(chunks)})"
            embed = self._make_embed(title, chunk, color=config.COLOR_PRIMARY)

            if i == len(chunks) - 1:
                footer_parts = []
                if result.duration_seconds > 0:
                    footer_parts.append(f"{result.duration_seconds:.1f}s")
                if result.eval_count > 0:
                    footer_parts.append(f"~{result.eval_count} tokens")
                if footer_parts:
                    embed.set_footer(text=" | ".join(footer_parts))
                embed.timestamp = datetime.now(timezone.utc)
            embeds.append(embed)

        return embeds if embeds else [self._make_embed("Chatbot", "...", color=config.COLOR_WARNING)]

    @staticmethod
    def _make_embed(title: str, description: str, color: int = config.COLOR_PRIMARY) -> discord.Embed:
        return discord.Embed(title=title, description=description, color=discord.Colour(color))


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(
        ChatBotCog(
            bot=bot,
            db=bot.db,
            queue_manager=bot.queue_manager,
        )
    )
