"""Discord Chatbot -- Entry point.

Initializes logging, database, Gemini client, queue manager,
loads the Discord cog, syncs slash commands, and runs the bot.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path regardless of how the script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent))

import logging
import traceback
from logging.handlers import RotatingFileHandler

import discord
from discord.ext import commands

import config
from core.database import Database
from core.gemini_client import GeminiClient
from core.queue_manager import QueueManager

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging() -> None:
    """Configure structured logging to both console and rotating file."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        config.LOG_DIR / "chatbot.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy libraries
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Bot subclass
# ---------------------------------------------------------------------------


class ChatBot(commands.Bot):
    """Custom Bot subclass that holds references to shared services."""

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.guilds = True

        super().__init__(
            command_prefix=config.BOT_PREFIX,
            intents=intents,
            help_command=None,
        )

        self.db = Database()
        self.gemini = GeminiClient(db=self.db) if config.GEMINI_API_KEY else None
        self.queue_manager = QueueManager(db=self.db, gemini=self.gemini)

    async def setup_hook(self) -> None:
        """Called before the bot connects to Discord. Initialize all services."""
        logger = logging.getLogger(__name__)

        # 1. Database
        await self.db.connect()
        logger.info("Database ready")

        # 2. Gemini client
        if self.gemini:
            await self.gemini.start()
            logger.info("Gemini client ready (API key configured)")
        else:
            logger.warning("No GEMINI_API_KEY set — bot will not be able to generate responses")

        # 3. Queue manager (single worker)
        await self.queue_manager.start()
        logger.info("Queue manager ready")

        # 4. Load cogs
        await self.load_extension("cogs.chatbot")
        logger.info("Chatbot cog loaded")

        # 5. Sync slash commands globally
        try:
            synced = await self.tree.sync()
            logger.info("Synced %d slash command(s)", len(synced))
        except discord.HTTPException as exc:
            logger.warning("Failed to sync slash commands: %s", exc)

    async def on_ready(self) -> None:
        logger = logging.getLogger(__name__)
        logger.info("Discord Chatbot is online as %s (ID: %s)", self.user, self.user.id if self.user else "?")
        logger.info("Connected to %d guild(s)", len(self.guilds))

        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name='@Discord Chatbot "Type Your Message"',
        )
        await self.change_presence(activity=activity)

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        """Global error handler for prefix commands."""
        if isinstance(error, commands.CommandNotFound):
            return
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.reply(f"Missing argument: `{error.param.name}`", mention_author=False)
            return
        if isinstance(error, commands.CommandOnCooldown):
            await ctx.reply(f"Slow down. Try again in {error.retry_after:.0f}s.", mention_author=False)
            return

        logger = logging.getLogger(__name__)
        logger.error("Command error in %s: %s", ctx.command, error)
        logger.debug("".join(traceback.format_exception(type(error), error, error.__traceback__)))

    async def close(self) -> None:
        """Graceful shutdown: stop all services."""
        logger = logging.getLogger(__name__)
        logger.info("Shutting down Discord Chatbot...")

        await self.queue_manager.stop()
        if self.gemini:
            await self.gemini.close()
        await self.db.close()

        await super().close()
        logger.info("Discord Chatbot shutdown complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    if not config.DISCORD_TOKEN:
        logger.error(
            "DISCORD_TOKEN not set. Copy .env.example to .env and add your bot token."
        )
        sys.exit(1)

    if not config.GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set — bot will start but cannot generate responses.")

    bot = ChatBot()

    try:
        bot.run(config.DISCORD_TOKEN, log_handler=None)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")


if __name__ == "__main__":
    main()
