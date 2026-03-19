# Discord Chatbot

A Discord bot that lets server members ask questions to Gemini LLMs. Mention the bot or use the `!chat` command to queue a question. The chatbot processes requests sequentially, streams responses into Discord embeds, and supports multi-turn conversation context through reply chains.

## Features

- **Mention or command:** `@Bot is this true?` or `!chat explain quantum computing`
- **Multi-turn threads:** Reply to a message and mention the bot to include context
- **Queue management:** SQLite-backed queue with crash recovery and per-user limits
- **Streaming responses:** Progressive embed updates while the model generates
- **Auto-start:** Windows startup script for boot-time startup
- **Metrics and logging:** Request logging, usage tracking, rotating log files

## Prerequisites

- Python 3.10+
- A Discord bot token ([create one here](https://discord.com/developers/applications))
- A Gemini API key

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/discord-chatbot.git
cd discord-chatbot

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
copy .env.example .env
# Edit .env and add your DISCORD_TOKEN and GEMINI_API_KEY

# 5. Run
python bot.py
```

## Configuration

All settings live in `.env`. See `.env.example` for the full list with defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `DISCORD_TOKEN` | (required) | Your Discord bot token |
| `GEMINI_API_KEY` | (required) | Google Gemini API key |
| `DEFAULT_MODEL` | `gemini-3.1-flash-lite-preview` | Default model |
| `MAX_QUEUE_SIZE` | `20` | Maximum queued requests |
| `MAX_USER_PENDING` | `3` | Max pending requests per user |
| `USER_COOLDOWN` | `0` | Seconds between requests per user |
| `MAX_THREAD_DEPTH` | `25` | Max messages in conversation context |
| `THREAD_EXPIRY_MINUTES` | `120` | Thread context auto-expires after this |

## Commands

| Command | Description |
|---------|-------------|
| `@Bot <question>` | Ask a question via mention |
| `!chat <question>` | Ask a question via prefix command |
| `/chat` | Slash command |
| `/chat-stats` | Usage dashboard |
| `/chat-config` | Per-server configuration |

## Auto-Start on Windows

Drop a shortcut to `start-chatbot.bat` in your Startup folder (`shell:startup`).

## Project Structure

```
discord-chatbot/
  bot.py                    # Entry point
  config.py                 # Centralized configuration
  .env.example
  .gitignore
  start-chatbot.bat          # Windows startup script
  start-chatbot.ps1          # Watchdog script
  cogs/
    chatbot.py               # Discord command/mention handler
  core/
    database.py             # Async SQLite layer
    models.py               # Data classes
    gemini_client.py         # Gemini API client
    queue_manager.py         # Single-worker async queue
    web_search.py            # Web search integration
  db/
    schema.sql              # Database schema
  logs/                     # Rotating log files (gitignored)
```

## Architecture

```
Discord API <--> ChatBotCog (discord.py)
                    |
            QueueManager (asyncio.Queue + SQLite persistence)
                    |
             GeminiClient (aiohttp)
                    |
              Gemini API
```

Single-worker design: one generation at a time to stay within rate limits.

## License

MIT
