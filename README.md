# Actionpackd Cohort Agent

A Telegram bot for the Actionpackd cohort. It uses **grammY** (long polling) and **Google Gemini** to answer questions with a system prompt from `instructions.md`.

## Prerequisites

- Node.js 20+
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Local setup

1. Clone the repo and install dependencies:

   ```bash
   npm install
   ```

2. Copy the env example and fill in your values (never commit `.env`):

   ```bash
   cp .env.example .env
   ```

   | Variable | Required | Description |
   |----------|----------|-------------|
   | `TELEGRAM_BOT_TOKEN` | Yes | Token from BotFather |
   | `GEMINI_API_KEY` | Yes | Google AI Studio API key |
   | `GEMINI_MODEL` | No | Primary model. Defaults to `gemini-3.6-flash` |
   | `GEMINI_FALLBACK_MODEL` | No | Second model. Defaults to `gemini-flash-lite-latest`, then `gemini-3.5-flash-lite` |
   | `GEMINI_FALLBACK_MODELS` | No | Comma-separated fallbacks. If set, replaces the fallback list |

3. Edit `instructions.md` with the cohort system prompt you want.

4. Run in development (TypeScript via `tsx`):

   ```bash
   npm run dev
   ```

   Or build and run the compiled output:

   ```bash
   npm run build
   npm start
   ```

5. Open your bot in Telegram and send `/start`.

## Deploy on Railway

1. Push this repo to GitHub (or connect the local folder in Railway).

2. In [Railway](https://railway.app), create a new project → **Deploy from GitHub** (or empty project + link this repo).

3. Set these **Variables** in the Railway service (Settings → Variables):

   - `TELEGRAM_BOT_TOKEN`
   - `GEMINI_API_KEY`
   - `GEMINI_MODEL` (optional; defaults to `gemini-3.6-flash`)
   - `GEMINI_FALLBACK_MODEL` (optional; defaults to `gemini-flash-lite-latest`)
   - `GEMINI_FALLBACK_MODELS` (optional comma-separated fallbacks)

4. Build / start (Railway usually detects these from `package.json`):

   - **Build:** `npm run build`
   - **Start:** `npm start`

5. Deploy. The bot uses **long polling** (no public URL or webhook required). Railway will send `SIGTERM` on restarts; the bot calls `bot.stop()` for a clean shutdown.

6. Message your bot on Telegram to confirm it is online.

## Commands

| Command | Action |
|---------|--------|
| `/start` | Reset chat history and show a welcome message |
| `/reset` | Clear chat history for that conversation |

Only text messages are supported. Chat history is kept **in memory** (last 20 messages per chat) and is lost on restart.
