import "dotenv/config";
import { readFileSync } from "fs";
import { join } from "path";
import { Bot, GrammyError } from "grammy";
import { ApiError, GoogleGenAI, type Content } from "@google/genai";

// --- Config (fail fast if required env vars are missing) ---

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value;
}

const TELEGRAM_BOT_TOKEN = requireEnv("TELEGRAM_BOT_TOKEN");
const GEMINI_API_KEY = requireEnv("GEMINI_API_KEY");
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-flash-latest";
const GEMINI_FALLBACK_MODEL =
  process.env.GEMINI_FALLBACK_MODEL || "gemini-2.5-flash";

// Load system prompt from repo root (works from both src/ and dist/)
const systemInstruction = readFileSync(
  join(__dirname, "..", "instructions.md"),
  "utf-8"
);

// --- Gemini client ---

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

// Chat history per Telegram chat id (last 20 messages max)
const chatHistory = new Map<number, Content[]>();
const MAX_HISTORY = 20;

// Wait times before each of the 3 retries
const RETRY_WAITS_MS = [1000, 3000, 6000];

function getHistory(chatId: number): Content[] {
  return chatHistory.get(chatId) ?? [];
}

function setHistory(chatId: number, history: Content[]): void {
  // Keep only the last MAX_HISTORY messages
  chatHistory.set(chatId, history.slice(-MAX_HISTORY));
}

function resetHistory(chatId: number): void {
  chatHistory.delete(chatId);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** True for transient Gemini failures worth retrying. */
function isRetryableGeminiError(err: unknown): boolean {
  const status = err instanceof ApiError ? err.status : undefined;
  return status === 503 || status === 429 || status === 500;
}

function is409Conflict(err: unknown): boolean {
  return err instanceof GrammyError && err.error_code === 409;
}

/** Split a long reply into chunks under Telegram's safe limit. */
function splitMessage(text: string, maxLen = 4000): string[] {
  if (text.length <= maxLen) return [text];

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > maxLen) {
    // Prefer splitting on a newline near the limit
    let cut = remaining.lastIndexOf("\n", maxLen);
    if (cut < maxLen / 2) cut = maxLen;
    chunks.push(remaining.slice(0, cut));
    remaining = remaining.slice(cut).trimStart();
  }

  if (remaining.length > 0) chunks.push(remaining);
  return chunks;
}

async function generateOnce(model: string, contents: Content[]) {
  return ai.models.generateContent({
    model,
    contents,
    config: {
      systemInstruction,
      temperature: 0.3,
      // No Google Search grounding
    },
  });
}

/** Call Gemini; on 503/429/500 retry up to 3 times (1s, 3s, 6s). */
async function generateWithRetries(model: string, contents: Content[]) {
  let lastErr: unknown;

  for (let attempt = 0; attempt <= RETRY_WAITS_MS.length; attempt++) {
    try {
      return await generateOnce(model, contents);
    } catch (err) {
      lastErr = err;
      const canRetry =
        isRetryableGeminiError(err) && attempt < RETRY_WAITS_MS.length;
      if (!canRetry) throw err;

      const wait = RETRY_WAITS_MS[attempt];
      console.warn(
        `Gemini error ${err instanceof ApiError ? err.status : "?"} on ${model}; retrying in ${wait}ms (${attempt + 1}/3)…`
      );
      await sleep(wait);
    }
  }

  throw lastErr;
}

async function askGemini(chatId: number, userText: string): Promise<string> {
  const history = getHistory(chatId);

  // Add the new user message to the contents we send
  const contents: Content[] = [
    ...history,
    { role: "user", parts: [{ text: userText }] },
  ];

  let response;
  try {
    response = await generateWithRetries(GEMINI_MODEL, contents);
  } catch (err) {
    // Still failing after retries → one shot with the fallback model
    if (!isRetryableGeminiError(err)) throw err;
    console.warn(
      `Primary model failed; trying fallback ${GEMINI_FALLBACK_MODEL}…`
    );
    response = await generateOnce(GEMINI_FALLBACK_MODEL, contents);
  }

  const reply = response.text?.trim() || "I don't have a response for that.";

  // Save user + model turns, then trim to last 20
  setHistory(chatId, [
    ...contents,
    { role: "model", parts: [{ text: reply }] },
  ]);

  return reply;
}

// --- Telegram bot ---

const bot = new Bot(TELEGRAM_BOT_TOKEN);

// /start — reset history and welcome the user
bot.command("start", async (ctx) => {
  const chatId = ctx.chat.id;
  resetHistory(chatId);

  await ctx.reply(
    "Hi! I'm the Actionpackd Cohort Agent 👋\n\n" +
      "Ask me anything about the cohort — I'll do my best to help.\n\n" +
      "Commands:\n" +
      "/start — start fresh\n" +
      "/reset — clear our conversation"
  );
});

// /reset — clear history and confirm
bot.command("reset", async (ctx) => {
  resetHistory(ctx.chat.id);
  await ctx.reply("Conversation cleared. What's next?");
});

// Text messages → Gemini
bot.on("message:text", async (ctx) => {
  const chatId = ctx.chat.id;
  const userText = ctx.message.text;

  // Show "typing…" while we wait for Gemini
  await ctx.replyWithChatAction("typing");

  try {
    const reply = await askGemini(chatId, userText);
    const chunks = splitMessage(reply);

    for (const chunk of chunks) {
      await ctx.reply(chunk);
    }
  } catch (err) {
    // Log the error (never log secrets) and keep the bot alive
    console.error("Gemini error:", err);
    await ctx.reply(
      "Sorry, something went wrong. Please try again in a minute."
    );
  }
});

// Non-text messages (photo, voice, stickers, etc.)
bot.on("message", async (ctx) => {
  // Text messages are handled above; this catches everything else
  if (ctx.message.text) return;

  await ctx.reply(
    "I can only read text messages for now. Please send your question as text!"
  );
});

// Catch unexpected errors so the bot never crashes
bot.catch((err) => {
  if (is409Conflict(err.error)) {
    console.warn(
      "Warning: 409 Conflict — another bot instance may be polling. Stop other instances."
    );
    return;
  }
  console.error("Bot error:", err);
});

// Clean shutdown for Railway restarts
process.once("SIGINT", () => bot.stop());
process.once("SIGTERM", () => bot.stop());

async function main() {
  // Clear any leftover webhook so long polling can run
  await bot.api.deleteWebhook({ drop_pending_updates: false });

  console.log("Actionpackd Cohort Agent is starting (long polling)…");

  try {
    await bot.start();
  } catch (err) {
    // getUpdates 409 should warn, not crash the process
    if (is409Conflict(err)) {
      console.warn(
        "Warning: 409 Conflict — another getUpdates is running. Stop other instances."
      );
      return;
    }
    throw err;
  }
}

main().catch((err) => {
  console.error("Fatal startup error:", err);
  process.exit(1);
});
