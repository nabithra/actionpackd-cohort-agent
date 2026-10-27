import "dotenv/config";
import { readFileSync } from "fs";
import { join } from "path";
import { Bot, Context, GrammyError } from "grammy";
import { ApiError, GoogleGenAI, type Content } from "@google/genai";
import { htmlToPlainText, markdownToTelegramHtml, splitMessage } from "./format";

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
const GEMINI_TIMEOUT_MS = 20_000;

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

class TimeoutError extends Error {
  constructor(ms: number) {
    super(`Gemini call timed out after ${ms}ms`);
    this.name = "TimeoutError";
  }
}

/** Reject if the promise takes longer than ms. */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new TimeoutError(ms)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** True for transient failures worth retrying (HTTP + network + timeout). */
function isRetryableGeminiError(err: unknown): boolean {
  if (err instanceof TimeoutError) return true;
  if (err instanceof ApiError) {
    return err.status === 503 || err.status === 429 || err.status === 500;
  }
  const msg = errorMessage(err).toLowerCase();
  return (
    msg.includes("fetch failed") ||
    msg.includes("timeout") ||
    msg.includes("timed out") ||
    msg.includes("econnreset") ||
    msg.includes("enotfound") ||
    msg.includes("network") ||
    msg.includes("socket")
  );
}

/** Model returned 400 about thinking — drop thinkingConfig and retry. */
function isThinkingConfigRejected(err: unknown): boolean {
  if (!(err instanceof ApiError) || err.status !== 400) return false;
  const msg = errorMessage(err).toLowerCase();
  return msg.includes("thinking") || msg.includes("budget");
}

function is409Conflict(err: unknown): boolean {
  return err instanceof GrammyError && err.error_code === 409;
}

async function callGemini(
  model: string,
  contents: Content[],
  withThinkingOff: boolean
) {
  return withTimeout(
    ai.models.generateContent({
      model,
      contents,
      config: {
        systemInstruction,
        temperature: 0.3,
        // No Google Search grounding
        ...(withThinkingOff
          ? { thinkingConfig: { thinkingBudget: 0 } }
          : {}),
      },
    }),
    GEMINI_TIMEOUT_MS
  );
}

/** One Gemini call with 20s timeout; skip thinkingConfig if the model rejects it. */
async function generateOnce(model: string, contents: Content[]) {
  try {
    // Turn off thinking for speed
    return await callGemini(model, contents, true);
  } catch (err) {
    if (!isThinkingConfigRejected(err)) throw err;
    console.warn(`Model ${model} rejected thinkingConfig; retrying without it`);
    return await callGemini(model, contents, false);
  }
}

/** Call Gemini; retry transient errors up to 3 times (1s, 3s, 6s). */
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
      const label =
        err instanceof ApiError
          ? String(err.status)
          : err instanceof TimeoutError
            ? "timeout"
            : "network";
      console.warn(
        `Gemini ${label} on ${model}; retrying in ${wait}ms (${attempt + 1}/3)…`
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

/** True when Telegram rejected parse_mode HTML ("can't parse entities"). */
function isHtmlParseError(err: unknown): boolean {
  if (!(err instanceof GrammyError) || err.error_code !== 400) return false;
  const description = `${err.description} ${err.message}`.toLowerCase();
  return description.includes("parse entities");
}

/** Send one chunk as HTML. If Telegram can't parse it, resend plain text. */
async function replyHtmlOrPlain(ctx: Context, html: string): Promise<void> {
  try {
    await ctx.reply(html, { parse_mode: "HTML" });
  } catch (err) {
    if (!isHtmlParseError(err)) throw err;
    console.warn("Telegram rejected the HTML; sending this part as plain text");
    const plain = htmlToPlainText(html).trim();
    if (plain) await ctx.reply(plain);
  }
}

// Text messages → Gemini
bot.on("message:text", async (ctx) => {
  const chatId = ctx.chat.id;
  const userText = ctx.message.text;

  // Keep "typing…" alive (Telegram expires it after a few seconds)
  const sendTyping = () =>
    ctx.replyWithChatAction("typing").catch(() => {
      /* ignore */
    });
  await sendTyping();
  const typingTimer = setInterval(sendTyping, 4000);

  try {
    const reply = await askGemini(chatId, userText);
    // History keeps Gemini's Markdown. Only the Telegram send is HTML.
    const chunks = splitMessage(markdownToTelegramHtml(reply)).filter((chunk) =>
      chunk.trim()
    );

    if (chunks.length === 0) {
      await ctx.reply("I don't have a response for that.");
    } else {
      for (const chunk of chunks) {
        await replyHtmlOrPlain(ctx, chunk);
      }
    }
  } catch (err) {
    // Log the error (never log secrets) and keep the bot alive
    console.error("Gemini error:", err);
    await ctx.reply(
      "Sorry, something went wrong. Please try again in a minute."
    );
  } finally {
    clearInterval(typingTimer);
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

// Log handler errors without crashing
bot.catch((err) => {
  if (is409Conflict(err.error)) {
    console.warn(
      "Warning: 409 Conflict — another bot instance may be polling."
    );
    return;
  }
  console.error("Bot error:", err);
});

// Clean shutdown for Railway restarts
process.once("SIGINT", () => {
  bot.stop().finally(() => process.exit(0));
});
process.once("SIGTERM", () => {
  bot.stop().finally(() => process.exit(0));
});

async function main() {
  // Clear any leftover webhook so long polling can run
  try {
    await bot.api.deleteWebhook({ drop_pending_updates: false });
  } catch (err) {
    console.warn("deleteWebhook failed (continuing):", err);
  }

  console.log("Actionpackd Cohort Agent is starting (long polling)…");

  // Keep the process alive; on 409 wait and start again
  while (true) {
    try {
      await bot.start();
      // bot.start() resolves after bot.stop() — exit the loop
      break;
    } catch (err) {
      if (is409Conflict(err)) {
        console.warn(
          "Warning: 409 Conflict — another getUpdates is running. Retrying in 5s…"
        );
        await sleep(5000);
        continue;
      }
      console.error("bot.start() error:", err);
      await sleep(5000);
    }
  }
}

main().catch((err) => {
  // Never exit here — log and keep the event loop from ending abruptly
  console.error("Startup error:", err);
});
