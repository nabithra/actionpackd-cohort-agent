import "dotenv/config";
import { readFileSync } from "fs";
import { join } from "path";
import { Bot, Context, GrammyError } from "grammy";
import { GoogleGenAI, type Content } from "@google/genai";
import { htmlToPlainText, markdownToTelegramHtml, splitMessage } from "./format";
import {
  listGeminiModels,
  TimeoutError,
  TOTAL_BUDGET_MS,
  tryModels,
  usableHistory,
} from "./gemini";

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
const GEMINI_MODELS = listGeminiModels();

// Load system prompt from repo root (works from both src/ and dist/)
const systemInstruction = readFileSync(
  join(__dirname, "..", "instructions.md"),
  "utf-8"
);

// --- Gemini client ---

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

// Chat history per Telegram chat id (last 20 messages max)
const chatHistory = new Map<number, Content[]>();

function getHistory(chatId: number): Content[] {
  return chatHistory.get(chatId) ?? [];
}

function setHistory(chatId: number, history: Content[]): void {
  // Keep pairs that start with the user. Never store a user turn that has no reply.
  chatHistory.set(chatId, usableHistory(history));
}

function resetHistory(chatId: number): void {
  chatHistory.delete(chatId);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
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

function is409Conflict(err: unknown): boolean {
  return err instanceof GrammyError && err.error_code === 409;
}

/** One Gemini call. No thinkingConfig — some models return 400 if it is set. */
function callGemini(model: string, contents: Content[], timeoutMs: number) {
  return withTimeout(
    ai.models.generateContent({
      model,
      contents,
      config: {
        systemInstruction,
        temperature: 0.3,
        // No Google Search grounding
      },
    }),
    timeoutMs
  );
}

async function askGemini(chatId: number, userText: string): Promise<string> {
  // Drop a broken history (starts with the model, or a user turn that failed).
  const history = usableHistory(getHistory(chatId));
  const contents: Content[] = [
    ...history,
    { role: "user", parts: [{ text: userText }] },
  ];

  const response = await tryModels(
    GEMINI_MODELS,
    (model, timeoutMs) => callGemini(model, contents, timeoutMs),
    TOTAL_BUDGET_MS
  );

  const reply = response.text?.trim() || "I don't have a response for that.";

  // Save only after a reply, so a failed call cannot leave two user turns in a row.
  setHistory(chatId, [
    ...contents,
    { role: "model", parts: [{ text: reply }] },
  ]);

  return reply;
}

// --- Telegram bot ---

const bot = new Bot(TELEGRAM_BOT_TOKEN);

// Same words as the greet in instructions.md. Blank line before the question.
const NINJA_GREETING =
  "Hi! I'm Ninja, an AI agent built at Actionpackd. In our cohort, you'll build agents like me.\n\nCan I ask a few quick questions to find the right fit for you?";

/** Clear the chat and send Ninja's intro once. The next message continues from here. */
async function greetFresh(ctx: Context): Promise<void> {
  if (!ctx.chat) return;
  const chatId = ctx.chat.id;
  resetHistory(chatId);
  await ctx.reply(NINJA_GREETING);
  // Gemini history must start with a user turn, so "/start" stands in for this command.
  setHistory(chatId, [
    { role: "user", parts: [{ text: "/start" }] },
    { role: "model", parts: [{ text: NINJA_GREETING }] },
  ]);
}

// /start and /reset — same greeting, no command list
bot.command("start", greetFresh);
bot.command("reset", greetFresh);

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

  // So /start and /reset show in Telegram's menu button
  try {
    await bot.api.setMyCommands([
      { command: "start", description: "Start fresh" },
      { command: "reset", description: "Clear our conversation" },
    ]);
  } catch (err) {
    console.warn("setMyCommands failed (continuing):", err);
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
