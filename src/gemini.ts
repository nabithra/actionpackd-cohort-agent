import { ApiError, type Content } from "@google/genai";

/**
 * How we call Gemini.
 * We never send thinkingConfig: several flash models answer 400
 * "invalid argument" and that message does not mention thinking.
 * Each model gets one 15s attempt and, only for a temporary failure,
 * one quick retry. Any error moves on to the next model.
 * The whole chain stops at about 40s.
 */

export const TOTAL_BUDGET_MS = 40_000;
const ATTEMPT_TIMEOUT_MS = 15_000;
const MIN_ATTEMPT_MS = 1_000;
const RETRY_PAUSE_MS = 1_000;
const MAX_HISTORY = 20;

const DEFAULT_MODEL = "gemini-3.6-flash";
const DEFAULT_FALLBACK = "gemini-flash-lite-latest";
const LAST_FALLBACK = "gemini-3.5-flash-lite";

export class TimeoutError extends Error {
  constructor(ms: number) {
    super(`Gemini call timed out after ${ms}ms`);
    this.name = "TimeoutError";
  }
}

function envOr(name: string, fallback: string): string {
  const value = process.env[name]?.trim();
  return value ? value : fallback;
}

/** Primary model, then fallbacks. GEMINI_FALLBACK_MODELS replaces the fallbacks. */
export function listGeminiModels(): string[] {
  const primary = envOr("GEMINI_MODEL", DEFAULT_MODEL);

  const custom = process.env.GEMINI_FALLBACK_MODELS?.trim();
  const fallbacks = custom
    ? custom.split(",").map((name) => name.trim()).filter(Boolean)
    : [envOr("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK), LAST_FALLBACK];

  const models: string[] = [];
  for (const name of [primary, ...fallbacks]) {
    if (!models.includes(name)) models.push(name);
  }
  return models;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** Status string for logs: HTTP code, "timeout", or "network". */
export function geminiFailureStatus(err: unknown): string {
  if (err instanceof TimeoutError) return "timeout";
  if (err instanceof ApiError) return String(err.status);
  const msg = errorMessage(err).toLowerCase();
  if (
    msg.includes("fetch failed") ||
    msg.includes("timeout") ||
    msg.includes("timed out") ||
    msg.includes("econnreset") ||
    msg.includes("enotfound") ||
    msg.includes("network") ||
    msg.includes("socket")
  ) {
    return "network";
  }
  return "error";
}

/** Temporary failures get one retry. 400 and 404 do not — we switch models. */
export function isRetryableGeminiError(err: unknown): boolean {
  if (err instanceof TimeoutError) return true;
  if (err instanceof ApiError) {
    return err.status === 503 || err.status === 429 || err.status === 500;
  }
  return geminiFailureStatus(err) === "network";
}

function timeLeft(deadline: number): number {
  return deadline - Date.now();
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Call one model. `send` must reject if it is still running after timeoutMs.
 * Logs "model (status)" on every failure.
 */
async function tryOneModel<T>(
  model: string,
  deadline: number,
  send: (model: string, timeoutMs: number) => Promise<T>
): Promise<T> {
  const once = () => {
    const timeoutMs = Math.min(ATTEMPT_TIMEOUT_MS, timeLeft(deadline));
    if (timeoutMs < MIN_ATTEMPT_MS) {
      throw new Error("time budget used up");
    }
    return send(model, timeoutMs);
  };

  try {
    return await once();
  } catch (err) {
    const status = geminiFailureStatus(err);
    const canRetry =
      isRetryableGeminiError(err) &&
      timeLeft(deadline) >= MIN_ATTEMPT_MS + RETRY_PAUSE_MS;
    if (!canRetry) {
      console.warn(`Gemini ${model} failed (${status})`);
      throw err;
    }
    console.warn(`Gemini ${model} failed (${status}); retrying once`);
    await sleep(RETRY_PAUSE_MS);
  }

  try {
    return await once();
  } catch (err) {
    console.warn(`Gemini ${model} failed (${geminiFailureStatus(err)})`);
    throw err;
  }
}

/** First model that answers. Any error, including 400, tries the next model. */
export async function tryModels<T>(
  models: string[],
  send: (model: string, timeoutMs: number) => Promise<T>,
  budgetMs = TOTAL_BUDGET_MS
): Promise<T> {
  const deadline = Date.now() + budgetMs;
  let lastErr: unknown;

  for (const model of models) {
    if (timeLeft(deadline) < MIN_ATTEMPT_MS) break;
    try {
      return await tryOneModel(model, deadline, send);
    } catch (err) {
      lastErr = err;
    }
  }

  throw lastErr ?? new Error("Every Gemini model failed");
}

/**
 * History Gemini will accept: starts with user, then user/model/user/model.
 * A trailing user turn has no reply (the call failed), so it is dropped.
 */
export function usableHistory(history: Content[]): Content[] {
  const turns: Content[] = [];

  for (const turn of history) {
    const role = turn.role === "model" ? "model" : "user";
    const text = (turn.parts ?? [])
      .map((part) => part.text ?? "")
      .join("")
      .trim();
    if (!text) continue;

    const next: Content = { role, parts: [{ text }] };
    const previous = turns[turns.length - 1];

    if (!previous) {
      if (role === "user") turns.push(next);
      continue;
    }

    if (previous.role === role) {
      // Two user turns in a row: the earlier one never got a reply.
      if (role === "user") turns[turns.length - 1] = next;
      continue;
    }

    turns.push(next);
  }

  if (turns.length > 0 && turns[turns.length - 1].role === "user") {
    turns.pop();
  }

  const trimmed = turns.slice(-MAX_HISTORY);
  while (trimmed.length > 0 && trimmed[0].role !== "user") {
    trimmed.shift();
  }
  return trimmed;
}
