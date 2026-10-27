/**
 * Gemini writes Markdown (**bold**, *italic*, `code`, "- " lists).
 * Telegram shows those stars unless we convert them to HTML first.
 *
 * Pipeline: escape & < > → bullets → code → bold → italic.
 * Long replies are split under 4000 characters without cutting a tag in half.
 */

const MAX_LEN = 4000;

/** & first, otherwise "&lt;" would become "&amp;lt;". */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/**
 * Turn one Gemini reply into HTML that Telegram can render (parse_mode HTML).
 */
export function markdownToTelegramHtml(markdown: string): string {
  let text = escapeHtml(markdown.replace(/\r\n/g, "\n"));

  // List lines. Gemini uses both "- item" and "* item".
  text = text.replace(/^(\s*)[-*] /gm, "$1• ");

  // Hold `code` aside so * and _ inside it stay literal.
  const codeSpans: string[] = [];
  text = text.replace(/`([^`\n]+)`/g, (_match, code: string) => {
    codeSpans.push(code);
    // Stand-in so bold/italic below do not touch the code. Swapped back later.
    return `\uE000${codeSpans.length - 1}\uE001`;
  });

  // Longer markers first, so **bold** is not read as italic.
  text = text.replace(/\*\*([^\n]+?)\*\*/g, (_match, inner: string) => {
    return `<b>${inner}</b>`;
  });
  text = text.replace(/__([^\n]+?)__/g, (_match, inner: string) => {
    return `<b>${inner}</b>`;
  });
  // Unpaired ** or __ would show up raw. Drop them.
  text = text.replace(/\*\*/g, "");
  text = text.replace(/__/g, "");

  // *italic* — the words must touch the stars, so "2 * 3" stays as-is.
  text = text.replace(
    /\*(\S(?:[^*\n<]*\S)?)\*/g,
    (_match, inner: string) => `<i>${inner}</i>`
  );
  // _italic_ only on a boundary, so snake_case is left alone.
  text = text.replace(
    /(^|[\s(>])_(\S(?:[^_\n<]*\S)?)_(?=[\s).,!?:;<"']|$)/g,
    (_match, before: string, inner: string) => `${before}<i>${inner}</i>`
  );

  // Unmatched * sits against a word ("*hello" or "hello*"). Drop it.
  // A spaced star ("2 * 3") is not a marker, so leave that one.
  text = text.replace(/(\S)\*/g, "$1");
  text = text.replace(/\*(\S)/g, "$1");

  text = text.replace(/\uE000(\d+)\uE001/g, (_match, id: string) => {
    const code = codeSpans[Number(id)] ?? "";
    return `<code>${code}</code>`;
  });

  return text;
}

/**
 * Readable text for a resend. Tags come off, entities become characters,
 * and leftover Markdown markers are removed.
 */
export function htmlToPlainText(html: string): string {
  let text = html.replace(/<[^>]*>/g, "");
  // A bad cut can leave a half tag such as "<b" or "</code" at the end.
  text = text.replace(/<\/?(?:code|cod|co|c|b|i)?$/g, "");
  text = text
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&");
  text = text.replace(/\*\*/g, "");
  text = text.replace(/__/g, "");
  text = text.replace(/`/g, "");
  // Same rule as above: drop *hello / hello*, keep "2 * 3".
  text = text.replace(/(\S)\*/g, "$1");
  text = text.replace(/\*(\S)/g, "$1");
  text = text.replace(/(^|\s)_(\S(?:[^_\n]*\S)?)_(?=\s|$)/g, "$1$2");
  return text;
}

/** Tags opened in this piece that do not yet have a closing tag. */
function unclosedTags(html: string): string[] {
  const stack: string[] = [];
  const re = /<(\/?)(b|i|code)>/g;
  let match: RegExpExecArray | null;
  while ((match = re.exec(html)) !== null) {
    const isClose = match[1] === "/";
    const name = match[2];
    if (!isClose) {
      stack.push(name);
    } else {
      const at = stack.lastIndexOf(name);
      if (at !== -1) stack.splice(at, 1);
    }
  }
  return stack;
}

function closeTags(tags: string[]): string {
  return tags
    .slice()
    .reverse()
    .map((tag) => `</${tag}>`)
    .join("");
}

function openTags(tags: string[]): string {
  return tags.map((tag) => `<${tag}>`).join("");
}

/**
 * Where to split. Prefer a newline near the limit.
 * Never end in the middle of <tag> or &amp; / &lt; / &gt;, or an emoji.
 */
function findCut(text: string, maxLen: number): number {
  let cut = text.lastIndexOf("\n", maxLen);
  if (cut < maxLen / 2) cut = maxLen;

  // Look at the head only (characters before `cut`).
  const headEnd = cut - 1;
  if (headEnd >= 0) {
    const lt = text.lastIndexOf("<", headEnd);
    const gt = text.lastIndexOf(">", headEnd);
    if (lt > gt && lt > 0) cut = lt;

    const amp = text.lastIndexOf("&", cut - 1);
    const semi = text.lastIndexOf(";", cut - 1);
    if (amp > semi && amp > 0 && cut - 1 - amp <= 8) cut = amp;
  }

  if (cut > 1) {
    const prev = text.charCodeAt(cut - 1);
    if (prev >= 0xd800 && prev <= 0xdbff) cut -= 1;
  }

  if (cut <= 0 || cut > text.length) cut = Math.min(maxLen, text.length);
  return cut;
}

/**
 * Split a reply into Telegram-sized chunks (4000 characters).
 * If a tag is still open at the cut, close it here and reopen it
 * on the next chunk so each message parses on its own.
 */
export function splitMessage(text: string, maxLen = MAX_LEN): string[] {
  if (text.length <= maxLen) return [text];

  const chunks: string[] = [];
  let remaining = text;

  while (remaining.length > maxLen) {
    const cut = findCut(remaining, maxLen);
    const head = remaining.slice(0, cut);
    const tail = remaining.slice(cut).trimStart();
    const open = unclosedTags(head);
    const repairedTail = openTags(open) + tail;

    // Tag repair must still shrink the text, or the loop would never finish.
    if (open.length > 0 && repairedTail.length < remaining.length) {
      chunks.push(head + closeTags(open));
      remaining = repairedTail;
    } else {
      chunks.push(head);
      remaining = tail.length < remaining.length ? tail : remaining.slice(cut);
    }
  }

  if (remaining.length > 0) chunks.push(remaining);
  return chunks;
}
