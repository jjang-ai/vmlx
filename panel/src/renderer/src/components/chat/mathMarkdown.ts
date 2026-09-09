import { renderToString } from 'katex'
import { marked } from 'marked'

// Preserve both CommonMark backtick fences and GFM tilde fences before any
// TeX normalization. Model-generated code frequently uses either spelling;
// rewriting `\times`, `$...$`, or `*` inside a tilde fence corrupts source.
// The unclosed-to-end-of-string alternatives mirror
// LITERAL_MARKDOWN_PROTECTED_RE below and are just as load-bearing here: a
// fence is unclosed for the whole time it streams (and permanently when the
// model ends its reply inside the block), and without them transformMath
// injected KaTeX wrapper HTML INTO the code body, which marked then rendered
// as literal escaped text inside the block.
const CODE_RE = /```[\s\S]*?```|~~~[\s\S]*?~~~|```[\s\S]*$|~~~[\s\S]*$|`[^`\n]*`/g
// The unclosed-fence alternatives are NOT redundant with the closed ones, and
// their position matters. A fence is unclosed for the whole time it streams, so
// without them every code block renders `&quot;` for `"` until the closing
// fence arrives — and permanently when a model ends its reply inside the block
// or is cut off by max_tokens, which is the common case for a trailing code
// answer.
//
// They must sit AFTER the closed forms (so a complete fence still wins) but
// BEFORE the inline-backtick form, which would otherwise match the first two
// backticks of an unclosed ``` as an empty inline span and leave the body
// exposed to escaping.
// Math delimiters deliberately get NO unclosed form, unlike fences above.
//
// An unclosed `$$`/`\[`/`$` does escape `<` to `&lt;` while it streams, and
// extending the same to-end-of-string trick does fix that — measured. But it
// then swallows the remainder of any message containing a stray opener, so
// `cost $$5 then <result>` stops escaping `<result>`, and the tag is dropped
// from the rendered output instead of shown as text. That is the exact failure
// escaping exists to prevent, and a lone `$` ("costs $5") is ordinary prose.
//
// The fence case is different and is fixed above: a fence body is rendered as
// literal code, so swallowing to end-of-message is the correct reading of an
// unterminated block, and the corruption there was proven live and PERMANENT
// (a model ending its reply inside the block). The math case is transient —
// the closing delimiter almost always arrives. Trading a rare permanent
// content drop for a rare transient artifact is a bad trade, so it is declined.
const LITERAL_MARKDOWN_PROTECTED_RE =
  /```[\s\S]*?```|~~~[\s\S]*?~~~|```[\s\S]*$|~~~[\s\S]*$|`[^`\n]*`|\\\[[\s\S]*?\\\]|\\\([^\n]*?\\\)|\$\$[\s\S]*?\$\$/g

const KATEX_OPTIONS = {
  output: 'html' as const,
  // A failed parse must take the escaped text fallback below. KaTeX's
  // throwOnError=false path emits a visible `.katex-error` node, which makes
  // malformed model scratch math look like renderer corruption.
  throwOnError: true,
  strict: 'ignore' as const,
  trust: false,
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function encodeMathSourceCodepoints(text: string): string {
  // Hex code points survive every later Markdown/TeX normalization pass
  // byte-for-byte, including lone surrogates, backslashes, quotes, and `*`.
  // The live attestor decodes this inert value from the completed DOM.
  return Array.from(text, (char) => char.codePointAt(0)!.toString(16)).join('-')
}

function looksLikeSingleDollarMath(text: string): boolean {
  const trimmed = text.trim()
  if (!trimmed || trimmed !== text) return false
  // Two currency amounts can otherwise be mistaken for one math span while
  // streaming (for example "$5<$10" temporarily matches "$5<$").
  //
  // That case is caught by the TRAILING operator, and rejecting a LEADING one
  // as well was too broad: models routinely continue a derivation onto its own
  // line as `$= 340 + 51$`, which is unambiguous math because no currency
  // amount ever begins with a relational operator. Live-reproduced on
  // gemma-4-E4B, which rendered `17 x 23 = 17 x (20 + 3)` correctly through
  // KaTeX and then showed the next three lines as literal `$= ...$` text.
  //
  // Leading `+ - * /` stay rejected: a signed currency amount ("$-5") is real,
  // so those remain genuinely ambiguous. A leading `= < >` is not.
  // Apply the incomplete-expression check to Unicode operators too. Otherwise
  // currency such as "$245.00 = **−$24.50**" pairs the two amount prefixes as
  // math and consumes the emphasis. Complete expressions still qualify below.
  if (/^[+\-*/]/.test(trimmed) || /[+\-*/=<>−×÷±≤≥≠≈]$/.test(trimmed)) return false
  if (/^\d+(?:[.,]\d{2})?$/.test(trimmed)) return false
  // A missing single-dollar closer must never let a later currency amount
  // terminate a prose-sized span. Strip TeX commands, then look at the
  // remaining alphabetic runs: real prose words give the span away, while a
  // short run of letters is almost always a product of variables.
  //
  // This rejected on runs of TWO or more, and its own comment claimed that
  // "keeps `$E=mc^2$` valid". MEASURED against the real predicate — it does
  // not. `mc` is a two-letter run and not a math function name, so the single
  // most recognisable inline formula rendered as literal text, as did
  // `$F = ma$` and `$ab = cd$`. Live-reproduced in the app: the model emitted
  // `$E = mc^2$` and the chat showed the dollar signs.
  //
  // Three is the right threshold. English prose words in these false-positive
  // spans are 3+ ("This", "seems", "the"), so the example the comment cites,
  // `$47 \times 19 ... This seems to be ... $43`, is still rejected. Two-letter
  // runs cannot sneak prose through on their own either, because a span still
  // has to show math STRUCTURE below — a TeX command, one of `{}_^=<>`, or an
  // infix operator — before this returns true. `$x is 5$` has none and is
  // still rejected.
  const lexicalView = trimmed
    .replace(/\\text\s*\{[^{}]*\}/g, '')
    .replace(/\\[A-Za-z]+/g, '')
  const proseWords = lexicalView.match(/[A-Za-z]{3,}/g) || []
  if (
    proseWords.some(
      (word) =>
        !/^(?:sin|cos|tan|cot|sec|csc|log|ln|exp|lim|max|min|mod|gcd|lcm|det)$/i.test(
          word,
        ),
    )
  ) {
    return false
  }
  if (/\\[A-Za-z]+/.test(trimmed)) return true
  if (/[{}_^=<>]/.test(trimmed)) return true
  if (/(?:[\dA-Za-z])\s*[+\-*/]\s*(?:[\dA-Za-z])/.test(trimmed)) return true
  if (/^[A-Za-z]$/.test(trimmed)) return true
  return false
}

function isEscapedAt(text: string, index: number): boolean {
  let precedingBackslashes = 0
  for (let cursor = index - 1; cursor >= 0 && text[cursor] === '\\'; cursor--) {
    precedingBackslashes += 1
  }
  return precedingBackslashes % 2 === 1
}

function findNextSingleDollar(text: string, start: number): number {
  for (let index = start; index < text.length; index++) {
    if (text[index] !== '$' || isEscapedAt(text, index)) continue
    return index
  }
  return -1
}

/**
 * A `$` that is welded to a preceding word AND followed by a digit is a
 * CURRENCY SYMBOL, not a math opener.
 *
 * `US$5 = CA$7` otherwise offers `5 = CA` as a candidate body, and that body is
 * accepted as math: `CA` is a two-letter run so it is not counted as prose, and
 * the `=` then satisfies the structure test. The line rendered as literal `US`,
 * math `$5 = CA$`, literal `7`.
 *
 * Counting two-letter runs as prose again is the wrong lever — that is what made
 * `$E=mc^2$` and `$F = ma$` render literally, and it also rejects legitimate
 * `$AB = CD$` geometry.
 *
 * But looking only BACKWARDS is too blunt, and shipped as a worse regression
 * than the bug it fixed. Models weld inline math to the preceding token
 * constantly, and every one of these rendered literally — with the TeX partly
 * normalized, so the user saw `1024$×$768` rather than either proper math or
 * clean source:
 *
 *     1024$\times$768    10$\mu$s    n$\ge$30    5$\sigma$    x$^2$
 *
 * What separates the two is what FOLLOWS the delimiter. Currency continues into
 * an amount; math continues into TeX — a command, a script marker, a brace. So a
 * welded `$` is only disqualified when a digit follows it. `US$5` is currency;
 * `10$\mu$s` is math.
 */
function isViableMathOpener(text: string, index: number): boolean {
  const before = text[index - 1]
  if (before === undefined || !/[A-Za-z0-9]/.test(before)) return true
  const after = text[index + 1]
  return after === undefined || !/[0-9]/.test(after)
}

function findNextOpeningDollar(text: string, start: number): number {
  let index = start
  while (index >= 0) {
    index = findNextSingleDollar(text, index)
    if (index < 0) return -1
    if (isViableMathOpener(text, index)) return index
    index += 1
  }
  return -1
}

function replaceSingleDollarMathLine(
  line: string,
  renderBody: (body: string) => string,
): string {
  let output = ''
  let unchangedStart = 0
  let opener = findNextOpeningDollar(line, 0)

  while (opener >= 0) {
    let closer = findNextSingleDollar(line, opener + 1)

    while (closer >= 0) {
      const body = line.slice(opener + 1, closer)
      if (looksLikeSingleDollarMath(body)) {
        output += line.slice(unchangedStart, opener)
        output += renderBody(body)
        unchangedStart = closer + 1
        opener = findNextOpeningDollar(line, unchangedStart)
        break
      }

      // A rejected pair must not consume the candidate closer. It may be the
      // valid opener that follows literal currency, as in
      // `$43 and $47 \times 19 = 893$`. It may equally be a currency `$` welded
      // to a word, which is not an opener at all, so re-qualify it rather than
      // adopting it outright.
      output += line.slice(unchangedStart, opener + 1)
      unchangedStart = opener + 1
      opener = isViableMathOpener(line, closer)
        ? closer
        : findNextOpeningDollar(line, closer + 1)
      if (opener < 0) break
      closer = findNextSingleDollar(line, opener + 1)
    }

    if (closer < 0) {
      return output + line.slice(unchangedStart)
    }
  }

  return output + line.slice(unchangedStart)
}

function replaceSingleDollarMath(
  text: string,
  renderBody: (body: string) => string,
): string {
  return text
    .split('\n')
    .map((line) => replaceSingleDollarMathLine(line, renderBody))
    .join('\n')
}

function normalizeEscapedUnicodeMath(text: string): string {
  // A model can begin a TeX command with `\` and then emit the equivalent
  // Unicode operator token instead of the command letters (`\×` rather than
  // `\times`). That byte sequence is invalid TeX but has one unambiguous
  // display meaning. Remove only the stray slash before known math glyphs;
  // ordinary backslashes and API payloads remain untouched.
  return text.replace(/\\([×÷·≈≤≥≠±→←∞π])/g, '$1')
}

type MathDelimiter = 'bracket' | 'double-dollar' | 'paren' | 'single-dollar'

// Renderer-owned math HTML must NOT flow through marked: marked re-parses it
// as Markdown, so a literal `\` inside KaTeX output (e.g. `\backslash`), a
// backslash directly before the injected tag, `_` glyphs pairing as emphasis
// across two spans, and the escaped-entity fallback all corrupt visibly.
// While a sink is active, transformMath emits inert private-use-area tokens
// instead of HTML; renderChatMarkdownHtml restores them AFTER marked runs.
// PUA delimiters survive marked untouched (they are plain text to CommonMark;
// NUL would be replaced with U+FFFD).
const MATH_HTML_TOKEN_RE = /(\d+)/g
let activeMathHtmlSink: string[] | null = null

function emitMathHtml(html: string): string {
  if (!activeMathHtmlSink) return html
  const index = activeMathHtmlSink.push(html) - 1
  return `${index}`
}

function renderMath(
  raw: string,
  displayMode: boolean,
  delimiter: MathDelimiter,
): string {
  const rawSource = raw.trim()
  const source = normalizeEscapedUnicodeMath(rawSource)
  if (!source) return ''

  // Bind the completed KaTeX node to the canonical source-span identity:
  // trimmed body, delimiter kind, and inline/display intent after explicit
  // duplicate-delimiter normalization. The stream and SQLite checks separately
  // retain the model's raw bytes. Keep the pre-KaTeX body here—including a
  // possible stray slash before a known Unicode operator—so the DOM remains
  // attributable to the persisted source even when the renderer safely
  // normalizes that operator for KaTeX. Code-point encoding keeps untrusted
  // model text out of HTML syntax and prevents later Markdown/TeX normalization
  // from rewriting the evidence value.
  const sourceAttributes = [
    `data-vmlx-math-source-codepoints="${encodeMathSourceCodepoints(rawSource)}"`,
    `data-vmlx-math-delimiter="${delimiter}"`,
    `data-vmlx-math-display-mode="${displayMode ? 'display' : 'inline'}"`,
  ].join(' ')

  try {
    const html = renderToString(source, {
      ...KATEX_OPTIONS,
      displayMode,
    })
    return displayMode
      ? `<div class="math-block" ${sourceAttributes}>${html}</div>`
      : `<span class="math-inline" ${sourceAttributes}>${html}</span>`
  } catch (_error) {
    const fallback = escapeHtml(source)
    return displayMode
      ? `<div class="math-block math-fallback" ${sourceAttributes}>${fallback}</div>`
      : `<span class="math-inline math-fallback" ${sourceAttributes}>${fallback}</span>`
  }
}

function normalizeBareLatexCommands(markdown: string): string {
  // Models sometimes emit a few TeX commands without delimiters. Do not try to
  // parse arbitrary surrounding prose as math here; just make common commands
  // readable so the UI never shows broken-looking backslash words in normal text.
  let out = normalizeEscapedUnicodeMath(markdown)
  for (let i = 0; i < 8; i++) {
    const next = out.replace(/\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, '$1/$2')
    if (next === out) break
    out = next
  }
  out = out
    .replace(/\\sqrt\s*\{([^{}]+)\}/g, '√($1)')
    .replace(/\\overline\s*\{([^{}]+)\}/g, (_match, body: string) =>
      [...body].map((char) => `${char}\u0305`).join('')
    )
    .replace(/\\text\s*\{([^{}]+)\}/g, '$1')
    .replace(/([A-Za-z0-9)])\^\{([+\-]?\d+)\}/g, (_match, base: string, exponent: string) => {
      const superscript: Record<string, string> = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻',
      }
      return `${base}${[...exponent].map((char) => superscript[char] || char).join('')}`
    })
  return out
    .replace(/\\\(/g, '(')
    .replace(/\\\)/g, ')')
    .replace(/\\times\b/g, '×')
    .replace(/\\div\b/g, '÷')
    .replace(/\\cdot\b/g, '·')
    .replace(/\\approx\b/g, '≈')
    .replace(/\\leq?\b/g, '≤')
    .replace(/\\geq?\b/g, '≥')
    .replace(/\\neq\b/g, '≠')
    .replace(/\\pm\b/g, '±')
    .replace(/\\rightarrow\b/g, '→')
    .replace(/\\leftarrow\b/g, '←')
    .replace(/\\infty\b/g, '∞')
    .replace(/\\ldots\b/g, '…')
    .replace(/\\pi\b/g, 'π')
    .replace(/\\%/g, '%')
    .replace(/\\,/g, ' ')
    .replace(/\\;/g, ' ')
    .replace(/\\!/g, '')
    .replace(/\\left\b/g, '')
    .replace(/\\right\b/g, '')
}

function escapeBareArithmeticAsterisks(markdown: string): string {
  // CommonMark may pair multiplication operators from separate expressions as
  // emphasis delimiters. A model list such as `37*28=...\n37*29=...` then
  // renders as `3728=...3729=...`, even though the API/SQLite bytes are intact.
  // Escape only operator runs directly between operands. Normal prose emphasis
  // (`*important*`) and code spans/fences remain untouched.
  return markdown.replace(
    /([\p{L}\p{N})\]])(\*{1,2})(?=[\p{L}\p{N}(\[])/gu,
    (_match, left: string, operator: string) =>
      `${left}${operator.replace(/\*/g, '\\*')}`,
  )
}

function normalizeRepeatedMathDelimiters(markdown: string): string {
  // Some model streams duplicate an adjacent opener while producing only one
  // closer (for example `\(\(47 \times 19\)`). Nested TeX math delimiters are
  // invalid, so collapse only immediately repeated delimiter tokens. This
  // keeps currency and ordinary parentheses untouched and gives KaTeX the
  // valid span the model clearly intended.
  return markdown
    .replace(/(?:\\\(\s*){2,}/g, '\\(')
    .replace(/(?:\\\)\s*){2,}/g, '\\)')
    .replace(/(?:\\\[\s*){2,}/g, '\\[')
    .replace(/(?:\\\]\s*){2,}/g, '\\]')
}

/**
 * Readable, allocation-light math view for the actively streaming reasoning
 * rail. The completed rail is rendered with KaTeX; this path only prevents
 * transient raw delimiters and common TeX commands from flashing while tokens
 * are still arriving.
 */
export function prepareStreamingPlainTextMath(markdown: string): string {
  if (!markdown) return ''
  // Code spans and fences (closed OR still streaming unclosed) must not have
  // their `\times`, `$..$`, or delimiter text rewritten — the rail was
  // visibly rewriting fence bodies while tokens streamed.
  const protectedSegments: string[] = []
  const protectedMarkdown = markdown.replace(CODE_RE, (segment) => {
    const index = protectedSegments.push(segment) - 1
    return ` STREAMCODE${index} `
  })
  const normalized = normalizeRepeatedMathDelimiters(protectedMarkdown)
  const rewritten = normalizeBareLatexCommands(
    replaceSingleDollarMath(
      normalized
      .replace(/\\\[([\s\S]*?)(?:\\\]|$)/g, '$1')
      .replace(/\\\(([^\n]*?)(?:\\\)|$)/g, '$1')
      .replace(/\$\$([\s\S]*?)(?:\$\$|$)/g, '$1'),
      (body) => body,
    )
  )
  return rewritten.replace(/ STREAMCODE(\d+) /g, (_match, indexText) =>
    protectedSegments[Number(indexText)] || '')
}

function transformMath(markdown: string): string {
  let out = markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_match, body) =>
      emitMathHtml(renderMath(body, true, 'bracket')))
    .replace(/\$\$([\s\S]*?)\$\$/g, (_match, body) =>
      emitMathHtml(renderMath(body, true, 'double-dollar')))
    // Inline math must not consume later paragraphs when a model leaves one
    // opener unmatched in a reasoning stream.
    .replace(/\\\(([^\n]*?)\\\)/g, (_match, body) =>
      emitMathHtml(renderMath(body, false, 'paren')))

  out = replaceSingleDollarMath(out, (body) =>
    emitMathHtml(renderMath(body, false, 'single-dollar')))

  out = normalizeBareLatexCommands(out)
  return escapeBareArithmeticAsterisks(out)
}

export function prepareMarkdownWithMath(markdown: string): string {
  if (!markdown) return ''

  const protectedSegments: string[] = []
  const protectedMarkdown = markdown.replace(CODE_RE, (segment) => {
    const index = protectedSegments.push(segment) - 1
    return `\u0000CODE${index}\u0000`
  })

  const transformed = transformMath(
    normalizeRepeatedMathDelimiters(protectedMarkdown)
  )

  return transformed.replace(/\u0000CODE(\d+)\u0000/g, (_match, indexText) => {
    const index = Number(indexText)
    return protectedSegments[index] || ''
  })
}

/**
 * Chat text is Markdown, not trusted/raw HTML. Marked otherwise treats a
 * placeholder such as `<human-readable size>` or structured model output such
 * as `<result status="ok">` as HTML; DOMPurify then removes unknown tags and
 * the UI visibly drops bytes that remain intact in SQLite and the API.
 *
 * Escape raw HTML before Markdown parsing while protecting code and TeX spans
 * so code fences stay literal and KaTeX still receives the original
 * comparison operators. Renderer-owned KaTeX HTML is injected only after the
 * raw chat text has been escaped.
 */
export function prepareLiteralMarkdownWithMath(markdown: string): string {
  if (!markdown) return ''

  const protectedSegments: string[] = []
  let protectedMarkdown = markdown.replace(LITERAL_MARKDOWN_PROTECTED_RE, (segment) => {
    const index = protectedSegments.push(segment) - 1
    return `\u0000CHATLITERAL${index}\u0000`
  })
  protectedMarkdown = replaceSingleDollarMath(protectedMarkdown, (body) => {
    const index = protectedSegments.push(`$${body}$`) - 1
    return `\u0000CHATLITERAL${index}\u0000`
  })
  const escapedMarkdown = escapeHtml(protectedMarkdown)
    // A raw backslash directly before a character that escapeHtml just
    // entity-encoded (e.g. model text `\"` becoming `\&quot;`) reads to
    // marked as a Markdown `\&` escape, which re-escapes the ampersand and
    // renders a literal `&quot;` to the user (seen live in gemma4 reasoning).
    // Entity-encode the whole backslash run so both characters stay literal,
    // which is this pipeline's contract for raw chat text.
    .replace(/\\+(?=&(?:amp|lt|gt|quot|#39);)/g, (run) => '&#92;'.repeat(run.length))
  const restoredMarkdown = escapedMarkdown.replace(
    /\u0000CHATLITERAL(\d+)\u0000/g,
    (_match, indexText) => protectedSegments[Number(indexText)] || '',
  )
  return prepareMarkdownWithMath(restoredMarkdown)
}

export function prepareUserMarkdownWithMath(markdown: string): string {
  return prepareLiteralMarkdownWithMath(markdown)
}

export function prepareAssistantMarkdownWithMath(markdown: string): string {
  return prepareLiteralMarkdownWithMath(markdown)
}

/**
 * Remove one narrowly identified model-output artifact from the assistant's
 * presentation only.  A Nemotron audio reply once ended with a plain fence
 * followed by confidence-prefixed copies of the answer lines, for example
 * ` ```1.0 answer`.  Those lines are not a distinct answer or code payload:
 * every one repeats a complete line already shown immediately above.  Keep
 * the persisted/API bytes unchanged and leave ordinary fences, user content,
 * and unique confidence text untouched.
 */
export function stripRepeatedConfidenceFenceArtifact(markdown: string): string {
  if (!markdown) return ''
  const lines = markdown.split('\n')
  if (lines.length < 4 || lines[lines.length - 1].trim() !== '```') return markdown

  const closingIndex = lines.length - 1
  let openingIndex = closingIndex - 1
  while (openingIndex >= 0 && lines[openingIndex].trim() !== '```') openingIndex -= 1
  if (openingIndex < 0) return markdown

  const artifactLines = lines.slice(openingIndex + 1, closingIndex)
  if (artifactLines.length < 2) return markdown
  const repeated = artifactLines.map((line) => {
    const match = line.match(/^```\d+(?:\.\d+)?\s+(.+)$/)
    return match ? match[1] : null
  })
  if (repeated.some((line) => line == null)) return markdown

  const precedingLines = lines.slice(0, openingIndex)
  const precedingText = precedingLines.join('\n')
  if (!repeated.every((line) => precedingText.includes(line!))) return markdown

  return precedingLines.join('\n').replace(/\n+$/, '')
}

export interface RenderChatMarkdownOptions {
  renderer?: InstanceType<typeof marked.Renderer>
  assistantContent?: boolean
}

/**
 * The one supported path from raw chat text to markdown HTML. Prepares the
 * text (literal HTML escaping + math), parses with marked using the unified
 * chat options, and only THEN splices in renderer-owned KaTeX HTML — marked
 * must never see that HTML, or it re-parses it as Markdown (a `\` inside
 * KaTeX output, a backslash before the injected tag, `_` glyphs pairing as
 * emphasis across spans, and the escaped-entity fallback all corrupted
 * visibly). Callers still sanitize the returned HTML before rendering.
 */
export function renderChatMarkdownHtml(
  markdown: string,
  options: RenderChatMarkdownOptions = {},
): string {
  if (!markdown) return ''
  const sink: string[] = []
  activeMathHtmlSink = sink
  let prepared: string
  try {
    // Strip pre-existing private-use token delimiters from untrusted text so
    // model output cannot forge a restore token.
    const rawMarkdown = String(markdown).replace(/[]/g, '')
    const presentationMarkdown = options.assistantContent
      ? stripRepeatedConfidenceFenceArtifact(rawMarkdown)
      : rawMarkdown
    prepared = prepareLiteralMarkdownWithMath(presentationMarkdown)
  } finally {
    activeMathHtmlSink = null
  }
  const parsed = marked.parse(prepared, {
    ...(options.renderer ? { renderer: options.renderer } : {}),
    breaks: true,
    gfm: true,
  }) as string
  return parsed.replace(MATH_HTML_TOKEN_RE, (_match, indexText) =>
    sink[Number(indexText)] || '')
}
