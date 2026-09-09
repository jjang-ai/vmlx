import { describe, expect, it } from 'vitest'
import { marked } from 'marked'
import {
  prepareMarkdownWithMath,
  prepareAssistantMarkdownWithMath,
  prepareStreamingPlainTextMath,
  prepareUserMarkdownWithMath,
  renderChatMarkdownHtml,
} from '../src/renderer/src/components/chat/mathMarkdown'

describe('currency amounts separated by Unicode arithmetic', () => {
  const invoice = 'Discount amount: 10% × $245.00 = **−$24.50**'

  it('preserves the stored invoice amounts and their Markdown emphasis', () => {
    const html = renderChatMarkdownHtml(invoice, { assistantContent: true })
    expect(html).toContain('$245.00')
    expect(html).toContain('<strong>−$24.50</strong>')
    expect(html).not.toContain('math-inline')
    expect(html).not.toContain('**')
  })

  it('does not consume currency delimiters at any streamed prefix', () => {
    for (let end = 1; end <= invoice.length; end++) {
      const prefix = invoice.slice(0, end)
      expect(prepareStreamingPlainTextMath(prefix)).toBe(prefix)
      expect(prepareMarkdownWithMath(prefix)).not.toContain('math-inline')
    }
  })

  it.each(['−', '×', '÷', '±', '≤', '≥', '≠', '≈'])(
    'does not treat an incomplete expression ending with %s as math',
    (operator) => {
      const source = 'Cost $245.00 = ' + operator + '$24.50'
      expect(prepareMarkdownWithMath(source)).toBe(source)
    },
  )

  it('retains complete Unicode arithmetic and welded TeX spans', () => {
    for (const source of ['$245.00 − 24.50 = 220.50$', '$= 2 − 1$', '1024$\\times$768']) {
      expect(prepareMarkdownWithMath(source)).toContain('class="katex"')
    }
  })
})

describe('prepareMarkdownWithMath', () => {
  it('renders inline TeX delimiters as readable math text', () => {
    const rendered = prepareMarkdownWithMath('Multiply first: \\(47 \\times 2 = 94\\)')

    expect(rendered).toContain('math-inline')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('47')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('\\times')
    expect(rendered).not.toContain('\\(')
  })

  it('renders TeX fractions without exposing raw commands', () => {
    const rendered = prepareMarkdownWithMath('Exact answer: \\(\\frac{47}{45}\\)')

    expect(rendered).toContain('math-inline')
    expect(rendered).toContain('class="mfrac"')
    expect(rendered).toContain('47')
    expect(rendered).toContain('45')
    expect(rendered).not.toContain('\\frac')
  })

  it('renders display math blocks without raw dollar delimiters', () => {
    const rendered = prepareMarkdownWithMath('Compare:\n$$43 \\times 17 = 731$$')

    expect(rendered).toContain('math-block')
    expect(rendered).toContain('class="katex-display"')
    expect(rendered).toContain('43')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('$$')
  })

  it('normalizes bare TeX commands emitted by models outside delimiters', () => {
    const rendered = prepareMarkdownWithMath(
      'Divide: (94 \\div 90 = \\frac{94}{90}); \\approx 1.4 \\times 10^{-6}',
    )

    expect(rendered).toContain('÷')
    expect(rendered).toContain('94/90')
    expect(rendered).toContain('≈ 1.4 × 10⁻⁶')
    expect(rendered).not.toContain('\\div')
    expect(rendered).not.toContain('\\frac')
  })

  it('keeps the active reasoning rail readable before TeX delimiters close', () => {
    const rendered = prepareStreamingPlainTextMath(
      'Work: \\(47 \\times 2 = 94 and \\frac{47}{45}',
    )

    expect(rendered).toBe('Work: 47 × 2 = 94 and 47/45')
    expect(rendered).not.toContain('\\(')
    expect(rendered).not.toContain('\\times')
  })

  it('does not let an unmatched inline opener consume later paragraphs', () => {
    const rendered = prepareMarkdownWithMath(
      'Draft \\(47/45\nLater valid math: \\(2 + 2 = 4\\)',
    )

    expect(rendered).toContain('Draft (47/45')
    expect(rendered).toContain('Later valid math:')
    expect(rendered.match(/class="katex"/g)).toHaveLength(1)
  })

  it('uses readable escaped fallback instead of a visible KaTeX error node', () => {
    const rendered = prepareMarkdownWithMath('Malformed: \\(\\frac{47}{\\)')

    expect(rendered).toContain('math-fallback')
    expect(rendered).not.toContain('katex-error')
    expect(rendered).toContain('\\frac{47}{')
  })

  it('repairs duplicated adjacent model delimiters before KaTeX rendering', () => {
    const malformed =
      'Path: panel/package.json — Size: 5.2 KB\n$43 and inline TeX: \\(\\(47 \\times 19 = 893 < 920 = 46 \\times 20\\)'
    const rendered = prepareMarkdownWithMath(malformed)

    expect(rendered).toContain('$43')
    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('math-fallback')
    expect(rendered).not.toContain('\\(')
    expect(rendered).not.toContain('\\times')
  })

  it('renders a Unicode operator emitted after a TeX escape', () => {
    const malformed = 'CURRENCY=$43 TEX=\\(47 \\× 19 = 893\\)'
    const rendered = prepareMarkdownWithMath(malformed)

    expect(rendered).toContain('$43')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('math-fallback')
    expect(rendered).not.toContain('\\×')
  })

  it('retains canonical TeX identity in inert source, delimiter, and mode attributes', () => {
    const rawSource = '47 \\× 19 = 893 < 920'
    const rendered = prepareMarkdownWithMath(
      `CURRENCY=$43 TEX=\\(${rawSource}\\)`,
    )
    const encodedSource = Array.from(
      rawSource,
      (char) => char.codePointAt(0)!.toString(16),
    ).join('-')

    expect(rendered).toContain(
      `data-vmlx-math-source-codepoints="${encodedSource}"`,
    )
    expect(rendered).toContain('data-vmlx-math-delimiter="paren"')
    expect(rendered).toContain('data-vmlx-math-display-mode="inline"')
    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('math-fallback')

    const hostile = prepareMarkdownWithMath(
      '\\(x" data-vmlx-injected="yes < y\\)',
    )
    expect(hostile).toMatch(/data-vmlx-math-source-codepoints="[0-9a-f-]+"/)
    expect(hostile).not.toContain('data-vmlx-injected="yes')
  })

  it('retains distinct display delimiter identities for bracket and dollar math', () => {
    const rendered = prepareMarkdownWithMath(
      '\\[x + 1\\]\n$$y + 2$$',
    )

    expect(rendered).toContain('data-vmlx-math-delimiter="bracket"')
    expect(rendered).toContain('data-vmlx-math-delimiter="double-dollar"')
    expect(rendered.match(/data-vmlx-math-display-mode="display"/g)).toHaveLength(2)
    expect(rendered.match(/class="katex-display"/g)).toHaveLength(2)
  })

  it('keeps duplicated adjacent delimiters readable during reasoning streaming', () => {
    const rendered = prepareStreamingPlainTextMath(
      'Draft: \\(\\(47 \\times 19 = 893 < 920',
    )

    expect(rendered).toBe('Draft: 47 × 19 = 893 < 920')
  })

  it('keeps an escaped Unicode operator readable during reasoning streaming', () => {
    const rendered = prepareStreamingPlainTextMath('Draft: \\(47 \\× 19 = 893')

    expect(rendered).toBe('Draft: 47 × 19 = 893')
  })

  it('does not treat plain dollar amounts as math', () => {
    const rendered = prepareMarkdownWithMath('The cost is $43 today.')

    expect(rendered).toBe('The cost is $43 today.')
  })

  it('does not consume dollars when comparing currency amounts', () => {
    const rendered = prepareMarkdownWithMath('Compare $5<$10 and $5 < $10.')

    expect(rendered).toBe('Compare $5<$10 and $5 < $10.')
    expect(rendered).not.toContain('math-inline')
  })

  it('retries a rejected currency closer as the next valid math opener', () => {
    const rendered = prepareMarkdownWithMath(
      'The literal currency string is $43 and $47 \\times 19 = 893 < 920 = 46 \\times 20$.',
    )

    expect(rendered).toContain('$43 and ')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('47')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('$47')
    expect(rendered).not.toContain('\\times')
  })

  it('does not pair an unclosed math opener with a later currency amount through prose', () => {
    const rendered = prepareMarkdownWithMath(
      'The literal currency string is $43 and $47 \\times 19 = 893 < 920 = 46 \\times 20. This seems to be the second line. There is another literal currency string $43.',
    )

    expect(rendered).not.toContain('class="katex"')
    expect(rendered).toContain('$43 and $47 × 19 = 893 < 920 = 46 × 20')
    expect(rendered).toContain('literal currency string $43')
  })

  it('preserves escaped currency before a later valid single-dollar span', () => {
    const rendered = prepareMarkdownWithMath(
      'Escaped currency is \\$43; math is $6 \\times 7 = 42$.',
    )

    expect(rendered).toContain('\\$43')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('×')
    expect(rendered).not.toContain('$6')
  })

  it('keeps the currency/math overlap readable during reasoning streaming', () => {
    const rendered = prepareStreamingPlainTextMath(
      'The literal currency string is $43 and $47 \\times 19 = 893$.',
    )

    expect(rendered).toBe(
      'The literal currency string is $43 and 47 × 19 = 893.',
    )
  })

  it('renders adjacent inline spans without mistaking their boundary for display math', () => {
    const rendered = marked.parse(
      prepareAssistantMarkdownWithMath('$x$$y$'),
    ) as string

    expect(rendered.match(/class="katex"/g)).toHaveLength(2)
    expect(rendered).not.toContain('$x')
    expect(rendered).not.toContain('$y')
  })

  it('renders multiplication without letting Markdown create emphasis', () => {
    const rendered = prepareMarkdownWithMath('Product: \\(2 * 3 * 4\\)')

    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('<em>')
    expect(rendered).not.toContain('*')
  })

  it('preserves bare arithmetic asterisks across repeated model calculations', () => {
    const prepared = prepareMarkdownWithMath(
      '37*28=1036 (sum 10)\n37*29=1073 (sum 11)\n37*30=1110 (sum 3)',
    )
    const rendered = marked.parse(prepared) as string

    expect(rendered).toContain('37*28=1036')
    expect(rendered).toContain('37*29=1073')
    expect(rendered).toContain('37*30=1110')
    expect(rendered).not.toContain('<em>')
    expect(rendered).not.toContain('<strong>')
  })

  it('keeps ordinary Markdown emphasis while escaping only arithmetic operators', () => {
    const prepared = prepareMarkdownWithMath('This is *important*, and x*y stays multiplication.')
    const rendered = marked.parse(prepared) as string

    expect(rendered).toContain('<em>important</em>')
    expect(rendered).toContain('x*y stays multiplication')
  })

  it('preserves raw TeX inside inline code and fenced code', () => {
    const rendered = prepareMarkdownWithMath('`\\frac{1}{2}`\n```txt\n47 \\times 2\n```')

    expect(rendered).toContain('`\\frac{1}{2}`')
    expect(rendered).toContain('47 \\times 2')
    expect(rendered).not.toContain('math-inline')
    expect(rendered).not.toContain('×')
  })

  it('escapes raw user HTML without corrupting protected TeX or code', () => {
    const prepared = prepareUserMarkdownWithMath(
      'SIZE=<human-readable size>; compare \\(893 < 920\\); keep `<tag>` literal.',
    )
    const rendered = marked.parse(prepared) as string

    expect(rendered).toContain('SIZE=&lt;human-readable size&gt;')
    expect(rendered).not.toContain('<human-readable')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('893')
    expect(rendered).toContain('<code>&lt;tag&gt;</code>')
  })

  it('preserves structured assistant XML as visible text alongside KaTeX', () => {
    const prepared = prepareAssistantMarkdownWithMath(
      'XML=<result status="ok">5.2 KB</result>; compare \\(893 < 920\\).',
    )
    const rendered = marked.parse(prepared) as string

    expect(rendered).toContain(
      'XML=&lt;result status=&quot;ok&quot;&gt;5.2 KB&lt;/result&gt;',
    )
    expect(rendered).not.toContain('<result')
    expect(rendered).toContain('class="katex"')
    expect(rendered).toContain('893')
  })
})

describe('single-dollar math with multi-letter symbols', () => {
  /**
   * The currency guard rejected any alphabetic run of TWO or more letters that
   * was not a math function name — while its own comment claimed the rule
   * "keeps `$E=mc^2$` valid". It did not: `mc` is a two-letter run, so the most
   * recognisable inline formula in physics rendered as literal dollar-wrapped
   * text, as did `$F = ma$` and `$ab = cd$`.
   *
   * LIVE-REPRODUCED in the app before the fix: asked for `$E = mc^2$` on its own
   * line, the chat rendered `<p>$E = mc^2$</p>` with zero .katex nodes while the
   * heading, list and inline code in the same reply rendered correctly.
   *
   * The threshold is now three letters. Two-letter runs still cannot smuggle
   * prose through, because a span must additionally show math STRUCTURE (a TeX
   * command, one of {}_^=<>, or an infix operator) before it is accepted.
   */
  const renders = (tex: string) =>
    prepareMarkdownWithMath(tex).includes('math-inline')

  it.each([
    ['$E=mc^2$'],
    ['$E = mc^2$'],
    ['$F = ma$'],
    ['$ab = cd$'],
  ])('renders %s instead of leaving the dollars visible', (input) => {
    expect(renders(input)).toBe(true)
  })

  it.each([
    ['$x^2 + y^2 = z^2$'],
    ['$47 \\times 19$'],
    ['$a$'],
    ['$\\alpha$'],
  ])('still renders %s', (input) => {
    expect(renders(input)).toBe(true)
  })

  it.each([
    ['$5$'],
    ['$100$'],
    ['$47 \\times 19 ... This seems to be ... $43'],
    ['$x is 5$'],
    ['$the answer is here$'],
  ])('still refuses %s', (input) => {
    expect(renders(input)).toBe(false)
  })

  it('a bare two-letter symbol with no math structure stays literal', () => {
    // Deliberate: `$dx$` alone is indistinguishable from prose or currency, and
    // accepting it would reopen the false positives above. Recorded so the
    // conservative choice is not mistaken for an oversight.
    expect(renders('$dx$')).toBe(false)
  })
})

describe('a dollar welded to a word is currency, not a math opener', () => {
  /**
   * Raising the prose threshold to three letters (so `$E=mc^2$` renders) let
   * `US$5 = CA$7` through: the candidate body `5 = CA` has no 3+ letter run, so
   * nothing marks it as prose, and the `=` then satisfies the structure test.
   * The line rendered as literal `US`, math `$5 = CA$`, literal `7`.
   *
   * The fix is positional rather than lexical, because lowering the threshold
   * again would re-break `$E=mc^2$` AND reject legitimate `$AB = CD$` geometry.
   * Real inline math opens at a boundary; `US$` and `CA$` attach the delimiter
   * to the preceding word.
   */
  const rendered = (source: string) => prepareMarkdownWithMath(source)
  const rendersMath = (source: string) => rendered(source).includes('math-inline')

  it('leaves two currency amounts alone', () => {
    expect(rendersMath('US$5 = CA$7')).toBe(false)
  })

  it.each([
    ['US$5 = CA$7'],
    ['AU$10 = NZ$11'],
    ['costs US$5 and CA$7 total'],
  ])('leaves %s entirely literal', (input) => {
    expect(rendersMath(input)).toBe(false)
  })

  it('still renders math that opens at a boundary', () => {
    expect(rendersMath('$E=mc^2$')).toBe(true)
    expect(rendersMath('the identity $E=mc^2$ holds')).toBe(true)
    expect(rendersMath('($E=mc^2$)')).toBe(true)
  })

  it('still renders geometry that the lexical rule would have rejected', () => {
    expect(rendersMath('$AB = CD$')).toBe(true)
  })

  it('a plain currency pair after real math still ends the math', () => {
    // Guards the reject path: the closer of a rejected pair must be re-qualified
    // as an opener rather than adopted outright.
    expect(rendersMath('$47 \\times 19 = 893$ and US$43')).toBe(true)
  })
})

describe('welded inline math still renders — currency is decided by what follows', () => {
  /**
   * Looking only BACKWARDS from the `$` shipped a worse regression than the
   * currency bug it fixed. Models weld inline math to the preceding token
   * constantly, and all of these rendered literally, with the TeX partly
   * normalized so the user saw `1024$×$768` — neither proper math nor clean
   * source. Measured live through the app's own module before the forward check.
   */
  const rendersMath = (source: string) =>
    prepareMarkdownWithMath(source).includes('math-inline')

  it.each([
    ['a 2$\\times$ speedup'],
    ['1024$\\times$768'],
    ['10$\\mu$s'],
    ['n$\\ge$30'],
    ['5$\\sigma$'],
    ['x$^2$'],
    ['x$_i$'],
  ])('renders %s', (input) => {
    expect(rendersMath(input)).toBe(true)
  })

  it.each([
    ['US$5 = CA$7'],
    ['AU$10 = NZ$11'],
    ['costs US$5 and CA$7 total'],
  ])('still leaves %s literal', (input) => {
    expect(rendersMath(input)).toBe(false)
  })

  it('a boundary opener is unaffected either way', () => {
    expect(rendersMath('$E=mc^2$')).toBe(true)
    expect(rendersMath('the identity $E=mc^2$ holds')).toBe(true)
    expect(rendersMath('$AB = CD$')).toBe(true)
  })
})

describe('a derivation continued onto its own line is math, not currency', () => {
  /**
   * Live-reproduced on gemma-4-E4B: the model rendered
   * `17 \times 23 = 17 \times (20 + 3)` correctly through KaTeX and then
   * emitted the remaining steps as `$= (17 \times 20) + (17 \times 3)$`,
   * `$= 340 + 51$`, `$= 391$` — every one of which showed as literal dollar
   * text in the chat. The guard rejected any body STARTING with an operator,
   * but the streaming case its comment cites ("$5<$10" matching "$5<$") is
   * caught by the TRAILING operator instead.
   */
  it('renders a step that opens with = and carries a TeX command', () => {
    const rendered = prepareMarkdownWithMath('$= (17 \\times 20) + (17 \\times 3)$')

    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('$=')
  })

  it('renders a bare arithmetic step that opens with =', () => {
    const rendered = prepareMarkdownWithMath('$= 340 + 51$')

    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('$=')
  })

  it('renders a single-value step that opens with =', () => {
    const rendered = prepareMarkdownWithMath('$= 391$')

    expect(rendered).toContain('class="katex"')
    expect(rendered).not.toContain('$=')
  })

  it('still leaves a body that ENDS with an operator alone', () => {
    // The streaming guard the original comment was actually written for.
    const rendered = prepareMarkdownWithMath('Compare $5<$10 today.')

    expect(rendered).toBe('Compare $5<$10 today.')
    expect(rendered).not.toContain('math-inline')
  })

  it('still leaves a leading sign alone, which a currency amount can carry', () => {
    const rendered = prepareMarkdownWithMath('The swing was $-5$ per share.')

    expect(rendered).toBe('The swing was $-5$ per share.')
    expect(rendered).not.toContain('math-inline')
  })
})

describe('a backslash before an HTML-escapable character stays literal', () => {
  // Live repro (gemma4 reasoning rail): model text `"\"` was escaped to
  // `\&quot;`, which marked read as a Markdown `\&` escape and re-escaped the
  // ampersand — the user saw a literal `&quot;` on screen.
  it('renders backslash-quote as backslash and quote, never a visible entity', () => {
    const html = marked.parse(
      prepareAssistantMarkdownWithMath('including "$" and "\\".'),
    ) as string

    expect(html).not.toContain('&amp;quot;')
    expect(html).toContain('&#92;&quot;')
  })

  it('keeps every backslash of a run before an escaped character', () => {
    const html = marked.parse(
      prepareAssistantMarkdownWithMath('double "\\\\" run.'),
    ) as string

    expect(html).not.toContain('&amp;')
    expect(html).toContain('&#92;&#92;&quot;')
  })

  it('covers < and & targets the same way', () => {
    const html = marked.parse(
      prepareAssistantMarkdownWithMath('a \\< b and \\& c'),
    ) as string

    expect(html).not.toContain('&amp;lt;')
    expect(html).not.toContain('&amp;amp;')
    expect(html).toContain('&#92;&lt;')
    expect(html).toContain('&#92;&amp;')
  })
})

import { renderChatMarkdownHtml } from '../src/renderer/src/components/chat/mathMarkdown'

describe('renderChatMarkdownHtml keeps renderer-owned math HTML out of marked', () => {
  it('does not inject math markup into an unclosed streaming code fence', () => {
    const html = renderChatMarkdownHtml('S:\n```js\nconst re = /\\[abc\\]/;\nconst s = "\\(group\\)";')

    expect(html).not.toContain('math-block')
    expect(html).not.toContain('math-inline')
    expect(html).toContain('\\[abc\\]')
  })

  it('renders math after an odd leading backslash without leaking the wrapper tag', () => {
    const html = renderChatMarkdownHtml('The answer is \\\\(2+2=4\\\\), confirmed.')

    expect(html).not.toContain('&lt;span class="math-inline')
    expect(html).toContain('math-inline')
  })

  it('keeps a KaTeX backslash glyph intact instead of showing a stray closing tag', () => {
    const html = renderChatMarkdownHtml('norm \\(x \\backslash y\\) mid \\(a+b\\) end')

    expect(html).not.toContain('&lt;/span&gt;')
    expect(html).toContain('math-inline')
  })

  it('KaTeX-error fallback shows backslash-quote literally, never a visible entity', () => {
    const html = renderChatMarkdownHtml('Try \\(\\errcmd \\"x\\"\\) done')

    expect(html).toContain('math-fallback')
    expect(html).not.toContain('&amp;quot;')
    expect(html).not.toContain('&amp;amp;')
  })

  it('does not pair underscores across two adjacent math spans as emphasis', () => {
    const html = renderChatMarkdownHtml('lit \\(a\\_b\\) and \\(c\\_d\\) end')

    expect(html).not.toContain('<em>')
  })

  it('keeps single newlines as line breaks (unified breaks option)', () => {
    const html = renderChatMarkdownHtml('line one\nline two\nline three')

    expect((html.match(/<br\s*\/?>/g) || []).length).toBe(2)
  })
})

describe('prepareStreamingPlainTextMath protects code while streaming', () => {
  it('leaves fence bodies untouched', () => {
    const out = prepareStreamingPlainTextMath('Code:\n```\na = "\\\\times 5"\nb = "$x^2$ z"\n```\ndone')

    expect(out).toContain('\\\\times 5')
    expect(out).toContain('$x^2$ z')
  })

  it('leaves an unclosed streaming fence untouched', () => {
    const out = prepareStreamingPlainTextMath('S:\n```js\nconst s = "\\(group\\)";')

    expect(out).toContain('\\(group\\)')
  })
})
