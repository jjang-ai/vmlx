import { describe, expect, it } from 'vitest'
import { AUTO_HIGHLIGHT_MAX_CHARS, escapeCodeHtml, highlightCodeBlock } from '../src/renderer/src/components/chat/MessageBubble'

// hljs.highlightAuto on large untagged blocks was 99 % of a 2.3 s render of a 370k-character pasted document (2026-10-17).
describe('code block highlighting', () => {
  it('escapes large untagged blocks instead of auto-detecting', () => {
    const big = '<script>alert("x")</script>\n'.repeat(Math.ceil(AUTO_HIGHLIGHT_MAX_CHARS / 20))
    expect(big.length).toBeGreaterThan(AUTO_HIGHLIGHT_MAX_CHARS)
    const out = highlightCodeBlock(big)
    expect(out).toBe(escapeCodeHtml(big))
    expect(out).not.toContain('<script>')
    const t0 = performance.now(); highlightCodeBlock('def f(x):\n    return x\n'.repeat(20000)); expect(performance.now() - t0).toBeLessThan(200)
  })
  it('still highlights tagged blocks and small untagged ones', () => {
    expect(highlightCodeBlock('def f(): pass', 'python')).toContain('hljs-')
    expect(highlightCodeBlock('def f(x):\n    return x + 1\n')).toContain('hljs-')
  })
})

describe('user content', () => {
  it('never auto-detects untagged blocks in user messages', () => {
    expect(highlightCodeBlock('def f(x):\n    return x + 1\n', undefined, true)).toBe(escapeCodeHtml('def f(x):\n    return x + 1\n'))
    expect(highlightCodeBlock('def f(): pass', 'python', true)).toContain('hljs-')
  })
})
