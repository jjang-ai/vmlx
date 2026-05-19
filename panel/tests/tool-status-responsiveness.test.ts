import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('tool status responsiveness contract', () => {
  it('yields the Electron main loop after visible tool-status transitions', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const flushToolStatusToRenderer = async () =>')
    expect(source).toContain('await flushToolStatusToRenderer();')
    expect(source).toContain('emitToolStatus(')
  })

  it('flushes tool-status events while draining an already-buffered SSE chunk', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const streamSseSource = source.slice(
      source.indexOf('const streamSSE = async'),
      source.indexOf('await streamSSE(reader);'),
    )

    expect(source).toContain('let toolStatusNeedsFlush = false')
    expect(source).toContain('toolStatusNeedsFlush = true')
    expect(streamSseSource).toContain('if (toolStatusNeedsFlush)')
    expect(streamSseSource).toContain('await flushToolStatusToRenderer();')
  })

  it('detects partial ZAYA/Zyphra XML tool prefixes before raw markup renders', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('<zyphra_tool_call\\b')
    expect(source).toContain('<function(?:=|\\b)')
    expect(source).toContain('emitToolStatus(\n                  "generating"')
    expect(source).toContain('currentEventType === "response.heartbeat"')
    expect(source).toContain('parsed.tool_call_generating')
    expect(source).toContain('if (!suppressVisibleToolDelta) {')
    expect(source).toContain('if (!isReasoningDelta && suppressVisibleToolDelta) return;')
  })
})
