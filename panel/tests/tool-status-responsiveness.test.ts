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
    expect(source).toContain('responsesEventType === "response.heartbeat"')
    expect(source).toContain('parsed.tool_call_generating')
    expect(source).toContain('if (!suppressVisibleToolDelta) {')
    expect(source).toContain('if (!isReasoningDelta && suppressVisibleToolDelta) return;')
  })

  it('has a stall watchdog while waiting for a buffered tool call to finish', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const streamSseSource = source.slice(
      source.indexOf('const streamSSE = async'),
      source.indexOf('await streamSSE(reader);'),
    )

    expect(source).toContain('TOOL_STREAM_STALL_TIMEOUT_MS')
    expect(streamSseSource).toContain('Promise.race')
    expect(streamSseSource).toContain('clientToolCallBuffering')
    expect(streamSseSource).toContain('Tool call generation stalled')
    expect(streamSseSource).toContain('await rdr.cancel()')
  })

  it('marks tool status done whenever any tool status was shown', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const doneIdx = source.indexOf('emitToolStatus("done", "", undefined')
    const preDone = source.substring(Math.max(0, doneIdx - 300), doneIdx)

    expect(doneIdx).toBeGreaterThan(0)
    expect(preDone).toContain('collectedToolStatuses.length > 0')
  })

  it('does not leave a completed message summarized as generating', () => {
    const source = readFileSync('src/renderer/src/components/chat/ToolCallStatus.tsx', 'utf8')

    expect(source).toContain("const isGenerating = isActive && lastStatus.phase === 'generating'")
  })

  it('executes Responses function calls from completed output items with final arguments', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const doneIdx = source.indexOf('responsesEventType === "response.output_item.done"')
    const executeIdx = source.indexOf('const executeToolCalls = async')
    const doneBlock = source.slice(doneIdx, source.indexOf('// Real-time usage', doneIdx))
    const executeBlock = source.slice(executeIdx, source.indexOf('const handleToolLoop', executeIdx))

    expect(doneIdx).toBeGreaterThan(0)
    expect(executeIdx).toBeGreaterThan(doneIdx)
    expect(doneBlock).toContain('parsed.item?.type === "function_call"')
    expect(doneBlock).toContain('arguments: item.arguments || "{}"')
    expect(doneBlock).toContain('emitToolStatus(')
    expect(doneBlock).toContain('item.arguments || "{}"')
    expect(executeBlock).toContain('arguments: tc.function.arguments')
    expect(executeBlock).toContain('JSON.parse(tc.function.arguments || "{}")')
  })
})
