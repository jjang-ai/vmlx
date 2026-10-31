import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

const readPanelSource = (path: string) =>
  readFileSync(new URL(`../${path}`, import.meta.url), 'utf8')

describe('tool status responsiveness contract', () => {
  it('yields the Electron main loop between visible SSE deltas', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
    const streamSseSource = source.slice(
      source.indexOf('const streamSSE = async'),
      source.indexOf('await streamSSE(reader);'),
    )

    expect(source).toContain('let rendererStreamNeedsFlush = false')
    expect(source).toContain('const flushStreamDeltaToRenderer = async () =>')
    expect(source).toContain('rendererStreamNeedsFlush = true')
    expect(streamSseSource).toContain('if (rendererStreamNeedsFlush)')
    expect(streamSseSource).toContain('await flushStreamDeltaToRenderer();')
  })

  it('yields the Electron main loop after visible tool-status transitions', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')

    expect(source).toContain('const flushToolStatusToRenderer = async () =>')
    expect(source).toContain('await flushToolStatusToRenderer();')
    expect(source).toContain('emitToolStatus(')
  })

  it('flushes tool-status events while draining an already-buffered SSE chunk', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
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
    const source = readPanelSource('src/main/ipc/chat.ts')
    // The marker pattern moved to the shared module so the restore path can
    // bypass re-detection (see responses-stream-recovery tests); chat.ts must
    // consume the shared export, and the shared module owns the dialects.
    const shared = readPanelSource('src/shared/responsesStreamRecovery.ts')

    expect(source).toContain('TOOL_CALL_MARKER_LINE_START')
    expect(shared).toContain('<zyphra_tool_call\\b')
    expect(shared).toContain('<function(?:=|\\b)')
    // Markup detected in emitDelta raises the generating status itself.
    const emitDeltaSource = source.slice(
      source.indexOf('const emitDelta = ('),
      source.indexOf('// Strip any leaked chat template tokens from the delta'),
    )
    expect(emitDeltaSource).toContain('if (observed.startedGenerating) {')
    expect(emitDeltaSource).toContain('emitToolStatus(\n                "generating"')
    expect(source).toContain('responsesEventType === "response.heartbeat"')
    expect(source).toContain('parsed.tool_call_generating')
    expect(source).toContain('if (!suppressVisibleToolDelta) {')
    expect(source).toContain('if (!isReasoningDelta && suppressVisibleToolDelta) return;')
  })

  it('emits one generating status per buffered tool call instead of one per heartbeat', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
    const responsesHeartbeatStart = source.indexOf('responsesEventType === "response.heartbeat"')
    const responsesHeartbeat = source.slice(
      responsesHeartbeatStart,
      source.indexOf('// Reasoning delta from OpenAI Responses', responsesHeartbeatStart),
    )
    const genericHeartbeat = source.slice(
      source.indexOf('// Detect server-side tool call buffering signal'),
      source.indexOf('// Handle tool_calls from streaming response'),
    )

    expect(responsesHeartbeat).toContain('if (!clientToolCallBuffering) {')
    expect(responsesHeartbeat.match(/emitToolStatus\(/g)).toHaveLength(1)
    expect(genericHeartbeat).toContain('if (!useResponsesApi && parsed.tool_call_generating)')
    expect(genericHeartbeat).toContain('if (!clientToolCallBuffering) {')
    expect(genericHeartbeat.match(/emitToolStatus\(/g)).toHaveLength(1)
  })

  it('has a stall watchdog while waiting for a buffered tool call to finish', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
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
    const source = readPanelSource('src/main/ipc/chat.ts')
    const doneIdx = source.indexOf('emitToolStatus("done", "", undefined')
    const preDone = source.substring(Math.max(0, doneIdx - 300), doneIdx)

    expect(doneIdx).toBeGreaterThan(0)
    expect(preDone).toContain('collectedToolStatuses.length > 0')
  })

  it('does not leave a completed message summarized as generating', () => {
    const source = readPanelSource('src/renderer/src/components/chat/ToolCallStatus.tsx')

    expect(source).toContain("const isGenerating = isActive && lastStatus.phase === 'generating'")
  })

  it('does not render a speculative buffering heartbeat as a completed zero-tool call', () => {
    const source = readPanelSource('src/renderer/src/components/chat/ToolCallStatus.tsx')

    expect(source).toContain('const hasStandaloneError')
    expect(source).toContain('if (!isActive && toolCount === 0 && !hasStandaloneError) return null')
  })

  it('executes Responses function calls from completed output items with final arguments', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
    const doneIdx = source.indexOf('responsesEventType === "response.output_item.done"')
    const executeIdx = source.indexOf('const executeToolCalls = async')
    const doneBlock = source.slice(doneIdx, source.indexOf('// Real-time usage', doneIdx))
    const executeBlock = source.slice(executeIdx, source.indexOf('const handleToolLoop', executeIdx))

    expect(doneIdx).toBeGreaterThan(0)
    expect(executeIdx).toBeGreaterThan(doneIdx)
    expect(doneBlock).toContain('parsed.item?.type === "function_call"')
    expect(doneBlock).toContain('const finalArguments =')
    expect(doneBlock).toContain('arguments: finalArguments')
    expect(doneBlock).not.toContain('emitToolStatus(')
    expect(executeBlock).toContain('emitToolStatus(\n              "calling"')
    expect(executeBlock).toContain('tc.function.arguments')
    expect(executeBlock).toContain('arguments: tc.function.arguments')
    expect(executeBlock).toContain('JSON.parse(tc.function.arguments || "{}")')
  })

  it('replays current reasoning before active tool calls in follow-up requests', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
    const executeIdx = source.indexOf('const executeToolCalls = async')
    const executeBlock = source.slice(executeIdx, source.indexOf('const handleToolLoop', executeIdx))

    expect(source).toContain('const currentReasoningSegment = () =>')
    expect(source).toContain('const responsesReasoningItem = (text: string) =>')
    expect(executeBlock).toContain('const toolReasoning = currentReasoningSegment();')
    expect(executeBlock).toContain('requestMessages.push(responsesReasoningItem(toolReasoning));')
    expect(executeBlock).toContain('assistantToolTurn.reasoning_content = toolReasoning;')
    expect(executeBlock.indexOf('responsesReasoningItem(toolReasoning)')).toBeLessThan(
      executeBlock.indexOf('type: "function_call"'),
    )
  })

  it('cleans a redundant namespaced tool preview before Responses continuation', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')

    expect(source).toContain('stripRedundantNamespacedToolPreview(')
    expect(source).toContain('clearedRedundantToolPreview')
    expect(source).toContain('Removed redundant namespaced tool preview')
  })

  it('recovers Responses function-call arguments from argument delta and done events', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
    const responsesParser = source.slice(
      source.indexOf('// ── Responses API SSE parsing ──'),
      source.indexOf('// Real-time usage from response.usage events'),
    )

    expect(source).toContain('responsesFunctionCallArgsByKey')
    expect(source).toContain('responsesFunctionCallItemKey')
    expect(responsesParser).toContain('response.function_call_arguments.delta')
    expect(responsesParser).toContain('response.function_call_arguments.done')
    expect(responsesParser).toContain('argsBuffer.value += parsed.delta')
    expect(responsesParser).toContain('argsBuffer.value = parsed.arguments')
    expect(responsesParser).toContain('const finalArguments =')
    expect(responsesParser).toContain('item.arguments || argsBuffer?.value || "{}"')
  })

  // d3eca9d12 (#309, "expose negotiated prefill timing on Chat Completions")
  // widened the private usage negotiation from Responses-only to every wire
  // API, still gated on the local engine. Remote providers never see it.
  it('negotiates incremental usage telemetry only with the local vMLX engine', () => {
    const source = readPanelSource('src/main/ipc/chat.ts')
    const responsesBody = source.slice(
      source.indexOf('if (useResponsesApi)'),
      source.indexOf('} else {', source.indexOf('if (useResponsesApi)')),
    )

    expect(responsesBody).not.toContain('stream_options: { include_usage: true }')
    const declStart = source.indexOf('const vmlxUsageHeaders: Record<string, string> =')
    expect(declStart).toBeGreaterThan(0)
    const decl = source.slice(declStart, source.indexOf(';', declStart))
    expect(decl.replace(/\s+/g, ' ')).toContain('= !isRemote ? { "X-vMLX-Stream-Usage": "incremental" } : {}')
    const initialFetch = source.slice(declStart, source.indexOf('const sendFollowUp = async'))
    // Both the Node streaming (loopback) and net.fetch branches carry it.
    expect(initialFetch.match(/\.\.\.vmlxUsageHeaders,/g)?.length).toBe(2)
    const followUp = source.slice(
      source.indexOf('const sendFollowUp = async'),
      source.indexOf(
        '// ─── Helper: execute tool calls',
        source.indexOf('const sendFollowUp = async'),
      ),
    )
    expect(followUp).toContain('...vmlxUsageHeaders')
  })
})
