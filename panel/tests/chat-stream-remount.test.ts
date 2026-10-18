import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'
import { extractResponsesWarnings } from '../src/renderer/src/lib/responsesWarnings'
import { readAssistantDisplayTimeline } from '../src/shared/assistantDisplayTimeline'

// Execute the production effect and helpers with deferred IPC promises. The
// real navigation/rendering counterpart is exercised in the Electron proof.
const source = readFileSync('src/renderer/src/components/chat/ChatInterface.tsx', 'utf8')
const ast = ts.createSourceFile('ChatInterface.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let effect = ''
const visit = (node: ts.Node) => {
  if (ts.isCallExpression(node) && node.expression.getText(ast) === 'useEffect' &&
      node.arguments[0]?.getText(ast).includes('onComplete(handleComplete)')) {
    effect = node.arguments[0].getText(ast)
  }
  ts.forEachChild(node, visit)
}
visit(ast)
if (!effect) throw new Error('ChatInterface stream effect not found')
const helpers = ast.statements.filter(node => ts.isFunctionDeclaration(node) &&
  ['hydrateMessages', 'latestPendingAssistantId', 'mergeToolStatusHistory'].includes(node.name?.text || ''))
const code = ts.transpileModule(`${helpers.map(node => node.getText(ast)).join('\n')}\nreturn ${effect}`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(r => { resolve = r })
  return { promise, resolve }
}
const flush = async () => { await Promise.resolve(); await Promise.resolve() }
const old = { id: 'old', chatId: 'chat', role: 'user', content: 'Earlier question', timestamp: 1 }
const pending = { id: 'reply', chatId: 'chat', role: 'assistant', content: '', timestamp: 2 }
const completion = {
  chatId: 'chat', messageId: 'reply', content: 'Final answer',
  reasoningContent: 'Final reasoning', reasoningSegments: ['Final reasoning'],
  metrics: { tokenCount: 7, tokensPerSecond: '12.0', ttft: '0.2' },
}

function mount() {
  const state: Record<string, any> = { messages: [], loading: false, streamingMessageId: null,
    currentMetrics: null, reasoningMap: {}, reasoningSegmentMap: {}, reasoningDoneMap: {},
    answerPassMap: {}, toolStatusMap: {}, askUserQuestion: null, askUserInput: '' }
  const messages = deferred<any[]>()
  const snapshots: Array<ReturnType<typeof deferred<boolean>>> = []
  const listeners: Record<string, (data: any) => void> = {}
  const chatIdRef = { current: 'chat' }
  const chat: Record<string, any> = {
    getMessages: vi.fn(() => messages.promise),
    isStreaming: vi.fn(() => { const d = deferred<boolean>(); snapshots.push(d); return d.promise }),
    abort: vi.fn(),
  }
  for (const event of ['Typing', 'Stream', 'Complete', 'ReasoningDone', 'AnswerPass', 'ToolStatus', 'AskUser']) {
    chat[`on${event}`] = (callback: (data: any) => void) => {
      listeners[event] = callback
      return () => { delete listeners[event] }
    }
  }
  const env: Record<string, unknown> = { chatId: 'chat', chatIdRef, window: { api: { chat } }, extractResponsesWarnings, readAssistantDisplayTimeline }
  for (const key of Object.keys(state)) {
    env[`set${key[0].toUpperCase()}${key.slice(1)}`] = (value: any) => {
      state[key] = typeof value === 'function' ? value(state[key]) : value
    }
  }
  const setup = new Function(...Object.keys(env), code)(...Object.values(env))
  const cleanup = setup() as () => void
  const emit = (event: string, data: any = {}) => listeners[event]({ chatId: 'chat', messageId: 'reply', ...data })
  const hydrate = async () => { messages.resolve([old, pending]); await flush() }
  const active = async () => { await hydrate(); snapshots.forEach(d => d.resolve(true)); await flush() }
  return { state, messages, snapshots, chatIdRef, chat, cleanup, emit, hydrate, active }
}

describe('ChatInterface stream ownership across navigation', () => {
  it('hydrates recorded chronology and keeps newer live snapshots through completion', async () => {
    const saved = { version: 1, items: [{ kind: 'reasoning', segment: 0, start: 0, end: 5 }] }
    const live = { version: 1, items: [...saved.items, { kind: 'tool', callId: 'call1' }] }
    const final = { version: 1, items: [...live.items, { kind: 'content', start: 0, end: 12 }] }
    const f = mount()
    f.messages.resolve([{ ...pending, reasoningContent: 'First', reasoningSegmentsJson: '["First",""]', generationRecordJson: JSON.stringify({ displayTimeline: saved }) }])
    await flush()
    expect(f.state.messages[0].displayTimeline).toEqual(saved)
    expect(f.state.reasoningSegmentMap.reply).toEqual(['First', ''])
    f.emit('ToolStatus', { phase: 'calling', toolName: 'read_file', toolCallId: 'call1', displayTimeline: live })
    expect(f.state.messages[0].displayTimeline).toEqual(live)
    f.emit('Complete', { ...completion, displayTimeline: final })
    expect(f.state.messages[0].displayTimeline).toEqual(final)
    f.emit('Typing', { messageId: 'next' })
    f.emit('Stream', { messageId: 'next', isReasoning: true, fullContent: 'First', reasoningSegments: ['First'], displayTimeline: saved })
    expect(f.state.messages.at(-1).displayTimeline).toEqual(saved)
    f.emit('Stream', { messageId: 'next', fullContent: 'Changed', displayTimeline: undefined })
    expect(f.state.messages.at(-1).displayTimeline).toBeUndefined()
  })
  it('settles the remounted instance on terminal completion without its own send promise', async () => {
    const f = mount()
    await f.active()
    expect(f.state.loading).toBe(true)
    f.emit('Stream', { fullContent: 'Partial', metrics: { tokenCount: 1 } })
    f.emit('Complete', completion)
    expect(f.state.loading).toBe(false)
    expect(f.state.streamingMessageId).toBeNull()
    expect(f.state.currentMetrics).toBeNull()
    expect(f.state.messages.at(-1)).toMatchObject({ content: 'Final answer', tokens: 7 })
    expect(f.state.reasoningMap.reply).toBe('Final reasoning')
    expect(f.state.reasoningDoneMap.reply).toBe(true)
  })

  it('preserves exchange-total scope in the live completion and reopened DB history', async () => {
    const metrics = { ...completion.metrics, promptTokens: 206454, cachedTokens: 103000,
      promptTokensScope: 'exchange' }
    const live = mount()
    await live.active()
    live.emit('Stream', { fullContent: 'Partial', metrics: { tokenCount: 1, promptTokens: 103273 } })
    live.emit('Complete', { ...completion, metrics })
    expect(live.state.messages.at(-1).metrics).toEqual(metrics)
    expect(live.state.currentMetrics).toBeNull()
    expect(live.state.streamingMessageId).toBeNull()

    const reopened = mount()
    reopened.messages.resolve([{ ...pending, content: completion.content, metricsJson: JSON.stringify(metrics) }])
    await flush()
    expect(reopened.state.messages[0].metrics).toEqual(metrics)
  })

  it('ignores both late active snapshots after a terminal event', async () => {
    const f = mount()
    await f.hydrate()
    expect(f.snapshots).toHaveLength(2)
    f.emit('Complete', completion)
    f.snapshots.forEach(d => d.resolve(true))
    await flush()
    expect(f.state.loading).toBe(false)
    expect(f.state.streamingMessageId).toBeNull()
  })

  it.each(['disposed', 'different chat'])('ignores stale hydration and active replies for %s', async mode => {
    const f = mount()
    if (mode === 'disposed') f.cleanup()
    else f.chatIdRef.current = 'other'
    const before = structuredClone(f.state)
    f.messages.resolve([old, pending])
    f.snapshots[0].resolve(true)
    await flush()
    f.snapshots.forEach(d => d.resolve(true))
    await flush()
    expect(f.state).toEqual(before)
    expect(f.chat.abort).not.toHaveBeenCalled()
  })

  it('retains live final content, reasoning and tools when older DB hydration arrives last', async () => {
    const f = mount()
    f.emit('Typing')
    f.emit('ToolStatus', { phase: 'done', toolName: 'read_file', toolCallId: 'call1' })
    f.emit('Complete', completion)
    f.messages.resolve([old, { ...pending, content: 'Stale partial', reasoningContent: 'Old reasoning',
      toolCallsJson: JSON.stringify([{ phase: 'running', toolName: 'read_file' }]) }])
    await flush()
    f.snapshots.forEach(d => d.resolve(true))
    await flush()
    expect(f.state.messages.map((m: any) => m.id)).toEqual(['old', 'reply'])
    expect(f.state.messages[1]).toMatchObject({ content: 'Final answer', tokens: 7 })
    expect(f.state.reasoningMap.reply).toBe('Final reasoning')
    expect(f.state.toolStatusMap.reply.at(-1).phase).toBe('done')
    expect(f.state.loading).toBe(false)
  })

  it('keeps a terminal message first observed before initial hydration', async () => {
    const f = mount()
    f.emit('Complete', completion)
    f.messages.resolve([old])
    await flush()
    expect(f.state.messages.map((m: any) => m.id)).toEqual(['old', 'reply'])
    expect(f.state.messages[1].content).toBe('Final answer')
  })

  it('merges a completed persisted tool transcript with its live suffix without duplicating a call', async () => {
    const f = mount()
    const first = { phase: 'calling', toolName: 'read_file', toolCallId: 'call1', iteration: 1 }
    const second = { ...first, toolCallId: 'call2', iteration: 2 }
    const result = { ...second, phase: 'result' }
    const fullDetail = 'x'.repeat(5000)
    f.emit('ToolStatus', { ...result, detail: fullDetail })
    f.emit('ToolStatus', { phase: 'done', toolName: '', iteration: 2 })
    f.emit('Complete', completion)
    f.messages.resolve([old, { ...pending, toolCallsJson: JSON.stringify([
      first, { ...first, phase: 'result', detail: 'earlier' }, second,
      { ...result, detail: fullDetail.slice(0, 4096) + '...' },
      { phase: 'done', toolName: '', iteration: 2 },
    ]) }])
    await flush()
    expect(f.state.toolStatusMap.reply).toHaveLength(5)
    expect(f.state.toolStatusMap.reply.filter((s: any) => s.phase === 'calling').map((s: any) => s.toolCallId))
      .toEqual(['call1', 'call2'])
    expect(f.state.toolStatusMap.reply[3].detail).toBe(fullDetail)
  })

  it('does not deduplicate unanchored status markers belonging to distinct events', async () => {
    const f = mount()
    f.emit('ToolStatus', { phase: 'processing', toolName: '' })
    f.messages.resolve([{ ...pending, toolCallsJson: JSON.stringify([{ phase: 'processing', toolName: '' }]) }])
    await flush()
    expect(f.state.toolStatusMap.reply).toHaveLength(2)
  })

  it('does not treat reasoning, answer-pass or tool events as terminal or accept another chat completion', async () => {
    const f = mount()
    await f.active()
    f.emit('ReasoningDone', { reasoningContent: 'Thoughts' })
    f.emit('AnswerPass')
    f.emit('ToolStatus', { phase: 'done', toolName: 'read_file' })
    f.emit('Complete', { ...completion, chatId: 'other' })
    expect(f.state.loading).toBe(true)
    f.emit('Complete', completion)
    expect(f.state.loading).toBe(false)
    f.emit('Typing', { messageId: 'next' })
    f.emit('Stream', { messageId: 'next', fullContent: 'Next answer' })
    f.emit('Complete', { ...completion, messageId: 'next', content: 'Next answer' })
    expect(f.state.messages.at(-1).content).toBe('Next answer')
    expect(f.chat.abort).not.toHaveBeenCalled()
  })
})
