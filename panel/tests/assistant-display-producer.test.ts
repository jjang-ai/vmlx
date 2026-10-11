import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'
import { AssistantDisplayTimelineRecorder } from '../src/shared/assistantDisplayTimeline'

const source = ts.createSourceFile('chat.ts', readFileSync('src/main/ipc/chat.ts', 'utf8'), ts.ScriptTarget.Latest, true)

function find(matches: (node: ts.Node) => boolean, root: ts.Node = source): ts.Node {
  if (matches(root)) return root
  let found: ts.Node | undefined
  root.forEachChild(node => { if (!found) { try { found = find(matches, node) } catch { /* continue */ } } })
  if (!found) throw new Error('Owning producer source not found')
  return found
}

function evaluate(expression: string, env: Record<string, unknown>): any {
  const js = ts.transpileModule(`const owned = ${expression};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS },
  }).outputText
  return new Function(...Object.keys(env), `${js}; return owned;`)(...Object.values(env))
}

function ownedCallback(name: string, env: Record<string, unknown>) {
  const declaration = find(node => ts.isVariableDeclaration(node) && node.name.getText(source) === name) as ts.VariableDeclaration
  return evaluate(declaration.initializer!.getText(source), env)
}

function streamSnapshot(record: AssistantDisplayTimelineRecorder, content: string, segments: string[], isReasoningDelta: boolean) {
  const observe = find(node => ts.isIfStatement(node) && node.expression.getText(source) === 'isReasoningDelta'
    && node.thenStatement.getText(source).includes('displayTimeline.observeReasoning'))
  evaluate(`() => { ${observe.getText(source)} }`, {
    displayTimeline: record, displayContent: content, reasoningSegments: segments, isReasoningDelta,
  })()
  const payload = find(node => ts.isObjectLiteralExpression(node) && node.properties.some(property =>
    ts.isPropertyAssignment(property) && property.name.getText(source) === 'fullContent'
      && property.initializer.getText(source) === 'displayContent'))
  return evaluate(payload.getText(source), {
    displayTimeline: record, displayContent: content, reasoningSegments: segments, isReasoningDelta,
    chatId: 'chat', assistantMessage: { id: 'reply' }, cumulativeTokenOffset: 0, iterationTokenCount: 1,
    promptTokens: 5, cachedTokens: 0, cacheDetail: '', streamTps: 1, prefillRates: {},
    ttft: 0.1, elapsed: 1, remoteMetricFields: () => ({}),
  })
}

describe('actual chat timeline IPC and SQLite producer', () => {
  it('records actual stream/call/result order and persists the same final timeline in the generation record', () => {
    const displayTimeline = new AssistantDisplayTimelineRecorder()
    const events: Array<{ name: string; payload: any }> = []
    const statuses: any[] = []
    const assistantMessage = { id: 'reply', content: 'Answer', generationRecordJson: '' }
    const emit = ownedCallback('emitToolStatus', {
      displayTimeline, collectedToolStatuses: statuses, toolStatusNeedsFlush: false,
      lastEmittedContentLength: 0, chatId: 'chat', assistantMessage,
      getWindow: () => ({ isDestroyed: () => false, webContents: { send: (name: string, payload: any) => events.push({ name, payload }) } }),
    })
    const first = streamSnapshot(displayTimeline, 'Plan', ['Plan'], true)
    emit('calling', 'write_file', '{}', 1, 'a')
    emit('executing', 'write_file', undefined, 1, 'a')
    emit('result', 'write_file', 'Written', 1, 'a')
    const second = streamSnapshot(displayTimeline, 'Plan\n\nCheck', ['Plan', 'Check'], true)
    emit('calling', 'read_file', '{}', 2, 'b')
    emit('result', 'read_file', 'Read', 2, 'b')
    const last = streamSnapshot(displayTimeline, 'Plan\n\nCheck\n\nVerified', ['Plan', 'Check', 'Verified'], true)
    const answer = streamSnapshot(displayTimeline, 'Answer', ['Plan', 'Check', 'Verified'], false)
    expect(first.reasoningSegments).toEqual(['Plan'])
    expect(second.reasoningSegments).toEqual(['Plan', 'Check'])
    expect(last.reasoningSegments).toEqual(['Plan', 'Check', 'Verified'])
    expect(events.map(event => event.payload.phase)).toEqual(['calling', 'executing', 'result', 'calling', 'result'])
    expect(events[2].payload.displayTimeline.items.map((item: any) => item.kind)).toEqual(['reasoning', 'tool'])
    expect(answer.displayTimeline.items.map((item: any) => item.kind)).toEqual(['reasoning', 'tool', 'reasoning', 'tool', 'reasoning', 'content'])
    expect(statuses.map(status => status.phase)).toEqual(events.map(event => event.payload.phase))

    const updates: any[] = []
    const generationRecord = { version: 1, status: 'completed', passes: [] }
    const requestMessages = [{ type: 'function_call', call_id: 'a' }, { type: 'function_call_output', call_id: 'a', output: 'Written' }]
    const save = ownedCallback('saveGenerationRecord', {
      generationRecord, assistantMessage, requestMessages, currentTurnToolStart: 0, displayTimeline,
      reasoningSegments: ['Plan', 'Check', 'Verified'],
      db: { updateMessageGenerationRecord: (id: string, record: string) => updates.push({ id, record }) },
    })
    save()
    const saved = JSON.parse(updates[0].record)
    expect(saved.displayTimeline).toEqual(answer.displayTimeline)
    expect(saved.toolExchange).toEqual(requestMessages)
    expect(JSON.parse(assistantMessage.generationRecordJson)).toEqual(saved)
  })

  it('emits empty reasoning slots unchanged for actual recorded segment indices', () => {
    const displayTimeline = new AssistantDisplayTimelineRecorder()
    streamSnapshot(displayTimeline, 'First', ['First'], true)
    displayTimeline.tool('a')
    displayTimeline.tool('b')
    const resumed = streamSnapshot(displayTimeline, 'First\n\nLast', ['First', '', 'Last'], true)
    expect(resumed.reasoningSegments).toEqual(['First', '', 'Last'])
    expect(resumed.displayTimeline.items.at(-1)).toEqual({ kind: 'reasoning', segment: 2, start: 0, end: 4 })
  })
})
