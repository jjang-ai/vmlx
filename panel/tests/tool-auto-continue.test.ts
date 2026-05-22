import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { shouldAutoContinueAfterToolUse } from '../src/shared/toolAutoContinue'

describe('tool auto-continue policy', () => {
  it('continues when a model stops after tools with no visible response', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: '',
        iterationTokenCount: 0,
        finishReason: 'stop',
        thresholdTokens: 100,
      }),
    ).toBe(true)
  })

  it('continues short content only when the model hit the length limit', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: 'partial sentence',
        iterationTokenCount: 4,
        finishReason: 'length',
        thresholdTokens: 100,
      }),
    ).toBe(true)
  })

  it('does not duplicate a short normal final answer after tool results', () => {
    expect(
      shouldAutoContinueAfterToolUse({
        content: 'Done after tools.',
        iterationTokenCount: 4,
        finishReason: 'stop',
        thresholdTokens: 100,
      }),
    ).toBe(false)
  })

  it('increments the auto-continue counter once per follow-up attempt', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('shouldAutoContinueAfterToolUse({'),
      source.indexOf('const hasContent = fullContent.trim().length > 0'),
    )

    expect(branch.match(/autoContinueCount\+\+/g) || []).toHaveLength(1)
  })

  it('resets text-chat tool streaming state before chained follow-up requests', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('receivedToolCalls = [];'),
      source.indexOf('if (!(await sendFollowUp())) break;', source.indexOf('receivedToolCalls = [];')),
    )

    for (const required of [
      'receivedToolCalls = []',
      'fullContent = ""',
      'rawAccumulated = ""',
      'lastFinishReason = undefined',
      'clientToolCallBuffering = false',
      'clientSideThinkParsing = false',
      'serverSendsUsage = false',
      'currentEventType = ""',
      'seenResponsesApiEvents.clear()',
    ]) {
      expect(branch).toContain(required)
    }
  })

  it('panel max tool iterations caps tool loops', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')
    const branch = source.slice(
      source.indexOf('const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10;'),
      source.indexOf('if (toolIteration > 0 || collectedToolStatuses.length > 0)'),
    )

    expect(branch).toContain('const MAX_TOOL_ITERATIONS = overrides?.maxToolIterations ?? 10')
    expect(branch).toContain('while (toolIteration < MAX_TOOL_ITERATIONS)')
    expect(branch).toContain('toolIteration++')
  })
})
