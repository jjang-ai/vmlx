import { describe, expect, it } from 'vitest'
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
})
