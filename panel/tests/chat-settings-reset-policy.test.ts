import { describe, expect, it } from 'vitest'

import {
  buildChatSettingsResetOverrides,
  type ChatSettingsResetOverrides,
} from '../src/shared/chatSettingsResetPolicy'

describe('chat settings reset policy', () => {
  it('keeps tool and coding controls but clears sticky prompt, api, reasoning, stop, and stale sampler fields', () => {
    const result = buildChatSettingsResetOverrides(
      {
        temperature: 1,
        topP: 1,
        topK: 1,
        minP: 0.5,
        repeatPenalty: 1.3,
        maxTokens: 128,
        maxThinkingTokens: 4096,
        stopSequences: '<bad>',
        systemPrompt: 'VERY LONG STICKY PROMPT SHOULD NOT SURVIVE RESET',
        wireApi: 'completions',
        enableThinking: true,
        reasoningEffort: 'max',
        workingDirectory: '/tmp/vmlx-tools',
        builtinToolsEnabled: true,
        webSearchEnabled: true,
        braveSearchEnabled: false,
        fetchUrlEnabled: true,
        fileToolsEnabled: false,
        searchToolsEnabled: true,
        shellEnabled: true,
        gitEnabled: false,
        utilityToolsEnabled: true,
        hideToolStatus: true,
        maxToolIterations: 5,
        toolResultMaxChars: 12345,
      },
      {
        temperature: 0.42,
        topP: 0.77,
        topK: 33,
        maxNewTokens: 2048,
        repeatPenalty: 1.07,
      },
    )

    expect(result).toMatchObject({
      temperature: 0.42,
      topP: 0.77,
      topK: 33,
      repeatPenalty: 1.07,
      workingDirectory: '/tmp/vmlx-tools',
      builtinToolsEnabled: true,
      webSearchEnabled: true,
      braveSearchEnabled: false,
      fetchUrlEnabled: true,
      fileToolsEnabled: false,
      searchToolsEnabled: true,
      shellEnabled: true,
      gitEnabled: false,
      utilityToolsEnabled: true,
      hideToolStatus: true,
      maxToolIterations: 5,
      toolResultMaxChars: 12345,
    })
    expect(result.systemPrompt).toBeUndefined()
    expect(result.wireApi).toBeUndefined()
    expect(result.stopSequences).toBeUndefined()
    expect(result.enableThinking).toBeUndefined()
    expect(result.reasoningEffort).toBeUndefined()
    expect(result.minP).toBeUndefined()
    expect(result.maxTokens).toBeUndefined()
    expect(result.maxThinkingTokens).toBeUndefined()
  })

  it('does not turn model max_new_tokens into a sticky per-chat max_tokens override', () => {
    const result = buildChatSettingsResetOverrides(
      { maxTokens: 128 } satisfies ChatSettingsResetOverrides,
      { maxNewTokens: 2048 },
    )

    expect(result.maxTokens).toBeUndefined()
  })

  it('does not turn model maxThinkingTokens into a sticky per-chat thinking budget override', () => {
    const result = buildChatSettingsResetOverrides(
      { maxThinkingTokens: 128 } satisfies ChatSettingsResetOverrides,
      { maxThinkingTokens: 2048 },
    )

    expect(result.maxThinkingTokens).toBeUndefined()
  })

  it('does not invent tool keys when they were unset', () => {
    const result = buildChatSettingsResetOverrides(
      { temperature: 1 } satisfies ChatSettingsResetOverrides,
      {},
    )

    expect(result).toEqual({})
  })
})
