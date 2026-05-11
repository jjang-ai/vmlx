import { readFileSync } from 'fs'
import { describe, expect, it } from 'vitest'
import {
  buildChatSettingsCompatibilityWarnings,
  type ChatSettingsCompatibilityInput,
} from '../src/renderer/src/components/chat/chatSettingsCompatibility'

function warnings(input: Partial<ChatSettingsCompatibilityInput>): string[] {
  return buildChatSettingsCompatibilityWarnings({
    messageCount: 3,
    currentModelPath: '/models/qwen',
    overrides: {},
    ...input,
  })
}

describe('chat settings cross-family compatibility warnings', () => {
  it('does not warn for empty chats', () => {
    expect(warnings({
      messageCount: 0,
      savedChatModelPath: '/models/old',
      currentModelPath: '/models/new',
      overrides: { enableThinking: true, reasoningEffort: 'medium' },
    })).toEqual([])
  })

  it('warns when a chat with history is opened against a different model path', () => {
    expect(warnings({
      savedChatModelPath: '/models/qwen36',
      currentModelPath: '/models/nemotron',
    })).toContain('This chat was started on qwen36 but is now attached to nemotron. Review saved per-chat settings before continuing.')
  })

  it('warns when saved Thinking On reaches a model with no reasoning parser', () => {
    expect(warnings({
      reasoningParser: undefined,
      overrides: { enableThinking: true },
    })).toContain('Saved Thinking On cannot take effect because this model has no detected reasoning parser.')
  })

  it('warns when stale reasoning effort reaches a parser that does not use effort levels', () => {
    expect(warnings({
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'medium' },
    })).toContain('Saved reasoning effort "medium" is not used by qwen3. Reset the chat setting or switch to Auto.')
  })

  it('allows Hy3 low/high effort even though it reuses the qwen3 text parser', () => {
    expect(warnings({
      detectedFamily: 'hy3',
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'low' },
    })).toEqual([])
  })

  it('warns when Mistral carries a non-high effort from another model family', () => {
    expect(warnings({
      reasoningParser: 'mistral',
      overrides: { reasoningEffort: 'medium' },
    })).toContain('Saved reasoning effort "medium" is not supported by Mistral. Use Auto or High.')
  })

  it('allows Hy3 low/high reasoning effort but warns on medium', () => {
    expect(warnings({
      detectedFamily: 'hy3',
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'low' },
    })).toEqual([])
    expect(warnings({
      detectedFamily: 'hy3',
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'high' },
    })).toEqual([])
    expect(warnings({
      detectedFamily: 'hy3',
      reasoningParser: 'qwen3',
      overrides: { reasoningEffort: 'medium' },
    })).toContain('Saved reasoning effort "medium" is not supported by Hy3. Use Auto or High.')
  })

  it('warns when built-in tools are enabled without a detected tool parser', () => {
    expect(warnings({
      toolParser: undefined,
      overrides: { builtinToolsEnabled: true },
    })).toContain('Built-in tools are enabled, but this model has no detected tool parser. Tool calls may not round-trip.')
  })

  it('disables Thinking buttons when no reasoning parser is detected', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain("const thinkingSupported = detectedFamily === 'deepseek-v4' || !!reasoningParser")
    expect(source).toContain("const showReasoningEffort = detectedFamily === 'hy3' || reasoningParser === 'openai_gptoss' || reasoningParser === 'mistral'")
    expect(source).toContain('const displayedEnableThinking = thinkingSupported ? overrides.enableThinking : undefined')
    expect(source).toContain('disabled={!thinkingSupported}')
  })

  it('shows Hy3 low/high effort controls without exposing medium', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain("detectedFamily === 'hy3' || reasoningParser === 'openai_gptoss' || reasoningParser === 'mistral'")
    expect(source).toContain("const showMediumEffort = reasoningParser !== 'mistral' && detectedFamily !== 'hy3'")
  })
})
