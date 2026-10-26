import { readFileSync } from 'fs'
import { describe, expect, it } from 'vitest'
import {
  buildChatSettingsCompatibilityWarnings,
  type ChatSettingsCompatibilityInput,
} from '../src/renderer/src/components/chat/chatSettingsCompatibility'
import { shouldWarnDsv4TopP } from '../src/shared/samplingParameterDomain'

function warnings(input: Partial<ChatSettingsCompatibilityInput>): string[] {
  return buildChatSettingsCompatibilityWarnings({
    messageCount: 3,
    currentModelPath: '/models/qwen',
    overrides: {},
    ...input,
  })
}

describe('chat settings cross-family compatibility warnings', () => {
  it('warns only when a known DSV4 Top P differs from 0.95', () => {
    expect(shouldWarnDsv4TopP('deepseek-v4', 0.95)).toBe(false)
    expect(shouldWarnDsv4TopP('deepseek-v4', 0.9500005)).toBe(false)
    expect(shouldWarnDsv4TopP('deepseek-v4', 1)).toBe(true)
    expect(shouldWarnDsv4TopP('qwen3', 1)).toBe(false)
    expect(shouldWarnDsv4TopP('deepseek-v4', undefined)).toBe(false)
    // DeepSeek-V4.1 rides the deepseek-v4 profile but its release default is top_p 1.0
    expect(shouldWarnDsv4TopP('deepseek-v4', 1, { deepseekVariant: 'v4.1' })).toBe(false)
    expect(shouldWarnDsv4TopP('deepseek-v4', 1, {})).toBe(true)
  })

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

  it('keeps local parser gating and uses advertised capabilities for remote controls', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('const [detectedSupportsThinking, setDetectedSupportsThinking]')
    expect(source).toContain('const resolvedReasoningParser = resolveEffectiveReasoningParser({')
    expect(source).toContain("const thinkingSupported = isRemote ? detectedSupportsThinking !== false : resolvedReasoningParser !== 'none' && (")
    expect(source).toContain('reasoningParserIsEnabled(resolvedReasoningParser)')
    expect(source).toContain('const showReasoningEffort = selectableReasoningEfforts.length > 0')
    expect(source).toContain('const displayedEnableThinking = thinkingSupported ? displayedOverrides.enableThinking : undefined')
    expect(source).toContain('disabled={!thinkingSupported}')
  })

  it('shows Hy3 low/high effort controls without exposing medium', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain("detectedFamily === 'hy3'")
    expect(source).toContain("? ['low', 'high']")
    expect(source).toContain('selectableReasoningEfforts.map(effort => (')
  })

  it('hides Thinking Off and exposes native effort levels when instruct mode is unsupported', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')
    const ipc = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const thinkingOffSupported = detectedSupportsInstructMode !== false')
    expect(source).toContain('{thinkingOffSupported && (')
    expect(source).toContain("'chat.settings.thinkingNativeOnlyHelp'")
    expect(source).toContain('setDetectedReasoningEfforts(detected?.supportedReasoningEfforts)')
    expect(ipc).toContain('supportsInstructMode === false && overrides?.enableThinking === false')
    expect(ipc).toContain('if (supportsInstructMode === false) return;')
  })

  it('renders DSV4 effort buttons from bundle metadata instead of a family-hardcoded tier list', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('detectedReasoningEfforts ?? (')
    expect(source).toContain('setDetectedDefaultReasoningEffort(detected?.defaultReasoningEffort)')
    expect(source).toContain('data-reasoning-effort={effort}')
    expect(source).toContain('onClick={() => updateThinkingMode(true, effort)}')
    expect(source).toContain("return t('chat.settings.effortMax')")
    expect(source).not.toContain('const dsv4MaxEnabled =')
    expect(source).not.toContain("sessionConfig?.dsv4ForceDirect")
    expect(source).not.toContain("sessionConfig?.dsv4RawMax === true")
  })

  it('shows the DSV4 Top P advisory after model hydration without changing the effective value', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('const dsv4TopPMismatch =')
    expect(source).toContain('hydrationCurrent && shouldWarnDsv4TopP(')
    expect(source).toContain('detectedFamily,')
    expect(source).toContain('displayedTopP,')
    expect(source).toContain('dsv4TopPMismatch && (')
    expect(source).toContain('data-vmlx-warning="dsv4-top-p-advisory"')
    expect(source).toContain("t('common.dsv4TopPAdvisory')")
    expect(source).not.toContain("update('topP', 0.95)")
    expect(source).not.toContain("setOverrides({ topP: 0.95 })")
  })

  it('distinguishes model-default effort from the separate Auto thinking mode', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('onClick={() => updateThinkingMode(undefined, undefined)}')
    expect(source).toContain('displayedEnableThinking == null')
    expect(source).toContain("t('chat.settings.effortDefault', {")
    expect(source).toContain("t('chat.settings.effortDefaultNoValue')")
    expect(source).toContain('reasoningEffortLabel(detectedDefaultReasoningEffort)')
  })

  it('accepts only the exact DSV4-0731 sidecar levels and flags stale Medium', () => {
    const contract = {
      detectedFamily: 'deepseek-v4',
      reasoningParser: 'deepseek_r1',
      supportedReasoningEfforts: ['low', 'high', 'max'] as const,
    }
    for (const effort of ['low', 'high', 'max'] as const) {
      expect(warnings({
        ...contract,
        supportedReasoningEfforts: [...contract.supportedReasoningEfforts],
        overrides: { reasoningEffort: effort },
      })).toEqual([])
    }
    expect(warnings({
      ...contract,
      supportedReasoningEfforts: [...contract.supportedReasoningEfforts],
      overrides: { reasoningEffort: 'medium' },
    })).toContain('Saved reasoning effort "medium" is not supported by this DSV4 bundle. Use Auto or Low/High/Max.')
  })

  it('does not silently mutate DSV4 output budgets when the user changes reasoning mode', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).not.toContain('DSV4_THINKING_MIN_TOKENS')
    expect(source).not.toContain('DSV4_MAX_MIN_TOKENS')
    expect(source).not.toContain('next.maxTokens = Math.max')
  })

  it('main IPC refuses stale local Thinking On when fresh detection has no reasoning parser', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    expect(source).toContain('const effectiveReasoningParser = resolveEffectiveReasoningParser({')
    expect(source).toContain('sessionHasReasoningParser = reasoningParserIsEnabled(')
    expect(source).toContain('supportsThinking: detected.supportsThinking,')
    expect(source).toContain('const effectiveEnableThinkingOverride =')
    expect(source).toContain('!sessionHasReasoningParser')
    expect(source).toContain('chatDetectedFamily !== "deepseek-v4"')
  })

  it('gates the Max Thinking Tokens field on engine-honoring or template budget support', () => {
    const source = readFileSync('src/renderer/src/components/chat/ChatSettings.tsx', 'utf8')

    expect(source).toContain('const [supportsThinkingBudget, setSupportsThinkingBudget]')
    expect(source).toContain('setSupportsThinkingBudget(detected?.supportsThinkingBudget)')
    expect(source).toContain('{(supportsThinkingBudget === true || thinkingBudgetSupported === true) && displayedEnableThinking !== false && (')
  })

  it('main IPC sends top-level max_thinking_tokens only for engine-honoring families', () => {
    const source = readFileSync('src/main/ipc/chat.ts', 'utf8')

    // The whole-block gate now keys off the registry/template budget capability,
    // and the deepseek-v4 special-case is gone from applyLocalThinkingBudget.
    expect(source).toContain('if (!(supportsThinkingBudget === true || thinkingBudgetSupported === true)) {')
    expect(source).toContain('supportsThinkingBudget = detected.supportsThinkingBudget;')
    expect(source).not.toContain('if (!sessionHasReasoningParser && chatDetectedFamily !== "deepseek-v4") {')
    // Template kwarg stays TEMPLATE-side only.
    expect(source).toContain('if (thinkingBudgetSupported !== false) {')
  })
})
