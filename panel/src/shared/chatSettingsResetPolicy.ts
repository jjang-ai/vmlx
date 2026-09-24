export interface ChatSettingsResetOverrides {
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  maxTokens?: number
  maxThinkingTokens?: number
  repeatPenalty?: number
  frequencyPenalty?: number
  presencePenalty?: number
  systemPrompt?: string
  stopSequences?: string
  wireApi?: 'completions' | 'responses' | string
  maxToolIterations?: number
  builtinToolsEnabled?: boolean
  workingDirectory?: string
  thinkingMode?: 'adaptive'
  enableThinking?: boolean
  reasoningEffort?: string
  hideToolStatus?: boolean
  webSearchEnabled?: boolean
  braveSearchEnabled?: boolean
  fetchUrlEnabled?: boolean
  fileToolsEnabled?: boolean
  searchToolsEnabled?: boolean
  shellEnabled?: boolean
  toolResultMaxChars?: number
  gitEnabled?: boolean
  utilityToolsEnabled?: boolean
}

export interface ChatSettingsGenerationDefaults {
  doSample?: boolean
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
  maxNewTokens?: number
  maxThinkingTokens?: number
}

export const CHAT_SETTINGS_RESET_PRESERVE_KEYS = [
  'workingDirectory',
  'builtinToolsEnabled',
  'webSearchEnabled',
  'braveSearchEnabled',
  'fetchUrlEnabled',
  'fileToolsEnabled',
  'searchToolsEnabled',
  'shellEnabled',
  'gitEnabled',
  'utilityToolsEnabled',
  'hideToolStatus',
  'maxToolIterations',
  'toolResultMaxChars',
] as const

export function buildChatSettingsResetOverrides<T extends ChatSettingsResetOverrides>(
  overrides: T,
  _genDefaults?: ChatSettingsGenerationDefaults | null,
): Partial<T> {
  const reset: Partial<T> = {}

  for (const key of CHAT_SETTINGS_RESET_PRESERVE_KEYS) {
    const value = overrides[key]
    if (value !== undefined && value !== null) {
      ;(reset as any)[key] = value
    }
  }

  // Model sampling defaults are display/runtime inheritance, not per-chat
  // overrides. Persisting a copy here makes Reset look correct only until the
  // bundle changes, then silently sends stale values ahead of the new config.
  // Leave all sampler fields absent; ChatSettings renders the current bundle
  // defaults and the engine resolves the same live metadata for the request.
  // Keep max_new_tokens model-owned. Saving it as a per-chat maxTokens
  // override makes the UI enforce a sticky output cap even though the server
  // can resolve the same bundle default when the request omits max_tokens.
  // Keep maxThinkingTokens model-owned too. The chat setting is an explicit
  // template budget override, not a profile/reset default.

  return reset
}
