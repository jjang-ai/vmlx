export interface ChatOverridePolicyInput {
  chatId: string
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  maxTokens?: number
  maxThinkingTokens?: number
  repeatPenalty?: number
  systemPrompt?: string
  stopSequences?: string
  wireApi?: string
  maxToolIterations?: number
  builtinToolsEnabled?: boolean
  workingDirectory?: string
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

const NEW_CHAT_TOOL_INHERIT_KEYS = [
  'builtinToolsEnabled',
  'webSearchEnabled',
  'braveSearchEnabled',
  'fetchUrlEnabled',
  'fileToolsEnabled',
  'searchToolsEnabled',
  'shellEnabled',
  'gitEnabled',
  'utilityToolsEnabled',
  'maxToolIterations',
  'workingDirectory',
  'hideToolStatus',
  'toolResultMaxChars',
] as const

export function buildNewChatInheritedOverrides<T extends ChatOverridePolicyInput>(
  existing: T,
  previous?: Partial<ChatOverridePolicyInput> | null,
): T {
  if (!previous) return existing

  const merged: ChatOverridePolicyInput = { ...existing }
  for (const key of NEW_CHAT_TOOL_INHERIT_KEYS) {
    const value = previous[key]
    if (value !== undefined) {
      ;(merged as any)[key] = value
    }
  }
  merged.chatId = existing.chatId
  return merged as T
}

const sanitizeFiniteNumber = (value: unknown, lo: number, hi: number): number | undefined => {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return undefined
  }
  return Math.max(lo, Math.min(hi, value))
}

const sanitizePositiveInteger = (value: unknown, hi: number): number | undefined => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
    return undefined
  }
  return Math.max(1, Math.min(hi, Math.floor(value)))
}

export function sanitizeChatOverrides<T extends ChatOverridePolicyInput>(overrides: T): T {
  const sanitized: ChatOverridePolicyInput = { ...overrides }
  if (sanitized.temperature != null) {
    const value = sanitizeFiniteNumber(sanitized.temperature, 0, 10)
    if (value == null) delete sanitized.temperature
    else sanitized.temperature = value
  }
  if (sanitized.topP != null) {
    const value = sanitizeFiniteNumber(sanitized.topP, 0, 1)
    if (value == null) delete sanitized.topP
    else sanitized.topP = value
  }
  if (sanitized.topK != null) {
    const value = sanitizeFiniteNumber(sanitized.topK, 0, 1000)
    if (value == null) delete sanitized.topK
    else sanitized.topK = value
  }
  if (sanitized.minP != null) {
    const value = sanitizeFiniteNumber(sanitized.minP, 0, 1)
    if (value == null) delete sanitized.minP
    else sanitized.minP = value
  }
  if (sanitized.maxTokens != null) {
    const maxTokens = sanitizePositiveInteger(sanitized.maxTokens, 1000000)
    if (maxTokens == null) {
      delete sanitized.maxTokens
    } else {
      sanitized.maxTokens = maxTokens
    }
  }
  if (sanitized.maxThinkingTokens != null) {
    const maxThinkingTokens = sanitizePositiveInteger(sanitized.maxThinkingTokens, 1000000)
    if (maxThinkingTokens == null) {
      delete sanitized.maxThinkingTokens
    } else {
      sanitized.maxThinkingTokens = maxThinkingTokens
    }
  }
  if (sanitized.repeatPenalty != null) {
    const value = sanitizeFiniteNumber(sanitized.repeatPenalty, 0, 10)
    if (value == null) delete sanitized.repeatPenalty
    else sanitized.repeatPenalty = value
  }
  if (sanitized.maxToolIterations != null) {
    const value = sanitizeFiniteNumber(sanitized.maxToolIterations, 1, 100)
    if (value == null) delete sanitized.maxToolIterations
    else sanitized.maxToolIterations = value
  }
  if (sanitized.toolResultMaxChars != null) {
    const value = sanitizeFiniteNumber(sanitized.toolResultMaxChars, 100, 500000)
    if (value == null) delete sanitized.toolResultMaxChars
    else sanitized.toolResultMaxChars = value
  }
  return sanitized as T
}

export { NEW_CHAT_TOOL_INHERIT_KEYS }
