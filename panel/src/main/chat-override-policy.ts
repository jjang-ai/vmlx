export interface ChatOverridePolicyInput {
  chatId: string
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  maxTokens?: number
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

export { NEW_CHAT_TOOL_INHERIT_KEYS }
