import { statSync } from 'node:fs'
import {
  TOP_K_MAX,
  sanitizeMinPOverride,
  sanitizeOpenAiTokenPenaltyOverride,
  sanitizeRepetitionPenaltyOverride,
  sanitizeTemperatureOverride,
  sanitizeTopKOverride,
  sanitizeTopPOverride,
} from '../shared/samplingParameterDomain'
import { REASONING_EFFORT_LEVELS } from '../shared/reasoningEffortPolicy'

export interface ChatOverridePolicyInput {
  chatId: string
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
  wireApi?: string
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

const CHAT_OVERRIDE_STRING_KEYS = [
  'systemPrompt',
  'stopSequences',
  'workingDirectory',
] as const

const CHAT_OVERRIDE_BOOLEAN_KEYS = [
  'builtinToolsEnabled',
  'enableThinking',
  'hideToolStatus',
  'webSearchEnabled',
  'braveSearchEnabled',
  'fetchUrlEnabled',
  'fileToolsEnabled',
  'searchToolsEnabled',
  'shellEnabled',
  'gitEnabled',
  'utilityToolsEnabled',
] as const

const CHAT_OVERRIDE_WIRE_APIS = new Set(['completions', 'responses'])
// Derived from the canonical list, never hand-rolled: a local copy silently
// DROPS any newly supported level on save. That is what kept Muse's 'xhigh'
// from persisting even though the button rendered and the engine honoured it.
const CHAT_OVERRIDE_REASONING_EFFORTS = new Set<string>(REASONING_EFFORT_LEVELS)

export const CHAT_TOP_K_HARD_MAX = TOP_K_MAX

function isUsableInheritedWorkingDirectory(value: unknown): value is string {
  if (typeof value !== 'string' || !value.trim()) return false
  try {
    return statSync(value).isDirectory()
  } catch {
    return false
  }
}

export function buildNewChatInheritedOverrides<T extends ChatOverridePolicyInput>(
  existing: T,
  previous?: Partial<ChatOverridePolicyInput> | null,
): T {
  if (!previous) return existing

  const merged: ChatOverridePolicyInput = { ...existing }
  for (const key of NEW_CHAT_TOOL_INHERIT_KEYS) {
    const value = previous[key]
    // A working directory is a live filesystem capability, not just a saved
    // preference. Temporary project folders can disappear between chats. Do
    // not seed a clean chat with a dead path and wait for its first file tool
    // to fail; leave the field unset so the settings surface names the missing
    // requirement before the model can spend a tool iteration on it.
    if (key === 'workingDirectory' && !isUsableInheritedWorkingDirectory(value)) {
      continue
    }
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
  const source = overrides as unknown as Record<string, unknown>
  const sanitized: ChatOverridePolicyInput = { chatId: overrides.chatId }
  const sanitizedRecord = sanitized as unknown as Record<string, unknown>
  const assignNumber = (
    key: keyof ChatOverridePolicyInput,
    sanitizer: (value: unknown) => number | undefined,
  ): void => {
    const value = sanitizer(source[key])
    if (value != null) sanitizedRecord[key] = value
  }

  assignNumber('temperature', sanitizeTemperatureOverride)
  assignNumber('topP', sanitizeTopPOverride)
  assignNumber('topK', sanitizeTopKOverride)
  assignNumber('minP', sanitizeMinPOverride)
  assignNumber('maxTokens', value => sanitizePositiveInteger(value, 1000000))
  assignNumber('maxThinkingTokens', value => sanitizePositiveInteger(value, 1000000))
  assignNumber('repeatPenalty', sanitizeRepetitionPenaltyOverride)
  assignNumber('frequencyPenalty', sanitizeOpenAiTokenPenaltyOverride)
  assignNumber('presencePenalty', sanitizeOpenAiTokenPenaltyOverride)
  assignNumber('maxToolIterations', value => sanitizeFiniteNumber(value, 1, 100))
  assignNumber('toolResultMaxChars', value => sanitizeFiniteNumber(value, 100, 500000))

  for (const key of CHAT_OVERRIDE_STRING_KEYS) {
    if (typeof source[key] === 'string') {
      sanitizedRecord[key] = source[key]
    }
  }
  for (const key of CHAT_OVERRIDE_BOOLEAN_KEYS) {
    if (typeof source[key] === 'boolean') {
      sanitizedRecord[key] = source[key]
    }
  }
  if (
    typeof source.wireApi === 'string' &&
    CHAT_OVERRIDE_WIRE_APIS.has(source.wireApi)
  ) {
    sanitized.wireApi = source.wireApi
  }
  if (
    typeof source.reasoningEffort === 'string' &&
    CHAT_OVERRIDE_REASONING_EFFORTS.has(source.reasoningEffort)
  ) {
    sanitized.reasoningEffort = source.reasoningEffort
  }

  if (source.thinkingMode === 'adaptive') {
    if (typeof source.enableThinking === 'boolean') {
      throw new Error('Adaptive thinking conflicts with an explicit On/Off choice.')
    }
    sanitized.thinkingMode = 'adaptive'
  }

  return sanitized as T
}

export function sanitizeChatProfileOverrides(value: unknown): Record<string, any> {
  const record =
    value && typeof value === 'object' && !Array.isArray(value)
      ? value as Record<string, any>
      : {}
  const sanitized = sanitizeChatOverrides({
    ...record,
    // Profiles have no chat owner. Put the temporary identity after untrusted
    // profile data so a stale/malicious chatId cannot replace it.
    chatId: '__profile__',
  })
  const { chatId: _profileIdentity, ...profile } = sanitized
  return profile
}

export { NEW_CHAT_TOOL_INHERIT_KEYS }
