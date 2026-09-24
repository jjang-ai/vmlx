export const REASONING_EFFORT_LEVELS = ['low', 'medium', 'high', 'xhigh', 'max'] as const

export type ReasoningEffort = typeof REASONING_EFFORT_LEVELS[number]
export type RemoteReasoningFormat = 'vmlx' | 'openai' | 'openrouter'

export function remoteReasoningFormatForUrl(url?: string): RemoteReasoningFormat {
  try {
    if (new URL(url || '').hostname === 'openrouter.ai') return 'openrouter'
  } catch { /* Unknown endpoints use the explicitly selected standard wire. */ }
  return 'openai'
}

const REASONING_EFFORT_SET = new Set<string>(REASONING_EFFORT_LEVELS)

export function normalizeReasoningEffort(value: unknown): ReasoningEffort | undefined {
  if (typeof value !== 'string') return undefined
  const normalized = value.trim().toLowerCase()
  return REASONING_EFFORT_SET.has(normalized)
    ? normalized as ReasoningEffort
    : undefined
}

export function normalizeReasoningEffortLevels(
  value: unknown,
): ReasoningEffort[] | undefined {
  if (!Array.isArray(value)) return undefined
  const levels: ReasoningEffort[] = []
  for (const entry of value) {
    const normalized = normalizeReasoningEffort(entry)
    if (normalized && !levels.includes(normalized)) levels.push(normalized)
  }
  return levels
}

export interface ReasoningRequestFieldsInput {
  thinkingMode?: 'adaptive'
  supportsAdaptiveThinking?: boolean
  enableThinking?: boolean
  reasoningEffort?: unknown
  isRemote: boolean
  sessionHasReasoningParser: boolean
  detectedFamily?: string
  supportedReasoningEfforts?: readonly ReasoningEffort[]
  remoteReasoningFormat?: RemoteReasoningFormat
  wireApi?: 'completions' | 'responses'
  defaultReasoningEffort?: ReasoningEffort
}

export function resolveReasoningEffortForRequest(
  input: ReasoningRequestFieldsInput,
): ReasoningEffort | undefined {
  const effort = normalizeReasoningEffort(input.reasoningEffort)
  if (!effort || input.enableThinking === false) {
    return undefined
  }

  if (
    input.supportedReasoningEfforts !== undefined &&
    !input.supportedReasoningEfforts.includes(effort)
  ) {
    const choices = input.supportedReasoningEfforts.length > 0
      ? `Auto or ${input.supportedReasoningEfforts.map(level => level[0].toUpperCase() + level.slice(1)).join('/')}`
      : 'Auto'
    throw new Error(
      `Reasoning effort "${effort}" is not supported by the loaded bundle. Choose ${choices}.`,
    )
  }

  if (input.detectedFamily === 'hy3' && input.enableThinking !== true) {
    return undefined
  }
  return input.isRemote || input.sessionHasReasoningParser || input.detectedFamily === 'deepseek-v4'
    ? effort
    : undefined
}

/**
 * Apply the reasoning fields used by the real chat request builder. The same
 * builder is reused for the initial HTTP request and every built-in-tool
 * continuation, so explicit bundle effort stays stable across both requests.
 */
export function applyReasoningRequestFields(
  body: Record<string, any>,
  input: ReasoningRequestFieldsInput,
): void {
  // Resolve (and validate) before mutating the body. A stale unsupported saved
  // effort fails clearly instead of partially serializing or falling back to
  // the bundle default under a different label.
  const effort = resolveReasoningEffortForRequest(input)
  if (input.thinkingMode === 'adaptive') {
    if (input.supportsAdaptiveThinking !== true ||
        (input.isRemote && input.remoteReasoningFormat !== 'vmlx')) {
      throw new Error('The loaded runtime does not advertise native adaptive thinking.')
    }
    if (input.enableThinking !== undefined) {
      throw new Error('Adaptive thinking conflicts with an explicit On/Off choice.')
    }
    body.thinking_mode = 'adaptive'
    body.chat_template_kwargs = { ...(body.chat_template_kwargs || {}), thinking_mode: 'adaptive' }
    if (effort) body.reasoning_effort = effort
    return
  }


  // A remote API returns structured reasoning; it does not need a local text
  // parser. Keep provider fields separate from vMLX template extensions. This
  // same function is used again for every tool-result continuation.
  if (input.isRemote && input.remoteReasoningFormat && input.remoteReasoningFormat !== 'vmlx') {
    if (input.remoteReasoningFormat === 'openrouter' && input.wireApi !== 'responses') {
      if (input.enableThinking !== undefined || effort) {
        body.reasoning = {
          ...(body.reasoning || {}),
          ...(input.enableThinking !== undefined ? { enabled: input.enableThinking } : {}),
          ...(effort ? { effort } : {}),
        }
      }
      return
    }
    const nativeEffort = input.enableThinking === false ? 'none'
      : effort ?? (input.enableThinking === true ? input.defaultReasoningEffort : undefined)
    // Empty reasoning objects are provider Auto, not a portable Thinking On.
    // Do not invent a tier or silently ignore the user's explicit choice.
    if (input.enableThinking === true && !nativeEffort) {
      throw new Error('Choose an explicit reasoning effort in Chat Settings for this remote API. Auto leaves the provider default unchanged.')
    }
    if (nativeEffort) {
      if (input.wireApi === 'responses') body.reasoning = { ...(body.reasoning || {}), effort: nativeEffort }
      else body.reasoning_effort = nativeEffort
    }
    return
  }

  if (input.enableThinking !== undefined) {
    body.enable_thinking = input.enableThinking
  }
  if (!input.isRemote && body.enable_thinking !== undefined) {
    body.chat_template_kwargs = {
      ...(body.chat_template_kwargs || {}),
      enable_thinking: body.enable_thinking,
    }
  }
  if (effort) {
    if (input.isRemote && input.wireApi === 'responses') {
      body.reasoning = { ...(body.reasoning || {}), effort }
    } else body.reasoning_effort = effort
  }

  // DSV4's versioned encoder is controlled by enable_thinking plus the exact
  // bundle effort. Do not add generic thinking_mode=reasoning here: the shared
  // API alias infers `medium`, which is not a 0731 tier. The DSV4 server path
  // maps enable_thinking=false to native Chat and validates the explicit effort.
  if (!input.isRemote && input.detectedFamily !== 'deepseek-v4') {
    if (body.enable_thinking === false) body.thinking_mode = 'instruct'
    else if (body.enable_thinking === true && effort === 'max') body.thinking_mode = 'max'
    // On with model-default effort must leave the tier absent. The legacy
    // `reasoning` alias otherwise inserts medium during API validation,
    // overriding the bundle's native default even though no tier was chosen.
    else if (body.enable_thinking === true && effort) body.thinking_mode = 'reasoning'
  }
}
