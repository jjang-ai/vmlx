export const DSV4_THINKING_MIN_TOKENS = 2048
export const DSV4_MAX_MIN_TOKENS = 8192
export const GEMMA4_REASONING_REQUIRED_TOOL_MIN_TOKENS = 512

function isGemma4Family(detectedFamily?: string): boolean {
  return detectedFamily === 'gemma4' || detectedFamily === 'gemma4-text'
}

export function dsv4OutputBudget(
  maxTokens: unknown,
  enableThinking: unknown,
  detectedFamily?: string,
  _reasoningEffort?: unknown,
  builtinToolsEnabled?: boolean,
): number | undefined {
  const explicit = typeof maxTokens === 'number' && Number.isFinite(maxTokens) && maxTokens > 0
    ? Math.floor(maxTokens)
    : undefined
  if (
    explicit != null &&
    explicit < GEMMA4_REASONING_REQUIRED_TOOL_MIN_TOKENS &&
    enableThinking === true &&
    builtinToolsEnabled === true &&
    isGemma4Family(detectedFamily)
  ) {
    return GEMMA4_REASONING_REQUIRED_TOOL_MIN_TOKENS
  }
  return explicit
}

export function dsv4FinalizerTokens(
  _enableThinking: unknown,
  _detectedFamily?: string,
  _sessionFinalizerTokens?: unknown,
): number | undefined {
  return undefined
}
