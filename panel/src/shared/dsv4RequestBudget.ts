export const DSV4_THINKING_MIN_TOKENS = 2048
export const DSV4_MAX_MIN_TOKENS = 8192

export function dsv4OutputBudget(
  maxTokens: unknown,
  _enableThinking: unknown,
  _detectedFamily?: string,
  _reasoningEffort?: unknown,
): number | undefined {
  return typeof maxTokens === 'number' && Number.isFinite(maxTokens) && maxTokens > 0
    ? Math.floor(maxTokens)
    : undefined
}

export function dsv4FinalizerTokens(
  _enableThinking: unknown,
  _detectedFamily?: string,
  _sessionFinalizerTokens?: unknown,
): number | undefined {
  return undefined
}
