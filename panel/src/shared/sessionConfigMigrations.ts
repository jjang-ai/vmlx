export const GENERATION_STARTUP_DEFAULTS_VERSION = 4
export const LEGACY_GENERIC_MAX_OUTPUT_TOKENS = new Set([4096, 12000, 12068, 32768])

function isMiniMaxSessionModel(modelPath?: string): boolean {
  const lower = String(modelPath || '').toLowerCase()
  return lower.includes('minimax-m2') || lower.includes('minimax_m2') || lower.includes('/minimax')
}

export function migrateLegacySessionStartupConfig(config: Record<string, any>, modelPath?: string): boolean {
  let changed = false
  if (LEGACY_GENERIC_MAX_OUTPUT_TOKENS.has(Number(config.maxTokens))) {
    config.maxTokens = 0
    config.generationStartupDefaultsVersion = GENERATION_STARTUP_DEFAULTS_VERSION
    changed = true
  }
  if (
    config.reasoningParser === 'minimax' ||
    config.reasoningParser === 'minimax_m2' ||
    config.reasoningParser === 'minimax_m2_5'
  ) {
    config.reasoningParser = 'minimax_m2'
    changed = true
  }
  if (isMiniMaxSessionModel(modelPath) && config.reasoningParser === 'qwen3') {
    config.reasoningParser = 'minimax_m2'
    changed = true
  }
  return changed
}
