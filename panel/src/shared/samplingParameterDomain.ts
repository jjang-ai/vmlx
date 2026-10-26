/** Shared Electron persistence/UI domains copied from the Python API validators. */
export const TEMPERATURE_MIN = 0
export const TEMPERATURE_MAX = 2
export const TOP_P_LOWER_EXCLUSIVE = 0
export const TOP_P_MAX = 1
export const TOP_P_UI_MIN = 0.01
export const TOP_K_MIN = 0
export const TOP_K_MAX = Number.MAX_SAFE_INTEGER
export const MIN_P_MIN = 0
export const MIN_P_MAX = 1
export const REPETITION_PENALTY_LOWER_EXCLUSIVE = 0
export const OPENAI_TOKEN_PENALTY_MIN = -2
export const OPENAI_TOKEN_PENALTY_MAX = 2
export const DSV4_RECOMMENDED_TOP_P = 0.95

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function finiteInClosedRange(value: unknown, min: number, max: number): number | undefined {
  const number = finiteNumber(value)
  return number != null && number >= min && number <= max ? number : undefined
}

export function sanitizeTemperatureOverride(value: unknown): number | undefined {
  return finiteInClosedRange(value, TEMPERATURE_MIN, TEMPERATURE_MAX)
}

export function sanitizeTopPOverride(value: unknown): number | undefined {
  const number = finiteNumber(value)
  return number != null && number > TOP_P_LOWER_EXCLUSIVE && number <= TOP_P_MAX
    ? number
    : undefined
}

export function sanitizeTopKOverride(value: unknown): number | undefined {
  const number = finiteNumber(value)
  return number != null &&
    Number.isSafeInteger(number) &&
    number >= TOP_K_MIN &&
    number <= TOP_K_MAX
    ? number
    : undefined
}

export function sanitizeMinPOverride(value: unknown): number | undefined {
  return finiteInClosedRange(value, MIN_P_MIN, MIN_P_MAX)
}

export function sanitizeRepetitionPenaltyOverride(value: unknown): number | undefined {
  const number = finiteNumber(value)
  return number != null && number > REPETITION_PENALTY_LOWER_EXCLUSIVE
    ? number
    : undefined
}

export function sanitizeOpenAiTokenPenaltyOverride(value: unknown): number | undefined {
  return finiteInClosedRange(value, OPENAI_TOKEN_PENALTY_MIN, OPENAI_TOKEN_PENALTY_MAX)
}

export function shouldWarnDsv4TopP(
  family: unknown,
  topP: unknown,
  architectureHints?: Record<string, unknown> | null,
): boolean {
  // DeepSeek-V4.1 rides the deepseek-v4 family profile, but the 0.95 advice comes from DSV4-0731 looping
  // measurements; V4.1's release default is top_p 1.0.
  if (architectureHints?.deepseekVariant === 'v4.1') return false
  const number = finiteNumber(topP)
  return family === 'deepseek-v4' &&
    number != null &&
    Math.abs(number - DSV4_RECOMMENDED_TOP_P) > 0.000001
}
