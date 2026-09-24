import {
  applyEffectiveSessionGenerationDefaults,
  type GenerationDefaultsLike,
} from './effectiveGenerationDefaults'

export interface ChatSettingsGenerationDefaults {
  doSample?: boolean
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
  maxNewTokens?: number
  maxThinkingTokens?: number
  thinkingBudgetSupported?: boolean
}

export interface ChatSettingsDetectedConfig {
  family?: string
  toolParser?: string
  reasoningParser?: string
  supportsThinking?: boolean
  supportsAdaptiveThinking?: boolean
  supportsInstructMode?: boolean
  /** Does the family's chat template actually READ `enable_thinking`?
   * Distinct from supportsThinking (does it reason at all). Muse reasons but
   * reads only `reasoning_strength`, so an Auto/On toggle rendered off
   * supportsThinking is inert — MEASURED byte-identical output for both. */
  honorsEnableThinking?: boolean
  supportedReasoningEfforts?: Array<'low' | 'medium' | 'high' | 'xhigh' | 'max'>
  defaultReasoningEffort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max'
  supportsThinkingBudget?: boolean
  nativeMtp?: { supported?: boolean }
}

export interface ChatSettingsHydrationLoaders<TOverrides extends object> {
  overrides: () => Promise<TOverrides | null | undefined>
  generationDefaults: () => Promise<ChatSettingsGenerationDefaults | null | undefined>
  detectedConfig: () => Promise<ChatSettingsDetectedConfig | null | undefined>
}

export interface ChatSettingsCompatibilityLoaders {
  chat: () => Promise<{ modelPath?: string } | null | undefined>
  messages: () => Promise<unknown>
}

export type ChatSettingsDefaultsState =
  | 'ready'
  | 'session-fallback'
  | 'engine-fallback'
  | 'unavailable'

export interface ChatSettingsHydrationResult<TOverrides extends object> {
  overrides: Partial<TOverrides>
  overridesLoaded: boolean
  modelDefaults: GenerationDefaultsLike
  defaultsState: ChatSettingsDefaultsState
  partialFailure: boolean
  generationDefaults?: ChatSettingsGenerationDefaults
  detectedConfig?: ChatSettingsDetectedConfig
}

export interface ChatSettingsCompatibilityResult {
  partialFailure: boolean
  savedChatModelPath?: string
  messageCount: number
}

type ChatSettingsHydrationSettlements<TOverrides extends object> = {
  overrides: PromiseSettledResult<TOverrides | null | undefined>
  generationDefaults: PromiseSettledResult<ChatSettingsGenerationDefaults | null | undefined>
  detectedConfig: PromiseSettledResult<ChatSettingsDetectedConfig | null | undefined>
}

export const CHAT_SETTINGS_IPC_TIMEOUT_MS = 5_000

function configRecord(config: string | Record<string, unknown> | undefined): Record<string, unknown> {
  if (!config) return {}
  if (typeof config !== 'string') return config
  try {
    const parsed = JSON.parse(config)
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {}
  } catch {
    return {}
  }
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function positiveNumber(value: unknown): number | undefined {
  const number = finiteNumber(value)
  return number != null && number > 0 ? number : undefined
}

function hasSamplerDefaults(defaults: GenerationDefaultsLike): boolean {
  return (
    defaults.temperature != null ||
    defaults.topP != null ||
    defaults.topK != null ||
    defaults.minP != null ||
    defaults.repeatPenalty != null
  )
}

function hasAnyDefaults(defaults: GenerationDefaultsLike): boolean {
  return Object.values(defaults).some(value => value != null)
}

/**
 * Reconstruct the bundle-derived defaults already persisted with a session.
 * These values are display seeds only; the live bundle read still wins after
 * hydration and untouched requests continue to omit sampler overrides.
 */
export function readPersistedSessionGenerationDefaults(
  config: string | Record<string, unknown> | undefined,
): GenerationDefaultsLike {
  const parsed = configRecord(config)
  const declared = parsed.defaultSamplingDefaultsDeclared === true
  const doSample = typeof parsed.defaultDoSample === 'boolean'
    ? parsed.defaultDoSample
    : undefined
  const defaults: GenerationDefaultsLike = {}

  if (declared) {
    if (doSample === false) {
      // `do_sample: false` is itself an explicit bundle policy, so these are
      // truthful effective values even when individual generation fields were
      // absent.
      defaults.temperature = 0
      defaults.topP = 1
      defaults.topK = 0
    } else {
      // Historical session rows persist absent fields as numeric zeroes and
      // only have one aggregate "declared" bit. A zero therefore cannot prove
      // that an individual bundle field existed. Preserve only non-zero
      // sentinel values here; the live generation-config read restores
      // explicit zeroes with field-level provenance before controls become
      // interactive.
      const temperature = positiveNumber(parsed.defaultTemperature)
      if (temperature != null) defaults.temperature = temperature / 100
      const topP = positiveNumber(parsed.defaultTopP)
      if (topP != null) defaults.topP = topP / 100
      const topK = positiveNumber(parsed.defaultTopK)
      if (topK != null) defaults.topK = Math.round(topK)
    }
    const minP = positiveNumber(parsed.defaultMinP)
    if (minP != null) defaults.minP = minP / 100
    const repeatPenalty = positiveNumber(parsed.defaultRepetitionPenalty)
    if (repeatPenalty != null) defaults.repeatPenalty = repeatPenalty / 100
  }

  const maxNewTokens = positiveNumber(parsed.defaultMaxNewTokens)
  if (maxNewTokens != null) defaults.maxTokens = Math.floor(maxNewTokens)
  return defaults
}

function modelDefaultsFromGeneration(
  generation: ChatSettingsGenerationDefaults | null | undefined,
): GenerationDefaultsLike {
  if (!generation) return {}
  const defaults: GenerationDefaultsLike = {}
  if (generation.doSample === false) {
    defaults.temperature = 0
    defaults.topP = 1
    defaults.topK = 0
  } else {
    if (generation.temperature != null) defaults.temperature = generation.temperature
    if (generation.topP != null) defaults.topP = generation.topP
    if (generation.topK != null) defaults.topK = generation.topK
  }
  if (generation.minP != null) defaults.minP = generation.minP
  if (generation.repeatPenalty != null) defaults.repeatPenalty = generation.repeatPenalty
  if (generation.maxNewTokens != null) defaults.maxTokens = generation.maxNewTokens
  if (generation.maxThinkingTokens != null) defaults.maxThinkingTokens = generation.maxThinkingTokens
  return defaults
}

function fulfilledValue<T>(result: PromiseSettledResult<T>): T | undefined {
  return result.status === 'fulfilled' ? result.value : undefined
}

function explicitOverrides<TOverrides extends object>(
  value: TOverrides | null | undefined,
): Partial<TOverrides> {
  return Object.fromEntries(
    Object.entries(value || {}).filter(([, entry]) => entry !== null && entry !== undefined),
  ) as Partial<TOverrides>
}

/**
 * Pure reduction from independently settled IPC results to one coherent UI
 * snapshot. Promise completion order cannot change which source owns a field.
 */
export function resolveChatSettingsHydration<TOverrides extends object>(
  sessionConfig: string | Record<string, unknown> | undefined,
  settled: ChatSettingsHydrationSettlements<TOverrides>,
): ChatSettingsHydrationResult<TOverrides> {
  const persistedDefaults = readPersistedSessionGenerationDefaults(sessionConfig)
  const generation = fulfilledValue(settled.generationDefaults)
  const detected = fulfilledValue(settled.detectedConfig)
  const liveDefaults = modelDefaultsFromGeneration(generation)
  const liveSamplerDefaults = hasSamplerDefaults(liveDefaults)

  let defaultsState: ChatSettingsDefaultsState
  let modelDefaults: GenerationDefaultsLike
  if (liveSamplerDefaults) {
    defaultsState = 'ready'
    modelDefaults = liveDefaults
  } else if (
    settled.generationDefaults.status === 'rejected' &&
    hasAnyDefaults(persistedDefaults)
  ) {
    defaultsState = 'session-fallback'
    modelDefaults = { ...persistedDefaults, ...liveDefaults }
  } else if (settled.generationDefaults.status === 'fulfilled') {
    defaultsState = 'engine-fallback'
    modelDefaults = liveDefaults
  } else {
    defaultsState = 'unavailable'
    modelDefaults = {}
  }

  modelDefaults = applyEffectiveSessionGenerationDefaults(
    modelDefaults,
    sessionConfig,
    detected?.nativeMtp,
  )

  const partialFailure = Object.values(settled).some(result => result.status === 'rejected')

  return {
    overrides: explicitOverrides(fulfilledValue(settled.overrides)),
    overridesLoaded: settled.overrides.status === 'fulfilled',
    modelDefaults,
    defaultsState,
    partialFailure,
    generationDefaults: generation || undefined,
    detectedConfig: detected || undefined,
  }
}

function start<T>(loader: () => Promise<T>): Promise<T> {
  try {
    return Promise.resolve(loader())
  } catch (error) {
    return Promise.reject(error)
  }
}

function startBounded<T>(
  name: string,
  loader: () => Promise<T>,
  timeoutMs: number,
): Promise<T> {
  const pending = start(loader)
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return pending

  let timer: ReturnType<typeof setTimeout> | undefined
  const timedOut = new Promise<never>((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`${name} did not settle within ${timeoutMs}ms`))
    }, timeoutMs)
  })
  return Promise.race([pending, timedOut]).finally(() => {
    if (timer !== undefined) clearTimeout(timer)
  })
}

/**
 * Start every settings-owned IPC read before awaiting any one of them. Each
 * source has its own real rejection deadline, so a hung detector cannot leave
 * the renderer permanently waiting or fabricate a replacement value.
 */
export async function loadChatSettingsHydration<TOverrides extends object>(
  sessionConfig: string | Record<string, unknown> | undefined,
  loaders: ChatSettingsHydrationLoaders<TOverrides>,
  timeoutMs = CHAT_SETTINGS_IPC_TIMEOUT_MS,
): Promise<ChatSettingsHydrationResult<TOverrides>> {
  const [overrides, generationDefaults, detectedConfig] =
    await Promise.allSettled([
      startBounded('chat overrides', loaders.overrides, timeoutMs),
      startBounded('generation defaults', loaders.generationDefaults, timeoutMs),
      startBounded('model detection', loaders.detectedConfig, timeoutMs),
    ])

  return resolveChatSettingsHydration(sessionConfig, {
    overrides,
    generationDefaults,
    detectedConfig,
  })
}

/**
 * Chat identity/history only feeds compatibility warnings. Load it on an
 * independent bounded path so it can never hold sampler/reasoning controls
 * behind an unrelated database read.
 */
export async function loadChatSettingsCompatibility(
  loaders: ChatSettingsCompatibilityLoaders,
  timeoutMs = CHAT_SETTINGS_IPC_TIMEOUT_MS,
): Promise<ChatSettingsCompatibilityResult> {
  const [chat, messages] = await Promise.allSettled([
    startBounded('chat metadata', loaders.chat, timeoutMs),
    startBounded('chat messages', loaders.messages, timeoutMs),
  ])
  const chatValue = fulfilledValue(chat)
  const messagesValue = fulfilledValue(messages)
  return {
    partialFailure: chat.status === 'rejected' || messages.status === 'rejected',
    savedChatModelPath: chatValue?.modelPath,
    messageCount: Array.isArray(messagesValue) ? messagesValue.length : 0,
  }
}
