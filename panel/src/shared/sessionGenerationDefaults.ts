export interface BundleGenerationDefaults {
  doSample?: boolean
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
  maxNewTokens?: number
  /** Effective running-server cap, not a declaration in the model bundle. */
  maxNewTokensFromEngine?: boolean
  source?: 'generation_config' | 'jang_config'
}

export interface SessionGenerationDefaultFields {
  defaultTemperature: number
  defaultTopP: number
  defaultTopK: number
  defaultMinP: number
  defaultRepetitionPenalty: number
  defaultMaxNewTokens: number
  defaultDoSample?: boolean
  defaultSamplingDefaultsDeclared: boolean
}

export interface DetectedDsv4PoolQuantDefault {
  family?: string
  dsv4PoolQuantDefault?: boolean
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function positiveInteger(value: unknown): number | undefined {
  const number = finiteNumber(value)
  return number != null && number > 0 ? Math.floor(number) : undefined
}

function objectRecord(value: unknown): Record<string, any> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, any>
    : undefined
}

/**
 * Resolve the bundle-owned sampling contract once for every main/renderer
 * settings surface. JANG chat metadata wins field-by-field over the standard
 * generation config; absent JANG fields retain valid generation defaults.
 */
export function resolveBundleGenerationDefaults(
  generationConfig: unknown,
  jangConfig: unknown,
  modelConfig: unknown,
): BundleGenerationDefaults | null {
  const defaults: BundleGenerationDefaults = {}
  const generation = objectRecord(generationConfig)
  if (generation) {
    const samplingDisabled = generation.do_sample === false
    if (typeof generation.do_sample === 'boolean') defaults.doSample = generation.do_sample
    const temperature = finiteNumber(generation.temperature)
    if (temperature != null) defaults.temperature = samplingDisabled ? 0 : temperature
    const topP = finiteNumber(generation.top_p)
    if (topP != null) defaults.topP = samplingDisabled ? 1 : topP
    const topK = finiteNumber(generation.top_k)
    if (topK != null) defaults.topK = samplingDisabled ? 0 : Math.max(0, Math.round(topK))
    const minP = finiteNumber(generation.min_p)
    if (minP != null) defaults.minP = minP
    const repeatPenalty = finiteNumber(generation.repetition_penalty)
    if (repeatPenalty != null) defaults.repeatPenalty = repeatPenalty
    const maxNewTokens = positiveInteger(generation.max_new_tokens)
    if (maxNewTokens != null) defaults.maxNewTokens = maxNewTokens
    if (Object.keys(defaults).length > 0) defaults.source = 'generation_config'
  }

  const jang = objectRecord(jangConfig)
  const sampling = objectRecord(jang?.chat?.sampling_defaults)
  if (sampling) {
    // JANG chat metadata owns sampling mode. A generation_config do_sample
    // value must not silently override a stamped JANG request contract.
    delete defaults.doSample
    const temperature = finiteNumber(sampling.temperature)
    if (temperature != null) defaults.temperature = temperature
    const topP = finiteNumber(sampling.top_p)
    if (topP != null) defaults.topP = topP
    const topK = finiteNumber(sampling.top_k)
    if (topK != null) defaults.topK = Math.max(0, Math.round(topK))
    const minP = finiteNumber(sampling.min_p)
    if (minP != null) defaults.minP = minP

    const defaultMode = jang?.chat?.reasoning?.default_mode
    const repThinking = finiteNumber(sampling.repetition_penalty_thinking)
    const repChat = finiteNumber(sampling.repetition_penalty_chat)
    const repScalar = finiteNumber(sampling.repetition_penalty)
    const modelType = objectRecord(modelConfig)?.model_type
    const repeatPenalty = modelType === 'deepseek_v4'
      ? (defaultMode === 'thinking'
        ? (repThinking ?? repScalar ?? repChat)
        : (repScalar ?? repChat ?? repThinking))
      : defaultMode === 'thinking'
        ? (repThinking ?? repChat ?? repScalar)
        : (repChat ?? repThinking ?? repScalar)
    if (repeatPenalty != null) defaults.repeatPenalty = repeatPenalty

    const maxNewTokens = positiveInteger(sampling.max_new_tokens)
    if (maxNewTokens != null) defaults.maxNewTokens = maxNewTokens
    defaults.source = 'jang_config'
  }

  return Object.keys(defaults).some((key) => key !== 'source') ? defaults : null
}

export function hasDeclaredBundleSamplingDefaults(
  defaults: BundleGenerationDefaults | null | undefined,
): boolean {
  return !!defaults && (
    defaults.doSample === false ||
    defaults.temperature != null ||
    defaults.topP != null ||
    defaults.topK != null ||
    defaults.minP != null ||
    defaults.repeatPenalty != null
  )
}

/**
 * Copy bundle-derived generation defaults into the fields displayed by the
 * session settings surfaces. These are inherited request defaults, not CLI
 * overrides, so absent values intentionally reset to the neutral sentinel.
 */
export function applyBundleGenerationDefaultsToSessionConfig<T extends object>(
  config: T,
  defaults: BundleGenerationDefaults | null | undefined,
): T & SessionGenerationDefaultFields {
  return {
    ...config,
    defaultTemperature: defaults?.temperature != null
      ? Math.round(defaults.temperature * 100)
      : 0,
    defaultTopP: defaults?.topP != null ? Math.round(defaults.topP * 100) : 0,
    defaultTopK: defaults?.topK != null ? Math.max(0, Math.round(defaults.topK)) : 0,
    defaultMinP: defaults?.minP != null ? Math.max(0, Math.round(defaults.minP * 100)) : 0,
    defaultRepetitionPenalty: defaults?.repeatPenalty != null
      ? Math.round(defaults.repeatPenalty * 100)
      : 0,
    // Health can include an explicit CLI override or a transient headroom cap.
    // Neither is a bundle default that should be advertised by Reset/Auto.
    // Keep config.maxTokens untouched; it owns the explicit server limit.
    defaultMaxNewTokens: defaults?.maxNewTokens != null && !defaults.maxNewTokensFromEngine
      ? Math.max(0, Math.round(defaults.maxNewTokens))
      : 0,
    defaultDoSample: typeof defaults?.doSample === 'boolean' ? defaults.doSample : undefined,
    defaultSamplingDefaultsDeclared: hasDeclaredBundleSamplingDefaults(defaults),
  }
}

/**
 * Reconcile the one DSV4 session field whose value is owned by the current
 * model bundle rather than by a previously saved session. Unknown detection
 * leaves the saved value alone; a known non-DSV4 family clears the stale,
 * family-specific field without touching any unrelated user settings.
 */
export function applyBundleDsv4PoolQuantToSessionConfig<
  T extends { dsv4PoolQuant?: boolean },
>(
  config: T,
  detected: DetectedDsv4PoolQuantDefault | null | undefined,
): T {
  if (!detected?.family || detected.family === 'unknown') return config

  const next = { ...config } as T
  if (
    detected.family === 'deepseek-v4'
    && typeof detected.dsv4PoolQuantDefault === 'boolean'
  ) {
    next.dsv4PoolQuant = detected.dsv4PoolQuantDefault
  } else {
    delete next.dsv4PoolQuant
  }
  return next
}
