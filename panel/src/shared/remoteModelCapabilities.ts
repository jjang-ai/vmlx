import { REASONING_EFFORT_LEVELS, remoteReasoningFormatForUrl, type RemoteReasoningFormat } from './reasoningEffortPolicy'
import { normalizeDetectedFamilyName } from './detectedFamilyNames'
import { remoteServerBaseUrl } from './remoteApiUrl'
export interface RemoteModelConnection {
  remoteUrl?: string
  remoteApiKey?: string
  remoteModel?: string
  remoteOrganization?: string
}

export interface RemoteGenerationDefaults {
  doSample?: boolean
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
  maxNewTokens?: number
  maxThinkingTokens?: number
  thinkingBudgetSupported?: boolean
  supportsThinkingBudget?: boolean
}

export interface RemoteDetectedConfig {
  family?: string
  toolParser?: string
  reasoningParser?: string
  supportsThinking?: boolean
  supportsTools?: boolean
  remoteReasoningFormat?: RemoteReasoningFormat
  supportsAdaptiveThinking?: boolean
  supportsInstructMode?: boolean
  supportedReasoningEfforts?: Array<'low' | 'medium' | 'high' | 'xhigh' | 'max'>
  defaultReasoningEffort?: 'low' | 'medium' | 'high' | 'xhigh' | 'max'
  supportsThinkingBudget?: boolean
  thinkInTemplate?: boolean
  dsv4PoolQuantDefault?: boolean
  cacheType?: string
  cacheSubtype?: string
  architectureHints?: Record<string, string | number | boolean>
  usePagedCache?: boolean
  enableAutoToolChoice?: boolean
  isMultimodal?: boolean
  forceTextOnly?: boolean
  quantizationLabel?: string
  nativeMtp?: {
    supported?: boolean
    depth?: number
    depthSource?: string
    runtimeScope?: 'text' | 'text+vl'
    nativeCacheType?: string
    requiresDeterministicSampling?: boolean
    defaultMode?: 'auto' | 'off'
    blockedReason?: string
  }
  description?: string
  maxContextLength?: number
}

type JsonRecord = Record<string, unknown>

const REASONING_EFFORTS = new Set<string>(REASONING_EFFORT_LEVELS)
const MEDIA_MODALITIES = new Set(['audio', 'image', 'omni', 'video', 'vision'])

function record(value: unknown): JsonRecord | undefined {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord
    : undefined
}

function finiteNumber(...values: unknown[]): number | undefined {
  return values.find(
    (value): value is number => typeof value === 'number' && Number.isFinite(value),
  )
}

function explicitBoolean(...values: unknown[]): boolean | undefined {
  return values.find((value): value is boolean => typeof value === 'boolean')
}

function positiveInteger(...values: unknown[]): number | undefined {
  const value = finiteNumber(...values)
  return value != null && value > 0 ? Math.floor(value) : undefined
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

/**
 * A fourth private copy of this rule used to live here, and it is exactly the
 * duplication detectedFamilyNames.ts was created to end. It also silently
 * lagged behind: it never learned the qwen3.5 / qwen3-next / nemotron-h
 * spellings, so remote sessions on those families missed their slow-family
 * timeout entirely.
 */
function normalizedFamily(value: unknown): string | undefined {
  return normalizeDetectedFamilyName(nonEmptyString(value))
}

function parserName(value: unknown): string | undefined {
  const parser = nonEmptyString(value)
  return parser && parser !== 'none' ? parser : undefined
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map(nonEmptyString).filter((entry): entry is string => Boolean(entry))
    : []
}

function reasoningEfforts(value: unknown): Array<'low' | 'medium' | 'high' | 'xhigh' | 'max'> {
  return stringArray(value)
    .map(level => level.toLowerCase())
    .filter((level): level is 'low' | 'medium' | 'high' | 'xhigh' | 'max' => REASONING_EFFORTS.has(level))
}

/**
 * Map the bundle-declared sampler defaults advertised by the live engine into
 * the shape consumed by Chat/Server Settings. Runtime fallbacks and server
 * overrides stay in effective_defaults and must not be presented as bundle
 * metadata; absent bundle fields therefore remain absent here.
 */
export function generationDefaultsFromRemoteCapabilities(
  value: unknown,
): RemoteGenerationDefaults | null {
  const capabilities = record(value)
  if (!capabilities) return null
  const sampling = record(capabilities.sampling_defaults) ?? {}
  const defaults: RemoteGenerationDefaults = {}

  if (typeof sampling.do_sample === 'boolean') defaults.doSample = sampling.do_sample
  const temperature = finiteNumber(sampling.temperature)
  if (temperature != null) defaults.temperature = temperature
  const topP = finiteNumber(sampling.top_p)
  if (topP != null) defaults.topP = topP
  const topK = finiteNumber(sampling.top_k)
  if (topK != null) defaults.topK = topK
  const minP = finiteNumber(sampling.min_p)
  if (minP != null) defaults.minP = minP
  const repeatPenalty = finiteNumber(
    sampling.repetition_penalty,
    sampling.repeat_penalty,
  )
  if (repeatPenalty != null) defaults.repeatPenalty = repeatPenalty
  const maxNewTokens = positiveInteger(
    sampling.max_new_tokens,
    sampling.max_tokens,
  )
  if (maxNewTokens != null) defaults.maxNewTokens = maxNewTokens
  const maxThinkingTokens = positiveInteger(
    sampling.max_thinking_tokens,
    sampling.thinking_budget,
  )
  // Runtime admission wins over a bundle's text-only default or legacy
  // template hint. Native media may share the endpoint but reject hard caps.
  const budgetSupported = explicitBoolean(
    capabilities.supports_thinking_budget,
    capabilities.thinking_budget_supported,
  )
  if (maxThinkingTokens != null && budgetSupported !== false) {
    defaults.maxThinkingTokens = maxThinkingTokens
  }
  if (typeof capabilities.supports_thinking_budget === 'boolean') {
    defaults.supportsThinkingBudget = capabilities.supports_thinking_budget
  }
  if (typeof capabilities.thinking_budget_supported === 'boolean') {
    defaults.thinkingBudgetSupported = budgetSupported
  }

  return Object.keys(defaults).length > 0 ? defaults : null
}

/** Map only explicitly advertised live capabilities; never infer from a name. */
export function detectedConfigFromRemoteCapabilities(
  value: unknown,
): RemoteDetectedConfig | null {
  const capabilities = record(value)
  if (!capabilities) return null
  const cache = record(capabilities.cache) ?? {}
  const nativeCache = record(cache.native) ?? {}
  const poolQuant = record(nativeCache.pool_quant) ?? {}
  const quantization = record(capabilities.quantization) ?? {}
  const media = record(capabilities.media) ?? {}
  const mtpPayload = record(capabilities.mtp)
  const mtp = mtpPayload ?? {}

  const mediaRuntimeModalities = stringArray(media.runtime_modalities)
  const runtimeModalities = mediaRuntimeModalities.length > 0
    ? mediaRuntimeModalities
    : stringArray(capabilities.modalities)
  const declaredModalities = stringArray(media.declared_modalities)
  const runtimeHasMedia = runtimeModalities.some(modality => MEDIA_MODALITIES.has(modality.toLowerCase()))
  const declaredHasMedia = declaredModalities.some(modality => MEDIA_MODALITIES.has(modality.toLowerCase()))
  const efforts = reasoningEfforts(capabilities.reasoning_efforts)
  const defaultEffort = nonEmptyString(capabilities.default_reasoning_effort)?.toLowerCase()
  const toolParser = capabilities.supports_tools === false
    ? undefined
    : parserName(capabilities.tool_parser)
  const reasoningParser = capabilities.supports_thinking === false
    ? undefined
    : parserName(capabilities.reasoning_parser)
  const family = normalizedFamily(capabilities.family)
  const detected: RemoteDetectedConfig = {}

  if (capabilities.reasoning_request_format === 'openrouter') {
    detected.remoteReasoningFormat = 'openrouter'
  } else if (typeof capabilities.supports_thinking === 'boolean' || reasoningParser) {
    detected.remoteReasoningFormat = 'vmlx'
  }
  if (typeof capabilities.supports_tools === 'boolean') detected.supportsTools = capabilities.supports_tools

  if (family) detected.family = family
  if (toolParser) detected.toolParser = toolParser
  if (reasoningParser) detected.reasoningParser = reasoningParser
  if (typeof capabilities.supports_thinking === 'boolean') {
    detected.supportsThinking = capabilities.supports_thinking
  }
  if (Array.isArray(capabilities.native_thinking_modes)) {
    detected.supportsAdaptiveThinking = capabilities.native_thinking_modes.includes('adaptive')
  }
  if (typeof capabilities.supports_instruct_mode === 'boolean') {
    detected.supportsInstructMode = capabilities.supports_instruct_mode
  }
  if (Array.isArray(capabilities.reasoning_efforts)) detected.supportedReasoningEfforts = efforts
  if (defaultEffort && REASONING_EFFORTS.has(defaultEffort)) {
    detected.defaultReasoningEffort = defaultEffort as 'low' | 'medium' | 'high' | 'xhigh' | 'max'
  }
  if (typeof capabilities.supports_thinking_budget === 'boolean') {
    detected.supportsThinkingBudget = capabilities.supports_thinking_budget
  }
  if (typeof capabilities.think_in_template === 'boolean') {
    detected.thinkInTemplate = capabilities.think_in_template
  }
  if (typeof cache.type === 'string' && cache.type.trim()) detected.cacheType = cache.type.trim()
  if (typeof cache.subtype === 'string' && cache.subtype.trim()) detected.cacheSubtype = cache.subtype.trim()
  if (typeof cache.paged === 'boolean') detected.usePagedCache = cache.paged
  if (typeof capabilities.supports_tools === 'boolean' || toolParser) {
    detected.enableAutoToolChoice = capabilities.supports_tools === true ||
      (capabilities.supports_tools !== false && Boolean(toolParser))
  }
  if (runtimeModalities.length > 0) detected.isMultimodal = runtimeHasMedia
  if (declaredModalities.length > 0 && runtimeModalities.length > 0) {
    detected.forceTextOnly = declaredHasMedia && !runtimeHasMedia
  }
  if (typeof poolQuant.requested === 'boolean') {
    detected.dsv4PoolQuantDefault = poolQuant.requested
  }
  const cacheSchema = nonEmptyString(nativeCache.schema)
  const nativeFamily = nonEmptyString(nativeCache.family)
  if (cacheSchema || nativeFamily) {
    detected.architectureHints = {
      ...(cacheSchema ? { cacheSchema } : {}),
      ...(nativeFamily ? { nativeCacheFamily: nativeFamily } : {}),
    }
  }
  const quantizationProfile = nonEmptyString(quantization.profile)
  const quantizationCodec = nonEmptyString(quantization.codec)
  if (quantizationProfile || quantizationCodec) {
    detected.quantizationLabel = quantizationProfile ?? quantizationCodec
  }
  if (mtpPayload) {
    const nativeMtp: NonNullable<RemoteDetectedConfig['nativeMtp']> = {}
    // Capability and activation are different axes. GLM deliberately reports
    // runtime_supported=true, runtime_available=false when its family default
    // is AR; treating the latter as capability false hides the explicit D1-D3
    // controls from remote/API-attached sessions. A validation block remains
    // an actual unavailable runtime and keeps its reason visible.
    const runtimeSupported = explicitBoolean(mtp.runtime_supported)
    const runtimeAvailable = explicitBoolean(mtp.runtime_available)
    const mtpSupported = mtp.runtime_validation_blocked === true
      ? false
      : runtimeSupported ?? runtimeAvailable
    if (mtpSupported != null) nativeMtp.supported = mtpSupported
    const mtpDepth = positiveInteger(mtp.effective_depth)
    if (mtpDepth != null) nativeMtp.depth = mtpDepth
    const mtpDepthSource = nonEmptyString(mtp.effective_depth_source)
    if (mtpDepthSource) nativeMtp.depthSource = mtpDepthSource
    const mtpRuntimeScope = nonEmptyString(mtp.runtime_scope)
    if (mtpRuntimeScope === 'text' || mtpRuntimeScope === 'text+vl') {
      nativeMtp.runtimeScope = mtpRuntimeScope
    }
    if (typeof mtp.requires_deterministic_sampling === 'boolean') {
      nativeMtp.requiresDeterministicSampling = mtp.requires_deterministic_sampling
    }
    const defaultMode = nonEmptyString(mtp.runtime_default_mode)
    if (defaultMode === 'auto' || defaultMode === 'off') {
      nativeMtp.defaultMode = defaultMode
    }
    if (mtpSupported === false) {
      const blockedReason = nonEmptyString(mtp.runtime_reason)
      if (blockedReason) nativeMtp.blockedReason = blockedReason
    }
    if (Object.keys(nativeMtp).length > 0) {
      const nativeCacheType = nonEmptyString(nativeCache.cache_type)
      if (nativeCacheType) nativeMtp.nativeCacheType = nativeCacheType
      detected.nativeMtp = nativeMtp
    }
  }
  const modelId = nonEmptyString(capabilities.id) ?? nonEmptyString(capabilities.loaded_model)
  if (modelId || family) {
    detected.description = `Live runtime capabilities: ${modelId ?? family}`
  }

  return Object.keys(detected).length > 0 ? detected : null
}

/** Provider-native metadata, selected by exact model id rather than its name. */
export function capabilitiesFromOpenRouterModel(value: unknown): JsonRecord | null {
  const model = record(value)
  const id = model && nonEmptyString(model.id)
  if (!model || !id) return null
  const parameters = Array.isArray(model.supported_parameters) ? stringArray(model.supported_parameters) : undefined
  const reasoning = record(model.reasoning)
  const result: JsonRecord = { id, reasoning_request_format: 'openrouter' }
  if (parameters) {
    result.supports_tools = parameters.includes('tools')
    result.supports_thinking = parameters.includes('reasoning') || parameters.includes('reasoning_effort')
  }
  if (reasoning) {
    result.supports_thinking = true
    if (typeof reasoning.mandatory === 'boolean') result.supports_instruct_mode = !reasoning.mandatory
    // Missing efforts is distinct from null (provider allows all its tiers).
    result.reasoning_efforts = reasoning.supported_efforts === null
      ? [...REASONING_EFFORT_LEVELS] : reasoningEfforts(reasoning.supported_efforts)
    if (typeof reasoning.default_effort === 'string') result.default_reasoning_effort = reasoning.default_effort
  }
  const modalities = stringArray(record(model.architecture)?.input_modalities)
  if (modalities.length) result.modalities = modalities
  return result
}

/**
 * Fetch model-specific capabilities first for multi-model gateways, then the
 * single-model compatibility route. Credentials stay in Electron main and are
 * never returned to the renderer.
 */
export async function fetchRemoteModelCapabilities(
  connection: RemoteModelConnection,
  fetchImpl: typeof fetch = fetch,
  timeoutMs = 4_500,
): Promise<JsonRecord | null> {
  const baseUrl = connection.remoteUrl && remoteServerBaseUrl(connection.remoteUrl)
  if (!baseUrl) return null
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (connection.remoteApiKey) headers.Authorization = `Bearer ${connection.remoteApiKey}`
  if (connection.remoteOrganization) headers['OpenAI-Organization'] = connection.remoteOrganization
  const model = nonEmptyString(connection.remoteModel)
  if (remoteReasoningFormatForUrl(connection.remoteUrl) === 'openrouter') {
    if (!model) return null
    try {
      const response = await fetchImpl(`${baseUrl}/v1/models`, {
        headers, signal: AbortSignal.timeout(Math.max(1, Math.floor(timeoutMs))),
      })
      if (!response.ok) return null
      const payload = record(await response.json())
      const models = Array.isArray(payload?.data) ? payload.data : []
      return capabilitiesFromOpenRouterModel(models.find(entry => record(entry)?.id === model))
    } catch { return null }
  }
  const attempts = [
    ...(model
      ? [{ path: `/v1/models/${encodeURIComponent(model)}/capabilities`, requireIdentity: false }]
      : []),
    { path: '/v1/capabilities', requireIdentity: Boolean(model) },
  ]
  const totalBudgetMs = Math.max(1, Math.floor(timeoutMs))
  const deadline = Date.now() + totalBudgetMs

  for (let index = 0; index < attempts.length; index += 1) {
    const attempt = attempts[index]
    const remainingMs = deadline - Date.now()
    if (remainingMs <= 0) break
    // Reserve time for the compatibility route instead of allowing a missing
    // model-specific route to consume the renderer's entire hydration budget.
    const attemptBudgetMs = model && index === 0
      ? Math.max(1, Math.floor(remainingMs / 2))
      : remainingMs
    try {
      const response = await fetchImpl(`${baseUrl}${attempt.path}`, {
        headers,
        signal: AbortSignal.timeout(attemptBudgetMs),
      })
      if (!response.ok) continue
      const payload = record(await response.json())
      if (!payload) continue
      if (attempt.requireIdentity && model) {
        const responseId = nonEmptyString(payload.id)
        const loadedModel = nonEmptyString(payload.loaded_model)
        if (responseId !== model && loadedModel !== model) continue
      }
      return payload
    } catch {
      // Try the compatibility route; callers preserve their existing fallback.
    }
  }
  return null
}
