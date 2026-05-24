import { useState, useEffect } from 'react'
import { ArrowLeft, ChevronRight } from 'lucide-react'
import { SessionConfigForm, SessionConfig, DEFAULT_CONFIG } from './SessionConfigForm'
import { useTranslation } from '../../i18n'
import { resolveCacheLaunchPolicy } from '../../../../shared/cacheControlPolicy'
import { buildMcpPolicyArgs } from '../../../../shared/mcpPolicy'
import { canonicalizeReasoningParserForCli } from '../../../../shared/reasoningParserAliases'
import { canonicalizeToolParserId } from '../../../../shared/toolParserAliases'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  config: string
  createdAt: number
  updatedAt: number
}

interface SessionSettingsProps {
  sessionId: string
  onBack: () => void
}

function normalizeDetectedFamilyName(family?: string): string | undefined {
  if (!family) return undefined
  if (family === 'deepseek_v4') return 'deepseek-v4'
  if (family === 'zaya1_vl') return 'zaya1-vl'
  if (family === 'bailing_hybrid') return 'ling'
  return family
}

function isZayaCcaFamily(family?: string): boolean {
  const normalized = normalizeDetectedFamilyName(family)
  return normalized === 'zaya' || normalized === 'zaya1-vl'
}

function cacheTypeRequiresPaged(cacheType?: string): boolean {
  return cacheType === 'hybrid' || cacheType === 'mamba'
}

const DSV4_PAGED_CACHE_BLOCK_SIZE = 256
const GENERIC_DEFAULT_TIMEOUT_SECONDS = 300
const DSV4_DEFAULT_TIMEOUT_SECONDS = 900

function effectiveSessionTimeoutSeconds(config: Partial<SessionConfig>, family?: string): number {
  const configured = config.timeout
  if (configured != null && configured <= 0) return 86400
  const normalizedFamily = normalizeDetectedFamilyName(family)
  if (normalizedFamily === 'deepseek-v4' && (configured == null || configured === GENERIC_DEFAULT_TIMEOUT_SECONDS)) {
    return DSV4_DEFAULT_TIMEOUT_SECONDS
  }
  return configured != null && configured > 0 ? configured : GENERIC_DEFAULT_TIMEOUT_SECONDS
}

function finitePositiveNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined
}

function finitePositiveInteger(value: unknown): number | undefined {
  const number = finitePositiveNumber(value)
  return number == null ? undefined : Math.max(1, Math.floor(number))
}

const ADDITIONAL_ARG_VALUE_FLAGS = new Set([
  '--block-disk-cache-dir',
  '--block-disk-cache-max-gb',
  '--allowed-origins',
  '--api-key',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--chat-template',
  '--chat-template-kwargs',
  '--cluster-secret',
  '--completion-batch-size',
  '--default-enable-thinking',
  '--default-min-p',
  '--default-repetition-penalty',
  '--default-temperature',
  '--default-top-k',
  '--default-top-p',
  '--distributed-mode',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--embedding-model',
  '--flash-moe-io-split',
  '--flash-moe-prefetch',
  '--flash-moe-slot-bank',
  '--host',
  '--image-mode',
  '--image-quantize',
  '--kv-cache-group-size',
  '--kv-cache-quantization',
  '--log-level',
  '--max-cache-blocks',
  '--max-num-seqs',
  '--max-prompt-tokens',
  '--max-tokens',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
  '--mflux-class',
  '--native-mtp-depth',
  '--native-mtp-sampling-policy',
  '--omni-backend',
  '--num-draft-tokens',
  '--paged-cache-block-size',
  '--pld-summary-interval',
  '--port',
  '--prefill-batch-size',
  '--prefill-step-size',
  '--prefix-cache-max-bytes',
  '--prefix-cache-size',
  '--rate-limit',
  '--reasoning-parser',
  '--served-model-name',
  '--smelt-experts',
  '--speculative-model',
  '--ssm-state-cache-mb',
  '--ssm-state-cache-size',
  '--stream-interval',
  '--timeout',
  '--tool-call-parser',
  '--worker-nodes',
])

const IMAGE_ADDITIONAL_ARG_BLOCKLIST = new Set([
  '--image-mode',
  '--image-quantize',
  '--served-model-name',
  '--mflux-class',
])

const DSV4_ADDITIONAL_ARG_BLOCKLIST = new Set([
  '--continuous-batching',
  '--no-continuous-batching',
  '--host',
  '--port',
  '--api-key',
  '--rate-limit',
  '--timeout',
  '--log-level',
  '--allowed-origins',
  '--served-model-name',
  '--dsv4-enable-prefix-cache',
  '--enable-prefix-cache',
  '--disable-prefix-cache',
  '--use-paged-cache',
  '--paged-cache-block-size',
  '--max-cache-blocks',
  '--kv-cache-quantization',
  '--kv-cache-group-size',
  '--max-num-seqs',
  '--prefill-batch-size',
  '--prefill-step-size',
  '--completion-batch-size',
  '--max-tokens',
  '--max-prompt-tokens',
  '--stream-interval',
  '--ssm-state-cache-size',
  '--ssm-state-cache-mb',
  '--enable-jit',
  '--no-memory-aware-cache',
  '--prefix-cache-size',
  '--prefix-cache-max-bytes',
  '--cache-memory-mb',
  '--cache-memory-percent',
  '--cache-ttl-minutes',
  '--enable-disk-cache',
  '--disk-cache-dir',
  '--disk-cache-max-gb',
  '--smelt',
  '--smelt-experts',
  '--flash-moe',
  '--flash-moe-slot-bank',
  '--flash-moe-prefetch',
  '--flash-moe-io-split',
  '--distributed',
  '--distributed-mode',
  '--worker-nodes',
  '--cluster-secret',
  '--speculative-model',
  '--num-draft-tokens',
  '--native-mtp-depth',
  '--native-mtp-sampling-policy',
  '--disable-native-mtp',
  '--enable-pld',
  '--pld-summary-interval',
  '--is-mllm',
  '--mcp-config',
  '--mcp-disabled-servers',
  '--mcp-disabled-tools',
  '--mcp-enabled-servers',
  '--mcp-enabled-tools',
  '--omni-backend',
  '--enable-auto-tool-choice',
  '--tool-call-parser',
  '--reasoning-parser',
  '--embedding-model',
  '--default-temperature',
  '--default-top-p',
  '--default-top-k',
  '--default-min-p',
  '--default-repetition-penalty',
  '--default-enable-thinking',
  '--chat-template',
  '--chat-template-kwargs',
])

function filterAdditionalArgs(raw: string | undefined, blockedFlags: Set<string>): string[] {
  if (!raw?.trim()) return []
  const extra = raw.trim().split(/\s+/).filter(Boolean)
  const filtered: string[] = []
  for (let i = 0; i < extra.length; i++) {
    const flag = extra[i]
    if (blockedFlags.has(flag)) {
      if (ADDITIONAL_ARG_VALUE_FLAGS.has(flag)) i++
      continue
    }
    filtered.push(flag)
  }
  return filtered
}

async function applyBundleGenerationDefaults(config: SessionConfig, modelPath: string): Promise<SessionConfig> {
  const next: SessionConfig = { ...config }
  try {
    const gen = await window.api.models.getGenerationDefaults(modelPath) as any
    next.defaultTemperature = gen?.temperature != null ? Math.round(gen.temperature * 100) : 0
    next.defaultTopP = gen?.topP != null ? Math.round(gen.topP * 100) : 0
    next.defaultTopK = gen?.topK != null ? Math.max(0, Math.round(gen.topK)) : 0
    next.defaultMinP = gen?.minP != null ? Math.round(gen.minP * 100) : 0
    next.defaultRepetitionPenalty = gen?.repeatPenalty != null ? Math.round(gen.repeatPenalty * 100) : 0
    next.defaultMaxNewTokens = gen?.maxNewTokens != null ? Math.round(gen.maxNewTokens) : 0
  } catch (_) {
    next.defaultTemperature = 0
    next.defaultTopP = 0
    next.defaultTopK = 0
    next.defaultMinP = 0
    next.defaultRepetitionPenalty = 0
    next.defaultMaxNewTokens = 0
  }
  return next
}

/**
 * Build a preview of the CLI command from config.
 * This MUST mirror the logic in sessions.ts buildArgs() exactly.
 * When editing buildArgs(), update this function too (and vice versa).
 */
function buildCommandPreview(
  modelPath: string,
  config: SessionConfig,
  detected?: { toolParser?: string; reasoningParser?: string; isMultimodal?: boolean; forceTextOnly?: boolean; isTurboQuant?: boolean; usePagedCache?: boolean; enableAutoToolChoice?: boolean; cacheType?: string; family?: string; nativeMtp?: { supported?: boolean; depth?: number; depthSource?: string } } | null
): string {
  const parts = ['vmlx-engine serve', modelPath]
  const requestedDistributed = !!(config as any).distributedEnabled
  const requestedFlashMoe = !!(config as any).flashMoe
  const detectedFamily = normalizeDetectedFamilyName(detected?.family)
  const dsv4Active = detectedFamily === 'deepseek-v4'
  const dsv4PrefixCacheOptIn = dsv4Active && config.dsv4PrefixCache !== false
  const omniBackendActive = detectedFamily === 'nemotron-h' && detected?.isMultimodal === true
  const effectiveSmelt = !!(config as any).smelt && !dsv4Active
  const isVLM = dsv4Active || effectiveSmelt || detected?.forceTextOnly ? false
    : detected?.isMultimodal ? true
      : config.isMultimodal === true ? true
        : config.isMultimodal === false ? false
          : false
  const zayaCcaActive = isZayaCcaFamily(detectedFamily)
  const turboQuantActive = !!detected?.isTurboQuant
  const hybridCacheActive = detected?.cacheType === 'hybrid' || detected?.cacheType === 'mamba'
  const effectiveDistributed = requestedDistributed && !dsv4Active
  const effectiveFlashMoe = requestedFlashMoe && !effectiveDistributed && !dsv4Active
  const effectiveEnableJit = !!config.enableJit && !isVLM && !effectiveFlashMoe && !effectiveDistributed && !dsv4Active && !zayaCcaActive && !turboQuantActive && !hybridCacheActive

  // Server settings
  parts.push('--host', config.host)
  parts.push('--port', config.port.toString())
  parts.push('--timeout', effectiveSessionTimeoutSeconds(config, detectedFamily).toString())

  if (config.apiKey) parts.push('# VLLM_API_KEY=*** (env var)')
  if (config.rateLimit && config.rateLimit > 0) parts.push('--rate-limit', config.rateLimit.toString())

  // Concurrent processing
  const effectiveMaxNumSeqs = dsv4Active ? 1 : config.maxNumSeqs
  if (effectiveMaxNumSeqs && effectiveMaxNumSeqs > 0) parts.push('--max-num-seqs', effectiveMaxNumSeqs.toString())
  if (!dsv4Active && config.prefillBatchSize && config.prefillBatchSize > 0) parts.push('--prefill-batch-size', config.prefillBatchSize.toString())
  if (!dsv4Active && config.prefillStepSize && config.prefillStepSize > 0) parts.push('--prefill-step-size', config.prefillStepSize.toString())
  if (!dsv4Active && config.completionBatchSize && config.completionBatchSize > 0) parts.push('--completion-batch-size', config.completionBatchSize.toString())

  if (isVLM) parts.push('--is-mllm')
  const cacheStackActive = dsv4Active ? true : config.continuousBatching !== false
  if (cacheStackActive) {
    parts.push('--continuous-batching')
  } else {
    parts.push('--no-continuous-batching')
  }

  // Parser resolution: User explicit choice -> Detected config -> Fallback
  // (mirrors buildArgs: user choice wins over detection)
  const effectiveToolParser = config.toolCallParser === ''
    ? undefined
    : canonicalizeToolParserId(config.toolCallParser && config.toolCallParser !== 'auto' ? config.toolCallParser
      : detected?.toolParser)
  const effectiveAutoTool = config.enableAutoToolChoice ?? detected?.enableAutoToolChoice
  const requestedReasoningParser = config.reasoningParser === ''
    ? undefined
    : (config.reasoningParser && config.reasoningParser !== 'auto' ? config.reasoningParser
      : detected?.reasoningParser)
  const effectiveReasoningParser = canonicalizeReasoningParserForCli(requestedReasoningParser)

  // Prefix cache (mirrors buildArgs): explicit user opt-out stays off even
  // when tools are configured. Tool sessions benefit from cache but do not
  // silently own the cache toggle.
  const zayaTypedCacheRequiresPaged = zayaCcaActive
  const architectureRequiresPagedCache =
    zayaTypedCacheRequiresPaged ||
    dsv4PrefixCacheOptIn ||
    (cacheTypeRequiresPaged(detected?.cacheType) && detected?.usePagedCache === true)
  const cacheLaunchPolicy = resolveCacheLaunchPolicy({
    continuousBatching: cacheStackActive,
    enablePrefixCache: dsv4Active
      ? dsv4PrefixCacheOptIn && config.enablePrefixCache !== false
      : config.enablePrefixCache !== false,
    usePagedCache: dsv4Active
      ? dsv4PrefixCacheOptIn
      : config.usePagedCache ?? detected?.usePagedCache ?? false,
    enableDiskCache: !!config.enableDiskCache,
    enableBlockDiskCache: dsv4Active
      ? dsv4PrefixCacheOptIn && !!config.enableBlockDiskCache
      : !!config.enableBlockDiskCache,
    architectureRequiresPagedCache,
  })
  const prefixCacheOff = cacheLaunchPolicy.prefixCacheOff
  const usePagedCache = cacheLaunchPolicy.effectiveUsePagedCache

  if (prefixCacheOff) {
    parts.push('--disable-prefix-cache')
  } else if (!dsv4Active) {
    if (config.noMemoryAwareCache) {
      parts.push('--no-memory-aware-cache')
      if (config.prefixCacheSize && config.prefixCacheSize > 0) parts.push('--prefix-cache-size', config.prefixCacheSize.toString())
      if (config.prefixCacheMaxBytes && config.prefixCacheMaxBytes > 0) parts.push('--prefix-cache-max-bytes', config.prefixCacheMaxBytes.toString())
    } else {
      if (!usePagedCache && config.cacheMemoryMb && config.cacheMemoryMb > 0) parts.push('--cache-memory-mb', config.cacheMemoryMb.toString())
      if (!usePagedCache && config.cacheMemoryPercent && config.cacheMemoryPercent > 0) parts.push('--cache-memory-percent', (config.cacheMemoryPercent / 100).toString())
      if (config.cacheTtlMinutes && config.cacheTtlMinutes > 0 && !usePagedCache) parts.push('--cache-ttl-minutes', config.cacheTtlMinutes.toString())
    }
  }

  // Paged cache — requires prefix cache ON (works for both LLM and VLM)
  if (!prefixCacheOff && usePagedCache) {
    parts.push('--use-paged-cache')
    const effectivePagedCacheBlockSize = dsv4Active
      ? DSV4_PAGED_CACHE_BLOCK_SIZE
      : config.pagedCacheBlockSize
    if (effectivePagedCacheBlockSize && effectivePagedCacheBlockSize > 0) parts.push('--paged-cache-block-size', effectivePagedCacheBlockSize.toString())
    if (config.maxCacheBlocks && config.maxCacheBlocks > 0) parts.push('--max-cache-blocks', config.maxCacheBlocks.toString())
  }

  // KV cache quantization — requires prefix cache ON (works for both LLM and VLM)
  // Hybrid/Mamba models allowed — Python scheduler only quantizes KVCache layers
  if (!prefixCacheOff && !dsv4Active && config.kvCacheQuantization && config.kvCacheQuantization !== 'auto') {
    parts.push('--kv-cache-quantization', config.kvCacheQuantization)
    if (config.kvCacheQuantization !== 'none' && config.kvCacheGroupSize && config.kvCacheGroupSize !== 64) {
      parts.push('--kv-cache-group-size', config.kvCacheGroupSize.toString())
    }
  }

  // Disk cache (L2 persistent cache) — mirrors sessions.ts buildArgs().
  if (cacheLaunchPolicy.enableLegacyDiskCache) {
    parts.push('--enable-disk-cache')
    if (config.diskCacheDir) parts.push('--disk-cache-dir', config.diskCacheDir)
    if (config.diskCacheMaxGb != null && config.diskCacheMaxGb >= 0) parts.push('--disk-cache-max-gb', config.diskCacheMaxGb.toString())
  }

  // Block-level disk cache (L2 for paged cache blocks) — mirrors sessions.ts buildArgs().
  if (cacheLaunchPolicy.enableBlockDiskCache) {
    parts.push('--enable-block-disk-cache')
    if (config.blockDiskCacheDir) parts.push('--block-disk-cache-dir', config.blockDiskCacheDir)
    if (config.blockDiskCacheMaxGb != null && config.blockDiskCacheMaxGb >= 0) parts.push('--block-disk-cache-max-gb', config.blockDiskCacheMaxGb.toString())
  }

  // Performance
  const streamInterval = finitePositiveInteger(config.streamInterval)
  if (streamInterval != null) parts.push('--stream-interval', streamInterval.toString())
  const maxTokens = finitePositiveInteger(config.maxTokens)
  if (maxTokens != null) {
    parts.push('--max-tokens', maxTokens.toString())
  }
  const maxContextLength = finitePositiveInteger(config.maxContextLength)
  if (maxContextLength != null) {
    parts.push('--max-prompt-tokens', maxContextLength.toString())
  }
  // Tool integration — mirrors buildArgs lines 1136-1147
  if (effectiveToolParser) {
    parts.push('--tool-call-parser', effectiveToolParser)
    if (effectiveAutoTool || config.enableAutoToolChoice === undefined) {
      parts.push('--enable-auto-tool-choice')
    }
  } else if (effectiveAutoTool) {
    parts.push('--enable-auto-tool-choice')
  }
  if (effectiveReasoningParser) parts.push('--reasoning-parser', effectiveReasoningParser)
  if (dsv4PrefixCacheOptIn) parts.push('--dsv4-enable-prefix-cache')

  if (config.mcpConfig) parts.push('--mcp-config', config.mcpConfig)
  parts.push(...buildMcpPolicyArgs(config))

  // Smelt mode (partial expert loading)
  if (effectiveSmelt) {
    parts.push('--smelt')
    const pct = (config as any).smeltExperts ?? 50
    if (pct !== 50) {
      parts.push('--smelt-experts', pct.toString())
    }
  }

  // Flash MoE (SSD expert streaming)
  if (effectiveFlashMoe) {
    parts.push('--flash-moe')
    const slotBank = (config as any).flashMoeSlotBank
    if (typeof slotBank === 'number' && slotBank > 0) {
      parts.push('--flash-moe-slot-bank', slotBank.toString())
    }
    const prefetch = (config as any).flashMoePrefetch
    if (prefetch && prefetch !== 'none') {
      parts.push('--flash-moe-prefetch', prefetch)
    }
    const ioSplit = (config as any).flashMoeIoSplit
    if (typeof ioSplit === 'number' && ioSplit > 0) {
      parts.push('--flash-moe-io-split', ioSplit.toString())
    }
  }

  // Distributed compute
  if (effectiveDistributed) {
    parts.push('--distributed')
    const mode = (config as any).distributedMode || 'pipeline'
    if (mode !== 'pipeline') {
      parts.push('--distributed-mode', mode)
    }
  }

  // Served model name
  if (config.servedModelName) parts.push('--served-model-name', config.servedModelName)

  // Custom chat template
  if ((config as any).chatTemplate) parts.push('--chat-template', '"..."')

  // Speculative decoding
  const compatibleExternalSpeculative = !dsv4Active && !isVLM && !cacheStackActive && !!config.speculativeModel
  if (compatibleExternalSpeculative) {
    parts.push('--speculative-model', config.speculativeModel)
    if (config.numDraftTokens && config.numDraftTokens !== 3) {
      parts.push('--num-draft-tokens', config.numDraftTokens.toString())
    }
  }

  // Native in-model MTP mirrors sessions.ts: deterministic mode owns the
  // startup sampling override needed for Qwen3.6 MTP to enter the native path.
  const nativeMtp = detected?.nativeMtp
  if (!dsv4Active && nativeMtp?.supported) {
    const mode = (config as any).nativeMtpMode || 'deterministic'
    if (mode === 'off') {
      parts.push('--disable-native-mtp')
    } else {
      const configuredDepth = (config as any).nativeMtpDepthOverride === true
        ? (config as any).nativeMtpDepth
        : nativeMtp.depth
      const depth = Math.max(1, Math.min(3, Math.round(Number(configuredDepth || nativeMtp.depth || 3))))
      parts.push('--native-mtp-depth', depth.toString())
      parts.push('--native-mtp-sampling-policy', mode === 'deterministic' ? 'deterministic-defaults' : 'compatible-only')
    }
  }

  // Generation defaults are resolved inside vmlx_engine.server from
  // jang_config/generation_config. The panel preview must not synthesize
  // --default-* flags that would override the engine's bundle lookup.

  // Embedding model
  if (config.embeddingModel) parts.push('--embedding-model', config.embeddingModel)

  // Thinking defaults are resolved by the engine and explicit chat/API requests.

  if (effectiveEnableJit) parts.push('--enable-jit')

  if (omniBackendActive && (config as any).omniBackend && (config as any).omniBackend !== 'stage1') {
    parts.push('--omni-backend', (config as any).omniBackend)
  }

  if (config.logLevel && config.logLevel !== 'INFO') {
    parts.push('--log-level', config.logLevel)
  }
  if (config.corsOrigins && config.corsOrigins !== '*') {
    parts.push('--allowed-origins', config.corsOrigins)
  }

  if (config.additionalArgs && config.additionalArgs.trim()) {
    const filtered = filterAdditionalArgs(
      config.additionalArgs,
      dsv4Active ? DSV4_ADDITIONAL_ARG_BLOCKLIST : IMAGE_ADDITIONAL_ARG_BLOCKLIST,
    )
    if (filtered.length) parts.push(filtered.join(' '))
  }

  return parts.join(' \\\n  ')
}

export function SessionSettings({ sessionId, onBack }: SessionSettingsProps) {
  const { t } = useTranslation()
  const [session, setSession] = useState<Session | null>(null)
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_CONFIG)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [detectedConfig, setDetectedConfig] = useState<{ toolParser?: string; reasoningParser?: string; cacheType?: string; isMultimodal?: boolean; forceTextOnly?: boolean; isTurboQuant?: boolean; usePagedCache?: boolean; enableAutoToolChoice?: boolean; family?: string; maxContextLength?: number; nativeMtp?: { supported?: boolean; depth?: number; depthSource?: string } } | null>(null)

  useEffect(() => {
    const load = async () => {
      const s = await window.api.sessions.get(sessionId)
      if (s) {
        setSession(s)
        // Parse stored config JSON, merge with defaults
        try {
          const stored = JSON.parse(s.config)
          const hydrated = await applyBundleGenerationDefaults({ ...DEFAULT_CONFIG, ...stored }, s.modelPath)
          setConfig(hydrated)
        } catch {
          setConfig(DEFAULT_CONFIG)
          setMessage({ type: 'error', text: 'Stored configuration was corrupted and has been reset to defaults. Save to persist.' })
          setDirty(true)
        }
        // Load auto-detected config for preview resolution
        try {
          const det = await window.api.models.detectConfig(s.modelPath)
          if (det && det.family !== 'unknown') setDetectedConfig(det)
        } catch (_) { }
      }
    }
    load()
  }, [sessionId])

  // Listen for session status changes
  useEffect(() => {
    const unsubStopped = window.api.sessions.onStopped((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'stopped', pid: undefined } : prev)
        setRestarting(false)
      }
    })
    const unsubReady = window.api.sessions.onReady((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'running' } : prev)
        setRestarting(false)
        setMessage({ type: 'success', text: 'Session restarted with new settings.' })
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'error' } : prev)
        setRestarting(false)
        setMessage({ type: 'error', text: `Restart failed: ${data.error}` })
      }
    })
    return () => {
      unsubStopped()
      unsubReady()
      unsubError()
    }
  }, [sessionId])

  const handleChange = <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    setDirty(true)
    setMessage(null)
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const result = await window.api.sessions.update(sessionId, config)
      if (result.success) {
        setDirty(false)
        setMessage({
          type: 'success',
          text: result.restartRequired
            ? `Settings saved. Restart the session for changes to take effect (${result.changedKeys?.join(', ')}).`
            : 'Settings saved.'
        })
        // Refresh session data
        const s = await window.api.sessions.get(sessionId)
        if (s) setSession(s)
      } else {
        setMessage({ type: 'error', text: result.error || 'Failed to save' })
      }
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }

  const handleSaveAndRestart = async () => {
    setSaving(true)
    setMessage(null)
    try {
      // Save first
      const saveResult = await window.api.sessions.update(sessionId, config)
      if (!saveResult.success) {
        setMessage({ type: 'error', text: saveResult.error || 'Failed to save' })
        setSaving(false)
        return
      }
      setDirty(false)

      // Stop and wait for the process to actually exit
      setRestarting(true)
      setMessage({ type: 'success', text: 'Stopping session...' })
      const stopResult = await window.api.sessions.stop(sessionId)
      if (!stopResult.success) {
        setMessage({ type: 'error', text: `Failed to stop: ${stopResult.error}` })
        setRestarting(false)
        setSaving(false)
        return
      }

      // Wait for port to free (backend uses up to 10s SIGKILL timeout)
      await new Promise(r => setTimeout(r, 2500))

      // Start with new config
      setMessage({ type: 'success', text: 'Starting session with new settings...' })
      const startResult = await window.api.sessions.start(sessionId)
      if (!startResult.success) {
        setMessage({ type: 'error', text: `Failed to start: ${startResult.error}` })
        setRestarting(false)
      }
      // Success/failure will be handled by event listeners above

      // Refresh session data
      const s = await window.api.sessions.get(sessionId)
      if (s) setSession(s)
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
      setRestarting(false)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    const base = { ...DEFAULT_CONFIG, host: config.host, port: config.port }
    // Re-run model detection to get proper defaults for this model
    if (session?.modelPath) {
      try {
        const detected = await window.api.models.detectConfig(session.modelPath)
        if (detected && detected.family !== 'unknown') {
          base.enableAutoToolChoice = detected.enableAutoToolChoice
          if (detected.family === 'deepseek-v4') {
            base.dsv4PrefixCache = true
            base.dsv4PoolQuant = true
            base.enablePrefixCache = true
            base.usePagedCache = true
            base.enableBlockDiskCache = true
            base.pagedCacheBlockSize = DSV4_PAGED_CACHE_BLOCK_SIZE
          } else {
            base.usePagedCache = detected.usePagedCache
          }
          // VLM models: set isMultimodal flag unless this model has a
          // runtime forceTextOnly policy (affine-JANG Qwen hybrid).
          if (detected.isMultimodal && !detected.forceTextOnly) {
            base.isMultimodal = true
          }
        }
        Object.assign(base, await applyBundleGenerationDefaults(base, session.modelPath))
      } catch (_) { }
    }
    setConfig(base)
    setDirty(true)
    setMessage(null)
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">{t('sessions.settings.loading')}</p>
      </div>
    )
  }

  const shortName = session.modelName || session.modelPath.split('/').pop() || session.modelPath
  const isRunning = session.status === 'running' || session.status === 'loading'

  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button onClick={onBack} className="text-muted-foreground hover:text-foreground flex items-center gap-1">
            <ArrowLeft className="h-3.5 w-3.5" /> {t('common.back')}
          </button>
          <h1 className="text-2xl font-bold">{t('sessions.settings.title')}</h1>
        </div>

        {/* Session Info */}
        <div className="mb-4 p-3 bg-card border border-border rounded">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-muted-foreground">{t('sessions.settings.modelLabel')}</span>
              <p className="font-medium text-sm truncate" title={session.modelPath}>{shortName}</p>
              <p className="text-xs text-muted-foreground truncate">{session.modelPath}</p>
            </div>
            <div className="text-right text-sm">
              <span className={`inline-flex items-center gap-1.5 ${isRunning ? 'text-primary' : 'text-muted-foreground'}`}>
                <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-primary' : 'bg-muted-foreground'}`} />
                {restarting ? t('status.restarting') : session.status}
              </span>
              <p className="text-xs text-muted-foreground">{config.host}:{config.port}</p>
            </div>
          </div>
        </div>

        {/* Running warning */}
        {isRunning && !restarting && (
          <div className="mb-4 p-3 bg-warning/10 border border-warning/30 rounded-lg text-sm text-warning">
            {t('sessions.settings.runningWarning')}
          </div>
        )}

        {/* Status message */}
        {message && (
          <div className={`mb-4 p-3 rounded-lg text-sm ${message.type === 'success'
            ? 'bg-primary/10 border border-primary/30 text-primary'
            : 'bg-destructive/10 border border-destructive/30 text-destructive'
            }`}>
            {message.text}
          </div>
        )}

        {/* Config Form */}
        <SessionConfigForm config={config} onChange={handleChange} onReset={handleReset} detectedCacheType={detectedConfig?.cacheType} detectedFamily={detectedConfig?.family} detectedIsTurboQuant={detectedConfig?.isTurboQuant} detectedIsMultimodal={detectedConfig?.isMultimodal} detectedForceTextOnly={detectedConfig?.forceTextOnly} detectedMaxContext={detectedConfig?.maxContextLength} detectedNativeMtp={(detectedConfig as any)?.nativeMtp} modelType={(() => { try { return JSON.parse(session.config || '{}').modelType } catch { return undefined } })()} sessionId={sessionId} />

        {/* Command Preview */}
        <div className="mt-4">
          <button
            onClick={() => setShowPreview(!showPreview)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            <ChevronRight className={`h-4 w-4 transition-transform ${showPreview ? 'rotate-90' : ''}`} /> {t('sessions.settings.cliCommandPreview')}
          </button>
          {showPreview && (
            <pre className="mt-2 p-3 bg-background/80 text-primary text-xs font-mono rounded-lg overflow-x-auto whitespace-pre-wrap">
              {buildCommandPreview(session.modelPath, config, detectedConfig)}
            </pre>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-3 mt-6 pb-6">
          <button onClick={onBack} className="px-4 py-2 border border-border rounded hover:bg-accent">
            {t('common.back')}
          </button>
          <button
            onClick={handleSave}
            disabled={!dirty || saving || restarting}
            className="px-6 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium disabled:opacity-40"
          >
            {saving && !restarting ? t('common.saving') : t('sessions.settings.saveSettings')}
          </button>
          {isRunning && (
            <button
              onClick={handleSaveAndRestart}
              disabled={saving || restarting}
              className="px-6 py-2 bg-success text-success-foreground rounded hover:bg-success/90 font-medium disabled:opacity-40"
            >
              {restarting ? t('sessions.settings.restarting') : t('sessions.settings.saveAndRestart')}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
