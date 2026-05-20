import { useEffect, useState, useRef } from 'react'
import { Modal } from '../ui/Modal'
import { DistributedNodeList } from './DistributedNodeList'
import { useTranslation } from '../../i18n'
import {
  cacheControlUpdatesForBlockDiskToggle,
  cacheControlUpdatesForDiskToggle,
  cacheControlUpdatesForPagedToggle,
  resolveCacheControlPolicy,
  type CacheControlUpdate,
} from '../../../../shared/cacheControlPolicy'
import { normalizeMcpPolicyList } from '../../../../shared/mcpPolicy'
export interface SessionConfig {
  host: string
  port: number
  apiKey: string
  rateLimit: number
  timeout: number
  maxNumSeqs: number
  prefillBatchSize: number
  prefillStepSize: number
  completionBatchSize: number
  continuousBatching: boolean
  enablePrefixCache: boolean
  prefixCacheSize: number
  prefixCacheMaxBytes: number
  cacheMemoryMb: number
  cacheMemoryPercent: number
  cacheTtlMinutes: number
  noMemoryAwareCache: boolean
  usePagedCache: boolean
  pagedCacheBlockSize: number
  maxCacheBlocks: number
  kvCacheQuantization: string
  kvCacheGroupSize: number
  // Nemotron-Omni multimodal backend. 'stage1' = bit-exact PyTorch+MPS
  // bridge (default, slower). 'stage2' = native MLX RADIO + Parakeet,
  // ~15–21× faster encoders + ~82 tok/s decode (the JANGQ-AI banner
  // numbers). Default-off pending Wave-4 quality validation.
  omniBackend: 'stage1' | 'stage2'
  enableDiskCache: boolean
  diskCacheMaxGb: number
  diskCacheDir: string
  enableBlockDiskCache: boolean
  blockDiskCacheMaxGb: number
  blockDiskCacheDir: string
  streamInterval: number
  maxTokens: number
  mcpConfig: string
  mcpEnabledServers: string
  mcpDisabledServers: string
  mcpEnabledTools: string
  mcpDisabledTools: string
  enableAutoToolChoice?: boolean
  toolCallParser: string
  reasoningParser: string
  isMultimodal?: boolean
  servedModelName: string
  speculativeModel: string
  numDraftTokens: number
  nativeMtpMode?: 'deterministic' | 'auto' | 'off'
  nativeMtpDepth?: number
  nativeMtpDepthOverride?: boolean
  smelt: boolean
  smeltExperts: number
  flashMoe: boolean
  flashMoeSlotBank: number
  flashMoePrefetch: 'none' | 'temporal'
  flashMoeIoSplit: number
  defaultTemperature: number
  defaultTopP: number
  defaultTopK?: number
  defaultMinP?: number
  defaultRepetitionPenalty: number
  defaultMaxNewTokens?: number
  defaultEnableThinking?: boolean
  dsv4PoolQuant?: boolean
  embeddingModel: string
  additionalArgs: string
  enableJit: boolean
  idleTimeoutSoftMin?: number
  idleTimeoutHardMin?: number
  autoSleepEnabled?: boolean
  logLevel: string
  corsOrigins: string
  maxContextLength: number
  chatTemplate?: string
  imageMode?: string
  imageQuantize?: number
  // VLM video sampling — forwarded as video_fps / video_max_frames on request
  videoFps?: number
  videoMaxFrames?: number
  // Distributed compute
  distributedEnabled?: boolean
  distributedMode?: 'pipeline' | 'tensor'
  distributedSecret?: string
  distributedNodes?: Array<{ address: string; port: number; hostname?: string }>
}

export const DEFAULT_CONFIG: SessionConfig = {
  host: '127.0.0.1',
  port: 8000,
  apiKey: '',
  rateLimit: 0,
  timeout: 300,
  maxNumSeqs: 1,
  // Default to the production cache stack: continuous batching is the backend
  // path that enables prefix, paged KV, block-L2, and stored-cache codecs.
  // Keep max sequences at one for normal local chat so users get the cache
  // features without reserving a large multi-user batch shape.
  prefillBatchSize: 512,
  prefillStepSize: 2048,
  completionBatchSize: 512,
  continuousBatching: true,
  enablePrefixCache: true,
  prefixCacheSize: 100,
  prefixCacheMaxBytes: 0,
  cacheMemoryMb: 0,
  cacheMemoryPercent: 15,
  cacheTtlMinutes: 0,
  noMemoryAwareCache: false,
  usePagedCache: true,
  pagedCacheBlockSize: 64,
  maxCacheBlocks: 1000,
  kvCacheQuantization: 'auto',
  kvCacheGroupSize: 64,
  omniBackend: 'stage1',
  enableDiskCache: true,
  diskCacheMaxGb: 10,
  diskCacheDir: '',
  enableBlockDiskCache: true,
  blockDiskCacheMaxGb: 10,
  blockDiskCacheDir: '',
  streamInterval: 1,
  maxTokens: 0,
  mcpConfig: '',
  mcpEnabledServers: '',
  mcpDisabledServers: '',
  mcpEnabledTools: '',
  mcpDisabledTools: '',
  // enableAutoToolChoice intentionally omitted (undefined = auto-detect from model config).
  // false blocks auto-detection because ?? doesn't fall through on false.
  toolCallParser: 'auto',
  reasoningParser: 'auto',
  isMultimodal: undefined,
  servedModelName: '',
  speculativeModel: '',
  numDraftTokens: 3,
  nativeMtpMode: 'deterministic',
  nativeMtpDepth: 3,
  nativeMtpDepthOverride: false,
  smelt: false,
  smeltExperts: 50,
  flashMoe: false,
  flashMoeSlotBank: 256,
  flashMoePrefetch: 'none',
  flashMoeIoSplit: 4,
  defaultTemperature: 0,
  defaultTopP: 0,
  defaultTopK: 0,
  defaultMinP: 0,
  defaultRepetitionPenalty: 0,
  defaultMaxNewTokens: 0,
  defaultEnableThinking: undefined,
  dsv4PoolQuant: false,
  embeddingModel: '',
  additionalArgs: '',
  enableJit: true,
  logLevel: 'INFO',
  corsOrigins: '*',
  maxContextLength: 0,
  imageMode: undefined,
  imageQuantize: undefined,
  // VLM defaults — 2 fps × 8 max frames = reasonable for Qwen 3.6 video (native
  // temporal embedding capacity). mlx_vlm/models/mllm.py DEFAULT_FPS=2.0.
  videoFps: 2,
  videoMaxFrames: 8,
}

const DSV4_PAGED_CACHE_BLOCK_SIZE = 256

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

// Expert = current defaults (backwards compatible, full control)
export const EXPERT_CONFIG = { ...DEFAULT_CONFIG }

// Casual: safest optimized defaults for low-compute machines.
// Keep cache codec on Auto so model architecture decides: calibrated TQ-KV for
// compatible plain KV rows, native typed cache for hybrid/DSV4/ZAYA rows.
// Resource ceilings lowered to prevent OOM on 32-48GB machines with large models.
export const CASUAL_CONFIG: SessionConfig = {
  ...DEFAULT_CONFIG,
  host: '127.0.0.1',         // Local-only (safer for beginners)
  maxNumSeqs: 1,              // Single user (saves memory from batch overhead)
  prefillBatchSize: 8,        // Low-memory default (override DEFAULT_CONFIG's 512)
  completionBatchSize: 32,    // Low-memory default (override DEFAULT_CONFIG's 512)
  cacheMemoryPercent: 15,     // 15% vs 30% — more headroom for model weights
  maxCacheBlocks: 500,        // Fewer paged blocks (half)
  prefixCacheSize: 50,        // Fewer cached prefixes
  kvCacheQuantization: 'auto', // Do not pass explicit q4; that disables calibrated live TQ-KV.
  maxTokens: 8192,            // 8K vs 32K — prevents huge KV allocation
  enableJit: true,            // JIT on by default (includes warmup for cold-start OOM prevention)
}

interface LiveMcpServer {
  name: string
  state?: string
  transport?: string
  tools_count?: number
  enabled?: boolean
  configured?: boolean
  error?: string | null
}

interface LiveMcpTool {
  name: string
  description?: string
  server?: string
  effective?: boolean
  enabled?: boolean
  transport?: string
  server_state?: string
  error?: string | null
}

interface SessionConfigFormProps {
  config: SessionConfig
  onChange: <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => void
  onReset?: () => void
  /** Detected model cache type ('kv', 'mamba', etc.) for feature gating */
  detectedCacheType?: string
  /** Detected model family for feature gating where cache type alone is ambiguous */
  detectedFamily?: string
  /** True for JANGTQ/MXTQ models whose live TurboQuant KV cache cannot be mx.compile traced */
  detectedIsTurboQuant?: boolean
  /** True for VLM/MLLM models detected from config/capabilities */
  detectedIsMultimodal?: boolean
  /** True when a model has media metadata but must use the text runtime */
  detectedForceTextOnly?: boolean
  /** Detected model max context length from config.json (max_position_embeddings) */
  detectedMaxContext?: number
  /** Native MTP capability from config/index metadata */
  detectedNativeMtp?: {
    supported?: boolean
    depth?: number
    depthSource?: string
    runtimeScope?: string
    requiresDeterministicSampling?: boolean
  }
  /** Model type — image models show minimal settings */
  modelType?: 'text' | 'image'
  /** Image mode — 'edit' or 'generate' (only relevant when modelType is 'image') */
  imageMode?: string
  /** Session ID for components that need to query the running backend (e.g. DistributedNodeList). Omit for the CreateSession form where the session doesn't exist yet. */
  sessionId?: string
}

export function SessionConfigForm({ config, onChange, onReset, detectedCacheType, detectedFamily, detectedIsTurboQuant, detectedIsMultimodal, detectedForceTextOnly, detectedMaxContext, detectedNativeMtp, modelType, imageMode, sessionId }: SessionConfigFormProps) {
  const { t } = useTranslation()
  const isImage = modelType === 'image'
  const isImageEdit = isImage && (imageMode === 'edit' || config.imageMode === 'edit')
  const [expandedSections, setExpandedSections] = useState({
    server: true,
    concurrent: false,
    distributed: false,
    prefixCache: false,
    pagedCache: false,
    kvCacheQuant: false,
    diskCache: false,
    power: false,
    performance: false,
    tools: false,
    specDecode: false,
    nativeMtp: true,
  })

  const [showCachingHelp, setShowCachingHelp] = useState(false)
  const [mcpStatus, setMcpStatus] = useState<{ servers: LiveMcpServer[]; tools: LiveMcpTool[]; error?: string } | null>(null)
  const [mcpStatusLoading, setMcpStatusLoading] = useState(false)
  const [mcpValidation, setMcpValidation] = useState<{ servers: any[]; serverCount?: number; error?: string } | null>(null)
  const [mcpValidationLoading, setMcpValidationLoading] = useState(false)
  const [mcpImportLoading, setMcpImportLoading] = useState(false)

  const normalizedDetectedFamily = normalizeDetectedFamilyName(detectedFamily)
  const dsv4Active = normalizedDetectedFamily === 'deepseek-v4'
  const effectiveSmeltActive = !!config.smelt && !dsv4Active
  const effectiveFlashMoeActive = !!config.flashMoe && !dsv4Active
  const effectiveDistributedActive = !!config.distributedEnabled && !dsv4Active
  const smeltActive = effectiveSmeltActive
  const flashMoeActive = effectiveFlashMoeActive
  const distributedActive = effectiveDistributedActive
  const zayaCcaActive = isZayaCcaFamily(normalizedDetectedFamily)
  const turboQuantActive = !!detectedIsTurboQuant
  const multimodalActive = !dsv4Active && !detectedForceTextOnly && (!!detectedIsMultimodal || config.isMultimodal === true)
  const hybridCacheActive = detectedCacheType === 'hybrid' || detectedCacheType === 'mamba'
  const effectiveContinuousBatching = dsv4Active ? true : config.continuousBatching
  const batchingOff = !effectiveContinuousBatching
  const effectivelyNoBatching = batchingOff
  const prefixOff = !config.enablePrefixCache
  const isMambaCache = detectedCacheType === 'mamba' || detectedCacheType === 'hybrid'
  const architectureRequiresPagedCache = zayaCcaActive || dsv4Active || isMambaCache
  const zayaTypedCacheRequiresPaged = zayaCcaActive && !batchingOff && !prefixOff
  const dsv4CompositeRequiresPaged = dsv4Active && !batchingOff && !prefixOff
  const nativeCacheRequiresPaged = isMambaCache && !batchingOff && !prefixOff
  const cacheControlState = {
    continuousBatching: effectiveContinuousBatching,
    enablePrefixCache: config.enablePrefixCache,
    usePagedCache: config.usePagedCache,
    enableDiskCache: config.enableDiskCache,
    enableBlockDiskCache: config.enableBlockDiskCache,
    architectureRequiresPagedCache,
  }
  const cachePolicy = resolveCacheControlPolicy(cacheControlState)
  const effectiveUsePagedCache = cachePolicy.effectiveUsePagedCache
  const effectivePagedCacheBlockSize = dsv4CompositeRequiresPaged
    ? DSV4_PAGED_CACHE_BLOCK_SIZE
    : config.pagedCacheBlockSize
  const pagedCacheSectionTitle = dsv4CompositeRequiresPaged
    ? 'DSV4 Native Cache'
    : t('sessions.config.pagedKVCache')
  const pagedCacheToggleLabel = dsv4CompositeRequiresPaged
    ? 'Native Composite Prefix Cache'
    : 'Use Paged KV Cache'
  const effectiveStoredCacheQuantization = dsv4Active ? 'auto' : config.kvCacheQuantization
  const effectiveMaxNumSeqs = dsv4Active ? 1 : config.maxNumSeqs
  const effectivePrefillBatchSize = dsv4Active ? 1 : config.prefillBatchSize
  const effectiveCompletionBatchSize = dsv4Active ? 1 : config.completionBatchSize
  const showVideoControls = !dsv4Active && !detectedForceTextOnly && multimodalActive
  const nativeMtpSupported = !!detectedNativeMtp?.supported
  const nativeMtpMode = config.nativeMtpMode || DEFAULT_CONFIG.nativeMtpMode || 'deterministic'
  const nativeMtpDepth = config.nativeMtpDepthOverride === true
    ? (config.nativeMtpDepth || detectedNativeMtp?.depth || 3)
    : (detectedNativeMtp?.depth || config.nativeMtpDepth || 3)
  const generationDefaultsSummary = [
    (config.defaultMaxNewTokens ?? 0) > 0 ? `max output tokens ${Math.floor(config.defaultMaxNewTokens ?? 0)}` : null,
    config.defaultTemperature > 0 ? `temperature ${(config.defaultTemperature / 100).toFixed(2)}` : null,
    config.defaultTopP > 0 ? `top-p ${(config.defaultTopP / 100).toFixed(2)}` : null,
    (config.defaultTopK ?? 0) > 0 ? `top-k ${Math.floor(config.defaultTopK ?? 0)}` : null,
    (config.defaultMinP ?? 0) > 0 ? `min-p ${((config.defaultMinP ?? 0) / 100).toFixed(2)}` : null,
    config.defaultRepetitionPenalty > 0 ? `repetition ${(config.defaultRepetitionPenalty / 100).toFixed(2)}` : null,
  ].filter(Boolean).join(', ')

  const toggleSection = (section: keyof typeof expandedSections) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  const applyCacheControlUpdates = (updates: CacheControlUpdate[]) => {
    updates.forEach(([key, value]) => onChange(key, value))
  }

  const browseMcpConfig = async () => {
    const result = await window.api.sessions.browseMcpConfig()
    if (!result?.canceled && result.filePath) {
      onChange('mcpConfig', result.filePath)
      validateMcpConfig(result.filePath)
    }
  }

  const importMcpConfig = async () => {
    setMcpImportLoading(true)
    try {
      const result = await window.api.sessions.importMcpConfig(config.mcpConfig?.trim() || undefined)
      if (result?.canceled) return
      if (result?.success && result.importedPath) {
        onChange('mcpConfig', result.importedPath)
        setMcpValidation({
          servers: Array.isArray(result.servers) ? result.servers : [],
          serverCount: result.serverCount,
        })
      } else {
        setMcpValidation({ servers: [], error: result?.error || 'MCP config import failed' })
      }
    } catch (error) {
      setMcpValidation({ servers: [], error: (error as Error).message })
    } finally {
      setMcpImportLoading(false)
    }
  }

  const validateMcpConfig = async (path = config.mcpConfig) => {
    if (!path?.trim()) {
      setMcpValidation({ servers: [], error: 'MCP config path is empty' })
      return
    }
    setMcpValidationLoading(true)
    try {
      const result = await window.api.sessions.validateMcpConfig(path)
      if (result?.success) {
        setMcpValidation({
          servers: Array.isArray(result.servers) ? result.servers : [],
          serverCount: result.serverCount,
        })
      } else {
        setMcpValidation({ servers: [], error: result?.error || 'MCP config validation failed' })
      }
    } catch (error) {
      setMcpValidation({ servers: [], error: (error as Error).message })
    } finally {
      setMcpValidationLoading(false)
    }
  }

  const refreshMcpStatus = async () => {
    if (!sessionId) return
    setMcpStatusLoading(true)
    try {
      const result = await window.api.sessions.mcpStatus(sessionId)
      if (result?.success) {
        setMcpStatus({
          servers: Array.isArray(result.servers) ? result.servers : [],
          tools: Array.isArray(result.tools) ? result.tools : [],
        })
      } else {
        setMcpStatus({ servers: [], tools: [], error: result?.error || 'MCP status unavailable' })
      }
    } catch (error) {
      setMcpStatus({ servers: [], tools: [], error: (error as Error).message })
    } finally {
      setMcpStatusLoading(false)
    }
  }

  useEffect(() => {
    if (expandedSections.tools && sessionId) {
      refreshMcpStatus()
    }
  }, [expandedSections.tools, sessionId])

  const joinPolicyList = (values: Iterable<string>) => Array.from(values).sort().join('\n')
  const policyServers = normalizeMcpPolicyList(config.mcpEnabledServers)
  const policyDisabledServers = normalizeMcpPolicyList(config.mcpDisabledServers)
  const policyEnabledTools = normalizeMcpPolicyList(config.mcpEnabledTools)
  const policyDisabledTools = normalizeMcpPolicyList(config.mcpDisabledTools)

  const toggleMcpServer = (serverName: string, enabled: boolean) => {
    const allowed = new Set(policyServers)
    const denied = new Set(policyDisabledServers)
    if (enabled) {
      denied.delete(serverName)
      if (allowed.size) allowed.add(serverName)
    } else {
      denied.add(serverName)
      allowed.delete(serverName)
    }
    onChange('mcpEnabledServers', joinPolicyList(allowed))
    onChange('mcpDisabledServers', joinPolicyList(denied))
  }

  const toggleMcpTool = (toolName: string, enabled: boolean) => {
    const allowed = new Set(policyEnabledTools)
    const denied = new Set(policyDisabledTools)
    if (enabled) {
      denied.delete(toolName)
      if (allowed.size) allowed.add(toolName)
    } else {
      denied.add(toolName)
      allowed.delete(toolName)
    }
    onChange('mcpEnabledTools', joinPolicyList(allowed))
    onChange('mcpDisabledTools', joinPolicyList(denied))
  }

  return (
    <div className="space-y-0">
      {/* Server Settings */}
      <Section title={t('sessions.config.serverSettings')} expanded={expandedSections.server} onToggle={() => toggleSection('server')}>
        <Field label={t('sessions.config.host')} tooltip="The network interface to bind to. Default 127.0.0.1 (localhost only). Change to 0.0.0.0 to allow connections from other machines on your network (LAN access). Use an API key when binding to 0.0.0.0.">
          <input type="text" value={config.host} onChange={e => onChange('host', e.target.value)} className="cfg-input" />
        </Field>
        <SliderField
          label={t('sessions.config.port')}
          tooltip="The TCP port the server listens on. Each running model instance needs a unique port. Ports are auto-assigned starting from 8000. You can manually set any port between 1024-65535 that isn't already in use."
          value={config.port}
          onChange={v => onChange('port', v)}
          min={1024}
          max={65535}
          step={1}
          defaultValue={DEFAULT_CONFIG.port}
        />
        <Field label={t('sessions.config.apiKey')} tooltip="Optional authentication key for the OpenAI-compatible API. When set, all API requests must include this key in the Authorization header. Leave empty to allow unauthenticated access (fine for local-only servers).">
          <input type="password" value={config.apiKey} onChange={e => onChange('apiKey', e.target.value)} placeholder={t('sessions.config.apiKeyPlaceholder')} className="cfg-input" />
        </Field>
        <Field label={t('sessions.config.servedModelName')} tooltip="Custom name to expose via the /v1/models API and in response objects. When set, API clients can use this name instead of the full model path. Both the custom name and the actual model name are listed in /v1/models. Leave empty to auto-derive from model path (e.g. 'mlx-community/Llama-3.2-3B').">
          <input type="text" value={config.servedModelName} onChange={e => onChange('servedModelName', e.target.value)} placeholder={t('sessions.config.servedModelNamePlaceholder')} className="cfg-input" />
        </Field>
        <SliderField
          label={t('sessions.config.rateLimit')}
          tooltip="Maximum number of API requests allowed per minute. Set to 0 to disable rate limiting. Useful when exposing the server to multiple users or external applications to prevent overloading."
          value={config.rateLimit}
          onChange={v => onChange('rateLimit', v)}
          min={1}
          max={1000}
          step={10}
          defaultValue={60}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={t('sessions.config.rateLimitDisabled')}
        />
        <SliderField
          label={t('sessions.config.timeout')}
          tooltip="Maximum time in seconds to wait for a single inference request to complete before timing out. Increase this for very long generations or slow models. Default 300s (5 minutes) should be sufficient for most use cases."
          value={config.timeout}
          onChange={v => onChange('timeout', v)}
          min={10}
          max={3600}
          step={10}
          defaultValue={DEFAULT_CONFIG.timeout}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={t('sessions.config.timeoutNoLimit')}
        />
        <Field label={t('sessions.config.logLevel')} tooltip="Controls how much detail the server logs. DEBUG shows everything (very verbose). INFO is the default. WARNING and ERROR reduce noise to only important messages.">
          <select value={config.logLevel || 'INFO'} onChange={e => onChange('logLevel', e.target.value)} className="cfg-input">
            <option value="DEBUG">{t('sessions.config.logLevelDebug')}</option>
            <option value="INFO">{t('sessions.config.logLevelInfo')}</option>
            <option value="WARNING">{t('sessions.config.logLevelWarning')}</option>
            <option value="ERROR">{t('sessions.config.logLevelError')}</option>
          </select>
        </Field>
        <Field label={t('sessions.config.corsOrigins')} tooltip="Allowed origins for cross-origin API requests (from web browsers). Use * to allow all origins, or a comma-separated list of specific origins (e.g. http://localhost:3000,https://myapp.com). Only matters when external web apps call your API.">
          <input type="text" value={config.corsOrigins || '*'} onChange={e => onChange('corsOrigins', e.target.value)} placeholder={t('sessions.config.corsPlaceholder')} className="cfg-input" />
        </Field>
      </Section>

      {/* Concurrent Processing */}
      {isImage && (
        <div className="px-4 py-3 text-xs text-muted-foreground border-b border-border">
          {isImageEdit
            ? <>{t('sessions.config.imageEditServerNote')}</>
            : <>{t('sessions.config.imageGenServerNote')}</>
          }
        </div>
      )}

      <Section title={t('sessions.config.concurrentProcessing')} expanded={expandedSections.concurrent} onToggle={() => toggleSection('concurrent')} hidden={isImage}>
        <div className="flex items-center gap-2 mb-2">
          <PerformanceHint text="Controls how many requests your server handles at once. Keep Continuous Batching ON to enable the caching engine." />
          <button
            onClick={(e) => { e.preventDefault(); e.stopPropagation(); setShowCachingHelp(true) }}
            className="w-6 h-6 flex items-center justify-center rounded-full bg-accent/50 text-accent-foreground hover:bg-accent hover:text-white transition-colors text-xs font-bold"
            title={t('sessions.config.cachingReferenceTitle')}
          >
            ?
          </button>
        </div>
        <SliderField
          label="Max Concurrent Sequences"
          tooltip="Maximum number of sequences (requests) that can be processed simultaneously. Higher values allow more parallel users but consume more memory. For single-user local use, 1-4 is sufficient. For multi-user servers, 16-256 depending on available RAM."
          value={effectiveMaxNumSeqs}
          onChange={v => onChange('maxNumSeqs', v)}
          min={1}
          max={1024}
          step={1}
          defaultValue={DEFAULT_CONFIG.maxNumSeqs}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Default (1)"
          disabled={dsv4Active}
        />
        <SliderField
          label="Prefill Batch Size"
          tooltip="Maximum number of concurrent prompts processed in parallel during the prefill (prompt processing) phase. Higher = more parallelism for multi-user workloads, more memory pressure during prompt ingest."
          value={effectivePrefillBatchSize}
          onChange={v => onChange('prefillBatchSize', v)}
          min={1}
          max={4096}
          step={64}
          defaultValue={DEFAULT_CONFIG.prefillBatchSize}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Default (512)"
          disabled={dsv4Active}
        />
        <SliderField
          label="Prefill Step Size"
          tooltip="Maximum number of tokens processed in a single prefill forward pass per sequence. Larger = fewer kernel launches and faster prefill, more transient memory. Reduce if you OOM mid-prompt on long contexts."
          value={config.prefillStepSize}
          onChange={v => onChange('prefillStepSize', v)}
          min={64}
          max={8192}
          step={64}
          defaultValue={DEFAULT_CONFIG.prefillStepSize}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Default (2048)"
          disabled={dsv4Active}
        />
        <SliderField
          label="Completion Batch Size"
          tooltip="Maximum number of tokens to generate in a single completion (token generation) step. Similar to prefill batch size but for the generation phase. Larger values can improve throughput for multi-user scenarios."
          value={effectiveCompletionBatchSize}
          onChange={v => onChange('completionBatchSize', v)}
          min={1}
          max={4096}
          step={64}
          defaultValue={DEFAULT_CONFIG.completionBatchSize}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel="Default (512)"
          disabled={dsv4Active}
        />
        <CheckField
          label="Smelt Mode"
          tooltip="Partial expert loading for MoE models. Loads backbone + N% of experts from SSD, reducing RAM by ~50% while maintaining ~97% baseline speed via cache-biased routing and native SwitchGLU kernels."
          checked={effectiveSmeltActive}
          onChange={v => {
            onChange('smelt', v)
            // Mutual exclusion: disable Flash MoE if enabling Smelt
            if (v && flashMoeActive) onChange('flashMoe', false)
          }}
          disabled={dsv4Active || effectiveFlashMoeActive}
        />
        {dsv4Active && (
          <IncompatWarning text="Smelt is disabled for DSV4 Flash. DSV4 uses the verified native JANG affine loader and composite cache path." />
        )}
        {flashMoeActive && (
          <IncompatWarning text="Smelt is disabled while Flash MoE is on. They both modify MoE expert layers — use one or the other." />
        )}
        {smeltActive && (
          <SliderField label="Smelt Experts %" value={config.smeltExperts} onChange={v => onChange('smeltExperts', v)} min={10} max={100} step={5} defaultValue={50} />
        )}
        {smeltActive && <PerformanceHint text={`Loading ${config.smeltExperts}% of experts per MoE layer. Lower = less RAM, slightly more routing bias.`} />}

        <CheckField
          label="Flash MoE (SSD Streaming)"
          tooltip="Streams MoE expert weights from SSD on-demand instead of keeping them all in RAM. Enables massive MoE models (35B-397B) to run on machines with limited RAM by caching only recently-used experts in a slot-bank cache. Incompatible with Smelt, Distributed, and JIT. ~50% slower than full-RAM mode due to on-demand disk loading."
          checked={effectiveFlashMoeActive}
          onChange={v => {
            onChange('flashMoe', v)
            // Mutual exclusion: disable conflicting features
            if (v) {
              if (smeltActive) onChange('smelt', false)
              if (distributedActive) onChange('distributedEnabled', false)
              if (config.enableJit) onChange('enableJit', false)
            }
          }}
          disabled={dsv4Active || effectiveSmeltActive || effectiveDistributedActive}
        />
        {dsv4Active && (
          <IncompatWarning text="Flash MoE is disabled for DSV4 Flash. DSV4 native expert hydration and SWA+CSA/HCA cache restore are not compatible with SSD expert streaming." />
        )}
        {(smeltActive || distributedActive) && !flashMoeActive && (
          <IncompatWarning text={`Flash MoE is disabled while ${smeltActive ? 'Smelt' : 'Distributed'} is on. Turn it off to enable Flash MoE.`} />
        )}
        {flashMoeActive && (
          <>
            <SliderField
              label="Slot Bank Size"
              tooltip="Number of expert weight sets cached in RAM. Higher = more cache hits but more RAM. Recommended: 64 for Nemotron/small MoE, 256+ for Qwen3.5 MoE, 512+ for MiniMax (256 experts)."
              value={config.flashMoeSlotBank}
              onChange={v => onChange('flashMoeSlotBank', v)}
              min={16}
              max={1024}
              step={16}
              defaultValue={64}
            />
            <SliderField
              label="I/O Workers"
              tooltip="Number of parallel disk I/O threads for loading experts. Higher = faster cold loads but more I/O pressure. Default 4 works well for most SSDs."
              value={config.flashMoeIoSplit}
              onChange={v => onChange('flashMoeIoSplit', v)}
              min={1}
              max={16}
              step={1}
              defaultValue={4}
            />
            <PerformanceHint text={`Streaming experts from SSD with ${config.flashMoeSlotBank}-slot LRU cache. Non-MoE models automatically pass through (no effect). JIT disabled (incompatible with on-demand loading).`} />
          </>
        )}
        <CheckField label="Continuous Batching" tooltip="Keep ON for best performance. This is the master switch for prefix cache, paged KV cache, block disk L2, and stored-cache codecs. Turning it off uses the direct single-request engine and disables the cache features below." checked={effectiveContinuousBatching} onChange={v => onChange('continuousBatching', v)} disabled={dsv4Active} />
        <PerformanceHint text="Keep ON for best overall behavior: it enables prefix reuse, paged cache, block disk L2, and architecture-specific cache restore while the default max sequence count stays at one for local chat." />
        {dsv4Active && <InfoNote text="DSV4 Flash uses the continuous-batching DSV4BatchGenerator path for native SWA+CSA/HCA cache correctness." />}
        {!effectiveContinuousBatching && config.enablePrefixCache && (
          <InfoNote text="Cache flags will be omitted at launch while continuous batching is off. Turn it back on to use prefix cache, paged KV cache, block disk L2, and stored-cache codecs." />
        )}
        {!effectiveContinuousBatching && (
          <InfoNote text="Turning this off disables: prefix caching, paged KV cache, KV cache quantization, and disk caching. Enable it to unlock these features." />
        )}
      </Section>

      {/* Prefix Cache */}
      <Section title={t('sessions.config.prefixCache')} expanded={expandedSections.prefixCache} onToggle={() => toggleSection('prefixCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text="Speeds up repeated conversations by remembering previous prompts. Makes follow-up messages much faster (lower time-to-first-token)." />}
        {batchingOff && <IncompatWarning text="Prefix cache requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above to enable prefix caching." />}
        <CheckField label="Enable Prefix Cache" tooltip="Caches prompt prefixes in memory. If you send the same system prompt or document multiple times, the server reuses the cached internal states instead of recomputing them, drastically reducing Time-To-First-Token (TTFT) and saving GPU compute. Highly recommended for agents and tool calling." checked={config.enablePrefixCache} onChange={v => onChange('enablePrefixCache', v)} />
        {config.enablePrefixCache && (
          <>
            {dsv4Active ? (
              <InfoNote text="DSV4 Flash stores native SWA+CSA/HCA prompt-boundary state through the paged prefix path. Generic memory-aware and legacy entry-count prefix-cache controls are not used." />
            ) : (
            <>
            <CheckField label="Legacy Entry-Count Cache" tooltip="Switches from memory-aware cache (which uses Cache Memory %, Cache Memory Limit, and Cache TTL controls) to a simpler entry-count cache. When ON: you control cache by max entries only. When OFF: you get fine-grained memory budget controls (% of RAM, MB limit, TTL expiration). Memory-aware mode is recommended for most users." checked={config.noMemoryAwareCache} onChange={v => onChange('noMemoryAwareCache', v)} disabled={dsv4Active} />
            {config.noMemoryAwareCache ? (
              <>
                <InfoNote text="Legacy mode active — Cache Memory %, Cache Memory Limit, and Cache TTL are hidden. Turn off 'Legacy Entry-Count Cache' above to use memory-aware caching with those controls." />
                <SliderField
                  label="Max Cache Entries"
                  tooltip="Maximum number of prefix cache entries to store when using legacy entry-count mode. Each entry stores the KV cache for one unique prefix. Higher values cache more prefixes but use more memory. For finer control over memory usage, switch to memory-aware mode by unchecking 'Legacy Entry-Count Cache' above."
                  value={config.prefixCacheSize}
                  onChange={v => onChange('prefixCacheSize', v)}
                  min={1}
                  max={10000}
                  step={10}
                  defaultValue={DEFAULT_CONFIG.prefixCacheSize}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="Default (100)"
                />
                <SliderField
                  label="Prefix Cache Max Bytes (MB)"
                  tooltip="Optional global byte budget for the legacy entry-count prefix cache. When set, eviction also fires when total cached bytes exceed this. Eviction priority is assistant → user → system, so shared system prompts persist across users/sessions. 0 = unlimited (entry-count only)."
                  value={Math.floor((config.prefixCacheMaxBytes || 0) / (1024 * 1024))}
                  onChange={v => onChange('prefixCacheMaxBytes', v * 1024 * 1024)}
                  min={0}
                  max={32768}
                  step={256}
                  defaultValue={0}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="Unlimited"
                />
              </>
            ) : (
              <>
                {effectiveUsePagedCache && (
                  <IncompatWarning text="Cache Memory Limit and Cache Memory % only apply to memory-aware non-paged prefix cache. With paged cache on, use Max Cache Blocks for L1 RAM capacity and Block Cache Max for L2 disk capacity." />
                )}
                <SliderField
                  label="Cache Memory Limit (MB)"
                  tooltip="Hard limit on memory used by the prefix cache in megabytes. Set to 'Auto-detect' to let the system auto-detect based on available RAM and the percentage setting below. Set an explicit value if you need to reserve memory for other applications."
                  value={config.cacheMemoryMb}
                  onChange={v => onChange('cacheMemoryMb', v)}
                  min={256}
                  max={65536}
                  step={256}
                  defaultValue={4096}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="Auto-detect"
                  disabled={effectiveUsePagedCache}
                />
                <SliderField
                  label="Cache Memory %"
                  tooltip="Percentage of available system RAM to allocate for the prefix cache. Only used when Cache Memory Limit is set to 'Auto-detect'. Default 20% is a balanced cache budget — lower this for large models that leave little headroom (e.g. 10-15% for 120GB+ models on 256GB systems). Higher values cache more prefixes but risk memory pressure during long generations."
                  value={config.cacheMemoryPercent}
                  onChange={v => onChange('cacheMemoryPercent', v)}
                  min={1}
                  max={100}
                  step={1}
                  defaultValue={DEFAULT_CONFIG.cacheMemoryPercent}
                  maxInput={100}
                  disabled={effectiveUsePagedCache}
                />
                {effectiveUsePagedCache && <IncompatWarning text="Cache TTL has no effect when paged cache is enabled — paged cache uses block-count LRU eviction instead. To control paged cache size, adjust 'Max Cache Blocks' in the Paged KV Cache section below. To use time-based TTL, disable 'Use Paged KV Cache' in the Paged KV Cache section." />}
                <SliderField
                  label="Cache TTL (minutes)"
                  tooltip="Time-to-live for memory-aware cache entries. Entries not accessed within this window are evicted to free memory. 'No expiration' means entries are only evicted by memory pressure. Note: this setting has no effect when Paged KV Cache is enabled (paged cache uses its own LRU eviction based on Max Cache Blocks)."
                  value={config.cacheTtlMinutes}
                  onChange={v => onChange('cacheTtlMinutes', v)}
                  min={1}
                  max={120}
                  step={5}
                  defaultValue={30}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="No expiration"
                  disabled={effectiveUsePagedCache}
                />
              </>
            )}
            </>
            )}

            {/* Caching Help Modal */}
            {showCachingHelp && (
              <Modal title="Caching & Compatibility Engine" onClose={() => setShowCachingHelp(false)} className="max-w-2xl max-h-[85vh] overflow-y-auto">
                <div className="space-y-6 text-sm">
                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">The Continuous Batching Engine</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      <strong>Continuous Batching</strong> is the heart of vMLX's server performance. Unlike simple mode (which processes exactly one request at a time), continuous batching allows multiple requests to be processed simultaneously. More importantly, <strong>it is required to enable all advanced caching features</strong> (Prefix Cache, Paged Cache, KV Quantization, and Disk Cache).
                    </p>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">Prefix Caching (Memory-Aware vs Legacy)</h3>
                    <p className="text-muted-foreground leading-relaxed mb-2">
                      Prefix caching drastically speeds up interactions by remembering previous prompts (like a system prompt or a long document), skipping the expensive prefill phase.
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                      <li><strong>Memory-Aware (Default):</strong> Intelligently manages the cache based on explicit memory boundaries (MB) or a percentage of total system RAM. It automatically evicts the oldest items when crossing these limits.</li>
                      <li><strong>Legacy Entry-Count:</strong> A simpler system that just stores a fixed number of complete prompt states regardless of their size. Useful if you want strict deterministic eviction.</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">Mamba & Hybrid Compatibility</h3>
                    <p className="text-muted-foreground leading-relaxed mb-2">
                      Newer models like Qwen 2.5/3, Falcon Mamba, and Jamba mix standard Attention (KV cache) with SSM blocks (Mamba/Arrays cache).
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                      <li><strong>KV Quantization:</strong> vMLX securely isolates Mamba layers. If you turn on KV Quantization (e.g. q8), it will safely compress the Attention layers while leaving the internal Mamba/SSM memory at full precision, ensuring no corruption or quality loss.</li>
                      <li><strong>Paged Cache Requirement:</strong> Since cumulative SSM states cannot be safely stored as continuous memory-aware blocks, the engine uses <code>--use-paged-cache</code> for these models when prefix caching is enabled.</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">KV Cache Quantization</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      By converting stored prompts to q8 or q4 precision, you can reduce the cache's RAM footprint by 2-4x. <strong>This only safely compresses saved prefixes</strong>. The actual text generation continues to run at standard full precision natively in MLX.
                    </p>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold tracking-tight text-foreground mb-2">Vision-Language (VL) Models</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      The core engine handles Vision models automatically. <strong>Prefix caching works for images too!</strong> If you repeatedly ask questions about the exact same image (like in a tool-calling flow analyzing a dashboard), the massive vision embedding prefill is cached and reused instantly.
                    </p>
                  </div>
                </div>
              </Modal>
            )}
          </>
        )}
      </Section>

      {/* Paged Cache */}
      <Section title={pagedCacheSectionTitle} expanded={expandedSections.pagedCache} onToggle={() => toggleSection('pagedCache')} hidden={isImage}>
        {!effectivelyNoBatching && !dsv4CompositeRequiresPaged && <PerformanceHint text="Reduces memory waste by splitting the KV cache into small blocks instead of one big chunk. Lets the server handle longer conversations without running out of RAM." />}
        {dsv4CompositeRequiresPaged && <PerformanceHint text="DSV4 Flash stores native SWA+CSA/HCA prompt-boundary snapshots for prefix reuse. This is not generic paged KV; the internal paged path is only the block index and L2 transport for DeepseekV4Cache state." />}
        {batchingOff && <IncompatWarning text="Paged cache requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above to enable paged cache." />}
        {!dsv4CompositeRequiresPaged && config.enableDiskCache && <IncompatWarning text="Paged cache and legacy Disk Cache cannot run simultaneously. Enabling paged cache will auto-disable legacy Disk Cache. For persistent caching with paged cache, use 'Block Disk Cache (L2)' below instead." />}
        {!dsv4CompositeRequiresPaged && !batchingOff && prefixOff && !cachePolicy.architectureRequiresPagedCache && <InfoNote text="Paged cache is a prefix-cache backend. Turning it on will enable Prefix Cache." />}
        {!batchingOff && prefixOff && cachePolicy.architectureRequiresPagedCache && <IncompatWarning text="This model uses native/paged cache when Prefix Cache is enabled. Enable Prefix Cache above to activate the architecture-specific cache stack." />}
        {zayaTypedCacheRequiresPaged && <InfoNote text="ZAYA typed CCA cache requires paged cache while prefix cache is enabled. Turn off Prefix Cache to disable this cache stack for ZAYA." />}
        {nativeCacheRequiresPaged && !zayaTypedCacheRequiresPaged && !dsv4CompositeRequiresPaged && <InfoNote text="Hybrid/Mamba cache models require paged cache while prefix cache is enabled so KV blocks and path-dependent state stay in the same cache contract." />}
        {dsv4CompositeRequiresPaged && <InfoNote text="DSV4 uses native SWA+CSA/HCA composite cache snapshots, so paged cache stays on and block size is fixed to 256 tokens for production decode compatibility." />}
        <CheckField label={pagedCacheToggleLabel} tooltip="Manages the KV cache in fixed-size pages instead of contiguous memory. Greatly reduces memory fragmentation and allows serving larger batches or larger contexts on limited GPU RAM. Extremely recommended for long conversations." checked={effectiveUsePagedCache} onChange={v => applyCacheControlUpdates(cacheControlUpdatesForPagedToggle(v, cacheControlState))} disabled={cachePolicy.pagedCacheDisabled} />
        {effectiveUsePagedCache && (
          <>
            <SliderField
              label="Block Size (tokens)"
              tooltip="Number of tokens per paged KV cache block. Smaller blocks reduce memory waste per sequence but increase overhead from managing more blocks. Default 64 is optimal for most models. DSV4 uses 256-token blocks for its native composite cache."
              value={effectivePagedCacheBlockSize}
              onChange={v => onChange('pagedCacheBlockSize', v)}
              min={1}
              max={1024}
              step={16}
              defaultValue={DEFAULT_CONFIG.pagedCacheBlockSize}
              disabled={dsv4CompositeRequiresPaged}
            />
            <SliderField
              label="Max Cache Blocks"
              tooltip="Maximum total number of KV cache blocks allocated. Total cache capacity = block_size x max_blocks tokens. Default 1000 blocks x 64 tokens = 64K tokens capacity. Increase for longer contexts, decrease to save memory."
              value={config.maxCacheBlocks}
              onChange={v => onChange('maxCacheBlocks', v)}
              min={1}
              max={100000}
              step={100}
              defaultValue={DEFAULT_CONFIG.maxCacheBlocks}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Default (1000)"
            />
          </>
        )}
        {!batchingOff && !effectiveUsePagedCache && <InfoNote text="Block Disk Cache is SSD L2 for paged/native cache blocks. Turning it on will enable Prefix Cache and Paged KV Cache." />}
        <CheckField
          label="Block Disk Cache (L2)"
          tooltip="Persist individual paged cache blocks to SSD. When a block is evicted from RAM, it's saved to disk and can be reloaded later without recomputation. Dramatically speeds up cache warm-up for repeated system prompts and common prefixes. Uses content-addressable storage with background writes so disk I/O doesn't block inference. Compatible runtimes store compressed blocks in their native codec; path-dependent architectures use typed cache records instead of generic TurboQuant."
          checked={cachePolicy.blockDiskCacheChecked}
          onChange={v => applyCacheControlUpdates(cacheControlUpdatesForBlockDiskToggle(v, cacheControlState))}
          disabled={!cachePolicy.blockDiskCacheVisible || cachePolicy.blockDiskCacheDisabled}
        />
        {cachePolicy.blockDiskCacheChecked && (
          <>
            <SliderField
              label="Block Cache Max (GB)"
              tooltip="Maximum disk space for cached blocks. Oldest blocks are evicted when exceeded. Each block is small (~100KB-1MB), so 10GB can hold tens of thousands of blocks. Set to 0 for unlimited."
              value={config.blockDiskCacheMaxGb}
              onChange={v => onChange('blockDiskCacheMaxGb', v)}
              min={0}
              max={100}
              step={1}
              defaultValue={10}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Unlimited"
            />
            <div className="block">
              <span className="text-xs font-medium text-muted-foreground">
                Block Cache Directory
                <Tooltip text="Directory for block-level disk cache files. A model-specific subdirectory is created automatically. Leave empty for default (~/.cache/vmlx-engine/block-cache/<model_hash>/)." />
              </span>
              <input
                type="text"
                value={config.blockDiskCacheDir || ''}
                onChange={e => onChange('blockDiskCacheDir', e.target.value)}
                placeholder={t('sessions.config.blockCachePlaceholder')}
                className="cfg-input text-xs"
              />
            </div>
          </>
        )}
      </Section>

      {/* KV Cache Quantization — split into two clearly-distinct controls so
          users stop assuming the dropdown's "None" default means "no cache
          compression at all". Auto mode intentionally omits the CLI flag:
          the engine can then use calibrated TurboQuant for compatible live
          KV caches, or native typed cache contracts for path-dependent
          architectures such as DSV4, ZAYA, and hybrid SSM. */}
      <Section title={t('sessions.config.kvCacheQuantization')} expanded={expandedSections.kvCacheQuant} onToggle={() => toggleSection('kvCacheQuant')} hidden={isImage}>
        {batchingOff && <IncompatWarning text="KV cache quantization requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above." />}
        {!batchingOff && prefixOff && <IncompatWarning text="KV cache quantization requires prefix cache. Enable 'Prefix Cache' above to use KV cache quantization." />}
        {!effectivelyNoBatching && !prefixOff && isMambaCache && <PerformanceHint text="Hybrid stateful cache detected — the engine keeps SSM/GLA state native and only uses cache codecs proven for that architecture. Generic TurboQuant KV is disabled unless a tested override exists." />}
        {!effectivelyNoBatching && !prefixOff && dsv4Active && <PerformanceHint text="DeepSeek-V4 keeps generic KV q4/q8 disabled. Prefix reuse uses native SWA+CSA/HCA records with optional DSV4-only pool quantization below." />}

        {/* Live/native cache codec — automatic per architecture. */}
        <div className="block">
          <span className="text-xs font-medium text-muted-foreground">
            Live Cache Codec
            <Tooltip text="Auto mode leaves the CLI flag unset so the engine can choose per architecture: calibrated TurboQuant for compatible plain KV/JANGTQ caches, native composite or typed caches for DSV4/ZAYA/hybrid SSM, and stored-prefix fallback only where that codec is valid." />
          </span>
          <div className="cfg-input flex items-center justify-between" style={{ background: 'var(--card)', cursor: 'default' }}>
            <span>Engine-selected native cache</span>
            <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--success-bg, rgba(34,197,94,0.15))', color: 'var(--success-fg, rgb(34,197,94))' }}>AUTO</span>
          </div>
        </div>

        {/* Stored prefix-cache compression — orthogonal to TurboQuant. */}
        <div className="block">
          <span className="text-xs font-medium text-muted-foreground">
            Stored Cache Quantization
            <Tooltip text="Controls how completed prompt states are stored in the prefix cache. Auto keeps the engine's production codec choice. None explicitly disables stored-cache quantization. q8/q4 force the generic stored-cache codec and also disable calibrated live TurboQuant so the explicit choice is honored." />
          </span>
          <select value={effectiveStoredCacheQuantization} onChange={e => onChange('kvCacheQuantization', e.target.value)} className="cfg-input" disabled={effectivelyNoBatching || prefixOff || dsv4Active}>
            <option value="auto">Auto (engine-selected: native/TurboQuant + stored fallback)</option>
            <option value="none">{t('sessions.config.kvQuantNone')}</option>
            <option value="q8">q8 (8-bit, ~2x stored cache savings)</option>
            <option value="q4">q4 (4-bit, ~4x stored cache savings)</option>
          </select>
        </div>
        {effectiveStoredCacheQuantization !== 'auto' && effectiveStoredCacheQuantization !== 'none' && (
          <SliderField
            label="Group Size"
            tooltip="Number of elements quantized together. Smaller groups preserve more precision but use slightly more memory for scale/zero-point metadata. Default 64 is optimal for most models."
            value={config.kvCacheGroupSize}
            onChange={v => onChange('kvCacheGroupSize', v)}
            min={32}
            max={128}
            step={32}
            defaultValue={DEFAULT_CONFIG.kvCacheGroupSize}
          />
        )}
      </Section>

      {/* Disk Cache (L2 Persistent) */}
      <Section title={t('sessions.config.diskCachePersistent')} expanded={expandedSections.diskCache} onToggle={() => toggleSection('diskCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text="Saves cached prompts to your SSD so they survive server restarts. Next time you load the same model, previous conversations warm up instantly." />}
        {dsv4Active ? (
          <InfoNote text="DSV4 Flash stores persistent prefix state through Block Disk Cache (L2) in the native cache section. Legacy disk cache is disabled because DSV4 restores typed SWA+CSA/HCA composite records, not generic KV entries." />
        ) : (
          <InfoNote text="Legacy disk cache works with memory-aware prefix cache. Block disk cache (in the Paged KV Cache section) works with paged cache. Only one can be active at a time." />
        )}
        {batchingOff && <IncompatWarning text="Disk cache requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above." />}
        {!effectivelyNoBatching && cachePolicy.legacyDiskCacheUnavailableReason === 'paged-cache-active' && <IncompatWarning text="Legacy disk cache is not compatible with paged cache. To use disk-based persistence with paged cache, use 'Block Disk Cache (L2)' in the Paged KV Cache section instead. To use this legacy disk cache, disable 'Use Paged KV Cache' first." />}
        {!effectivelyNoBatching && cachePolicy.legacyDiskCacheUnavailableReason === 'architecture-requires-paged-cache' && <IncompatWarning text="This architecture requires native/paged cache when Prefix Cache is enabled. Use 'Block Disk Cache (L2)' in the Paged KV Cache section for persistent cache storage." />}
        {!batchingOff && prefixOff && !cachePolicy.legacyDiskCacheDisabled && <InfoNote text="Disk cache is persistent L2 behind Prefix Cache. Turning it on will enable Prefix Cache and disable paged/block cache." />}
        <CheckField
          label="Enable Disk Cache"
          tooltip="Persist prompt caches to disk for reuse across server restarts. Acts as L2 cache behind the in-memory prefix cache — when a prompt isn't found in memory, it's loaded from disk instead of recomputing. Dramatically speeds up repeated prompts (system prompts, common prefixes). Compatible runtimes store compressed cache data in their native format; path-dependent caches use typed restore records. Requires prefix cache to be enabled. Note: not compatible with paged cache (uses different storage format)."
          checked={cachePolicy.legacyDiskCacheChecked}
          onChange={v => applyCacheControlUpdates(cacheControlUpdatesForDiskToggle(v, cacheControlState))}
          disabled={cachePolicy.legacyDiskCacheDisabled}
        />
        {cachePolicy.legacyDiskCacheChecked && (
          <>
            <SliderField
              label="Max Cache Size (GB)"
              tooltip="Maximum disk space for cached prompt states. Oldest entries are evicted when this limit is exceeded. Set to 0 for unlimited. Each cached prompt typically uses 50-500MB depending on model size and prompt length."
              value={config.diskCacheMaxGb}
              onChange={v => onChange('diskCacheMaxGb', v)}
              min={0}
              max={100}
              step={1}
              defaultValue={10}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Unlimited"
            />
            <div className="block">
              <span className="text-xs font-medium text-muted-foreground">
                Cache Directory
                <Tooltip text="Base directory for disk cache files (.safetensors). A model-specific subdirectory is created automatically. Leave empty for the default location (~/.cache/vmlx-engine/prompt-cache/<model>/). Set a custom path if you want to use a specific drive." />
              </span>
              <input
                type="text"
                value={config.diskCacheDir || ''}
                onChange={e => onChange('diskCacheDir', e.target.value)}
                placeholder={t('sessions.config.diskCachePathPlaceholder')}
                className="cfg-input text-xs"
              />
            </div>
          </>
        )}
      </Section>

      {/* Power Management — visible for ALL model types (text + image) */}
      <Section title={t('sessions.config.powerManagement')} expanded={expandedSections.power} onToggle={() => toggleSection('power')}>
        <PerformanceHint text="Control when idle models automatically sleep to free GPU memory. Sleeping models auto-wake when a new request arrives." />
        <Field label="Auto-Sleep" tooltip="Automatically put the model to sleep after a period of inactivity to free memory. Light sleep clears caches but keeps the model loaded (instant wake). Deep sleep unloads the model entirely (2-15s wake). Models auto-wake when a new request arrives.">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={config.autoSleepEnabled !== false}
              onChange={e => onChange('autoSleepEnabled', e.target.checked)}
              className="rounded border-input"
            />
            <span className="text-xs text-muted-foreground">
              Put model to sleep when idle (frees GPU memory)
            </span>
          </label>
        </Field>
        {config.autoSleepEnabled !== false && (
          <>
            <SliderField
              label="Light Sleep After"
              tooltip="Minutes of inactivity before entering light sleep. Light sleep clears KV/prefix caches to free memory but keeps the model loaded in GPU. Wake is instant — no reload needed. Set to 0 to disable light sleep."
              value={config.idleTimeoutSoftMin ?? (isImage ? 5 : 10)}
              onChange={v => onChange('idleTimeoutSoftMin', v)}
              min={0}
              max={120}
              step={1}
              defaultValue={isImage ? 5 : 10}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Disabled"
            />
            <SliderField
              label="Deep Sleep After"
              tooltip="Minutes of inactivity before entering deep sleep. Deep sleep unloads the model entirely from GPU memory. The server process stays alive and the model auto-reloads when a new request arrives (2-15 seconds for most models). Set to 0 to disable deep sleep."
              value={config.idleTimeoutHardMin ?? (isImage ? 15 : 30)}
              onChange={v => onChange('idleTimeoutHardMin', v)}
              min={0}
              max={240}
              step={1}
              defaultValue={isImage ? 15 : 30}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Disabled"
            />
          </>
        )}
      </Section>

      {/* Performance */}
      <Section title={t('sessions.config.performanceGeneration')} expanded={expandedSections.performance} onToggle={() => toggleSection('performance')} hidden={isImage}>
        <PerformanceHint text="Controls how tokens stream to you and the max response length. For chat, keep stream interval at 1. Max tokens limits how long a single reply can be." />
        {/* JIT is not available for image models or VLM chat models. */}
        <Field label="JIT Compile (mx.compile)" tooltip="Enable Metal kernel fusion via mx.compile on the model forward pass. This optimizes GPU operations for faster inference after a one-time warmup on the first request. May not work with all models — falls back gracefully if compilation fails. Requires restart.">
          <label className={`flex items-center gap-2 ${flashMoeActive || distributedActive || dsv4Active || zayaCcaActive || turboQuantActive || multimodalActive || hybridCacheActive ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>
            <input
              type="checkbox"
              checked={!!config.enableJit && !flashMoeActive && !distributedActive && !dsv4Active && !zayaCcaActive && !turboQuantActive && !multimodalActive && !hybridCacheActive}
              onChange={e => onChange('enableJit', e.target.checked)}
              disabled={flashMoeActive || distributedActive || dsv4Active || zayaCcaActive || turboQuantActive || multimodalActive || hybridCacheActive}
              className="rounded border-input"
            />
            <span className="text-xs text-muted-foreground">
              Fuse Metal operations for faster inference (experimental)
            </span>
          </label>
        </Field>
        {(flashMoeActive || distributedActive || dsv4Active || zayaCcaActive || turboQuantActive || multimodalActive || hybridCacheActive) && (
          <IncompatWarning text={dsv4Active
            ? "JIT is disabled for DeepSeek-V4 native composite cache. DSV4 uses path-dependent SWA+CSA/HCA state that must stay on the uncompiled scheduler path."
            : zayaCcaActive
            ? "JIT is disabled for ZAYA typed CCA cache. CCA state is path-dependent and the full cache stack benchmarks faster on the uncompiled scheduler path."
            : hybridCacheActive
            ? "JIT is disabled for hybrid SSM/Mamba cache models. Their path-dependent Python cache objects are not mx.compile safe."
            : multimodalActive
            ? "JIT is disabled for multimodal/VLM models. The mlx-vlm streaming path owns image/video preprocessing and stream context state that is not safe to trace with mx.compile."
            : turboQuantActive
            ? "Server-level mx.compile is disabled for JANGTQ/TurboQuant KV because the live cache uses custom TurboQuant objects that mx.compile cannot trace. JANGTQ fused Metal kernels still run."
            : flashMoeActive
            ? "JIT is disabled while Flash MoE is on. Flash MoE's on-demand expert loading is incompatible with mx.compile tracing."
            : "JIT is disabled while distributed mode is on. Distributed orchestration cannot safely compile the local coordinator graph."} />
        )}

        {dsv4Active && (
          <>
            <InfoNote text="DeepSeek-V4 uses a native SWA+CSA/HCA composite cache. These controls map to DSV4-only engine env vars and require a restart." />
            <CheckField
              label="DSV4 Pool Quantization"
              tooltip="Enables the experimental native CSA/HCA pool codec with DSV4_POOL_QUANT=1. Keep off for production correctness; use only for local memory/restore experiments."
              checked={!!config.dsv4PoolQuant}
              onChange={v => onChange('dsv4PoolQuant', v)}
            />
          </>
        )}

        <SliderField
          label="Stream Interval"
          tooltip="Controls how often streaming tokens are sent to the client. A value of 1 sends each token immediately (smoothest streaming). Higher values batch multiple tokens together, which improves throughput but makes streaming feel chunkier. Set to 1 for chat use, higher for batch processing."
          value={config.streamInterval}
          onChange={v => onChange('streamInterval', v)}
          min={1}
          max={100}
          step={1}
          defaultValue={DEFAULT_CONFIG.streamInterval}
        />
        <SliderField
          label="Max Context Tokens"
          tooltip="Maximum prompt/context tokens accepted by this server before prefill. This maps to --max-prompt-tokens and rejects over-limit prompts with prompt_too_long. It does not trim history and does not cap generated output; per-chat/API max_tokens controls output length."
          value={config.maxContextLength}
          onChange={v => onChange('maxContextLength', v)}
          min={1}
          max={1000000}
          step={1024}
          defaultValue={detectedMaxContext && detectedMaxContext > 0 ? detectedMaxContext : DEFAULT_CONFIG.maxContextLength}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={detectedMaxContext && detectedMaxContext > 0 ? `Auto (${detectedMaxContext} model context)` : "Auto (memory-safe)"}
        />
        <InfoNote text={`Generation defaults are resolved by the engine from generation_config.json/jang_config when present${generationDefaultsSummary ? `. Current model-declared values: ${generationDefaultsSummary}` : ''}. The app does not synthesize missing sampling values; per-chat and API request parameters override model defaults.`} />
      </Section>

      {/* Tool Integration */}
      <Section title={t('sessions.config.toolIntegrationMCP')} expanded={expandedSections.tools} onToggle={() => toggleSection('tools')} hidden={isImage}>
        <PerformanceHint text="Lets the model call external tools (web search, code execution, etc.) during conversations. Requires a model that supports tool calling." />
        <Field label="MCP Config File" tooltip="Path to a JSON config file defining MCP (Model Context Protocol) tool servers. When configured, the model can call external tools during generation. The config file defines tool server endpoints, authentication, and available capabilities.">
          <div className="flex gap-2">
            <input type="text" value={config.mcpConfig} onChange={e => onChange('mcpConfig', e.target.value)} placeholder={t('sessions.config.mcpConfigPlaceholder')} className="cfg-input flex-1" />
            <button type="button" onClick={browseMcpConfig} className="px-3 py-1.5 rounded border border-border text-sm hover:bg-accent">Browse</button>
            <button type="button" onClick={importMcpConfig} className="px-3 py-1.5 rounded border border-border text-sm hover:bg-accent" disabled={mcpImportLoading}>
              {mcpImportLoading ? 'Importing' : 'Import'}
            </button>
            <button type="button" onClick={() => validateMcpConfig()} className="px-3 py-1.5 rounded border border-border text-sm hover:bg-accent" disabled={mcpValidationLoading}>
              {mcpValidationLoading ? 'Validating' : 'Validate'}
            </button>
          </div>
        </Field>
        {mcpValidation && (
          <div className="rounded border border-border/60 bg-background/60 px-2 py-1.5 text-xs">
            {mcpValidation.error ? (
              <span className="text-destructive">{mcpValidation.error}</span>
            ) : (
              <div className="space-y-1">
                <div className="text-muted-foreground">{mcpValidation.serverCount ?? mcpValidation.servers.length} configured MCP servers</div>
                {mcpValidation.servers.slice(0, 4).map(server => (
                  <div key={server.name} className="flex items-center justify-between gap-2">
                    <span className="font-medium">{server.name}</span>
                    <span className="text-muted-foreground">{server.transport || 'mcp'} · {server.enabled === false ? 'disabled' : 'enabled'}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        <Field label="Enabled MCP Servers" tooltip="Comma or newline separated server names from mcp.json. Empty means every configured server is eligible for this session.">
          <textarea value={config.mcpEnabledServers} onChange={e => onChange('mcpEnabledServers', e.target.value)} placeholder="filesystem,github" className="cfg-input" rows={2} />
        </Field>
        <Field label="Disabled MCP Servers" tooltip="Comma or newline separated server names to block even when they are present in mcp.json.">
          <textarea value={config.mcpDisabledServers} onChange={e => onChange('mcpDisabledServers', e.target.value)} placeholder="browser_automation&#10;postgres_readonly" className="cfg-input" rows={2} />
        </Field>
        <Field label="Enabled MCP Tools" tooltip="Comma or newline separated MCP tool names, usually server__tool. Empty means every server-eligible tool is eligible unless denied below.">
          <textarea value={config.mcpEnabledTools} onChange={e => onChange('mcpEnabledTools', e.target.value)} placeholder="filesystem__read_file&#10;github__search_repositories" className="cfg-input" rows={3} />
        </Field>
        <Field label="Disabled MCP Tools" tooltip="Comma or newline separated MCP tool names to block even if the model asks for them.">
          <textarea value={config.mcpDisabledTools} onChange={e => onChange('mcpDisabledTools', e.target.value)} placeholder="filesystem__write_file" className="cfg-input" rows={2} />
        </Field>
        {sessionId && (
          <div className="rounded border border-border bg-background/60 p-2 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-muted-foreground">Live MCP Status</span>
              <button type="button" onClick={refreshMcpStatus} className="text-xs px-2 py-1 rounded border border-border hover:bg-accent" disabled={mcpStatusLoading}>
                {mcpStatusLoading ? 'Refreshing' : 'Refresh'}
              </button>
            </div>
            {mcpStatus?.error && (
              <div className="text-xs text-destructive">{mcpStatus.error}</div>
            )}
            {(mcpStatus?.servers?.length || 0) > 0 && (
              <div className="space-y-1">
                <div className="text-[11px] text-muted-foreground">Servers</div>
                {(mcpStatus?.servers || []).map(server => {
                  const allowListActive = policyServers.length > 0
                  const checked = !policyDisabledServers.includes(server.name) && (allowListActive ? policyServers.includes(server.name) : server.enabled !== false)
                  return (
                    <label key={server.name} className="flex items-center justify-between gap-2 rounded border border-border/60 px-2 py-1 text-xs">
                      <span className="min-w-0">
                        <span className="font-medium">{server.name}</span>
                        <span className="ml-2 text-muted-foreground">{server.transport || 'mcp'} · {server.state || 'unknown'} · {server.tools_count ?? 0} tools</span>
                      </span>
                      <input type="checkbox" checked={checked} onChange={e => toggleMcpServer(server.name, e.target.checked)} />
                    </label>
                  )
                })}
              </div>
            )}
            {(mcpStatus?.tools?.length || 0) > 0 && (
              <div className="space-y-1">
                <div className="text-[11px] text-muted-foreground">Tools</div>
                {(mcpStatus?.tools || []).map(tool => (
                  <label key={tool.name} className="grid grid-cols-[1fr_auto] gap-2 rounded border border-border/60 px-2 py-1 text-xs">
                    <span className="min-w-0">
                      <span className="font-medium break-all">{tool.name}</span>
                      <span className={`ml-2 ${tool.effective ? 'text-primary' : 'text-muted-foreground'}`}>
                        {tool.effective ? 'effective' : 'blocked'}
                      </span>
                      {tool.description && <span className="block truncate text-muted-foreground">{tool.description}</span>}
                    </span>
                    <input type="checkbox" checked={tool.effective !== false && !policyDisabledTools.includes(tool.name)} onChange={e => toggleMcpTool(tool.name, e.target.checked)} />
                  </label>
                ))}
              </div>
            )}
          </div>
        )}
        <CheckField label="Enable Auto Tool Choice" tooltip="When enabled, the model automatically decides when to call tools based on the conversation context. Requires a model that supports tool calling (Qwen, Llama 3+, Mistral, Gemma 3, Phi-4, Hermes, DeepSeek, GLM, Granite, Kimi, xLAM, Functionary, MiniMax, StepFun). The model will format tool calls according to the selected parser. Leave unchecked for auto-detection (recommended)." checked={config.enableAutoToolChoice ?? false} onChange={v => onChange('enableAutoToolChoice', v || undefined)} />
        {config.enableAutoToolChoice === undefined && (
          <InfoNote text="Auto-detect: most models enable this automatically when a tool parser is detected." />
        )}
        <ParserField
          label="Tool Call Parser"
          tooltip="Specifies how to parse the model's tool call output. Each model family uses a different format (Qwen, Llama, Mistral, Hermes, DeepSeek, GLM, etc). 'Auto-detect' reads config.json to pick the right one. If auto-detection fails (e.g. GGUF, renamed fine-tunes), select the parser matching your model's base architecture. Click '?' to see format examples and supported models for each parser."
          value={config.toolCallParser}
          onChange={v => onChange('toolCallParser', v)}
          options={TOOL_PARSER_OPTIONS}
        />
        <ParserField
          label="Reasoning Parser"
          tooltip="Separates reasoning/thinking from final content. Use Auto-detect unless it picks wrong. Qwen3: Qwen, QwQ, MiniMax, StepFun (strict <think> tags). DeepSeek R1: DeepSeek-R1, GLM-4.7, Phi-4, Nemotron (lenient <think> tags). GPT-OSS: GLM-4.7 Flash (Harmony protocol). Mistral 4: Mistral Small/Large 4 ([THINK] tags). Click '?' for full model list."
          value={config.reasoningParser}
          onChange={v => onChange('reasoningParser', v)}
          options={REASONING_PARSER_OPTIONS}
        />
        <Field label="Custom Chat Template" tooltip="Override the model's built-in Jinja2 chat template. Useful when the default template is incompatible with your client (e.g., JetBrains AI Chat). Leave empty to use the model's built-in template. The template receives 'messages' and 'add_generation_prompt' variables.">
          <textarea
            value={config.chatTemplate ?? ''}
            onChange={e => onChange('chatTemplate', e.target.value || undefined)}
            placeholder={t('sessions.config.chatTemplatePlaceholder')}
            rows={3}
            className="cfg-input font-mono text-xs"
            style={{ resize: 'vertical', minHeight: '3rem' }}
          />
        </Field>
        <SelectField
          label="Multimodal Support (VLM)"
          tooltip="Vision-Language Model mode for models like Qwen2-VL, Qwen3-VL, Pixtral, InternVL, or LLaVA. Auto-detected VLMs launch with the MLLM scheduler even if an older saved session says off. Smelt and documented unsafe runtimes use text-only loading."
          value={dsv4Active || smeltActive || detectedForceTextOnly ? 'off' : config.isMultimodal === true ? 'on' : config.isMultimodal === false ? 'off' : 'auto'}
          onChange={v => onChange('isMultimodal', v === 'on' ? true : v === 'off' ? false : undefined)}
          options={[
            { value: 'auto', label: 'Auto (detect from model)' },
            { value: 'on', label: 'Force On' },
            { value: 'off', label: 'Force Off' },
          ]}
          disabled={dsv4Active || smeltActive || detectedForceTextOnly}
        />
        {dsv4Active && (
          <InfoNote text="DSV4 Flash is served through the text runtime. Image/video controls stay hidden because this bundle has no VL processor path." />
        )}
        {smeltActive && (
          <IncompatWarning text="VLM is disabled when Smelt Mode is active. Smelt uses text-only loading for partial expert support." />
        )}
        {detectedForceTextOnly && (
          <IncompatWarning text="This model has media metadata, but vMLX is using the text runtime because its current VLM language path is not production-safe. Use an MXFP4 or JANGTQ/MXTQ variant for image/video input." />
        )}
        {!dsv4Active && !smeltActive && !detectedForceTextOnly && config.isMultimodal === true && (
          <InfoNote text="VLM mode is active — the MLLM scheduler handles image/video processing with full prefix cache, paged KV cache, and KV quantization support." />
        )}
        {!dsv4Active && !smeltActive && !detectedForceTextOnly && config.isMultimodal === false && (
          <InfoNote text="VLM mode is off only when the model is not auto-detected as multimodal. Detected VLM bundles launch with image/video support." />
        )}
        {/* Video sampling — only relevant for VL models that accept video_url.
            Qwen 3.6 / Qwen3.5-VL both have native video understanding via
            temporal position embeddings, so 2 fps × 8 frames is typical. */}
        {showVideoControls && (
          <>
            <SliderField
              label="Video Frames/Second"
              tooltip="For VL models with video support (Qwen 3.6, Qwen3.5-VL). Controls how many frames per second are sampled from an uploaded video clip. Lower = fewer frames = faster prefill but less temporal detail. Qwen 3.6's temporal embeddings tolerate up to ~4 fps; 2 fps is a good default."
              value={config.videoFps ?? 2}
              onChange={v => onChange('videoFps', v)}
              min={1}
              max={8}
              step={1}
              defaultValue={2}
            />
            <SliderField
              label="Max Video Frames"
              tooltip="Maximum number of frames extracted from a single video, regardless of fps or duration. Caps prefill cost on long clips. Qwen 3.6 supports up to 32+ frames but most prompts work well with 8."
              value={config.videoMaxFrames ?? 8}
              onChange={v => onChange('videoMaxFrames', v)}
              min={2}
              max={64}
              step={2}
              defaultValue={8}
            />
          </>
        )}
      </Section>

      {/* Native in-model MTP */}
      <Section title="Native MTP" expanded={expandedSections.nativeMtp} onToggle={() => toggleSection('nativeMtp')} hidden={isImage || dsv4Active || !nativeMtpSupported}>
        <PerformanceHint text="Uses the model's own preserved MTP heads. Current Qwen3.6 runtime is deterministic and uses measured model-local depth when present, with D3 as the generic fallback." />
        {nativeMtpMode === 'auto' && (
          <InfoNote text="Auto mode only activates MTP for API/chat requests that already use deterministic sampling. Sampled requests fall back to autoregressive decode and the server logs the reason." />
        )}
        {nativeMtpMode === 'deterministic' && (
          <InfoNote text={`Default mode applies D${nativeMtpDepth} and deterministic startup sampling so normal app chats actually enter the native MTP path. Explicit API sampling parameters still win per request.`} />
        )}
        <SelectField
          label="Native MTP Mode"
          tooltip="Deterministic mode uses the bundle's measured MTP depth when available, otherwise D3. Auto leaves sampling defaults alone and only uses MTP when a request is already compatible. Off disables the in-model MTP runtime."
          value={nativeMtpMode}
          onChange={v => onChange('nativeMtpMode', v as 'deterministic' | 'auto' | 'off')}
          options={[
            { value: 'deterministic', label: 'Deterministic tuned default' },
            { value: 'auto', label: 'Auto gate only' },
            { value: 'off', label: 'Off' },
          ]}
        />
        <SliderField
          label="Native MTP Depth"
          tooltip="Number of tokens drafted per native-MTP verification cycle. Model-local tuning picks the measured default; changing this slider creates a manual override."
          value={nativeMtpDepth}
          onChange={v => {
            onChange('nativeMtpDepth', v)
            onChange('nativeMtpDepthOverride', true)
          }}
          min={1}
          max={3}
          step={1}
          defaultValue={3}
          disabled={nativeMtpMode === 'off'}
        />
        <InfoNote text={`Detected scope: ${detectedNativeMtp?.runtimeScope || 'text'}; depth source: ${detectedNativeMtp?.depthSource || 'default'}. Hybrid cache bundles use paged cache while prefix cache is enabled so KV blocks and SSM state stay in one cache contract.`} />
      </Section>

      {/* Speculative Decoding */}
      <Section title={t('sessions.config.specDecoding')} expanded={expandedSections.specDecode} onToggle={() => toggleSection('specDecode')} hidden={isImage || dsv4Active}>
        <PerformanceHint text="Use a small draft model to propose tokens, then verify them in a single target model pass. Can give 20-90% speedup with zero quality loss." />
        {config.continuousBatching && <IncompatWarning text="Speculative decoding is incompatible with continuous batching. The draft model is omitted at launch while the cache-stack scheduler is active." />}
        {multimodalActive && <IncompatWarning text="Speculative decoding is incompatible with multimodal (VLM) models. The draft model is omitted at launch for VLM requests." />}
        <Field label="Draft Model" tooltip="Path or HuggingFace name of a small draft model. Must use the same tokenizer as the main model. Example: mlx-community/Llama-3.2-1B-Instruct-4bit for a Llama 3 target model. Leave empty to disable speculative decoding.">
          <input type="text" value={config.speculativeModel} onChange={e => onChange('speculativeModel', e.target.value)} placeholder={t('sessions.config.specModelPlaceholder')} className="cfg-input" disabled={config.continuousBatching || multimodalActive || dsv4Active} />
        </Field>
        {config.speculativeModel && (
          <SliderField
            label="Draft Tokens per Step"
            tooltip="Number of tokens the draft model proposes per speculative decoding step. Higher values = more potential speedup but lower acceptance rate. Sweet spot is typically 2-5."
            value={config.numDraftTokens}
            onChange={v => onChange('numDraftTokens', v)}
            min={1}
            max={20}
            step={1}
            defaultValue={DEFAULT_CONFIG.numDraftTokens}
            disabled={config.continuousBatching || multimodalActive || dsv4Active}
          />
        )}
      </Section>

      {/* Distributed Compute */}
      <Section title={t('sessions.config.distributed')} expanded={expandedSections.distributed} onToggle={() => toggleSection('distributed')} hidden={isImage || dsv4Active}>
        <div className="mx-4 mt-3 mb-2 rounded-md border-2 border-amber-500 bg-amber-500/15 px-3 py-3 text-xs text-amber-800 dark:text-amber-100">
          <div className="font-bold uppercase tracking-wide text-[11px] mb-1.5 text-amber-900 dark:text-amber-50">
            ⚠ Pre-Alpha — localhost loopback only
          </div>
          <div className="leading-relaxed text-amber-900/90 dark:text-amber-100/90 space-y-1.5">
            <p>
              <strong>This feature is under active development and is not
              safe to expose on any network you don't fully control.</strong>
            </p>
            <p>
              Known gaps: cluster secret is sent plaintext over the wire (no
              TLS, no HMAC); worker crash recovery is not implemented;
              coordinator-loss re-election recovery is a stub; protocol has
              no version handshake; tensor parallelism is stubbed.
            </p>
            <p>
              Recommended usage today: run one <code>vmlx-worker</code> on
              the same Mac you're running the coordinator on (different port),
              bound to <code>127.0.0.1</code>, as a smoke test. Real multi-Mac
              deployment is blocked behind Phase 2 hardening. See
              <code>docs/guides/distributed-setup.md</code>.
            </p>
          </div>
        </div>
        <PerformanceHint text="Pipeline parallelism splits transformer layers across nodes. Each request passes hidden states over a TCP connection between workers. In localhost loopback testing, the overhead is dominated by the loopback memcpy — useful for verifying correctness, not performance." />
        <CheckField
          label="Enable Distributed Inference"
          tooltip="Split the model across multiple Macs. Requires vmlx-worker running on each additional Mac. The coordinator (this Mac) handles tokenization, embedding, and final projection."
          checked={!!config.distributedEnabled}
          onChange={v => {
            onChange('distributedEnabled', v)
            // Mutual exclusion: disable Flash MoE and JIT if enabling distributed
            if (v && flashMoeActive) onChange('flashMoe', false)
            if (v && config.enableJit) onChange('enableJit', false)
          }}
          disabled={flashMoeActive}
        />
        {flashMoeActive && (
          <IncompatWarning text="Distributed is disabled while Flash MoE is on. Flash MoE patches local model layers — distributed workers have their own model copies." />
        )}
        {config.distributedEnabled && (
          <>
            <SelectField
              label="Parallelism Mode"
              tooltip="Pipeline: split layers across nodes (simple, works with any network). Tensor: split weights within layers (requires high bandwidth, 10GbE+ recommended)."
              value={config.distributedMode || 'pipeline'}
              onChange={v => onChange('distributedMode', v as 'pipeline' | 'tensor')}
              options={[
                { value: 'pipeline', label: 'Pipeline Parallelism (split layers)' },
                { value: 'tensor', label: 'Tensor Parallelism (split weights) — coming soon' },
              ]}
            />
            {config.distributedMode === 'tensor' && (
              <IncompatWarning text="Tensor parallelism is not yet implemented. Use pipeline parallelism for now." />
            )}
            <Field label="Cluster Secret" tooltip="Shared secret for authenticating worker nodes. All workers must use the same secret. Leave empty for no authentication (only safe on trusted networks).">
              <input
                type="password"
                value={config.distributedSecret || ''}
                onChange={e => onChange('distributedSecret', e.target.value)}
                placeholder={t('sessions.config.clusterSecretPlaceholder')}
                className="cfg-input"
              />
            </Field>
            <InfoNote text="Worker nodes: Install vMLX on each Mac and run 'vmlx-worker --secret YOUR_SECRET' from Terminal. Workers auto-advertise via Bonjour — the coordinator discovers them automatically." />
            <DistributedNodeList enabled={!!config.distributedEnabled} sessionId={sessionId} />
            <div className="px-4 py-3 space-y-2">
              <div className="text-xs font-medium text-foreground">Setup Guide</div>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>1. Connect Macs via <strong>Thunderbolt 5 cable</strong> (fastest) or Ethernet/WiFi</p>
                <p>2. On each worker Mac: <code className="bg-muted px-1 rounded">pip install vmlx && vmlx-worker --secret YOUR_SECRET</code></p>
                <p>3. Workers appear automatically above via Bonjour discovery</p>
                <p>4. Or click "Add Manual" to add by IP if discovery doesn't find them</p>
                <p className="text-muted-foreground/70 pt-1">Thunderbolt 5: ~120 Gbps, 0.1ms latency (best). 1GbE: works fine for pipeline parallelism. WiFi: works but slower. Any network that can ping the other Mac will work.</p>
              </div>
            </div>
          </>
        )}
      </Section>

      {/* Embedding Model */}
      {!isImage && (
      <div className="mb-2">
        <Field label={t('sessions.config.embeddingModel')} tooltip="Pre-load a separate embedding model at startup for the /v1/embeddings endpoint. Runs alongside the main chat model. Example: mlx-community/embeddinggemma-300m-6bit. Leave empty to disable embeddings endpoint.">
          <input type="text" value={config.embeddingModel} onChange={e => onChange('embeddingModel', e.target.value)} placeholder={t('sessions.config.embeddingPlaceholder')} className="cfg-input" />
        </Field>
      </div>
      )}

      {/* Additional */}
      <div className="mb-4">
        <Field label={t('sessions.config.additionalArgs')} tooltip="Raw command-line arguments appended to the serve command. Use this for flags not exposed in the UI above. Example: --log-level DEBUG. Arguments are split by whitespace and passed directly to the CLI.">
          <input type="text" value={config.additionalArgs} onChange={e => onChange('additionalArgs', e.target.value)} placeholder={t('sessions.config.additionalArgsPlaceholder')} className="cfg-input" />
        </Field>
      </div>

      {/* Reset to Defaults */}
      {onReset && (
        <div className="pt-2 pb-1 border-t border-border">
          <button
            onClick={onReset}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Reset all parameters to defaults
          </button>
        </div>
      )}
    </div>
  )
}

// ─── Shared Helper Components ─────────────────────────────────────────────────

export function Tooltip({ text }: { text: string }) {
  const [show, setShow] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [above, setAbove] = useState(true)
  const [hAnchor, setHAnchor] = useState<'center' | 'left' | 'right'>('center')
  const triggerRef = useRef<HTMLSpanElement>(null)

  const updatePosition = () => {
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect()
      setAbove(rect.top > 130)
      // Horizontal: tooltip is w-72 (288px). Need ~144px of clearance on each
      // side of the trigger for centered layout. If not enough room on one
      // side, anchor to that side so the tooltip extends toward the other.
      const vw = window.innerWidth
      const triggerCenter = rect.left + rect.width / 2
      const half = 150 // 288/2 + small buffer
      if (triggerCenter - half < 8) {
        setHAnchor('left')          // anchor to left of trigger, extends right
      } else if (triggerCenter + half > vw - 8) {
        setHAnchor('right')         // anchor to right of trigger, extends left
      } else {
        setHAnchor('center')
      }
    }
  }

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    updatePosition()
    const willPin = !pinned
    setPinned(willPin)
    setShow(willPin)
  }

  const handleEnter = () => {
    if (!pinned) {
      updatePosition()
      setShow(true)
    }
  }

  const handleLeave = () => {
    if (!pinned) setShow(false)
  }

  return (
    <span className="relative inline-flex ml-1">
      <span
        ref={triggerRef}
        className={`inline-flex items-center justify-center w-3.5 h-3.5 rounded-full text-[10px] font-bold cursor-help select-none ${pinned ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}
        onClick={handleClick}
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
      >
        ?
      </span>
      {show && (
        <div
          className={`absolute w-72 max-w-[calc(100vw-1rem)] p-2.5 bg-popover text-popover-foreground text-xs rounded-lg shadow-lg border border-border z-50 leading-relaxed ${
            above ? 'bottom-full mb-2' : 'top-full mt-2'
          } ${
            hAnchor === 'left' ? 'left-0'
              : hAnchor === 'right' ? 'right-0'
              : 'left-1/2 -translate-x-1/2'
          }`}
        >
          {text}
          <div className={`absolute border-4 border-transparent ${
            above ? 'top-full -mt-px border-t-border' : 'bottom-full -mb-px border-b-border'
          } ${
            hAnchor === 'left' ? 'left-2'
              : hAnchor === 'right' ? 'right-2'
              : 'left-1/2 -translate-x-1/2'
          }`} />
        </div>
      )}
    </span>
  )
}

// ─── Parser Options with Format Examples ──────────────────────────────────────

interface ParserOption {
  value: string
  label: string
  format?: string  // Example of the format for tooltip
  models?: string[]  // Specific models that use this parser (shown in help panel)
}

const TOOL_PARSER_OPTIONS: ParserOption[] = [
  { value: 'auto', label: 'Auto-detect (recommended)' },
  { value: '', label: 'None (disable tool parsing)' },
  {
    value: 'qwen', label: 'Qwen — Qwen3.5 / Qwen3 / Qwen2.5 / QwQ', format: '<tool_call>{"name":"fn","arguments":{...}}</tool_call>', models: [
      'Qwen3.5-VL (0.8B\u2013122B MoE, native vision)', 'Qwen3 (0.6B\u2013235B)', 'Qwen3-Coder',
      'Qwen3-MoE (22B/57B)', 'Qwen3-VL (2B/32B/72B)', 'QwQ-32B',
      'Qwen2.5 (0.5B\u201372B)', 'Qwen2.5-Coder (0.5B\u201332B)',
      'Qwen2.5-VL (3B\u201372B)', 'Qwen2 (0.5B\u201372B)', 'Qwen2-VL (2B\u201372B)',
    ]
  },
  {
    value: 'llama', label: 'Llama — Llama 4 / 3.x / Yi', format: '<function=name>{"arg":"val"}</function>', models: [
      'Llama 4 Scout (17Bx16E MoE)', 'Llama 4 Maverick (17Bx128E MoE)',
      'Llama 3.3 (70B)', 'Llama 3.2 (1B/3B/11B/90B)', 'Llama 3.1 (8B/70B/405B)', 'Llama 3 (8B/70B)',
      'Yi / Yi-1.5 (Llama architecture)',
    ]
  },
  {
    value: 'mistral', label: 'Mistral — Mistral / Mixtral / Pixtral / Codestral', format: '[TOOL_CALLS][{"name":"fn","arguments":{...}}]', models: [
      'Mistral Large (123B)', 'Mistral Small 3.1 (24B)', 'Mistral Nemo (12B)', 'Mistral 7B v0.3',
      'Mixtral 8x7B / 8x22B', 'Pixtral 12B / Pixtral Large', 'Codestral (22B)', 'Devstral Small (24B)',
    ]
  },
  {
    value: 'hermes', label: 'Hermes — Phi-4 / Hermes fine-tunes', format: '<tool_call>{"name":"fn","arguments":{...}}</tool_call>', models: [
      'Phi-4 Mini (3.8B)', 'Phi-4 Medium (14B)',
      'Phi-4 Reasoning (14B)', 'Hermes 2 / 3 / 4', 'Any Hermes-format fine-tune',
    ]
  },
  {
    value: 'gemma3', label: 'Gemma 3 / 3n — Google tool_code', format: '```tool_code\nfn(arg="val")\n```', models: [
      'Gemma 3 (1B/4B/12B/27B)',
      'Gemma 3n (E2B/E4B)',
      'Use this for model_type=gemma3/gemma3n; do not use Hermes for Google tool_code bundles',
    ]
  },
  {
    value: 'deepseek', label: 'DeepSeek / GLM5 / Ling — DeepSeek-style tools', format: '\u{ff5c}<tool_call>name\n{"arg":"val"}</tool_call>\u{ff5c}', models: [
      'DeepSeek-V3 (671B MoE)', 'DeepSeek-V2.5 (236B MoE)', 'DeepSeek-V2 (236B MoE)',
      'DeepSeek-R1 (671B native)', 'DeepSeek-Coder-V2 (236B)',
      'GLM-5.1 / GLM MoE DSA', 'Ling / Bailing hybrid',
      '\u26A0 R1-Distill-Qwen/Llama use qwen/llama parsers',
    ]
  },
  {
    value: 'deepseek_v4', label: 'DeepSeek V4 / DSV4-Flash — DSML', format: '<｜DSML｜invoke name="fn"><｜DSML｜parameter name="arg" string="true">val</｜DSML｜parameter></｜DSML｜invoke>', models: [
      'DeepSeek-V4-Flash / DSV4-Flash JANG, JANGTQ, and DQ bundles',
      'Use this for deepseek_v4 model_type; DeepSeek V3/R1 use the DeepSeek parser above',
    ]
  },
  {
    value: 'hy_v3', label: 'Hy3 / Hunyuan — Tencent XML tools', format: '<tool_calls><tool_call>fn<tool_sep><arg_key>arg</arg_key><arg_value>val</arg_value></tool_call></tool_calls>', models: [
      'Hy3-preview / Hunyuan model_type=hy_v3 bundles',
      'Hunyuan/Tencent XML tool-call contract',
    ]
  },
  {
    value: 'zaya_xml', label: 'ZAYA / Zyphra — XML tools', format: '<function=fn>{"arg":"val"}</function>', models: [
      'ZAYA1 / ZAYA1-VL JANGTQ and MXFP bundles',
      'Zyphra XML tool-call contract',
    ]
  },
  {
    value: 'nemotron', label: 'Nemotron — Nemotron / Qwen3-Next', format: '<tool_call><function=fn><parameter=p>val</parameter></function></tool_call>', models: [
      'Nemotron-H (8B/47B/56B)', 'Nemotron-4 Nano/Super/Ultra',
      'Qwen3-Next / Qwen3-Coder-Next (hybrid Mamba)',
      '\u26A0 Llama/Qwen fine-tunes named "Nemotron" use their base parser',
    ]
  },
  {
    value: 'glm47', label: 'GLM / GPT-OSS — GLM-4 / GLM-4.7 / GLM-Z1', format: '<tool_call>name\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>', models: [
      'GLM-4 (9B)', 'GLM-4.7 (9B)', 'GLM-4.7 Flash (9B MoE)', 'GLM-Z1 (32B)', 'GPT-OSS-20B/120B',
    ]
  },
  {
    value: 'granite', label: 'Granite — IBM Granite 3.x / Granite-Code', format: '<|tool_call|>[{"name":"fn","arguments":{...}}]', models: [
      'Granite 3.0/3.1/3.2/3.3 (2B/8B)', 'Granite-Code (3B/8B/20B/34B)',
    ]
  },
  {
    value: 'functionary', label: 'Functionary — MeetKai Functionary v2/v3/v4r', format: '<|from|>assistant\n<|recipient|>fn\n<|content|>{"arg":"val"}', models: [
      'Functionary v2 (7B)', 'Functionary v3 (8B/70B)', 'Functionary v4r (8B)',
    ]
  },
  {
    value: 'minimax', label: 'MiniMax — MiniMax-M1 / M2 / M2.5', format: '<minimax:tool_call><invoke name="fn"><parameter name="arg">val</parameter></invoke></minimax:tool_call>', models: [
      'MiniMax-M1 (40B MoE)', 'MiniMax-M2 (230B MoE)', 'MiniMax-M2.5 (230B MoE)',
    ]
  },
  {
    value: 'xlam', label: 'xLAM — Salesforce xLAM-v2 series', format: '[{"name":"fn","arguments":{...}}]', models: [
      'xLAM-1B', 'xLAM-7B', 'xLAM-v2 (8x7B/8x22B)',
    ]
  },
  {
    value: 'kimi', label: 'Kimi — Kimi-K2/K2.5/K2.6 / Moonshot', format: '<|tool_calls_section_begin|><|tool_call_begin|>fn<|tool_call_argument_begin|>{...}<|tool_call_end|>', models: [
      'Kimi-K2 (1T MoE)', 'Kimi-K2.5 / kimi_k25', 'Kimi-K2.6 VL', 'Moonshot-v1',
    ]
  },
  {
    value: 'step3p5', label: 'StepFun — Step-3.5 Flash / Step-3.5', format: '<tool_call><function=fn><parameter=arg>val</parameter></function></tool_call>', models: [
      'Step-3.5 Flash (8B MoE)', 'Step-3.5',
    ]
  },
  {
    value: 'gemma4', label: 'Gemma 4 — Google Gemma 4', format: '<|tool_call>call:fn{key:value}<tool_call|>', models: [
      'Gemma 4 27B-A4B (text+vision, MoE)',
      'Gemma 4 31B (text+vision, dense)',
    ]
  },
]

const REASONING_PARSER_OPTIONS: ParserOption[] = [
  { value: 'auto', label: 'Auto-detect (recommended)' },
  { value: '', label: 'None (disable reasoning extraction)' },
  {
    value: 'qwen3', label: 'Qwen3 — Qwen / QwQ / MiniMax / StepFun', format: '<think>...reasoning...</think>content  (strict: both tags required)', models: [
      'Qwen3.5-VL (0.8B\u2013122B MoE, vision+reasoning)', 'Qwen3 (0.6B\u2013235B, all sizes)',
      'Qwen3-Coder (all sizes)', 'Qwen3-MoE (22B/57B)', 'QwQ-32B',
      'MiniMax-M2 (46B)', 'MiniMax-M2.5 (172B MoE)',
      'MiniMax Prism Pro (80B)', 'StepFun Step-3.5 Flash (8B MoE)',
      'StepFun Step-3.5', 'StepFun Step-1V (vision)',
    ]
  },
  {
    value: 'deepseek_r1', label: 'DeepSeek R1 — DeepSeek / Gemma / GLM / Phi / Nemotron', format: '<think>...reasoning...</think>content  (lenient: handles missing <think>)', models: [
      'DeepSeek-R1 (671B native)', 'DeepSeek-R1-0528',
      'GLM-4.7 (9B) \u2014 NOT GLM-4.7 Flash', 'GLM-Z1 (32B)',
      'Phi-4 Reasoning / Reasoning Plus (14B)',
      'Nemotron (hybrid Mamba+attention)',
      '\u26A0 R1-Distill-Qwen/Llama: must select manually (auto-detect has no reasoning)',
    ]
  },
  {
    value: 'openai_gptoss', label: 'GPT-OSS / Harmony — GLM-4.7 Flash / GPT-OSS', format: '<|channel|>analysis<|message|>reasoning...<|channel|>final<|message|>content', models: [
      'GLM-4.7 Flash (9B MoE) \u2014 uses Harmony, NOT deepseek_r1',
      'GPT-OSS-20B', 'GPT-OSS-120B',
    ]
  },
  {
    value: 'mistral', label: 'Mistral 4 — Mistral Small/Large 4', format: '[THINK]...reasoning...[/THINK]content', models: [
      'Mistral Small 4 (24B/119B MoE, text+vision)',
      'Mistral Large 4 (text+vision)',
      'Any Mistral model with [THINK]/[/THINK] reasoning tokens',
    ]
  },
  {
    value: 'gemma4', label: 'Gemma 4 — Google Gemma 4', format: '<|channel>thought...reasoning...<channel|>content', models: [
      'Gemma 4 27B-A4B (text+vision, MoE)',
      'Gemma 4 31B (text+vision, dense)',
      'Any Gemma 4 model with <|channel>thought protocol',
    ]
  },
]

function ParserField({ label, tooltip, value, onChange, options }: {
  label: string; tooltip: string; value: string; onChange: (v: string) => void; options: ParserOption[]
}) {
  const { t } = useTranslation()
  const [showHelp, setShowHelp] = useState(false)
  const selected = options.find(o => o.value === value)
  // Show help panel when explicitly toggled OR when a non-auto parser is manually selected
  const helpVisible = showHelp || (value !== 'auto' && value !== '')

  return (
    <div className="block">
      <span className="text-xs font-medium text-muted-foreground">
        {label}
        <Tooltip text={tooltip} />
        <button
          type="button"
          onClick={() => setShowHelp(!showHelp)}
          className="ml-1 inline-flex items-center justify-center w-3.5 h-3.5 rounded-full bg-muted text-muted-foreground text-[10px] font-bold cursor-help select-none hover:bg-accent"
          title={t('sessions.config.modelCompatTitle')}
        >
          ?
        </button>
      </span>
      <select value={value} onChange={e => onChange(e.target.value)} className="cfg-input">
        {options.map(o => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      {helpVisible && (
        <div className="mt-1.5 bg-background border border-border rounded p-2 text-xs max-h-48 overflow-auto space-y-2">
          {options.filter(o => o.format).map(o => {
            const isSelected = o.value === value
            return (
              <div key={o.value} className={`pl-1.5 border-l-2 ${isSelected ? 'border-primary' : 'border-transparent'}`}>
                <div className={`font-medium leading-snug ${isSelected ? 'text-primary' : 'text-foreground'}`}>
                  {o.label}
                </div>
                <code className="block mt-0.5 text-[10px] bg-muted text-muted-foreground px-1.5 py-0.5 rounded break-all leading-snug">
                  {o.format}
                </code>
                {o.models && o.models.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {o.models.map((m, i) => (
                      <span key={i} className={`inline-block text-[10px] px-1.5 py-px rounded-sm leading-tight ${m.startsWith('\u26A0') ? 'bg-warning/15 text-warning border border-warning/30' : 'bg-muted text-muted-foreground'
                        }`}>{m}</span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
          <div className="pt-1 border-t border-border text-[10px] text-muted-foreground/70 italic leading-snug">
            Fine-tunes inherit the base model&apos;s parser. A Llama fine-tune uses llama, a Qwen fine-tune uses qwen, regardless of its marketing name. When auto-detect fails, select the parser matching the base architecture.
          </div>
        </div>
      )}
      {selected?.format && !helpVisible && (
        <p className="text-[10px] text-muted-foreground mt-0.5 font-mono truncate" title={selected.format}>
          {selected.format}
        </p>
      )}
    </div>
  )
}

function IncompatWarning({ text }: { text: string }) {
  return (
    <div className="px-2 py-1.5 mb-1 rounded text-[11px] bg-warning/10 border border-warning/30 text-warning leading-tight">
      {text}
    </div>
  )
}

function InfoNote({ text }: { text: string }) {
  return (
    <div className="px-2 py-1.5 mb-1 rounded text-[11px] bg-primary/10 border border-primary/30 text-primary leading-tight">
      {text}
    </div>
  )
}

function PerformanceHint({ text }: { text: string }) {
  return (
    <div className="px-2 py-1.5 mb-1 rounded text-[11px] text-muted-foreground/70 italic leading-tight">
      {text}
    </div>
  )
}

export function Section({ title, expanded, onToggle, children, hidden }: {
  title: string; expanded: boolean; onToggle: () => void; children: React.ReactNode; hidden?: boolean
}) {
  if (hidden) return null
  return (
    <div className="mb-3 border border-border rounded">
      <button onClick={onToggle} className="w-full flex items-center gap-2 px-3 py-2 text-sm font-medium hover:bg-accent rounded-t">
        <span className={`transition-transform ${expanded ? 'rotate-90' : ''}`}>&#9654;</span>
        {title}
      </button>
      {expanded && <div className="px-3 pb-3 space-y-3">{children}</div>}
    </div>
  )
}

export function Field({ label, tooltip, children }: { label: string; tooltip?: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs font-medium text-muted-foreground">
        {label}
        {tooltip && <Tooltip text={tooltip} />}
      </span>
      {children}
    </label>
  )
}

export function CheckField({ label, tooltip, checked, onChange, disabled }: {
  label: string; tooltip?: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean
}) {
  return (
    <label className={`flex items-center gap-2 ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
      <input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)} disabled={disabled} />
      <span className="text-sm">{label}</span>
      {tooltip && <Tooltip text={tooltip} />}
    </label>
  )
}

export function SelectField({ label, tooltip, value, onChange, options, disabled }: {
  label: string; tooltip?: string; value: string; onChange: (v: string) => void
  options: { value: string; label: string }[]; disabled?: boolean
}) {
  return (
    <Field label={label} tooltip={tooltip}>
      <select value={value} onChange={e => onChange(e.target.value)} disabled={disabled} className="cfg-input">
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </Field>
  )
}

interface SliderFieldProps {
  label: string
  tooltip?: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  defaultValue: number
  allowUnlimited?: boolean
  unlimitedValue?: number
  unlimitedLabel?: string
  disabled?: boolean
  /** Hard upper limit for number input (prevents server crash from out-of-range values) */
  maxInput?: number
}

export function SliderField({
  label, tooltip, value, onChange, min, max, step, defaultValue,
  allowUnlimited = false, unlimitedValue = 0, unlimitedLabel = 'Unlimited',
  disabled = false, maxInput
}: SliderFieldProps) {
  const isUnlimited = allowUnlimited && value === unlimitedValue
  // Local string state for the number input so typing isn't clamped mid-keystroke.
  // Without this, min=1024 causes typing "1" to immediately snap to 1024.
  const [localInput, setLocalInput] = useState<string | null>(null)

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange(Number(e.target.value))
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    // Store raw typed value locally — no clamping until blur
    setLocalInput(e.target.value)
  }

  const handleInputFocus = () => {
    // Initialize local state with current value when focus starts
    setLocalInput(isUnlimited ? '' : String(value))
  }

  const handleInputBlur = () => {
    const raw = localInput ?? ''
    setLocalInput(null)
    if (raw === '') {
      onChange(isUnlimited ? unlimitedValue : defaultValue)
      return
    }
    const num = Math.round(Number(raw))
    if (isNaN(num)) {
      onChange(defaultValue)
    } else {
      // Clamp to valid range — maxInput enforces hard server-side limits
      const clamped = maxInput != null ? Math.min(maxInput, Math.max(min, num)) : Math.max(min, num)
      onChange(clamped)
    }
  }

  const toggleUnlimited = () => {
    if (isUnlimited) {
      onChange(defaultValue)
    } else {
      onChange(unlimitedValue)
    }
  }

  // Clamp slider display value to range (for when input allows beyond max)
  const sliderValue = isUnlimited ? min : Math.min(Math.max(value, min), max)
  // Show local input while editing, parent value otherwise
  const displayValue = localInput !== null ? localInput : (isUnlimited ? '' : value)

  return (
    <div className={`block ${disabled ? 'opacity-50 pointer-events-none' : ''}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">
          {label}
          {tooltip && <Tooltip text={tooltip} />}
        </span>
        {allowUnlimited && (
          <button
            type="button"
            onClick={toggleUnlimited}
            disabled={disabled}
            className={`text-[10px] px-1.5 py-0.5 rounded border transition-colors ${isUnlimited
              ? 'bg-primary/15 border-primary/40 text-primary'
              : 'border-border text-muted-foreground hover:text-foreground hover:border-foreground/30'
              }`}
          >
            {unlimitedLabel}
          </button>
        )}
      </div>
      <div className="flex items-center gap-2 mt-1">
        <input
          type="range"
          className="cfg-slider flex-1"
          min={min}
          max={max}
          step={step}
          value={sliderValue}
          onChange={handleSliderChange}
          disabled={disabled || isUnlimited}
        />
        <input
          type="number"
          className="w-20 px-2 py-1 bg-background border border-input rounded text-sm text-right tabular-nums"
          value={displayValue}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onBlur={handleInputBlur}
          placeholder={isUnlimited ? unlimitedLabel : undefined}
          disabled={disabled}
          min={min}
          step={step}
        />
      </div>
    </div>
  )
}
