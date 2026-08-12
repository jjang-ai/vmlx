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
import {
  pagedCacheCapacityText,
  pagedCacheControlsState,
  pagedCacheMemoryIgnoredText,
} from '../../../../shared/cacheCapacityDisplay'
import { metalWiredLimitHelpText } from '../../../../shared/metalWiredLimit'
import { isLagunaMixedSwaTurboQuantEffective } from '../../../../shared/lagunaCachePolicy'
import { normalizeMcpPolicyList } from '../../../../shared/mcpPolicy'
import { canonicalizeToolParserId } from '../../../../shared/toolParserAliases'
import { shouldWarnDsv4TopP } from '../../../../shared/samplingParameterDomain'
import { resolveEffectiveModelFamily } from '../../../../shared/dsv4Env'
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
  // Manual model-family override. undefined = autodetect (default). When set
  // to a registry family name, sessions.ts emits --model-family to the engine.
  modelFamily?: string
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
  defaultDoSample?: boolean
  defaultSamplingDefaultsDeclared?: boolean
  defaultEnableThinking?: boolean
  dsv4PrefixCache?: boolean
  dsv4PoolQuant?: boolean
  dsv4ActivationQat?: boolean
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
  imageTokenBudget?: number
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
  usePagedCache: false,
  pagedCacheBlockSize: 64,
  // 4097 blocks x 64 tokens = 262,144 indexable tokens. The old flat 1000
  // addressed only 63,936 and silently capped prefix reuse; the main process
  // also backstops this on every create path.
  maxCacheBlocks: 4097,
  kvCacheQuantization: 'auto',
  kvCacheGroupSize: 64,
  omniBackend: 'stage1',
  enableDiskCache: false,
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
  nativeMtpMode: 'auto',
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
  defaultDoSample: undefined,
  defaultSamplingDefaultsDeclared: false,
  defaultEnableThinking: undefined,
  dsv4PrefixCache: false,
  dsv4PoolQuant: undefined,
  dsv4ActivationQat: false,
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

export const DSV4_PAGED_CACHE_BLOCK_SIZE = 256
export const DSV4_MAX_CACHE_BLOCKS = 4097

// Engine family_name values (vmlx_engine/model_configs.py) offered by the
// manual Model-Family override. These are passed verbatim to --model-family,
// so they MUST be the engine's underscore-form names (not the panel registry's
// dotted/hyphenated display names). 'auto' keeps autodetection.
export const MODEL_FAMILY_OVERRIDE_NAMES: string[] = [
  'qwen3_5', 'qwen3_5_moe', 'qwen3', 'qwen3_moe', 'qwen3_vl', 'qwen3_next',
  'qwen2', 'qwen2_vl', 'qwen_mamba',
  'llama', 'llama4', 'mistral', 'mistral4', 'mistral3', 'ministral3',
  'devstral', 'codestral', 'pixtral',
  'deepseek', 'deepseek_v4', 'deepseek_vl',
  'glm5', 'glm4_moe', 'glm_z1', 'gpt_oss',
  'gemma', 'gemma3', 'gemma3_text', 'gemma4', 'gemma4_text', 'medgemma',
  'phi4', 'phi4_reasoning', 'phi4_multimodal', 'phi3',
  'nemotron', 'nemotron_h', 'cohere', 'granite', 'granitemoehybrid', 'lfm2',
  'minimax', 'minicpm', 'kimi', 'kimi_k25', 'ling', 'zaya', 'zaya1_vl', 'mimo_v2',
  'hy_v3', 'step', 'step_vl', 'step3p7', 'hermes', 'mamba', 'jamba',
  'openpangu_v2',
]

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
  maxTokens: 0,               // Bundle/engine-owned output cap. Users can set an explicit cap per server/chat/API request.
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
  detectedUsePagedCache?: boolean
  /** Detected architecture cache subtype for KV models with typed native cache contracts */
  detectedCacheSubtype?: string
  /** Detected model family for feature gating where cache type alone is ambiguous */
  detectedFamily?: string
  /** Bundle-grounded per-architecture hints that do not change generic cache controls */
  detectedArchitectureHints?: Record<string, string | number | boolean>
  detectedToolParser?: string
  detectedReasoningParser?: string
  detectedEnableAutoToolChoice?: boolean
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
    nativeCacheType?: string
    requiresDeterministicSampling?: boolean
    blockedReason?: string
  }
  /** Model type — image models show minimal settings */
  modelType?: 'text' | 'image'
  /** Image mode — 'edit' or 'generate' (only relevant when modelType is 'image') */
  imageMode?: string
  /** Session ID for components that need to query the running backend (e.g. DistributedNodeList). Omit for the CreateSession form where the session doesn't exist yet. */
  sessionId?: string
  /** Model path/name used only for artifact-specific policy labels (for example Bonsai's q8 exception). */
  modelIdentity?: string
}

export function SessionConfigForm({ config, onChange, onReset, detectedCacheType, detectedUsePagedCache, detectedCacheSubtype, detectedFamily, detectedArchitectureHints, detectedToolParser, detectedReasoningParser, detectedEnableAutoToolChoice, detectedIsTurboQuant, detectedIsMultimodal, detectedForceTextOnly, detectedMaxContext, detectedNativeMtp, modelType, imageMode, sessionId, modelIdentity }: SessionConfigFormProps) {
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
  const normalizedEffectiveFamily = normalizeDetectedFamilyName(
    resolveEffectiveModelFamily(config.modelFamily, normalizedDetectedFamily),
  )
  const minicpmCacheCodecRestricted = normalizedEffectiveFamily === 'minicpm'
  const dsv4Active = normalizedEffectiveFamily === 'deepseek-v4'
  const m3Active = normalizedDetectedFamily === 'minimax_m3'
  const hy3Active = normalizedDetectedFamily === 'hy_v3' || normalizedDetectedFamily === 'hy3'
  const openPanguExactTypedCache = normalizedDetectedFamily === 'openpangu_v2'
  const effectiveSmeltActive = !!config.smelt && !dsv4Active
  const effectiveFlashMoeActive = !!config.flashMoe && !dsv4Active
  const effectiveDistributedActive = !!config.distributedEnabled && !dsv4Active
  const smeltActive = effectiveSmeltActive
  const flashMoeActive = effectiveFlashMoeActive
  const distributedActive = effectiveDistributedActive
  const zayaCcaActive = isZayaCcaFamily(normalizedDetectedFamily)
  const turboQuantActive = !!detectedIsTurboQuant
  const multimodalActive = !dsv4Active && !detectedForceTextOnly && (!!detectedIsMultimodal || config.isMultimodal === true)
  const hybridCacheActive =
    detectedCacheType === 'hybrid' ||
    detectedCacheType === 'mamba' ||
    detectedCacheType === 'rotating_kv'
  const effectiveContinuousBatching = dsv4Active ? true : config.continuousBatching
  const batchingOff = !effectiveContinuousBatching
  const effectivelyNoBatching = batchingOff
  const effectivePrefixCacheEnabled = config.enablePrefixCache
  const prefixOff = !effectivePrefixCacheEnabled
  const lagunaMixedSwaTurboQuantActive = isLagunaMixedSwaTurboQuantEffective({
    detected: {
      family: normalizedDetectedFamily,
      architectureHints: detectedArchitectureHints,
    },
    kvCacheQuantization: config.kvCacheQuantization,
    explicitKvCacheQuantizationApplied:
      !effectivelyNoBatching &&
      !prefixOff &&
      !!config.kvCacheQuantization &&
      config.kvCacheQuantization !== 'auto',
  })
  const isMambaCache =
    detectedCacheType === 'mamba' ||
    detectedCacheType === 'hybrid' ||
    detectedCacheType === 'rotating_kv'
  const mixedSwaBlockDiskOnlySupported =
    detectedCacheType === 'rotating_kv' ||
    detectedCacheSubtype === 'mixed_swa_kv' ||
    detectedCacheSubtype === 'step3p7_full_sliding_kv'
  const stepMixedSwaBlockDiskOnly = detectedCacheSubtype === 'step3p7_full_sliding_kv'
  const architectureBlockDiskOnlySupported =
    (detectedCacheType === 'mamba' ||
      detectedCacheType === 'hybrid' ||
      mixedSwaBlockDiskOnlySupported ||
      m3Active ||
      dsv4Active) &&
    !zayaCcaActive &&
    !openPanguExactTypedCache
  const normalizedModelIdentity = (modelIdentity || '').toLowerCase()
  const bonsaiActive = normalizedModelIdentity.includes('bonsai')
  const qwenHybridTqActive = isMambaCache && (normalizedDetectedFamily || '').startsWith('qwen')
  const qwenFullTqActive = !isMambaCache && (normalizedDetectedFamily || '').startsWith('qwen')
  const mixedSwaCacheActive =
    detectedCacheType === 'rotating_kv' ||
    detectedCacheSubtype === 'mixed_swa_kv'
  const subtypeRequiresPagedCache =
    detectedCacheSubtype === 'step3p7_full_sliding_kv' ||
    detectedCacheSubtype === 'mixed_swa_kv'
  // 2026-07-12 parity: a native cache-type/subtype only FORCES paged in the UI
  // when detection actually resolved paged ON for this family. Gemma mixed-SWA is
  // rotating_kv but paged-OFF (detectedUsePagedCache=false), so it must NOT show a
  // forced/checked-disabled paged box while the launch emits --no-paged-cache. This
  // mirrors the launch gate (sessions.ts architectureRequiresPagedCache).
  const nativePagedFamilyActive =
    (isMambaCache || subtypeRequiresPagedCache) && detectedUsePagedCache === true
  const architectureRequiresPagedCache = zayaCcaActive || nativePagedFamilyActive
  const zayaTypedCacheRequiresPaged = zayaCcaActive && !batchingOff && !prefixOff
  const cacheControlState = {
    continuousBatching: effectiveContinuousBatching,
    enablePrefixCache: effectivePrefixCacheEnabled,
    usePagedCache: openPanguExactTypedCache ? false : config.usePagedCache,
    enableDiskCache: dsv4Active ? false : config.enableDiskCache,
    enableBlockDiskCache: openPanguExactTypedCache ? false : config.enableBlockDiskCache,
    architectureRequiresPagedCache,
    architectureSupportsBlockDiskOnly: architectureBlockDiskOnlySupported,
  }
  const cachePolicy = resolveCacheControlPolicy(cacheControlState)
  const nativeCacheRequiresPaged = cachePolicy.architectureForcedPagedActive && nativePagedFamilyActive
  const effectiveUsePagedCache = cachePolicy.effectiveUsePagedCache
  const blockDiskOnly = cachePolicy.blockDiskCacheChecked && !effectiveUsePagedCache
  const genericPagedCacheToggleDisabled = cachePolicy.pagedCacheDisabled || openPanguExactTypedCache
  const effectivePagedCacheBlockSize = dsv4Active ? DSV4_PAGED_CACHE_BLOCK_SIZE : config.pagedCacheBlockSize
  const pagedCacheUiState = pagedCacheControlsState(effectiveUsePagedCache, blockDiskOnly)
  const effectivePagedCapacityText = pagedCacheCapacityText({
    blockSize: effectivePagedCacheBlockSize,
    maxBlocks: config.maxCacheBlocks,
    defaultBlockSize: DEFAULT_CONFIG.pagedCacheBlockSize,
    defaultMaxBlocks: DEFAULT_CONFIG.maxCacheBlocks,
  })
  const pagedCacheSectionTitle = t('sessions.config.pagedKVCache')
  const nativeTypedCacheOwnsStoredCodec = dsv4Active || m3Active || openPanguExactTypedCache
  // openPangu's typed snapshot stays full precision and explicitly opts out of
  // generic live/stored KV codecs. Do not leave the disabled selector saying
  // "Auto / TurboQuant" when the runtime contract is intentionally "None".
  const effectiveStoredCacheQuantization = openPanguExactTypedCache
    ? 'none'
    : minicpmCacheCodecRestricted && (config.kvCacheQuantization === 'q4' || config.kvCacheQuantization === 'q8')
      ? 'auto'
      : nativeTypedCacheOwnsStoredCodec ? 'auto' : config.kvCacheQuantization
  const explicitStoredCacheCodec = effectiveStoredCacheQuantization !== 'auto'
  const liveCacheCodecLabel = openPanguExactTypedCache
    ? 'openPangu typed composite cache'
    : dsv4Active
      ? 'DeepSeek-V4 native composite cache'
      : m3Active
        ? 'MiniMax-M3 native MSA cache'
        : explicitStoredCacheCodec
          ? effectiveStoredCacheQuantization === 'none'
            ? 'Live TurboQuant and stored quantization disabled'
            : `Live TurboQuant disabled; stored cache ${effectiveStoredCacheQuantization}`
          : hy3Active
            ? 'Native HY3 KV + TQ4 stored prefixes'
            : mixedSwaCacheActive
              ? 'Engine-selected mixed-SWA live cache + q4 stored prefixes'
              : qwenFullTqActive
                ? 'TQ4 bulk attention KV + TQ8 boundary layers'
              : qwenHybridTqActive
                ? bonsaiActive
                  ? 'TQ8 attention KV + native hybrid state'
                  : 'TQ4 attention KV + native hybrid state'
              : 'Engine-selected native cache'
  const liveCacheCodecBadge =
    openPanguExactTypedCache || dsv4Active || m3Active || explicitStoredCacheCodec
      ? 'TURBOQUANT OFF'
      : hy3Active
        ? 'TQ4 AUTO'
        : mixedSwaCacheActive
          ? 'MIXED AUTO'
          : qwenFullTqActive
            ? 'MIXED TQ4/8 AUTO'
          : qwenHybridTqActive
            ? bonsaiActive ? 'TQ8 AUTO' : 'TQ4 AUTO'
            : 'AUTO'
  const effectiveMaxNumSeqs = dsv4Active ? 1 : config.maxNumSeqs
  const effectivePrefillBatchSize = dsv4Active ? 1 : config.prefillBatchSize
  const effectiveCompletionBatchSize = dsv4Active ? 1 : config.completionBatchSize
  const detectedRuntimeVideoCapable = [
    'qwen3-vl',
    'qwen3.5',
    'qwen3.5-moe',
    'qwen2-vl',
    'gemma4',
    'nemotron-h',
    'mistral3',
    'mistral4',
    'pixtral',
    'kimi-k25',
  ].includes(normalizedDetectedFamily || '')
  const showVideoControls = !dsv4Active && !detectedForceTextOnly && multimodalActive && (
    detectedRuntimeVideoCapable ||
    (!normalizedDetectedFamily && config.isMultimodal === true)
  )
  const nativeMtpDetected = detectedNativeMtp !== undefined
  const nativeMtpSupported = !!detectedNativeMtp?.supported
  const omniBackendVisible = normalizedDetectedFamily === 'nemotron-h' && multimodalActive
  const nativeMtpMode = config.nativeMtpMode || DEFAULT_CONFIG.nativeMtpMode || 'auto'
  const nativeMtpDepth = config.nativeMtpDepthOverride === true
    ? (config.nativeMtpDepth || detectedNativeMtp?.depth || 3)
    : (detectedNativeMtp?.depth || config.nativeMtpDepth || 3)
  const hasDeclaredSamplingDefaults =
    config.defaultSamplingDefaultsDeclared === true ||
    config.defaultDoSample === false ||
    config.defaultTemperature > 0 ||
    config.defaultTopP > 0 ||
    (config.defaultTopK ?? 0) > 0 ||
    (config.defaultMinP ?? 0) > 0 ||
    config.defaultRepetitionPenalty > 0
  const generationDefaultsSummary = [
    (config.defaultMaxNewTokens ?? 0) > 0 ? `max output tokens ${Math.floor(config.defaultMaxNewTokens ?? 0)}` : null,
    config.defaultDoSample === false ? 'sampling off' : null,
    hasDeclaredSamplingDefaults ? `temperature ${(config.defaultTemperature / 100).toFixed(2)}` : null,
    config.defaultTopP > 0 ? `top-p ${(config.defaultTopP / 100).toFixed(2)}` : null,
    hasDeclaredSamplingDefaults ? ((config.defaultTopK ?? 0) > 0 ? `top-k ${Math.floor(config.defaultTopK ?? 0)}` : 'top-k off') : null,
    (config.defaultMinP ?? 0) > 0 ? `min-p ${((config.defaultMinP ?? 0) / 100).toFixed(2)}` : null,
    config.defaultRepetitionPenalty > 0 ? `repetition ${(config.defaultRepetitionPenalty / 100).toFixed(2)}` : null,
  ].filter(Boolean).join(', ')
  const lagunaXsTopKMetadataWarning =
    normalizedDetectedFamily === 'laguna' &&
    detectedArchitectureHints?.lagunaVariant === 'xs-2.1' &&
    Number(config.defaultTopK ?? 0) !== 20
  const dsv4TopPMismatch = shouldWarnDsv4TopP(
    normalizedDetectedFamily,
    Number(config.defaultTopP) / 100,
  )

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

  useEffect(() => {
    if (
      minicpmCacheCodecRestricted &&
      (config.kvCacheQuantization === 'q4' || config.kvCacheQuantization === 'q8')
    ) {
      onChange('kvCacheQuantization', 'auto')
    }
  }, [minicpmCacheCodecRestricted, config.kvCacheQuantization, onChange])

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
      {dsv4TopPMismatch && (
        <IncompatWarning text={t('common.dsv4TopPAdvisory')} />
      )}
      {lagunaXsTopKMetadataWarning && (
        <IncompatWarning text={t('sessions.config.lagunaXsTopKWarning')} />
      )}
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
          tooltip="Maximum time in seconds to wait for a single inference request to complete before timing out. Increase this for very long generations or slow models. Default is 300s for most models; MiniMax-M3 and DSV4 auto-use 900s unless you choose a custom value."
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
          {!dsv4Active && <PerformanceHint text="Controls how many requests your server handles at once. Keep Continuous Batching ON to enable the caching engine." />}
          {!dsv4Active && (
            <button
              onClick={(e) => { e.preventDefault(); e.stopPropagation(); setShowCachingHelp(true) }}
              className="w-6 h-6 flex items-center justify-center rounded-full bg-accent/50 text-accent-foreground hover:bg-accent hover:text-white transition-colors text-xs font-bold"
              title={t('sessions.config.cachingReferenceTitle')}
            >
              ?
            </button>
          )}
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
              defaultValue={DEFAULT_CONFIG.flashMoeSlotBank}
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
        <CheckField
          label="Continuous Batching"
          tooltip={dsv4Active
            ? "DSV4 Flash requires the continuous-batching DSV4BatchGenerator path. Prefix reuse uses its native SWA+CSA/HCA typed cache; the RAM paged tier and persistent Block Disk L2 remain independently configurable."
            : "Keep ON for best performance. This is the master switch for Prefix Cache, In-Memory Paged Cache (RAM), Block Disk Cache (SSD / L2), and stored-cache codecs. Turning it off uses the direct single-request engine and disables the cache features below."}
          checked={effectiveContinuousBatching}
          onChange={v => onChange('continuousBatching', v)}
          disabled={dsv4Active}
        />
        {!dsv4Active && <PerformanceHint text="Keep ON for best overall behavior: it enables prefix reuse, the in-memory RAM tier, persistent SSD L2, and architecture-specific cache restore while the default max sequence count stays at one for local chat." />}
        {dsv4Active && <InfoNote text="DSV4 Flash stays on its native DSV4BatchGenerator path. Prefix reuse defaults On and Block Disk Cache (SSD / L2) defaults On as the warm/cold stack. In-Memory Paged Cache (RAM) remains optional and bounded when enabled; the CSA/HCA pool codec remains bundle-derived." />}
        {!effectiveContinuousBatching && effectivePrefixCacheEnabled && (
          <InfoNote text="Cache flags will be omitted at launch while continuous batching is off. Turn it back on to use Prefix Cache, In-Memory Paged Cache (RAM), Block Disk Cache (SSD / L2), and stored-cache codecs." />
        )}
        {!effectiveContinuousBatching && (
          <InfoNote text="Turning this off disables Prefix Cache, In-Memory Paged Cache (RAM), KV cache quantization, and disk caching. Enable it to unlock these features." />
        )}
        <InfoNote text={metalWiredLimitHelpText} />
      </Section>

      {/* Prefix Cache */}
      <Section title={t('sessions.config.prefixCache')} expanded={expandedSections.prefixCache} onToggle={() => toggleSection('prefixCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text="Speeds up repeated conversations by remembering previous prompts. Makes follow-up messages much faster (lower time-to-first-token)." />}
        {dsv4Active && <InfoNote text="DSV4 prefix reuse preserves the native SWA plus CSA/HCA composite state. It is controlled by this standard Prefix Cache switch; there is no separate hidden DSV4 cache toggle." />}
        {openPanguExactTypedCache && <InfoNote text="openPangu v2 uses exact typed N-1 prompt snapshots. Memory and prompt-disk L2 preserve MLA KV, DSA indexer state, rotating-SWA metadata, and all three causal-convolution states together. Generic paged blocks, reverse truncation, and generic KV q4/q8 stay off." />}
        {batchingOff && <IncompatWarning text="Prefix cache requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above to enable prefix caching." />}
        <CheckField label="Enable Prefix Cache" tooltip="Caches prompt prefixes so repeated system prompts, documents, and conversation history can reuse their computed state instead of repeating prefill. The selected RAM and/or SSD tier below owns where reusable blocks are retained." checked={effectivePrefixCacheEnabled} onChange={v => onChange('enablePrefixCache', v)} />
        {!dsv4Active && effectivePrefixCacheEnabled && (
          <>
            {openPanguExactTypedCache && <InfoNote text="Memory-aware mode is required for openPangu's non-aliasing typed cache clone. The legacy entry-count backend is unavailable for this family." />}
            <CheckField label="Legacy Entry-Count Cache" tooltip="Switches from memory-aware cache (which uses Cache Memory %, Cache Memory Limit, and Cache TTL controls) to a simpler entry-count cache. When ON: you control cache by max entries only. When OFF: you get fine-grained memory budget controls (% of RAM, MB limit, TTL expiration). Memory-aware mode is recommended for most users." checked={openPanguExactTypedCache ? false : config.noMemoryAwareCache} onChange={v => onChange('noMemoryAwareCache', v)} disabled={openPanguExactTypedCache} />
            {!dsv4Active && !openPanguExactTypedCache && config.noMemoryAwareCache ? (
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
                  <IncompatWarning text={pagedCacheMemoryIgnoredText} />
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
                  disabled={pagedCacheUiState.memoryBudgetControlsDisabled}
                />
                <SliderField
                  label="Cache Memory %"
                  tooltip="Percentage of available system RAM to allocate for the prefix cache. Only used when Cache Memory Limit is set to 'Auto-detect'. Default 15% leaves headroom for model weights and active generation. Higher values cache more prefixes but risk memory pressure during long generations."
                  value={config.cacheMemoryPercent}
                  onChange={v => onChange('cacheMemoryPercent', v)}
                  min={1}
                  max={100}
                  step={1}
                  defaultValue={DEFAULT_CONFIG.cacheMemoryPercent}
                  maxInput={100}
                  disabled={pagedCacheUiState.memoryBudgetControlsDisabled}
                />
                {blockDiskOnly && <IncompatWarning text="In-Memory Paged Cache (RAM) is Off and Block Disk Cache (SSD / L2) is authoritative, so Cache Memory Limit, Cache Memory %, and Cache TTL do not apply. Block Size and Max Cache Blocks bound the in-memory hash/index capacity; Block Cache Max bounds SSD usage." />}
                <SliderField
                  label="Cache TTL (minutes)"
                  tooltip="Time-to-live for memory-aware cache entries. Entries not accessed within this window are evicted to free memory. 'No expiration' means entries are only evicted by memory pressure. This setting has no effect while In-Memory Paged Cache (RAM) is on; that tier uses LRU eviction based on Max Cache Blocks."
                  value={config.cacheTtlMinutes}
                  onChange={v => onChange('cacheTtlMinutes', v)}
                  min={1}
                  max={120}
                  step={5}
                  defaultValue={30}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel="No expiration"
                  disabled={pagedCacheUiState.cacheTtlDisabled}
                />
              </>
            )}

            {/* Caching Help Modal */}
            {!dsv4Active && showCachingHelp && (
              <Modal title="Caching & Compatibility Engine" onClose={() => setShowCachingHelp(false)} className="max-w-2xl max-h-[85vh] overflow-y-auto">
                <div className="space-y-6 text-sm">
                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">{t('sessions.config.continuousBatchingEngine')}</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      <strong>{t('sessions.config.continuousBatching')}</strong> is the heart of vMLX's server performance. Unlike simple mode (which processes exactly one request at a time), continuous batching allows multiple requests to be processed simultaneously. More importantly, <strong>it is required to enable all advanced caching features</strong> (Prefix Cache, In-Memory Paged Cache, KV Quantization, and Disk Cache).
                    </p>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">{t('sessions.config.prefixCachingModes')}</h3>
                    <p className="text-muted-foreground leading-relaxed mb-2">
                      Prefix caching drastically speeds up interactions by remembering previous prompts (like a system prompt or a long document), skipping the expensive prefill phase.
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                      <li><strong>{t('sessions.config.memoryAwareDefault')}</strong> Intelligently manages the cache based on explicit memory boundaries (MB) or a percentage of total system RAM. It automatically evicts the oldest items when crossing these limits.</li>
                      <li><strong>{t('sessions.config.legacyEntryCount')}</strong> A simpler system that just stores a fixed number of complete prompt states regardless of their size. Useful if you want strict deterministic eviction.</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">{t('sessions.config.mambaHybridCompat')}</h3>
                    <p className="text-muted-foreground leading-relaxed mb-2">
                      Newer models like Qwen 2.5/3, Falcon Mamba, and Jamba mix standard Attention (KV cache) with SSM blocks (Mamba/Arrays cache).
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                      <li><strong>{t('sessions.config.kvQuantizationLabel')}</strong> vMLX securely isolates Mamba layers. If you turn on KV Quantization (e.g. q8), it will safely compress the Attention layers while leaving the internal Mamba/SSM memory at full precision, ensuring no corruption or quality loss.</li>
                      <li><strong>{t('sessions.config.inMemoryPagedCache')}</strong> Some models use this RAM tier when Prefix Cache is enabled so attention KV blocks and path-dependent state share one cache contract. Supported models can instead use Block Disk Cache as an SSD-only tier.</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">{t('sessions.config.kvCacheQuantization')}</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      By converting stored prompts to q8 or q4 precision, you can reduce the cache's RAM footprint by 2-4x. <strong>{t('sessions.config.onlyCompressesSavedPrefixes')}</strong>. The actual text generation continues to run at standard full precision natively in MLX.
                    </p>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold tracking-tight text-foreground mb-2">{t('sessions.config.visionLanguageModels')}</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {t('sessions.config.coreEngineHandlesVision')} <strong>{t('sessions.config.prefixCachingWorksForImages')}</strong> If you repeatedly ask questions about the exact same image (like in a tool-calling flow analyzing a dashboard), the massive vision embedding prefill is cached and reused instantly.
                    </p>
                  </div>
                </div>
              </Modal>
            )}
          </>
        )}
      </Section>

      {/* In-memory paged cache (RAM) */}
      <Section title={pagedCacheSectionTitle} expanded={expandedSections.pagedCache} onToggle={() => toggleSection('pagedCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text="Keeps reusable prefix blocks in Apple unified memory as small pages instead of one large allocation. This is the fast RAM tier; Block Disk Cache (SSD / L2) below is the persistent tier." />}
        {dsv4Active && <InfoNote text="DSV4 uses 256-token typed composite blocks. The bounded RAM tier defaults On for hot reuse, with Block Disk Cache as the persistent warm/cold tier. You can turn RAM Off and keep SSD-only reuse." />}
        {batchingOff && <IncompatWarning text="In-Memory Paged Cache (RAM) requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above to enable the RAM cache tier." />}
        {!dsv4Active && config.enableDiskCache && <IncompatWarning text="In-Memory Paged Cache (RAM) and legacy Disk Cache cannot run simultaneously. Enabling the RAM tier will auto-disable legacy Disk Cache. For persistent SSD caching, use 'Block Disk Cache (SSD / L2)' below instead." />}
        {!dsv4Active && !batchingOff && prefixOff && !cachePolicy.architectureRequiresPagedCache && <InfoNote text="In-Memory Paged Cache (RAM) is a prefix-cache backend. Turning it on will enable Prefix Cache." />}
        {!batchingOff && prefixOff && cachePolicy.architectureRequiresPagedCache && <IncompatWarning text="This model uses a native/in-memory paged cache when Prefix Cache is enabled. Enable Prefix Cache above to activate the architecture-specific cache stack." />}
        {zayaTypedCacheRequiresPaged && <InfoNote text="ZAYA typed CCA cache requires the in-memory paged tier while Prefix Cache is enabled. Turn off Prefix Cache to disable this cache stack for ZAYA." />}
        {nativeCacheRequiresPaged && !zayaTypedCacheRequiresPaged && <InfoNote text="This native cache route requires the in-memory paged tier while Prefix Cache is enabled so KV blocks and path-dependent state stay in the same cache contract." />}
        {dsv4Active && cachePolicy.blockDiskCacheChecked && <InfoNote text={blockDiskOnly
          ? "DSV4 SSD-only mode preserves typed SWA plus CSA/HCA state in Block Disk L2 without claiming a retained RAM payload tier."
          : "DSV4 uses the RAM paged tier as L1 and Block Disk L2 as the persistent fallback, with its pool codec derived from the loaded bundle."} />}
        {architectureBlockDiskOnlySupported && !m3Active && !dsv4Active && cachePolicy.blockDiskCacheChecked && <InfoNote text={mixedSwaBlockDiskOnlySupported
          ? stepMixedSwaBlockDiskOnly
            ? "Step full/sliding-KV SSD-only mode is available: typed KV blocks and rotating metadata stay in Block Disk L2 without RAM payloads. Under tight Metal headroom, long cold-prompt stores can be skipped to avoid an unsafe second clean prefill; existing SSD blocks remain reusable."
            : "Native sliding/mixed-SWA SSD-only mode is available: turn In-Memory Paged Cache Off to keep typed KV blocks and rotating-window metadata in Block Disk L2 without retaining RAM payloads."
          : "Hybrid/Mamba SSD-only mode is available: turn In-Memory Paged Cache Off to keep attention KV blocks in Block Disk L2 while restoring full-precision SSM/GDN companion state from its typed SSD store or clean-prefill rederive."} />}
        {m3Active && <InfoNote text={blockDiskOnly
          ? "MiniMax-M3 SSD-only mode preserves native MSA keys, values, idx_keys, and absolute offsets in Block Disk L2 while keeping persistent RAM payloads disabled."
          : "MiniMax-M3 uses a native typed MSA paged cache that preserves keys, values, idx_keys, and absolute offsets. Block Disk Cache provides its persistent L2; generic KV q4/q8 remains disabled."} />}
        {openPanguExactTypedCache && <InfoNote text="openPangu does not use generic paged blocks: causal-convolution state is cumulative and cannot be reconstructed from an arbitrary block. Use Prefix Cache plus prompt-level Disk Cache (L2) instead." />}
        <CheckField label="In-Memory Paged Cache (RAM)" tooltip="Keeps reusable prompt-prefix and native cache blocks in Apple unified memory (shared by CPU and GPU) for faster repeated prompts. This is the fast RAM tier and is not persistent. Block Disk Cache (SSD / L2) can remain enabled when this RAM tier is Off." checked={effectiveUsePagedCache} onChange={v => applyCacheControlUpdates(cacheControlUpdatesForPagedToggle(v, cacheControlState))} disabled={genericPagedCacheToggleDisabled} />
        {(effectiveUsePagedCache || cachePolicy.blockDiskCacheChecked) && (
          <>
            <InfoNote text={blockDiskOnly
              ? effectivePagedCapacityText.replace('Effective in-memory cache capacity', 'Effective SSD block-index capacity')
              : effectivePagedCapacityText} />
            <SliderField
              label="Block Size (tokens)"
              tooltip={dsv4Active
                ? "DSV4 native SWA+CSA/HCA cache records require fixed 256-token blocks. This value is read-only for this architecture."
                : "Number of tokens per content-addressed cache block in the in-memory paged tier or Block Disk Cache. Smaller blocks reduce waste per sequence but increase management overhead. Default 64 is optimal for most models."}
              value={effectivePagedCacheBlockSize}
              onChange={v => onChange('pagedCacheBlockSize', v)}
              min={1}
              max={1024}
              step={16}
              defaultValue={dsv4Active ? DSV4_PAGED_CACHE_BLOCK_SIZE : DEFAULT_CONFIG.pagedCacheBlockSize}
              disabled={dsv4Active}
            />
            <SliderField
              label="Max Cache Blocks"
              tooltip="Maximum total number of KV cache blocks allocated. Block 0 is permanently reserved as the null/placeholder block, so usable token capacity = block_size x (max_blocks - 1). Increase for longer contexts, decrease to save memory."
              value={config.maxCacheBlocks}
              onChange={v => onChange('maxCacheBlocks', v)}
              min={2}
              max={100000}
              step={100}
              defaultValue={dsv4Active ? DSV4_MAX_CACHE_BLOCKS : DEFAULT_CONFIG.maxCacheBlocks}
              maxInput={100000}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="Default (1000)"
            />
          </>
        )}
        {!batchingOff && !effectiveUsePagedCache && <InfoNote text="Block Disk Cache can run as a pure SSD prefix tier while In-Memory Paged Cache remains Off. It keeps only the content-addressed block index in memory, restores KV payloads transiently from SSD, and still requires Prefix Cache." />}
        <CheckField
          label="Block Disk Cache (SSD / L2)"
          tooltip="Persist content-addressed prefix blocks to SSD. With In-Memory Paged Cache (RAM) On, SSD is L2 behind the RAM tier. With the RAM tier Off, SSD is the authoritative block tier and KV payloads are restored only transiently for reconstruction. Compatible runtimes preserve native TurboQuant or typed cache records."
          checked={cachePolicy.blockDiskCacheChecked}
          onChange={v => applyCacheControlUpdates(cacheControlUpdatesForBlockDiskToggle(v, cacheControlState))}
          disabled={!cachePolicy.blockDiskCacheVisible || cachePolicy.blockDiskCacheDisabled || openPanguExactTypedCache}
        />
        {cachePolicy.blockDiskCacheChecked && (
          <>
            <SliderField
              label="Block Cache Max (GB)"
              tooltip="Maximum physical disk space for the managed block-cache root, shared across model/config namespaces and typed companion state. Least-recently-used entries are evicted when the aggregate root exceeds the limit. If multiple live sessions share the root, the smallest finite limit is enforced. Set to 0 for unlimited only when no live session supplies a finite limit."
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
                {t('sessions.config.blockCacheDirectory')}
                <Tooltip text="Managed root for block-level disk cache files. A model/config-specific subdirectory is created automatically, and the size limit applies across all managed subdirectories and typed companions in this root. Leave empty for ~/.cache/vmlx-engine/block-cache/." />
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
        {!effectivelyNoBatching && !prefixOff && mixedSwaCacheActive && <PerformanceHint text="Mixed sliding/full attention cache detected — Auto preserves the model's native cache-slot and rotating-window metadata and applies q4 at a compatible live or stored-cache boundary selected by the engine. The running health panel reports whether live full-attention TurboQuant or storage-only q4 is active. Explicit None disables both." />}
        {!effectivelyNoBatching && !prefixOff && hy3Active && <PerformanceHint text="HY3 plain-KV cache detected — Auto uses TQ4 for RAM/SSD L2 stored prefixes while live decode stays on the native KV cache. Native MTP D1 copies this cache independently before batch split/verify." />}
        {!effectivelyNoBatching && !prefixOff && qwenHybridTqActive && !mixedSwaCacheActive && <PerformanceHint text={bonsaiActive ? 'Bonsai hybrid cache detected — Auto applies TQ8 only to compatible attention KV and preserves native SSM/GLA companion state.' : 'Qwen hybrid cache detected — Auto applies TQ4 only to compatible attention KV and preserves native SSM/GLA companion state.'} />}
        {!effectivelyNoBatching && !prefixOff && qwenFullTqActive && <PerformanceHint text="Qwen full-KV cache detected — Auto stores bulk attention KV with TQ4 and protects the first/last six boundary layers with TQ8. Explicit None disables both live TQ-KV and stored quantization." />}
        {!effectivelyNoBatching && !prefixOff && isMambaCache && !qwenHybridTqActive && !mixedSwaCacheActive && !dsv4Active && !m3Active && !openPanguExactTypedCache && <PerformanceHint text="Hybrid stateful cache detected — the engine keeps SSM/GLA state native and only uses cache codecs proven for that architecture. Generic TurboQuant KV is disabled unless a tested override exists." />}
        {!effectivelyNoBatching && dsv4Active && <PerformanceHint text="DeepSeek-V4 keeps generic TurboQuant KV q4/q8 disabled. Prefix RAM/SSD records preserve native SWA+CSA/HCA state, while CSA/HCA pool quantization is read from the loaded bundle rather than a generic cache-codec override." />}
        {!effectivelyNoBatching && m3Active && <PerformanceHint text="MiniMax-M3 keeps generic KV q4/q8 disabled. Prefix reuse uses native MSA snapshots with keys, values, idx_keys, and absolute offsets; generic stored-KV codecs cannot preserve that cache format." />}
        {!effectivelyNoBatching && openPanguExactTypedCache && <PerformanceHint text="openPangu keeps generic KV q4/q8 disabled. Its exact typed snapshot owns MLA KV, DSA indexer, rotating-SWA metadata, and causal-convolution state as one full-precision record." />}
        {!effectivelyNoBatching && minicpmCacheCodecRestricted && <PerformanceHint text="MiniCPM keeps generic KV q4/q8 disabled because live source-artifact validation reproduced cold/warm output divergence. Auto uses native raw KV; None explicitly disables cache quantization." />}
        {/* Live/native cache codec — automatic per architecture. */}
        <div className="block">
          <span className="text-xs font-medium text-muted-foreground">
            {t('sessions.config.liveCacheCodec')}
            <Tooltip text="Auto mode leaves the CLI flag unset so the engine can choose per architecture: calibrated TurboQuant for compatible plain KV/JANGTQ caches, native composite or typed caches for DSV4/ZAYA/hybrid SSM, and stored-prefix fallback only where that codec is valid." />
          </span>
          <div className="cfg-input flex items-center justify-between" style={{ background: 'var(--card)', cursor: 'default' }}>
            <span>{liveCacheCodecLabel}</span>
            <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--success-bg, rgba(34,197,94,0.15))', color: 'var(--success-fg, rgb(34,197,94))' }}>
              {liveCacheCodecBadge}
            </span>
          </div>
        </div>
        {dsv4Active && (
          <>
            <div className="block">
              <span className="text-xs font-medium text-muted-foreground">
                {t('sessions.config.nativePoolCodec')}
                <Tooltip text="This is DSV4's architecture-native compressed-pool codec, not generic TurboQuant KV. The value is detected from the loaded bundle and is not user-overridable from this generic cache panel." />
              </span>
              <div className="cfg-input flex items-center justify-between" style={{ background: 'var(--card)', cursor: 'default' }}>
                <span>{t('sessions.config.dsv4PoolQuantization')}</span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--success-bg, rgba(34,197,94,0.15))', color: 'var(--success-fg, rgb(34,197,94))' }}>
                  {config.dsv4PoolQuant === true
                    ? 'ON (BUNDLE)'
                    : config.dsv4PoolQuant === false
                      ? 'OFF (BUNDLE)'
                      : 'ENGINE / BUNDLE DEFAULT'}
                </span>
              </div>
            </div>
            <CheckField
              label="DSV4 Activation QAT"
              tooltip="Restart required. On enables source-native E4M3 round-trips for attention KV and compressed pools plus Hadamard-128 + FP4 E2M1 indexer round-trips. Off skips only those activation-QAT transforms to avoid their runtime overhead; FP32 compressor staging remains enabled and is not controlled here."
              checked={config.dsv4ActivationQat === true}
              onChange={v => onChange('dsv4ActivationQat', v)}
            />
            <InfoNote text="Default Off. Enable only when you want the source-native DSV4 activation-QAT graph and accept its extra runtime work. This switch does not change weights, sampling, cache pool quantization, or FP32 compressor staging." />
          </>
        )}

        {/* Stored prefix-cache compression — orthogonal to TurboQuant. */}
        <div className="block">
          <span className="text-xs font-medium text-muted-foreground">
            {t('sessions.config.storedCacheQuantization')}
            <Tooltip text="Controls how completed prompt states are stored in the prefix cache. Auto keeps the engine's production codec choice. None explicitly disables stored-cache quantization. q8/q4 force the generic stored-cache codec and also disable calibrated live TurboQuant so the explicit choice is honored." />
          </span>
          <select value={effectiveStoredCacheQuantization} onChange={e => onChange('kvCacheQuantization', e.target.value)} className="cfg-input" disabled={effectivelyNoBatching || prefixOff || nativeTypedCacheOwnsStoredCodec}>
            <option value="auto">{dsv4Active ? 'Native typed codec (bundle-derived)' : 'Auto (engine-selected: native/TurboQuant + stored fallback)'}</option>
            <option value="none">{t('sessions.config.kvQuantNone')}</option>
            {!minicpmCacheCodecRestricted && <option value="q8">q8 (8-bit, ~2x stored cache savings)</option>}
            {!minicpmCacheCodecRestricted && <option value="q4">q4 (4-bit, ~4x stored cache savings)</option>}
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
          <InfoNote text="DSV4 uses Block Disk Cache (SSD / L2) above for persistent native composite blocks. This legacy whole-prompt disk format remains unavailable for DSV4 typed cache records." />
        ) : (
          <InfoNote text="Legacy prompt disk cache works with the memory-aware prefix backend. Block Disk Cache (SSD / L2) persists content-addressed blocks whether In-Memory Paged Cache (RAM) is On or explicitly Off. Only one disk format can be active at a time." />
        )}
        {openPanguExactTypedCache && <InfoNote text="For openPangu this prompt-level disk cache stores the exact typed N-1 composite and restores it across process restarts. Block Disk Cache remains unavailable." />}
        {batchingOff && <IncompatWarning text="Disk cache requires continuous batching. Turn on 'Continuous Batching' in the Concurrent Processing section above." />}
        {!effectivelyNoBatching && cachePolicy.legacyDiskCacheUnavailableReason === 'paged-cache-active' && <IncompatWarning text="Legacy disk cache is not compatible with In-Memory Paged Cache (RAM). For persistent SSD storage, use 'Block Disk Cache (SSD / L2)' in that section instead. To use legacy Disk Cache, disable the RAM tier first." />}
        {!effectivelyNoBatching && cachePolicy.legacyDiskCacheUnavailableReason === 'architecture-requires-paged-cache' && <IncompatWarning text="This architecture requires a native/in-memory paged cache while Prefix Cache is enabled. Use 'Block Disk Cache (SSD / L2)' in the In-Memory Paged Cache section for persistent SSD storage." />}
        {!batchingOff && prefixOff && !cachePolicy.legacyDiskCacheDisabled && <InfoNote text="Disk cache is persistent L2 behind Prefix Cache. Turning it on will enable Prefix Cache and disable the in-memory and block-cache backends." />}
        <CheckField
          label="Enable Disk Cache"
          tooltip="Persist whole-prompt caches to disk for reuse across server restarts. This legacy format acts as L2 behind the memory-aware prefix backend. It requires Prefix Cache and is not compatible with In-Memory Paged Cache (RAM); use Block Disk Cache (SSD / L2) for persistent content-addressed blocks instead."
          checked={cachePolicy.legacyDiskCacheChecked}
          onChange={v => applyCacheControlUpdates(cacheControlUpdatesForDiskToggle(v, cacheControlState))}
          disabled={dsv4Active || cachePolicy.legacyDiskCacheDisabled}
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
                {t('sessions.config.cacheDirectory')}
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
              {t('sessions.config.sleepWhenIdle')}
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
        <PerformanceHint text="Controls token streaming, response length, and prompt-window limits. Max Output Tokens caps generated tokens; Max Context Tokens caps accepted prompt/context tokens." />
        {/* Whole-model JIT is not available for path-dependent cache models. */}
        <Field label="Model-wide JIT (mx.compile)" tooltip="Compile the entire model forward graph for Metal kernel fusion. This is separate from model-native compiled kernels, which remain automatic when supported. Whole-model JIT requires a trace-safe cache topology and a restart.">
          <label className={`flex items-center gap-2 ${flashMoeActive || distributedActive || dsv4Active || m3Active || zayaCcaActive || turboQuantActive || lagunaMixedSwaTurboQuantActive || multimodalActive || hybridCacheActive ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>
            <input
              type="checkbox"
              checked={!!config.enableJit && !flashMoeActive && !distributedActive && !dsv4Active && !m3Active && !zayaCcaActive && !turboQuantActive && !lagunaMixedSwaTurboQuantActive && !multimodalActive && !hybridCacheActive}
              onChange={e => onChange('enableJit', e.target.checked)}
              disabled={flashMoeActive || distributedActive || dsv4Active || m3Active || zayaCcaActive || turboQuantActive || lagunaMixedSwaTurboQuantActive || multimodalActive || hybridCacheActive}
              className="rounded border-input"
            />
            <span className="text-xs text-muted-foreground">
              {t('sessions.config.fuseMetalOps')}
            </span>
          </label>
        </Field>
        {(flashMoeActive || distributedActive || dsv4Active || m3Active || zayaCcaActive || turboQuantActive || lagunaMixedSwaTurboQuantActive || multimodalActive || hybridCacheActive) && (
          <IncompatWarning text={dsv4Active
            ? "Whole-model JIT is not trace-safe for DeepSeek-V4's path-dependent SWA+CSA/HCA cache. Native compiled decode remains automatic: the supported DSV4 runtime uses compiled router/SwiGLU operations and a fused Metal mHC single-token decode kernel."
            : m3Active
            ? "JIT is disabled for MiniMax-M3 native MSA cache. The Lightning-Indexer idx_keys path must stay on the uncompiled scheduler path."
            : zayaCcaActive
            ? "JIT is disabled for ZAYA typed CCA cache. CCA state is path-dependent and the full cache stack benchmarks faster on the uncompiled scheduler path."
            : multimodalActive
            ? "JIT is disabled for multimodal/VLM models. The mlx-vlm streaming path owns image/video preprocessing and stream context state that is not safe to trace with mx.compile."
            : hybridCacheActive
            ? "JIT is disabled for hybrid SSM/Mamba cache models. Their path-dependent Python cache objects are not mx.compile safe."
            : turboQuantActive
            ? "Server-level mx.compile is disabled for JANGTQ/TurboQuant KV because the live cache uses custom TurboQuant objects that mx.compile cannot trace. JANGTQ fused Metal kernels still run."
            : lagunaMixedSwaTurboQuantActive
            ? "JIT is disabled for Laguna while Auto cache quantization uses TurboQuantKVCache on full-attention slots and preserves native rotating sliding-window slots. Choose an explicit stored-cache codec (including None) to disable the live TurboQuant wrapper before enabling JIT."
            : flashMoeActive
            ? "JIT is disabled while Flash MoE is on. Flash MoE's on-demand expert loading is incompatible with mx.compile tracing."
            : "JIT is disabled while distributed mode is on. Distributed orchestration cannot safely compile the local coordinator graph."} />
        )}
        {dsv4Active && (
          <PerformanceHint text="Native compiled decode: Automatic. Only unsafe whole-model cache tracing is unavailable." />
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
          label="Max Output Tokens"
          tooltip="Default generated-token cap for this local server. This maps to --max-tokens and only limits response length; it does not change prompt/context length. Leave on Bundle / engine default unless you intentionally want a server-level cap."
          value={config.maxTokens}
          onChange={v => onChange('maxTokens', v)}
          min={1}
          max={32768}
          step={256}
          defaultValue={(config.defaultMaxNewTokens ?? 0) > 0 ? Math.floor(config.defaultMaxNewTokens ?? 0) : 4096}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={(config.defaultMaxNewTokens ?? 0) > 0 ? `Bundle (${Math.floor(config.defaultMaxNewTokens ?? 0)})` : 'Bundle / engine default'}
          maxInput={1000000}
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
              <span className="text-xs font-medium text-muted-foreground">{t('sessions.config.liveMcpStatus')}</span>
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
        <SelectField
          label="Automatic Tool Choice"
          tooltip="Auto follows the detected model/tool-parser contract. On explicitly enables automatic tool selection. Off explicitly disables it."
          value={config.enableAutoToolChoice === undefined ? 'auto' : config.enableAutoToolChoice ? 'on' : 'off'}
          onChange={value => onChange(
            'enableAutoToolChoice',
            value === 'auto' ? undefined : value === 'on',
          )}
          options={[
            {
              value: 'auto',
              label: `Auto (detected: ${detectedEnableAutoToolChoice ? 'On' : 'Off'})`,
            },
            { value: 'on', label: 'On' },
            { value: 'off', label: 'Off' },
          ]}
        />
        {config.enableAutoToolChoice === undefined && (
          <InfoNote text={`Auto-detect is currently ${detectedEnableAutoToolChoice ? 'On' : 'Off'} for this model.`} />
        )}
        <ParserField
          label="Tool Call Parser"
          tooltip="Specifies how to parse the model's tool call output. Each model family uses a different format (Qwen, Llama, Mistral, Hermes, DeepSeek, GLM, etc). 'Auto-detect' reads config.json to pick the right one. If auto-detection fails (e.g. GGUF, renamed fine-tunes), select the parser matching your model's base architecture. Click '?' to see format examples and supported models for each parser."
          value={canonicalizeToolParserId(config.toolCallParser) ?? 'auto'}
          onChange={v => onChange('toolCallParser', v)}
          options={TOOL_PARSER_OPTIONS}
          detectedValue={detectedToolParser}
        />
        <ParserField
          label="Reasoning Parser"
          tooltip="Separates reasoning/thinking from final content. Use Auto-detect unless it picks wrong. Qwen3: Qwen, QwQ, MiniMax, StepFun (strict <think> tags). DeepSeek R1: DeepSeek-R1, GLM-4.7, Phi-4, Nemotron (lenient <think> tags). GPT-OSS: GLM-4.7 Flash (Harmony protocol). Mistral 4: Mistral Small/Large 4 ([THINK] tags). Click '?' for full model list."
          value={config.reasoningParser === 'none' ? '' : config.reasoningParser}
          onChange={v => onChange('reasoningParser', v)}
          options={REASONING_PARSER_OPTIONS}
          detectedValue={detectedReasoningParser}
        />
        <SelectField
          label="Model Family (override)"
          tooltip="Force the model family instead of letting the engine autodetect it from jang_config.json / config.json. Use this for renamed fine-tunes, GGUF, or custom merges where detection picks the wrong family. The chosen family's cache + tool/reasoning parser contract is applied. Leave on Auto unless detection is wrong — forcing a family whose cache contract mismatches the weights (e.g. forcing a plain-KV family onto a hybrid/SSM or MLA model) can produce garbled output."
          value={config.modelFamily ?? 'auto'}
          onChange={v => onChange('modelFamily', v === 'auto' ? undefined : v)}
          options={[
            { value: 'auto', label: `Auto (detected: ${detectedFamily ?? 'unknown'})` },
            ...MODEL_FAMILY_OVERRIDE_NAMES.map(name => ({ value: name, label: name })),
          ]}
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
          tooltip="Vision-Language Model mode for models like Qwen2-VL, Qwen3-VL, Pixtral, InternVL, or LLaVA. Auto uses the detected model/runtime policy. Force Off remains an explicit user override. Smelt and documented unsafe runtimes use text-only loading."
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
          <IncompatWarning text="This bundle includes media metadata, but its detected vMLX runtime is currently text-only. Attachments stay disabled until that family's native media path is live-verified; changing quantization format alone does not make the media route available." />
        )}
        {!dsv4Active && !smeltActive && !detectedForceTextOnly && config.isMultimodal === true && (
          <InfoNote text="VLM mode is active — the MLLM scheduler handles image/video processing with Prefix Cache, In-Memory Paged Cache (RAM), and KV quantization support." />
        )}
        {!dsv4Active && !smeltActive && !detectedForceTextOnly && config.isMultimodal === false && (
          <InfoNote text="VLM mode is off only when the model is not auto-detected as multimodal. Detected VLM bundles launch with image/video support." />
        )}
        {omniBackendVisible && (
          <SelectField
            label="Omni Backend"
            tooltip="Nemotron-Omni encoder backend. Stage 1 is the correctness-first PyTorch/MPS bridge. Stage 2 maps to --omni-backend stage2 / VMLX_OMNI_BACKEND=stage2 for native MLX RADIO + Parakeet benchmarking."
            value={config.omniBackend || 'stage1'}
            onChange={v => onChange('omniBackend', v as 'stage1' | 'stage2')}
            options={[
              { value: 'stage1', label: 'Stage 1 correctness' },
              { value: 'stage2', label: 'Stage 2 native MLX' },
            ]}
          />
        )}
        {normalizedDetectedFamily === 'gemma4' && multimodalActive && (
          <SelectField
            label="Image Token Budget"
            tooltip="Gemma 4 visual soft-token budget per image. 280 is the bundle default; use 560 or 1120 for OCR and small text at higher prefill cost. The selected value is sent as image_token_budget and is part of the media cache identity."
            value={String(config.imageTokenBudget ?? 280)}
            onChange={v => onChange('imageTokenBudget', Number(v))}
            options={[
              { value: '70', label: '70 — fastest / lowest detail' },
              { value: '140', label: '140 — low detail' },
              { value: '280', label: '280 — bundle default' },
              { value: '560', label: '560 — detailed' },
              { value: '1120', label: '1120 — OCR / small text' },
            ]}
          />
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
      <Section title="Native MTP" expanded={expandedSections.nativeMtp} onToggle={() => toggleSection('nativeMtp')} hidden={isImage || dsv4Active || !nativeMtpDetected}>
        {!nativeMtpSupported && (
          <IncompatWarning text={detectedNativeMtp?.blockedReason || 'Native MTP weights were detected, but this bundle has not passed the runtime compatibility gate. Autoregressive decode remains active.'} />
        )}
        {nativeMtpSupported && (
          <>
        <PerformanceHint text="Uses the model's own preserved MTP heads and measured model-local depth when present, with D3 as the generic fallback." />
        {nativeMtpMode === 'auto' && (
          <InfoNote text="Auto preserves the bundle's generation_config/jang_config sampling defaults. It activates MTP only for compatible requests; sampled requests fall back to autoregressive decode and the server logs the reason." />
        )}
        {nativeMtpMode === 'deterministic' && (
          <InfoNote text={`Explicit deterministic mode applies D${nativeMtpDepth} and greedy startup sampling so omitted API/chat sampling values enter the native MTP path. Explicit per-request sampling parameters still win.`} />
        )}
        <SelectField
          label="Native MTP Mode"
          tooltip="Auto preserves bundle sampling defaults and uses MTP only when a request is compatible. Deterministic mode explicitly replaces omitted sampling values with greedy defaults. Off disables the in-model MTP runtime."
          value={nativeMtpMode}
          onChange={v => onChange('nativeMtpMode', v as 'deterministic' | 'auto' | 'off')}
          options={[
            { value: 'auto', label: 'Auto (bundle defaults)' },
            { value: 'deterministic', label: 'Deterministic override' },
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
        <InfoNote text={`Detected scope: ${detectedNativeMtp?.runtimeScope || 'text'}; native cache: ${detectedNativeMtp?.nativeCacheType || detectedCacheSubtype || detectedCacheType || 'unknown'}; depth source: ${detectedNativeMtp?.depthSource || 'default'}. Hybrid cache bundles use the in-memory paged tier while Prefix Cache is enabled so KV blocks and SSM state stay in one cache contract.`} />
          </>
        )}
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
              {t('sessions.config.recommendedUsageToday')} <code>vmlx-worker</code> on
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
              <div className="text-xs font-medium text-foreground">{t('sessions.config.setupGuide')}</div>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>1. Connect Macs via <strong>{t('sessions.config.thunderboltCable')}</strong> (fastest) or Ethernet/WiFi</p>
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
            {t('sessions.config.resetAllParameters')}
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
    // Tooltip triggers commonly live inside <label> elements. Prevent the
    // label's default activation so opening help never toggles the owning
    // checkbox or changes an unsaved server setting.
    e.preventDefault()
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
    <span
      ref={triggerRef}
      className="relative inline-flex ml-1"
      onClick={handleClick}
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
    >
      <span
        className={`inline-flex items-center justify-center w-3.5 h-3.5 rounded-full text-[10px] font-bold cursor-help select-none ${pinned ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground'}`}
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
    value: 'openpangu', label: 'openPangu — openPangu-2.0-Flash', format: '<|tool_call_start|>[{"name":"fn","arguments":{...}}]<|tool_call_end|>', models: [
      'openPangu-2.0-Flash (92B MoE, 6B active)',
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
    value: 'dsml', label: 'DeepSeek V4 / DSV4-Flash — DSML', format: '<｜DSML｜invoke name="fn"><｜DSML｜parameter name="arg" string="true">val</｜DSML｜parameter></｜DSML｜invoke>', models: [
      'DeepSeek-V4-Flash / DSV4-Flash JANG, JANGTQ, and DQ bundles',
      'Use this for deepseek_v4 model_type; DeepSeek V3/R1 use the DeepSeek parser above',
    ]
  },
  {
    value: 'hunyuan', label: 'Hy3 / Hunyuan — Tencent XML tools', format: '<tool_calls><tool_call>fn<tool_sep><arg_key>arg</arg_key><arg_value>val</arg_value></tool_call></tool_calls>', models: [
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
    value: 'xml_function', label: 'MiMo / generic XML function', format: '<tool_call><function=fn><parameter=arg>val</parameter></function></tool_call>', models: [
      'MiMo-V2.5 JANG bundles',
      'Generic XML function-call templates with <parameter=...> values',
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
    value: 'lfm2', label: 'Liquid LFM2 — Liquid AI LFM2 / LFM2-MoE', format: '<|tool_call_start|>[fn(arg=val)]<|tool_call_end|>', models: [
      'LFM2.5-8B-A1B', 'LFM2-MoE',
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
  {
    value: 'atem', label: 'Muse Glimmer — ATEM', format: '<atem:function_calls><atem:invoke name="fn"><atem:parameter name="arg">val</atem:parameter></atem:invoke></atem:function_calls>', models: [
      'Muse Glimmer 30B (text+vision+video)',
    ]
  },
  {
    value: 'minimax_m3', label: 'MiniMax M3 — MiniMax-M3 (sparse MSA + Lightning-Indexer)', format: 'native tool_call (MiniMax M3 parser)', models: [
      'MiniMax-M3 (REAP22 / JANG_2L)',
      'Auto-detected for minimax_m3 / minimax_m3_vl bundles.',
    ]
  },
]

const REASONING_PARSER_OPTIONS: ParserOption[] = [
  { value: 'auto', label: 'Auto-detect (recommended)' },
  { value: '', label: 'None (disable reasoning extraction)' },
  {
    value: 'qwen3', label: 'Qwen3 — Qwen / QwQ / StepFun', format: '<think>...reasoning...</think>content  (strict: both tags required)', models: [
      'Qwen3.5-VL (0.8B\u2013122B MoE, vision+reasoning)', 'Qwen3 (0.6B\u2013235B, all sizes)',
      'Qwen3-Coder (all sizes)', 'Qwen3-MoE (22B/57B)', 'QwQ-32B',
      'StepFun Step-3.7 Flash JANG/VL', 'StepFun Step-3.5 Flash (8B MoE)', 'StepFun Step-3.5', 'StepFun Step-1V (vision)',
    ]
  },
  {
    value: 'minimax_m2', label: 'MiniMax M2 — MiniMax M2 / M2.5', format: '<think>...reasoning...</think>content  (MiniMax M2 parser)', models: [
      'MiniMax-M2 (46B)', 'MiniMax-M2.5 (172B MoE)', 'MiniMax Prism Pro (80B)',
      'Use this when a stale bundle sidecar still says qwen3; Auto normalizes MiniMax to minimax_m2.',
    ]
  },
  {
    value: 'minimax_m3', label: 'MiniMax M3 — MiniMax-M3 (sparse MSA)', format: '<mm:think>...reasoning...</mm:think>content  (MiniMax M3 parser)', models: [
      'MiniMax-M3 (REAP22 / JANG_2L)',
      'Auto-detected for minimax_m3 / minimax_m3_vl bundles.',
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
  {
    value: 'muse_glimmer', label: 'Muse Glimmer — recipient channels', format: '<|start|>assistant to=self<|message|>reasoning<|eom|> then to=user<|message|>answer<|eot|>', models: [
      'Muse Glimmer 30B (reasoning_strength: low/medium/high/xhigh)',
    ]
  },
  {
    value: 'think_xml', label: 'Think XML — MiMo XML reasoning', format: '<think>...reasoning...</think>content  (XML reasoning blocks)', models: [
      'MiMo V2.5 JANG 2L',
      'Use only when model metadata selects think_xml; MiMo generation quality remains separately gated.',
    ]
  },
]

function ParserField({ label, tooltip, value, onChange, options, detectedValue }: {
  label: string; tooltip: string; value: string; onChange: (v: string) => void; options: ParserOption[]; detectedValue?: string
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
          <option key={o.value} value={o.value}>{o.value === 'auto' && detectedValue ? `Auto (detected: ${detectedValue})` : o.label}</option>
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
    // Keep raw text locally so partially typed values are never clamped
    // mid-keystroke. Once it is already a valid in-range number, publish it
    // immediately so Save cannot observe the previous parent value.
    const raw = e.target.value
    setLocalInput(raw)
    if (raw === '') return

    const parsed = Math.round(Number(raw))
    const withinHardMaximum = maxInput == null || parsed <= maxInput
    if (Number.isFinite(parsed) && parsed >= min && withinHardMaximum) {
      onChange(parsed)
    }
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

  // Anchor the range track to the step grid so round values (64, 512, 1000,
  // 1024…) are representable. With a raw min of 1 the grid is 1 + k·step, so a
  // value like 64 (step 16) falls between 49 and 65 and the browser snaps the
  // thumb to 65 while the number field shows the true 64 — a visible off-by-one
  // between the paired controls. Anchor the range's min/max and thumb to the
  // step grid; the number input keeps the exact semantic value/min.
  const sliderMin = Math.ceil(min / step) * step
  const sliderMax = Math.max(sliderMin, Math.floor(max / step) * step)
  const snapToGrid = (v: number) => sliderMin + Math.round((v - sliderMin) / step) * step
  const sliderValue = isUnlimited
    ? sliderMin
    : Math.min(Math.max(snapToGrid(value), sliderMin), sliderMax)
  // Show local input while editing, parent value otherwise
  const displayValue = localInput !== null ? localInput : (isUnlimited ? '' : value)

  return (
    <div
      className={`block ${disabled ? 'opacity-50 pointer-events-none' : ''}`}
      data-setting-label={label}
      data-setting-value={String(value)}
      data-unlimited-active={String(isUnlimited)}
    >
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
            aria-pressed={isUnlimited}
            aria-label={`${label}: ${unlimitedLabel} ${isUnlimited ? 'active' : 'inactive'}`}
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
          min={sliderMin}
          max={sliderMax}
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

/** Commit a pending number edit before a settings action reads parent state. */
export function commitActiveSettingsInput() {
  const active = document.activeElement
  if (active instanceof HTMLInputElement && active.type === 'number') {
    active.blur()
  }
}
