import { DEFAULT_BLOCK_DISK_CACHE_PERCENT } from '../../../../shared/cacheDefaults'
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
  pagedCacheControlsState,
  resolvePagedCacheCapacity,
} from '../../../../shared/cacheCapacityDisplay'
import { metalWiredLimitCommand } from '../../../../shared/metalWiredLimit'
import { isLagunaMixedSwaTurboQuantEffective } from '../../../../shared/lagunaCachePolicy'
import { normalizeMcpPolicyList } from '../../../../shared/mcpPolicy'
import { canonicalizeToolParserId, describeDetectedToolParser } from '../../../../shared/toolParserAliases'
import { shouldWarnDsv4TopP } from '../../../../shared/samplingParameterDomain'
import { resolveEffectiveModelFamily } from '../../../../shared/dsv4Env'
import {
  isZayaCcaFamily,
  normalizeDetectedFamilyName,
  usesExactTypedPromptDiskCache,
} from '../../../../shared/detectedFamilyNames'
import { computeEffectiveJit, isJitSuppressedByRuntime } from '../../../../shared/jitPolicy'
import { isMixedSwaBundle } from '../../../../shared/storedKvQuantPolicy'
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
  blockDiskCacheMaxGb?: number
  blockDiskCacheMaxPercent: number
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
  /** Per-frame pixel budget for sampled video frames (aspect preserved); unset = processor default. */
  videoMaxPixels?: number
  /** Whole-clip vision-token budget (derived into the loader's per-clip pixel budget); unset = processor default. */
  videoTokenBudget?: number
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
  // Intentionally absent, not 0: the engine reads 0 as UNLIMITED, so a seeded
  // 0 is emitted at launch and hands the whole disk to the cache while this
  // form shows a 10% budget.

  blockDiskCacheMaxPercent: DEFAULT_BLOCK_DISK_CACHE_PERCENT,
  blockDiskCacheDir: '',
  streamInterval: 8,
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
  nativeMtpDepthOverride: true,
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
  'minimax', 'kimi', 'kimi_k25', 'ling', 'zaya', 'zaya1_vl', 'mimo_v2',
  'hy_v3', 'step', 'step_vl', 'step3p7', 'hermes', 'mamba', 'jamba',
  'openpangu_v2',
]



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
  // Auto omits the CLI flag. The engine's production default preserves each
  // architecture's native cache objects and imposes no generic TQ codec.
  kvCacheQuantization: 'auto',
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
    defaultMode?: 'auto' | 'off'
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

export function SessionConfigForm({ config, onChange, onReset, detectedCacheType, detectedCacheSubtype, detectedFamily, detectedArchitectureHints, detectedToolParser, detectedReasoningParser, detectedEnableAutoToolChoice, detectedIsTurboQuant, detectedIsMultimodal, detectedForceTextOnly, detectedMaxContext, detectedNativeMtp, modelType, imageMode, sessionId, modelIdentity }: SessionConfigFormProps) {
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
  // Clearing the SSD cache for THIS session. The engine owns the block store
  // while it is running, so this goes through the engine rather than deleting
  // files underneath it; the button says so when the session is stopped.
  const [clearingSsdCache, setClearingSsdCache] = useState(false)
  const [ssdClearResult, setSsdClearResult] = useState<string | null>(null)
  const handleClearSsdCache = async () => {
    setClearingSsdCache(true)
    setSsdClearResult(null)
    try {
      const endpoint = { host: String(config.host || '127.0.0.1'), port: Number(config.port) }
      const res: any = await (window as any).api?.cache?.clear('prefix', endpoint, sessionId)
      // The engine reports WHICH tiers it cleared and, separately, which it
      // refused to touch — `paged_prefix:blocks_in_use` means a live request
      // still holds those blocks, so nothing was freed there. Reporting a flat
      // "Cleared." over a skipped tier would tell the user the cache is gone
      // when it is not.
      const cleared: string[] = Array.isArray(res?.caches) ? res.caches : []
      const skipped: string[] = Array.isArray(res?.skipped) ? res.skipped : []
      if (skipped.length > 0) {
        setSsdClearResult(
          t('sessions.config.clearSsdCachePartial', {
            cleared: cleared.length,
            skipped: skipped.join(', '),
          }),
        )
      } else if (cleared.length > 0) {
        setSsdClearResult(t('sessions.config.clearSsdCacheDone', { count: cleared.length }))
      } else {
        setSsdClearResult(t('sessions.config.clearSsdCacheEmpty'))
      }
    } catch (e: any) {
      setSsdClearResult(t('sessions.config.clearSsdCacheFailed', { error: String(e?.message || e) }))
    } finally {
      setClearingSsdCache(false)
    }
  }
  const [mcpStatus, setMcpStatus] = useState<{ servers: LiveMcpServer[]; tools: LiveMcpTool[]; error?: string } | null>(null)
  const [mcpStatusLoading, setMcpStatusLoading] = useState(false)
  const [mcpValidation, setMcpValidation] = useState<{ servers: any[]; serverCount?: number; error?: string } | null>(null)
  const [mcpValidationLoading, setMcpValidationLoading] = useState(false)
  const [mcpImportLoading, setMcpImportLoading] = useState(false)

  const normalizedDetectedFamily = normalizeDetectedFamilyName(detectedFamily)
  const normalizedEffectiveFamily = normalizeDetectedFamilyName(
    resolveEffectiveModelFamily(config.modelFamily, normalizedDetectedFamily),
  )
  const dsv4Active = normalizedEffectiveFamily === 'deepseek-v4'
  const m3Active = normalizedDetectedFamily === 'minimax_m3'
  const hy3Active = normalizedDetectedFamily === 'hy_v3' || normalizedDetectedFamily === 'hy3'
  const openPanguExactTypedCache = normalizedDetectedFamily === 'openpangu_v2'
  const exactTypedPromptDiskCache = usesExactTypedPromptDiskCache(normalizedDetectedFamily)
  const effectiveSmeltActive = !!config.smelt && !dsv4Active
  const effectiveFlashMoeActive = !!config.flashMoe && !dsv4Active
  const effectiveDistributedActive = !!config.distributedEnabled && !dsv4Active
  const smeltActive = effectiveSmeltActive
  const flashMoeActive = effectiveFlashMoeActive
  const distributedActive = effectiveDistributedActive
  const zayaCcaActive = isZayaCcaFamily(normalizedDetectedFamily)
  const turboQuantActive = !!detectedIsTurboQuant
  const multimodalActive = !dsv4Active && !detectedForceTextOnly && (!!detectedIsMultimodal || config.isMultimodal === true)
  const dflash2Speculative = /dflash2/i.test(config.speculativeModel || '')
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
  const jitSuppressedByRuntime = isJitSuppressedByRuntime({
    isMultimodal: multimodalActive,
    flashMoeActive,
    distributedActive,
    dsv4Active,
    m3Active,
    zayaCcaActive,
    turboQuantActive,
    lagunaMixedSwaTurboQuantActive,
    hybridCacheActive,
  })
  const mixedSwaBundle = isMixedSwaBundle({
    cacheType: detectedCacheType,
    cacheSubtype: detectedCacheSubtype,
    architectureHints: detectedArchitectureHints,
  })
  const isMambaCache =
    detectedCacheType === 'mamba' ||
    detectedCacheType === 'hybrid' ||
    detectedCacheType === 'rotating_kv'
  // Same question as mixedSwaCacheActive below; both were hand-copied
  // three-condition chains. One detector now answers it.
  const mixedSwaBlockDiskOnlySupported = mixedSwaBundle
  const stepMixedSwaBlockDiskOnly = detectedCacheSubtype === 'step3p7_full_sliding_kv'
  const architectureBlockDiskOnlySupported =
    (detectedCacheType === 'mamba' ||
      detectedCacheType === 'hybrid' ||
      mixedSwaBlockDiskOnlySupported ||
      m3Active ||
      dsv4Active) &&
    !zayaCcaActive &&
    !exactTypedPromptDiskCache
  const normalizedModelIdentity = (modelIdentity || '').toLowerCase()
  const bonsaiActive = normalizedModelIdentity.includes('bonsai')
  const qwenHybridTqActive = isMambaCache && (normalizedDetectedFamily || '').startsWith('qwen')
  const qwenFullTqActive = !isMambaCache && (normalizedDetectedFamily || '').startsWith('qwen')
  // step3p7_full_sliding_kv belongs here too. It is already listed as
  // mixed-SWA in mixedSwaBlockDiskOnlySupported above and in the shared
  // storedKvQuantPolicy subtype set, and the ENGINE classifies it as mixed-SWA
  // (mllm_scheduler's detector returns true for it alongside mixed_swa_kv).
  // Omitting it here alone made the cache codec badge read a generic "AUTO"
  // with the engine-native description, for a family the rest of the app —
  // and the engine — treats as mixed-SWA. Display-only, but display-only is
  // exactly where a wrong label goes unnoticed.
  const mixedSwaCacheActive = mixedSwaBundle
  // No detected family may force the retired RAM tier back on. ZAYA remains an
  // explicit SSD-reconstruction gap until its CCA companion state is restorable;
  // the UI says that directly instead of silently showing a forced RAM toggle.
  const architectureRequiresPagedCache = false
  const zayaSsdReuseUnavailable = zayaCcaActive && !batchingOff && !prefixOff
  const cacheControlState = {
    continuousBatching: effectiveContinuousBatching,
    enablePrefixCache: effectivePrefixCacheEnabled,
    usePagedCache: exactTypedPromptDiskCache ? false : config.usePagedCache,
    enableDiskCache: dsv4Active ? false : config.enableDiskCache,
    enableBlockDiskCache: exactTypedPromptDiskCache ? false : config.enableBlockDiskCache,
    architectureRequiresPagedCache,
    architectureSupportsBlockDiskOnly: architectureBlockDiskOnlySupported,
  }
  const cachePolicy = resolveCacheControlPolicy(cacheControlState)
  const effectiveUsePagedCache = cachePolicy.effectiveUsePagedCache
  const blockDiskOnly = cachePolicy.blockDiskCacheChecked && !effectiveUsePagedCache
  const genericPagedCacheToggleDisabled = cachePolicy.pagedCacheDisabled || exactTypedPromptDiskCache
  const effectivePagedCacheBlockSize = dsv4Active ? DSV4_PAGED_CACHE_BLOCK_SIZE : config.pagedCacheBlockSize
  const pagedCacheUiState = pagedCacheControlsState(effectiveUsePagedCache, blockDiskOnly)
  // The shared module still owns the ARITHMETIC (and its English sentence, which
  // the main process reuses in error messages); the renderer owns the wording so
  // this sentence localizes like every other label around it. Live-caught: with
  // the app in Korean this line and the Metal wired-limit note were the only
  // prose left in English on the whole form.
  const pagedCapacity = resolvePagedCacheCapacity({
    blockSize: effectivePagedCacheBlockSize,
    maxBlocks: config.maxCacheBlocks,
    defaultBlockSize: DEFAULT_CONFIG.pagedCacheBlockSize,
    defaultMaxBlocks: DEFAULT_CONFIG.maxCacheBlocks,
  })
  const effectivePagedCapacityText = t('sessions.config.pagedCacheCapacity', {
    blockSize: pagedCapacity.blockSize,
    usableBlocks: pagedCapacity.usableBlocks,
    maxBlocks: pagedCapacity.maxBlocks,
    tokens: pagedCapacity.capacityTokens.toLocaleString(),
  })
  // A SEPARATE key, not string surgery on the localized sentence. The previous
  // version took the paged text above and swapped the English phrase naming
  // in-memory capacity for one naming the SSD block index. That is a no-op in
  // every non-English locale, so block-disk-only mode mislabelled the SSD block
  // index as RAM capacity everywhere except English.
  const effectiveBlockDiskCapacityText = t('sessions.config.blockDiskCapacity', {
    blockSize: pagedCapacity.blockSize,
    usableBlocks: pagedCapacity.usableBlocks,
    maxBlocks: pagedCapacity.maxBlocks,
    tokens: pagedCapacity.capacityTokens.toLocaleString(),
  })
  const pagedCacheSectionTitle = t('sessions.config.pagedKVCache')
  // One production representation: preserve the loaded architecture's native
  // cache state and add no generic stored codec. Persisted stale values are
  // migrated by the main process; the renderer never echoes them as effective.
  const effectiveStoredCacheQuantization = 'auto'
  const liveCacheCodecLabel = openPanguExactTypedCache
    ? t('sessions.config.codecOpenPangu')
    : dsv4Active
      ? t('sessions.config.codecDsv4')
      : m3Active
        ? t('sessions.config.codecM3')
        : hy3Active
            ? t('sessions.config.codecHy3')
            : mixedSwaCacheActive
              ? t('sessions.config.codecMixedSwa')
              : qwenFullTqActive
                ? t('sessions.config.codecQwenFull')
              : qwenHybridTqActive
                ? bonsaiActive
                  ? t('sessions.config.codecBonsaiHybrid')
                  : t('sessions.config.codecQwenHybrid')
              : t('sessions.config.codecEngineNative')
  const liveCacheCodecBadge = 'NATIVE · GENERIC TQ OFF'
  const effectiveMaxNumSeqs = dsv4Active ? 1 : config.maxNumSeqs
  const effectivePrefillBatchSize = dsv4Active ? 1 : config.prefillBatchSize
  const effectiveCompletionBatchSize = dsv4Active ? 1 : config.completionBatchSize
  const detectedRuntimeVideoCapable = [
    'qwen3-vl',
    'qwen3.5',
    'qwen3.5-moe',
    // Qwen3.8 Flash-Next (engine family qwen4_exp): native video path,
    // live-proven with per-request fps / frame / pixel / token controls.
    'qwen4-exp',
    'qwen4_exp',
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
  const nativeMtpDepthPolicy = config.nativeMtpDepthOverride === true
    ? 'fixed'
    : 'adaptive'
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
        setMcpValidation({ servers: [], error: result?.error || t('sessions.config.mcpImportFailed') })
      }
    } catch (error) {
      setMcpValidation({ servers: [], error: (error as Error).message })
    } finally {
      setMcpImportLoading(false)
    }
  }

  const validateMcpConfig = async (path = config.mcpConfig) => {
    if (!path?.trim()) {
      setMcpValidation({ servers: [], error: t('sessions.config.mcpPathEmpty') })
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
        setMcpValidation({ servers: [], error: result?.error || t('sessions.config.mcpValidationFailed') })
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
        setMcpStatus({ servers: [], tools: [], error: result?.error || t('sessions.config.mcpStatusUnavailable') })
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
      {dsv4TopPMismatch && (
        <IncompatWarning text={t('common.dsv4TopPAdvisory')} />
      )}
      {lagunaXsTopKMetadataWarning && (
        <IncompatWarning text={t('sessions.config.lagunaXsTopKWarning')} />
      )}
      {/* Product-wide cache contract: the retained paged-RAM tier is locked off
          and SSD is authoritative. Keep this visible outside the collapsed
          section so users can confirm the launched memory policy at a glance. */}
      <InfoNote text={t('sessions.config.ramCacheTradeoffNotice')} />

      {/* Server Settings */}
      <Section title={t('sessions.config.serverSettings')} expanded={expandedSections.server} onToggle={() => toggleSection('server')}>
        <Field label={t('sessions.config.host')} tooltip={t('sessions.config.hostTooltip')}>
          <input type="text" value={config.host} onChange={e => onChange('host', e.target.value)} className="cfg-input" />
        </Field>
        <SliderField
          label={t('sessions.config.port')}
          tooltip={t('sessions.config.portTooltip')}
          value={config.port}
          onChange={v => onChange('port', v)}
          min={1024}
          max={65535}
          step={1}
          defaultValue={DEFAULT_CONFIG.port}
        />
        <Field label={t('sessions.config.apiKey')} tooltip={t('sessions.config.apiKeyTooltip')}>
          <input type="password" value={config.apiKey} onChange={e => onChange('apiKey', e.target.value)} placeholder={t('sessions.config.apiKeyPlaceholder')} className="cfg-input" />
        </Field>
        <Field label={t('sessions.config.servedModelName')} tooltip={t('sessions.config.servedModelNameTooltip')}>
          <input type="text" value={config.servedModelName} onChange={e => onChange('servedModelName', e.target.value)} placeholder={t('sessions.config.servedModelNamePlaceholder')} className="cfg-input" />
        </Field>
        <SliderField
          label={t('sessions.config.rateLimit')}
          tooltip={t('sessions.config.rateLimitTooltip')}
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
          tooltip={t('sessions.config.timeoutTooltip')}
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
        <Field label={t('sessions.config.logLevel')} tooltip={t('sessions.config.logLevelTooltip')}>
          <select value={config.logLevel || 'INFO'} onChange={e => onChange('logLevel', e.target.value)} className="cfg-input">
            <option value="DEBUG">{t('sessions.config.logLevelDebug')}</option>
            <option value="INFO">{t('sessions.config.logLevelInfo')}</option>
            <option value="WARNING">{t('sessions.config.logLevelWarning')}</option>
            <option value="ERROR">{t('sessions.config.logLevelError')}</option>
          </select>
        </Field>
        <Field label={t('sessions.config.corsOrigins')} tooltip={t('sessions.config.corsOriginsTooltip')}>
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
          {!dsv4Active && <PerformanceHint text={t('sessions.config.concurrentHint')} />}
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
          label={t('sessions.config.maxConcurrentSequences')}
          tooltip={t('sessions.config.maxConcurrentSequencesTooltip')}
          value={effectiveMaxNumSeqs}
          onChange={v => onChange('maxNumSeqs', v)}
          min={1}
          max={1024}
          step={1}
          defaultValue={DEFAULT_CONFIG.maxNumSeqs}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={t('sessions.config.defaultWithValue', { n: 1 })}
          disabled={dsv4Active}
        />
        <SliderField
          label={t('sessions.config.prefillBatchSize')}
          tooltip={t('sessions.config.prefillBatchSizeTooltip')}
          value={effectivePrefillBatchSize}
          onChange={v => onChange('prefillBatchSize', v)}
          min={1}
          max={4096}
          step={64}
          defaultValue={DEFAULT_CONFIG.prefillBatchSize}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={t('sessions.config.defaultWithValue', { n: 512 })}
          disabled={dsv4Active}
        />
        <SliderField
          label={t('sessions.config.prefillStepSize')}
          tooltip={t('sessions.config.prefillStepSizeTooltip')}
          value={config.prefillStepSize}
          onChange={v => onChange('prefillStepSize', v)}
          min={64}
          max={8192}
          step={64}
          defaultValue={DEFAULT_CONFIG.prefillStepSize}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={t('sessions.config.defaultWithValue', { n: 2048 })}
          disabled={dsv4Active}
        />
        <SliderField
          label={t('sessions.config.completionBatchSize')}
          tooltip={t('sessions.config.completionBatchSizeTooltip')}
          value={effectiveCompletionBatchSize}
          onChange={v => onChange('completionBatchSize', v)}
          min={1}
          max={4096}
          step={64}
          defaultValue={DEFAULT_CONFIG.completionBatchSize}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={t('sessions.config.defaultWithValue', { n: 512 })}
          disabled={dsv4Active}
        />
        <CheckField
          label={t('sessions.config.smeltMode')}
          tooltip={t('sessions.config.smeltModeTooltip')}
          checked={effectiveSmeltActive}
          onChange={v => {
            onChange('smelt', v)
            // Mutual exclusion: disable Flash MoE if enabling Smelt
            if (v && flashMoeActive) onChange('flashMoe', false)
          }}
          disabled={dsv4Active || effectiveFlashMoeActive}
        />
        {dsv4Active && (
          <IncompatWarning text={t('sessions.config.smeltDisabledDsv4')} />
        )}
        {flashMoeActive && (
          <IncompatWarning text={t('sessions.config.smeltDisabledFlashMoe')} />
        )}
        {smeltActive && (
          <SliderField label={t('sessions.config.smeltExpertsPercent')} value={config.smeltExperts} onChange={v => onChange('smeltExperts', v)} min={10} max={100} step={5} defaultValue={50} />
        )}
        {smeltActive && <PerformanceHint text={t('sessions.config.smeltExpertsHint', { percent: config.smeltExperts })} />}

        <CheckField
          label={t('sessions.config.flashMoe')}
          tooltip={t('sessions.config.flashMoeTooltip')}
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
          <IncompatWarning text={t('sessions.config.flashMoeDisabledDsv4')} />
        )}
        {(smeltActive || distributedActive) && !flashMoeActive && (
          <IncompatWarning text={smeltActive ? t('sessions.config.flashMoeDisabledSmelt') : t('sessions.config.flashMoeDisabledDistributed')} />
        )}
        {flashMoeActive && (
          <>
            <SliderField
              label={t('sessions.config.slotBankSize')}
              tooltip={t('sessions.config.slotBankSizeTooltip')}
              value={config.flashMoeSlotBank}
              onChange={v => onChange('flashMoeSlotBank', v)}
              min={16}
              max={1024}
              step={16}
              defaultValue={DEFAULT_CONFIG.flashMoeSlotBank}
            />
            <SliderField
              label={t('sessions.config.ioWorkers')}
              tooltip={t('sessions.config.ioWorkersTooltip')}
              value={config.flashMoeIoSplit}
              onChange={v => onChange('flashMoeIoSplit', v)}
              min={1}
              max={16}
              step={1}
              defaultValue={4}
            />
            <PerformanceHint text={t('sessions.config.flashMoeStreamingHint', { slots: config.flashMoeSlotBank })} />
          </>
        )}
        <CheckField
          label={t('sessions.config.continuousBatching')}
          tooltip={dsv4Active
            ? t('sessions.config.continuousBatchingTooltipDsv4')
            : t('sessions.config.continuousBatchingTooltip')}
          checked={effectiveContinuousBatching}
          onChange={v => onChange('continuousBatching', v)}
          disabled={dsv4Active}
        />
        {!dsv4Active && <PerformanceHint text={t('sessions.config.continuousBatchingHint')} />}
        {dsv4Active && <InfoNote text={t('sessions.config.dsv4BatchPathNote')} />}
        {!effectiveContinuousBatching && effectivePrefixCacheEnabled && (
          <InfoNote text={t('sessions.config.cacheFlagsOmittedNote')} />
        )}
        {!effectiveContinuousBatching && (
          <InfoNote text={t('sessions.config.batchingOffDisablesNote')} />
        )}
        <InfoNote text={t('sessions.config.metalWiredLimitHelp', { command: metalWiredLimitCommand })} />
      </Section>

      {/* Prefix Cache */}
      <Section title={t('sessions.config.prefixCache')} expanded={expandedSections.prefixCache} onToggle={() => toggleSection('prefixCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text={t('sessions.config.prefixCacheHint')} />}
        {dsv4Active && <InfoNote text={t('sessions.config.dsv4PrefixReuseNote')} />}
        {openPanguExactTypedCache && <InfoNote text={t('sessions.config.openPanguTypedCacheNote')} />}
        {batchingOff && <IncompatWarning text={t('sessions.config.prefixCacheRequiresBatching')} />}
        <CheckField label={t('sessions.config.enablePrefixCache')} tooltip={t('sessions.config.enablePrefixCacheTooltip')} checked={effectivePrefixCacheEnabled} onChange={v => onChange('enablePrefixCache', v)} />
        {!dsv4Active && effectivePrefixCacheEnabled && (
          <>
            {openPanguExactTypedCache && <InfoNote text={t('sessions.config.openPanguMemoryAwareNote')} />}
            <CheckField label={t('sessions.config.legacyEntryCountCache')} tooltip={t('sessions.config.legacyEntryCountCacheTooltip')} checked={exactTypedPromptDiskCache ? false : config.noMemoryAwareCache} onChange={v => onChange('noMemoryAwareCache', v)} disabled={exactTypedPromptDiskCache} />
            {!dsv4Active && !exactTypedPromptDiskCache && config.noMemoryAwareCache ? (
              <>
                <InfoNote text={t('sessions.config.legacyModeActiveNote')} />
                <SliderField
                  label={t('sessions.config.maxCacheEntries')}
                  tooltip={t('sessions.config.maxCacheEntriesTooltip')}
                  value={config.prefixCacheSize}
                  onChange={v => onChange('prefixCacheSize', v)}
                  min={1}
                  max={10000}
                  step={10}
                  defaultValue={DEFAULT_CONFIG.prefixCacheSize}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel={t('sessions.config.defaultWithValue', { n: 100 })}
                />
                <SliderField
                  label={t('sessions.config.prefixCacheMaxBytes')}
                  tooltip={t('sessions.config.prefixCacheMaxBytesTooltip')}
                  value={Math.floor((config.prefixCacheMaxBytes || 0) / (1024 * 1024))}
                  onChange={v => onChange('prefixCacheMaxBytes', v * 1024 * 1024)}
                  min={0}
                  max={32768}
                  step={256}
                  defaultValue={0}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel={t('sessions.cache.unlimited')}
                />
              </>
            ) : (
              <>
                {effectiveUsePagedCache && (
                  <IncompatWarning text={t('sessions.config.pagedCacheMemoryIgnored')} />
                )}
                <SliderField
                  label={t('sessions.config.cacheMemoryLimitMb')}
                  tooltip={t('sessions.config.cacheMemoryLimitTooltip')}
                  value={config.cacheMemoryMb}
                  onChange={v => onChange('cacheMemoryMb', v)}
                  min={256}
                  max={65536}
                  step={256}
                  defaultValue={4096}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel={t('sessions.config.autoDetect')}
                  disabled={pagedCacheUiState.memoryBudgetControlsDisabled}
                />
                <SliderField
                  label={t('sessions.config.cacheMemoryPercent')}
                  tooltip={t('sessions.config.cacheMemoryPercentTooltip')}
                  value={config.cacheMemoryPercent}
                  onChange={v => onChange('cacheMemoryPercent', v)}
                  min={1}
                  max={100}
                  step={1}
                  defaultValue={DEFAULT_CONFIG.cacheMemoryPercent}
                  maxInput={100}
                  disabled={pagedCacheUiState.memoryBudgetControlsDisabled}
                />
                {blockDiskOnly && <IncompatWarning text={t('sessions.config.blockDiskOnlyBudgetNote')} />}
                <SliderField
                  label={t('sessions.config.cacheTtlMinutes')}
                  tooltip={t('sessions.config.cacheTtlTooltip')}
                  value={config.cacheTtlMinutes}
                  onChange={v => onChange('cacheTtlMinutes', v)}
                  min={1}
                  max={120}
                  step={5}
                  defaultValue={30}
                  allowUnlimited
                  unlimitedValue={0}
                  unlimitedLabel={t('sessions.config.noExpiration')}
                  disabled={pagedCacheUiState.cacheTtlDisabled}
                />
              </>
            )}

            {/* Caching Help Modal */}
            {!dsv4Active && showCachingHelp && (
              <Modal title={t('sessions.config.cachingHelpHeader')} onClose={() => setShowCachingHelp(false)} className="max-w-2xl max-h-[85vh] overflow-y-auto">
                <div className="space-y-6 text-sm">
                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">{t('sessions.config.continuousBatchingEngine')}</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      <strong>{t('sessions.config.continuousBatching')}</strong> {t('sessions.config.cachingHelpBatchBody')}
                    </p>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">{t('sessions.config.prefixCachingModes')}</h3>
                    <p className="text-muted-foreground leading-relaxed mb-2">
                      {t('sessions.config.cachingHelpPrefixBody')}
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                      <li><strong>{t('sessions.config.memoryAwareDefault')}</strong> {t('sessions.config.memoryAwareDefaultBody')}</li>
                      <li><strong>{t('sessions.config.legacyEntryCount')}</strong> {t('sessions.config.legacyEntryCountBody')}</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">{t('sessions.config.mambaHybridCompat')}</h3>
                    <p className="text-muted-foreground leading-relaxed mb-2">
                      {t('sessions.config.cachingHelpMambaBody')}
                    </p>
                    <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                      <li><strong>{t('sessions.config.kvQuantizationLabel')}</strong> {t('sessions.config.kvQuantizationBody')}</li>
                      <li><strong>{t('sessions.config.inMemoryPagedCache')}</strong> {t('sessions.config.inMemoryPagedCacheBody')}</li>
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-foreground mb-2">{t('sessions.config.kvCacheQuantization')}</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {t('sessions.config.kvQuantHelpBody1')} <strong>{t('sessions.config.onlyCompressesSavedPrefixes')}</strong>. {t('sessions.config.kvQuantHelpBody2')}
                    </p>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold tracking-tight text-foreground mb-2">{t('sessions.config.visionLanguageModels')}</h3>
                    <p className="text-muted-foreground leading-relaxed">
                      {t('sessions.config.coreEngineHandlesVision')} <strong>{t('sessions.config.prefixCachingWorksForImages')}</strong> {t('sessions.config.vlHelpBody')}
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
        <PerformanceHint text={t('sessions.config.pagedCacheHint')} />
        {dsv4Active && <InfoNote text={t('sessions.config.dsv4PagedNote')} />}
        {zayaSsdReuseUnavailable && <IncompatWarning text={t('sessions.config.zayaTypedCacheNote')} />}
        {dsv4Active && cachePolicy.blockDiskCacheChecked && <InfoNote text={blockDiskOnly
          ? t('sessions.config.dsv4SsdOnlyNote')
          : t('sessions.config.dsv4RamL1Note')} />}
        {architectureBlockDiskOnlySupported && !m3Active && !dsv4Active && cachePolicy.blockDiskCacheChecked && <InfoNote text={mixedSwaBlockDiskOnlySupported
          ? stepMixedSwaBlockDiskOnly
            ? t('sessions.config.stepSsdOnlyNote')
            : t('sessions.config.mixedSwaSsdOnlyNote')
          : t('sessions.config.hybridSsdOnlyNote')} />}
        {m3Active && <InfoNote text={blockDiskOnly
          ? t('sessions.config.m3SsdOnlyNote')
          : t('sessions.config.m3NativeMsaNote')} />}
        {openPanguExactTypedCache && <InfoNote text={t('sessions.config.openPanguNoPagedNote')} />}
        <CheckField label={t('sessions.config.pagedKVCache')} tooltip={t('sessions.config.pagedKVCacheTooltip')} checked={effectiveUsePagedCache} onChange={v => applyCacheControlUpdates(cacheControlUpdatesForPagedToggle(v, cacheControlState))} disabled={genericPagedCacheToggleDisabled} />
        {(effectiveUsePagedCache || cachePolicy.blockDiskCacheChecked) && (
          <>
            <InfoNote text={blockDiskOnly
              ? effectiveBlockDiskCapacityText
              : effectivePagedCapacityText} />
            <SliderField
              label={t('sessions.config.blockSizeTokens')}
              tooltip={dsv4Active
                ? t('sessions.config.blockSizeTooltipDsv4')
                : t('sessions.config.blockSizeTooltip')}
              value={effectivePagedCacheBlockSize}
              onChange={v => onChange('pagedCacheBlockSize', v)}
              min={1}
              max={1024}
              step={16}
              defaultValue={dsv4Active ? DSV4_PAGED_CACHE_BLOCK_SIZE : DEFAULT_CONFIG.pagedCacheBlockSize}
              disabled={dsv4Active}
            />
            <SliderField
              label={t('sessions.config.maxCacheBlocks')}
              tooltip={t('sessions.config.maxCacheBlocksTooltip')}
              value={config.maxCacheBlocks}
              onChange={v => onChange('maxCacheBlocks', v)}
              min={2}
              max={100000}
              step={100}
              defaultValue={dsv4Active ? DSV4_MAX_CACHE_BLOCKS : DEFAULT_CONFIG.maxCacheBlocks}
              maxInput={100000}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel={t('sessions.config.defaultWithValue', { n: 1000 })}
            />
          </>
        )}
        {!batchingOff && !effectiveUsePagedCache && <InfoNote text={t('sessions.config.blockDiskPureSsdNote')} />}
        <CheckField
          label={t('sessions.cache.blockDiskCache')}
          tooltip={t('sessions.config.blockDiskCacheTooltip')}
          checked={cachePolicy.blockDiskCacheChecked}
          onChange={v => applyCacheControlUpdates(cacheControlUpdatesForBlockDiskToggle(v, cacheControlState))}
          disabled={!cachePolicy.blockDiskCacheVisible || cachePolicy.blockDiskCacheDisabled || exactTypedPromptDiskCache}
        />
        {cachePolicy.blockDiskCacheChecked && (
          <>
            <SliderField
              label={t('sessions.config.blockCacheMaxPercent')}
              tooltip={t('sessions.config.blockCacheMaxPercentTooltip')}
              value={config.blockDiskCacheMaxPercent}
              onChange={v => onChange('blockDiskCacheMaxPercent', v)}
              min={0}
              max={90}
              step={1}
              defaultValue={10}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel={t('sessions.cache.unlimited')}
            />
            {/* The engine trims ONE cache root shared by every session, so this
                budget is a total rather than a per-session allowance. Saying so
                here keeps the control honest: two sessions set to different
                percentages would otherwise look independent while contending
                over the same pool. */}
            <InfoNote text={t('sessions.config.blockCacheSharedBudgetNote')} />
            {/* The pool spans cache ARCHITECTURES, not just sessions: typed SSM
                companions, DSV4 composite records, rotating-SWA state and
                multimodal blocks are all trimmed out of the same root. Someone
                reading only the per-session note would reasonably assume their
                DSV4 session had its own budget. */}
            <InfoNote text={t('sessions.config.blockCacheArchitecturePoolNote')} />
            <div className="flex items-center gap-2 mt-1">
              <button
                type="button"
                onClick={handleClearSsdCache}
                disabled={clearingSsdCache}
                title={t('sessions.config.clearSsdCacheTooltip')}
                className="px-3 py-1.5 text-xs border border-destructive/50 text-destructive rounded hover:bg-destructive/10 disabled:opacity-50"
              >
                {clearingSsdCache
                  ? t('sessions.config.clearSsdCacheBusy')
                  : t('sessions.config.clearSsdCache')}
              </button>
              {ssdClearResult && (
                <span className="text-[11px] text-muted-foreground">{ssdClearResult}</span>
              )}
            </div>
            <div className="block">
              <span className="text-xs font-medium text-muted-foreground">
                {t('sessions.config.blockCacheDirectory')}
                <Tooltip text={t('sessions.config.blockCacheDirectoryTooltip')} />
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

      {/* Cache representation. Auto intentionally omits the CLI flag and the
          engine preserves model.make_cache() exactly: full KV, rotating/SWA,
          recurrent, sparse, or native compressed/typed composite state. q4/q8
          remain explicit diagnostic stored codecs where the architecture
          allows them; they are never silently selected by Auto. */}
      <Section title={t('sessions.config.kvCacheQuantization')} expanded={expandedSections.kvCacheQuant} onToggle={() => toggleSection('kvCacheQuant')} hidden={isImage}>
        {batchingOff && <IncompatWarning text={t('sessions.config.kvQuantRequiresBatching')} />}
        {!batchingOff && prefixOff && <IncompatWarning text={t('sessions.config.kvQuantRequiresPrefix')} />}
        {!effectivelyNoBatching && !prefixOff && mixedSwaCacheActive && <PerformanceHint text={t('sessions.config.mixedSwaAutoHint')} />}
        {!effectivelyNoBatching && !prefixOff && hy3Active && <PerformanceHint text={t('sessions.config.hy3AutoHint')} />}
        {!effectivelyNoBatching && !prefixOff && qwenHybridTqActive && !mixedSwaCacheActive && <PerformanceHint text={bonsaiActive ? t('sessions.config.bonsaiHybridHint') : t('sessions.config.qwenHybridHint')} />}
        {!effectivelyNoBatching && !prefixOff && qwenFullTqActive && <PerformanceHint text={t('sessions.config.qwenFullKvHint')} />}
        {!effectivelyNoBatching && !prefixOff && isMambaCache && !qwenHybridTqActive && !mixedSwaCacheActive && !dsv4Active && !m3Active && !openPanguExactTypedCache && <PerformanceHint text={t('sessions.config.hybridStatefulHint')} />}
        {!effectivelyNoBatching && dsv4Active && <PerformanceHint text={t('sessions.config.dsv4KvQuantHint')} />}
        {!effectivelyNoBatching && m3Active && <PerformanceHint text={t('sessions.config.m3KvQuantHint')} />}
        {!effectivelyNoBatching && openPanguExactTypedCache && <PerformanceHint text={t('sessions.config.openPanguKvQuantHint')} />}
        {/* Live/native cache representation — automatic per architecture. */}
        <div className="block">
          <span className="text-xs font-medium text-muted-foreground">
            {t('sessions.config.liveCacheCodec')}
            <Tooltip text={t('sessions.config.liveCacheCodecTooltip')} />
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
                <Tooltip text={t('sessions.config.nativePoolCodecTooltip')} />
              </span>
              <div className="cfg-input flex items-center justify-between" style={{ background: 'var(--card)', cursor: 'default' }}>
                <span>{t('sessions.config.dsv4PoolQuantization')}</span>
                <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--success-bg, rgba(34,197,94,0.15))', color: 'var(--success-fg, rgb(34,197,94))' }}>
                  {config.dsv4PoolQuant === true
                    ? t('sessions.config.poolQuantOnBundle')
                    : config.dsv4PoolQuant === false
                      ? t('sessions.config.poolQuantOffBundle')
                      : t('sessions.config.poolQuantEngineDefault')}
                </span>
              </div>
            </div>
            <CheckField
              label={t('sessions.config.dsv4ActivationQat')}
              tooltip={t('sessions.config.dsv4ActivationQatTooltip')}
              checked={config.dsv4ActivationQat === true}
              onChange={v => onChange('dsv4ActivationQat', v)}
            />
            <InfoNote text={t('sessions.config.dsv4ActivationQatNote')} />
          </>
        )}

        {/* Stored prefix representation. Auto adds no codec. */}
        <div className="block">
          <span className="text-xs font-medium text-muted-foreground">
            {t('sessions.config.storedCacheQuantization')}
            <Tooltip text={t('sessions.config.storedCacheQuantTooltip')} />
          </span>
          <select value={effectiveStoredCacheQuantization} className="cfg-input" disabled>
            <option value="auto">{dsv4Active ? t('sessions.config.storedQuantNativeTyped') : t('sessions.config.storedQuantAuto')}</option>
          </select>
        </div>
      </Section>

      {/* Disk Cache (L2 Persistent) */}
      <Section title={t('sessions.config.diskCachePersistent')} expanded={expandedSections.diskCache} onToggle={() => toggleSection('diskCache')} hidden={isImage}>
        {!effectivelyNoBatching && <PerformanceHint text={t('sessions.config.diskCacheHint')} />}
        {dsv4Active ? (
          <InfoNote text={t('sessions.config.dsv4LegacyDiskNote')} />
        ) : (
          <InfoNote text={t('sessions.config.legacyDiskNote')} />
        )}
        {openPanguExactTypedCache && <InfoNote text={t('sessions.config.openPanguDiskNote')} />}
        {batchingOff && <IncompatWarning text={t('sessions.config.diskCacheRequiresBatching')} />}
        {!effectivelyNoBatching && cachePolicy.legacyDiskCacheUnavailableReason === 'paged-cache-active' && <IncompatWarning text={t('sessions.config.legacyDiskPagedConflict')} />}
        {!effectivelyNoBatching && cachePolicy.legacyDiskCacheUnavailableReason === 'architecture-requires-paged-cache' && <IncompatWarning text={t('sessions.config.legacyDiskArchConflict')} />}
        {!batchingOff && prefixOff && !cachePolicy.legacyDiskCacheDisabled && <InfoNote text={t('sessions.config.diskCacheEnablesPrefix')} />}
        <CheckField
          label={t('sessions.config.enableDiskCache')}
          tooltip={t('sessions.config.enableDiskCacheTooltip')}
          checked={cachePolicy.legacyDiskCacheChecked}
          onChange={v => applyCacheControlUpdates(cacheControlUpdatesForDiskToggle(v, cacheControlState))}
          disabled={dsv4Active || cachePolicy.legacyDiskCacheDisabled}
        />
        {cachePolicy.legacyDiskCacheChecked && (
          <>
            <SliderField
              label={t('sessions.config.maxCacheSizeGb')}
              tooltip={t('sessions.config.maxCacheSizeTooltip')}
              value={config.diskCacheMaxGb}
              onChange={v => onChange('diskCacheMaxGb', v)}
              min={0}
              max={100}
              step={1}
              defaultValue={10}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel={t('sessions.cache.unlimited')}
            />
            <div className="block">
              <span className="text-xs font-medium text-muted-foreground">
                {t('sessions.config.cacheDirectory')}
                <Tooltip text={t('sessions.config.cacheDirectoryTooltip')} />
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
        <PerformanceHint text={t('sessions.config.powerManagementDesc')} />
        <Field label={t('sessions.config.autoSleep')} tooltip={t('sessions.config.autoSleepDesc')}>
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
              label={t('sessions.config.lightSleepAfter')}
              tooltip={t('sessions.config.lightSleepAfterTooltip')}
              value={config.idleTimeoutSoftMin ?? (isImage ? 5 : 10)}
              onChange={v => onChange('idleTimeoutSoftMin', v)}
              min={0}
              max={120}
              step={1}
              defaultValue={isImage ? 5 : 10}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel={t('sessions.config.rateLimitDisabled')}
            />
            <SliderField
              label={t('sessions.config.deepSleepAfter')}
              tooltip={t('sessions.config.deepSleepAfterTooltip')}
              value={config.idleTimeoutHardMin ?? (isImage ? 15 : 30)}
              onChange={v => onChange('idleTimeoutHardMin', v)}
              min={0}
              max={240}
              step={1}
              defaultValue={isImage ? 15 : 30}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel={t('sessions.config.rateLimitDisabled')}
            />
          </>
        )}
      </Section>

      {/* Performance */}
      <Section title={t('sessions.config.performanceGeneration')} expanded={expandedSections.performance} onToggle={() => toggleSection('performance')} hidden={isImage}>
        <PerformanceHint text={t('sessions.config.performanceHint')} />
        {/* Whole-model JIT is not available for path-dependent cache models. */}
        <Field label={t('sessions.config.modelWideJit')} tooltip={t('sessions.config.modelWideJitTooltip')}>
          <label className={`flex items-center gap-2 ${jitSuppressedByRuntime ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'}`}>
            <input
              type="checkbox"
              checked={computeEffectiveJit({
                enableJitRequested: !!config.enableJit,
                isMultimodal: multimodalActive,
                flashMoeActive,
                distributedActive,
                dsv4Active,
                m3Active,
                zayaCcaActive,
                turboQuantActive,
                lagunaMixedSwaTurboQuantActive,
                hybridCacheActive,
              })}
              onChange={e => onChange('enableJit', e.target.checked)}
              disabled={jitSuppressedByRuntime}
              className="rounded border-input"
            />
            <span className="text-xs text-muted-foreground">
              {t('sessions.config.fuseMetalOps')}
            </span>
          </label>
        </Field>
        {(jitSuppressedByRuntime) && (
          <IncompatWarning text={dsv4Active
            ? t('sessions.config.jitDisabledDsv4')
            : m3Active
            ? t('sessions.config.jitDisabledM3')
            : zayaCcaActive
            ? t('sessions.config.jitDisabledZaya')
            : multimodalActive
            ? t('sessions.config.jitDisabledMultimodal')
            : hybridCacheActive
            ? t('sessions.config.jitDisabledHybrid')
            : turboQuantActive
            ? t('sessions.config.jitDisabledTurboQuant')
            : lagunaMixedSwaTurboQuantActive
            ? t('sessions.config.jitDisabledLaguna')
            : flashMoeActive
            ? t('sessions.config.jitDisabledFlashMoe')
            : t('sessions.config.jitDisabledDistributed')} />
        )}
        {dsv4Active && (
          <PerformanceHint text={t('sessions.config.dsv4CompiledDecodeHint')} />
        )}

        <SliderField
          label={t('sessions.config.streamInterval')}
          tooltip={t('sessions.config.streamIntervalTooltip')}
          value={config.streamInterval}
          onChange={v => onChange('streamInterval', v)}
          min={1}
          max={100}
          step={1}
          defaultValue={DEFAULT_CONFIG.streamInterval}
        />
        <SliderField
          label={t('sessions.config.maxOutputTokens')}
          tooltip={t('sessions.config.maxOutputTokensTooltip')}
          value={config.maxTokens}
          onChange={v => onChange('maxTokens', v)}
          min={1}
          max={32768}
          step={256}
          defaultValue={(config.defaultMaxNewTokens ?? 0) > 0 ? Math.floor(config.defaultMaxNewTokens ?? 0) : 4096}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={(config.defaultMaxNewTokens ?? 0) > 0 ? t('sessions.config.bundleWithValue', { n: Math.floor(config.defaultMaxNewTokens ?? 0) }) : t('sessions.config.bundleEngineDefault')}
          maxInput={1000000}
        />
        <SliderField
          label={t('sessions.config.maxContextTokens')}
          tooltip={t('sessions.config.maxContextTokensTooltip')}
          value={config.maxContextLength}
          onChange={v => onChange('maxContextLength', v)}
          min={1}
          max={1000000}
          step={1024}
          defaultValue={detectedMaxContext && detectedMaxContext > 0 ? detectedMaxContext : DEFAULT_CONFIG.maxContextLength}
          allowUnlimited
          unlimitedValue={0}
          unlimitedLabel={detectedMaxContext && detectedMaxContext > 0 ? t('sessions.config.autoModelContext', { n: detectedMaxContext }) : t('sessions.config.autoMemorySafe')}
        />
        <InfoNote text={generationDefaultsSummary ? t('sessions.config.generationDefaultsNoteWithValues', { values: generationDefaultsSummary }) : t('sessions.config.generationDefaultsNote')} />
      </Section>

      {/* Tool Integration */}
      <Section title={t('sessions.config.toolIntegrationMCP')} expanded={expandedSections.tools} onToggle={() => toggleSection('tools')} hidden={isImage}>
        <PerformanceHint text={t('sessions.config.mcpHint')} />
        <Field label={t('sessions.config.mcpConfigFile')} tooltip={t('sessions.config.mcpConfigFileTooltip')}>
          <div className="flex gap-2">
            <input type="text" value={config.mcpConfig} onChange={e => onChange('mcpConfig', e.target.value)} placeholder={t('sessions.config.mcpConfigPlaceholder')} className="cfg-input flex-1" />
            <button type="button" onClick={browseMcpConfig} className="px-3 py-1.5 rounded border border-border text-sm hover:bg-accent">{t('common.browse')}</button>
            <button type="button" onClick={importMcpConfig} className="px-3 py-1.5 rounded border border-border text-sm hover:bg-accent" disabled={mcpImportLoading}>
              {mcpImportLoading ? t('sessions.config.importing') : t('chat.list.import')}
            </button>
            <button type="button" onClick={() => validateMcpConfig()} className="px-3 py-1.5 rounded border border-border text-sm hover:bg-accent" disabled={mcpValidationLoading}>
              {mcpValidationLoading ? t('sessions.config.validating') : t('sessions.config.validate')}
            </button>
          </div>
        </Field>
        {mcpValidation && (
          <div className="rounded border border-border/60 bg-background/60 px-2 py-1.5 text-xs">
            {mcpValidation.error ? (
              <span className="text-destructive">{mcpValidation.error}</span>
            ) : (
              <div className="space-y-1">
                <div className="text-muted-foreground">{t('sessions.config.mcpConfiguredServers', { n: mcpValidation.serverCount ?? mcpValidation.servers.length })}</div>
                {mcpValidation.servers.slice(0, 4).map(server => (
                  <div key={server.name} className="flex items-center justify-between gap-2">
                    <span className="font-medium">{server.name}</span>
                    <span className="text-muted-foreground">{server.transport || 'mcp'} · {server.enabled === false ? t('sessions.cache.statusDisabled') : t('sessions.cache.statusEnabled')}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        <Field label={t('sessions.config.enabledMcpServers')} tooltip={t('sessions.config.enabledMcpServersTooltip')}>
          <textarea value={config.mcpEnabledServers} onChange={e => onChange('mcpEnabledServers', e.target.value)} placeholder="filesystem,github" className="cfg-input" rows={2} />
        </Field>
        <Field label={t('sessions.config.disabledMcpServers')} tooltip={t('sessions.config.disabledMcpServersTooltip')}>
          <textarea value={config.mcpDisabledServers} onChange={e => onChange('mcpDisabledServers', e.target.value)} placeholder="browser_automation&#10;postgres_readonly" className="cfg-input" rows={2} />
        </Field>
        <Field label={t('sessions.config.enabledMcpTools')} tooltip={t('sessions.config.enabledMcpToolsTooltip')}>
          <textarea value={config.mcpEnabledTools} onChange={e => onChange('mcpEnabledTools', e.target.value)} placeholder="filesystem__read_file&#10;github__search_repositories" className="cfg-input" rows={3} />
        </Field>
        <Field label={t('sessions.config.disabledMcpTools')} tooltip={t('sessions.config.disabledMcpToolsTooltip')}>
          <textarea value={config.mcpDisabledTools} onChange={e => onChange('mcpDisabledTools', e.target.value)} placeholder="filesystem__write_file" className="cfg-input" rows={2} />
        </Field>
        {sessionId && (
          <div className="rounded border border-border bg-background/60 p-2 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-medium text-muted-foreground">{t('sessions.config.liveMcpStatus')}</span>
              <button type="button" onClick={refreshMcpStatus} className="text-xs px-2 py-1 rounded border border-border hover:bg-accent" disabled={mcpStatusLoading}>
                {mcpStatusLoading ? t('sessions.config.refreshing') : t('common.refresh')}
              </button>
            </div>
            {mcpStatus?.error && (
              <div className="text-xs text-destructive">{mcpStatus.error}</div>
            )}
            {(mcpStatus?.servers?.length || 0) > 0 && (
              <div className="space-y-1">
                <div className="text-[11px] text-muted-foreground">{t('sessions.config.mcpServers')}</div>
                {(mcpStatus?.servers || []).map(server => {
                  const allowListActive = policyServers.length > 0
                  const checked = !policyDisabledServers.includes(server.name) && (allowListActive ? policyServers.includes(server.name) : server.enabled !== false)
                  return (
                    <label key={server.name} className="flex items-center justify-between gap-2 rounded border border-border/60 px-2 py-1 text-xs">
                      <span className="min-w-0">
                        <span className="font-medium">{server.name}</span>
                        <span className="ml-2 text-muted-foreground">{server.transport || 'mcp'} · {server.state || t('sessions.performance.unknown')} · {t('sessions.config.mcpToolsCount', { n: server.tools_count ?? 0 })}</span>
                      </span>
                      <input type="checkbox" checked={checked} onChange={e => toggleMcpServer(server.name, e.target.checked)} />
                    </label>
                  )
                })}
              </div>
            )}
            {(mcpStatus?.tools?.length || 0) > 0 && (
              <div className="space-y-1">
                <div className="text-[11px] text-muted-foreground">{t('app.mode.tools')}</div>
                {(mcpStatus?.tools || []).map(tool => (
                  <label key={tool.name} className="grid grid-cols-[1fr_auto] gap-2 rounded border border-border/60 px-2 py-1 text-xs">
                    <span className="min-w-0">
                      <span className="font-medium break-all">{tool.name}</span>
                      <span className={`ml-2 ${tool.effective ? 'text-primary' : 'text-muted-foreground'}`}>
                        {tool.effective ? t('sessions.config.mcpEffective') : t('sessions.config.mcpBlocked')}
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
          label={t('sessions.config.automaticToolChoice')}
          tooltip={t('sessions.config.automaticToolChoiceTooltip')}
          value={config.enableAutoToolChoice === undefined ? 'auto' : config.enableAutoToolChoice ? 'on' : 'off'}
          onChange={value => onChange(
            'enableAutoToolChoice',
            value === 'auto' ? undefined : value === 'on',
          )}
          options={[
            {
              value: 'auto',
              label: t('sessions.config.autoDetectedWithValue', { value: detectedEnableAutoToolChoice ? t('chat.settings.thinkingOn') : t('chat.settings.thinkingOff') }),
            },
            { value: 'on', label: t('chat.settings.thinkingOn') },
            { value: 'off', label: t('chat.settings.thinkingOff') },
          ]}
        />
        {config.enableAutoToolChoice === undefined && (
          <InfoNote text={t('sessions.config.autoDetectCurrently', { value: detectedEnableAutoToolChoice ? t('chat.settings.thinkingOn') : t('chat.settings.thinkingOff') })} />
        )}
        <ParserField
          label={t('sessions.config.toolCallParser')}
          tooltip={t('sessions.config.toolCallParserTooltip')}
          noneLabel={t('sessions.config.parserNoneToolCalls')}
          value={config.toolCallParser === 'none' ? '' : canonicalizeToolParserId(config.toolCallParser) ?? 'auto'}
          onChange={v => onChange('toolCallParser', v)}
          options={TOOL_PARSER_OPTIONS}
          detectedValue={describeDetectedToolParser(detectedToolParser)}
        />
        <ParserField
          label={t('sessions.config.reasoningParser')}
          tooltip={t('sessions.config.reasoningParserTooltip')}
          noneLabel={t('sessions.config.parserNoneReasoning')}
          value={config.reasoningParser === 'none' ? '' : config.reasoningParser}
          onChange={v => onChange('reasoningParser', v)}
          options={REASONING_PARSER_OPTIONS}
          detectedValue={detectedReasoningParser}
        />
        <SelectField
          label={t('sessions.config.modelFamilyOverride')}
          tooltip={t('sessions.config.modelFamilyOverrideTooltip')}
          value={config.modelFamily ?? 'auto'}
          onChange={v => onChange('modelFamily', v === 'auto' ? undefined : v)}
          options={[
            { value: 'auto', label: t('sessions.config.autoDetectedWithValue', { value: detectedFamily ?? t('sessions.performance.unknown') }) },
            ...MODEL_FAMILY_OVERRIDE_NAMES.map(name => ({ value: name, label: name })),
          ]}
        />
        <Field label={t('sessions.config.customChatTemplate')} tooltip={t('sessions.config.customChatTemplateTooltip')}>
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
          label={t('sessions.config.multimodalSupport')}
          tooltip={t('sessions.config.multimodalSupportTooltip')}
          value={dsv4Active || smeltActive || detectedForceTextOnly ? 'off' : config.isMultimodal === true ? 'on' : config.isMultimodal === false ? 'off' : 'auto'}
          onChange={v => onChange('isMultimodal', v === 'on' ? true : v === 'off' ? false : undefined)}
          options={[
            { value: 'auto', label: t('sessions.config.autoDetectFromModel') },
            { value: 'on', label: t('sessions.config.forceOn') },
            { value: 'off', label: t('sessions.config.forceOff') },
          ]}
          disabled={dsv4Active || smeltActive || detectedForceTextOnly}
        />
        {dsv4Active && (
          <InfoNote text={t('sessions.config.dsv4TextRuntimeNote')} />
        )}
        {smeltActive && (
          <IncompatWarning text={t('sessions.config.vlmDisabledSmelt')} />
        )}
        {detectedForceTextOnly && (
          <IncompatWarning text={t('sessions.config.forceTextOnlyNote')} />
        )}
        {!dsv4Active && !smeltActive && !detectedForceTextOnly && config.isMultimodal === true && (
          <InfoNote text={t('sessions.config.vlmActiveNote')} />
        )}
        {!dsv4Active && !smeltActive && !detectedForceTextOnly && config.isMultimodal === false && (
          <InfoNote text={t('sessions.config.vlmOffNote')} />
        )}
        {omniBackendVisible && (
          <SelectField
            label={t('sessions.config.omniBackend')}
            tooltip={t('sessions.config.omniBackendTooltip')}
            value={config.omniBackend || 'stage1'}
            onChange={v => onChange('omniBackend', v as 'stage1' | 'stage2')}
            options={[
              { value: 'stage1', label: t('sessions.config.omniStage1') },
              { value: 'stage2', label: t('sessions.config.omniStage2') },
            ]}
          />
        )}
        {normalizedDetectedFamily === 'gemma4' && multimodalActive && (
          <SelectField
            label={t('sessions.config.imageTokenBudget')}
            tooltip={t('sessions.config.imageTokenBudgetTooltip')}
            value={String(config.imageTokenBudget ?? 280)}
            onChange={v => onChange('imageTokenBudget', Number(v))}
            options={[
              { value: '70', label: t('sessions.config.imageBudget70') },
              { value: '140', label: t('sessions.config.imageBudget140') },
              { value: '280', label: t('sessions.config.imageBudget280') },
              { value: '560', label: t('sessions.config.imageBudget560') },
              { value: '1120', label: t('sessions.config.imageBudget1120') },
            ]}
          />
        )}
        {/* Video sampling — only relevant for VL models that accept video_url.
            Qwen 3.6 / Qwen3.5-VL both have native video understanding via
            temporal position embeddings, so 2 fps × 8 frames is typical. */}
        {showVideoControls && (
          <>
            <SliderField
              settingKey="videoFps"
              label={t('sessions.config.videoFps')}
              tooltip={t('sessions.config.videoFpsTooltip')}
              value={config.videoFps ?? 2}
              onChange={v => onChange('videoFps', v)}
              min={1}
              max={8}
              step={1}
              defaultValue={2}
            />
            <SliderField
              settingKey="videoMaxFrames"
              label={t('sessions.config.maxVideoFrames')}
              tooltip={t('sessions.config.maxVideoFramesTooltip')}
              value={config.videoMaxFrames ?? 8}
              onChange={v => onChange('videoMaxFrames', v)}
              min={2}
              max={64}
              step={2}
              defaultValue={8}
            />
            <Field settingKey="videoMaxPixels" label={t('sessions.config.videoMaxPixels')} tooltip={t('sessions.config.videoMaxPixelsTooltip')}>
              <select
                className="cfg-input"
                data-vmlx-setting="videoMaxPixels"
                value={config.videoMaxPixels ? String(config.videoMaxPixels) : ''}
                onChange={e => onChange('videoMaxPixels', e.target.value ? Number(e.target.value) : undefined)}
              >
                <option value="">{t('sessions.config.videoMaxPixelsDefault')}</option>
                <option value="100352">128 px² · 100k ({t('sessions.config.videoMaxPixelsFastest')})</option>
                <option value="200704">256 px² · 200k</option>
                <option value="301056">301k</option>
                <option value="401408">401k</option>
                <option value="602112">602k ({t('sessions.config.videoMaxPixelsMax')})</option>
              </select>
            </Field>
            <Field settingKey="videoTokenBudget" label={t('sessions.config.videoTokenBudget')} tooltip={t('sessions.config.videoTokenBudgetTooltip')}>
              <select
                className="cfg-input"
                data-vmlx-setting="videoTokenBudget"
                value={config.videoTokenBudget ? String(config.videoTokenBudget) : ''}
                onChange={e => onChange('videoTokenBudget', e.target.value ? Number(e.target.value) : undefined)}
              >
                <option value="">{t('sessions.config.videoMaxPixelsDefault')}</option>
                {[256, 512, 1024, 2048, 4096, 8192, 16384].map(n => (
                  <option key={n} value={String(n)}>{n}</option>
                ))}
              </select>
            </Field>
          </>
        )}
      </Section>

      {/* Native in-model MTP */}
      <Section title={t('sessions.config.nativeMtp')} expanded={expandedSections.nativeMtp} onToggle={() => toggleSection('nativeMtp')} hidden={isImage || dsv4Active || !nativeMtpDetected}>
        {!nativeMtpSupported && (
          <IncompatWarning text={detectedNativeMtp?.blockedReason || t('sessions.config.nativeMtpBlockedFallback')} />
        )}
        {nativeMtpSupported && (
          <>
        <PerformanceHint text={t('sessions.config.nativeMtpHint')} />
        {nativeMtpMode === 'auto' && detectedNativeMtp?.defaultMode === 'off' && (
          <InfoNote text={t('sessions.config.nativeMtpDefaultOffNote')} />
        )}
        {nativeMtpMode === 'auto' && detectedNativeMtp?.defaultMode !== 'off' && (
          <InfoNote text={t('sessions.config.nativeMtpAutoNote')} />
        )}
        {nativeMtpMode === 'deterministic' && (
          <InfoNote text={t('sessions.config.nativeMtpDeterministicNote', { depth: nativeMtpDepth })} />
        )}
        <SelectField
          label={t('sessions.config.nativeMtpMode')}
          tooltip={t('sessions.config.nativeMtpModeTooltip')}
          value={nativeMtpMode}
          onChange={v => onChange('nativeMtpMode', v as 'deterministic' | 'auto' | 'off')}
          options={[
            { value: 'auto', label: t('sessions.config.mtpAutoBundleDefaults') },
            { value: 'deterministic', label: t('sessions.config.mtpDeterministicOverride') },
            { value: 'off', label: t('chat.settings.thinkingOff') },
          ]}
        />
        <SelectField
          label={t('sessions.config.nativeMtpDepthPolicy')}
          tooltip={t('sessions.config.nativeMtpDepthPolicyTooltip')}
          value={nativeMtpDepthPolicy}
          onChange={v => {
            const fixed = v === 'fixed'
            if (fixed) onChange('nativeMtpDepth', nativeMtpDepth)
            onChange('nativeMtpDepthOverride', fixed)
          }}
          options={[
            { value: 'adaptive', label: t('sessions.config.nativeMtpDepthAdaptive') },
            { value: 'fixed', label: t('sessions.config.nativeMtpDepthFixed') },
          ]}
          disabled={nativeMtpMode === 'off'}
        />
        <SliderField
          label={t('sessions.config.nativeMtpDepth')}
          tooltip={t('sessions.config.nativeMtpDepthTooltip')}
          value={nativeMtpDepth}
          onChange={v => {
            onChange('nativeMtpDepth', v)
            onChange('nativeMtpDepthOverride', true)
          }}
          min={1}
          max={3}
          step={1}
          defaultValue={3}
          disabled={nativeMtpMode === 'off' || nativeMtpDepthPolicy !== 'fixed'}
        />
        <InfoNote text={nativeMtpDepthPolicy === 'fixed'
          ? t('sessions.config.nativeMtpDepthFixedNote', { depth: nativeMtpDepth })
          : t('sessions.config.nativeMtpDepthAdaptiveNote', { depth: nativeMtpDepth })}
        />
        <InfoNote text={t('sessions.config.nativeMtpDetectedNote', { scope: detectedNativeMtp?.runtimeScope || 'text', cache: detectedNativeMtp?.nativeCacheType || detectedCacheSubtype || detectedCacheType || 'unknown', depthSource: detectedNativeMtp?.depthSource || 'default' })} />
          </>
        )}
      </Section>

      {/* Speculative Decoding */}
      <Section title={t('sessions.config.specDecoding')} expanded={expandedSections.specDecode} onToggle={() => toggleSection('specDecode')} hidden={isImage || dsv4Active}>
        <PerformanceHint text={t('sessions.config.specDecodeHint')} />
        {config.continuousBatching && !dflash2Speculative && !multimodalActive && <IncompatWarning text={t('sessions.config.specDecodeIncompatBatching')} />}
        {multimodalActive && config.speculativeModel && !dflash2Speculative && <IncompatWarning text={t('sessions.config.specDecodeIncompatVlm')} />}
        <Field label={t('sessions.config.draftModel')} tooltip={t('sessions.config.draftModelTooltip')}>
          <input type="text" value={config.speculativeModel} onChange={e => onChange('speculativeModel', e.target.value)} placeholder={t('sessions.config.specModelPlaceholder')} className="cfg-input" disabled={dsv4Active || (!multimodalActive && config.continuousBatching)} />
        </Field>
        {config.speculativeModel && (
          <SliderField
            label={t('sessions.config.draftTokensPerStep')}
            tooltip={t('sessions.config.draftTokensPerStepTooltip')}
            value={config.numDraftTokens}
            onChange={v => onChange('numDraftTokens', v)}
            min={1}
            max={20}
            step={1}
            defaultValue={DEFAULT_CONFIG.numDraftTokens}
            disabled={dsv4Active || (!dflash2Speculative && (config.continuousBatching || multimodalActive))}
          />
        )}
      </Section>

      {/* Distributed Compute */}
      <Section title={t('sessions.config.distributed')} expanded={expandedSections.distributed} onToggle={() => toggleSection('distributed')} hidden={isImage || dsv4Active}>
        <div className="mx-4 mt-3 mb-2 rounded-md border-2 border-amber-500 bg-amber-500/15 px-3 py-3 text-xs text-amber-800 dark:text-amber-100">
          <div className="font-bold uppercase tracking-wide text-[11px] mb-1.5 text-amber-900 dark:text-amber-50">
            {t('sessions.config.preAlphaHeader')}
          </div>
          <div className="leading-relaxed text-amber-900/90 dark:text-amber-100/90 space-y-1.5">
            <p>
              <strong>{t('sessions.config.preAlphaWarnBody1')}</strong>
            </p>
            <p>{t('sessions.config.preAlphaWarnBody2')}</p>
            <p>{t('sessions.config.preAlphaUsage')}</p>
          </div>
        </div>
        <PerformanceHint text={t('sessions.config.distributedHint')} />
        <CheckField
          label={t('sessions.config.enableDistributed')}
          tooltip={t('sessions.config.enableDistributedTooltip')}
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
          <IncompatWarning text={t('sessions.config.distributedDisabledFlashMoe')} />
        )}
        {config.distributedEnabled && (
          <>
            <SelectField
              label={t('sessions.config.parallelismMode')}
              tooltip={t('sessions.config.parallelismModeTooltip')}
              value={config.distributedMode || 'pipeline'}
              onChange={v => onChange('distributedMode', v as 'pipeline' | 'tensor')}
              options={[
                { value: 'pipeline', label: t('sessions.config.pipelineParallelism') },
                { value: 'tensor', label: t('sessions.config.tensorParallelism') },
              ]}
            />
            {config.distributedMode === 'tensor' && (
              <IncompatWarning text={t('sessions.config.tensorNotImplemented')} />
            )}
            <Field label={t('sessions.config.clusterSecret')} tooltip={t('sessions.config.clusterSecretTooltip')}>
              <input
                type="password"
                value={config.distributedSecret || ''}
                onChange={e => onChange('distributedSecret', e.target.value)}
                placeholder={t('sessions.config.clusterSecretPlaceholder')}
                className="cfg-input"
              />
            </Field>
            <InfoNote text={t('sessions.config.workerNodesNote')} />
            <DistributedNodeList enabled={!!config.distributedEnabled} sessionId={sessionId} />
            <div className="px-4 py-3 space-y-2">
              <div className="text-xs font-medium text-foreground">{t('sessions.config.setupGuide')}</div>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>{t('sessions.config.setupStep1')}</p>
                <p>{t('sessions.config.setupStep2')} <code className="bg-muted px-1 rounded">pip install vmlx && vmlx-worker --secret YOUR_SECRET</code></p>
                <p>{t('sessions.config.setupStep3')}</p>
                <p>{t('sessions.config.setupStep4')}</p>
                <p className="text-muted-foreground/70 pt-1">{t('sessions.config.setupNetworkNote')}</p>
              </div>
            </div>
          </>
        )}
      </Section>

      {/* Embedding Model */}
      {!isImage && (
      <div className="mb-2">
        <Field label={t('sessions.config.embeddingModel')} tooltip={t('sessions.config.embeddingModelTooltip')}>
          <input type="text" value={config.embeddingModel} onChange={e => onChange('embeddingModel', e.target.value)} placeholder={t('sessions.config.embeddingPlaceholder')} className="cfg-input" />
        </Field>
      </div>
      )}

      {/* Additional */}
      <div className="mb-4">
        <Field label={t('sessions.config.additionalArgs')} tooltip={t('sessions.config.additionalArgsTooltip')}>
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
    value: 'glm_xml_args', label: 'GLM-5.3 — arg_key/arg_value XML (glm47 dialect)', format: '<tool_call>name\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>', models: [
      'GLM-5.3-Flash (JANG)',
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
    value: 'dots', label: 'dots3-note — dots XML', format: '<dots_function_call><invoke name="fn"><parameter name="arg">\nval\n</parameter></invoke></dots_function_call>', models: [
      'dots3-note (280B omni MoE, text+vision+video+audio)',
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
    value: 'glm_think_block', label: 'GLM-5.3 — <think> rail (R1 contract)', format: '<think>...reasoning...</think>content  (template force-opens <think>)', models: [
      'GLM-5.3-Flash (JANG)',
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
    value: 'dots3', label: 'dots3-note — plain think rail', format: '<think>...reasoning...</think>content (thinking ON by default; off = <no_think> + closed-block prefill)', models: [
      'dots3-note (280B omni MoE)',
    ]
  },
  {
    value: 'think_xml', label: 'Think XML — MiMo XML reasoning', format: '<think>...reasoning...</think>content  (XML reasoning blocks)', models: [
      'MiMo V2.5 JANG 2L',
      'Use only when model metadata selects think_xml; MiMo generation quality remains separately gated.',
    ]
  },
]

function ParserField({ label, tooltip, value, onChange, options, detectedValue, noneLabel }: {
  label: string; tooltip: string; value: string; onChange: (v: string) => void; options: ParserOption[]; detectedValue?: string; noneLabel?: string
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
          <option key={o.value} value={o.value}>{o.value === 'auto' ? (detectedValue ? t('sessions.config.autoDetectedWithValue', { value: detectedValue }) : t('sessions.config.parserAutoRecommended')) : o.value === '' && noneLabel ? noneLabel : o.label}</option>
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
            {t('sessions.config.parserFinetunesNote')}
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

export function Section({ title, expanded, onToggle, children, hidden, sectionKey }: {
  title: string; expanded: boolean; onToggle: () => void; children: React.ReactNode; hidden?: boolean
  /** Stable, untranslated identity: data-vmlx-section on the wrapper, data-vmlx-state open/closed on the toggle. */
  sectionKey?: string
}) {
  if (hidden) return null
  return (
    <div className="mb-3 border border-border rounded" data-vmlx-section={sectionKey}>
      <button onClick={onToggle} data-vmlx-control={sectionKey ? `section-${sectionKey}` : undefined} data-vmlx-state={expanded ? 'open' : 'closed'} className="w-full flex items-center gap-2 px-3 py-2 text-sm font-medium hover:bg-accent rounded-t">
        <span className={`transition-transform ${expanded ? 'rotate-90' : ''}`}>&#9654;</span>
        {title}
      </button>
      {expanded && <div className="px-3 pb-3 space-y-3">{children}</div>}
    </div>
  )
}

export function Field({ label, tooltip, children, settingKey }: { label: string; tooltip?: string; children: React.ReactNode
  /** Stable, untranslated identity: data-vmlx-setting on the field wrapper. */
  settingKey?: string }) {
  return (
    <label className="block" data-vmlx-setting={settingKey}>
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

export function SelectField({ label, tooltip, value, onChange, options, disabled, settingKey }: {
  label: string; tooltip?: string; value: string; onChange: (v: string) => void
  options: { value: string; label: string }[]; disabled?: boolean; settingKey?: string
}) {
  return (
    <Field label={label} tooltip={tooltip} settingKey={settingKey}>
      <select value={value} onChange={e => onChange(e.target.value)} disabled={disabled} className="cfg-input" data-vmlx-control={settingKey ? `setting-${settingKey}` : undefined}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </Field>
  )
}

interface SliderFieldProps {
  /** Stable, untranslated identity: data-vmlx-setting on the field wrapper. */
  settingKey?: string
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
  disabled = false, maxInput,
  settingKey
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
      data-vmlx-setting={settingKey} data-setting-label={label}
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
