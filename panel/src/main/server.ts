/**
 * Shared types for vMLX Engine server configuration and process detection.
 * The actual multi-instance management lives in sessions.ts (SessionManager).
 */

export interface ServerConfig {
  modelPath: string

  // Model type — auto-detected from directory structure
  modelType?: 'text' | 'image'

  // Image-specific settings (stored in config, passed as CLI flags)
  imageMode?: 'generate' | 'edit'
  imageQuantize?: number
  mfluxClass?: string

  // Server settings
  host: string
  port: number
  apiKey?: string
  rateLimit?: number
  timeout: number

  // Concurrent processing
  maxNumSeqs?: number
  prefillBatchSize?: number
  prefillStepSize?: number
  completionBatchSize?: number
  continuousBatching: boolean

  // Prefix cache
  enablePrefixCache?: boolean
  prefixCacheSize?: number
  prefixCacheMaxBytes?: number
  cacheMemoryMb?: number
  cacheMemoryPercent?: number
  noMemoryAwareCache?: boolean

  // Paged cache
  usePagedCache: boolean
  pagedCacheBlockSize: number
  maxCacheBlocks: number

  // KV cache quantization
  kvCacheQuantization?: string
  kvCacheGroupSize?: number
  cacheStackStartupDefaultsVersion?: number
  modelParserDefaultsVersion?: number

  // Disk cache (L2 persistent cache)
  enableDiskCache?: boolean
  diskCacheDir?: string
  diskCacheMaxGb?: number

  // Block-level disk cache (L2 for paged cache blocks)
  enableBlockDiskCache?: boolean
  blockDiskCacheDir?: string
  blockDiskCacheMaxGb?: number
  /** Block-disk budget as a percent of the cache volume. The engine trims ONE
   *  shared cache root, so this is a total across sessions, not per-session. */
  blockDiskCacheMaxPercent?: number

  // MoE runtime modes
  smelt?: boolean
  smeltExperts?: number
  flashMoe?: boolean
  flashMoeSlotBank?: number
  flashMoePrefetch?: 'none' | 'temporal'
  flashMoeIoSplit?: number

  // Performance
  streamInterval: number
  maxTokens?: number
  defaultMaxNewTokens?: number
  defaultDoSample?: boolean
  defaultTopK?: number
  defaultMinP?: number
  defaultSamplingDefaultsDeclared?: boolean

  // Tool integration
  mcpConfig?: string
  mcpEnabledServers?: string
  mcpDisabledServers?: string
  mcpEnabledTools?: string
  mcpDisabledTools?: string
  enableAutoToolChoice?: boolean
  toolCallParser?: string

  // Reasoning
  reasoningParser?: string

  // Manual model-family override (--model-family). undefined = autodetect.
  modelFamily?: string

  // DSV4 Flash runtime env controls
  dsv4PrefixCache?: boolean
  dsv4PoolQuant?: boolean
  dsv4ActivationQat?: boolean

  // Custom API model name (--served-model-name)
  servedModelName?: string

  // Multimodal (VLM)
  isMultimodal?: boolean
  omniBackend?: 'stage1' | 'stage2'
  imageTokenBudget?: number
  videoFps?: number
  videoMaxFrames?: number

  // Cache TTL
  cacheTtlMinutes?: number

  // Speculative decoding
  speculativeModel?: string
  /** Use a DFlash2 drafter shipped inside the model bundle (default true). */
  useBundledDflash2?: boolean
  numDraftTokens?: number

  // Native in-model MTP: 'adaptive' (engine-governed speculation) or 'off'
  // (plain AR). 'auto'/'deterministic' are legacy persisted spellings that
  // normalize to 'adaptive'; legacy nativeMtpDepth/Override/AutoSamplingPolicy
  // keys in old rows are ignored.
  nativeMtpMode?: 'adaptive' | 'off' | 'auto' | 'deterministic'
  /** Separate choice: old native-MTP defaults did not opt into Bolt's drafter. */
  lilicorrMode?: 'adaptive' | 'off'

  // Generation defaults
  defaultTemperature?: number
  defaultTopP?: number
  defaultRepetitionPenalty?: number

  // Embedding model (separate from main model)
  embeddingModel?: string

  // Additional
  additionalArgs?: string

  // Display-only legacy field; session startup does not pass enable_thinking.
  // Chat/API requests carry explicit values and the engine resolves model defaults.
  defaultEnableThinking?: boolean

  // JIT compilation (mx.compile)
  enableJit?: boolean

  // Logging
  logLevel?: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'

  // CORS — comma-separated allowed origins (default '*' = all)
  corsOrigins?: string

  // Max context length override (0 = use model default)
  maxContextLength?: number

  // Custom template / distributed / sleep controls
  chatTemplate?: string
  distributedEnabled?: boolean
  distributedMode?: 'pipeline' | 'tensor'
  distributedSecret?: string
  distributedNodes?: Array<{ address: string; port: number; hostname?: string }>
  idleTimeoutSoftMin?: number
  idleTimeoutHardMin?: number
  autoSleepEnabled?: boolean
}

export interface DetectedProcess {
  pid: number
  port: number
  modelPath: string
  healthy: boolean
  modelName?: string
  nativeMtpSamplingPolicy?: 'compatible-only' | 'deterministic-defaults' | 'greedy-only' | 'disabled'
  nativeMtpDepthPolicy?: 'fixed' | 'adaptive'
  nativeMtpDepth?: number
  nativeMtpDisabled?: boolean
  standbyDepth?: 'soft' | 'deep' | null  // null = running, 'soft'/'deep' = in standby
}
