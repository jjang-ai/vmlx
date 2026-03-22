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
  completionBatchSize?: number
  continuousBatching: boolean

  // Prefix cache
  enablePrefixCache?: boolean
  prefixCacheSize?: number
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

  // Disk cache (L2 persistent cache)
  enableDiskCache?: boolean
  diskCacheDir?: string
  diskCacheMaxGb?: number

  // Block-level disk cache (L2 for paged cache blocks)
  enableBlockDiskCache?: boolean
  blockDiskCacheDir?: string
  blockDiskCacheMaxGb?: number

  // SSD disk-streaming mode (per-layer weight recycling for models > RAM)
  streamFromDisk?: boolean
  streamMemoryPercent?: number
  ssdMemoryBudget?: number
  ssdPrefetchLayers?: number

  // Performance
  streamInterval: number
  maxTokens?: number

  // Tool integration
  mcpConfig?: string
  enableAutoToolChoice?: boolean
  toolCallParser?: string

  // Reasoning
  reasoningParser?: string

  // Custom API model name (--served-model-name)
  servedModelName?: string

  // Multimodal (VLM)
  isMultimodal?: boolean

  // Cache TTL
  cacheTtlMinutes?: number

  // Speculative decoding
  speculativeModel?: string
  numDraftTokens?: number

  // Generation defaults
  defaultTemperature?: number
  defaultTopP?: number

  // Embedding model (separate from main model)
  embeddingModel?: string

  // Additional
  additionalArgs?: string

  // Server-level default for enable_thinking (from model_settings.reasoning_mode)
  defaultEnableThinking?: boolean

  // JIT compilation (mx.compile)
  enableJit?: boolean

  // Logging
  logLevel?: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'

  // CORS — comma-separated allowed origins (default '*' = all)
  corsOrigins?: string

  // Max context length override (0 = use model default)
  maxContextLength?: number
}

export interface DetectedProcess {
  pid: number
  port: number
  modelPath: string
  healthy: boolean
  modelName?: string
  standbyDepth?: 'soft' | 'deep' | null  // null = running, 'soft'/'deep' = in standby
}
