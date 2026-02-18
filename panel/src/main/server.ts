/**
 * Shared types for vLLM-MLX server configuration and process detection.
 * The actual multi-instance management lives in sessions.ts (SessionManager).
 */

export interface ServerConfig {
  modelPath: string

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

  // Performance
  streamInterval: number
  maxTokens?: number

  // Tool integration
  mcpConfig?: string
  enableAutoToolChoice?: boolean
  toolCallParser?: string

  // Reasoning
  reasoningParser?: string

  // Additional
  additionalArgs?: string

  // Legacy
  logLevel?: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
}

export interface DetectedProcess {
  pid: number
  port: number
  modelPath: string
  healthy: boolean
  modelName?: string
}
