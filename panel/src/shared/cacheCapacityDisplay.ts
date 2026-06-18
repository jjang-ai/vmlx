export interface PagedCacheCapacityInput {
  blockSize: number
  maxBlocks: number
  defaultBlockSize?: number
  defaultMaxBlocks?: number
}

export interface PagedCacheControlsState {
  memoryBudgetControlsDisabled: boolean
  cacheTtlDisabled: boolean
  memoryBudgetIgnored: boolean
}

const DEFAULT_PAGED_BLOCK_SIZE = 64
const DEFAULT_MAX_CACHE_BLOCKS = 1000

function finitePositiveInteger(value: number | undefined, fallback: number): number {
  if (Number.isFinite(value) && Math.floor(value as number) > 0) {
    return Math.floor(value as number)
  }
  return fallback
}

function formatInteger(value: number): string {
  return Math.floor(value).toLocaleString('en-US')
}

export const pagedCacheMemoryIgnoredText =
  'Cache Memory Limit, Cache Memory %, and Cache TTL are ignored while paged cache is active. Use Max Cache Blocks and Block Size for in-RAM paged capacity.'

export function resolvePagedCacheCapacity(input: PagedCacheCapacityInput): {
  blockSize: number
  maxBlocks: number
  capacityTokens: number
} {
  const blockSize = finitePositiveInteger(
    input.blockSize,
    finitePositiveInteger(input.defaultBlockSize, DEFAULT_PAGED_BLOCK_SIZE),
  )
  const maxBlocks = finitePositiveInteger(
    input.maxBlocks,
    finitePositiveInteger(input.defaultMaxBlocks, DEFAULT_MAX_CACHE_BLOCKS),
  )
  return {
    blockSize,
    maxBlocks,
    capacityTokens: blockSize * maxBlocks,
  }
}

export function pagedCacheCapacityText(input: PagedCacheCapacityInput): string {
  const resolved = resolvePagedCacheCapacity(input)
  return `Effective paged capacity: ${resolved.blockSize} tokens/block x ${resolved.maxBlocks} blocks = ${formatInteger(resolved.capacityTokens)} tokens`
}

export function pagedCacheControlsState(effectiveUsePagedCache: boolean): PagedCacheControlsState {
  return {
    memoryBudgetControlsDisabled: effectiveUsePagedCache,
    cacheTtlDisabled: effectiveUsePagedCache,
    memoryBudgetIgnored: effectiveUsePagedCache,
  }
}
