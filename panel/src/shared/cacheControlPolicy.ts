export type CacheControlKey =
  | 'enablePrefixCache'
  | 'usePagedCache'
  | 'enableDiskCache'
  | 'enableBlockDiskCache'
  | 'dsv4PrefixCache'
  | 'dsv4PoolQuant'

export type CacheControlUpdate = [CacheControlKey, boolean]

export interface CacheControlState {
  continuousBatching: boolean
  enablePrefixCache: boolean
  usePagedCache: boolean
  enableDiskCache: boolean
  enableBlockDiskCache: boolean
  architectureRequiresPagedCache?: boolean
}

export interface CacheControlPolicy {
  batchingOff: boolean
  prefixOff: boolean
  architectureRequiresPagedCache: boolean
  architectureForcedPagedActive: boolean
  userPagedCacheActive: boolean
  effectiveUsePagedCache: boolean
  pagedCacheDisabled: boolean
  legacyDiskCacheDisabled: boolean
  blockDiskCacheVisible: boolean
  blockDiskCacheDisabled: boolean
  legacyDiskCacheChecked: boolean
  blockDiskCacheChecked: boolean
  legacyDiskCacheUnavailableReason?: 'batching-off' | 'paged-cache-active' | 'architecture-requires-paged-cache'
}

export interface CacheLaunchPolicy {
  prefixCacheOff: boolean
  effectiveUsePagedCache: boolean
  enableLegacyDiskCache: boolean
  enableBlockDiskCache: boolean
}

export function resolveCacheControlPolicy(state: CacheControlState): CacheControlPolicy {
  const batchingOff = !state.continuousBatching
  const prefixOff = !state.enablePrefixCache
  const architectureRequiresPagedCache = !!state.architectureRequiresPagedCache
  const architectureForcedPagedActive = architectureRequiresPagedCache && !batchingOff && !prefixOff
  const userPagedCacheActive = !batchingOff && !prefixOff && !!state.usePagedCache
  const effectiveUsePagedCache = architectureForcedPagedActive || userPagedCacheActive
  const pagedCacheDisabled = batchingOff || architectureForcedPagedActive
  const legacyDiskCacheDisabled = batchingOff || effectiveUsePagedCache || (!batchingOff && architectureRequiresPagedCache)
  const blockDiskCacheVisible = !batchingOff
  const blockDiskCacheDisabled = batchingOff
  const legacyDiskCacheChecked = !!state.enableDiskCache && !legacyDiskCacheDisabled && !prefixOff
  const blockDiskCacheChecked = !!state.enableBlockDiskCache && !batchingOff && !prefixOff && effectiveUsePagedCache
  const legacyDiskCacheUnavailableReason = batchingOff
    ? 'batching-off'
    : architectureRequiresPagedCache
      ? 'architecture-requires-paged-cache'
      : effectiveUsePagedCache
        ? 'paged-cache-active'
        : undefined

  return {
    batchingOff,
    prefixOff,
    architectureRequiresPagedCache,
    architectureForcedPagedActive,
    userPagedCacheActive,
    effectiveUsePagedCache,
    pagedCacheDisabled,
    legacyDiskCacheDisabled,
    blockDiskCacheVisible,
    blockDiskCacheDisabled,
    legacyDiskCacheChecked,
    blockDiskCacheChecked,
    legacyDiskCacheUnavailableReason,
  }
}

export function resolveCacheLaunchPolicy(state: CacheControlState): CacheLaunchPolicy {
  const batchingOff = !state.continuousBatching
  const architectureRequiresPagedCache = !!state.architectureRequiresPagedCache
  const prefixEnabled = !batchingOff && !!state.enablePrefixCache
  const architectureForcedPagedActive = architectureRequiresPagedCache && prefixEnabled
  const effectiveUsePagedCache = prefixEnabled && (
    architectureForcedPagedActive ||
    !!state.usePagedCache
  )

  return {
    prefixCacheOff: !prefixEnabled,
    effectiveUsePagedCache,
    enableLegacyDiskCache: !!state.enableDiskCache && prefixEnabled && !effectiveUsePagedCache,
    enableBlockDiskCache: !!state.enableBlockDiskCache && prefixEnabled && effectiveUsePagedCache,
  }
}

export function cacheControlUpdatesForPagedToggle(enabled: boolean, state: CacheControlState): CacheControlUpdate[] {
  const updates: CacheControlUpdate[] = []
  if (enabled && !state.enablePrefixCache) updates.push(['enablePrefixCache', true])
  if (enabled && state.enableDiskCache) updates.push(['enableDiskCache', false])
  if (!enabled && state.enableBlockDiskCache) updates.push(['enableBlockDiskCache', false])
  updates.push(['usePagedCache', enabled])
  return updates
}

export function cacheControlUpdatesForDiskToggle(enabled: boolean, state: CacheControlState): CacheControlUpdate[] {
  const updates: CacheControlUpdate[] = []
  if (enabled && !state.enablePrefixCache) updates.push(['enablePrefixCache', true])
  if (enabled && state.usePagedCache) updates.push(['usePagedCache', false])
  if (enabled && state.enableBlockDiskCache) updates.push(['enableBlockDiskCache', false])
  updates.push(['enableDiskCache', enabled])
  return updates
}

export function cacheControlUpdatesForBlockDiskToggle(enabled: boolean, state: CacheControlState): CacheControlUpdate[] {
  const updates: CacheControlUpdate[] = []
  if (enabled && !state.enablePrefixCache) updates.push(['enablePrefixCache', true])
  if (enabled && !state.usePagedCache) updates.push(['usePagedCache', true])
  if (enabled && state.enableDiskCache) updates.push(['enableDiskCache', false])
  updates.push(['enableBlockDiskCache', enabled])
  return updates
}

export function cacheControlUpdatesForDsv4CompositeToggle(enabled: boolean): CacheControlUpdate[] {
  const updates: CacheControlUpdate[] = [
    ['dsv4PrefixCache', enabled],
    ['enablePrefixCache', enabled],
    ['usePagedCache', enabled],
    ['enableBlockDiskCache', enabled],
  ]
  if (!enabled) updates.splice(1, 0, ['dsv4PoolQuant', false])
  return updates
}

export function cacheControlUpdatesForDsv4BlockDiskToggle(enabled: boolean): CacheControlUpdate[] {
  if (!enabled) return [['enableBlockDiskCache', false]]
  return [
    ['dsv4PrefixCache', true],
    ['enablePrefixCache', true],
    ['usePagedCache', true],
    ['enableDiskCache', false],
    ['enableBlockDiskCache', true],
  ]
}

export function cacheControlUpdatesForDsv4PoolQuantToggle(enabled: boolean): CacheControlUpdate[] {
  if (!enabled) return [['dsv4PoolQuant', false]]
  return [
    ['dsv4PrefixCache', true],
    ['enablePrefixCache', true],
    ['usePagedCache', true],
    ['enableBlockDiskCache', true],
    ['dsv4PoolQuant', true],
  ]
}
