export type CacheControlKey =
  | 'enablePrefixCache'
  | 'usePagedCache'
  | 'enableDiskCache'
  | 'enableBlockDiskCache'

export type CacheControlUpdate = [CacheControlKey, boolean]

export interface CacheControlState {
  continuousBatching: boolean
  /** DFlash2 owns an SSD store independently of the batched prefix stack. */
  nativeSpeculativeCache?: boolean
  enablePrefixCache: boolean
  usePagedCache: boolean
  enableDiskCache: boolean
  enableBlockDiskCache: boolean
  architectureRequiresPagedCache?: boolean
  /**
   * The architecture has non-KV companion state, but the engine can pair the
   * disk-only KV block index with a typed companion SSD store/rederive path.
   * This is true for hybrid/Mamba SSM caches and native rotating/mixed-SWA
   * caches whose typed metadata is carried in each block record. This includes
   * DSV4, M3, and hybrid/Mamba cache contracts with a complete
   * typed block-disk representation. It remains false for ZAYA/openPangu contracts
   * that still require their architecture-specific in-memory or prompt-level
   * cache lane.
   */
  architectureSupportsBlockDiskOnly?: boolean
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
  // Three user-chosen tiers, never chosen by a family or a registry capability:
  //   RAM only  : Prefix Cache on, Block Disk Cache off (memory-aware prefix cache);
  //   SSD only  : Block Disk Cache on, RAM copy off (the default);
  //   RAM + SSD : Block Disk Cache on + "Keep a RAM copy" (paged RAM blocks in front of the SSD blocks).
  // The RAM copy is a front for the block store, so it is live only with Block Disk Cache (and never for the
  // native speculative lanes, which own their SSD store).
  const architectureRequiresPagedCache = false
  const architectureForcedPagedActive = false
  const blockDiskCacheActive = (state.nativeSpeculativeCache || (!batchingOff && !prefixOff)) && !!state.enableBlockDiskCache
  const ramCopyAvailable = blockDiskCacheActive && !state.nativeSpeculativeCache
  const userPagedCacheActive = ramCopyAvailable && !!state.usePagedCache
  const effectiveUsePagedCache = userPagedCacheActive
  const pagedCacheDisabled = !ramCopyAvailable
  const legacyDiskCacheDisabled = batchingOff || blockDiskCacheActive
  const blockDiskCacheVisible = !!state.nativeSpeculativeCache || !batchingOff
  const blockDiskCacheDisabled = batchingOff && !state.nativeSpeculativeCache
  const legacyDiskCacheChecked = !!state.enableDiskCache && !legacyDiskCacheDisabled && !prefixOff
  const blockDiskCacheChecked = blockDiskCacheActive
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
  const prefixEnabled = !batchingOff && !!state.enablePrefixCache
  const enableBlockDiskCache = !!state.enableBlockDiskCache && (prefixEnabled || !!state.nativeSpeculativeCache)
  // The RAM copy (paged RAM blocks) only ever comes from the user's own checkbox, and only in front of the
  // block store. Neither a per-family registry capability (`detected.usePagedCache`) nor an architecture
  // "requires paged" claim may turn it on -- both once did, which is how 18 families ended up launching with
  // --use-paged-cache nobody asked for.
  const effectiveUsePagedCache = !!state.usePagedCache && enableBlockDiskCache && !state.nativeSpeculativeCache

  return {
    prefixCacheOff: !prefixEnabled,
    effectiveUsePagedCache,
    enableLegacyDiskCache: !!state.enableDiskCache && prefixEnabled && !enableBlockDiskCache,
    enableBlockDiskCache,
  }
}

export function cacheControlUpdatesForPagedToggle(enabled: boolean, state: CacheControlState): CacheControlUpdate[] {
  // "Keep a RAM copy" fronts the block store: turning it on also turns Block Disk Cache (and so Prefix Cache) on,
  // so the saved state always launches what the checkbox shows (RAM + SSD).
  const updates: CacheControlUpdate[] = []
  if (enabled && !state.enablePrefixCache) updates.push(['enablePrefixCache', true])
  if (enabled && state.enableDiskCache) updates.push(['enableDiskCache', false])
  if (enabled && !state.enableBlockDiskCache) updates.push(['enableBlockDiskCache', true])
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
  if (enabled && state.enableDiskCache) updates.push(['enableDiskCache', false])
  // Block Disk Cache off = RAM only: the RAM copy has nothing to front, so it goes off with it (it would otherwise
  // come back on with the next Block Disk Cache click without the user asking).
  if (!enabled && state.usePagedCache) updates.push(['usePagedCache', false])
  updates.push(['enableBlockDiskCache', enabled])
  return updates
}
