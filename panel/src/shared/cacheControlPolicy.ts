export type CacheControlKey =
  | 'enablePrefixCache'
  | 'usePagedCache'
  | 'enableDiskCache'
  | 'enableBlockDiskCache'

export type CacheControlUpdate = [CacheControlKey, boolean]

/**
 * The saved cache keys (unchanged on disk, so every existing session keeps its
 * meaning). The panel shows them as exactly two choices, SSD cache and RAM
 * cache; `resolveSimpleCacheTier` is the only place that maps between the two.
 */
export interface CacheControlState {
  continuousBatching: boolean
  /** DFlash2 / DSpark single-request lanes own an SSD store independently of the batched prefix stack. */
  nativeSpeculativeCache?: boolean
  enablePrefixCache: boolean
  usePagedCache: boolean
  enableDiskCache: boolean
  enableBlockDiskCache: boolean
  architectureRequiresPagedCache?: boolean
  architectureSupportsBlockDiskOnly?: boolean
}

export interface CacheLaunchPolicy {
  prefixCacheOff: boolean
  effectiveUsePagedCache: boolean
  enableLegacyDiskCache: boolean
  enableBlockDiskCache: boolean
  /**
   * A RAM tier the engine bounds with `--cache-memory-percent`: the memory-aware
   * prefix cache (RAM only) or the paged RAM blocks in front of the block store
   * (RAM + SSD). False for SSD only, for both-off, and for the native
   * speculative and exact typed prompt-snapshot lanes.
   */
  ramCacheActive: boolean
}

export type SimpleCacheTier = 'off' | 'ram' | 'ssd' | 'ram-ssd'

export interface SimpleCacheLane {
  /**
   * openPangu / native GLM: the SSD choice is the exact typed prompt store
   * (`enableDiskCache`, `--enable-disk-cache --disk-cache-pool`). That lane
   * keeps no RAM copy next to its SSD store (the launcher passes
   * `--cache-memory-mb 0`), so RAM is offered only while SSD is off.
   */
  exactTypedPromptDiskCache?: boolean
}

export interface SimpleCacheTierState {
  ssdOn: boolean
  ramOn: boolean
  /** Whether the RAM choice can be turned on from the current state of this lane. */
  ramAvailable: boolean
  tier: SimpleCacheTier
}

function batchedPrefixActive(state: CacheControlState): boolean {
  return !!state.continuousBatching && !!state.enablePrefixCache
}

/**
 * Old saved keys -> the two user-facing choices.
 *
 *   SSD on  = Block Disk Cache on (generic) / Disk Cache on (exact typed lane) / native SSD store on (speculative lane)
 *   RAM on  = Prefix Cache on with no SSD store (the memory-aware prefix cache, "RAM only"),
 *             or Prefix Cache + Block Disk Cache + the saved RAM copy (`usePagedCache`, "RAM + SSD")
 *   both off = Prefix Cache off (no reuse)
 */
export function resolveSimpleCacheTier(state: CacheControlState, lane: SimpleCacheLane = {}): SimpleCacheTierState {
  if (state.nativeSpeculativeCache) {
    const ssdOn = !!state.enableBlockDiskCache
    return { ssdOn, ramOn: false, ramAvailable: false, tier: ssdOn ? 'ssd' : 'off' }
  }
  const prefix = batchedPrefixActive(state)
  if (lane.exactTypedPromptDiskCache) {
    const ssdOn = prefix && !!state.enableDiskCache
    const ramOn = prefix && !ssdOn
    return { ssdOn, ramOn, ramAvailable: !!state.continuousBatching && !ssdOn, tier: ssdOn ? 'ssd' : ramOn ? 'ram' : 'off' }
  }
  const ssdOn = prefix && !!state.enableBlockDiskCache
  const ramOn = prefix && (!ssdOn || !!state.usePagedCache)
  const tier: SimpleCacheTier = !prefix ? 'off' : ssdOn ? (ramOn ? 'ram-ssd' : 'ssd') : 'ram'
  return { ssdOn, ramOn, ramAvailable: !!state.continuousBatching, tier }
}

export function resolveCacheLaunchPolicy(state: CacheControlState): CacheLaunchPolicy {
  const prefixEnabled = batchedPrefixActive(state)
  const enableBlockDiskCache = !!state.enableBlockDiskCache && (prefixEnabled || !!state.nativeSpeculativeCache)
  // Paged RAM blocks only ever come from the user's own RAM choice, and only in front of the block store. Neither a
  // per-family registry capability nor an architecture "requires paged" claim may turn it on -- both once did, which
  // is how 18 families ended up launching with --use-paged-cache nobody asked for.
  const effectiveUsePagedCache = !!state.usePagedCache && enableBlockDiskCache && !state.nativeSpeculativeCache
  const enableLegacyDiskCache = !!state.enableDiskCache && prefixEnabled && !enableBlockDiskCache
  const ramCacheActive =
    prefixEnabled &&
    !state.nativeSpeculativeCache &&
    (effectiveUsePagedCache || (!enableBlockDiskCache && !enableLegacyDiskCache))

  return {
    prefixCacheOff: !prefixEnabled,
    effectiveUsePagedCache,
    enableLegacyDiskCache,
    enableBlockDiskCache,
    ramCacheActive,
  }
}

/** The SSD choice. Turning it on or off never silently changes whether RAM caching was wanted. */
export function cacheControlUpdatesForSsdToggle(
  enabled: boolean,
  state: CacheControlState,
  lane: SimpleCacheLane = {},
): CacheControlUpdate[] {
  if (state.nativeSpeculativeCache) return [['enableBlockDiskCache', enabled]]
  const current = resolveSimpleCacheTier(state, lane)
  if (lane.exactTypedPromptDiskCache) {
    if (enabled) {
      return [
        ['enablePrefixCache', true],
        ['usePagedCache', false],
        ['enableBlockDiskCache', false],
        ['enableDiskCache', true],
      ]
    }
    // Off = RAM only (the lane's memory-aware cache) while reuse stays wanted. The GLM text lane reads a saved
    // native-pool On bit as the same disk choice, so an explicit Off clears that bit too.
    return [['enableDiskCache', false], ['enableBlockDiskCache', false], ['usePagedCache', false]]
  }
  if (enabled) {
    return [
      ['enablePrefixCache', true],
      ['enableDiskCache', false],
      ['enableBlockDiskCache', true],
      // RAM only -> RAM + SSD keeps the RAM choice; SSD only when RAM was off.
      ['usePagedCache', current.ramOn],
    ]
  }
  return [
    ['enableBlockDiskCache', false],
    ['usePagedCache', false],
    // RAM + SSD -> RAM only; SSD only -> off (no reuse).
    ['enablePrefixCache', current.ramOn],
  ]
}

/** The RAM choice. Turning it on or off never changes the SSD choice. */
export function cacheControlUpdatesForRamToggle(
  enabled: boolean,
  state: CacheControlState,
  lane: SimpleCacheLane = {},
): CacheControlUpdate[] {
  if (state.nativeSpeculativeCache) return []
  const current = resolveSimpleCacheTier(state, lane)
  if (enabled) {
    if (current.ssdOn && lane.exactTypedPromptDiskCache) return []
    return [
      ['enablePrefixCache', true],
      // With SSD on: the RAM copy in front of the block store (RAM + SSD). Without: the memory-aware cache (RAM only).
      ['usePagedCache', current.ssdOn],
    ]
  }
  if (current.ssdOn) return [['usePagedCache', false]]
  return [['enablePrefixCache', false], ['usePagedCache', false]]
}
