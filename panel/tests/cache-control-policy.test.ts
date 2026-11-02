import { describe, expect, it } from 'vitest'

import {
  cacheControlUpdatesForRamToggle,
  cacheControlUpdatesForSsdToggle,
  resolveCacheLaunchPolicy,
  resolveSimpleCacheTier,
  type CacheControlState,
  type CacheControlUpdate,
} from '../src/shared/cacheControlPolicy'

const base: CacheControlState = {
  continuousBatching: true,
  enablePrefixCache: true,
  usePagedCache: false,
  enableDiskCache: false,
  enableBlockDiskCache: true,
}

function apply(state: CacheControlState, updates: CacheControlUpdate[]): CacheControlState {
  const next = { ...state }
  for (const [key, value] of updates) next[key] = value
  return next
}

describe('saved cache keys -> the two choices (SSD cache, RAM cache)', () => {
  it('maps every saved combination onto exactly one of the four tiers', () => {
    // The shipped default: Prefix Cache on, Block Disk Cache on, no RAM copy.
    expect(resolveSimpleCacheTier(base)).toEqual({ ssdOn: true, ramOn: false, ramAvailable: true, tier: 'ssd' })
    // Prefix Cache on, Block Disk Cache off = the memory-aware prefix cache.
    expect(resolveSimpleCacheTier({ ...base, enableBlockDiskCache: false })).toMatchObject({ ssdOn: false, ramOn: true, tier: 'ram' })
    // Block Disk Cache + the saved RAM copy.
    expect(resolveSimpleCacheTier({ ...base, usePagedCache: true })).toMatchObject({ ssdOn: true, ramOn: true, tier: 'ram-ssd' })
    // Prefix Cache off = no reuse, whatever the stale tier bits say.
    expect(resolveSimpleCacheTier({ ...base, enablePrefixCache: false, usePagedCache: true })).toMatchObject({ ssdOn: false, ramOn: false, tier: 'off' })
  })

  it('a stale RAM copy saved without Block Disk Cache is RAM only, never paged RAM', () => {
    const state = { ...base, usePagedCache: true, enableBlockDiskCache: false }
    expect(resolveSimpleCacheTier(state).tier).toBe('ram')
    const launch = resolveCacheLaunchPolicy(state)
    expect(launch.effectiveUsePagedCache).toBe(false)
    expect(launch.ramCacheActive).toBe(true)
  })

  it('continuous batching off means both choices are off and unavailable', () => {
    expect(resolveSimpleCacheTier({ ...base, continuousBatching: false })).toEqual({ ssdOn: false, ramOn: false, ramAvailable: false, tier: 'off' })
  })

  it('exact typed prompt lanes (openPangu / native GLM) read Disk Cache as the SSD choice and offer no RAM copy next to it', () => {
    const lane = { exactTypedPromptDiskCache: true }
    const typed = { ...base, enableDiskCache: true, enableBlockDiskCache: false }
    expect(resolveSimpleCacheTier(typed, lane)).toEqual({ ssdOn: true, ramOn: false, ramAvailable: false, tier: 'ssd' })
    expect(resolveSimpleCacheTier({ ...typed, enableDiskCache: false }, lane)).toEqual({ ssdOn: false, ramOn: true, ramAvailable: true, tier: 'ram' })
    expect(cacheControlUpdatesForRamToggle(true, typed, lane)).toEqual([])
  })

  it('DFlash2 / DSpark single-request lanes have only the SSD choice', () => {
    const native = { ...base, continuousBatching: false, nativeSpeculativeCache: true }
    expect(resolveSimpleCacheTier(native)).toEqual({ ssdOn: true, ramOn: false, ramAvailable: false, tier: 'ssd' })
    expect(cacheControlUpdatesForSsdToggle(false, native)).toEqual([['enableBlockDiskCache', false]])
    expect(cacheControlUpdatesForRamToggle(true, native)).toEqual([])
  })
})

describe('the two toggles never lose the other choice', () => {
  it('SSD off while RAM is on keeps RAM (RAM + SSD -> RAM only); SSD off while RAM is off turns reuse off', () => {
    const ramSsd = { ...base, usePagedCache: true }
    expect(resolveSimpleCacheTier(apply(ramSsd, cacheControlUpdatesForSsdToggle(false, ramSsd))).tier).toBe('ram')
    expect(resolveSimpleCacheTier(apply(base, cacheControlUpdatesForSsdToggle(false, base))).tier).toBe('off')
  })

  it('SSD on while RAM is on keeps RAM (RAM only -> RAM + SSD); SSD on from off is SSD only', () => {
    const ramOnly = { ...base, enableBlockDiskCache: false }
    expect(resolveSimpleCacheTier(apply(ramOnly, cacheControlUpdatesForSsdToggle(true, ramOnly))).tier).toBe('ram-ssd')
    const off = { ...base, enablePrefixCache: false, enableBlockDiskCache: false }
    const next = apply(off, cacheControlUpdatesForSsdToggle(true, off))
    expect(resolveSimpleCacheTier(next).tier).toBe('ssd')
    expect(next.enablePrefixCache).toBe(true)
    expect(next.enableDiskCache).toBe(false)
  })

  it('RAM on with SSD on is RAM + SSD; RAM on with SSD off is RAM only', () => {
    expect(resolveSimpleCacheTier(apply(base, cacheControlUpdatesForRamToggle(true, base))).tier).toBe('ram-ssd')
    const off = { ...base, enablePrefixCache: false, enableBlockDiskCache: false }
    const next = apply(off, cacheControlUpdatesForRamToggle(true, off))
    expect(resolveSimpleCacheTier(next).tier).toBe('ram')
    expect(next.usePagedCache).toBe(false)
  })

  it('RAM off with SSD on is SSD only; RAM off with SSD off turns reuse off', () => {
    const ramSsd = { ...base, usePagedCache: true }
    expect(resolveSimpleCacheTier(apply(ramSsd, cacheControlUpdatesForRamToggle(false, ramSsd))).tier).toBe('ssd')
    const ramOnly = { ...base, enableBlockDiskCache: false }
    expect(resolveSimpleCacheTier(apply(ramOnly, cacheControlUpdatesForRamToggle(false, ramOnly))).tier).toBe('off')
  })

  it('exact typed lane: SSD off becomes RAM only and clears the GLM native-pool bit; SSD on clears the block store', () => {
    const lane = { exactTypedPromptDiskCache: true }
    const typed = { ...base, enableDiskCache: true, enableBlockDiskCache: true }
    const off = apply(typed, cacheControlUpdatesForSsdToggle(false, typed, lane))
    expect(off.enableDiskCache).toBe(false)
    expect(off.enableBlockDiskCache).toBe(false)
    expect(resolveSimpleCacheTier(off, lane).tier).toBe('ram')
    const on = apply(off, cacheControlUpdatesForSsdToggle(true, off, lane))
    expect(on).toMatchObject({ enablePrefixCache: true, enableDiskCache: true, enableBlockDiskCache: false, usePagedCache: false })
  })
})

describe('launch policy', () => {
  it('keeps prefix cache off as the master switch', () => {
    const policy = resolveCacheLaunchPolicy({ ...base, enablePrefixCache: false, usePagedCache: true })
    expect(policy).toEqual({
      prefixCacheOff: true,
      effectiveUsePagedCache: false,
      enableLegacyDiskCache: false,
      enableBlockDiskCache: false,
      ramCacheActive: false,
    })
  })

  it('names the RAM tier the percent bounds in each mode', () => {
    expect(resolveCacheLaunchPolicy(base).ramCacheActive).toBe(false) // SSD only: no retained RAM payloads
    expect(resolveCacheLaunchPolicy({ ...base, usePagedCache: true }).ramCacheActive).toBe(true) // paged RAM blocks
    expect(resolveCacheLaunchPolicy({ ...base, enableBlockDiskCache: false }).ramCacheActive).toBe(true) // memory-aware
    expect(resolveCacheLaunchPolicy({ ...base, enableDiskCache: true, enableBlockDiskCache: false }).ramCacheActive).toBe(false) // typed SSD lane
    expect(resolveCacheLaunchPolicy({ ...base, continuousBatching: false, nativeSpeculativeCache: true }).ramCacheActive).toBe(false)
  })

  it('an architecture "requires paged" claim never turns the RAM copy on by itself', () => {
    const policy = resolveCacheLaunchPolicy({ ...base, architectureRequiresPagedCache: true, architectureSupportsBlockDiskOnly: true })
    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.enableBlockDiskCache).toBe(true)
  })

  it('a hybrid architecture without the block store launches the memory-aware RAM tier, never paged cache', () => {
    const policy = resolveCacheLaunchPolicy({ ...base, enableBlockDiskCache: false, architectureRequiresPagedCache: true, architectureSupportsBlockDiskOnly: true })
    expect(policy.prefixCacheOff).toBe(false)
    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.enableLegacyDiskCache).toBe(false)
    expect(policy.enableBlockDiskCache).toBe(false)
    expect(policy.ramCacheActive).toBe(true)
  })
})
