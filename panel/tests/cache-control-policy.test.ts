import { describe, expect, it } from 'vitest'

import {
  cacheControlUpdatesForBlockDiskToggle,
  cacheControlUpdatesForDiskToggle,
  cacheControlUpdatesForPagedToggle,
  resolveCacheLaunchPolicy,
  resolveCacheControlPolicy,
} from '../src/shared/cacheControlPolicy'

describe('cache control policy', () => {
  it('ignores a stale saved paged toggle when prefix cache is off', () => {
    const policy = resolveCacheControlPolicy({
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: true,
      enableDiskCache: false,
      enableBlockDiskCache: false,
    })

    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.userPagedCacheActive).toBe(false)
    // Paged RAM is retired product-wide: the control is hard-disabled, so the
    // stale saved toggle can never re-arm it from the UI either.
    expect(policy.pagedCacheDisabled).toBe(true)
    expect(policy.legacyDiskCacheDisabled).toBe(false)
  })

  it('lets legacy disk cache opt in by enabling prefix and clearing paged/block cache', () => {
    const updates = cacheControlUpdatesForDiskToggle(true, {
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: true,
      enableDiskCache: false,
      enableBlockDiskCache: true,
    })

    expect(updates).toEqual([
      ['enablePrefixCache', true],
      ['usePagedCache', false],
      ['enableBlockDiskCache', false],
      ['enableDiskCache', true],
    ])
  })

  it('turning the RAM copy on turns on the block store it fronts (RAM + SSD), with Prefix Cache', () => {
    const updates = cacheControlUpdatesForPagedToggle(true, {
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: false,
      enableDiskCache: true,
      enableBlockDiskCache: false,
    })

    expect(updates).toEqual([
      ['enablePrefixCache', true],
      ['enableDiskCache', false],
      ['enableBlockDiskCache', true],
      ['usePagedCache', true],
    ])
  })

  it('turning Block Disk Cache off (RAM only) also drops the RAM copy it fronted', () => {
    const updates = cacheControlUpdatesForBlockDiskToggle(false, {
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: true,
      enableDiskCache: false,
      enableBlockDiskCache: true,
    })

    expect(updates).toEqual([['usePagedCache', false], ['enableBlockDiskCache', false]])
  })

  it('keeps block disk cache when paged RAM is turned off', () => {
    const updates = cacheControlUpdatesForPagedToggle(false, {
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: true,
      enableDiskCache: false,
      enableBlockDiskCache: true,
    })

    expect(updates).toEqual([['usePagedCache', false]])
  })

  it('lets block disk cache opt in by enabling prefix without forcing paged RAM', () => {
    const updates = cacheControlUpdatesForBlockDiskToggle(true, {
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: false,
      enableDiskCache: true,
      enableBlockDiskCache: false,
    })

    expect(updates).toEqual([
      ['enablePrefixCache', true],
      ['enableDiskCache', false],
      ['enableBlockDiskCache', true],
    ])
  })

  it('keeps block disk cache visible when prefix and paged cache are both off', () => {
    const policy = resolveCacheControlPolicy({
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: false,
      enableDiskCache: false,
      enableBlockDiskCache: false,
      architectureRequiresPagedCache: true,
    })

    expect(policy.blockDiskCacheVisible).toBe(true)
    expect(policy.blockDiskCacheDisabled).toBe(false)
    expect(policy.blockDiskCacheChecked).toBe(false)
  })

  it('ignores an architecture paged-cache requirement — legacy disk gating never cites paged cache', () => {
    // OLD contract: architectureRequiresPagedCache=true forced paged active and
    // disabled legacy disk with reason 'architecture-requires-paged-cache'.
    // NEW contract: no architecture may require the retired RAM tier, so the
    // input is ignored and legacy disk stays available (prefix-off state).
    const policy = resolveCacheControlPolicy({
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: false,
      enableDiskCache: false,
      enableBlockDiskCache: false,
      architectureRequiresPagedCache: true,
    })

    expect(policy.architectureRequiresPagedCache).toBe(false)
    expect(policy.architectureForcedPagedActive).toBe(false)
    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.legacyDiskCacheDisabled).toBe(false)
    expect(policy.legacyDiskCacheUnavailableReason).toBeUndefined()
  })

  it('launch policy keeps prefix cache off as the master switch', () => {
    const policy = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: false,
      enableDiskCache: true,
      enableBlockDiskCache: false,
    })

    expect(policy.prefixCacheOff).toBe(true)
    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.enableLegacyDiskCache).toBe(false)
    expect(policy.enableBlockDiskCache).toBe(false)
  })

  it('launch policy runs RAM + SSD only from the saved RAM copy toggle together with block disk cache', () => {
    const ramSsd = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: true,
      enableDiskCache: true,
      enableBlockDiskCache: true,
    })
    expect(ramSsd.prefixCacheOff).toBe(false)
    expect(ramSsd.effectiveUsePagedCache).toBe(true)
    expect(ramSsd.enableLegacyDiskCache).toBe(false)
    expect(ramSsd.enableBlockDiskCache).toBe(true)

    // A saved toggle without the block store it fronts launches no paged RAM.
    const noBlock = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: true,
      enableDiskCache: false,
      enableBlockDiskCache: false,
    })
    expect(noBlock.effectiveUsePagedCache).toBe(false)
    // Nor for a native speculative lane, which owns its SSD store.
    expect(resolveCacheLaunchPolicy({
      continuousBatching: false,
      nativeSpeculativeCache: true,
      enablePrefixCache: true,
      usePagedCache: true,
      enableDiskCache: false,
      enableBlockDiskCache: true,
    }).effectiveUsePagedCache).toBe(false)
  })

  it('an architecture "requires paged" claim never turns the RAM copy on by itself', () => {
    const policy = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: false,
      enableDiskCache: false,
      enableBlockDiskCache: true,
      architectureRequiresPagedCache: true,
      architectureSupportsBlockDiskOnly: false,
    })

    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.enableBlockDiskCache).toBe(true)
    expect(policy.enableLegacyDiskCache).toBe(false)
  })

  it('launch policy emits a disk-only block backend when paged RAM is off', () => {
    const policy = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: false,
      enableDiskCache: true,
      enableBlockDiskCache: true,
    })

    expect(policy.prefixCacheOff).toBe(false)
    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.enableLegacyDiskCache).toBe(false)
    expect(policy.enableBlockDiskCache).toBe(true)
  })

  it('keeps block SSD/L2 toggle available whether paged RAM is on or off', () => {
    for (const usePagedCache of [true, false]) {
      const policy = resolveCacheControlPolicy({
        continuousBatching: true,
        enablePrefixCache: true,
        usePagedCache,
        enableDiskCache: false,
        enableBlockDiskCache: true,
      })

      expect(policy.blockDiskCacheVisible).toBe(true)
      expect(policy.blockDiskCacheDisabled).toBe(false)
      expect(policy.blockDiskCacheChecked).toBe(true)
      expect(policy.legacyDiskCacheDisabled).toBe(true)
    }
  })

  it('allows legacy disk cache only after both paged RAM and block SSD/L2 are off', () => {
    const policy = resolveCacheControlPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: false,
      enableDiskCache: true,
      enableBlockDiskCache: false,
    })

    expect(policy.blockDiskCacheVisible).toBe(true)
    expect(policy.blockDiskCacheDisabled).toBe(false)
    expect(policy.legacyDiskCacheDisabled).toBe(false)
    expect(policy.legacyDiskCacheChecked).toBe(true)
  })

  it('lets a hybrid SSM architecture use block SSD L2 without paged RAM', () => {
    const ui = resolveCacheControlPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: false,
      enableDiskCache: false,
      enableBlockDiskCache: true,
      architectureRequiresPagedCache: true,
      architectureSupportsBlockDiskOnly: true,
    })
    const launch = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: false,
      enableDiskCache: false,
      enableBlockDiskCache: true,
      architectureRequiresPagedCache: true,
      architectureSupportsBlockDiskOnly: true,
    })

    expect(ui.architectureForcedPagedActive).toBe(false)
    // The RAM copy is the user's choice for every family, hybrid included: available, not forced.
    expect(ui.pagedCacheDisabled).toBe(false)
    expect(ui.effectiveUsePagedCache).toBe(false)
    expect(launch.effectiveUsePagedCache).toBe(false)
    expect(launch.enableBlockDiskCache).toBe(true)
  })

  it('lets a rotating mixed-SWA architecture use typed block SSD L2 without paged RAM', () => {
    const policy = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: false,
      enableDiskCache: false,
      enableBlockDiskCache: true,
      architectureRequiresPagedCache: true,
      architectureSupportsBlockDiskOnly: true,
    })

    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.enableBlockDiskCache).toBe(true)
    expect(policy.enableLegacyDiskCache).toBe(false)
  })

  it('never escalates a hybrid architecture without block SSD L2 to paged cache — it launches with no cache tier', () => {
    // OLD contract: this exact state force-enabled paged RAM as the fallback
    // tier. NEW contract: hybrid/Mamba drop the memory-aware lane instead of
    // escalating — no paged RAM, no L2, prefix reuse simply absent. That costs
    // speed, never correctness.
    const policy = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: false,
      enableDiskCache: false,
      enableBlockDiskCache: false,
      architectureRequiresPagedCache: true,
      architectureSupportsBlockDiskOnly: true,
    })

    expect(policy.prefixCacheOff).toBe(false)
    expect(policy.effectiveUsePagedCache).toBe(false)
    expect(policy.enableLegacyDiskCache).toBe(false)
    expect(policy.enableBlockDiskCache).toBe(false)
  })

})
