import { describe, expect, it } from 'vitest'

import {
  cacheControlUpdatesForDsv4BlockDiskToggle,
  cacheControlUpdatesForDsv4CompositeToggle,
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
    expect(policy.pagedCacheDisabled).toBe(false)
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

  it('lets paged cache opt in by enabling prefix and clearing legacy disk cache', () => {
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
      ['usePagedCache', true],
    ])
  })

  it('clears block disk cache when paged cache is turned off', () => {
    const updates = cacheControlUpdatesForPagedToggle(false, {
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: true,
      enableDiskCache: false,
      enableBlockDiskCache: true,
    })

    expect(updates).toEqual([
      ['enableBlockDiskCache', false],
      ['usePagedCache', false],
    ])
  })

  it('lets block disk cache opt in by enabling prefix and paged cache', () => {
    const updates = cacheControlUpdatesForBlockDiskToggle(true, {
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: false,
      enableDiskCache: true,
      enableBlockDiskCache: false,
    })

    expect(updates).toEqual([
      ['enablePrefixCache', true],
      ['usePagedCache', true],
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

  it('keeps legacy disk cache disabled for architectures that force paged cache when prefix is enabled', () => {
    const policy = resolveCacheControlPolicy({
      continuousBatching: true,
      enablePrefixCache: false,
      usePagedCache: false,
      enableDiskCache: false,
      enableBlockDiskCache: false,
      architectureRequiresPagedCache: true,
    })

    expect(policy.legacyDiskCacheDisabled).toBe(true)
    expect(policy.legacyDiskCacheUnavailableReason).toBe('architecture-requires-paged-cache')
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

  it('launch policy emits block disk cache after UI has saved prefix and paged prerequisites', () => {
    const policy = resolveCacheLaunchPolicy({
      continuousBatching: true,
      enablePrefixCache: true,
      usePagedCache: true,
      enableDiskCache: true,
      enableBlockDiskCache: true,
    })

    expect(policy.prefixCacheOff).toBe(false)
    expect(policy.effectiveUsePagedCache).toBe(true)
    expect(policy.enableLegacyDiskCache).toBe(false)
    expect(policy.enableBlockDiskCache).toBe(true)
  })

  it('DSV4 composite cache opt-in updates the DSV4 master flag and prerequisites together', () => {
    expect(cacheControlUpdatesForDsv4CompositeToggle(true)).toEqual([
      ['dsv4PrefixCache', true],
      ['enablePrefixCache', true],
      ['usePagedCache', true],
      ['enableBlockDiskCache', true],
    ])

    expect(cacheControlUpdatesForDsv4CompositeToggle(false)).toEqual([
      ['dsv4PrefixCache', false],
      ['enablePrefixCache', false],
      ['usePagedCache', false],
      ['enableBlockDiskCache', false],
    ])
  })

  it('DSV4 block L2 opt-in enables the native composite cache prerequisites', () => {
    expect(cacheControlUpdatesForDsv4BlockDiskToggle(true)).toEqual([
      ['dsv4PrefixCache', true],
      ['enablePrefixCache', true],
      ['usePagedCache', true],
      ['enableDiskCache', false],
      ['enableBlockDiskCache', true],
    ])

    expect(cacheControlUpdatesForDsv4BlockDiskToggle(false)).toEqual([
      ['enableBlockDiskCache', false],
    ])
  })
})
