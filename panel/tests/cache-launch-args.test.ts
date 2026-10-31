import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import { buildCacheLaunchArgs } from '../src/shared/cacheLaunchArgs'

const SSD_ONLY = {
  continuousBatching: true,
  enablePrefixCache: true,
  usePagedCache: false,
  enableDiskCache: false,
  enableBlockDiskCache: true,
  cacheMemoryMb: 8192,
  cacheMemoryPercent: 35,
  cacheTtlMinutes: 60,
  effectivePagedCacheBlockSize: 64,
  maxCacheBlocks: 4097,
  blockDiskCacheMaxGb: 0,
  blockDiskCacheMaxPercent: 10,
}

describe('authoritative cache launch arguments', () => {
  it('builds RAM + SSD argv from the saved RAM copy: paged RAM blocks with their RAM budget, then the block store', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, usePagedCache: true })

    expect(result.policy.effectiveUsePagedCache).toBe(true)
    expect(result.blockDiskOnly).toBe(false)
    expect(result.args).toEqual([
      '--use-paged-cache',
      '--no-vision-memory-cache',
      '--ssm-state-cache-size', '0',
      '--ssm-state-cache-mb', '0',
      '--cache-memory-mb', '8192',
      '--cache-memory-percent', '0.35',
      '--cache-ttl-minutes', '60',
      '--paged-cache-block-size', '64',
      '--max-cache-blocks', '4097',
      '--enable-block-disk-cache',
      '--block-disk-cache-max-percent', '10',
    ])
    expect(result.args).not.toContain('--no-paged-cache')
  })

  it('a saved RAM copy without Block Disk Cache launches RAM only (memory-aware), never paged RAM', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, usePagedCache: true, enableBlockDiskCache: false })

    expect(result.policy.effectiveUsePagedCache).toBe(false)
    expect(result.args[0]).toBe('--no-paged-cache')
    expect(result.args).toContain('--disable-block-disk-cache')
    expect(result.args).not.toContain('--use-paged-cache')
  })

  it('builds SSD-only argv with no RAM tier or RAM budget', () => {
    const result = buildCacheLaunchArgs(SSD_ONLY)

    expect(result.policy.effectiveUsePagedCache).toBe(false)
    expect(result.args).toEqual([
      '--no-paged-cache',
      '--no-vision-memory-cache',
      '--ssm-state-cache-size', '0',
      '--ssm-state-cache-mb', '0',
      '--paged-cache-block-size', '64',
      '--max-cache-blocks', '4097',
      '--enable-block-disk-cache',
      '--block-disk-cache-max-percent', '10',
    ])
    expect(result.args).not.toContain('--use-paged-cache')
    expect(result.args).not.toContain('--enable-vision-memory-cache')
    expect(result.args).not.toContain('--cache-memory-mb')
    expect(result.args).not.toContain('--cache-memory-percent')
    expect(result.args).not.toContain('--block-disk-cache-max-gb')
  })

  it('emits the engine-default override when the visible SSD toggle is off', () => {
    const result = buildCacheLaunchArgs({
      ...SSD_ONLY,
      usePagedCache: false,
      enableBlockDiskCache: false,
    })

    expect(result.args).toContain('--no-paged-cache')
    expect(result.args).toContain('--disable-block-disk-cache')
    expect(result.args).not.toContain('--enable-block-disk-cache')
  })

  it('keeps the prefix master switch authoritative', () => {
    const result = buildCacheLaunchArgs({
      ...SSD_ONLY,
      enablePrefixCache: false,
    })

    expect(result.args).toEqual([
      '--no-paged-cache',
      '--no-vision-memory-cache',
      '--ssm-state-cache-size', '0',
      '--ssm-state-cache-mb', '0',
      '--disable-prefix-cache',
    ])
  })

  it('puts exact typed prompt snapshots in the managed SSD pool with the block budget, not a flat GB', () => {
    const typed = {
      ...SSD_ONLY,
      enableDiskCache: true,
      enableBlockDiskCache: false,
      forceMemoryAwareCache: true,
      diskCacheMaxGb: 10,
      blockDiskCacheMaxPercent: 7,
    }
    const pooled = buildCacheLaunchArgs({ ...typed, typedPromptSsdPool: true }).args
    expect(pooled).toContain('--enable-disk-cache')
    expect(pooled).toContain('--disk-cache-pool')
    expect(pooled).not.toContain('--disk-cache-max-gb')
    expect(pooled[pooled.indexOf('--block-disk-cache-max-percent') + 1]).toBe('7')
    // An explicit positive block GB still wins over the percent, exactly as for the block store.
    const gb = buildCacheLaunchArgs({ ...typed, typedPromptSsdPool: true, blockDiskCacheMaxGb: 40 }).args
    expect(gb[gb.indexOf('--block-disk-cache-max-gb') + 1]).toBe('40')
    // Without the flag (legacy whole-prompt cache) nothing changes.
    const legacy = buildCacheLaunchArgs(typed).args
    expect(legacy).not.toContain('--disk-cache-pool')
    expect(legacy[legacy.indexOf('--disk-cache-max-gb') + 1]).toBe('10')
  })

  it('passes typedPromptSsdPool from BOTH the launcher and the command preview', () => {
    for (const file of ['../src/main/sessions.ts', '../src/renderer/src/components/sessions/SessionSettings.tsx']) {
      const src = readFileSync(new URL(file, import.meta.url), 'utf8')
      expect(src, file).toContain('typedPromptSsdPool: exactTypedPromptDiskCache')
    }
  })

  it('keeps openPangu-style exact prompt L2 memory-aware and non-paged', () => {
    const result = buildCacheLaunchArgs({
      ...SSD_ONLY,
      enableDiskCache: true,
      enableBlockDiskCache: false,
      noMemoryAwareCache: true,
      forceMemoryAwareCache: true,
      diskCacheDir: '/tmp/prompt-l2',
      diskCacheMaxGb: 10,
    })

    expect(result.args).toContain('--no-paged-cache')
    expect(result.args).not.toContain('--no-memory-aware-cache')
    expect(result.args).toContain('--enable-disk-cache')
    const budgetIndex = result.args.indexOf('--cache-memory-mb')
    expect(budgetIndex).toBeGreaterThan(-1)
    expect(result.args[budgetIndex + 1]).toBe('0')
    expect(result.args).not.toContain('--cache-memory-percent')
    expect(result.args).not.toContain('--cache-ttl-minutes')
    expect(result.args).toContain('--disable-block-disk-cache')
  })

  it('is the cache argv source for both the visual preview and real spawn path', () => {
    const renderer = readFileSync('src/renderer/src/components/sessions/SessionSettings.tsx', 'utf8')
    const launcher = readFileSync('src/main/sessions.ts', 'utf8')

    expect(renderer).toContain("import { buildCacheLaunchArgs }")
    expect(renderer).toContain('const cacheLaunch = buildCacheLaunchArgs({')
    expect(renderer).toContain('parts.push(...cacheLaunch.args)')
    expect(launcher).toContain("import { buildCacheLaunchArgs }")
    expect(launcher).toContain('const cacheLaunch = buildCacheLaunchArgs({')
    expect(launcher).toContain('args.push(...cacheLaunch.args)')
    expect(renderer).not.toContain("parts.push('--use-paged-cache')")
    expect(launcher).not.toContain("args.push('--use-paged-cache')")
  })
})


describe('native DFlash2 SSD settings', () => {
  it('passes the saved SSD location and budget despite the disabled batched prefix stack', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, continuousBatching: false,
      nativeSpeculativeCache: true, blockDiskCacheDir: '/private/test-cache',
      blockDiskCacheMaxPercent: 7 })
    expect(result.args).toContain('--disable-prefix-cache')
    expect(result.args).toContain('--enable-block-disk-cache')
    expect(result.args[result.args.indexOf('--block-disk-cache-dir') + 1]).toBe('/private/test-cache')
    expect(result.args[result.args.indexOf('--block-disk-cache-max-percent') + 1]).toBe('7')
    expect(result.args).not.toContain('--cache-memory-mb')
    expect(result.args).not.toContain('--paged-cache-block-size')
  })

  it('honors the SSD opt-out without activating a RAM fallback', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, continuousBatching: false,
      nativeSpeculativeCache: true, enableBlockDiskCache: false })
    expect(result.args).toContain('--disable-block-disk-cache')
    expect(result.args).not.toContain('--enable-block-disk-cache')
    expect(result.args).not.toContain('--cache-memory-mb')
    expect(result.args).not.toContain('--cache-memory-percent')
  })
})
