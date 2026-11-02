import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import { buildCacheLaunchArgs } from '../src/shared/cacheLaunchArgs'

const SSD_ONLY = {
  continuousBatching: true,
  enablePrefixCache: true,
  usePagedCache: false,
  enableDiskCache: false,
  enableBlockDiskCache: true,
  cacheMemoryPercent: 35,
  effectivePagedCacheBlockSize: 64,
  maxCacheBlocks: 4097,
  blockDiskCacheMaxGb: 0,
  blockDiskCacheMaxPercent: 10,
}

const RETAINED_RAM_OFF = ['--no-vision-memory-cache', '--ssm-state-cache-size', '0', '--ssm-state-cache-mb', '0']

describe('authoritative cache launch arguments: the four modes', () => {
  it('SSD only (the default): block store with its disk percent, no RAM tier or RAM budget', () => {
    const result = buildCacheLaunchArgs(SSD_ONLY)

    expect(result.policy.effectiveUsePagedCache).toBe(false)
    expect(result.policy.ramCacheActive).toBe(false)
    expect(result.blockDiskOnly).toBe(true)
    expect(result.args).toEqual([
      '--no-paged-cache',
      ...RETAINED_RAM_OFF,
      '--paged-cache-block-size', '64',
      '--max-cache-blocks', '4097',
      '--enable-block-disk-cache',
      '--block-disk-cache-max-percent', '10',
    ])
    expect(result.args).not.toContain('--cache-memory-percent')
    expect(result.args).not.toContain('--block-disk-cache-max-gb')
  })

  it('RAM only: the memory-aware prefix cache bounded by --cache-memory-percent, block store off', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, enableBlockDiskCache: false })

    expect(result.policy.ramCacheActive).toBe(true)
    expect(result.args).toEqual([
      '--no-paged-cache',
      ...RETAINED_RAM_OFF,
      '--cache-memory-percent', '0.35',
      '--disable-block-disk-cache',
    ])
  })

  it('RAM + SSD: paged RAM blocks bounded by the same --cache-memory-percent, then the block store', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, usePagedCache: true })

    expect(result.policy.effectiveUsePagedCache).toBe(true)
    expect(result.policy.ramCacheActive).toBe(true)
    expect(result.blockDiskOnly).toBe(false)
    expect(result.args).toEqual([
      '--use-paged-cache',
      ...RETAINED_RAM_OFF,
      '--cache-memory-percent', '0.35',
      '--paged-cache-block-size', '64',
      '--max-cache-blocks', '4097',
      '--enable-block-disk-cache',
      '--block-disk-cache-max-percent', '10',
    ])
  })

  it('both off: prefix cache off, no RAM tier, SSD store explicitly off', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, enablePrefixCache: false, enableBlockDiskCache: false })

    expect(result.args).toEqual(['--no-paged-cache', ...RETAINED_RAM_OFF, '--disable-prefix-cache'])
  })

  it('prefix off beats stale tier bits (the master switch)', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, enablePrefixCache: false, usePagedCache: true })
    expect(result.args).toEqual(['--no-paged-cache', ...RETAINED_RAM_OFF, '--disable-prefix-cache'])
  })

  it('the percent is the ONLY RAM budget the panel sends: no mb, TTL or entry-count flags from saved values', () => {
    const stale = {
      ...SSD_ONLY,
      usePagedCache: true,
      cacheMemoryMb: 8192,
      cacheTtlMinutes: 60,
      noMemoryAwareCache: true,
      prefixCacheSize: 100,
      prefixCacheMaxBytes: 1024,
    } as Record<string, unknown>
    for (const state of [stale, { ...stale, usePagedCache: false, enableBlockDiskCache: false }]) {
      const args = buildCacheLaunchArgs(state as any).args
      for (const flag of ['--cache-memory-mb', '--cache-ttl-minutes', '--no-memory-aware-cache', '--prefix-cache-size', '--prefix-cache-max-bytes']) {
        expect(args, flag).not.toContain(flag)
      }
      expect(args[args.indexOf('--cache-memory-percent') + 1]).toBe('0.35')
    }
  })

  it('a saved RAM copy without Block Disk Cache launches RAM only (memory-aware), never paged RAM', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, usePagedCache: true, enableBlockDiskCache: false })

    expect(result.policy.effectiveUsePagedCache).toBe(false)
    expect(result.args[0]).toBe('--no-paged-cache')
    expect(result.args).toContain('--disable-block-disk-cache')
    expect(result.args).toContain('--cache-memory-percent')
  })

  it('puts exact typed prompt snapshots in the managed SSD pool with the block budget, not a flat GB', () => {
    const typed = {
      ...SSD_ONLY,
      enableDiskCache: true,
      enableBlockDiskCache: false,
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

  it('keeps openPangu-style exact prompt L2 memory-aware, non-paged and without a retained RAM copy', () => {
    const result = buildCacheLaunchArgs({
      ...SSD_ONLY,
      enableDiskCache: true,
      enableBlockDiskCache: false,
      diskCacheDir: '/tmp/prompt-l2',
      diskCacheMaxGb: 10,
    })

    expect(result.policy.ramCacheActive).toBe(false)
    expect(result.args).toContain('--no-paged-cache')
    expect(result.args).not.toContain('--no-memory-aware-cache')
    expect(result.args).toContain('--enable-disk-cache')
    const budgetIndex = result.args.indexOf('--cache-memory-mb')
    expect(budgetIndex).toBeGreaterThan(-1)
    expect(result.args[budgetIndex + 1]).toBe('0')
    expect(result.args).not.toContain('--cache-memory-percent')
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
    // Neither caller forwards the removed knobs behind the user's back.
    for (const src of [renderer, launcher]) {
      for (const key of ['cacheMemoryMb: config.cacheMemoryMb', 'cacheTtlMinutes: config.cacheTtlMinutes', 'noMemoryAwareCache: !!config.noMemoryAwareCache', 'prefixCacheSize: config.prefixCacheSize']) {
        expect(src).not.toContain(key)
      }
      expect(src).toContain('cacheMemoryPercent: config.cacheMemoryPercent')
    }
  })
})

describe('native DFlash2 / DSpark SSD settings', () => {
  it('passes the saved SSD location and budget despite the disabled batched prefix stack', () => {
    const result = buildCacheLaunchArgs({ ...SSD_ONLY, continuousBatching: false,
      nativeSpeculativeCache: true, blockDiskCacheDir: '/private/test-cache',
      blockDiskCacheMaxPercent: 7 })
    expect(result.args).toContain('--disable-prefix-cache')
    expect(result.args).toContain('--enable-block-disk-cache')
    expect(result.args[result.args.indexOf('--block-disk-cache-dir') + 1]).toBe('/private/test-cache')
    expect(result.args[result.args.indexOf('--block-disk-cache-max-percent') + 1]).toBe('7')
    expect(result.args).not.toContain('--cache-memory-mb')
    expect(result.args).not.toContain('--cache-memory-percent')
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

describe('session start keeps the RAM copy for the typed DSV4 / MiniMax-M3 caches', () => {
  // The start path re-applies each family's cache policy; it used to set usePagedCache=false unconditionally for
  // DSV4 and MiniMax-M3, so a saved RAM + SSD launched --no-paged-cache (seen live: panel saved usePagedCache=true,
  // the engine argv still had --no-paged-cache).
  const sessions = readFileSync('src/main/sessions.ts', 'utf-8')
  it('DSV4 and MiniMax-M3 keep a saved RAM copy only together with Block Disk Cache', () => {
    expect(sessions).toContain("const dsv4RamCopy = config.usePagedCache === true && config.enableBlockDiskCache !== false")
    expect(sessions).toContain('config.usePagedCache = dsv4RamCopy')
    expect(sessions).toContain("const m3RamCopy = config.usePagedCache === true && config.enableBlockDiskCache !== false")
    expect(sessions).toContain('config.usePagedCache = m3RamCopy')
  })
})
