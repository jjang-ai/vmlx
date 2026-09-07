import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { buildCacheLaunchArgs } from '../src/shared/cacheLaunchArgs'
import { usesExactTypedPromptDiskCache } from '../src/shared/detectedFamilyNames'

/**
 * The SSD budget slider must reach the engine.
 *
 * v1.6.35 moved the block-disk budget from a flat GB number to a percent of the
 * volume. The engine resolves explicit-GB-wins, so passing only
 * `--block-disk-cache-max-gb` pins the budget and makes the percent slider
 * decorative — which is exactly what shipped in the first cut: fresh configs
 * seeded `blockDiskCacheMaxGb: 10` and buildArgs emitted only the GB flag, so
 * the 10%-of-disk default could never take effect.
 *
 * These assertions read the REAL sessions.ts, deliberately. `settings-flow.test.ts`
 * builds args through a hand-written mirror of buildArgs; a mirror agrees with
 * itself no matter what the shipped launcher does, and that is why it stayed
 * green through the defect. Anything asserting "the user's setting reaches the
 * process" has to look at the source that spawns the process.
 */
describe('block-disk budget percent reaches the engine argv', () => {
  const sessions = readFileSync('src/main/sessions.ts', 'utf-8')

  it('buildArgs emits --block-disk-cache-max-percent', () => {
    const { args } = buildCacheLaunchArgs({
      continuousBatching: true,
      enablePrefixCache: true,
      enableBlockDiskCache: true,
      blockDiskCacheMaxPercent: 10,
    })
    expect(args).toContain('--block-disk-cache-max-percent')
    expect(args[args.indexOf('--block-disk-cache-max-percent') + 1]).toBe('10')
    expect(sessions).toContain('args.push(...cacheLaunch.args)')
  })

  it('emits the percent alongside the GB flag, not as an either/or', () => {
    const { args } = buildCacheLaunchArgs({
      continuousBatching: true,
      enablePrefixCache: true,
      enableBlockDiskCache: true,
      blockDiskCacheMaxGb: 40,
      blockDiskCacheMaxPercent: 10,
    })
    const gbIdx = args.indexOf('--block-disk-cache-max-gb')
    const pctIdx = args.indexOf('--block-disk-cache-max-percent')
    expect(gbIdx).toBeGreaterThan(-1)
    expect(pctIdx).toBeGreaterThan(gbIdx)
    expect(args[gbIdx + 1]).toBe('40')
    expect(args[pctIdx + 1]).toBe('10')
  })

  it('fresh configs seed NO GB value at all, so the percent owns the budget', () => {
    // This assertion previously demanded a seeded 0 and passed while the
    // shipped launcher ran an UNBOUNDED cache: the engine reads
    // `explicit_gb is not None`, and BlockDiskStore documents
    // `max_size_gb: 0 = unlimited`. 0 is not "unset" — absent is.
    //
    // Behaviour, rather than these string greps, is asserted in
    // block-disk-budget-behaviour.test.ts. Every literal this file checked was
    // individually correct while the composed result was wrong, which is the
    // whole reason a grep-based test could not catch it.
    expect(sessions).toContain(
      "setConfigValue(mutable, 'blockDiskCacheMaxPercent', DEFAULT_BLOCK_DISK_CACHE_PERCENT)",
    )
    expect(sessions).not.toContain("setConfigValue(mutable, 'blockDiskCacheMaxGb', 0)")
    expect(sessions).not.toContain("setConfigValue(mutable, 'blockDiskCacheMaxGb', 10)")
  })

  it('both arg allow-lists know the percent flag', () => {
    // Two lists gate which flags survive engine adoption / arg comparison. A
    // flag missing from either is silently dropped when an existing engine is
    // reused, so the setting appears to work and then does not.
    const occurrences = sessions.split("'--block-disk-cache-max-percent',").length - 1
    expect(occurrences).toBeGreaterThanOrEqual(2)
  })

  it('the default percent has exactly one definition', () => {
    const shared = readFileSync('src/shared/cacheDefaults.ts', 'utf-8')
    expect(shared).toContain('export const DEFAULT_BLOCK_DISK_CACHE_PERCENT = 10')
    const form = readFileSync(
      'src/renderer/src/components/sessions/SessionConfigForm.tsx',
      'utf-8',
    )
    expect(form).toContain('blockDiskCacheMaxPercent: DEFAULT_BLOCK_DISK_CACHE_PERCENT')
    expect(sessions).toContain('DEFAULT_BLOCK_DISK_CACHE_PERCENT')
  })
})

describe('the v17 SSD-first default runs as a post-pass', () => {
  const sessions = readFileSync('src/main/sessions.ts', 'utf-8')

  it('applies after the legacy tuple migrations, not before', () => {
    const legacyIdx = sessions.indexOf('const legacyChanged = applyLegacyCacheStackMigrations(config, modelPath)')
    const ssdIdx = sessions.indexOf('const ssdFirstChanged = applySsdFirstCacheDefaults(')
    expect(legacyIdx).toBeGreaterThan(-1)
    expect(ssdIdx).toBeGreaterThan(legacyIdx)
  })

  it('skips passes where the version stamp declined', () => {
    // markCacheStackStartupDefaultsCurrent refuses to stamp while the bundle is
    // unreachable so the migration can retry later. Mutating usePagedCache or
    // blockDiskCacheMaxGb on such a pass rewrites the exact-tuple fingerprint
    // the retry matches on, and the retry then never fires.
    expect(sessions).toContain(
      'Number(config.cacheStackStartupDefaultsVersion || 0) !==\n    CACHE_STACK_STARTUP_DEFAULTS_VERSION',
    )
  })

  it('retires paged RAM for every family while preserving openPangu disk format', () => {
    const fn = sessions.slice(
      sessions.indexOf('function applySsdFirstCacheDefaults('),
      sessions.indexOf('function applyCacheStackStartupDefaultMigration('),
    )
    expect(fn).toContain('config.usePagedCache = false')
    expect(fn).not.toContain('isZayaCacheStackMigrationTarget')
    // the exact-typed prompt-L2 families (openPangu v2, GLM5-Next) keep their own disk format: the
    // post-pass returns before rewriting it, through the shared family helper
    expect(fn).toContain('if (usesExactTypedPromptDiskCache(detectedFamily)) return changed')
    expect(usesExactTypedPromptDiskCache('openpangu_v2')).toBe(true)
    expect(usesExactTypedPromptDiskCache('qwen3_5')).toBe(false)
  })
})

describe('an unrecognised family still gets the SSD-first default', () => {
  const sessions = readFileSync('src/main/sessions.ts', 'utf-8')

  it('gates the version stamp on bundle REACHABILITY, not on family detection', () => {
    // The stamp declines so an unmounted drive can retry later. It used to
    // decide that by asking whether the family resolved, which conflates "the
    // drive is not mounted" with "this bundle reads fine but is not in the
    // registry" -- and the second case then never migrates at all.
    //
    // Live: 21 real sessions, 20 migrated to the SSD-first default. The one
    // that did not was an LFM2.5-VL HF snapshot (model_type lfm2_vl) that
    // detects as family 'unknown'; it kept the in-RAM paged cache on.
    const fn = sessions.slice(
      sessions.indexOf('function markCacheStackStartupDefaultsCurrent('),
      sessions.indexOf('function applySsdFirstCacheDefaults('),
    )
    expect(fn).toContain("existsSync(join(targetPath, 'config.json'))")
    expect(fn).not.toContain("effectiveFamily === 'unknown'")
  })
})
