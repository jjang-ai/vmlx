import {
  resolveCacheLaunchPolicy,
  type CacheControlState,
  type CacheLaunchPolicy,
} from './cacheControlPolicy'
import {
  finiteNonNegativeNumber,
  finitePositiveInteger,
  finitePositiveNumber,
} from './launchArgValues'

/**
 * Every input used to build the prefix-cache tier arguments shared by the
 * renderer command preview and the main-process engine launcher.
 *
 * Keep this interface deliberately independent of ServerConfig/SessionConfig:
 * importing either UI or main-process types here would make the shared policy
 * depend on one of the two consumers it exists to keep in agreement.
 */
export interface CacheLaunchArgsInput extends CacheControlState {
  /**
   * The one RAM number the panel shows ("RAM cache, max % of RAM"). It is the
   * only RAM budget the panel sends: `--cache-memory-percent`, which bounds the
   * memory-aware prefix cache (RAM only) and the paged RAM blocks' resident
   * bytes (RAM + SSD). The CLI still accepts --cache-memory-mb / --cache-ttl-minutes
   * / the entry-count flags for scripts; the panel never emits them, so no hidden
   * value can override the percent the label promises.
   */
  cacheMemoryPercent?: unknown
  effectivePagedCacheBlockSize?: unknown
  maxCacheBlocks?: unknown
  diskCacheDir?: string
  diskCacheMaxGb?: unknown
  blockDiskCacheDir?: string
  blockDiskCacheMaxGb?: unknown
  blockDiskCacheMaxPercent?: unknown
  /**
   * Exact typed prompt snapshots (openPangu, typed GLM) keep their whole-prompt SSD records in the managed aggregate
   * pool, budgeted by the same SSD % (or explicit block GB) as the block store and visible to the SSD pool notice.
   * Their former flat 10 GB budget held ~3 openPangu records and evicted on almost every turn.
   */
  typedPromptSsdPool?: boolean
}

export interface CacheLaunchArgsResult {
  args: string[]
  policy: CacheLaunchPolicy
  blockDiskOnly: boolean
}

/**
 * Build the authoritative cache-tier argv fragment.
 *
 * This is intentionally consumed by BOTH SessionSettings' visible command
 * preview and sessions.ts' spawned argv. Before this helper existed those two
 * hand-maintained blocks had already drifted in two user-visible ways:
 *
 *  - the preview emitted `--block-disk-cache-max-gb 0`, which means unlimited,
 *    while the launcher omitted the unset legacy value and honored the percent;
 *  - the preview could omit `--disable-block-disk-cache` while the launcher
 *    emitted it to override the engine's default-on SSD tier.
 *
 * The two panel choices map to four launches:
 *   SSD only  : `--no-paged-cache --enable-block-disk-cache --block-disk-cache-max-percent P`
 *   RAM only  : `--no-paged-cache --disable-block-disk-cache --cache-memory-percent R/100` (memory-aware prefix cache)
 *   RAM + SSD : `--use-paged-cache --cache-memory-percent R/100 --enable-block-disk-cache --block-disk-cache-max-percent P`
 *   both off  : `--no-paged-cache --disable-prefix-cache --disable-block-disk-cache`
 * `--use-paged-cache` only ever comes from the RAM choice, and only together with the block store.
 */
export function buildCacheLaunchArgs(input: CacheLaunchArgsInput): CacheLaunchArgsResult {
  const policy = resolveCacheLaunchPolicy(input)
  // These tokens are the retained-RAM contract (no paged RAM blocks unless the user turned RAM cache on next to SSD,
  // no media or SSM RAM caches). Keep them ahead of every early return so disabled prefix caching cannot accidentally
  // fall back to an engine default that re-enables a hidden RAM tier.
  const args: string[] = [
    policy.effectiveUsePagedCache ? '--use-paged-cache' : '--no-paged-cache',
    '--no-vision-memory-cache',
    '--ssm-state-cache-size', '0',
    '--ssm-state-cache-mb', '0',
  ]
  const blockDiskOnly = policy.enableBlockDiskCache && !policy.effectiveUsePagedCache

  if (policy.prefixCacheOff) {
    args.push('--disable-prefix-cache')
    if (!input.nativeSpeculativeCache) return { args, policy, blockDiskOnly }
  }

  if (policy.enableLegacyDiskCache) {
    // Exact typed prompt snapshots use the memory-aware restore adapter, but
    // completed payloads belong on SSD just like native block records. Zero
    // explicitly disables its retained L1; omitting it enables the RAM default.
    args.push('--cache-memory-mb', '0')
  } else if (policy.ramCacheActive) {
    // RAM only: MemoryCacheConfig.max_memory_percent (memory_cache.py) = this share of the RAM available at startup,
    // LRU-evicted (oldest first) when exceeded. RAM + SSD: the same percent is the paged RAM blocks' resident ceiling
    // (resolve_paged_resident_policy -> PagedCacheManager max_resident_bytes). The engine default (0.15) applies when
    // the saved value is unusable, which is the same 15 the panel shows.
    const cacheMemoryPercent = finitePositiveNumber(input.cacheMemoryPercent)
    if (cacheMemoryPercent != null) {
      args.push('--cache-memory-percent', (cacheMemoryPercent / 100).toString())
    }
  }

  if (policy.enableBlockDiskCache && !input.nativeSpeculativeCache) {
    // Not user-facing any more: the saved/default block geometry still travels so a session launches exactly as it
    // was proven (DSV4 256-token blocks, generic 64) and the index keeps its 262k-token / 1M-token capacity.
    const blockSize = finitePositiveInteger(input.effectivePagedCacheBlockSize)
    if (blockSize != null) args.push('--paged-cache-block-size', blockSize.toString())
    const maxCacheBlocks = finitePositiveInteger(input.maxCacheBlocks)
    if (maxCacheBlocks != null) args.push('--max-cache-blocks', maxCacheBlocks.toString())
  }

  if (policy.enableLegacyDiskCache) {
    args.push('--enable-disk-cache')
    if (input.diskCacheDir) args.push('--disk-cache-dir', input.diskCacheDir)
    if (input.typedPromptSsdPool) {
      // Pool budget = the block store's: an explicit positive block GB, else the SSD % slider (0 = unlimited).
      args.push('--disk-cache-pool')
      if (input.blockDiskCacheDir) args.push('--block-disk-cache-dir', input.blockDiskCacheDir)
      const poolGb = finiteNonNegativeNumber(input.blockDiskCacheMaxGb)
      if (poolGb != null && poolGb > 0) args.push('--block-disk-cache-max-gb', poolGb.toString())
      const poolPercent = finiteNonNegativeNumber(input.blockDiskCacheMaxPercent)
      if (poolPercent != null) args.push('--block-disk-cache-max-percent', poolPercent.toString())
    } else {
      const diskCacheMaxGb = finiteNonNegativeNumber(input.diskCacheMaxGb)
      if (diskCacheMaxGb != null) args.push('--disk-cache-max-gb', diskCacheMaxGb.toString())
    }
  }

  if (policy.enableBlockDiskCache) {
    args.push('--enable-block-disk-cache')
    if (input.blockDiskCacheDir) args.push('--block-disk-cache-dir', input.blockDiskCacheDir)

    // A stale zero in the removed GB control means "unset" in product state,
    // but means UNLIMITED to the engine. Only a positive legacy GB value is an
    // explicit cap. Zero/unlimited is represented by the still-visible percent
    // control, whose zero has the same engine meaning without overriding itself.
    const blockDiskCacheMaxGb = finiteNonNegativeNumber(input.blockDiskCacheMaxGb)
    if (blockDiskCacheMaxGb != null && blockDiskCacheMaxGb > 0) {
      args.push('--block-disk-cache-max-gb', blockDiskCacheMaxGb.toString())
    }
    const blockDiskCacheMaxPercent = finiteNonNegativeNumber(input.blockDiskCacheMaxPercent)
    if (blockDiskCacheMaxPercent != null) {
      args.push('--block-disk-cache-max-percent', blockDiskCacheMaxPercent.toString())
    }
  } else {
    // The engine defaults compatible SSD L2 on. A visible user opt-out must be
    // explicit in both the preview and the spawned command.
    args.push('--disable-block-disk-cache')
  }

  return { args, policy, blockDiskOnly }
}
