/**
 * DSV4 Flash runtime env mapping.
 *
 * DSV4-specific knobs are exposed via `VMLX_*` environment variables on the
 * vmlx-engine subprocess. The panel UI accepts them as session-config fields;
 * this helper normalizes them to the env dict that gets merged into spawnEnv.
 *
 * Knobs:
 *   - `dsv4FinalizerTokens` and `dsv4ForceDirect` are retained only for
 *     migration compatibility with older saved sessions. They intentionally
 *     do not emit env vars: the runtime must not inject thinking tags or
 *     silently flip requested reasoning rails.
 *   - `dsv4PrefixCache` -> `VMLX_DSV4_ENABLE_PREFIX_CACHE=1` — diagnostic
 *     opt-in for native SWA+CSA/HCA composite prefix reuse. Default off until
 *     deterministic cached-vs-no-cache equivalence is proven.
 *   - `dsv4PoolQuant` is retained for old-session migration only. The live
 *     pool codec is disabled because even after append-only writes, attention
 *     reads still dequantize/concatenate the historical CSA/HCA pool during
 *     decode; production DSV4 launches always set `DSV4_POOL_QUANT=0`.
 *
 * Natural model behavior wins: bundle chat/generation config plus explicit
 * per-request controls are the only model-behavior inputs.
 */

export interface Dsv4EnvConfig {
  /** Kept for config migration compatibility; raw max is no longer env-gated. */
  dsv4RawMax?: boolean
  dsv4FinalizerTokens?: number
  dsv4ForceDirect?: boolean
  dsv4PrefixCache?: boolean
  dsv4PoolQuant?: boolean
}

export interface Dsv4EnvOptions {
  dsv4Active?: boolean
}

export function dsv4EnvFromConfig(
  config: Dsv4EnvConfig | null | undefined,
  options: Dsv4EnvOptions = {},
): Record<string, string> {
  if (!config) return {}
  const env: Record<string, string> = {}

  if (options.dsv4Active === true) {
    env.DSV4_LONG_CTX = '1'
    env.DSV4_POOL_QUANT = '0'
    if (config.dsv4PrefixCache === true) {
      env.VMLX_DSV4_ENABLE_PREFIX_CACHE = '1'
    }
  }

  return env
}
