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
 *   - `dsv4PrefixCache` -> `VMLX_DSV4_ENABLE_PREFIX_CACHE=1` for native
 *     SWA+CSA/HCA composite prefix reuse.
 *   - `dsv4PoolQuant` -> `DSV4_POOL_QUANT=1` when native composite prefix
 *     cache is active. Generic TurboQuant KV is still suppressed for DSV4
 *     composite cache.
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
    const prefixEnabled = config.dsv4PrefixCache !== false
    const poolQuantEnabled = prefixEnabled && config.dsv4PoolQuant !== false
    env.DSV4_LONG_CTX = '1'
    env.DSV4_POOL_QUANT = poolQuantEnabled ? '1' : '0'
    if (prefixEnabled) {
      env.VMLX_DSV4_ENABLE_PREFIX_CACHE = '1'
    }
  }

  return env
}
