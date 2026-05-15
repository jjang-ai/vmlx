/**
 * DSV4 Flash runtime env mapping.
 *
 * DSV4-specific knobs are exposed via `VMLX_*` environment variables on the
 * vmlx-engine subprocess. The panel UI accepts them as session-config fields;
 * this helper normalizes them to the env dict that gets merged into spawnEnv.
 *
 * Knobs:
 *   - `dsv4PoolQuant` -> `DSV4_POOL_QUANT=1` — experimental native CSA/HCA
 *     pool codec. Production DSV4 launches explicitly set `DSV4_POOL_QUANT=0`
 *     unless this is enabled.
 *
 * Helper only emits runtime cache/diagnostic env vars. Reasoning mode is
 * carried by the normal request/API path.
 */

export interface Dsv4EnvConfig {
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
    env.DSV4_POOL_QUANT = config.dsv4PoolQuant === true ? '1' : '0'
  }

  return env
}
