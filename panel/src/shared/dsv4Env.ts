/**
 * DSV4 Flash runtime env mapping.
 *
 * DSV4-specific knobs are exposed via `VMLX_*` environment variables on the
 * vmlx-engine subprocess. The panel UI accepts them as session-config fields;
 * this helper normalizes them to the env dict that gets merged into spawnEnv.
 *
 * Knobs:
 *   - `dsv4RawMax` -> `VMLX_DSV4_RAW_MAX=1` — opt in to genuine
 *     reasoning_effort=max template (default downgrades to high). See
 *     docs/internal/AUDIT_dsv4_flash_long_form_max_thinking_2026_05_09.md.
 *   - `dsv4FinalizerTokens` -> `VMLX_DSV4_FINALIZER_TOKENS=<int>` — visible
 *     answer budget after forced `</think>` (default 2048). Bump for long-form
 *     content where 2048 is still too tight.
 *   - `dsv4ForceDirect` -> `VMLX_DSV4_FORCE_DIRECT_RAIL=1` — disable thinking
 *     entirely (visible-only direct rail).
 *
 * Engine reconciles precedence (`forceDirect` short-circuits ahead of
 * `rawMax`); helper just emits the env vars.
 */

export interface Dsv4EnvConfig {
  dsv4RawMax?: boolean
  dsv4FinalizerTokens?: number
  dsv4ForceDirect?: boolean
}

export function dsv4EnvFromConfig(
  config: Dsv4EnvConfig | null | undefined,
): Record<string, string> {
  if (!config) return {}
  const env: Record<string, string> = {}

  if (config.dsv4RawMax === true) {
    env.VMLX_DSV4_RAW_MAX = '1'
  }

  if (typeof config.dsv4FinalizerTokens === 'number'
      && Number.isFinite(config.dsv4FinalizerTokens)
      && config.dsv4FinalizerTokens > 0) {
    env.VMLX_DSV4_FINALIZER_TOKENS = String(Math.floor(config.dsv4FinalizerTokens))
  }

  if (config.dsv4ForceDirect === true) {
    env.VMLX_DSV4_FORCE_DIRECT_RAIL = '1'
  }

  return env
}
