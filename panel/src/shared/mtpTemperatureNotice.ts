/**
 * Explain how the selected native-MTP mode affects sampling.
 *
 * Auto uses greedy startup defaults but honors explicit request sampling.
 * Deterministic intentionally pins greedy values for every request.
 * Off preserves sampling too, but disables native MTP entirely.
 *
 * States worth surfacing, all keyed on the model ACTUALLY having MTP:
 *  - `default`  Auto currently shows greedy, without enforcing it.
 *  - `pinned`   Deterministic mode forces greedy sampling.
 *  - `active`   Auto mode is using the displayed sampling temperature.
 *  - `inactive` a stale nonzero value contradicts Deterministic mode.
 */

export type MtpTemperatureNoticeKind = 'default' | 'pinned' | 'active' | 'inactive'

export interface MtpTemperatureNotice {
  kind: MtpTemperatureNoticeKind
  /** The temperature that caused an `inactive` verdict. */
  temperature?: number
}

export interface MtpTemperatureNoticeInput {
  /** True only when the bundle really carries native MTP heads. */
  nativeMtpSupported: boolean
  /** Session `nativeMtpMode`: 'auto' | 'deterministic' | 'off'. */
  mode: string | undefined
  /** Effective temperature shown in the box. */
  temperature: number | undefined
}

/**
 * Read `nativeMtpMode` out of a session config (stored as a JSON string).
 * Anything unreadable or absent means the app default, which is `auto`.
 */
export function parseSessionNativeMtpMode(
  sessionConfig: string | Record<string, unknown> | undefined,
): string {
  if (!sessionConfig) return 'auto'
  let parsed: unknown = sessionConfig
  if (typeof sessionConfig === 'string') {
    try {
      parsed = JSON.parse(sessionConfig)
    } catch {
      return 'auto'
    }
  }
  if (!parsed || typeof parsed !== 'object') return 'auto'
  const mode = (parsed as Record<string, unknown>).nativeMtpMode
  return typeof mode === 'string' && mode ? mode : 'auto'
}

export function resolveMtpTemperatureNotice(
  input: MtpTemperatureNoticeInput,
): MtpTemperatureNotice | null {
  // No MTP heads -> temperature has nothing to do with MTP. Say nothing.
  if (!input.nativeMtpSupported) return null

  const mode = typeof input.mode === 'string' && input.mode ? input.mode : 'auto'
  // Explicitly disabled by the user: temperature is unrelated to MTP.
  if (mode === 'off') return null

  const temperature = input.temperature
  if (temperature == null) return null
  if (mode === 'deterministic') {
    if (temperature > 0) return { kind: 'inactive', temperature }
    return { kind: 'pinned' }
  }
  // Auto launches with the deterministic-defaults policy: the STARTUP
  // DEFAULT is pinned to greedy, but an explicit nonzero temperature the
  // user set is still honored by the engine (API kwargs win).
  if (temperature > 0) return { kind: 'active', temperature }
  return { kind: 'default' }
}
