import { normalizeNativeMtpMode } from './nativeMtpLaunchArgs'

/**
 * Explain how Adaptive MTP relates to the displayed sampling temperature.
 *
 * Adaptive MTP never changes the sampler: greedy requests verify by identity,
 * sampled requests verify by rejection sampling. The only thing worth telling
 * the user is therefore that a nonzero temperature is honored and verified
 * stochastically. Off and non-MTP bundles say nothing.
 */

export type MtpTemperatureNoticeKind = 'active'

export interface MtpTemperatureNotice {
  kind: MtpTemperatureNoticeKind
  temperature: number
}

export interface MtpTemperatureNoticeInput {
  /** Remote connections do not own or enforce the server's startup mode. */
  isRemote?: boolean
  /** True only when the bundle really carries native MTP heads. */
  nativeMtpSupported: boolean
  /** Loaded runtime can disable MTP, including when an external drafter wins. */
  runtimeAvailable?: boolean
  /** Session `nativeMtpMode` (new or legacy spelling). */
  mode: string | undefined
  /** Bundle-level measured default from model detection. */
  modelDefaultMode?: 'auto' | 'off'
  /** Effective temperature shown in the box. */
  temperature: number | undefined
}

/**
 * Read the adapter's saved choice and normalize legacy spellings.
 * Absent/unreadable means "not chosen"; legacy native defaults do not opt
 * a separate LiLiCorr drafter in.
 */
export function parseSessionNativeMtpMode(
  sessionConfig: string | Record<string, unknown> | undefined,
  adapter?: string,
): 'adaptive' | 'off' | undefined {
  if (!sessionConfig) return undefined
  let parsed: unknown = sessionConfig
  if (typeof sessionConfig === 'string') {
    try {
      parsed = JSON.parse(sessionConfig)
    } catch {
      return undefined
    }
  }
  if (!parsed || typeof parsed !== 'object') return undefined
  const key = adapter === 'nemotron_lilicorr' ? 'lilicorrMode' : 'nativeMtpMode'
  return normalizeNativeMtpMode((parsed as Record<string, unknown>)[key])
}

export function resolveMtpTemperatureNotice(
  input: MtpTemperatureNoticeInput,
): MtpTemperatureNotice | null {
  if (input.isRemote || !input.nativeMtpSupported || input.runtimeAvailable === false) return null
  const mode = input.mode === 'off'
    ? 'off'
    : input.mode === 'adaptive' || input.mode === 'deterministic'
      ? 'adaptive'
      : input.modelDefaultMode === 'off' ? 'off' : 'adaptive'
  if (mode === 'off') return null
  const temperature = input.temperature
  if (temperature == null || !(temperature > 0)) return null
  return { kind: 'active', temperature }
}
