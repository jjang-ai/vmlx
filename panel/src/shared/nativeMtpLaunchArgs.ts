/**
 * Native in-model MTP: ONE user decision, two values.
 *
 *   'off'      — plain autoregressive decode (AR). The bundle's MTP head is
 *                loaded nowhere; the engine receives --disable-native-mtp.
 *   'adaptive' — the engine's adaptive speculative policy. The engine chooses
 *                the draft depth itself: it measures the completed verify-cycle
 *                cost against its own AR baseline, promotes toward the verifier
 *                ceiling when a deeper rung wins, steps down (-> D1 -> AR) when a
 *                window loses, and re-probes with backoff. Bundle stamps, tuning
 *                sidecars and family defaults only pick the starting rung.
 *                The request/bundle sampler is never changed: greedy requests
 *                use identity verification, sampled requests use
 *                rejection-sampling acceptance (`compatible-only`).
 *
 * There is deliberately NO user-facing depth (D1/D2/D3), no "fixed" policy and
 * no "deterministic" greedy override any more (Eric, 2026-10-04: "users no
 * longer have adaptive/d1/d2/d3 buttons but only ar (mtp off) and adaptive
 * mtp"). The engine keeps `--native-mtp-depth N --native-mtp-depth-policy
 * fixed` as a measurement lever for benchmarks; the app never emits it.
 *
 * Legacy persisted values ('auto', 'deterministic', nativeMtpDepth,
 * nativeMtpDepthOverride, nativeMtpAutoSamplingPolicy) are read-only history:
 * 'auto'/'deterministic' normalize to 'adaptive'; the depth/override/sampling
 * fields are ignored everywhere. A saved deterministic session therefore
 * stops forcing temperature 0 on every request — that enforcement was a
 * sampler clamp hidden inside an MTP control.
 */

export type NativeMtpMode = 'adaptive' | 'off'

/** Values that may still exist in persisted session rows. */
export type LegacyNativeMtpMode = NativeMtpMode | 'auto' | 'deterministic'

/** The only sampling policy the app ever launches. */
export const NATIVE_MTP_SAMPLING_POLICY = 'compatible-only' as const

/**
 * Map any stored/legacy/malformed value onto the two product modes.
 * `undefined` stays `undefined` so callers can tell "not chosen" from "off".
 */
export function normalizeNativeMtpMode(raw: unknown): NativeMtpMode | undefined {
  if (raw === undefined || raw === null || raw === '') return undefined
  if (raw === 'off') return 'off'
  if (raw === 'adaptive' || raw === 'auto' || raw === 'deterministic') return 'adaptive'
  return undefined
}

/**
 * Fresh/reset session default. An explicit saved choice always wins.
 *
 * Adaptive is the default for every bundle that passed the runtime gate: the
 * AR-safety valve guarantees a losing request falls back to plain decode, so
 * there is no family-level reason to start Off. A bundle whose measured
 * verifier is slower than AR declares `defaultMode: 'off'` through model
 * detection (GLM-5.3 today); that is a model-derived default, not a user
 * setting, and the user can still choose Adaptive explicitly.
 */
export function resolveNativeMtpStartupMode(
  _family?: string,
  configured?: unknown,
  modelDefaultMode?: 'auto' | 'off',
): NativeMtpMode {
  const explicit = normalizeNativeMtpMode(configured)
  if (explicit !== undefined) return explicit
  return modelDefaultMode === 'off' ? 'off' : 'adaptive'
}

export interface NativeMtpLaunchPolicyInput {
  supported: boolean
  /** Session `nativeMtpMode` (new or legacy spelling). */
  mode?: unknown
  /** Bundle-level measured default from model detection. */
  modelDefaultMode?: 'auto' | 'off'
  externalSpeculativeActive?: boolean
}

/**
 * Effective mode for a session.
 *
 * A legacy 'auto' row on a bundle that declares `defaultMode: 'off'` keeps its
 * old meaning (AR): that row never recorded an explicit opt-in. A row that
 * stores the new 'adaptive' value IS an explicit opt-in and runs adaptive even
 * on such a bundle. 'deterministic' was always an explicit MTP opt-in.
 */
export function resolveNativeMtpMode(
  input: Pick<NativeMtpLaunchPolicyInput, 'mode' | 'modelDefaultMode'>,
): NativeMtpMode {
  const raw = input.mode
  if (raw === 'off') return 'off'
  if (raw === 'adaptive' || raw === 'deterministic') return 'adaptive'
  // undefined / legacy 'auto' / unknown: the model-derived default decides.
  return input.modelDefaultMode === 'off' ? 'off' : 'adaptive'
}

/** One source of truth for the Electron CLI preview and the process launcher. */
export function buildNativeMtpLaunchArgs(input: NativeMtpLaunchPolicyInput): string[] {
  if (!input.supported) return []
  const mode = resolveNativeMtpMode(input)
  if (mode === 'off' || input.externalSpeculativeActive) {
    return ['--disable-native-mtp']
  }
  // No --native-mtp-depth: that flag is the engine's explicit
  // VMLINUX_NATIVE_MTP_DEPTH override, which pins the start depth and bypasses
  // tuning sidecars and bundle stamps. The engine resolves the ceiling itself.
  return [
    '--native-mtp-depth-policy',
    'adaptive',
    '--native-mtp-sampling-policy',
    NATIVE_MTP_SAMPLING_POLICY,
  ]
}
