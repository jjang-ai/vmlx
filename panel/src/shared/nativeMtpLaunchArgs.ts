import { finitePositiveInteger } from './launchArgValues'

export interface NativeMtpLaunchPolicyInput {
  supported: boolean
  detectedDepth?: number
  configuredDepth?: number
  depthOverride?: boolean
  mode?: 'auto' | 'deterministic' | 'off'
  modelDefaultMode?: 'auto' | 'off'
  externalSpeculativeActive?: boolean
}

export function resolveNativeMtpMode(
  input: Pick<NativeMtpLaunchPolicyInput, 'mode' | 'modelDefaultMode' | 'depthOverride'>,
): 'auto' | 'deterministic' | 'off' {
  const configured = input.mode || 'auto'
  // GLM-5.3 is measured slower under its current verifier, so its bundle-level
  // Auto policy is AR. A fixed D1-D3 selection is an explicit opt-in and must
  // remain available for measurement/tuning.
  if (
    configured === 'auto'
    && input.modelDefaultMode === 'off'
    && input.depthOverride !== true
  ) return 'off'
  return configured
}

/** The form and launcher must display/use the same supported fixed ceiling. */
export function resolveFixedNativeMtpDepth(configuredDepth?: number, detectedDepth?: number): number {
  return Math.max(1, Math.min(3,
    finitePositiveInteger(configuredDepth) || finitePositiveInteger(detectedDepth) || 1,
  ))
}

/** One source of truth for Electron preview and the process launcher. */
export function buildNativeMtpLaunchArgs(
  input: NativeMtpLaunchPolicyInput,
): string[] {
  if (!input.supported) return []
  const mode = resolveNativeMtpMode(input)
  if (mode === 'off' || input.externalSpeculativeActive) {
    return ['--disable-native-mtp']
  }

  // Native MTP enabled in Server settings pins the STARTUP DEFAULTS to
  // greedy (temperature 0, top_p 1, top_k dropped, min_p 0) for every
  // surface — chat settings and API alike — while an EXPLICIT request
  // temperature in API kwargs still wins (engine deterministic-defaults
  // policy). Deterministic mode goes further and hard-pins greedy for every
  // request (greedy-only). 'auto' previously sent compatible-only, which
  // silently kept the bundle's sampled temperature as the default and made
  // every app-managed MTP session run stochastic rejection sampling.
  const samplingArgs = [
    '--native-mtp-sampling-policy',
    mode === 'deterministic' ? 'greedy-only' : 'deterministic-defaults',
  ]
  if (input.depthOverride !== true) {
    // Adaptive policy: emit NO explicit depth. --native-mtp-depth becomes
    // the engine's explicit VMLINUX_NATIVE_MTP_DEPTH override, which pins
    // the start depth and bypasses tuning sidecars, bundle stamps, and the
    // session-scoped adaptive profile. This launcher used to always send
    // it with detectedDepth — and detectedDepth comes from
    // mtp_num_hidden_layers, a HEAD LAYER COUNT (1), not a draft depth —
    // so every "adaptive" app session was silently pinned to fixed D1
    // (live-proven on Flash-Next JANG_4M: effective_depth=1
    // source=VMLINUX_NATIVE_MTP_DEPTH).
    return ['--native-mtp-depth-policy', 'adaptive', ...samplingArgs]
  }
  const depth = resolveFixedNativeMtpDepth(input.configuredDepth, input.detectedDepth)
  return [
    '--native-mtp-depth',
    depth.toString(),
    '--native-mtp-depth-policy',
    'fixed',
    ...samplingArgs,
  ]
}
