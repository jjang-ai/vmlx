/**
 * JANG profile and hybrid-family compatibility warnings for the converter UI.
 *
 * Background: GatedDeltaNet/Mamba conv1d weights are stored on disk as
 * (out, 1, kernel) but MLX Conv1d expects (out, kernel, 1). PyPI `jang <
 * 2.5.27` did not preflip on convert; mlx_lm direct loads of the resulting
 * bundles crash at the first conv1d layer with a "input channels" mismatch.
 *
 * vMLX engine has an idempotent load-time backstop
 * (_sanitize_grouped_conv1d_layout), but this is still a compatibility note:
 * direct mlx_lm/mlx_vlm users need a fresh conversion, and vMLX coherence still
 * depends on the normal live model gate.
 *
 * Cross-reference: docs/internal/AUDIT_jang_hybrid_conv1d_compat_2026_05_09.md
 */

export interface FamilyCompatEntry {
  /** JANG profile names (e.g. "JANG_2L") that should surface a warning. */
  profiles_warn: string[]
  /** User-facing warning message rendered in the UI. */
  message: string
}

const HYBRID_JANG_PROFILES = [
  'JANG_1L',
  'JANG_2S',
  'JANG_2M',
  'JANG_2L',
  'JANG_3S',
  'JANG_3M',
  'JANG_3L',
  'JANG_4S',
  'JANG_4M',
  'JANG_4L',
  'JANG_6M',
]

const HYBRID_LOAD_NOTE =
  'Hybrid GatedDeltaNet/Mamba conv1d may crash on direct mlx_lm/mlx_vlm load with jang < 2.5.27. ' +
  'vMLX applies a runtime grouped-Conv1d backstop, but this is not a coherence guarantee. ' +
  'For external loaders, re-convert with jang >= 2.5.27.'

export const KNOWN_INCOMPATIBLE_FAMILIES: Record<string, FamilyCompatEntry> = {
  qwen3_next: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  qwen3_5: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  qwen3_5_text: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  qwen3_5_moe: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  qwen3_5_moe_text: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  nemotron_h: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  nemotron_h_v2: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  mamba: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  mamba2: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  codestral_mamba: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  bailing_hybrid: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  bailing_moe_v2_5: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  lfm2_moe: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
  granitemoehybrid: {
    profiles_warn: HYBRID_JANG_PROFILES,
    message: HYBRID_LOAD_NOTE,
  },
}

/**
 * Returns a warning string for risky family/profile combos, or null.
 *
 * Pure function: no IPC, no DOM. Caller is responsible for fetching
 * `model_type` (via `window.api.models.detectConfig`) and the active
 * profile name from the JANG_PRESETS map.
 */
export function getJangCompatWarning(
  modelType: string | null | undefined,
  profile: string | null | undefined,
): string | null {
  if (!modelType || !profile) return null
  if (!profile.startsWith('JANG_')) return null
  const family = modelType.toLowerCase()
  const entry = KNOWN_INCOMPATIBLE_FAMILIES[family]
  if (!entry) return null
  if (!entry.profiles_warn.includes(profile)) return null
  return entry.message
}
