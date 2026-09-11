/**
 * Canonical JIT (mx.compile) suppression policy.
 *
 * This rule previously existed as three hand-copied boolean expressions -- the
 * renderer checkbox in SessionConfigForm, the launch-argument builder in
 * SessionSettings, and the main-process builder in sessions.ts. Two of them
 * spelled the multimodal condition `isVLM` while the third spelled it
 * `multimodalActive`, so the checkbox a user saw and the flag the engine
 * received were computed by separate code paths that could drift apart.
 *
 * Every consumer now calls `computeEffectiveJit` so the displayed state and the
 * launched state cannot disagree.
 *
 * Measured 2026-08-12: JIT is worth ~0.3% on decode (Nanbeige-3B 81.9 vs 82.2
 * t/s; Gemma-4-E4B 94.6 vs 94.3 t/s) and costs TTFT on the VLM (0.33s vs
 * 0.19s) because of the warmup pass. The suppression list is therefore about
 * correctness/compatibility, not speed -- do not treat lifting an entry here as
 * a performance win without an A/B that logs the JIT lines.
 */
/** Missing legacy/new-session values use the same default as the settings UI. */
export function resolveRequestedJit(saved: boolean | undefined): boolean {
  return saved !== false
}

export interface JitSuppressionInput {
  /** The saved session toggle. */
  enableJitRequested: boolean
  /** Vision/multimodal bundle (`isVLM` / `multimodalActive` at the call sites). */
  isMultimodal: boolean
  flashMoeActive: boolean
  distributedActive: boolean
  dsv4Active: boolean
  m3Active: boolean
  zayaCcaActive: boolean
  turboQuantActive: boolean
  lagunaMixedSwaTurboQuantActive: boolean
  hybridCacheActive: boolean
}

/** Runtimes that own their own compiled/fused kernels or path-dependent cache. */
export function jitSuppressionReason(
  input: JitSuppressionInput,
): string | null {
  if (!input.enableJitRequested) return "disabled"
  if (input.isMultimodal) return "multimodal"
  if (input.flashMoeActive) return "flash_moe"
  if (input.distributedActive) return "distributed"
  if (input.dsv4Active) return "dsv4"
  if (input.m3Active) return "m3"
  if (input.zayaCcaActive) return "zaya_cca"
  if (input.turboQuantActive) return "turboquant"
  if (input.lagunaMixedSwaTurboQuantActive) return "laguna_mixed_swa_turboquant"
  if (input.hybridCacheActive) return "hybrid_cache"
  return null
}

export function computeEffectiveJit(input: JitSuppressionInput): boolean {
  return jitSuppressionReason(input) === null
}

/**
 * Runtime-only suppression, ignoring the saved toggle.
 *
 * The checkbox needs this separately from `computeEffectiveJit`: `checked`
 * depends on the toggle, while `disabled`, the dimmed styling and the
 * incompatibility warning depend only on whether some runtime owns its own
 * kernels. Keeping both derived from one condition list stops the box from
 * rendering enabled while the launcher suppresses it (or vice versa).
 */
export function isJitSuppressedByRuntime(
  input: Omit<JitSuppressionInput, "enableJitRequested">,
): boolean {
  return jitSuppressionReason({ ...input, enableJitRequested: true }) !== null
}
