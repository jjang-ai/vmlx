/**
 * Canonical mapping from an ENGINE family_name to the panel's REGISTRY name.
 *
 * The engine and the panel deliberately spell some families differently
 * (`deepseek_v4` vs `deepseek-v4`, `zaya1_vl` vs `zaya1-vl`, `bailing_hybrid`
 * vs `ling`). Getting that translation wrong is not a cosmetic bug: a lookup
 * keyed on the wrong spelling silently misses, the setting appears to apply,
 * and the fix reads as a no-op — this project has shipped that failure before.
 *
 * These two rules previously existed as three byte-identical private copies
 * each (src/main/sessions.ts, SessionConfigForm.tsx, SessionSettings.tsx),
 * spanning the main AND renderer processes. They live here so a new family
 * alias is added once, and so main and renderer can never disagree about what
 * a family is called.
 */

/**
 * Engine `family_name` -> panel registry name.
 *
 * The engine and the registry do not spell families the same way, and the
 * registry is not internally consistent about it either: qwen3_5 -> "qwen3.5"
 * and nemotron_h -> "nemotron-h", but minimax_m3 and openpangu_v2 keep their
 * underscores because those ARE their registry names.
 *
 * The last four entries were missing, which only shows on the REMOTE path
 * (locally, detectModelConfigFromDir already returns registry names). Measured
 * consequence: SLOW_FAMILY_TIMEOUTS is keyed on registry names, so a remote
 * Qwen3.5 / Qwen3.5-MoE / Qwen3-Next / Nemotron-H session missed its 900s
 * entry and was aborted at the generic 300s while the remote engine — which
 * keeps its own copy in engine spelling, cli.py _SLOW_FAMILY_TIMEOUTS — served
 * happily to 900. That is the exact "Request timed out after 300s" symptom
 * slowFamilyTimeouts.ts documents as fixed.
 *
 * Passing an already-normalized registry name through is a no-op: no registry
 * name collides with an engine spelling.
 */
const ENGINE_FAMILY_TO_REGISTRY: Readonly<Record<string, string>> = Object.freeze({
  deepseek_v4: 'deepseek-v4',
  zaya1_vl: 'zaya1-vl',
  bailing_hybrid: 'ling',
  qwen3_5: 'qwen3.5',
  qwen3_5_moe: 'qwen3.5-moe',
  qwen3_next: 'qwen3-next',
  qwen4_exp: 'qwen4-exp',
  nemotron_h: 'nemotron-h',
  glm5_next: 'glm5-next',
  glm5_next_text: 'glm5-next',
  ernie4_5: 'ernie4.5',
})

export function normalizeDetectedFamilyName(family?: string): string | undefined {
  if (!family) return undefined
  return ENGINE_FAMILY_TO_REGISTRY[family] ?? family
}

/**
 * ZAYA's CCA cache is path-dependent, so several settings surfaces gate on it.
 * Always ask through the normalizer — the engine spells the VL variant
 * `zaya1_vl`, the registry spells it `zaya1-vl`.
 */
export function isZayaCcaFamily(family?: string): boolean {
  const normalized = normalizeDetectedFamilyName(family)
  return normalized === 'zaya' || normalized === 'zaya1-vl'
}

/**
 * Families whose architecture-native state is persisted as one exact typed
 * N-1 prompt snapshot. Generic content-addressed block records cannot
 * reconstruct their path-dependent recurrent/indexer state, so the product
 * must select prompt-level disk L2 and keep block-disk L2 off.
 */
export function usesExactTypedPromptDiskCache(family?: string): boolean {
  const normalized = normalizeDetectedFamilyName(family)
  return normalized === 'openpangu_v2' || normalized === 'glm5-next'
}
