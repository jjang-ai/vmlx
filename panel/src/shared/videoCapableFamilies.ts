/**
 * Which local sessions get the per-request video controls (fps / frames /
 * pixels / token budget) in the settings drawer.
 *
 * Two sources decide it, and either one is sufficient:
 *
 * 1. The bundle's own `capabilities.modalities` sidecar (jang_config.json).
 *    This is the SAME artifact the engine reads in
 *    `_bundle_declares_native_video` before it advertises "video" in
 *    /capabilities, so a bundle that says video gets the controls without the
 *    panel having to know the family at all.
 * 2. A registry family known to run video through the engine's native video
 *    path or its sampled-frame fallback. This list exists for bundles whose
 *    sidecar predates the modalities field.
 *
 * Neither source is a guard: an unlisted family with a video-declaring sidecar
 * shows the controls, and a listed family shows them even when the sidecar is
 * silent. Muse Glimmer was missing from the list while the registry already
 * described it as "vision + video" and the engine served its videos — the
 * drawer hid the controls on a session that honoured them over the API.
 */
export const RUNTIME_VIDEO_CAPABLE_FAMILIES: readonly string[] = Object.freeze([
  'qwen3-vl',
  'qwen3.5',
  'qwen3.5-moe',
  // Qwen3.8 Flash-Next (engine family qwen4_exp): native video path,
  // live-proven with per-request fps / frame / pixel / token controls.
  'qwen4-exp',
  'qwen4_exp',
  'qwen2-vl',
  'gemma4',
  'nemotron-h',
  'mistral3',
  'mistral4',
  'pixtral',
  'kimi-k25',
  // Muse Glimmer: registry family "muse-glimmer" (engine muse_glimmer),
  // vision + video, <|video|> placeholder in its template.
  'muse-glimmer',
  'muse_glimmer',
])

export function modalitiesIncludeVideo(modalities?: readonly string[] | null): boolean {
  if (!Array.isArray(modalities)) return false
  return modalities.some((item) => String(item || '').toLowerCase() === 'video')
}

/**
 * `normalizedFamily` is the registry spelling (run the engine family through
 * `normalizeDetectedFamilyName` first); `runtimeModalities` is the bundle's
 * declared modality list when detection could read one.
 */
export function isRuntimeVideoCapable(input: {
  normalizedFamily?: string
  runtimeModalities?: readonly string[] | null
  liveRuntimeModalities?: readonly string[] | null
}): boolean {
  // A source-matched live report wins over offline family/sidecar guesses.
  // Missing metadata is unknown, not a claim that the model is text-only.
  if (Array.isArray(input.liveRuntimeModalities)) {
    return modalitiesIncludeVideo(input.liveRuntimeModalities)
  }
  if (modalitiesIncludeVideo(input.runtimeModalities)) return true
  return RUNTIME_VIDEO_CAPABLE_FAMILIES.includes(input.normalizedFamily || '')
}
