/**
 * What a running image server's LOADED model actually accepts, as reported
 * by the engine on /health (`image` block). Derived there from the mflux class
 * and the model's generate signature, so the UI shows only effective controls.
 */
export interface ImageCapabilities {
  loaded: boolean
  model?: string | null
  mflux_class?: string
  quantize?: number | null
  mode?: 'generate' | 'edit'
  /** Adapter actually uses negative conditioning; a signature alone is insufficient (Flux1 accepts but ignores it). */
  negative_prompt?: boolean
  /** False only when the loaded adapter establishes that guidance is unused. */
  guidance?: boolean | null
  /** img2img strength on a generation model (the gallery's Iterate). */
  variation_strength?: boolean
  /** strength on an edit model: false for every edit class the engine accepts (their branches never forward it). */
  edit_strength?: boolean | null
  /** "required" for Flux Fill inpainting, otherwise "none". */
  mask?: 'required' | 'none'
  /** generation models return n images; edit always returns one. */
  count?: boolean
  error?: string
}

export type ImageServerStatus = 'stopped' | 'starting' | 'running' | 'standby' | 'error'

/** Keep non-negative finite user guidance. There is no universal upper bound:
 * the established Fill preset itself is 30. Empty/invalid input retains the
 * existing zero behavior rather than substituting an unrelated model default. */
export function imageGuidanceFromInput(raw: string): number {
  const value = Number.parseFloat(raw)
  return Number.isFinite(value) ? Math.max(0, value) : 0
}

/** Native mflux Config.init_time_step skips more denoising at higher values.
 * Scope this wording to the adapters whose source-preservation contract is
 * established; other adapters must not inherit an assumed direction. */
export function imageVariationPreservesSource(capabilities?: ImageCapabilities | null): boolean {
  return capabilities?.loaded === true && capabilities.variation_strength === true &&
    (capabilities.mflux_class === 'Flux1' || capabilities.mflux_class === 'ZImage')
}

/** HTTP liveness is not proof that diffusion weights are ready. */
export function imageRuntimeSnapshot(body: unknown): { status: ImageServerStatus; capabilities: ImageCapabilities | null } {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return { status: 'error', capabilities: null }
  const health = body as Record<string, any>
  if (health.wake_in_progress === true || health.status === 'no_model' || health.status === 'loading') {
    return { status: 'starting', capabilities: null }
  }
  const image = health.image && typeof health.image === 'object' && !Array.isArray(health.image)
    ? health.image as ImageCapabilities : null
  const loaded = image ? image.loaded === true && health.model_loaded !== false
    : health.model_type === 'image' && health.model_loaded === true
  // Sleeping is not request-ready, but a still-loaded adapter retains its
  // capabilities. Dropping these made unsupported edit controls reappear.
  if (health.status === 'standby_soft' || health.status === 'standby_deep') {
    return { status: 'standby', capabilities: loaded ? image : null }
  }
  return health.status === 'healthy' && loaded
    ? { status: 'running', capabilities: image }
    : { status: 'error', capabilities: null }
}
