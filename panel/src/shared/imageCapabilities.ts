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
  /** generate_image takes a negative prompt (Flux1, QwenImage, QwenImageEdit do; Klein does not). */
  negative_prompt?: boolean
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
  if (health.status === 'standby_soft' || health.status === 'standby_deep') return { status: 'standby', capabilities: null }
  const image = health.image && typeof health.image === 'object' && !Array.isArray(health.image)
    ? health.image as ImageCapabilities : null
  const loaded = image ? image.loaded === true && health.model_loaded !== false
    : health.model_type === 'image' && health.model_loaded === true
  return health.status === 'healthy' && loaded
    ? { status: 'running', capabilities: image }
    : { status: 'error', capabilities: null }
}
