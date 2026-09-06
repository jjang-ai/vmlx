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

/** Fetch the capability block from a running image server; null when the engine has none (older engine) or on error. */
export async function fetchImageCapabilities(port: number): Promise<ImageCapabilities | null> {
  try {
    const resp = await fetch(`http://127.0.0.1:${port}/health`)
    if (!resp.ok) return null
    const body = await resp.json()
    return body && typeof body === 'object' && body.image && typeof body.image === 'object' ? (body.image as ImageCapabilities) : null
  } catch {
    return null
  }
}
