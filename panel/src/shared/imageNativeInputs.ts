/** Native upscaling accepts a shortest-edge pixel count or a scale factor. */
export function parseUpscaleResolution(value: unknown): number | string | undefined {
  if (typeof value === 'number') return Number.isSafeInteger(value) && value >= 2 ? value : undefined
  if (typeof value !== 'string') return undefined
  const text = value.trim()
  if (/^\d+$/.test(text)) return parseUpscaleResolution(Number(text))
  if (/^\d+(?:\.\d+)?x$/i.test(text) && Number(text.slice(0, -1)) > 0 && Number.isFinite(Number(text.slice(0, -1)))) return text
  return undefined
}

/** FIBO itself parses JSON; never turn plain prose into an invented schema. */
export function isNativeJsonPrompt(prompt: string): boolean {
  try { JSON.parse(prompt); return true } catch { return false }
}

export interface ImageEditInputs {
  prompt: string; model: string; imageBase64: string; maskBase64?: string
  width: number; height: number; steps: number; guidance: number; strength: number
  negativePrompt?: string; seed?: number
  operation?: 'upscale'; resolution?: number | string; softness?: number
}

/** Preserve the established edit wire contract, with a separate native upscale branch. */
export function imageEditBody(params: ImageEditInputs): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: params.model,
    image: params.imageBase64.replace(/^data:image\/[\w+.-]+;base64,/, ''),
    n: 1, response_format: 'b64_json',
  }
  if (params.seed != null) body.seed = params.seed
  if (params.operation === 'upscale') {
    const resolution = parseUpscaleResolution(params.resolution ?? 384)
    const softness = params.softness ?? 0
    if (resolution === undefined) throw new Error('Enter an upscale resolution in pixels or a positive scale such as 2x')
    if (!Number.isFinite(softness) || softness < 0 || softness > 1) throw new Error('Upscale softness must be between 0 and 1')
    body.resolution = resolution
    body.softness = softness
    return body
  }
  Object.assign(body, {prompt: params.prompt, size: `${params.width}x${params.height}`, steps: params.steps, guidance: params.guidance, strength: params.strength})
  if (params.negativePrompt) body.negative_prompt = params.negativePrompt
  if (params.maskBase64) body.mask = params.maskBase64.replace(/^data:image\/[\w+.-]+;base64,/, '')
  return body
}

/** Structured native captions remain one intact JSON document on variation turns. */
export function imagePromptForRequest(prompt: string, iteratePrompt: string | null | undefined, nativeJson: boolean): string {
  if (nativeJson) return prompt
  if (iteratePrompt != null) return prompt.trim() ? `${iteratePrompt}, ${prompt.trim()}` : iteratePrompt
  return prompt.trim()
}
