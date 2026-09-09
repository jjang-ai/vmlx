import { resolveImageModelFromDirectoryName } from './imageModels'

/** Discovery eligibility, NOT a promise that every tensor loads or output is good.
 * Reuse the supported family registry; require mflux metadata and model weights.
 * Local preflight remains the authority for a downloaded bundle.
 */
export function isMfluxImageCandidate(value: unknown): boolean {
  if (!value || typeof value !== 'object') return false
  const m = value as Record<string, any>
  const id = m.modelId || m.id
  if (typeof id !== 'string' || !/^[^/\s]+\/[^/\s]+$/.test(id)) return false
  const tags: string[] = Array.isArray(m.tags) ? m.tags.filter((t: unknown) => typeof t === 'string') : []
  const lower = tags.map(t => t.toLowerCase())
  if (m.library_name !== 'mflux' && !lower.includes('mflux')) return false
  if (m.pipeline_tag && !['text-to-image', 'image-to-image'].includes(m.pipeline_tag)) return false
  if (lower.some(t => ['gguf', 'exl3', 'nvfp4', 'lora', 'adapter', 'controlnet'].includes(t))) return false
  if (/(?:^|[-_/])(gguf|exl3|nvfp4|lora|controlnet)(?:[-_/.]|$)/i.test(id)) return false
  const bases = Array.isArray(m.cardData?.base_model) ? m.cardData.base_model : [m.cardData?.base_model]
  const names = [id, ...bases, ...tags.filter(t => t.startsWith('base_model:')).map(t => t.split(':').pop())]
  const supported = names.some(name => typeof name === 'string' &&
    resolveImageModelFromDirectoryName(name.replace(/-mflux(?:-(?:q\d+|\d+bit))?$/i, '')))
  if (!supported) return false
  const files: string[] = Array.isArray(m.siblings)
    ? m.siblings.map((s: any) => s?.rfilename).filter((s: unknown) => typeof s === 'string') : []
  // A card, LoRA or generic pipeline tag is not a complete diffusion model.
  const roots = new Set(files.flatMap(f => {
    const match = /^((?:.*\/)?)transformer\/[^/]+\.safetensors$/.exec(f)
    return match ? [match[1]] : []
  }))
  return files.some(f => [...roots].some(root => f.startsWith(root + 'text_encoder/') &&
    /^text_encoder\/[^/]+\.safetensors$/.test(f.slice(root.length))))
}

export function mfluxImageSearchParams(params: URLSearchParams): URLSearchParams {
  const image = new URLSearchParams(params)
  image.set('filter', 'mflux')
  // Explicit expansion preserves the metadata needed for eligibility and display.
  for (const field of ['tags', 'library_name', 'pipeline_tag', 'siblings', 'cardData', 'downloads', 'likes', 'lastModified', 'createdAt']) {
    image.append('expand[]', field)
  }
  return image
}
