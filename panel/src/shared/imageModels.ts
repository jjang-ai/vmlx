/**
 * Single source of truth for all image model definitions.
 *
 * Every model has explicit mfluxClass + mfluxName fields — NO regex or
 * directory name matching. The engine uses these directly to pick the
 * correct Python class and mflux ModelConfig.
 *
 * To add a new model:
 * 1. Add an entry here with the correct mfluxClass and mfluxName
 * 2. Add the class import in image_gen.py's MODEL_CLASS_MAP
 * 3. That's it — all UI/download/startup code reads from this registry
 */

export interface ImageModelDef {
  /** Canonical model ID used everywhere: 'schnell', 'qwen-image-edit', etc. */
  id: string
  /** Display name: 'Flux Schnell', 'Qwen Image Edit' */
  name: string
  /** UI description shown in the model picker */
  desc: string
  /** Generation or editing model */
  category: 'generate' | 'edit'
  /** Default inference steps */
  steps: number
  /** Display size string: '~6-24 GB' */
  size: string
  /** Allowed quantization options, e.g. [4, 8, 0] where 0 = full precision */
  quantizeOptions: number[]
  /** Maps quantize level -> HuggingFace repo ID for downloading */
  repoMap: Record<number, string>
  /** RAM tier hint for the UI */
  tier?: 'small' | 'medium' | 'large'
  /**
   * Explicit mflux Python class name — used by image_gen.py to import the correct class.
   * NO regex. NO directory name matching. This is the single source of truth.
   * Examples: 'Flux1', 'Flux2Klein', 'ZImage', 'Flux1Kontext', 'QwenImageEdit'
   */
  mfluxClass: string
  /**
   * Canonical mflux model name — passed to ModelConfig.from_name().
   * Must match one of the names in mflux's AVAILABLE_MODELS registry.
   * Examples: 'schnell', 'dev', 'z-image-turbo', 'flux2-klein-4b', 'dev-kontext'
   */
  mfluxName: string
  /** Whether this model supports img2img (source image + strength) */
  supportsImg2Img: boolean
}

/**
 * Master list of all supported image models.
 *
 * To add a new model, add an entry here. All UI components, download logic,
 * and validation will pick it up automatically.
 */
export const IMAGE_MODELS: ImageModelDef[] = [
  // ── Image Generation ──────────────────────────────────────────────────

  // Flux Schnell — fastest, 4 steps, dual encoder
  {
    id: 'schnell',
    name: 'Flux Schnell',
    desc: 'Fastest generation (4 steps)',
    category: 'generate',
    steps: 4,
    size: '~6-24 GB',
    quantizeOptions: [4, 8, 0],
    repoMap: {
      4: 'dhairyashil/FLUX.1-schnell-mflux-4bit',
      8: 'dhairyashil/FLUX.1-schnell-mflux-8bit',
      0: 'black-forest-labs/FLUX.1-schnell',
    },
    tier: 'small',
    mfluxClass: 'Flux1',
    mfluxName: 'schnell',
    supportsImg2Img: true,
  },

  // Z-Image Turbo — fast, 4 steps, single encoder
  {
    id: 'z-image-turbo',
    name: 'Z-Image Turbo',
    desc: 'Fast turbo generation (4 steps)',
    category: 'generate',
    steps: 4,
    size: '~6-24 GB',
    quantizeOptions: [4, 8, 0],
    repoMap: {
      4: 'filipstrand/Z-Image-Turbo-mflux-4bit',
      8: 'carsenk/z-image-turbo-mflux-8bit',
      0: 'Tongyi-MAI/Z-Image-Turbo',
    },
    tier: 'small',
    mfluxClass: 'ZImage',
    mfluxName: 'z-image-turbo',
    supportsImg2Img: true,
  },

  // Flux Dev — highest quality, 20 steps, dual encoder
  {
    id: 'dev',
    name: 'Flux Dev',
    desc: 'High quality generation (20 steps)',
    category: 'generate',
    steps: 20,
    size: '~6-24 GB',
    quantizeOptions: [4, 8, 0],
    repoMap: {
      4: 'dhairyashil/FLUX.1-dev-mflux-4bit',
      8: 'dhairyashil/FLUX.1-dev-mflux-8bit',
      0: 'black-forest-labs/FLUX.1-dev',
    },
    tier: 'medium',
    mfluxClass: 'Flux1',
    mfluxName: 'dev',
    supportsImg2Img: true,
  },

  // FLUX.2 Klein 4B — fast small model, single encoder, uses Flux2Klein class
  {
    id: 'klein-4b',
    name: 'FLUX.2 Klein 4B',
    desc: 'Fast & small (20 steps)',
    category: 'generate',
    steps: 20,
    size: '~4-8 GB',
    quantizeOptions: [4, 0],
    repoMap: {
      4: 'RunPod/FLUX.2-klein-4B-mflux-4bit',
      0: 'black-forest-labs/FLUX.2-klein-4B',
    },
    tier: 'small',
    mfluxClass: 'Flux2Klein',
    mfluxName: 'flux2-klein-4b',
    supportsImg2Img: true,
  },

  // FLUX.2 Klein 9B — medium quality, single encoder
  {
    id: 'klein-9b',
    name: 'FLUX.2 Klein 9B',
    desc: 'Medium quality (20 steps)',
    category: 'generate',
    steps: 20,
    size: '~8-18 GB',
    quantizeOptions: [0],
    repoMap: {
      0: 'black-forest-labs/FLUX.2-klein-9B',
    },
    tier: 'medium',
    mfluxClass: 'Flux2Klein',
    mfluxName: 'flux2-klein-9b',
    supportsImg2Img: true,
  },

  // Qwen Image (generation) — large model, good prompt understanding
  {
    id: 'qwen-image',
    name: 'Qwen Image',
    desc: 'Strong prompt understanding (20 steps)',
    category: 'generate',
    steps: 20,
    size: '~20-40 GB',
    quantizeOptions: [4, 0],
    repoMap: {
      4: 'carsenk/qwen-image-mflux-4bit',
      0: 'Qwen/Qwen-Image',
    },
    tier: 'large',
    mfluxClass: 'QwenImage',
    mfluxName: 'qwen-image',
    supportsImg2Img: true,
  },

  // ── Image Editing ────────────────────────────────────────────────────

  // Qwen Image Edit — instruction-based editing
  {
    id: 'qwen-image-edit',
    name: 'Qwen Image Edit',
    desc: 'Instruction-based editing (28 steps)',
    category: 'edit',
    steps: 28,
    size: '~54 GB',
    quantizeOptions: [0],
    repoMap: {
      0: 'Qwen/Qwen-Image-Edit',
    },
    tier: 'large',
    mfluxClass: 'QwenImageEdit',
    mfluxName: 'qwen-image-edit',
    supportsImg2Img: false,
  },

  // Flux Kontext — subject-consistent editing via img2img
  {
    id: 'kontext',
    name: 'Flux Kontext',
    desc: 'Subject-consistent editing (24 steps)',
    category: 'edit',
    steps: 24,
    size: '~6-24 GB',
    quantizeOptions: [4, 0],
    repoMap: {
      4: 'akx/FLUX.1-Kontext-dev-mflux-4bit',
      0: 'black-forest-labs/FLUX.1-Kontext-dev',
    },
    tier: 'medium',
    mfluxClass: 'Flux1Kontext',
    mfluxName: 'dev-kontext',
    supportsImg2Img: true,
  },

  // Flux Fill — inpainting with mask
  {
    id: 'fill',
    name: 'Flux Fill',
    desc: 'Inpainting with mask (20 steps)',
    category: 'edit',
    steps: 20,
    size: '~24 GB',
    quantizeOptions: [0],
    repoMap: {
      0: 'black-forest-labs/FLUX.1-Fill-dev',
    },
    tier: 'large',
    mfluxClass: 'Flux1Fill',
    mfluxName: 'dev-fill',
    supportsImg2Img: true,
  },
]

/** Look up a model definition by its canonical ID */
export function getImageModel(id: string): ImageModelDef | undefined {
  return IMAGE_MODELS.find(m => m.id === id)
}

/** Get default inference steps for a model ID */
export function getDefaultSteps(modelId: string): number {
  return getImageModel(modelId)?.steps ?? 4
}

/**
 * Resolve a named image model to its HF repo ID based on quantization level.
 * Exact match first, then closest available quantization.
 */
export function resolveImageModelRepo(modelName: string, quantize: number): string | null {
  const model = getImageModel(modelName)
  if (!model) return null
  const repos = model.repoMap
  // Exact match first
  if (repos[quantize]) return repos[quantize]
  // Fall back to closest quantization level
  const available = Object.keys(repos).map(Number).sort((a, b) => Math.abs(a - quantize) - Math.abs(b - quantize))
  return repos[available[0]] || null
}
