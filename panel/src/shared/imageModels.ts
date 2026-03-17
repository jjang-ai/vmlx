/**
 * Single source of truth for all image model definitions.
 *
 * Replaces duplicated model lists in:
 *   - ImageModelPicker.tsx  (NAMED_MODELS)
 *   - ImageTopBar.tsx       (AVAILABLE_MODELS)
 *   - ImageTab.tsx          (getDefaultSteps)
 *   - models.ts             (IMAGE_MODEL_REPOS)
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
  /** Whether the model uses a single or dual text encoder (for validation) */
  encoderType: 'single' | 'dual'
}

/**
 * Master list of all supported image models.
 *
 * To add a new model, add an entry here. All UI components, download logic,
 * and validation will pick it up automatically.
 */
export const IMAGE_MODELS: ImageModelDef[] = [
  // ── Image Generation ──────────────────────────────────────────────────
  // Small (16GB RAM friendly)
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
    encoderType: 'dual',
  },
  {
    id: 'flux2-klein-4b',
    name: 'FLUX.2 Klein 4B',
    desc: 'Compact next-gen model (20 steps)',
    category: 'generate',
    steps: 20,
    size: '~4-8 GB',
    quantizeOptions: [8, 0],
    repoMap: {
      8: 'AITRADER/FLUX2-klein-4B-mlx-8bit',
      0: 'black-forest-labs/FLUX.2-klein-4B',
    },
    tier: 'small',
    encoderType: 'single',
  },
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
    encoderType: 'single',
  },
  // Medium
  {
    id: 'flux2-klein-9b',
    name: 'FLUX.2 Klein 9B',
    desc: 'Next-gen mid-size model (20 steps)',
    category: 'generate',
    steps: 20,
    size: '~16 GB',
    quantizeOptions: [0],
    repoMap: {
      0: 'black-forest-labs/FLUX.2-klein-9B',
    },
    tier: 'medium',
    encoderType: 'single',
  },
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
    encoderType: 'dual',
  },
  // ── Image Editing (full precision only) ────────────────────────────────
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
    encoderType: 'single',
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
 * Build a flat repo-map keyed by model ID, equivalent to the old IMAGE_MODEL_REPOS.
 * Used by resolveImageModelRepo in main process.
 */
export function buildRepoMap(): Record<string, Record<number, string>> {
  const map: Record<string, Record<number, string>> = {}
  for (const m of IMAGE_MODELS) {
    map[m.id] = { ...m.repoMap }
  }
  return map
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
