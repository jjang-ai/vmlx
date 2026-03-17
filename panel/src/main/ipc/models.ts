import { ipcMain, dialog, BrowserWindow } from 'electron'
import { readdir, stat, access, readFile, mkdir, writeFile, unlink, rm, realpath } from 'fs/promises'
import { existsSync, readFileSync } from 'fs'
import { join, basename } from 'path'
import { homedir } from 'os'
import { spawn, ChildProcess } from 'child_process'
import { db } from '../database'
import { detectModelConfigFromDir } from '../model-config-registry'
import { getBundledPythonPath } from '../engine-manager'
import { IMAGE_MODELS, resolveImageModelRepo as _resolveImageModelRepo } from '../../shared/imageModels'

/** Generation defaults read from a model's generation_config.json */
export interface GenerationDefaults {
  temperature?: number
  topP?: number
  topK?: number
  minP?: number
  repeatPenalty?: number
}

/** Read generation_config.json from a model directory and extract sampling defaults */
export async function readGenerationDefaults(modelPath: string): Promise<GenerationDefaults | null> {
  try {
    const configPath = join(modelPath, 'generation_config.json')
    const raw = await readFile(configPath, 'utf-8')
    const config = JSON.parse(raw)
    const defaults: GenerationDefaults = {}

    if (typeof config.temperature === 'number') defaults.temperature = config.temperature
    if (typeof config.top_p === 'number') defaults.topP = config.top_p
    if (typeof config.top_k === 'number') defaults.topK = config.top_k
    if (typeof config.min_p === 'number') defaults.minP = config.min_p
    if (typeof config.repetition_penalty === 'number') defaults.repeatPenalty = config.repetition_penalty

    // Only return if at least one param was found
    if (Object.keys(defaults).length === 0) return null
    return defaults
  } catch {
    return null
  }
}

interface ModelInfo {
  id: string
  name: string
  path: string
  size?: string
  format?: 'mlx' | 'gguf' | 'unknown'
  quantization?: string
}

/** Check if a model directory contains MLX-format files or diffusers format.
 *  Strict: config.json must have model_type or architectures to be a real model,
 *  not just any directory with a stray config.json. */
async function detectModelFormat(modelPath: string): Promise<'mlx' | 'diffusers' | 'gguf' | 'unknown'> {
  try {
    const files = await readdir(modelPath)

    // Diffusers / mflux image models:
    // - model_index.json at root (standard diffusers)
    // - transformer/ + text_encoder/ subdirs (mflux quantized models)
    // - transformer/config.json without root config.json model_type (custom diffusion arch)
    if (files.includes('model_index.json')) return 'diffusers'
    if (files.includes('transformer') && files.includes('text_encoder')) return 'diffusers'
    if (files.includes('transformer') && files.includes('vae')) return 'diffusers'

    const hasGGUF = files.some(f => f.endsWith('.gguf') || f.endsWith('.gguf.part'))
    const hasSafetensors = files.some(f => f.endsWith('.safetensors'))
    const hasConfig = files.includes('config.json')

    // Strict MLX model check: config.json must contain model_type or architectures
    // This prevents parent directories with stray config.json from being detected as models
    if (hasSafetensors && hasConfig) {
      try {
        const cfg = JSON.parse(readFileSync(join(modelPath, 'config.json'), 'utf-8'))
        if (cfg.model_type || cfg.architectures || cfg.quantization) return 'mlx'
        // Also accept JANG models
        if (files.some(f => f === 'jang_config.json' || f === 'jang_cfg.json' || f === 'jjqf_config.json' || f === 'mxq_config.json')) return 'mlx'
        // config.json without model_type/architectures is not a model (could be scheduler config, etc.)
        return 'unknown'
      } catch {
        return 'unknown'
      }
    }
    // JANG-only model (has jang_config.json + safetensors but config.json may lack model_type)
    if (hasSafetensors && files.some(f => f === 'jang_config.json' || f === 'jang_cfg.json' || f === 'jjqf_config.json' || f === 'mxq_config.json')) return 'mlx'
    if (hasGGUF) return 'gguf'
    return 'unknown'
  } catch {
    return 'unknown'
  }
}

const BUILTIN_MODEL_PATHS = [
  join(homedir(), '.mlxstudio/models'),
  join(homedir(), '.cache/huggingface/hub'),
  join(homedir(), '.exo/models'),
]

const SETTINGS_KEY = 'model_scan_directories'
const IMAGE_SETTINGS_KEY = 'image_model_scan_directories'

const BUILTIN_IMAGE_PATHS = [
  join(homedir(), '.mlxstudio/models/image'),
  join(homedir(), '.mlxstudio/models/xcreates'),
]

// ─── Image Model HF Repo Mapping ────────────────────────────────────────────
// Now derived from the shared registry in shared/imageModels.ts.
// Use resolveImageModelRepo() to resolve a model ID + quantize to an HF repo.
const resolveImageModelRepo = _resolveImageModelRepo

/**
 * Validate that an image model directory has ALL required components.
 *
 * mflux models need more than just transformer/ weights — they also need
 * text encoders, tokenizers, and VAE. Without these, the model appears
 * "downloaded" but the server hangs trying to fetch missing components.
 *
 * Two formats exist:
 *   mflux-quantized (pre-quantized all-in-one): transformer/ + text_encoder/ both with .safetensors
 *   diffusers (full precision): model_index.json lists all components; verify each exists
 *
 * Returns { complete, missing[] } — missing lists which components are absent.
 */
export function validateImageModelCompleteness(modelDir: string, encoderType?: 'single' | 'dual'): { complete: boolean; missing: string[] } {
  const { readdirSync } = require('fs')
  const missing: string[] = []

  let files: string[]
  try {
    files = readdirSync(modelDir)
  } catch {
    return { complete: false, missing: ['directory unreadable'] }
  }

  // Check if transformer/ has weight files (required for ALL image models)
  let hasTransformerWeights = false
  if (files.includes('transformer')) {
    try {
      const tFiles: string[] = readdirSync(join(modelDir, 'transformer'))
      hasTransformerWeights = tFiles.some((f: string) => f.endsWith('.safetensors'))
    } catch { }
  }
  if (!hasTransformerWeights) {
    // Some mflux models have root-level .safetensors (rare, but allow it)
    const hasRootWeights = files.some((f: string) => f.endsWith('.safetensors'))
    if (!hasRootWeights) {
      missing.push('transformer/ weights')
    }
  }

  // If model_index.json exists, use it to verify all listed components
  if (files.includes('model_index.json')) {
    try {
      const indexRaw = readFileSync(join(modelDir, 'model_index.json'), 'utf-8')
      const index = JSON.parse(indexRaw)
      // model_index.json lists component dirs as keys with [class, subclass] values
      // Standard keys to check: text_encoder, text_encoder_2, transformer, vae, tokenizer, etc.
      // Only verify directories that are declared in model_index.json
      for (const [key, value] of Object.entries(index)) {
        if (key.startsWith('_')) continue  // skip _class_name, _diffusers_version, etc.
        if (!Array.isArray(value)) continue  // skip non-component entries
        const [className] = value as string[]
        if (!className || className === 'None' || className === null) continue
        // Check that the component directory exists
        if (!files.includes(key)) {
          missing.push(`${key}/`)
        }
      }
    } catch {
      // model_index.json parse failed — fall through to heuristic check
    }
  } else {
    // No model_index.json -> mflux-quantized format. Must have text_encoder/ with weights.
    if (files.includes('text_encoder')) {
      try {
        const teFiles: string[] = readdirSync(join(modelDir, 'text_encoder'))
        const hasTeWeights = teFiles.some((f: string) => f.endsWith('.safetensors'))
        if (!hasTeWeights) missing.push('text_encoder/ weights')
      } catch {
        missing.push('text_encoder/ (unreadable)')
      }
    } else {
      missing.push('text_encoder/')
    }

    // Determine encoder type: prefer the explicit parameter, then look up from the
    // shared registry, and finally fall back to directory-name heuristic.
    let isSingleEncoder: boolean
    if (encoderType) {
      isSingleEncoder = encoderType === 'single'
    } else {
      // Try to match the directory name against the shared image model registry
      const dirLower = modelDir.toLowerCase()
      const modelDef = IMAGE_MODELS.find(m => dirLower.includes(m.id))
      if (modelDef) {
        isSingleEncoder = modelDef.encoderType === 'single'
      } else {
        // Legacy fallback: string-based heuristic
        isSingleEncoder = dirLower.includes('z-image') || dirLower.includes('zimage') || dirLower.includes('klein')
      }
    }
    if (!isSingleEncoder && !files.includes('text_encoder_2')) {
      missing.push('text_encoder_2/')
    }
  }

  return { complete: missing.length === 0, missing }
}

/** Check if a named image model is available locally using the DB-backed path store.
 *  Verifies the stored path still exists on disk; removes stale entries. */
function checkImageModelLocal(modelName: string, quantize: number): { available: boolean; localPath?: string; repoId?: string; missing?: string[] } {
  const stored = db.getImageModelPath(modelName, quantize)
  if (stored) {
    try {
      if (existsSync(stored.localPath)) {
        // Validate completeness — check for essential subdirectories
        const missing: string[] = []
        const modelDir = stored.localPath
        if (!existsSync(join(modelDir, 'transformer'))) missing.push('transformer')
        if (!existsSync(join(modelDir, 'text_encoder'))) missing.push('text_encoder')
        // Check for .vmlx-downloading marker (download in progress)
        if (existsSync(join(modelDir, '.vmlx-downloading'))) missing.push('download incomplete')

        if (missing.length > 0) {
          return { available: false, localPath: stored.localPath, repoId: stored.repoId, missing }
        }
        return { available: true, localPath: stored.localPath, repoId: stored.repoId }
      } else {
        // Path was deleted — remove stale entry from DB
        db.deleteImageModelPath(modelName, quantize)
      }
    } catch { /* fs error — treat as unavailable */ }
  }
  const repoId = resolveImageModelRepo(modelName, quantize)
  return { available: false, repoId: repoId || undefined }
}

/** Get the list of directories to scan: user-configured + built-in defaults */
function getModelDirectories(modelType?: string): string[] {
  const key = modelType === 'image' ? IMAGE_SETTINGS_KEY : SETTINGS_KEY
  const builtins = modelType === 'image' ? BUILTIN_IMAGE_PATHS : BUILTIN_MODEL_PATHS
  const saved = db.getSetting(key)
  if (saved) {
    try {
      const userDirs: string[] = JSON.parse(saved)
      const all = [...userDirs]
      for (const d of builtins) {
        if (!all.includes(d)) all.push(d)
      }
      return all
    } catch {
      return builtins
    }
  }
  return builtins
}

/** Get only user-configured directories (not the built-in defaults) */
function getUserDirectories(modelType?: string): string[] {
  const key = modelType === 'image' ? IMAGE_SETTINGS_KEY : SETTINGS_KEY
  const saved = db.getSetting(key)
  if (saved) {
    try {
      return JSON.parse(saved)
    } catch {
      return []
    }
  }
  return []
}

function setUserDirectories(dirs: string[], modelType?: string): void {
  const key = modelType === 'image' ? IMAGE_SETTINGS_KEY : SETTINGS_KEY
  db.setSetting(key, JSON.stringify(dirs))
}

async function getDirectorySize(dirPath: string): Promise<number> {
  let totalSize = 0
  try {
    const files = await readdir(dirPath, { withFileTypes: true })
    for (const file of files) {
      const filePath = join(dirPath, file.name)
      if (file.isDirectory()) {
        totalSize += await getDirectorySize(filePath)
      } else {
        const stats = await stat(filePath)
        totalSize += stats.size
      }
    }
  } catch (error) {
    console.error('Error calculating directory size:', error)
  }
  return totalSize
}

function formatSize(bytes: number): string {
  const tb = bytes / (1024 * 1024 * 1024 * 1024)
  if (tb >= 1) return `~${tb.toFixed(1)} TB`
  const gb = bytes / (1024 * 1024 * 1024)
  if (gb >= 1) return `~${gb.toFixed(1)} GB`
  const mb = bytes / (1024 * 1024)
  return `~${mb.toFixed(0)} MB`
}

/** Parse size string like "~4.2 GB" back to bytes for sorting */
function parseSizeBytes(size: string | undefined): number {
  if (!size) return 0
  const match = size.match(/~?([\d.]+)\s*(TB|GB|MB)/i)
  if (!match) return 0
  const val = parseFloat(match[1])
  const unit = match[2].toUpperCase()
  if (unit === 'TB') return val * 1024 * 1024 * 1024 * 1024
  if (unit === 'GB') return val * 1024 * 1024 * 1024
  return val * 1024 * 1024
}

async function scanModelsInPath(basePath: string, modelType?: string, skipDirs?: Set<string>): Promise<ModelInfo[]> {
  const models: ModelInfo[] = []
  // Skip common system and git directories. We keep 'snapshots' unskipped so it can descend into standard HF Hub caches if needed.
  const SKIP_DIRS = ['.locks', 'blobs', 'refs', '.git', '.cache']

  async function scanRecursive(currentPath: string, depth: number, maxDepth: number) {
    if (depth > maxDepth) return

    try {
      // Check if current directory is a valid model (MLX text or diffusers image)
      const format = await detectModelFormat(currentPath)

      // Diffusers models: stop recursion here (don't descend into transformer/, vae/, etc.)
      if (format === 'diffusers') {
        // Skip image models when scanning for text models only
        if (modelType === 'text') return

        const size = await getDirectorySize(currentPath)
        let id = basename(currentPath)
        if (depth > 1) {
          const parent = basename(join(currentPath, '..'))
          if (parent !== basename(basePath)) id = `${parent}/${id}`
        }
        models.push({
          id,
          name: id,
          path: currentPath,
          size: formatSize(size),
          format: 'mlx',  // Treat as compatible for UI purposes
          quantization: 'Image Model',
        })
        return  // Don't recurse into subdirectories of diffusers models
      }

      // MLX text models (.safetensors + config.json)
      if (format === 'mlx') {
        // Skip text models when scanning for image models only
        if (modelType === 'image') return
        const size = await getDirectorySize(currentPath)

        // Skip empty models (less than 1MB)
        if (size >= 1024 * 1024) {
          // If the model is a subdirectory, use its relative-ish name (org/model-name)
          let id = basename(currentPath)
          if (depth > 1) {
            const parent = basename(join(currentPath, '..'))
            if (parent !== basename(basePath)) {
              id = `${parent}/${id}`
            }
          }

          // Detect quantization: try config.json first, fall back to directory name
          let quantization: string | undefined
          try {
            const configRaw = await readFile(join(currentPath, 'config.json'), 'utf-8')
            const configJson = JSON.parse(configRaw)
            if (typeof configJson.quantization === 'string' && configJson.quantization) {
              quantization = configJson.quantization
            }
          } catch { /* no config or parse error */ }
          // JANG format: read jang_config.json (or legacy mxq_config.json) for actual bit width
          if (!quantization) {
            for (const cfgName of ['jang_config.json', 'jang_cfg.json', 'jjqf_config.json', 'mxq_config.json']) {
              try {
                const jangRaw = await readFile(join(currentPath, cfgName), 'utf-8')
                const jangConfig = JSON.parse(jangRaw)
                if (jangConfig.format === 'jang' || jangConfig.format === 'jjqf' || jangConfig.format === 'mxq') {
                  const profile = jangConfig.quantization?.profile
                  const bits = jangConfig.quantization?.actual_bits || jangConfig.quantization?.target_bits
                  quantization = profile ? `${profile} (${bits}b)` : (bits ? `JANG ${bits}-bit` : 'JANG')
                  break
                }
              } catch { /* not this config */ }
            }
          }
          if (!quantization) {
            const nameMatch = id.toLowerCase().match(/\b(4bit|8bit|3bit|6bit|fp16|bf16|fp32)\b/)
            if (nameMatch) quantization = nameMatch[1]
          }

          models.push({
            id,
            name: id,
            path: currentPath,
            size: formatSize(size),
            format,
            quantization
          })
        }

        // Do NOT recurse into a valid model directory. 
        // This prevents treating a model's subdirectories (like code/) as distinct org parent folders.
        return
      }

      // If it's not a model, and we haven't reached max depth, recurse into its subdirectories
      if (depth < maxDepth) {
        const entries = await readdir(currentPath, { withFileTypes: true })
        for (const entry of entries) {
          if (!entry.isDirectory()) continue
          if (SKIP_DIRS.includes(entry.name)) continue

          // Skip directories that belong to the other model type's scan paths
          // (e.g., when scanning text models from ~/.mlxstudio/models, skip image/ and xcreates/ subdirs)
          const childPath = join(currentPath, entry.name)
          if (skipDirs?.has(childPath)) continue

          // Special case: ignore in-progress vmlx downloads
          try {
            await access(join(childPath, '.vmlx-downloading'))
            continue
          } catch (_) { /* not downloading */ }

          await scanRecursive(childPath, depth + 1, maxDepth)
        }
      }
    } catch (e) {
      // Access errors (restricted permissions, etc) can be ignored safely
    }
  }

  try {
    await access(basePath)
    // Start at depth 0, max depth 3 (allows scanning ~/.cache/huggingface/hub/models.../snapshots/xyz/)
    // For standard org/model, they are found at depth 2
    await scanRecursive(basePath, 0, 3)
  } catch (error) {
    // Directory doesn't exist or not accessible — skip silently
  }

  return models
}

/** Kill active download subprocess — call on app quit to prevent orphans */
let _killActiveDownload: (() => void) | null = null
export function killActiveDownload(): void {
  _killActiveDownload?.()
}

export function registerModelHandlers(): void {
  // Scan for available models in all configured directories
  ipcMain.handle('models:scan', async (_, modelType?: string) => {
    const dirs = getModelDirectories(modelType)
    console.log('[MODELS] Scanning directories:', dirs)
    const allModels: ModelInfo[] = []

    // When explicitly filtering by type, skip the other type's directories and filter results.
    // 'text' scan: skip image dirs (don't descend into ~/.mlxstudio/models/image or xcreates)
    // 'image' scan: only return diffusers-format models
    // undefined: return all models (no filtering)
    let skipDirs: Set<string> | undefined
    if (modelType === 'text') {
      const imageDirs = getModelDirectories('image')
      skipDirs = new Set(imageDirs.map(d => d.replace(/\/+$/, '')))
    }
    const scanFilter = modelType === 'image' ? 'image' : (modelType === 'text' ? 'text' : undefined)

    for (const basePath of dirs) {
      try {
        const models = await scanModelsInPath(basePath, scanFilter, skipDirs)
        allModels.push(...models)
        console.log(`[MODELS] Found ${models.length} models in ${basePath}`)
      } catch (error) {
        console.error(`[MODELS] Error scanning ${basePath}:`, error)
      }
    }

    // Deduplicate by resolved absolute path — prevents duplicates when scan paths overlap
    // (e.g., ~/.mlxstudio/models and ~/.mlxstudio/models/image both scan the same model)
    // Resolve symlinks for dedup, but keep original paths for display
    const resolvedPaths = await Promise.all(allModels.map(async m => {
      try { return await realpath(m.path) } catch { return m.path }
    }))
    const seen = new Set<string>()
    const deduped = allModels.filter((m, i) => {
      const resolved = resolvedPaths[i].replace(/\/+$/, '')
      if (seen.has(resolved)) return false
      seen.add(resolved)
      return true
    })
    console.log(`[MODELS] Total models found: ${deduped.length} (${allModels.length - deduped.length} duplicates removed)`)
    return deduped
  })

  // Get model info by path
  ipcMain.handle('models:info', async (_, modelPath: string) => {
    try {
      const size = await getDirectorySize(modelPath)
      const name = basename(modelPath)

      return {
        id: name,
        name,
        path: modelPath,
        size: formatSize(size)
      }
    } catch (error) {
      throw new Error(`Failed to get model info: ${(error as Error).message}`)
    }
  })

  // Get all scan directories (user + built-in) — optional modelType for separate image dirs
  ipcMain.handle('models:getDirectories', async (_, modelType?: string) => {
    const builtins = modelType === 'image' ? BUILTIN_IMAGE_PATHS : BUILTIN_MODEL_PATHS
    return {
      directories: getModelDirectories(modelType),
      userDirectories: getUserDirectories(modelType),
      builtinDirectories: builtins
    }
  })

  // Add a directory to the scan list — optional modelType for separate image dirs
  ipcMain.handle('models:addDirectory', async (_, dirPath: string, modelType?: string) => {
    const userDirs = getUserDirectories(modelType)
    const builtins = modelType === 'image' ? BUILTIN_IMAGE_PATHS : BUILTIN_MODEL_PATHS
    const normalized = dirPath.replace(/\/+$/, '')
    try {
      await access(normalized)
    } catch {
      return { success: false, error: 'Directory does not exist or is not accessible' }
    }
    // Resolve symlinks for robust duplicate detection
    let resolved: string
    try { resolved = await realpath(normalized) } catch { resolved = normalized }
    const allExisting = [...userDirs, ...builtins]
    for (const existing of allExisting) {
      try {
        const existingResolved = await realpath(existing)
        if (existingResolved === resolved) {
          return { success: false, error: 'Directory already in scan list' }
        }
      } catch {
        // If we can't resolve an existing entry, fall back to string comparison
        if (existing === normalized) {
          return { success: false, error: 'Directory already in scan list' }
        }
      }
    }
    userDirs.push(normalized)
    setUserDirectories(userDirs, modelType)
    return { success: true }
  })

  // Remove a user directory from the scan list — optional modelType
  ipcMain.handle('models:removeDirectory', async (_, dirPath: string, modelType?: string) => {
    const userDirs = getUserDirectories(modelType)
    const filtered = userDirs.filter(d => d !== dirPath)
    setUserDirectories(filtered, modelType)
    return { success: true }
  })

  // Detect model config (tool/reasoning parser, cache type) from model directory
  ipcMain.handle('models:detect-config', async (_, modelPath: string) => {
    return detectModelConfigFromDir(modelPath)
  })

  // Detect model types (image vs text) by checking file structure, not names
  ipcMain.handle('models:detectTypes', async (_, modelPaths: string[]) => {
    // Return a Record<path, type> so callers can look up by path (not by index).
    // Previously returned an array, but CreateSession.tsx uses types?.[modelPath]
    // which requires a keyed object.
    const result: Record<string, 'text' | 'image' | 'unknown'> = {}
    for (const p of modelPaths) {
      try {
        // Diffusers / mflux image models
        if (existsSync(join(p, 'model_index.json'))) { result[p] = 'image'; continue }
        if (existsSync(join(p, 'transformer')) && existsSync(join(p, 'text_encoder'))) { result[p] = 'image'; continue }
        if (existsSync(join(p, 'transformer')) && existsSync(join(p, 'vae'))) { result[p] = 'image'; continue }
        // Standard text models have config.json with model_type or architectures
        if (existsSync(join(p, 'config.json'))) {
          try {
            const cfg = JSON.parse(readFileSync(join(p, 'config.json'), 'utf-8'))
            if (cfg.pipeline_tag === 'text-to-image' || cfg.pipeline_tag === 'image-to-image') { result[p] = 'image'; continue }
            if (cfg.model_type || cfg.architectures) { result[p] = 'text'; continue }
          } catch {}
        }
        // JANG models are text models (check all config variants)
        if (existsSync(join(p, 'jang_config.json')) || existsSync(join(p, 'jang_cfg.json')) || existsSync(join(p, 'jjqf_config.json')) || existsSync(join(p, 'mxq_config.json'))) { result[p] = 'text'; continue }
        // Remote sessions
        if (p.startsWith('remote://')) { result[p] = 'text'; continue }
        result[p] = 'unknown'
      } catch {
        result[p] = 'unknown'
      }
    }
    return result
  })

  // Open a native directory picker dialog
  ipcMain.handle('models:browseDirectory', async () => {
    // Default to LM Studio models folder if it exists, otherwise home
    const lmStudioPath = join(homedir(), '.mlxstudio', 'models')
    let defaultPath: string | undefined
    try {
      await access(lmStudioPath)
      defaultPath = lmStudioPath
    } catch {
      defaultPath = homedir()
    }
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory'],
      securityScopedBookmarks: true,
      title: 'Select Model Directory',
      defaultPath
    })
    if (result.canceled || result.filePaths.length === 0) {
      return { canceled: true }
    }

    // In MacOS App Sandbox, save the security scoped bookmark for future reboots
    if (result.bookmarks && result.bookmarks.length > 0) {
      db.saveBookmark(result.filePaths[0], result.bookmarks[0])
    }

    return { canceled: false, path: result.filePaths[0] }
  })

  // Read generation defaults from model's generation_config.json
  ipcMain.handle('models:getGenerationDefaults', async (_, modelPath: string) => {
    return await readGenerationDefaults(modelPath)
  })

  // ─── HuggingFace Search & Download ─────────────────────────────────────────

  const DOWNLOAD_DIR_KEY = 'model_download_directory'

  function getDownloadDirectory(): string {
    return db.getSetting(DOWNLOAD_DIR_KEY) || join(homedir(), '.cache/huggingface/hub')
  }

  // ─── Download Manager ────────────────────────────────────────────────────────

  interface DownloadProgress {
    percent: number      // 0-100
    speed: string        // e.g. "45.2MB/s"
    downloaded: string   // e.g. "1.2GB"
    total: string        // e.g. "4.5GB"
    eta: string          // e.g. "2:30"
    currentFile: string  // e.g. "model-00001-of-00003.safetensors"
    filesProgress: string // e.g. "2/15"
    raw: string          // raw stderr line
  }

  interface DownloadJob {
    id: string
    repoId: string
    status: 'queued' | 'downloading' | 'paused' | 'complete' | 'cancelled' | 'error'
    progress?: DownloadProgress
    error?: string
    process?: ChildProcess
    modelDir: string
    wasCancelled?: boolean
    wasPaused?: boolean
    /** For image model downloads: the canonical model ID (e.g. 'schnell') */
    imageModelName?: string
    /** For image model downloads: the quantization level (e.g. 4, 8, 0) */
    imageQuantize?: number
  }

  let jobIdCounter = 0
  const downloadQueue: DownloadJob[] = []
  const activeJobs: Map<string, DownloadJob> = new Map()  // Multiple concurrent downloads
  const MAX_CONCURRENT = 3
  const completedJobs: DownloadJob[] = []
  const MAX_COMPLETED_JOBS = 100
  const trackCompleted = (job: DownloadJob) => {
    completedJobs.push(job)
    if (completedJobs.length > MAX_COMPLETED_JOBS) completedJobs.splice(0, completedJobs.length - MAX_COMPLETED_JOBS)
  }

  // Expose kill function for app quit cleanup
  _killActiveDownload = () => {
    for (const [, job] of activeJobs) {
      if (job.process) {
        console.log(`[DOWNLOADS] Killing active download on quit: ${job.repoId}`)
        try { job.process.kill('SIGKILL') } catch (_) { }
      }
    }
  }

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
  }

  // parseTqdmProgress removed — download manager now uses structured JSON stdout

  function emitToRenderer(channel: string, data: any) {
    // Send to ALL windows (main + download popup) so both stay in sync
    try {
      for (const win of BrowserWindow.getAllWindows()) {
        if (!win.isDestroyed()) {
          win.webContents.send(channel, data)
        }
      }
    } catch (_) { }
  }

  async function getPythonPath(): Promise<string> {
    const bundledPython = getBundledPythonPath()
    if (bundledPython) {
      try {
        await access(bundledPython)
        return bundledPython
      } catch { /* fall through */ }
    }
    return 'python3'
  }

  async function processQueue() {
    // Start downloads up to MAX_CONCURRENT — skip paused jobs (don't block queue)
    while (activeJobs.size < MAX_CONCURRENT && downloadQueue.length > 0) {
      // Find the first non-paused job
      const idx = downloadQueue.findIndex(j => j.status !== 'paused')
      if (idx === -1) break  // All remaining jobs are paused
      const job = downloadQueue.splice(idx, 1)[0]
      // Register in activeJobs BEFORE async work to prevent over-dispatch
      activeJobs.set(job.id, job)
      job.status = 'downloading'
      startDownloadJob(job)
    }
  }

  async function startDownloadJob(job: DownloadJob) {
    // job.status and activeJobs.set already done by processQueue (synchronous, before await)
    emitToRenderer('models:downloadStarted', { jobId: job.id, repoId: job.repoId })

    const pythonPath = await getPythonPath()
    // Pass HF token to snapshot_download for gated model access (reads HF_TOKEN env var)
    // Download script that reports total progress across all files as JSON lines on stdout.
    // huggingface_hub's snapshot_download uses tqdm per-file, which only shows shard progress.
    // Instead, we use hf_hub_download per file and track cumulative bytes ourselves.
    const script = [
      'import sys, json, os, time',
      'from huggingface_hub import HfApi, hf_hub_download',
      'from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError',
      'repo_id = sys.argv[1]',
      'local_dir = sys.argv[2]',
      'token = os.environ.get("HF_TOKEN") or None',
      'try:',
      '    api = HfApi()',
      '    tree = list(api.list_repo_tree(repo_id, token=token, recursive=True))',
      '    files = [f for f in tree if hasattr(f, "rfilename") and not f.rfilename.startswith(".")]',
      '    total_bytes = sum(getattr(f, "size", 0) or 0 for f in files)',
      '    downloaded_bytes = 0',
      '    total_files = len(files)',
      '    print(json.dumps({"type":"init","total_bytes":total_bytes,"total_files":total_files}), flush=True)',
      '    for i, f in enumerate(files):',
      '        est_size = getattr(f, "size", 0) or 0',
      '        print(json.dumps({"type":"file_start","file":f.rfilename,"file_num":i+1,"total_files":total_files,"downloaded_bytes":downloaded_bytes,"total_bytes":total_bytes}), flush=True)',
      '        t0 = time.time()',
      '        dl_path = hf_hub_download(repo_id, f.rfilename, local_dir=local_dir, token=token, local_dir_use_symlinks=False)',
      '        actual_size = os.path.getsize(dl_path) if os.path.exists(dl_path) else est_size',
      '        downloaded_bytes += actual_size',
      '        elapsed = time.time() - t0',
      '        speed = actual_size / elapsed if elapsed > 0 else 0',
      '        pct = int(downloaded_bytes * 100 / total_bytes) if total_bytes > 0 else int((i+1) * 100 / total_files)',
      '        print(json.dumps({"type":"file_done","file":f.rfilename,"file_num":i+1,"total_files":total_files,"downloaded_bytes":downloaded_bytes,"total_bytes":total_bytes,"speed":speed,"percent":pct}), flush=True)',
      '    print(json.dumps({"status":"complete","path":local_dir}), flush=True)',
      'except KeyboardInterrupt:',
      '    print(json.dumps({"status":"cancelled"}), flush=True)',
      '    sys.exit(0)',
      'except (GatedRepoError, RepositoryNotFoundError) as e:',
      '    print(json.dumps({"status":"error","error":str(e),"gated":True}), flush=True)',
      '    sys.exit(1)',
      'except Exception as e:',
      '    print(json.dumps({"status":"error","error":str(e)}), flush=True)',
      '    sys.exit(1)',
    ].join('\n')

    // Write marker file
    const markerFile = join(job.modelDir, '.vmlx-downloading')
    try {
      await mkdir(job.modelDir, { recursive: true })
      await writeFile(markerFile, `${job.repoId}\n${Date.now()}`, 'utf-8')
    } catch (_) { /* dir may already exist */ }

    console.log(`[DOWNLOADS] Starting: ${job.repoId} → ${job.modelDir}`)

    // Build spawn environment — inject HF_TOKEN if user has configured one
    const downloadEnv: Record<string, string | undefined> = { ...process.env }
    const hfToken = db.getSetting('hf_api_key')
    if (hfToken) {
      downloadEnv.HF_TOKEN = hfToken
    }

    const proc = spawn(pythonPath, ['-u', '-c', script, job.repoId, job.modelDir], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: downloadEnv
    })
    job.process = proc

    let stdout = ''
    let lastProgress: Partial<DownloadProgress> = {}
    let stdoutBuffer = ''

    // Parse JSON progress lines from stdout (our custom script outputs structured data)
    proc.stdout?.on('data', (data: Buffer) => {
      stdoutBuffer += data.toString()
      const lines = stdoutBuffer.split('\n')
      stdoutBuffer = lines.pop() || '' // Keep incomplete line in buffer
      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed) continue
        stdout += trimmed + '\n'
        try {
          const msg = JSON.parse(trimmed)
          if (msg.type === 'init') {
            lastProgress = {
              percent: 0,
              downloaded: '0 B',
              total: formatBytes(msg.total_bytes),
              filesProgress: `0/${msg.total_files}`,
              speed: '',
              eta: 'calculating...',
              currentFile: '',
              raw: trimmed,
            }
          } else if (msg.type === 'file_start') {
            // Use file count as progress fallback when byte totals are 0
            const pct = msg.total_bytes > 0
              ? Math.round((msg.downloaded_bytes / msg.total_bytes) * 100)
              : Math.round(((msg.file_num - 1) / msg.total_files) * 100)
            lastProgress = {
              ...lastProgress,
              percent: pct,
              downloaded: msg.downloaded_bytes > 0 ? formatBytes(msg.downloaded_bytes) : `${msg.file_num - 1} files`,
              total: msg.total_bytes > 0 ? formatBytes(msg.total_bytes) : `${msg.total_files} files`,
              currentFile: msg.file,
              filesProgress: `${msg.file_num}/${msg.total_files}`,
              eta: 'downloading...',
              raw: trimmed,
            }
          } else if (msg.type === 'file_done') {
            // Use percent from Python script (handles both byte and file-count modes)
            const pct = msg.percent ?? (msg.total_bytes > 0
              ? Math.round((msg.downloaded_bytes / msg.total_bytes) * 100)
              : Math.round((msg.file_num / msg.total_files) * 100))
            lastProgress = {
              ...lastProgress,
              percent: pct,
              downloaded: msg.downloaded_bytes > 0 ? formatBytes(msg.downloaded_bytes) : `${msg.file_num} files`,
              total: msg.total_bytes > 0 ? formatBytes(msg.total_bytes) : `${msg.total_files} files`,
              currentFile: msg.file,
              filesProgress: `${msg.file_num}/${msg.total_files}`,
              speed: msg.speed ? formatBytes(msg.speed) + '/s' : '',
              raw: trimmed,
            }
            // Estimate ETA from speed
            const remaining = msg.total_bytes - msg.downloaded_bytes
            if (msg.speed > 0 && remaining > 0) {
              const secs = Math.round(remaining / msg.speed)
              lastProgress.eta = secs > 60 ? `${Math.round(secs / 60)}m ${secs % 60}s` : `${secs}s`
            } else if (msg.file_num < msg.total_files) {
              lastProgress.eta = `${msg.total_files - msg.file_num} files left`
            } else {
              lastProgress.eta = ''
            }
          }
          job.progress = lastProgress as DownloadProgress
          emitToRenderer('models:downloadProgress', {
            jobId: job.id,
            repoId: job.repoId,
            progress: lastProgress,
          })
        } catch {
          // Not JSON — ignore (could be stray output)
        }
      }
    })

    // Capture stderr for error reporting but don't parse tqdm
    proc.stderr?.on('data', (data: Buffer) => {
      // Log stderr but don't parse — our progress comes from structured stdout
      const chunk = data.toString().trim()
      if (chunk) console.log(`[DOWNLOADS] stderr: ${chunk.slice(0, 200)}`)
    })

    proc.on('close', async (code: number | null) => {
      console.log(`[DOWNLOADS] Process exited: ${job.repoId} code=${code} cancelled=${job.wasCancelled}`)

      if (job.wasPaused) {
        // Paused — keep files for resume, don't emit completion
        console.log(`[DOWNLOADS] Paused (files preserved): ${job.repoId}`)
        return  // Don't clean up — job is already back in queue
      } else if (job.wasCancelled) {
        try { await unlink(markerFile) } catch (_) { }
        // Remove partial model directory to prevent incomplete models from appearing in scan
        try { await rm(job.modelDir, { recursive: true, force: true }) } catch (_) { }
        job.status = 'cancelled'
        emitToRenderer('models:downloadComplete', { jobId: job.id, repoId: job.repoId, status: 'cancelled' })
      } else if (code === 0) {
        try { await unlink(markerFile) } catch (_) { }
        job.status = 'complete'
        // Store downloaded image model path in DB for fast lookup
        if (job.imageModelName != null && job.imageQuantize != null) {
          db.setImageModelPath(job.imageModelName, job.imageQuantize, job.modelDir, job.repoId)
          console.log(`[DOWNLOADS] Stored image model path: ${job.imageModelName} q=${job.imageQuantize} → ${job.modelDir}`)
        }
        emitToRenderer('models:downloadComplete', { jobId: job.id, repoId: job.repoId, status: 'complete', path: job.modelDir })
      } else {
        let errorMsg = `Download failed (exit ${code})`
        try {
          const lines = stdout.trim().split('\n').filter(Boolean)
          if (lines.length > 0) {
            const result = JSON.parse(lines[lines.length - 1])
            if (result.status === 'cancelled') {
              try { await unlink(markerFile) } catch (_) { }
              job.status = 'cancelled'
              emitToRenderer('models:downloadComplete', { jobId: job.id, repoId: job.repoId, status: 'cancelled' })
              activeJobs.delete(job.id)
              job.process = undefined
              trackCompleted(job)
              processQueue()
              return
            }
            errorMsg = result.error || errorMsg
          }
        } catch { }
        // stderr is logged to console but not parsed for progress anymore
        try { await unlink(markerFile) } catch (_) { }
        job.status = 'error'
        job.error = errorMsg
        // Detect gated/auth errors (401/403) to prompt user to add HF token
        const isGatedError = /40[13]|gated|access.*denied|authentication|authorized/i.test(errorMsg)
        emitToRenderer('models:downloadError', { jobId: job.id, repoId: job.repoId, error: errorMsg, gated: isGatedError })
      }

      activeJobs.delete(job.id)
      job.process = undefined
      trackCompleted(job)
      // Process next in queue
      processQueue()
    })

    proc.on('error', async (err: Error) => {
      job.status = 'error'
      job.error = `Failed to start download: ${err.message}`
      activeJobs.delete(job.id)
      job.process = undefined
      trackCompleted(job)
      try { await unlink(markerFile) } catch (_) { }
      emitToRenderer('models:downloadError', { jobId: job.id, repoId: job.repoId, error: job.error })
      processQueue()
    })
  }

  // ─── Stale marker cleanup on startup ──────────────────────────────────────────
  async function cleanStaleMarkers() {
    const dirs = getModelDirectories()
    for (const baseDir of dirs) {
      try {
        const orgs = await readdir(baseDir, { withFileTypes: true })
        for (const org of orgs) {
          if (!org.isDirectory()) continue
          const orgPath = join(baseDir, org.name)
          try {
            const models = await readdir(orgPath, { withFileTypes: true })
            for (const model of models) {
              if (!model.isDirectory()) continue
              const markerPath = join(orgPath, model.name, '.vmlx-downloading')
              try {
                await access(markerPath)
                const content = await readFile(markerPath, 'utf-8')
                const ts = parseInt(content.split('\n')[1], 10)
                // Remove markers older than 1 hour (stale from crashed downloads)
                if (!isNaN(ts) && Date.now() - ts > 3600000) {
                  console.log(`[DOWNLOADS] Removing stale marker: ${markerPath}`)
                  await unlink(markerPath)
                }
              } catch { /* no marker */ }
            }
          } catch { /* can't read org dir */ }
        }
      } catch { /* base dir doesn't exist */ }
    }
  }
  cleanStaleMarkers().catch(() => { })

  // ─── One-time migration: scan existing image models into DB ──────────────────
  // On first launch after this update, scan ~/.mlxstudio/models/image/ for
  // already-downloaded models that aren't yet tracked in the DB. Match directory
  // names against the shared registry's repoMap to identify model IDs.
  async function migrateExistingImageModels() {
    const imageDir = join(homedir(), '.mlxstudio', 'models', 'image')
    try {
      if (!existsSync(imageDir)) return
      const dirs = await readdir(imageDir)
      let migrated = 0
      for (const dir of dirs) {
        const fullPath = join(imageDir, dir)
        const dirStat = await stat(fullPath).catch(() => null)
        if (!dirStat?.isDirectory()) continue
        // Check if this directory matches any known model repo
        const dirLower = dir.toLowerCase()
        for (const model of IMAGE_MODELS) {
          for (const [qStr, repoId] of Object.entries(model.repoMap)) {
            const quantize = Number(qStr)
            // Already in DB? Skip.
            const existing = db.getImageModelPath(model.id, quantize)
            if (existing) continue
            // Match: the directory name matches the repo name
            const repoName = repoId.split('/').pop()?.toLowerCase() || ''
            if (dirLower === repoName) {
              // Validate completeness before registering
              const validation = validateImageModelCompleteness(fullPath, model.encoderType)
              if (validation.complete) {
                db.setImageModelPath(model.id, quantize, fullPath, repoId)
                migrated++
                console.log(`[IMAGE] Migrated existing model: ${model.id} q=${quantize} → ${fullPath}`)
              }
            }
          }
        }
      }
      if (migrated > 0) {
        console.log(`[IMAGE] Migration complete: registered ${migrated} existing image model(s) in DB`)
      }
    } catch (err) {
      console.error('[IMAGE] Migration scan failed:', err)
    }
  }
  migrateExistingImageModels().catch(() => {})

  // Search HuggingFace for MLX models

  /** Map raw HF API model object to our HFModel shape.
   *  The list endpoint omits lastModified, author, and safetensors —
   *  use createdAt as date fallback, extract author from modelId. */
  function mapHFModel(m: any) {
    const modelId = m.modelId || m.id || ''
    return {
      id: modelId,
      author: m.author || modelId.split('/')[0] || 'Unknown',
      downloads: m.downloads ?? 0,
      likes: m.likes ?? 0,
      lastModified: m.lastModified || m.createdAt,
      tags: m.tags || [],
      pipelineTag: m.pipeline_tag,
      size: extractModelSize(m)
    }
  }

  /** Extract total model size from HF API safetensors metadata */
  function extractModelSize(m: any): string | undefined {
    try {
      // safetensors.total: total bytes across all safetensors files
      const total = m.safetensors?.total
      if (typeof total === 'number' && total > 0) return formatSize(total)
      // Fallback: sum parameter counts from safetensors.parameters
      const params = m.safetensors?.parameters
      if (params && typeof params === 'object') {
        const totalParams = Object.values(params).reduce((sum: number, v: any) => sum + (typeof v === 'number' ? v : 0), 0)
        if (totalParams > 0) {
          // Rough estimate: 2 bytes per param for fp16/bf16 (most MLX models)
          return formatSize(totalParams * 2)
        }
      }
    } catch (_) { }
    return undefined
  }

  ipcMain.handle('models:searchHF', async (_, query: string, sortBy?: string, sortDir?: string, modelType?: string) => {
    const params = new URLSearchParams({
      search: query,
      limit: '30'
    })
    // Filter by model type: 'image' searches both text-to-image AND image-to-image (for edit models),
    // default searches MLX text models
    const wantAsc = sortDir === 'asc'
    // HF API sort options: downloads, likes, lastModified, trending
    // "relevance" = omit sort (HF default), "size" = client-side sort
    if (sortBy && sortBy !== 'relevance' && sortBy !== 'size') {
      params.set('sort', sortBy)
      params.set('direction', '-1')
    } else if (!sortBy || sortBy === 'downloads') {
      params.set('sort', 'downloads')
      params.set('direction', '-1')
    }

    let models: any[]
    if (modelType === 'image') {
      // Search both text-to-image and image-to-image pipelines in parallel
      const p1 = new URLSearchParams(params)
      p1.set('filter', 'text-to-image')
      const p2 = new URLSearchParams(params)
      p2.set('filter', 'image-to-image')
      console.log(`[MODELS] Searching HuggingFace image models: ${query} (sort=${sortBy || 'downloads'} dir=${sortDir || 'desc'})`)
      const [r1, r2] = await Promise.all([
        fetch(`https://huggingface.co/api/models?${p1}`, { signal: AbortSignal.timeout(15000) }),
        fetch(`https://huggingface.co/api/models?${p2}`, { signal: AbortSignal.timeout(15000) })
      ])
      if (!r1.ok) throw new Error(`HuggingFace API error: ${r1.status}`)
      const m1 = await r1.json()
      const m2 = r2.ok ? await r2.json() : []
      // Deduplicate by model ID
      const seen = new Set<string>()
      models = []
      for (const m of [...m1, ...m2]) {
        if (!seen.has(m.modelId || m.id)) {
          seen.add(m.modelId || m.id)
          models.push(m)
        }
      }
    } else {
      params.set('filter', 'mlx')
      const url = `https://huggingface.co/api/models?${params}`
      console.log(`[MODELS] Searching HuggingFace: ${query} (sort=${sortBy || 'downloads'} dir=${sortDir || 'desc'})`)
      const response = await fetch(url, { signal: AbortSignal.timeout(15000) })
      if (!response.ok) throw new Error(`HuggingFace API error: ${response.status}`)
      models = await response.json()
    }

    let results = models.map((m: any) => mapHFModel(m))

    // HF API only returns descending — reverse for ascending
    if (wantAsc && sortBy !== 'size') results.reverse()

    // Client-side sort by model size (HF API doesn't support this)
    if (sortBy === 'size') {
      results.sort((a: any, b: any) => {
        const sizeA = parseSizeBytes(a.size)
        const sizeB = parseSizeBytes(b.size)
        return sortDir === 'asc' ? sizeA - sizeB : sizeB - sizeA
      })
    }

    return results
  })

  // Fetch model README from HuggingFace
  ipcMain.handle('models:fetchReadme', async (_, repoId: string) => {
    try {
      const res = await fetch(`https://huggingface.co/${repoId}/raw/main/README.md`, {
        signal: AbortSignal.timeout(10000),
      })
      if (!res.ok) return null
      const text = await res.text()
      // Strip YAML frontmatter
      const stripped = text.replace(/^---[\s\S]*?---\s*/, '')
      // Truncate to ~3000 chars for display
      return stripped.length > 3000 ? stripped.slice(0, 3000) + '\n\n...(truncated)' : stripped
    } catch {
      return null
    }
  })

  // Get recommended models from JANGQ-AI
  ipcMain.handle('models:getRecommendedModels', async () => {
    console.log('[MODELS] Fetching JANGQ-AI recommended models')
    const urls = [
      `https://huggingface.co/api/models?author=JANGQ-AI&sort=downloads&direction=-1`,
    ]
    const allModels: any[] = []
    for (const url of urls) {
      try {
        const response = await fetch(url, { signal: AbortSignal.timeout(15000) })
        if (response.ok) {
          const models = await response.json()
          allModels.push(...models)
        }
      } catch (err) {
        console.warn(`[MODELS] Failed to fetch from ${url}:`, err)
      }
    }
    // Deduplicate by ID, sort by downloads
    const seen = new Set<string>()
    const unique = allModels.filter(m => {
      if (seen.has(m.id || m.modelId)) return false
      seen.add(m.id || m.modelId)
      return true
    })
    unique.sort((a, b) => (b.downloads || 0) - (a.downloads || 0))
    return unique.map((m: any) => mapHFModel(m))
  })

  // Start a download (non-blocking — returns jobId immediately)
  ipcMain.handle('models:startDownload', async (_, repoId: string) => {
    // Check if already active or queued
    for (const [, aj] of activeJobs) {
      if (aj.repoId === repoId) return { jobId: aj.id, status: 'already_downloading' }
    }
    const queued = downloadQueue.find(j => j.repoId === repoId)
    if (queued) return { jobId: queued.id, status: 'already_queued' }

    // Validate repoId to prevent path traversal (must be "org/model" format)
    if (!repoId || repoId.includes('..') || repoId.startsWith('/') || !/^[^/]+\/[^/]+$/.test(repoId)) {
      throw new Error(`Invalid repository ID: ${repoId}`)
    }

    const id = `dl_${++jobIdCounter}_${Date.now()}`
    const targetDir = getDownloadDirectory()
    const modelDir = join(targetDir, repoId)

    const job: DownloadJob = { id, repoId, status: 'queued', modelDir }
    downloadQueue.push(job)
    console.log(`[DOWNLOADS] Queued: ${repoId} (${downloadQueue.length} in queue, ${activeJobs.size} active)`)

    // Notify UI about new queue entry immediately
    emitToRenderer('models:downloadQueued', { jobId: id, repoId })

    // Start if slots available
    processQueue()

    return { jobId: id, status: activeJobs.has(id) ? 'downloading' : 'queued', queuePosition: downloadQueue.length }
  })

  // Cancel a download by jobId (or cancel active if no jobId)
  ipcMain.handle('models:cancelDownload', async (_, jobId?: string) => {
    // Cancel from queue first (includes paused jobs)
    const qIdx = downloadQueue.findIndex(j => jobId ? j.id === jobId : true)
    if (qIdx >= 0) {
      const removed = downloadQueue.splice(qIdx, 1)[0]
      removed.status = 'cancelled'
      trackCompleted(removed)
      console.log(`[DOWNLOADS] Cancelled queued/paused: ${removed.repoId}`)
      // Clean up partial files for paused jobs (active jobs clean up in close handler)
      if (removed.wasPaused && removed.modelDir) {
        try { await rm(removed.modelDir, { recursive: true, force: true }) } catch (_) { }
      }
      emitToRenderer('models:downloadComplete', { jobId: removed.id, repoId: removed.repoId, status: 'cancelled' })
      return { success: true }
    }

    // Cancel active job by ID
    if (jobId && activeJobs.has(jobId)) {
      const aj = activeJobs.get(jobId)!
      console.log(`[DOWNLOADS] Cancelling active: ${aj.repoId}`)
      aj.wasCancelled = true
      aj.process?.kill('SIGTERM')
      return { success: true }
    }
    // Cancel first active if no jobId specified
    if (!jobId && activeJobs.size > 0) {
      const aj = activeJobs.values().next().value!
      console.log(`[DOWNLOADS] Cancelling active: ${aj.repoId}`)
      aj.wasCancelled = true
      aj.process?.kill('SIGTERM')
      return { success: true }
    }

    return { success: false, error: 'No matching download found' }
  })

  // Pause a download — kills the process but keeps job for resume
  ipcMain.handle('models:pauseDownload', async (_, jobId: string) => {
    if (activeJobs.has(jobId)) {
      const job = activeJobs.get(jobId)!
      console.log(`[DOWNLOADS] Pausing: ${job.repoId}`)
      job.wasPaused = true
      job.wasCancelled = true  // Signals the close handler to not delete files
      job.process?.kill('SIGTERM')
      // Move back to queue front so it can be resumed
      job.status = 'paused'
      activeJobs.delete(jobId)
      downloadQueue.unshift(job)
      emitToRenderer('models:downloadPaused', { jobId: job.id, repoId: job.repoId })
      processQueue()  // Start next queued job if any
      return { success: true }
    }
    return { success: false, error: 'Download not active' }
  })

  // Resume a paused download
  ipcMain.handle('models:resumeDownload', async (_, jobId: string) => {
    const idx = downloadQueue.findIndex(j => j.id === jobId && j.status === 'paused')
    if (idx >= 0) {
      const job = downloadQueue.splice(idx, 1)[0]
      job.wasPaused = false
      job.wasCancelled = false
      job.status = 'queued'
      job.progress = undefined
      console.log(`[DOWNLOADS] Resuming: ${job.repoId}`)
      // hf_hub_download skips already-downloaded files (per-file resume, not byte-level)
      if (activeJobs.size < MAX_CONCURRENT) {
        job.status = 'downloading'
        activeJobs.set(job.id, job)
        startDownloadJob(job)
      } else {
        downloadQueue.unshift(job)
        emitToRenderer('models:downloadQueued', { jobId: job.id, repoId: job.repoId })
      }
      return { success: true }
    }
    return { success: false, error: 'Download not found or not paused' }
  })

  // Get current download status (all jobs)
  ipcMain.handle('models:getDownloadStatus', async () => {
    const active = Array.from(activeJobs.values()).map(j => ({
      jobId: j.id, repoId: j.repoId, progress: j.progress
    }))
    return {
      active: active[0] || null,  // Backward compat: first active as primary
      activeAll: active,           // All active downloads
      queue: downloadQueue.map(j => ({ jobId: j.id, repoId: j.repoId, status: j.status })),
      completed: completedJobs.slice(-10).map(j => ({ jobId: j.id, repoId: j.repoId, status: j.status, error: j.error }))
    }
  })

  // Legacy blocking download (backward compat — wraps the new queue system)
  ipcMain.handle('models:downloadModel', async (_, repoId: string) => {
    // Validate repoId to prevent path traversal (must be "org/model" format)
    if (!repoId || repoId.includes('..') || repoId.startsWith('/') || !/^[^/]+\/[^/]+$/.test(repoId)) {
      throw new Error(`Invalid repository ID: ${repoId}`)
    }

    // Check if already queued or downloading
    const alreadyActive = Array.from(activeJobs.values()).some(j => j.repoId === repoId)
    if (alreadyActive || downloadQueue.find(j => j.repoId === repoId)) {
      throw new Error(`Already downloading ${repoId}`)
    }

    const id = `dl_${++jobIdCounter}_${Date.now()}`
    const targetDir = getDownloadDirectory()
    const modelDir = join(targetDir, repoId)
    const job: DownloadJob = { id, repoId, status: 'queued', modelDir }
    downloadQueue.push(job)
    processQueue()

    // Poll for completion
    return new Promise<any>((resolve, reject) => {
      const check = () => {
        const done = completedJobs.find(j => j.id === id)
        if (done) {
          if (done.status === 'complete') resolve({ status: 'complete', path: done.modelDir })
          else if (done.status === 'cancelled') resolve({ status: 'cancelled' })
          else reject(new Error(done.error || 'Download failed'))
        } else {
          setTimeout(check, 500)
        }
      }
      check()
    })
  })

  // Get download directory
  ipcMain.handle('models:getDownloadDir', async () => {
    return getDownloadDirectory()
  })

  // Set download directory (also adds to scan list)
  ipcMain.handle('models:setDownloadDir', async (_, dir: string) => {
    db.setSetting(DOWNLOAD_DIR_KEY, dir)
    // Also add to scan directories so downloaded models are found
    const userDirs = getUserDirectories()
    const normalized = dir.replace(/\/+$/, '')
    if (!userDirs.includes(normalized) && !BUILTIN_MODEL_PATHS.includes(normalized)) {
      userDirs.push(normalized)
      setUserDirectories(userDirs)
    }
    return { success: true }
  })

  // Browse for download directory
  ipcMain.handle('models:browseDownloadDir', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory', 'createDirectory'],
      securityScopedBookmarks: true,
      title: 'Select Model Download Directory',
      defaultPath: getDownloadDirectory()
    })
    if (result.canceled || result.filePaths.length === 0) {
      return { canceled: true }
    }

    // In MacOS App Sandbox, save the security scoped bookmark for future reboots
    if (result.bookmarks && result.bookmarks.length > 0) {
      db.saveBookmark(result.filePaths[0], result.bookmarks[0])
    }

    return { canceled: false, path: result.filePaths[0] }
  })

  // ─── Image Model Download Support ──────────────────────────────────────────
  // Check if a named image model is available locally before starting server
  ipcMain.handle('models:checkImageModel', async (_, modelName: string, quantize: number = 4) => {
    const result = checkImageModelLocal(modelName, quantize)
    return result
  })

  // Download a named image model (resolves to HF repo, uses existing download queue)
  ipcMain.handle('models:downloadImageModel', async (_, modelName: string, quantize: number = 4) => {
    const repoId = resolveImageModelRepo(modelName, quantize)
    if (!repoId) {
      throw new Error(`No known HuggingFace repository for image model: ${modelName} at ${quantize}-bit`)
    }

    // Check if already downloaded
    const local = checkImageModelLocal(modelName, quantize)
    if (local.available) {
      return { status: 'already_downloaded', localPath: local.localPath, repoId }
    }

    // Check HF token for gated models (Flux repos require authentication)
    const hfToken = db.getSetting('hf_api_key')
    if (!hfToken) {
      // Warn but don't block — some repos may be public
      console.log(`[DOWNLOADS] No HF token set. Image model ${repoId} may require authentication.`)
    }

    // Check if already active or queued
    for (const [, aj] of activeJobs) {
      if (aj.repoId === repoId) return { jobId: aj.id, status: 'already_downloading', repoId }
    }
    const queued = downloadQueue.find(j => j.repoId === repoId)
    if (queued) return { jobId: queued.id, status: 'already_queued', repoId }

    // Download image models to ~/.mlxstudio/models/image/ so they are found by
    // both checkImageModelLocal and image:startServer's local path check.
    // Use the repo name as the directory name (e.g., "flux.1-schnell-all-in-one-T5xxl-4bit")
    const id = `dl_${++jobIdCounter}_${Date.now()}`
    const imageModelsDir = join(homedir(), '.mlxstudio', 'models', 'image')
    const repoName = repoId.split('/').pop() || repoId
    const modelDir = join(imageModelsDir, repoName)

    const job: DownloadJob = { id, repoId, status: 'queued', modelDir, imageModelName: modelName, imageQuantize: quantize }
    downloadQueue.push(job)
    console.log(`[DOWNLOADS] Queued image model: ${repoId} → ${modelDir} (${downloadQueue.length} in queue, ${activeJobs.size} active)`)

    // Notify UI about new queue entry immediately
    emitToRenderer('models:downloadQueued', { jobId: id, repoId })

    processQueue()

    return { jobId: id, status: activeJobs.has(id) ? 'downloading' : 'queued', queuePosition: downloadQueue.length, repoId }
  })

  // ─── Image Model Paths IPC ──────────────────────────────────────────────────
  // Return all stored image model paths so the UI can check availability
  ipcMain.handle('image:getModelPaths', async () => {
    return db.getAllImageModelPaths()
  })
}
