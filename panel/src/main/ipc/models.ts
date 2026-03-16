import { ipcMain, dialog, BrowserWindow } from 'electron'
import { readdir, stat, access, readFile, mkdir, writeFile, unlink, rm, realpath } from 'fs/promises'
import { existsSync, readFileSync } from 'fs'
import { join, basename } from 'path'
import { homedir } from 'os'
import { spawn, ChildProcess } from 'child_process'
import { db } from '../database'
import { detectModelConfigFromDir } from '../model-config-registry'
import { getBundledPythonPath } from '../engine-manager'

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
        if (files.some(f => f === 'jang_config.json' || f === 'jjqf_config.json' || f === 'mxq_config.json')) return 'mlx'
        // config.json without model_type/architectures is not a model (could be scheduler config, etc.)
        return 'unknown'
      } catch {
        return 'unknown'
      }
    }
    // JANG-only model (has jang_config.json + safetensors but config.json may lack model_type)
    if (hasSafetensors && files.some(f => f === 'jang_config.json' || f === 'jjqf_config.json' || f === 'mxq_config.json')) return 'mlx'
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
// Maps named mflux model IDs to HuggingFace repos for pre-downloading.
// These repos contain the quantized mflux-format weights that mflux can load locally.
// Without pre-download, mflux downloads silently with no progress UI.
const IMAGE_MODEL_REPOS: Record<string, Record<number, string>> = {
  'schnell': {
    4: 'madroid/flux.1-schnell-all-in-one-T5xxl-4bit',
    8: 'dhairyashil/FLUX.1-schnell-mflux-8bit',
    0: 'black-forest-labs/FLUX.1-schnell',
  },
  'dev': {
    4: 'dhairyashil/FLUX.1-dev-mflux-4bit',
    8: 'dhairyashil/FLUX.1-dev-mflux-8bit',
    0: 'black-forest-labs/FLUX.1-dev',
  },
  'z-image-turbo': {
    4: 'mflux-community/Z-Image-Turbo-Alpha-v0.1-4bit',
    8: 'carsenk/z-image-turbo-mflux-8bit',
    0: 'mflux-community/Z-Image-Turbo-Alpha-v0.1',
  },
  'flux2-klein-4b': {
    8: 'AITRADER/FLUX2-klein-4B-mlx-8bit',
    0: 'black-forest-labs/FLUX.2-klein-4B',
  },
  'flux2-klein-9b': {
    0: 'black-forest-labs/FLUX.2-klein-9B',
  },
}

/** Resolve a named image model to its HF repo ID based on quantization level */
function resolveImageModelRepo(modelName: string, quantize: number): string | null {
  const repos = IMAGE_MODEL_REPOS[modelName]
  if (!repos) return null
  // Exact match first, then closest available
  if (repos[quantize]) return repos[quantize]
  // Fall back to closest quantization level
  const available = Object.keys(repos).map(Number).sort((a, b) => Math.abs(a - quantize) - Math.abs(b - quantize))
  return repos[available[0]] || null
}

/** Check if a named image model is available locally in HF cache */
function checkImageModelInHFCache(repoId: string): string | null {
  // HF hub cache: ~/.cache/huggingface/hub/models--org--name/snapshots/<hash>/
  const cacheBase = join(homedir(), '.cache/huggingface/hub')
  const cacheDirName = `models--${repoId.replace('/', '--')}`
  const cacheDir = join(cacheBase, cacheDirName, 'snapshots')
  try {
    if (!existsSync(cacheDir)) return null
    const { readdirSync } = require('fs')
    const snapshots = readdirSync(cacheDir)
    if (snapshots.length === 0) return null
    // Return the most recent snapshot path
    return join(cacheDir, snapshots[snapshots.length - 1])
  } catch {
    return null
  }
}

/** Check if a named image model is available locally (HF cache or mlxstudio/models/image) */
function checkImageModelLocal(modelName: string, quantize: number): { available: boolean; localPath?: string; repoId?: string } {
  const repoId = resolveImageModelRepo(modelName, quantize)
  if (!repoId) return { available: false }

  // Check HF cache
  const hfPath = checkImageModelInHFCache(repoId)
  if (hfPath) return { available: true, localPath: hfPath, repoId }

  // Check ~/.mlxstudio/models/image/ for locally saved mflux models
  const imageDir = join(homedir(), '.mlxstudio/models/image')
  try {
    const { readdirSync } = require('fs')
    const dirs = readdirSync(imageDir)
    // Look for model name patterns (e.g., flux1-schnell-4bit, z-image-turbo-4bit)
    const namePatterns = [
      `${modelName}-${quantize}bit`,
      `${modelName.replace(/-/g, '')}-${quantize}bit`,
      modelName,
    ]
    for (const dir of dirs) {
      const lower = dir.toLowerCase()
      for (const pattern of namePatterns) {
        if (lower.includes(pattern.toLowerCase())) {
          const fullPath = join(imageDir, dir)
          // Verify it's actually a model directory (has transformer/ or safetensors)
          try {
            const files = readdirSync(fullPath)
            if (files.includes('transformer') || files.some((f: string) => f.endsWith('.safetensors'))) {
              return { available: true, localPath: fullPath, repoId }
            }
          } catch { }
        }
      }
    }
  } catch { }

  return { available: false, repoId }
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
            for (const cfgName of ['jang_config.json', 'jjqf_config.json', 'mxq_config.json']) {
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
    return modelPaths.map(p => {
      try {
        // Diffusers / mflux image models
        if (existsSync(join(p, 'model_index.json'))) return 'image'
        if (existsSync(join(p, 'transformer')) && existsSync(join(p, 'text_encoder'))) return 'image'
        if (existsSync(join(p, 'transformer')) && existsSync(join(p, 'vae'))) return 'image'
        // Standard text models have config.json with model_type or architectures
        if (existsSync(join(p, 'config.json'))) {
          try {
            const cfg = JSON.parse(readFileSync(join(p, 'config.json'), 'utf-8'))
            // Check if it has pipeline_tag for image generation
            if (cfg.pipeline_tag === 'text-to-image' || cfg.pipeline_tag === 'image-to-image') return 'image'
            // Has model_type or architectures = text model (transformers)
            if (cfg.model_type || cfg.architectures) return 'text'
          } catch {}
        }
        // JANG models are text models (check all config variants)
        if (existsSync(join(p, 'jang_config.json')) || existsSync(join(p, 'jjqf_config.json')) || existsSync(join(p, 'mxq_config.json'))) return 'text'
        // Remote sessions
        if (p.startsWith('remote://')) return 'text'
        return 'unknown'
      } catch {
        return 'unknown'
      }
    })
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
    status: 'queued' | 'downloading' | 'complete' | 'cancelled' | 'error'
    progress?: DownloadProgress
    error?: string
    process?: ChildProcess
    modelDir: string
    wasCancelled?: boolean
  }

  let jobIdCounter = 0
  const downloadQueue: DownloadJob[] = []
  let activeJob: DownloadJob | null = null
  const completedJobs: DownloadJob[] = []
  const MAX_COMPLETED_JOBS = 100
  const trackCompleted = (job: DownloadJob) => {
    completedJobs.push(job)
    if (completedJobs.length > MAX_COMPLETED_JOBS) completedJobs.splice(0, completedJobs.length - MAX_COMPLETED_JOBS)
  }

  // Expose kill function for app quit cleanup
  _killActiveDownload = () => {
    if (activeJob?.process) {
      console.log(`[DOWNLOADS] Killing active download on quit: ${activeJob.repoId}`)
      try { activeJob.process.kill('SIGKILL') } catch (_) { }
    }
  }

  /** Parse HuggingFace tqdm progress from stderr */
  function parseTqdmProgress(line: string): Partial<DownloadProgress> {
    // Strip ANSI escape codes before parsing
    const clean = line.replace(/\x1b\[[0-9;]*[a-zA-Z]/g, '')
    const result: Partial<DownloadProgress> = { raw: clean.trim() }

    // Match filename from "Downloading <filename>:" or "Fetching N files:" patterns
    const fileMatch = clean.match(/(?:Downloading|Fetching)\s+(.+?)(?:\s*:|\s*\|)/)
    if (fileMatch) result.currentFile = fileMatch[1].trim()

    // Match tqdm bar: "  45%|████      | 1.2G/4.5G [01:30<02:30, 45.2MB/s]"
    const tqdmMatch = clean.match(/(\d+)%\|[^|]*\|\s*([\d.]+\w*)\/([\d.]+\w*)\s*\[([^\]<]*)<([^\],]*),\s*([^\]]+)\]/)
    if (tqdmMatch) {
      result.percent = parseInt(tqdmMatch[1], 10)
      result.downloaded = tqdmMatch[2]
      result.total = tqdmMatch[3]
      result.eta = tqdmMatch[5].trim()
      result.speed = tqdmMatch[6].trim()
    } else {
      // Simpler percent match: " 45%|" or "45%|" (tqdm may start at line beginning after \r)
      const simplePercent = clean.match(/\b(\d+)%\|/)
      if (simplePercent) result.percent = parseInt(simplePercent[1], 10)
    }

    // Match "Fetching N files:  45%|" for files progress
    const filesMatch = clean.match(/Fetching\s+(\d+)\s+files.*?(\d+)%/)
    if (filesMatch) result.filesProgress = `${Math.round(parseInt(filesMatch[2]) * parseInt(filesMatch[1]) / 100)}/${filesMatch[1]}`

    return result
  }

  function emitToRenderer(channel: string, data: any) {
    try {
      const win = BrowserWindow.getAllWindows()[0]
      if (win && !win.isDestroyed()) {
        win.webContents.send(channel, data)
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
    if (activeJob || downloadQueue.length === 0) return

    const job = downloadQueue.shift()!
    job.status = 'downloading'
    activeJob = job

    emitToRenderer('models:downloadStarted', { jobId: job.id, repoId: job.repoId })

    const pythonPath = await getPythonPath()
    // Pass HF token to snapshot_download for gated model access (reads HF_TOKEN env var)
    const script = [
      'import sys, json, os',
      'from huggingface_hub import snapshot_download',
      'repo_id = sys.argv[1]',
      'local_dir = sys.argv[2]',
      'token = os.environ.get("HF_TOKEN") or None',
      'try:',
      '    path = snapshot_download(repo_id, local_dir=local_dir, token=token)',
      '    print(json.dumps({"status": "complete", "path": path}), flush=True)',
      'except KeyboardInterrupt:',
      '    print(json.dumps({"status": "cancelled"}), flush=True)',
      '    sys.exit(0)',
      'except Exception as e:',
      '    print(json.dumps({"status": "error", "error": str(e)}), flush=True)',
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
    let lastStderr = ''
    let lastProgress: Partial<DownloadProgress> = {}

    proc.stdout?.on('data', (data: Buffer) => {
      stdout += data.toString()
    })

    proc.stderr?.on('data', (data: Buffer) => {
      const chunk = data.toString()
      lastStderr = chunk
      // tqdm uses \r to overwrite progress lines — split on \r and \n to get each update
      const segments = chunk.split(/[\r\n]+/).filter(s => s.trim())
      // Parse each segment and keep the best (highest percent) result
      let bestParsed: Partial<DownloadProgress> = { raw: chunk.trim() }
      let bestPercent = lastProgress.percent ?? -1
      for (const seg of segments) {
        const parsed = parseTqdmProgress(seg)
        if (parsed.percent != null && parsed.percent >= bestPercent) {
          bestParsed = { ...bestParsed, ...parsed }
          bestPercent = parsed.percent
        } else if (parsed.currentFile || parsed.filesProgress) {
          // Merge metadata even if percent didn't improve
          if (parsed.currentFile) bestParsed.currentFile = parsed.currentFile
          if (parsed.filesProgress) bestParsed.filesProgress = parsed.filesProgress
        }
      }
      // If no segment had percent but we got metadata, still merge
      if (bestParsed.percent == null && segments.length > 0) {
        const lastSeg = parseTqdmProgress(segments[segments.length - 1])
        bestParsed = { ...bestParsed, ...lastSeg }
      }
      // Merge with last known progress (tqdm lines may only have partial info)
      lastProgress = { ...lastProgress, ...bestParsed }
      job.progress = lastProgress as DownloadProgress
      emitToRenderer('models:downloadProgress', {
        jobId: job.id,
        repoId: job.repoId,
        progress: lastProgress
      })
    })

    proc.on('close', async (code: number | null) => {
      console.log(`[DOWNLOADS] Process exited: ${job.repoId} code=${code} cancelled=${job.wasCancelled}`)

      if (job.wasCancelled) {
        try { await unlink(markerFile) } catch (_) { }
        // Remove partial model directory to prevent incomplete models from appearing in scan
        try { await rm(job.modelDir, { recursive: true, force: true }) } catch (_) { }
        job.status = 'cancelled'
        emitToRenderer('models:downloadComplete', { jobId: job.id, repoId: job.repoId, status: 'cancelled' })
      } else if (code === 0) {
        try { await unlink(markerFile) } catch (_) { }
        job.status = 'complete'
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
              activeJob = null
              job.process = undefined
              trackCompleted(job)
              processQueue()
              return
            }
            errorMsg = result.error || errorMsg
          }
        } catch { }
        if (!errorMsg.includes(lastStderr.slice(0, 100)) && lastStderr.trim()) {
          errorMsg += `: ${lastStderr.slice(0, 200)}`
        }
        try { await unlink(markerFile) } catch (_) { }
        job.status = 'error'
        job.error = errorMsg
        // Detect gated/auth errors (401/403) to prompt user to add HF token
        const isGatedError = /40[13]|gated|access.*denied|authentication|authorized/i.test(errorMsg)
        emitToRenderer('models:downloadError', { jobId: job.id, repoId: job.repoId, error: errorMsg, gated: isGatedError })
      }

      activeJob = null
      job.process = undefined
      trackCompleted(job)
      // Process next in queue
      processQueue()
    })

    proc.on('error', async (err: Error) => {
      job.status = 'error'
      job.error = `Failed to start download: ${err.message}`
      activeJob = null
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
    // Filter by model type: 'image' searches text-to-image models, default searches MLX text models
    if (modelType === 'image') {
      params.set('filter', 'text-to-image')
    } else {
      params.set('filter', 'mlx')
    }
    // HF API only supports direction=-1 (descending). For ascending, we fetch
    // descending and reverse client-side.
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
    const url = `https://huggingface.co/api/models?${params}`
    console.log(`[MODELS] Searching HuggingFace: ${query} (sort=${sortBy || 'downloads'} dir=${sortDir || 'desc'})`)

    const response = await fetch(url, { signal: AbortSignal.timeout(15000) })
    if (!response.ok) throw new Error(`HuggingFace API error: ${response.status}`)
    const models = await response.json()

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
    // Check if already queued or downloading
    if (activeJob?.repoId === repoId) return { jobId: activeJob.id, status: 'already_downloading' }
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
    console.log(`[DOWNLOADS] Queued: ${repoId} (${downloadQueue.length} in queue)`)

    // Kick off processing (no-op if already running)
    processQueue()

    return { jobId: id, status: 'queued', queuePosition: downloadQueue.length }
  })

  // Cancel a download by jobId (or cancel active if no jobId)
  ipcMain.handle('models:cancelDownload', async (_, jobId?: string) => {
    // Cancel from queue first
    const qIdx = downloadQueue.findIndex(j => jobId ? j.id === jobId : true)
    if (qIdx >= 0) {
      const removed = downloadQueue.splice(qIdx, 1)[0]
      removed.status = 'cancelled'
      trackCompleted(removed)
      console.log(`[DOWNLOADS] Cancelled queued: ${removed.repoId}`)
      emitToRenderer('models:downloadComplete', { jobId: removed.id, repoId: removed.repoId, status: 'cancelled' })
      return { success: true }
    }

    // Cancel active
    if (activeJob && (!jobId || activeJob.id === jobId)) {
      console.log(`[DOWNLOADS] Cancelling active: ${activeJob.repoId}`)
      activeJob.wasCancelled = true
      activeJob.process?.kill('SIGTERM')
      return { success: true }
    }

    return { success: false, error: 'No matching download found' }
  })

  // Get current download status (all jobs)
  ipcMain.handle('models:getDownloadStatus', async () => {
    return {
      active: activeJob ? { jobId: activeJob.id, repoId: activeJob.repoId, progress: activeJob.progress } : null,
      queue: downloadQueue.map(j => ({ jobId: j.id, repoId: j.repoId })),
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
    if (activeJob?.repoId === repoId || downloadQueue.find(j => j.repoId === repoId)) {
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

    // Check if already queued or downloading
    if (activeJob?.repoId === repoId) return { jobId: activeJob.id, status: 'already_downloading', repoId }
    const queued = downloadQueue.find(j => j.repoId === repoId)
    if (queued) return { jobId: queued.id, status: 'already_queued', repoId }

    // Queue the download using the standard system
    const id = `dl_${++jobIdCounter}_${Date.now()}`
    const targetDir = getDownloadDirectory()
    const modelDir = join(targetDir, repoId)

    const job: DownloadJob = { id, repoId, status: 'queued', modelDir }
    downloadQueue.push(job)
    console.log(`[DOWNLOADS] Queued image model: ${repoId} (${downloadQueue.length} in queue)`)

    processQueue()

    return { jobId: id, status: 'queued', queuePosition: downloadQueue.length, repoId }
  })
}
