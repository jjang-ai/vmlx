import { ipcMain, dialog, BrowserWindow } from 'electron'
import { readdir, stat, access, readFile, mkdir, writeFile, unlink } from 'fs/promises'
import { join, basename } from 'path'
import { homedir } from 'os'
import { spawn, ChildProcess } from 'child_process'
import { db } from '../database'
import { detectModelConfigFromDir } from '../model-config-registry'
import { getBundledPythonPath } from '../vllm-manager'

/** Generation defaults read from a model's generation_config.json */
export interface GenerationDefaults {
  temperature?: number
  topP?: number
  topK?: number
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
}

/** Check if a model directory contains MLX-format files (safetensors + config.json) */
async function detectModelFormat(modelPath: string): Promise<'mlx' | 'gguf' | 'unknown'> {
  try {
    const files = await readdir(modelPath)
    const hasGGUF = files.some(f => f.endsWith('.gguf') || f.endsWith('.gguf.part'))
    const hasSafetensors = files.some(f => f.endsWith('.safetensors'))
    const hasConfig = files.includes('config.json')

    if (hasSafetensors && hasConfig) return 'mlx'
    if (hasGGUF) return 'gguf'
    return 'unknown'
  } catch {
    return 'unknown'
  }
}

const BUILTIN_MODEL_PATHS = [
  join(homedir(), '.lmstudio/models'),
  join(homedir(), '.cache/huggingface/hub')
]

const SETTINGS_KEY = 'model_scan_directories'

/** Get the list of directories to scan: user-configured + built-in defaults */
function getModelDirectories(): string[] {
  const saved = db.getSetting(SETTINGS_KEY)
  if (saved) {
    try {
      const userDirs: string[] = JSON.parse(saved)
      // Merge: user dirs first, then built-in defaults (deduplicated)
      const all = [...userDirs]
      for (const d of BUILTIN_MODEL_PATHS) {
        if (!all.includes(d)) all.push(d)
      }
      return all
    } catch {
      return BUILTIN_MODEL_PATHS
    }
  }
  return BUILTIN_MODEL_PATHS
}

/** Get only user-configured directories (not the built-in defaults) */
function getUserDirectories(): string[] {
  const saved = db.getSetting(SETTINGS_KEY)
  if (saved) {
    try {
      return JSON.parse(saved)
    } catch {
      return []
    }
  }
  return []
}

function setUserDirectories(dirs: string[]): void {
  db.setSetting(SETTINGS_KEY, JSON.stringify(dirs))
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
  const gb = bytes / (1024 * 1024 * 1024)
  if (gb >= 1) {
    return `~${gb.toFixed(1)} GB`
  }
  const mb = bytes / (1024 * 1024)
  return `~${mb.toFixed(0)} MB`
}

async function scanModelsInPath(basePath: string): Promise<ModelInfo[]> {
  const models: ModelInfo[] = []

  // Skip directories that are not actual models
  const SKIP_DIRS = ['.locks', 'blobs', 'refs', 'snapshots', '.git', '.cache']

  try {
    await access(basePath) // Verify path exists
    const entries = await readdir(basePath, { withFileTypes: true })

    for (const entry of entries) {
      if (!entry.isDirectory()) continue
      if (SKIP_DIRS.includes(entry.name)) continue

      const modelPath = join(basePath, entry.name)

      // Check for nested structure (org/model-name)
      try {
        const subEntries = await readdir(modelPath, { withFileTypes: true })
        const hasSubDirs = subEntries.some(e => e.isDirectory() && !SKIP_DIRS.includes(e.name))

        if (hasSubDirs) {
          // Scan subdirectories (likely org/model-name structure)
          for (const subEntry of subEntries) {
            if (!subEntry.isDirectory()) continue
            if (SKIP_DIRS.includes(subEntry.name)) continue

            const subModelPath = join(modelPath, subEntry.name)

            // Skip in-progress downloads
            try { await access(join(subModelPath, '.vmlx-downloading')); continue } catch (_) { /* not downloading */ }

            const size = await getDirectorySize(subModelPath)

            // Skip empty models
            if (size < 1024 * 1024) continue // Less than 1 MB

            const format = await detectModelFormat(subModelPath)
            // Skip GGUF models — vllm-mlx only supports MLX (safetensors)
            if (format === 'gguf') continue

            models.push({
              id: `${entry.name}/${subEntry.name}`,
              name: `${entry.name}/${subEntry.name}`,
              path: subModelPath,
              size: formatSize(size),
              format,
            })
          }
        } else {
          // Single-level model directory

          // Skip in-progress downloads
          try { await access(join(modelPath, '.vmlx-downloading')); continue } catch (_) { /* not downloading */ }

          const size = await getDirectorySize(modelPath)

          // Skip empty models
          if (size < 1024 * 1024) continue // Less than 1 MB

          const format = await detectModelFormat(modelPath)
          // Skip GGUF models — vllm-mlx only supports MLX (safetensors)
          if (format === 'gguf') continue

          models.push({
            id: entry.name,
            name: entry.name,
            path: modelPath,
            size: formatSize(size),
            format,
          })
        }
      } catch (error) {
        console.error(`Error scanning ${modelPath}:`, error)
      }
    }
  } catch (error) {
    // Directory doesn't exist or not accessible — skip silently
  }

  return models
}

export function registerModelHandlers(): void {
  // Scan for available models in all configured directories
  ipcMain.handle('models:scan', async () => {
    const dirs = getModelDirectories()
    console.log('[MODELS] Scanning directories:', dirs)
    const allModels: ModelInfo[] = []

    for (const basePath of dirs) {
      try {
        const models = await scanModelsInPath(basePath)
        allModels.push(...models)
        console.log(`[MODELS] Found ${models.length} models in ${basePath}`)
      } catch (error) {
        console.error(`[MODELS] Error scanning ${basePath}:`, error)
      }
    }

    console.log(`[MODELS] Total models found: ${allModels.length}`)
    return allModels
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

  // Get all scan directories (user + built-in)
  ipcMain.handle('models:getDirectories', async () => {
    return {
      directories: getModelDirectories(),
      userDirectories: getUserDirectories(),
      builtinDirectories: BUILTIN_MODEL_PATHS
    }
  })

  // Add a directory to the scan list
  ipcMain.handle('models:addDirectory', async (_, dirPath: string) => {
    const userDirs = getUserDirectories()
    // Normalize and deduplicate
    const normalized = dirPath.replace(/\/+$/, '')
    if (userDirs.includes(normalized) || BUILTIN_MODEL_PATHS.includes(normalized)) {
      return { success: false, error: 'Directory already in scan list' }
    }
    // Verify the directory exists
    try {
      await access(normalized)
    } catch {
      return { success: false, error: 'Directory does not exist or is not accessible' }
    }
    userDirs.push(normalized)
    setUserDirectories(userDirs)
    return { success: true }
  })

  // Remove a user directory from the scan list
  ipcMain.handle('models:removeDirectory', async (_, dirPath: string) => {
    const userDirs = getUserDirectories()
    const filtered = userDirs.filter(d => d !== dirPath)
    setUserDirectories(filtered)
    return { success: true }
  })

  // Detect model config (tool/reasoning parser, cache type) from model directory
  ipcMain.handle('models:detect-config', async (_, modelPath: string) => {
    return detectModelConfigFromDir(modelPath)
  })

  // Open a native directory picker dialog
  ipcMain.handle('models:browseDirectory', async () => {
    const result = await dialog.showOpenDialog({
      properties: ['openDirectory'],
      title: 'Select Model Directory'
    })
    if (result.canceled || result.filePaths.length === 0) {
      return { canceled: true }
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

  // Active download tracking
  let activeDownload: { process: ChildProcess; repoId: string } | null = null

  // Search HuggingFace for MLX models
  ipcMain.handle('models:searchHF', async (_, query: string) => {
    const params = new URLSearchParams({
      search: query,
      filter: 'mlx',
      sort: 'downloads',
      direction: '-1',
      limit: '30'
    })
    const url = `https://huggingface.co/api/models?${params}`
    console.log(`[MODELS] Searching HuggingFace: ${query}`)

    const response = await fetch(url, { signal: AbortSignal.timeout(15000) })
    if (!response.ok) throw new Error(`HuggingFace API error: ${response.status}`)
    const models = await response.json()

    return models.map((m: any) => ({
      id: m.modelId || m.id,
      author: m.author,
      downloads: m.downloads,
      likes: m.likes,
      lastModified: m.lastModified,
      tags: m.tags || [],
      pipelineTag: m.pipeline_tag
    }))
  })

  // Get recommended models from shieldstackllc
  ipcMain.handle('models:getRecommendedModels', async () => {
    const url = `https://huggingface.co/api/models?author=shieldstackllc&sort=downloads&direction=-1`
    console.log('[MODELS] Fetching shieldstackllc recommended models')

    const response = await fetch(url, { signal: AbortSignal.timeout(15000) })
    if (!response.ok) throw new Error(`HuggingFace API error: ${response.status}`)
    const models = await response.json()

    return models.map((m: any) => ({
      id: m.modelId || m.id,
      author: m.author,
      downloads: m.downloads,
      likes: m.likes,
      lastModified: m.lastModified,
      tags: m.tags || [],
      pipelineTag: m.pipeline_tag
    }))
  })

  // Download a model from HuggingFace
  ipcMain.handle('models:downloadModel', async (_, repoId: string) => {
    if (activeDownload) {
      throw new Error(`Already downloading ${activeDownload.repoId}`)
    }

    const targetDir = getDownloadDirectory()
    const modelDir = join(targetDir, repoId)

    // Find Python to use (bundled first, fallback to system)
    const bundledPython = getBundledPythonPath()
    let pythonPath: string
    if (bundledPython) {
      try {
        await access(bundledPython)
        pythonPath = bundledPython
      } catch {
        pythonPath = 'python3'
      }
    } else {
      pythonPath = 'python3'
    }

    const script = [
      'import sys, json',
      'from huggingface_hub import snapshot_download',
      'repo_id = sys.argv[1]',
      'local_dir = sys.argv[2]',
      'try:',
      '    path = snapshot_download(repo_id, local_dir=local_dir)',
      '    print(json.dumps({"status": "complete", "path": path}), flush=True)',
      'except KeyboardInterrupt:',
      '    print(json.dumps({"status": "cancelled"}), flush=True)',
      '    sys.exit(0)',
      'except Exception as e:',
      '    print(json.dumps({"status": "error", "error": str(e)}), flush=True)',
      '    sys.exit(1)',
    ].join('\n')

    console.log(`[MODELS] Downloading ${repoId} → ${modelDir}`)

    // Write marker file to indicate download in progress (model scanner skips these)
    const markerFile = join(modelDir, '.vmlx-downloading')
    try {
      await mkdir(modelDir, { recursive: true })
      await writeFile(markerFile, `${repoId}\n${Date.now()}`, 'utf-8')
    } catch (_) { /* dir may already exist */ }

    return new Promise((resolve, reject) => {
      let wasCancelled = false

      const proc = spawn(pythonPath, ['-u', '-c', script, repoId, modelDir], {
        stdio: ['pipe', 'pipe', 'pipe']
      })

      activeDownload = { process: proc, repoId }

      let stdout = ''
      let lastStderr = ''

      proc.stdout?.on('data', (data: Buffer) => {
        stdout += data.toString()
      })

      proc.stderr?.on('data', (data: Buffer) => {
        const line = data.toString()
        lastStderr = line
        // Emit progress to renderer
        try {
          const win = BrowserWindow.getAllWindows()[0]
          if (win && !win.isDestroyed()) {
            win.webContents.send('models:downloadProgress', {
              repoId,
              progress: line.trim()
            })
          }
        } catch (_) { }
      })

      proc.on('close', async (code: number | null) => {
        activeDownload = null
        console.log(`[MODELS] Download process exited with code ${code}, cancelled=${wasCancelled}`)

        // Handle user-initiated cancel (SIGTERM may exit non-zero without writing JSON)
        if (wasCancelled) {
          resolve({ status: 'cancelled' })
          return
        }

        if (code === 0) {
          // Remove marker on success
          try { await unlink(markerFile) } catch (_) { }
          try {
            const lines = stdout.trim().split('\n')
            const result = JSON.parse(lines[lines.length - 1])
            resolve(result)
          } catch {
            resolve({ status: 'complete', path: modelDir })
          }
        } else {
          try {
            const lines = stdout.trim().split('\n').filter(Boolean)
            if (lines.length > 0) {
              const result = JSON.parse(lines[lines.length - 1])
              if (result.status === 'cancelled') {
                resolve({ status: 'cancelled' })
                return
              }
              reject(new Error(result.error || `Download failed (exit ${code})`))
              return
            }
          } catch { }
          reject(new Error(`Download failed (exit ${code}): ${lastStderr.slice(0, 200)}`))
        }
      })

      proc.on('error', (err: Error) => {
        activeDownload = null
        reject(new Error(`Failed to start download: ${err.message}`))
      })

      // Expose cancel flag setter for the cancel handler
      ;(proc as any).__cancelDownload = () => { wasCancelled = true }
    })
  })

  // Cancel active download
  ipcMain.handle('models:cancelDownload', async () => {
    if (activeDownload) {
      console.log(`[MODELS] Cancelling download: ${activeDownload.repoId}`)
      // Set cancel flag before killing so close handler knows this was intentional
      ;(activeDownload.process as any).__cancelDownload?.()
      activeDownload.process.kill('SIGTERM')
      activeDownload = null
      return { success: true }
    }
    return { success: false, error: 'No active download' }
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
      title: 'Select Model Download Directory',
      defaultPath: getDownloadDirectory()
    })
    if (result.canceled || result.filePaths.length === 0) {
      return { canceled: true }
    }
    return { canceled: false, path: result.filePaths[0] }
  })
}
