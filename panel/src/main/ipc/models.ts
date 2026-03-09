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
  join(homedir(), '.cache/huggingface/hub'),
  join(homedir(), '.exo/models'),
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
  // Skip common system and git directories. We keep 'snapshots' unskipped so it can descend into standard HF Hub caches if needed.
  const SKIP_DIRS = ['.locks', 'blobs', 'refs', '.git', '.cache']

  async function scanRecursive(currentPath: string, depth: number, maxDepth: number) {
    if (depth > maxDepth) return

    try {
      // Check if current directory is a valid MLX model
      const format = await detectModelFormat(currentPath)

      // We only support MLX format (.safetensors + config.json). GGUF and unknown formats are ignored.
      if (format === 'mlx') {
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

          models.push({
            id,
            name: id,
            path: currentPath,
            size: formatSize(size),
            format
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

          // Special case: ignore in-progress vmlx downloads
          try {
            await access(join(currentPath, entry.name, '.vmlx-downloading'))
            continue
          } catch (_) { /* not downloading */ }

          await scanRecursive(join(currentPath, entry.name), depth + 1, maxDepth)
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
    // Default to LM Studio models folder if it exists, otherwise home
    const lmStudioPath = join(homedir(), '.lmstudio', 'models')
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

    // Write marker file
    const markerFile = join(job.modelDir, '.vmlx-downloading')
    try {
      await mkdir(job.modelDir, { recursive: true })
      await writeFile(markerFile, `${job.repoId}\n${Date.now()}`, 'utf-8')
    } catch (_) { /* dir may already exist */ }

    console.log(`[DOWNLOADS] Starting: ${job.repoId} → ${job.modelDir}`)

    const proc = spawn(pythonPath, ['-u', '-c', script, job.repoId, job.modelDir], {
      stdio: ['pipe', 'pipe', 'pipe']
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
              completedJobs.push(job)
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
        emitToRenderer('models:downloadError', { jobId: job.id, repoId: job.repoId, error: errorMsg })
      }

      activeJob = null
      job.process = undefined
      completedJobs.push(job)
      // Process next in queue
      processQueue()
    })

    proc.on('error', async (err: Error) => {
      job.status = 'error'
      job.error = `Failed to start download: ${err.message}`
      activeJob = null
      job.process = undefined
      completedJobs.push(job)
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

    return models.map((m: any) => mapHFModel(m))
  })

  // Get recommended models from shieldstackllc
  ipcMain.handle('models:getRecommendedModels', async () => {
    const url = `https://huggingface.co/api/models?author=shieldstackllc&sort=downloads&direction=-1`
    console.log('[MODELS] Fetching shieldstackllc recommended models')

    const response = await fetch(url, { signal: AbortSignal.timeout(15000) })
    if (!response.ok) throw new Error(`HuggingFace API error: ${response.status}`)
    const models = await response.json()

    return models.map((m: any) => mapHFModel(m))
  })

  // Start a download (non-blocking — returns jobId immediately)
  ipcMain.handle('models:startDownload', async (_, repoId: string) => {
    // Check if already queued or downloading
    if (activeJob?.repoId === repoId) return { jobId: activeJob.id, status: 'already_downloading' }
    const queued = downloadQueue.find(j => j.repoId === repoId)
    if (queued) return { jobId: queued.id, status: 'already_queued' }

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
      completedJobs.push(removed)
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
}
