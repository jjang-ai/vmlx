import { spawn, ChildProcess, execSync, execFileSync, execFile as execFileCallback } from 'child_process'
import { promisify } from 'util'
import { lookup } from 'dns'
import { EventEmitter } from 'events'
import { existsSync, readdirSync, statSync } from 'fs'
import { createServer } from 'net'
import { homedir, totalmem, freemem } from 'os'
import { join } from 'path'
import { v4 as uuidv4 } from 'uuid'
import { db, Session } from './database'

const execFileAsync = promisify(execFileCallback)

export type { ServerConfig, DetectedProcess } from './server'
import type { ServerConfig, DetectedProcess } from './server'
import { detectModelConfigFromDir } from './model-config-registry'
import { getBundledPythonPath } from './engine-manager'

/** Result of findEnginePath: either bundled Python or a system binary */
type EnginePath =
  | { type: 'bundled'; pythonPath: string }
  | { type: 'system'; binaryPath: string }

interface ManagedProcess {
  process: ChildProcess | null
  adoptedPid: number | null
  lastStderr?: string  // Last stderr line for error reporting
  exitCode?: number | null
  exitSignal?: string | null  // Signal that killed the process (e.g. SIGKILL for OOM)
  intentionalStop?: boolean   // Set true when stopSession sends SIGTERM — prevents crash misreport
}

/** Normalize model paths for consistent matching: resolve and strip trailing slashes */
function normalizePath(p: string): string {
  return p.replace(/\/+$/, '')
}

/** Resolve bind address to connectable address (0.0.0.0 → 127.0.0.1) */
export function connectHost(host: string): string {
  return host === '0.0.0.0' ? '127.0.0.1' : host
}

/** Estimate model memory usage from safetensors file sizes. Returns bytes or 0 if unknown. */
function estimateModelMemory(modelPath: string): number {
  try {
    const files = readdirSync(modelPath)
    let totalBytes = 0
    for (const file of files) {
      if (file.endsWith('.safetensors')) {
        totalBytes += statSync(join(modelPath, file)).size
      }
    }
    // Model size on disk + ~30% overhead for KV cache, activations, framework
    return Math.round(totalBytes * 1.3)
  } catch (_) {
    return 0
  }
}

/**
 * Resolve .local (mDNS/Bonjour) hostnames to IPv4 before fetch.
 * Node.js/undici's fetch resolves .local to IPv6 link-local (fe80::...)
 * which is unreachable without a zone ID, causing "fetch failed".
 * This replaces the hostname with the resolved IPv4 address.
 *
 * Results are cached for 60s to avoid redundant DNS lookups on every
 * message send and health check (previously added 50-100ms per call).
 */
const resolvedUrlCache = new Map<string, { url: string; timestamp: number }>()
const RESOLVE_URL_CACHE_TTL = 60_000 // 60 seconds

export async function resolveUrl(url: string): Promise<string> {
  const cached = resolvedUrlCache.get(url)
  if (cached && Date.now() - cached.timestamp < RESOLVE_URL_CACHE_TTL) {
    return cached.url
  }

  try {
    const parsed = new URL(url)
    if (parsed.hostname.endsWith('.local')) {
      const ip = await new Promise<string>((resolve, reject) => {
        lookup(parsed.hostname, { family: 4 }, (err, addr) => {
          if (err) reject(err); else resolve(addr)
        })
      })
      parsed.hostname = ip
      const resolved = parsed.toString().replace(/\/+$/, '')
      console.log(`[DNS] Resolved .local: ${url} → ${resolved}`)
      resolvedUrlCache.set(url, { url: resolved, timestamp: Date.now() })
      return resolved
    }
  } catch (e) {
    console.log(`[DNS] Failed to resolve ${url}:`, e)
  }
  resolvedUrlCache.set(url, { url, timestamp: Date.now() })
  return url
}

export class SessionManager extends EventEmitter {
  private processes = new Map<string, ManagedProcess>()
  private monitorInterval: ReturnType<typeof setInterval> | null = null
  private failCounts = new Map<string, number>()
  /** Per-session operation lock to prevent concurrent start/stop races */
  private operationLocks = new Map<string, Promise<void>>()
  /** Global creation lock to prevent port assignment races between concurrent createSession calls */
  private creationLock: Promise<void> = Promise.resolve()
  /** Timestamp of last successful health check per session (used to skip redundant per-message checks) */
  private lastHealthyAt = new Map<string, number>()
  /** Per-session ring buffer for log lines (capped at LOG_BUFFER_MAX_LINES) */
  private logBuffers = new Map<string, string[]>()
  private static readonly LOG_BUFFER_MAX_LINES = 2000
  // Allow up to 60 consecutive health check failures (5s * 60 = 5 min)
  // before marking session as down. Long prefill operations (e.g. 44k+
  // tokens) can block the server's event loop for 30+ seconds.
  private static readonly MAX_FAIL_COUNT = 60

  // ── Idle / Sleep tracking ──
  /** Timestamp of last API request per session (for idle detection) */
  private lastRequestAt = new Map<string, number>()
  /** Default idle timeouts in milliseconds */
  private static readonly DEFAULT_SOFT_TIMEOUT_TEXT_MS = 10 * 60 * 1000   // 10 min
  private static readonly DEFAULT_HARD_TIMEOUT_TEXT_MS = 30 * 60 * 1000   // 30 min
  private static readonly DEFAULT_SOFT_TIMEOUT_IMAGE_MS = 5 * 60 * 1000   // 5 min
  private static readonly DEFAULT_HARD_TIMEOUT_IMAGE_MS = 15 * 60 * 1000  // 15 min

  constructor() {
    super()
  }

  /** Get timestamp of last successful health check for a session (0 if never checked) */
  getLastHealthyAt(sessionId: string): number {
    return this.lastHealthyAt.get(sessionId) || 0
  }

  // Loading progress patterns — matched against engine stdout/stderr to detect loading phase
  private static readonly LOAD_PROGRESS_PATTERNS: Array<{ pattern: RegExp; label: string; progress: number }> = [
    { pattern: /Loading model:/, label: 'Initializing...', progress: 5 },
    { pattern: /System memory before load/, label: 'Checking memory...', progress: 10 },
    { pattern: /Loading model with (?:Simple|Batched)Engine/, label: 'Creating engine...', progress: 15 },
    { pattern: /JANG v2 detected/, label: 'Loading JANG weights...', progress: 20 },
    { pattern: /Loading JANG VL model/, label: 'Loading JANG VL model...', progress: 20 },
    { pattern: /Loading MLLM:/, label: 'Loading vision model...', progress: 20 },
    { pattern: /Loading image model:/, label: 'Loading image model...', progress: 20 },
    { pattern: /Loading \d+ safetensors shards/, label: 'Loading weights...', progress: 30 },
    { pattern: /JANG v[12] loaded in/, label: 'Weights loaded', progress: 50 },
    { pattern: /Model loaded successfully/, label: 'Model loaded', progress: 55 },
    { pattern: /MLLM loaded successfully/, label: 'Vision model loaded', progress: 55 },
    { pattern: /JANG VL model loaded/, label: 'JANG VL loaded', progress: 55 },
    { pattern: /Image model loaded in/, label: 'Image model loaded', progress: 55 },
    { pattern: /model loaded \((?:simple|batched) mode\)/, label: 'Engine ready', progress: 60 },
    { pattern: /Metal GPU memory after load/, label: 'Configuring GPU...', progress: 70 },
    { pattern: /Saved \d+\/\d+ layer weights to SSD/, label: 'Saving weights to SSD...', progress: 72 },
    { pattern: /SSD weight index:/, label: 'Building weight index...', progress: 73 },
    { pattern: /SSD per-layer weight recycling configured/, label: 'SSD streaming ready', progress: 74 },
    { pattern: /KV cache quantization/, label: 'Setting up KV cache...', progress: 75 },
    { pattern: /(?:Chat template loaded|Applied custom chat template)/, label: 'Loading chat template...', progress: 80 },
    { pattern: /Native tool format enabled/, label: 'Configuring tools...', progress: 85 },
    { pattern: /Default max tokens:/, label: 'Finalizing config...', progress: 90 },
    { pattern: /Uvicorn running on/, label: 'Server started', progress: 95 },
  ]

  // Track last emitted progress per session to avoid duplicate events
  private loadProgressState = new Map<string, number>()

  /** Check a log line for loading progress and emit event if phase advanced */
  private checkLoadProgress(sessionId: string, text: string): void {
    for (const { pattern, label, progress } of SessionManager.LOAD_PROGRESS_PATTERNS) {
      if (pattern.test(text)) {
        const current = this.loadProgressState.get(sessionId) ?? 0
        if (progress > current) {
          this.loadProgressState.set(sessionId, progress)
          this.emit('session:loadProgress', { sessionId, label, progress })
        }
        break
      }
    }
  }

  /** Append log data to the per-session ring buffer */
  pushLog(sessionId: string, data: string): void {
    let buffer = this.logBuffers.get(sessionId)
    if (!buffer) {
      buffer = []
      this.logBuffers.set(sessionId, buffer)
    }
    const timestamp = new Date().toISOString().slice(11, 23) // HH:mm:ss.SSS
    const lines = data.split('\n')
    for (const line of lines) {
      if (!line && lines.length > 1) continue // skip empty splits from trailing newline
      buffer.push(`[${timestamp}] ${line}`)
    }
    if (buffer.length > SessionManager.LOG_BUFFER_MAX_LINES) {
      buffer.splice(0, buffer.length - SessionManager.LOG_BUFFER_MAX_LINES)
    }
    // Parse log for loading progress indicators
    this.checkLoadProgress(sessionId, data)
  }

  /** Get all buffered log lines for a session */
  getLogs(sessionId: string): string[] {
    return this.logBuffers.get(sessionId) || []
  }

  /** Clear the log buffer for a session */
  clearLogs(sessionId: string): void {
    this.logBuffers.delete(sessionId)
  }

  /**
   * Acquire a per-session operation lock. Serializes start/stop operations
   * for the same session to prevent race conditions (e.g. stop during start,
   * start during stop, rapid start/stop/start).
   *
   * Uses promise-chaining: each caller atomically chains onto the tail of
   * the previous operation. No TOCTOU window between await and set.
   */
  private withSessionLock(sessionId: string, fn: () => Promise<void>): Promise<void> {
    const prev = this.operationLocks.get(sessionId) ?? Promise.resolve()
    const next = prev.catch(() => {}).then(() => fn())
    const tail = next.catch(() => {})
    // Store the chain tail so the next caller awaits us
    this.operationLocks.set(sessionId, tail)
    // Clean up once our operation settles (avoids unbounded map growth)
    tail.then(() => {
      if (this.operationLocks.get(sessionId) === tail) {
        this.operationLocks.delete(sessionId)
      }
    })
    return next
  }

  // ─── Process Detection (reused from ServerManager) ─────────────────

  async detect(): Promise<DetectedProcess[]> {
    const detected: DetectedProcess[] = []

    try {
      const { stdout: output } = await execFileAsync('ps', ['aux'], { encoding: 'utf-8', timeout: 5000 })
      const lines = output.split('\n')

      for (const line of lines) {
        if (line.includes('grep')) continue
        // Detect `vmlx-engine serve`, `python -m vmlx_engine.cli serve`, and `python -m vmlx_engine.server` processes
        const isCliServe = line.includes('vmlx-engine') && line.includes('serve')
        const isPythonModule = line.includes('vmlx_engine') && (line.includes('.cli') || line.includes('.server') || line.includes('--model'))
        if (!isCliServe && !isPythonModule) continue

        const parsed = this.parsePsLine(line)
        if (!parsed) continue

        let healthy = false
        let modelName: string | undefined
        let standbyDepth: 'soft' | 'deep' | null = null
        try {
          const res = await fetch(
            `http://127.0.0.1:${parsed.port}/health`,
            { signal: AbortSignal.timeout(2000) }
          )
          if (res.ok) {
            const data = await res.json()
            healthy = true
            modelName = data.model_name
            // Detect standby state for proper re-adoption
            if (data.status === 'standby_soft') standbyDepth = 'soft'
            else if (data.status === 'standby_deep') standbyDepth = 'deep'
          }
        } catch (_) { }

        detected.push({
          pid: parsed.pid,
          port: parsed.port,
          modelPath: parsed.modelPath,
          healthy,
          modelName,
          standbyDepth
        })
      }
    } catch (_) { }

    return detected
  }

  private parsePsLine(line: string): { pid: number; port: number; modelPath: string } | null {
    try {
      const parts = line.trim().split(/\s+/)
      const pid = parseInt(parts[1])
      if (isNaN(pid)) return null

      const cmdStart = parts.slice(10).join(' ')

      let modelPath = ''

      // Try `serve <model-path> --...` format first (vmlx-engine CLI)
      const serveIdx = cmdStart.indexOf('serve ')
      if (serveIdx !== -1) {
        const afterServe = cmdStart.substring(serveIdx + 6).trim()
        modelPath = afterServe.split(/\s+--/)[0].trim()
      }

      // Try `--model <path>` format (python -m vmlx_engine.server)
      if (!modelPath) {
        const modelMatch = cmdStart.match(/--model\s+(\S+)/)
        if (modelMatch) modelPath = modelMatch[1]
      }

      if (!modelPath) return null

      // Normalize: strip trailing slashes for consistent matching
      modelPath = normalizePath(modelPath)

      let port = 8000
      const portMatch = cmdStart.match(/--port\s+(\d+)/)
      if (portMatch) port = parseInt(portMatch[1])

      return { pid, port, modelPath }
    } catch (_) {
      return null
    }
  }

  // ─── Session Lifecycle ─────────────────────────────────────────────

  async createSession(modelPath: string, config: Partial<ServerConfig>): Promise<Session> {
    // Serialize all session creation to prevent port assignment race conditions.
    // Without this, concurrent createSession calls can both see the same DB snapshot
    // and assign the same port (TOCTOU race in findAvailablePort).
    let unlock!: () => void
    const prev = this.creationLock
    this.creationLock = new Promise<void>(r => { unlock = r })
    await prev
    try {
      return await this._createSessionInner(modelPath, config)
    } finally {
      unlock()
    }
  }

  private async _createSessionInner(modelPath: string, config: Partial<ServerConfig>): Promise<Session> {
    // Normalize path to prevent trailing-slash mismatches
    modelPath = normalizePath(modelPath)

    // Check if session already exists for this model path
    const existing = db.getSessionByModelPath(modelPath)
    if (existing) {
      // Merge new config into existing (don't overwrite unspecified fields)
      let existingConfig: Record<string, any> = {}
      try { existingConfig = JSON.parse(existing.config || '{}') } catch (_) { }
      const host = (config.host as string) || existing.host
      const port = (config.port as number) || existing.port
      const merged = { ...existingConfig, ...config, modelPath, host, port }
      db.updateSession(existing.id, {
        config: JSON.stringify(merged),
        host,
        port
      })
      return db.getSession(existing.id)!
    }

    const id = uuidv4()
    const host = config.host || '0.0.0.0'
    const port = config.port || await this.findAvailablePort()
    const now = Date.now()

    const session: Session = {
      id,
      modelPath,
      modelName: modelPath.split('/').pop() || modelPath,
      host,
      port,
      status: 'stopped',
      config: JSON.stringify({ ...config, modelPath, port, host }),
      createdAt: now,
      updatedAt: now,
      type: 'local'
    }

    db.createSession(session)
    this.emit('session:created', session)
    return session
  }

  async createRemoteSession(params: {
    remoteUrl: string
    remoteApiKey?: string
    remoteModel: string
    remoteOrganization?: string
  }): Promise<Session> {
    const url = new URL(params.remoteUrl)
    const modelPath = `remote://${params.remoteModel}@${url.host}`

    const existing = db.getSessionByModelPath(modelPath)
    if (existing) {
      db.updateSession(existing.id, {
        remoteUrl: params.remoteUrl,
        remoteApiKey: params.remoteApiKey,
        remoteModel: params.remoteModel,
        remoteOrganization: params.remoteOrganization
      })
      return db.getSession(existing.id)!
    }

    const id = uuidv4()
    const host = url.hostname
    // Remote sessions don't bind a local port — the port field is just a DB key.
    // Use findAvailablePort to avoid UNIQUE constraint conflicts when multiple
    // remote sessions point to different models on the same host (e.g., port 443).
    const port = await this.findAvailablePort()
    const now = Date.now()

    const session: Session = {
      id,
      modelPath,
      modelName: params.remoteModel,
      host,
      port,
      status: 'stopped',
      config: JSON.stringify({ timeout: 300 }),
      createdAt: now,
      updatedAt: now,
      type: 'remote',
      remoteUrl: params.remoteUrl,
      remoteApiKey: params.remoteApiKey,
      remoteModel: params.remoteModel,
      remoteOrganization: params.remoteOrganization
    }

    db.createSession(session)
    this.emit('session:created', session)
    return session
  }

  async startSession(sessionId: string): Promise<void> {
    // Remote sessions connect instead of starting a local process
    const session = db.getSession(sessionId)
    if (session?.type === 'remote') {
      // Guard: skip if already running or connecting
      if (session.status === 'running' || session.status === 'loading') {
        console.log(`[SESSIONS] Remote session ${sessionId} already ${session.status}, skipping connect`)
        return
      }
      return this._connectRemoteSession(session)
    }

    // Serialize start/stop operations per session to prevent races
    await this.withSessionLock(sessionId, () => this._startSessionInner(sessionId))
  }

  private async _startSessionInner(sessionId: string): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    const managed = this.processes.get(sessionId)
    if (managed?.process || managed?.adoptedPid) {
      throw new Error('Session is already running')
    }

    const config: ServerConfig = JSON.parse(session.config)
    config.modelPath = session.modelPath
    config.host = session.host
    config.port = session.port

    // Apply model_settings.reasoning_mode as server-level default for external API clients
    const modelSettings = db.getModelSettings(session.modelPath)
    if (modelSettings?.reasoning_mode === 'on') config.defaultEnableThinking = true
    else if (modelSettings?.reasoning_mode === 'off') config.defaultEnableThinking = false

    const engineResult = this.findEnginePath()
    if (!engineResult) throw new Error('vmlx-engine not found. Please install it first.')

    // Image models may use mflux named models (e.g., "schnell") that are NOT filesystem paths
    // — skip path/format validation for image sessions, let the Python server handle it
    const isImageSession = config.modelType === 'image'

    if (!isImageSession) {
      if (!existsSync(config.modelPath)) throw new Error(`Model not found at: ${config.modelPath}`)

      // Block starting a session with an actively downloading model
      const downloadMarker = join(config.modelPath, '.vmlx-downloading')
      if (existsSync(downloadMarker)) {
        throw new Error('This model is still downloading. Please wait for the download to complete before starting a session.')
      }

      // Validate model format: vmlx-engine only supports MLX (safetensors) models
      try {
        const files = readdirSync(config.modelPath)
        const hasGGUF = files.some(f => f.endsWith('.gguf') || f.endsWith('.gguf.part'))
        const hasSafetensors = files.some(f => f.endsWith('.safetensors'))
        const hasConfig = files.includes('config.json')

        if (hasGGUF && !hasSafetensors) {
          throw new Error(
            'This model is in GGUF format, which is not supported by vmlx-engine. ' +
            'Please download an MLX-format version (safetensors) from HuggingFace Hub.'
          )
        }
        // Diffusers image models have model_index.json instead of config.json — that's valid
        const hasModelIndex = files.includes('model_index.json')
        const hasTransformerDir = files.includes('transformer')
        if (!hasConfig && !hasModelIndex && !hasTransformerDir) {
          throw new Error(
            'Model directory is missing config.json (text) or model_index.json (image). ' +
            'vmlx-engine requires MLX-format models with config.json and .safetensors files, ' +
            'or diffusers-format image models with model_index.json.'
          )
        }
      } catch (e) {
        if ((e as Error).message.includes('GGUF format') || (e as Error).message.includes('missing config.json') || (e as Error).message.includes('model_index.json')) throw e
        // Ignore filesystem errors — let the server handle them
      }
    } // end if (!isImageSession)

    // Re-detect model config from disk — handles case where model files were
    // replaced with a different model (same folder name, different model_type).
    // User-set overrides (port, host, apiKey, etc.) are preserved.
    if (!isImageSession) {
      try {
        const freshConfig = detectModelConfigFromDir(config.modelPath)
        if (freshConfig) {
          const oldFamily = config.toolCallParser
          // Update auto-detected fields only if user hasn't explicitly overridden them
          // Use === checks, not falsy — '' means "None/disabled" (explicit user choice)
          if (config.toolCallParser === undefined || config.toolCallParser === 'auto') {
            config.toolCallParser = freshConfig.toolParser || 'auto'
          }
          if (config.reasoningParser === undefined || config.reasoningParser === 'auto') {
            config.reasoningParser = freshConfig.reasoningParser || 'auto'
          }
          // Refresh multimodal detection from disk — but only if user hasn't explicitly overridden
          // (undefined = auto-detect, true/false = user forced via "Force On"/"Force Off" in settings)
          if (config.isMultimodal === undefined) {
            config.isMultimodal = freshConfig.isMultimodal
          }
          // Log if model type changed
          if (oldFamily && oldFamily !== 'auto' && freshConfig.toolParser && oldFamily !== freshConfig.toolParser) {
            this.pushLog(sessionId, `[INFO] Model config re-detected from disk (was: ${oldFamily}, now: ${freshConfig.toolParser})`)
          }
          // Update DB with refreshed config
          db.updateSession(sessionId, { config: JSON.stringify(config) })
        }
      } catch (e) {
        this.pushLog(sessionId, `[WARN] Could not re-detect model config: ${(e as Error).message}`)
      }
    }

    // Memory estimation: warn if model is too large for available RAM
    const modelSizeBytes = estimateModelMemory(config.modelPath)
    if (modelSizeBytes > 0) {
      const availableBytes = freemem()
      const totalBytes = totalmem()
      const usagePercent = ((totalBytes - availableBytes) / totalBytes) * 100
      const modelGB = (modelSizeBytes / 1e9).toFixed(1)
      const availGB = (availableBytes / 1e9).toFixed(1)
      const totalGB = (totalBytes / 1e9).toFixed(0)
      console.log(`[SESSION] Model estimate: ~${modelGB} GB | RAM: ${availGB} GB free / ${totalGB} GB total (${usagePercent.toFixed(0)}% used)`)
      this.emit('session:log', { sessionId, data: `Model estimate: ~${modelGB} GB | RAM: ${availGB} GB free / ${totalGB} GB total\n` })
      if (modelSizeBytes > availableBytes * 0.9) {
        console.warn(`[SESSION] WARNING: Model (~${modelGB} GB) may exceed available memory (${availGB} GB free). Risk of system instability.`)
        this.emit('session:log', { sessionId, data: `⚠️  Memory warning: Model requires ~${modelGB} GB but only ${availGB} GB free. Loading may cause system instability or swap.\n` })
      } else if (modelSizeBytes > availableBytes * 0.7) {
        console.log(`[SESSION] Model will use most of available RAM (${modelGB} GB / ${availGB} GB free)`)
        this.emit('session:log', { sessionId, data: `Note: Model (~${modelGB} GB) will use most available memory. KV cache may be limited.\n` })
      }
    }

    // Kill anything on this port first
    await this.killByPort(session.port)

    db.updateSession(sessionId, {
      status: 'loading',
      lastStartedAt: Date.now()
    })
    this.loadProgressState.delete(sessionId) // Reset loading progress for fresh start
    this.emit('session:starting', { sessionId, modelPath: session.modelPath })

    const args = this.buildArgs(config)

    // Ensure PATH includes pyenv/homebrew so the engine finds its Python
    const extraPath = [
      join(homedir(), '.pyenv', 'shims'),
      join(homedir(), '.pyenv', 'bin'),
      '/opt/homebrew/bin',
      '/usr/local/bin',
    ].join(':')
    const spawnEnv: Record<string, string | undefined> = { ...process.env, PATH: `${extraPath}:${process.env.PATH || ''}` }
    // Pass API key via env var (not CLI arg) to avoid exposure in ps aux
    if (config.apiKey) {
      spawnEnv.VLLM_API_KEY = config.apiKey
    }
    // Pass HuggingFace token for gated model access (used by mflux, transformers, etc.)
    const hfToken = db.getSetting('hf_api_key')
    if (hfToken) {
      spawnEnv.HF_TOKEN = hfToken
    }
    // NOTE: We previously set HF_HUB_OFFLINE=1 for image models to prevent mflux from
    // silently downloading multi-GB models. This was removed because it also blocks mflux
    // from reading already-cached files in ~/.cache/huggingface/hub/. Instead, we rely on
    // validateImageModelCompleteness() to warn users about incomplete downloads before start,
    // and the logs panel shows startup progress if mflux does need to fetch missing components.

    let proc: ChildProcess
    if (engineResult.type === 'bundled') {
      // Bundled Python: spawn python3 -s -m vmlx_engine.cli serve <model> --host ... --port ...
      // -s: suppress user site-packages (~/.local/lib/python3.12/site-packages)
      // This avoids shebang path issues with relocatable Python and ensures
      // the app uses ONLY its bundled engine, never system-installed mlx-lm/vmlx-engine.
      const bundledEnv: Record<string, string | undefined> = {
        ...spawnEnv,
        PYTHONNOUSERSITE: '1',  // Extra safety: disable user site-packages
        PYTHONPATH: undefined,  // Clear any inherited PYTHONPATH
      }
      const fullCmd = `${engineResult.pythonPath} -s -m vmlx_engine.cli ${args.join(' ')}`
      this.pushLog(sessionId, `$ ${fullCmd}`)
      this.emit('session:log', { sessionId, data: `$ ${fullCmd}\n` })
      proc = spawn(engineResult.pythonPath, ['-s', '-m', 'vmlx_engine.cli', ...args], {
        env: bundledEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: true,  // Separate process group so we can kill entire group
      })
    } else {
      // System binary: spawn vmlx-engine directly
      const fullCmd = `${engineResult.binaryPath} ${args.join(' ')}`
      this.pushLog(sessionId, `$ ${fullCmd}`)
      this.emit('session:log', { sessionId, data: `$ ${fullCmd}\n` })
      proc = spawn(engineResult.binaryPath, args, {
        env: spawnEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: true,
      })
    }

    this.processes.set(sessionId, { process: proc, adoptedPid: null })

    proc.stdout?.on('data', (data) => {
      const text = data.toString()
      this.pushLog(sessionId, text)
      this.emit('session:log', { sessionId, data: text })
    })
    proc.stderr?.on('data', (data) => {
      const text = data.toString()
      this.pushLog(sessionId, text)
      // Log errors to main console for diagnostics
      if (text.includes('ERROR') || text.includes('Traceback') || text.includes('Exception')) {
        console.error(`[SERVER] ${text.trimEnd()}`)
      }
      this.emit('session:log', { sessionId, data: text })
      // Capture most meaningful stderr line for error reporting.
      // Python exceptions print the error type several lines before the
      // final output (e.g., RuntimeError on line N, then "library not found"
      // on line N+3). Prefer exception lines over the last line.
      const managed = this.processes.get(sessionId)
      if (managed) {
        const lines = text.trim().split('\n').filter((l: string) => l.trim())
        // Look for Python exception lines (most informative)
        const exceptionLine = lines.find((l: string) =>
          /^(RuntimeError|ImportError|ModuleNotFoundError|OSError|ValueError|TypeError|MemoryError|FileNotFoundError):/.test(l.trim()) ||
          /^(mlx|mflux|torch|jax)\./.test(l.trim()) && l.includes('Error')
        )
        if (exceptionLine) {
          managed.lastStderr = exceptionLine.trim()
        } else if (!managed.lastStderr || !/^(RuntimeError|ImportError|ModuleNotFoundError|OSError|ValueError|TypeError|MemoryError|FileNotFoundError):/.test(managed.lastStderr)) {
          // Only overwrite if we haven't already captured an exception line
          const lastLine = lines.pop()
          if (lastLine) managed.lastStderr = lastLine
        }
      }
    })
    proc.stdout?.on('error', () => { })
    proc.stderr?.on('error', () => { })

    proc.on('exit', (code, signal) => {
      const managed = this.processes.get(sessionId)
      const lastStderr = managed?.lastStderr
      const intentional = managed?.intentionalStop === true
      this.processes.delete(sessionId)
      this.failCounts.delete(sessionId)
      const killed = signal === 'SIGKILL'
      const crashed = !intentional && (killed || (code !== null && code !== 0))
      db.updateSession(sessionId, {
        status: crashed ? 'error' : 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now()
      })
      if (crashed) {
        let reason: string
        if (killed) {
          reason = 'Process was killed (SIGKILL) — likely out of memory. Try a smaller/more quantized model, reduce cache size, or close other apps.'
        } else if (lastStderr) {
          reason = `Process exited with code ${code}: ${lastStderr}`
        } else {
          reason = `Process exited with code ${code}`
        }
        this.pushLog(sessionId, `[ERROR] ${reason}`)
        this.emit('session:error', { sessionId, error: reason })
      } else {
        this.pushLog(sessionId, `[INFO] Process stopped (exit code ${code})`)
      }
      // Store exit info for waitForReady to access
      this.processes.set(sessionId, { process: null, adoptedPid: null, exitCode: code, exitSignal: signal, lastStderr })
      this.emit('session:stopped', { sessionId, code, signal })
      // Clean up the exit info after a delay so waitForReady can read it
      setTimeout(() => {
        const m = this.processes.get(sessionId)
        if (m && !m.process && !m.adoptedPid) this.processes.delete(sessionId)
      }, 5000)
    })

    proc.on('error', (error) => {
      this.processes.delete(sessionId)
      this.failCounts.delete(sessionId)
      db.updateSession(sessionId, {
        status: 'error',
        pid: undefined
      })
      this.emit('session:error', { sessionId, error: error.message })
    })

    if (proc.pid) {
      db.updateSession(sessionId, { pid: proc.pid })
    }

    // Wait for health endpoint — use session timeout (min 120s for large models)
    const startupTimeoutMs = Math.max((config.timeout || 300) * 1000, 120000)
    try {
      await this.waitForReady(session.host, session.port, startupTimeoutMs, sessionId)
      db.updateSession(sessionId, { status: 'running' })
      this.touchSession(sessionId)  // Start idle timer from model-ready time
      this.emit('session:ready', { sessionId, port: session.port })
    } catch (err) {
      db.updateSession(sessionId, { status: 'error' })
      this.emit('session:error', { sessionId, error: (err as Error).message })
      throw err
    }
  }

  private async _connectRemoteSession(session: Session): Promise<void> {
    db.updateSession(session.id, { status: 'loading', lastStartedAt: Date.now() })
    this.emit('session:starting', { sessionId: session.id, modelPath: session.modelPath })
    this.pushLog(session.id, `[INFO] Connecting to remote endpoint...`)

    const baseUrl = session.remoteUrl!.replace(/\/+$/, '')
    const headers: Record<string, string> = {}
    if (session.remoteApiKey) headers['Authorization'] = `Bearer ${session.remoteApiKey}`
    if (session.remoteOrganization) headers['OpenAI-Organization'] = session.remoteOrganization

    const url = `${baseUrl}/v1/models`
    const resolvedUrl = await resolveUrl(url)
    this.pushLog(session.id, `[INFO] GET ${url}${resolvedUrl !== url ? ` (resolved: ${resolvedUrl})` : ''}`)
    console.log(`[SESSION] Connecting to remote: ${url}${resolvedUrl !== url ? ` (resolved: ${resolvedUrl})` : ''}`)

    // Retry up to 3 times with increasing delay to handle transient DNS/network issues
    let lastErr: Error | null = null
    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        const res = await fetch(resolvedUrl, {
          headers,
          signal: AbortSignal.timeout(10000)
        })
        if (!res.ok) throw new Error(`Server returned HTTP ${res.status}`)

        this.pushLog(session.id, `[INFO] Connected to remote endpoint (attempt ${attempt})`)
        console.log(`[SESSION] Remote connected: ${url} (attempt ${attempt})`)
        db.updateSession(session.id, { status: 'running' })
        this.lastHealthyAt.set(session.id, Date.now())
        this.emit('session:ready', { sessionId: session.id, port: session.port })
        return
      } catch (err) {
        lastErr = err as Error
        this.pushLog(session.id, `WARNING: Connect attempt ${attempt}/3 failed: ${lastErr.message}`)
        console.log(`[SESSION] Remote connect attempt ${attempt}/3 failed: ${lastErr.message}`)
        if (attempt < 3) await new Promise(r => setTimeout(r, attempt * 1000))
      }
    }

    this.pushLog(session.id, `[ERROR] Cannot connect to remote endpoint: ${lastErr!.message}`)
    db.updateSession(session.id, { status: 'error' })
    this.emit('session:error', { sessionId: session.id, error: `${lastErr!.message} (${url})` })
    throw new Error(`Cannot connect to remote endpoint ${url}: ${lastErr!.message}`)
  }

  async stopSession(sessionId: string): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    // Remote sessions just disconnect (no process to kill) — no lock needed
    if (session.type === 'remote') {
      this.pushLog(sessionId, '[INFO] Disconnected from remote endpoint')
      this.failCounts.delete(sessionId)
      db.updateSession(sessionId, { status: 'stopped', lastStoppedAt: Date.now() })
      this.emit('session:stopped', { sessionId })
      return
    }

    // Serialize start/stop operations per session to prevent races
    await this.withSessionLock(sessionId, async () => {
      this.failCounts.delete(sessionId)
      const managed = this.processes.get(sessionId)

      if (managed?.process) {
        managed.intentionalStop = true
        await this.killChildProcess(managed.process)
        this.processes.delete(sessionId)
      } else if (managed?.adoptedPid) {
        this.killPid(managed.adoptedPid)
        await new Promise(r => setTimeout(r, 1500))
        try { process.kill(managed.adoptedPid, 0); this.killPid(managed.adoptedPid, 'SIGKILL') } catch (_) { }
        this.processes.delete(sessionId)
      } else if (session.pid) {
        // Fallback: kill by stored PID
        this.killPid(session.pid)
        await new Promise(r => setTimeout(r, 1500))
        try { process.kill(session.pid, 0); this.killPid(session.pid, 'SIGKILL') } catch (_) { }
      } else {
        // Last resort: kill by port
        await this.killByPort(session.port)
      }

      db.updateSession(sessionId, {
        status: 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now(),
        standbyDepth: null
      })
      // Clear idle tracking and log buffer
      this.lastRequestAt.delete(sessionId)
      this.logBuffers.delete(sessionId)
      this.emit('session:stopped', { sessionId })
    })
  }

  async deleteSession(sessionId: string): Promise<void> {
    // Stop first if running
    const session = db.getSession(sessionId)
    if (session && (session.status === 'running' || session.status === 'loading' || session.status === 'standby')) {
      await this.stopSession(sessionId)
    }

    // Acquire lock to prevent race with concurrent startSession
    await this.withSessionLock(sessionId, async () => {
      this.processes.delete(sessionId)
      this.failCounts.delete(sessionId)
      this.logBuffers.delete(sessionId)
      db.deleteSession(sessionId)
      this.emit('session:deleted', { sessionId })
    })
  }

  /** Config keys that require a session restart to take effect (all CLI args). */
  private static readonly RESTART_REQUIRED_KEYS = new Set([
    'port', 'host', 'modelPath', 'continuousBatching', 'enablePrefixCache',
    'usePagedCache', 'pagedCacheBlockSize', 'maxCacheBlocks',
    'noMemoryAwareCache', 'cacheMemoryMb', 'cacheMemoryPercent',
    'kvCacheQuantization', 'kvCacheGroupSize',
    'enableDiskCache', 'diskCacheMaxGb', 'diskCacheDir',
    'enableBlockDiskCache', 'blockDiskCacheMaxGb', 'blockDiskCacheDir',
    'prefixCacheSize', 'cacheTtlMinutes', 'isMultimodal',
    'toolCallParser', 'reasoningParser',
    'maxNumSeqs', 'prefillBatchSize', 'completionBatchSize',
    'timeout', 'streamInterval', 'apiKey', 'rateLimit',
    'maxTokens', 'mcpConfig', 'servedModelName',
    'speculativeModel', 'numDraftTokens',
    'defaultTemperature', 'defaultTopP',
    'embeddingModel', 'additionalArgs', 'mfluxClass',
    'enableAutoToolChoice',
    'logLevel', 'corsOrigins',
    'enableJit',
    'imageMode', 'imageQuantize',
    'streamFromDisk', 'streamMemoryPercent', 'ssdMemoryBudget', 'ssdPrefetchLayers',
  ])

  async updateSessionConfig(sessionId: string, config: Partial<ServerConfig>): Promise<{ restartRequired: boolean; changedKeys: string[] }> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    // Validate port if provided
    if (config.port !== undefined) {
      if (config.port < 1024 || config.port > 65535) {
        throw new Error(`Invalid port ${config.port}. Must be between 1024 and 65535.`)
      }
      // Check for port conflicts with other LOCAL sessions (remote sessions don't bind ports).
      // Only block if another session is actually running or loading on that port.
      const allSessions = db.getSessions()
      const conflicting = allSessions.find(s =>
        s.port === config.port &&
        s.id !== sessionId &&
        s.type === 'local' &&
        (s.status === 'running' || s.status === 'loading' || s.status === 'standby')
      )
      if (conflicting) {
        throw new Error(`Port ${config.port} is in use by running session "${conflicting.modelName || conflicting.modelPath}".`)
      }
    }

    let currentConfig: Record<string, unknown> = {}
    try {
      currentConfig = JSON.parse(session.config)
    } catch {
      // Corrupted config in DB — start fresh
    }
    // Strip undefined values before merging — prevents config spread from
    // overwriting existing DB values with undefined (which JSON.stringify would then drop)
    const cleanConfig: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(config as Record<string, unknown>)) {
      if (v !== undefined) cleanConfig[k] = v
    }
    const merged = { ...currentConfig, ...cleanConfig }

    // Log sleep config changes
    if ('idleTimeoutSoftMin' in cleanConfig || 'idleTimeoutHardMin' in cleanConfig || 'autoSleepEnabled' in cleanConfig) {
      console.log(`[SLEEP] Config saved for ${sessionId.slice(0, 8)}: soft=${merged.idleTimeoutSoftMin}min, hard=${merged.idleTimeoutHardMin}min, enabled=${merged.autoSleepEnabled}`)
    }

    // Always keep host/port in sync between JSON blob and DB columns
    // Extract from merged to ensure JSON blob and DB columns always agree
    const host = (merged.host as string) || session.host
    const port = (merged.port as number) || session.port

    db.updateSession(sessionId, {
      config: JSON.stringify(merged),
      host,
      port
    })

    // H6: Determine if changed keys require a restart
    const isRunning = session.status === 'running' || session.status === 'loading'
    const changedKeys = Object.keys(cleanConfig).filter(k =>
      SessionManager.RESTART_REQUIRED_KEYS.has(k) &&
      (cleanConfig as Record<string, unknown>)[k] !== currentConfig[k]
    )
    return {
      restartRequired: isRunning && changedKeys.length > 0,
      changedKeys,
    }
  }

  // ─── Discovery & Adoption ─────────────────────────────────────────

  async detectAndAdoptAll(): Promise<Session[]> {
    const processes = await this.detect()
    const adopted: Session[] = []

    for (const proc of processes) {
      if (!proc.healthy) continue

      // Normalize detected path for consistent DB matching
      proc.modelPath = normalizePath(proc.modelPath)

      // Check if we already have a session for this model path
      let session = db.getSessionByModelPath(proc.modelPath)

      // Determine correct session status from health response
      const adoptStatus = proc.standbyDepth ? 'standby' : 'running'
      const adoptStandbyDepth = proc.standbyDepth || null

      if (!session) {
        // Create a new session record for this detected process
        // Use full defaults so the settings page shows complete config
        const id = uuidv4()
        const now = Date.now()
        // Auto-detect model config for proper defaults (paged cache, parsers, etc.)
        const detected = detectModelConfigFromDir(proc.modelPath)
        const defaultConfig: ServerConfig = {
          modelPath: proc.modelPath,
          host: '0.0.0.0',
          port: proc.port,
          timeout: 300,
          maxNumSeqs: 256,
          prefillBatchSize: 0,
          completionBatchSize: 0,
          // VLM/MLLM models support continuous batching via MLLMScheduler
          continuousBatching: true,
          enablePrefixCache: true,
          prefixCacheSize: 100,
          cacheMemoryMb: 0,
          cacheMemoryPercent: 30,
          noMemoryAwareCache: false,
          usePagedCache: detected.usePagedCache ?? true,
          pagedCacheBlockSize: 64,
          maxCacheBlocks: 1000,
          streamInterval: 1,
          maxTokens: 32768,
          toolCallParser: 'auto',
          reasoningParser: 'auto',
          enableAutoToolChoice: detected.enableAutoToolChoice
        }
        session = {
          id,
          modelPath: proc.modelPath,
          modelName: proc.modelName || proc.modelPath.split('/').pop() || proc.modelPath,
          host: '0.0.0.0',
          port: proc.port,
          pid: proc.pid,
          status: adoptStatus,
          config: JSON.stringify(defaultConfig),
          createdAt: now,
          updatedAt: now,
          lastStartedAt: now,
          type: 'local',
          standbyDepth: adoptStandbyDepth
        }
        db.createSession(session)
      } else {
        // Update existing session with live process info (also normalize stored path)
        db.updateSession(session.id, {
          status: adoptStatus,
          standbyDepth: adoptStandbyDepth,
          pid: proc.pid,
          port: proc.port,
          modelPath: normalizePath(session.modelPath),
          modelName: proc.modelName || session.modelName,
          lastStartedAt: Date.now()
        })
        session = db.getSession(session.id)!
      }

      this.processes.set(session.id, { process: null, adoptedPid: proc.pid })
      adopted.push(session)
    }

    // Mark sessions that were running but no longer have a process
    const allSessions = db.getSessions()
    for (const s of allSessions) {
      if (s.status === 'running' || s.status === 'loading' || s.status === 'standby') {
        if (s.type === 'remote') {
          console.log(`[SESSIONS] Resetting stale remote session "${s.modelName}" to stopped (was ${s.status})`)
          db.updateSession(s.id, { status: 'stopped', standbyDepth: null })
          this.emit('session:stopped', { sessionId: s.id })
        } else if (!adopted.find(a => a.id === s.id)) {
          db.updateSession(s.id, { status: 'stopped', pid: undefined, standbyDepth: null })
        }
      }
    }

    return adopted
  }

  // ─── Global Health Monitor ─────────────────────────────────────────

  startGlobalMonitor(): void {
    if (this.monitorInterval) return

    this.monitorInterval = setInterval(async () => {
      const sessions = db.getSessions()

      for (const session of sessions) {
        // Skip stopped/error sessions. Standby sessions are monitored for health but not fail-counted.
        if (session.status === 'stopped' || session.status === 'error') continue

        // Standby sessions: just check process is alive, don't fail-count
        if (session.status === 'standby') {
          if (session.type !== 'remote' && session.pid) {
            const alive = this.isProcessAlive(session.id, session.pid)
            if (!alive) {
              db.updateSession(session.id, { status: 'stopped', standbyDepth: null })
              this.emit('session:stopped', { sessionId: session.id })
              this.pushLog(session.id, '[Sleep] Process died during standby')
              continue
            }
            // Check if model was woken externally (e.g., external curl triggered JIT wake)
            try {
              const res = await fetch(
                `http://${connectHost(session.host)}:${session.port}/health`,
                { signal: AbortSignal.timeout(3000) }
              )
              if (res.ok) {
                const data = await res.json()
                if (data.status === 'healthy') {
                  // Model woke externally — sync DB to running
                  db.updateSession(session.id, { status: 'running', standbyDepth: null })
                  this.touchSession(session.id)
                  this.emit('session:ready', { sessionId: session.id, port: session.port })
                  this.pushLog(session.id, '[Wake] Model woke externally — synced to running')
                }
              }
            } catch {
              // Health check failed — process alive but server unresponsive, keep standby
            }
          }
          continue
        }

        if (session.status !== 'running' && session.status !== 'loading') continue

        // Remote sessions: check /v1/models instead of /health
        if (session.type === 'remote') {
          if (!session.remoteUrl) {
            db.updateSession(session.id, { status: 'error' })
            this.emit('session:error', { sessionId: session.id, error: 'Missing remote URL' })
            continue
          }
          try {
            const remoteBase = session.remoteUrl.replace(/\/+$/, '')
            const remoteHeaders: Record<string, string> = {}
            if (session.remoteApiKey) remoteHeaders['Authorization'] = `Bearer ${session.remoteApiKey}`
            if (session.remoteOrganization) remoteHeaders['OpenAI-Organization'] = session.remoteOrganization
            const resolvedHealthUrl = await resolveUrl(`${remoteBase}/v1/models`)
            const pingStart = Date.now()
            const res = await fetch(resolvedHealthUrl, {
              headers: remoteHeaders,
              signal: AbortSignal.timeout(10000)
            })
            const latencyMs = Date.now() - pingStart
            if (res.ok) {
              this.failCounts.delete(session.id)
              this.lastHealthyAt.set(session.id, Date.now())
              if (session.status === 'loading') {
                db.updateSession(session.id, { status: 'running' })
                this.emit('session:ready', { sessionId: session.id, port: session.port })
              }
              this.emit('session:health', {
                sessionId: session.id,
                running: true,
                modelName: session.remoteModel,
                port: session.port,
                latencyMs
              })
            } else {
              this.incrementFailAndCheck(session.id)
            }
          } catch (_) {
            // Remote server unresponsive — likely busy with inference.
            // Use dampened counting (every 3rd failure) like local sessions,
            // since remote servers have no PID to check liveness.
            this.emit('session:health', {
              sessionId: session.id,
              running: true,
              busy: true,
              modelName: session.remoteModel,
              port: session.port
            })
            const count = this.failCounts.get(session.id) || 0
            if (count % 3 === 0) {
              this.incrementFailAndCheck(session.id)
            } else {
              this.failCounts.set(session.id, count + 1)
            }
          }
          continue
        }

        try {
          const res = await fetch(
            `http://${connectHost(session.host)}:${session.port}/health`,
            { signal: AbortSignal.timeout(10000) }
          )
          if (res.ok) {
            const data = await res.json()
            // Handle standby states from server
            const isStandby = data.status?.startsWith('standby_')
            // Only count as truly healthy if the model is loaded (status: "healthy")
            // The server returns "no_model" while still loading in lifespan()
            const modelReady = data.status === 'healthy'
            if (isStandby) {
              // Server is in standby — keep session alive, don't fail-count
              this.failCounts.delete(session.id)
            } else if (modelReady) {
              // Reset fail counter on success
              this.failCounts.delete(session.id)
              this.lastHealthyAt.set(session.id, Date.now())
              if (data.model_name && data.model_name !== session.modelName) {
                db.updateSession(session.id, { modelName: data.model_name })
              }
              if (session.status === 'loading') {
                db.updateSession(session.id, { status: 'running', standbyDepth: null })
                this.touchSession(session.id)
                this.emit('session:ready', { sessionId: session.id, port: session.port })
              }
              // Sync server-side last_request_time to idle timer — catches direct API
              // requests (curl, benchmarks, external tools) that bypass Electron IPC
              if (data.last_request_time) {
                const serverLastReq = Math.round(data.last_request_time * 1000) // Python epoch → JS epoch
                const electronLastReq = this.lastRequestAt.get(session.id) || 0
                if (serverLastReq > electronLastReq) {
                  this.lastRequestAt.set(session.id, serverLastReq)
                  db.updateSession(session.id, { lastRequestAt: serverLastReq })
                }
              }
            } else if (isStandby && session.status === 'loading') {
              // Wake failed — server reverted to standby but DB says loading.
              // Sync DB back to standby so user can retry.
              const depth = data.status === 'standby_deep' ? 'deep' : 'soft'
              db.updateSession(session.id, { status: 'standby', standbyDepth: depth })
              this.emit('session:standby', { sessionId: session.id, depth })
              this.pushLog(session.id, `[Wake] Model reload failed — reverted to ${depth} sleep`)
            } else {
              // Server is up but model not loaded yet — keep as loading
            }
            this.emit('session:health', {
              sessionId: session.id,
              running: modelReady,
              status: modelReady ? 'ok' : 'loading',
              modelName: data.model_name,
              port: session.port,
              memory: data.memory  // { active_mb, peak_mb, cache_mb } from /health
            })
          } else {
            this.incrementFailAndCheck(session.id)
          }
        } catch (_) {
          // Health check timed out or failed — check if process is still alive
          // Long prefills block the event loop, so the server can't respond
          // but the process is still running fine
          if (this.isProcessAlive(session.id, session.pid)) {
            // Process alive but unresponsive (likely busy with long prefill)
            // Emit a "busy" health event so the UI knows the server isn't dead
            this.emit('session:health', {
              sessionId: session.id,
              running: true,
              busy: true,
              modelName: session.modelName,
              port: session.port
            })
            // Only count every 3rd failure to avoid false positives
            const count = this.failCounts.get(session.id) || 0
            if (count % 3 === 0) {
              this.incrementFailAndCheck(session.id)
            } else {
              this.failCounts.set(session.id, count + 1)
            }
          } else {
            // Process is truly dead — fast-track to marking down
            this.incrementFailAndCheck(session.id)
          }
        }
      }

      // Check for idle sessions that should enter sleep
      await this.checkIdleSessions()
    }, 5000)
  }

  stopGlobalMonitor(): void {
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval)
      this.monitorInterval = null
    }
  }

  // ── Idle / Sleep Management ──

  /** Mark a session as having received a request (resets idle timer) */
  touchSession(sessionId: string): void {
    const now = Date.now()
    const prev = this.lastRequestAt.get(sessionId) || 0
    this.lastRequestAt.set(sessionId, now)
    db.updateSession(sessionId, { lastRequestAt: now })
    // Log touch events to help debug idle timer issues (only if previous touch was >10s ago)
    if (now - prev > 10000) {
      console.log(`[SLEEP] touchSession ${sessionId.slice(0, 8)} — idle timer reset`)
    }
  }

  /** Get idle timeouts for a session based on its model type */
  private getIdleTimeouts(session: import('./database').Session): { softMs: number; hardMs: number } {
    // Determine if this is an image session + read per-session overrides in one parse
    let isImage = false
    let perSessionSoft: number | undefined
    let perSessionHard: number | undefined
    try {
      const cfg = JSON.parse(session.config)
      isImage = cfg.modelType === 'image'
      // Accept each timeout independently (don't require BOTH to be set)
      if (typeof cfg.idleTimeoutSoftMin === 'number') perSessionSoft = cfg.idleTimeoutSoftMin
      if (typeof cfg.idleTimeoutHardMin === 'number') perSessionHard = cfg.idleTimeoutHardMin
    } catch {}

    // Defaults based on model type
    const defaultSoftMs = isImage ? SessionManager.DEFAULT_SOFT_TIMEOUT_IMAGE_MS : SessionManager.DEFAULT_SOFT_TIMEOUT_TEXT_MS
    const defaultHardMs = isImage ? SessionManager.DEFAULT_HARD_TIMEOUT_IMAGE_MS : SessionManager.DEFAULT_HARD_TIMEOUT_TEXT_MS

    // Check global settings
    const globalSoftStr = db.getSetting('idle_timeout_soft_min')
    const globalHardStr = db.getSetting('idle_timeout_hard_min')

    // Priority: per-session > global > model-type default (each timeout resolved independently)
    const softMs = perSessionSoft != null ? perSessionSoft * 60 * 1000
      : globalSoftStr ? parseInt(globalSoftStr) * 60 * 1000
      : defaultSoftMs
    const hardMs = perSessionHard != null ? perSessionHard * 60 * 1000
      : globalHardStr ? parseInt(globalHardStr) * 60 * 1000
      : defaultHardMs

    return { softMs, hardMs }
  }

  /** Check if auto-sleep is enabled (global setting, default true) */
  private isAutoSleepEnabled(): boolean {
    const setting = db.getSetting('auto_sleep_enabled')
    return setting !== '0' && setting !== 'false'
  }

  /** Trigger soft sleep on a session — clear caches, model stays loaded */
  async softSleep(sessionId: string): Promise<{ success: boolean; error?: string }> {
    const session = db.getSession(sessionId)
    if (!session || session.status !== 'running') {
      return { success: false, error: 'Session not running' }
    }
    if (session.type === 'remote') {
      return { success: false, error: 'Cannot sleep remote sessions' }
    }

    try {
      const host = connectHost(session.host)
      const res = await fetch(`http://${host}:${session.port}/admin/soft-sleep`, {
        method: 'POST',
        signal: AbortSignal.timeout(10000)
      })
      if (res.ok) {
        db.updateSession(sessionId, { status: 'standby', standbyDepth: 'soft' })
        this.emit('session:standby', { sessionId, depth: 'soft' })
        this.pushLog(sessionId, '[Sleep] Entered soft sleep — caches cleared, model loaded')
        return { success: true }
      }
      return { success: false, error: `Server returned ${res.status}` }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  }

  /** Trigger deep sleep on a session — unload model, process stays alive */
  async deepSleep(sessionId: string): Promise<{ success: boolean; error?: string }> {
    const session = db.getSession(sessionId)
    if (!session || (session.status !== 'running' && session.status !== 'standby')) {
      return { success: false, error: 'Session not running or standby' }
    }
    if (session.type === 'remote') {
      return { success: false, error: 'Cannot sleep remote sessions' }
    }

    try {
      const host = connectHost(session.host)
      const res = await fetch(`http://${host}:${session.port}/admin/deep-sleep`, {
        method: 'POST',
        signal: AbortSignal.timeout(10000)
      })
      if (res.ok) {
        db.updateSession(sessionId, { status: 'standby', standbyDepth: 'deep' })
        this.emit('session:standby', { sessionId, depth: 'deep' })
        this.pushLog(sessionId, '[Sleep] Entered deep sleep — model unloaded, port alive')
        return { success: true }
      }
      return { success: false, error: `Server returned ${res.status}` }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  }

  /** Wake a session from any sleep state — reload model */
  async wakeSession(sessionId: string): Promise<{ success: boolean; error?: string }> {
    const session = db.getSession(sessionId)
    if (!session || session.status !== 'standby') {
      return { success: false, error: 'Session not in standby' }
    }
    if (session.type === 'remote') {
      return { success: false, error: 'Cannot wake remote sessions' }
    }

    try {
      const host = connectHost(session.host)
      // 300s timeout — admin/wake does synchronous model load. Disk-streaming mode
      // with large models (60GB+ JANG MoE) can take >120s for mmap initialization.
      // Matches the JIT wake timeout on the Python side.
      const res = await fetch(`http://${host}:${session.port}/admin/wake`, {
        method: 'POST',
        signal: AbortSignal.timeout(300000)
      })
      if (res.ok) {
        db.updateSession(sessionId, { status: 'loading', standbyDepth: null })
        this.emit('session:starting', { sessionId })
        this.pushLog(sessionId, '[Wake] Waking from sleep — reloading model...')
        // The global monitor will pick up the 'loading' status and wait for /health
        this.touchSession(sessionId)
        return { success: true }
      }
      return { success: false, error: `Server returned ${res.status}` }
    } catch (e) {
      return { success: false, error: (e as Error).message }
    }
  }

  /** Check idle sessions and trigger sleep transitions (called from monitor) */
  private async checkIdleSessions(): Promise<void> {
    if (!this.isAutoSleepEnabled()) return

    const sessions = db.getSessions()
    const now = Date.now()

    for (const session of sessions) {
      if (session.type === 'remote') continue
      if (session.status !== 'running' && session.status !== 'standby') continue

      // Check per-session autoSleepEnabled override
      let autoSleepDisabled = false
      try {
        const cfg = JSON.parse(session.config)
        if (cfg.autoSleepEnabled === false) autoSleepDisabled = true
      } catch {}
      if (autoSleepDisabled) continue

      const mapTs = this.lastRequestAt.get(session.id)
      const lastReq = mapTs || session.lastRequestAt || session.lastStartedAt || 0
      if (!lastReq) continue

      const idleMs = now - lastReq
      const { softMs, hardMs } = this.getIdleTimeouts(session)

      // Skip if timeouts are 0 (disabled)
      if (softMs <= 0 && hardMs <= 0) continue

      if (session.status === 'running' && softMs > 0 && idleMs >= softMs) {
        // Running and idle past soft timeout → soft sleep
        console.log(`[SLEEP] Session ${session.id.slice(0, 8)} idle ${Math.round(idleMs / 1000)}s >= soft ${Math.round(softMs / 1000)}s → soft sleep`)
        this.pushLog(session.id, `[Sleep] Idle for ${Math.round(idleMs / 60000)}min — entering soft sleep (timeout: ${Math.round(softMs / 60000)}min)`)
        await this.softSleep(session.id)
      } else if (session.status === 'standby' && session.standbyDepth === 'soft' && hardMs > 0 && idleMs >= hardMs) {
        // In soft sleep and idle past hard timeout → deep sleep
        console.log(`[SLEEP] Session ${session.id.slice(0, 8)} idle ${Math.round(idleMs / 1000)}s >= hard ${Math.round(hardMs / 1000)}s → deep sleep`)
        this.pushLog(session.id, `[Sleep] Idle for ${Math.round(idleMs / 60000)}min — entering deep sleep (timeout: ${Math.round(hardMs / 60000)}min)`)
        await this.deepSleep(session.id)
      }
    }
  }

  private incrementFailAndCheck(sessionId: string): void {
    const count = (this.failCounts.get(sessionId) || 0) + 1
    this.failCounts.set(sessionId, count)

    const session = db.getSession(sessionId)

    // For local sessions: if process is dead, mark down immediately
    // For remote sessions: skip this check — they have no PID, so isProcessAlive
    // always returns false. Use the normal fail-count threshold instead.
    if (session?.type !== 'remote' && session && !this.isProcessAlive(sessionId, session.pid)) {
      console.log(`[SESSIONS] Process dead for session ${sessionId} (fail #${count}), marking down`)
      this.failCounts.delete(sessionId)
      this.handleSessionDown(sessionId)
      return
    }

    // Scale max failures with session timeout (5s interval → timeout/5, min 60)
    let maxFails = SessionManager.MAX_FAIL_COUNT
    if (session) {
      try {
        const cfg = JSON.parse(session.config)
        if (cfg.timeout && cfg.timeout > 0) {
          maxFails = Math.max(60, Math.ceil(cfg.timeout / 5))
        }
      } catch (_) { }
    }

    if (count >= maxFails) {
      console.log(`[SESSIONS] Health check failed ${count}x for session ${sessionId} (limit ${maxFails}), marking down`)
      this.failCounts.delete(sessionId)
      this.handleSessionDown(sessionId)
    } else if (count % 10 === 0) {
      // Only log every 10th failure to reduce noise (process is likely doing a long prefill)
      console.log(`[SESSIONS] Health check failed ${count}/${maxFails} for session ${sessionId} (process alive, likely busy)`)
    }
  }

  private handleSessionDown(sessionId: string): void {
    const session = db.getSession(sessionId)
    if (session && (session.status === 'running' || session.status === 'loading')) {
      if (session.type === 'remote') {
        // Remote endpoint truly unreachable after sustained failures — mark as error.
        // Unlike local sessions, there's no process to kill. The user needs to know
        // the endpoint is down so they can fix it or restart.
        this.pushLog(sessionId, '[ERROR] Remote endpoint unreachable after sustained failures')
        console.log(`[SESSIONS] handleSessionDown: remote session ${sessionId} ("${session.modelName}") unreachable, marking error`)
        db.updateSession(sessionId, { status: 'error' })
        this.failCounts.delete(sessionId)
        this.emit('session:error', { sessionId, error: 'Remote endpoint unreachable' })
        return
      } else {
        // Kill the process before marking stopped — without this, the Python
        // process continues running as an orphan consuming RAM/CPU.
        const managed = this.processes.get(sessionId)
        const pid = managed?.adoptedPid ?? managed?.process?.pid ?? session.pid
        if (pid) {
          console.log(`[SESSIONS] handleSessionDown: killing PID ${pid} for session ${sessionId}`)
          this.killPid(pid, 'SIGTERM')
          // Schedule SIGKILL escalation after 3s (non-blocking)
          setTimeout(() => {
            try {
              process.kill(pid, 0) // Check if still alive
              console.log(`[SESSIONS] handleSessionDown: escalating to SIGKILL for PID ${pid}`)
              this.killPid(pid, 'SIGKILL')
            } catch (_) {
              // Already dead — good
            }
          }, 3000)
        } else if (session.port) {
          // Fallback: kill by port
          this.killByPort(session.port).catch(() => { })
        }
        this.processes.delete(sessionId)
      }
      this.failCounts.delete(sessionId)
      // Abort any in-flight SSE streams before marking down
      if (session.host && session.port) {
        this.emit('session:abortInference', { sessionId, host: session.host, port: session.port })
      }
      db.updateSession(sessionId, {
        status: 'error',
        pid: undefined,
        lastStoppedAt: Date.now()
      })
      this.emit('session:error', { sessionId, error: 'Session became unresponsive' })
    }
  }

  // ─── Stop All ──────────────────────────────────────────────────────

  async stopAll(): Promise<void> {
    this.stopGlobalMonitor()

    const processes = await this.detect()

    // B3: Send SIGTERM first to all processes for graceful shutdown
    for (const proc of processes) {
      this.killPid(proc.pid, 'SIGTERM')
    }
    for (const [, managed] of this.processes) {
      if (managed.process) {
        try { managed.process.kill('SIGTERM') } catch (_) { }
      } else if (managed.adoptedPid) {
        this.killPid(managed.adoptedPid, 'SIGTERM')
      }
    }

    // Wait for graceful shutdown, then SIGKILL any survivors
    if (processes.length > 0 || this.processes.size > 0) {
      await new Promise(r => setTimeout(r, 3000))
      for (const proc of processes) {
        try { process.kill(proc.pid, 'SIGKILL') } catch (_) { }
      }
      for (const [, managed] of this.processes) {
        if (managed.process) {
          try { managed.process.kill('SIGKILL') } catch (_) { }
        } else if (managed.adoptedPid) {
          try { process.kill(managed.adoptedPid, 'SIGKILL') } catch (_) { }
        }
      }
    }

    this.processes.clear()

    // Mark all sessions as stopped in DB (including standby — their processes were killed above)
    const sessions = db.getSessions()
    for (const s of sessions) {
      if (s.status === 'running' || s.status === 'loading' || s.status === 'standby') {
        db.updateSession(s.id, { status: 'stopped', pid: undefined, lastStoppedAt: Date.now(), standbyDepth: null })
      }
    }
  }

  // ─── Queries ───────────────────────────────────────────────────────

  getSessions(): Session[] {
    return db.getSessions()
  }

  getSession(id: string): Session | undefined {
    return db.getSession(id)
  }

  getSessionByModelPath(modelPath: string): Session | undefined {
    return db.getSessionByModelPath(normalizePath(modelPath))
  }

  // ─── Helpers (from ServerManager) ──────────────────────────────────

  buildArgs(config: ServerConfig): string[] {
    const args = ['serve', config.modelPath]
    const isImage = config.modelType === 'image'

    // Server settings — always pass explicitly (both text and image)
    args.push('--host', config.host)
    args.push('--port', config.port.toString())
    args.push('--timeout', (config.timeout != null && config.timeout > 0 ? config.timeout : 86400).toString())

    if (config.rateLimit && config.rateLimit > 0) args.push('--rate-limit', config.rateLimit.toString())
    // API key passed via VLLM_API_KEY env var in spawn (not CLI arg) to avoid exposure in ps aux

    // Image models: skip all text-specific flags (parsers, batching, cache, etc.)
    // The Python server auto-detects image vs text from the model directory
    if (isImage) {
      // Image-specific settings (explicit flags, not via additionalArgs)
      if (config.imageMode === 'edit') args.push('--image-mode', 'edit')
      if (config.imageQuantize && config.imageQuantize > 0) args.push('--image-quantize', config.imageQuantize.toString())
      if (config.servedModelName) args.push('--served-model-name', config.servedModelName)
      if (config.mfluxClass) args.push('--mflux-class', config.mfluxClass)
      // Logging + CORS still apply to image servers
      if (config.logLevel && config.logLevel !== 'INFO') args.push('--log-level', config.logLevel)
      if (config.corsOrigins && config.corsOrigins !== '*') args.push('--allowed-origins', config.corsOrigins)
      // Strip image-specific flags from additionalArgs to prevent duplication
      // (stale additionalArgs may survive config merge from a previous session)
      if (config.additionalArgs?.trim()) {
        const imageFlags = new Set(['--image-mode', '--image-quantize', '--served-model-name', '--mflux-class'])
        const extra = config.additionalArgs.trim().split(/\s+/).filter(Boolean)
        const filtered: string[] = []
        for (let i = 0; i < extra.length; i++) {
          if (imageFlags.has(extra[i])) {
            i++ // skip the flag's value argument too
          } else {
            filtered.push(extra[i])
          }
        }
        if (filtered.length) args.push(...filtered)
      }
      return args
    }

    // === Text model flags below ===

    // Concurrent processing
    // When value is 0 ("No limit" in UI), omit the flag so backend uses its default.
    // When value > 0, pass it explicitly to override the backend default.
    if (config.maxNumSeqs && config.maxNumSeqs > 0) {
      args.push('--max-num-seqs', config.maxNumSeqs.toString())
    }
    if (config.prefillBatchSize && config.prefillBatchSize > 0) {
      args.push('--prefill-batch-size', config.prefillBatchSize.toString())
    }
    if (config.completionBatchSize && config.completionBatchSize > 0) {
      args.push('--completion-batch-size', config.completionBatchSize.toString())
    }

    // Auto-detect tool/reasoning parser from model's config.json model_type field.
    // No name-based regex detection — config.json is authoritative.
    const detected = detectModelConfigFromDir(config.modelPath)

    // VLM detection: tri-state — undefined=auto, true=force on, false=force off.
    // Only respect explicit user choice (true/false); undefined defers to auto-detect.
    const isVLM = config.isMultimodal === true ? true
      : config.isMultimodal === false ? false
        : !!detected.isMultimodal
    if (isVLM) args.push('--is-mllm')

    if (config.continuousBatching) args.push('--continuous-batching')

    // Parser resolution: User explicit choice -> Detected config -> Fallback logic
    // Empty string "" = user explicitly chose "None" (disabled) — always respected.
    const userToolParser = config.toolCallParser
    const effectiveToolParser = userToolParser === ''
      ? undefined                     // User explicitly chose "None"
      : (userToolParser && userToolParser !== 'auto' ? userToolParser
        : detected.toolParser)       // Fallback to detection if auto or missing

    const effectiveAutoTool = config.enableAutoToolChoice ?? detected.enableAutoToolChoice

    const userReasoningParser = config.reasoningParser
    const effectiveReasoningParser = userReasoningParser === ''
      ? undefined                     // User explicitly chose "None"
      : (userReasoningParser && userReasoningParser !== 'auto' ? userReasoningParser
        : detected.reasoningParser)  // Fallback to detection if auto or missing

    // Pass resolved parsers directly to the CLI so backend doesn't guess.
    // When a tool parser is set, --enable-auto-tool-choice is required by the engine
    // (cli.py gates on both flags). Enable it unless user explicitly disabled auto-tool-choice.
    if (effectiveToolParser) {
      args.push('--tool-call-parser', effectiveToolParser)
      // Ensure --enable-auto-tool-choice is set when a parser is present
      if (effectiveAutoTool || config.enableAutoToolChoice === undefined) {
        args.push('--enable-auto-tool-choice')
      }
    } else if (effectiveAutoTool) {
      args.push('--enable-auto-tool-choice')
    }
    if (effectiveReasoningParser) {
      args.push('--reasoning-parser', effectiveReasoningParser)
    }
    // Pass custom served model name if configured
    if (config.servedModelName) {
      args.push('--served-model-name', config.servedModelName)
    }

    console.log(`[SESSION] Model family: ${detected.family} | tool: ${effectiveToolParser || 'none'} (user=${userToolParser}, detected=${detected.toolParser || 'none'}) | reasoning: ${effectiveReasoningParser || 'none'} (user=${userReasoningParser}, detected=${detected.reasoningParser || 'none'}) | autoTool: ${effectiveAutoTool} | VLM: ${isVLM}`)

    // SSD disk-streaming mode — per-layer weight recycling from SSD
    if (config.streamFromDisk) {
      args.push('--stream-from-disk')
      if (config.streamMemoryPercent != null && config.streamMemoryPercent !== 90) {
        args.push('--stream-memory-percent', config.streamMemoryPercent.toString())
      }
    }

    // Prefix cache — requires --continuous-batching to take effect in vmlx-engine
    // When MCP tools + auto-tool-choice are enabled, force prefix cache ON.
    // Tool follow-up requests share most of the prompt with the original request;
    // without prefix cache each follow-up re-processes the entire prompt (~16s).
    const toolsNeedCache = !!(effectiveAutoTool && config.mcpConfig)
    const prefixCacheOff = config.enablePrefixCache === false && !toolsNeedCache

    if (prefixCacheOff) {
      args.push('--disable-prefix-cache')
    } else {
      // Auto-enable continuous batching when prefix cache is on (required by vmlx-engine).
      if (!config.continuousBatching && !args.includes('--continuous-batching')) {
        args.push('--continuous-batching')
      }
      if (config.noMemoryAwareCache) {
        args.push('--no-memory-aware-cache')
        if (config.prefixCacheSize && config.prefixCacheSize > 0) {
          args.push('--prefix-cache-size', config.prefixCacheSize.toString())
        }
      } else {
        if (config.cacheMemoryMb && config.cacheMemoryMb > 0) {
          args.push('--cache-memory-mb', config.cacheMemoryMb.toString())
        }
        if (config.cacheMemoryPercent && config.cacheMemoryPercent > 0) {
          args.push('--cache-memory-percent', (config.cacheMemoryPercent / 100).toString())
        }
        // Cache TTL (time-to-live for cache entries) — only meaningful for memory-aware cache, not paged cache
        if (config.cacheTtlMinutes && config.cacheTtlMinutes > 0 && !(config.usePagedCache ?? detected.usePagedCache)) {
          args.push('--cache-ttl-minutes', config.cacheTtlMinutes.toString())
        }
      }
    }

    // Paged cache is a prefix cache backend — works for both LLMs and VLMs
    if (!prefixCacheOff && (config.usePagedCache ?? detected.usePagedCache)) {
      args.push('--use-paged-cache')
      if (config.pagedCacheBlockSize && config.pagedCacheBlockSize > 0) {
        args.push('--paged-cache-block-size', config.pagedCacheBlockSize.toString())
      }
      if (config.maxCacheBlocks && config.maxCacheBlocks > 0) {
        args.push('--max-cache-blocks', config.maxCacheBlocks.toString())
      }
    }

    // KV cache quantization — works for both LLMs and VLMs
    // The Python scheduler only quantizes KVCache layers, non-KV layers pass through.
    if (!prefixCacheOff && config.kvCacheQuantization && config.kvCacheQuantization !== 'none') {
      args.push('--kv-cache-quantization', config.kvCacheQuantization)
      if (config.kvCacheGroupSize && config.kvCacheGroupSize !== 64) {
        args.push('--kv-cache-group-size', config.kvCacheGroupSize.toString())
      }
    }

    // Disk cache (L2 persistent cache) — only meaningful with prefix cache, incompatible with paged cache
    if (!prefixCacheOff && config.enableDiskCache && !(config.usePagedCache ?? detected.usePagedCache)) {
      args.push('--enable-disk-cache')
      if (config.diskCacheDir) {
        args.push('--disk-cache-dir', config.diskCacheDir)
      }
      if (config.diskCacheMaxGb != null && config.diskCacheMaxGb >= 0) {
        args.push('--disk-cache-max-gb', config.diskCacheMaxGb.toString())
      }
    }

    // Block-level disk cache (L2 for paged cache blocks) — works for both LLMs and VLMs
    // Must mirror the paged cache guard condition above
    if (!prefixCacheOff && (config.usePagedCache ?? detected.usePagedCache) && config.enableBlockDiskCache) {
      args.push('--enable-block-disk-cache')
      if (config.blockDiskCacheDir) {
        args.push('--block-disk-cache-dir', config.blockDiskCacheDir)
      }
      if (config.blockDiskCacheMaxGb != null && config.blockDiskCacheMaxGb >= 0) {
        args.push('--block-disk-cache-max-gb', config.blockDiskCacheMaxGb.toString())
      }
    }

    // Performance
    if (config.streamInterval && config.streamInterval > 0) {
      args.push('--stream-interval', config.streamInterval.toString())
    }
    // maxTokens: 0 = "No limit" → pass a very large value so the model context window is the limit
    if (config.maxTokens && config.maxTokens > 0) {
      args.push('--max-tokens', config.maxTokens.toString())
    } else {
      args.push('--max-tokens', '1000000')
    }

    // Tool integration (parsers and --enable-auto-tool-choice already pushed above)
    if (config.mcpConfig) args.push('--mcp-config', config.mcpConfig)

    // Speculative decoding
    if (config.speculativeModel) {
      args.push('--speculative-model', config.speculativeModel)
      if (config.numDraftTokens && config.numDraftTokens !== 3) {
        args.push('--num-draft-tokens', config.numDraftTokens.toString())
      }
    }

    // Generation defaults (slider value is integer ×100, convert to float)
    // Slider uses 0 as "Server default" sentinel (unlimitedValue=0), so > 0 is correct.
    // The minimum real value on the slider is step=5 (temp=0.05).
    if (config.defaultTemperature != null && config.defaultTemperature > 0) {
      args.push('--default-temperature', (config.defaultTemperature / 100).toFixed(2))
    }
    if (config.defaultTopP != null && config.defaultTopP > 0) {
      args.push('--default-top-p', (config.defaultTopP / 100).toFixed(2))
    }

    // Embedding model
    if (config.embeddingModel) {
      args.push('--embedding-model', config.embeddingModel)
    }

    // Server-level default for enable_thinking (applies to all API clients)
    if (config.defaultEnableThinking === true) args.push('--default-enable-thinking', 'true')
    else if (config.defaultEnableThinking === false) args.push('--default-enable-thinking', 'false')

    // JIT compilation
    if (config.enableJit) args.push('--enable-jit')

    // Logging
    if (config.logLevel && config.logLevel !== 'INFO') {
      args.push('--log-level', config.logLevel)
    }

    // CORS
    if (config.corsOrigins && config.corsOrigins !== '*') {
      args.push('--allowed-origins', config.corsOrigins)
    }

    // Additional arguments — strip stale image-only flags from old session configs
    if (config.additionalArgs?.trim()) {
      const staleImageFlags = new Set(['--mflux-class', '--image-mode', '--image-quantize'])
      // Strip commas — common when users copy flag lists from docs (e.g. "--flag1, --flag2")
      const extra = config.additionalArgs.trim().replace(/,/g, ' ').split(/\s+/).filter(Boolean)
      const filtered: string[] = []
      for (let i = 0; i < extra.length; i++) {
        if (staleImageFlags.has(extra[i])) {
          i++ // skip the flag's value too
        } else {
          filtered.push(extra[i])
        }
      }
      if (filtered.length) args.push(...filtered)
    }

    return args
  }

  findEnginePath(): EnginePath | null {
    // Bundled Python: use python3 -m vmlx_engine.cli instead of vmlx-engine binary
    // This avoids shebang path issues in relocatable Python builds
    const bundledPython = getBundledPythonPath()
    if (bundledPython) {
      try {
        execSync(`"${bundledPython}" -s -c "import vmlx_engine"`, {
          encoding: 'utf-8',
          timeout: 10000,
          env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: '' },
        })
        return { type: 'bundled', pythonPath: bundledPython }
      } catch (_) {
        console.log('[SESSIONS] Bundled Python found but vmlx_engine import failed, trying system')
      }
    }

    // System binary search
    const home = homedir()
    const locations = [
      join(home, '.local', 'bin', 'vmlx-engine'),     // uv tool / pip --user
      '/opt/homebrew/bin/vmlx-engine',                  // Homebrew (Apple Silicon)
      '/usr/local/bin/vmlx-engine',                     // Homebrew (Intel) / system pip
      '/usr/bin/vmlx-engine',                           // System pip
      join(home, 'miniforge3', 'bin', 'vmlx-engine'),  // Miniforge
      join(home, 'anaconda3', 'bin', 'vmlx-engine'),   // Anaconda
      join(home, 'miniconda3', 'bin', 'vmlx-engine'),  // Miniconda
    ]

    // Scan pyenv versions (common on macOS)
    const pyenvRoot = join(home, '.pyenv', 'versions')
    try {
      if (existsSync(pyenvRoot)) {
        for (const ver of readdirSync(pyenvRoot)) {
          locations.push(join(pyenvRoot, ver, 'bin', 'vmlx-engine'))
        }
      }
    } catch (_) { }

    for (const loc of locations) {
      if (existsSync(loc)) return { type: 'system', binaryPath: loc }
    }

    // Fallback: check PATH via login shell (picks up pyenv, nvm, etc.)
    for (const shell of ['/bin/zsh', '/bin/bash']) {
      try {
        const result = execSync(
          `${shell} -lc "which vmlx-engine"`,
          { encoding: 'utf-8', timeout: 5000 }
        ).trim()
        if (result && existsSync(result)) return { type: 'system', binaryPath: result }
      } catch (_) { }
    }

    // Last resort: plain which
    try {
      const result = execSync('which vmlx-engine', { encoding: 'utf-8', timeout: 3000 }).trim()
      if (result && existsSync(result)) return { type: 'system', binaryPath: result }
    } catch (_) { }

    // Development fallback: project .venv relative to source directory
    try {
      const sourceDir = join(__dirname, '..', '..', '..')
      const venvPython = join(sourceDir, '.venv', 'bin', 'python3')
      if (existsSync(venvPython)) {
        try {
          execFileSync(venvPython, ['-s', '-c', 'import vmlx_engine'], {
            encoding: 'utf-8',
            timeout: 10000,
            env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: '' },
          })
          console.log(`[SESSIONS] Using project venv: ${venvPython}`)
          return { type: 'bundled', pythonPath: venvPython }
        } catch (_) { }
      }
    } catch (_) { }

    return null
  }

  private async findAvailablePort(): Promise<number> {
    const sessions = db.getSessions()
    // Check ALL session ports (DB has UNIQUE constraint on port column)
    const usedPorts = new Set(sessions.map(s => s.port))
    let port = 8000
    while (usedPorts.has(port) || !(await this.isPortFree(port))) {
      port++
      if (port > 65535) throw new Error('No available ports')
    }
    return port
  }

  private isPortFree(port: number): Promise<boolean> {
    return new Promise(resolve => {
      const server = createServer()
      server.once('error', () => resolve(false))
      server.once('listening', () => {
        server.close(() => setTimeout(() => resolve(true), 10))
      })
      server.listen(port, '127.0.0.1')
    })
  }

  /** Check if a session's process is still alive (not zombie) via PID probe. */
  private isProcessAlive(sessionId: string, dbPid?: number): boolean {
    // Check managed process first (spawned or adopted)
    const managed = this.processes.get(sessionId)
    const pid = managed?.adoptedPid ?? managed?.process?.pid ?? dbPid
    if (!pid) return false
    try {
      process.kill(pid, 0) // Signal 0: doesn't kill, just checks existence
    } catch (_) {
      return false
    }
    // M7: kill(pid, 0) succeeds for zombies. Check process state to filter them out.
    try {
      const state = execFileSync('ps', ['-o', 'state=', '-p', String(pid)],
        { timeout: 1000 }).toString().trim()
      if (state.startsWith('Z')) return false // Zombie process
    } catch (_) {
      // ps failed — process may have exited between checks
      return false
    }
    return true
  }

  private killPid(pid: number, signal: NodeJS.Signals = 'SIGTERM'): void {
    // Try process group kill first (negative PID kills entire group).
    // This ensures MCP subprocesses and uvicorn workers are also killed.
    // Falls back to single-PID kill if group kill fails (e.g., not a group leader).
    try { process.kill(-pid, signal) } catch (_) {
      try { process.kill(pid, signal) } catch (_) { }
    }
  }

  private async killByPort(port: number): Promise<void> {
    try {
      const output = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf-8', timeout: 5000 }).trim()
      if (output) {
        const pids = output.split('\n').map(s => parseInt(s)).filter(n => !isNaN(n))
        for (const pid of pids) this.killPid(pid)
        await new Promise(r => setTimeout(r, 1500))
        // Escalate to SIGKILL if processes still hold the port
        try {
          const check = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf-8', timeout: 3000 }).trim()
          if (check) {
            for (const pidStr of check.split('\n')) this.killPid(parseInt(pidStr), 'SIGKILL')
            await new Promise(r => setTimeout(r, 500))
          }
        } catch (_) { /* port freed */ }
      }
    } catch (_) { }
  }

  private async killChildProcess(proc: ChildProcess): Promise<void> {
    const pid = proc.pid
    return new Promise((resolve) => {
      // Escalate to SIGKILL after 10s if SIGTERM doesn't work
      const killTimeout = setTimeout(() => {
        // Kill entire process group (detached spawn)
        if (pid) { try { process.kill(-pid, 'SIGKILL') } catch (_) { } }
        try { proc.kill('SIGKILL') } catch (_) { }
      }, 10000)

      // B4: Final safety — resolve after 15s even if process never exits
      const hardTimeout = setTimeout(() => {
        clearTimeout(killTimeout)
        resolve()
      }, 15000)

      proc.once('exit', () => {
        clearTimeout(killTimeout)
        clearTimeout(hardTimeout)
        resolve()
      })

      // Send SIGTERM to process group first, then to the process directly
      if (pid) { try { process.kill(-pid, 'SIGTERM') } catch (_) { } }
      try { proc.kill('SIGTERM') } catch (_) {
        clearTimeout(killTimeout)
        clearTimeout(hardTimeout)
        resolve()
      }
    })
  }

  private async waitForReady(host: string, port: number, maxWait = 120000, sessionId?: string): Promise<void> {
    const startTime = Date.now()
    const healthUrl = `http://${connectHost(host)}:${port}/health`

    while (Date.now() - startTime < maxWait) {
      // Abort early if the process exited while we were waiting
      if (sessionId) {
        const managed = this.processes.get(sessionId)
        if (managed && !managed.process && !managed.adoptedPid) {
          // Process exited — include the reason if available
          let reason: string
          if (managed.exitSignal === 'SIGKILL') {
            reason = 'Process was killed (SIGKILL) — likely out of memory. Try a smaller/more quantized model, reduce cache size, or close other apps.'
          } else {
            reason = managed.lastStderr || `exit code ${managed.exitCode ?? 'unknown'}`
          }
          throw new Error(`Process exited before becoming ready: ${reason}`)
        }
        if (!managed) {
          throw new Error('Process exited before becoming ready')
        }
      }

      try {
        const response = await fetch(healthUrl, { signal: AbortSignal.timeout(1000) })
        if (response.ok) return
      } catch (_) { }
      await new Promise(resolve => setTimeout(resolve, 500))
    }

    throw new Error('Server failed to start within timeout period')
  }
}

export const sessionManager = new SessionManager()
