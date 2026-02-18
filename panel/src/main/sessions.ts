import { spawn, ChildProcess, execSync } from 'child_process'
import { lookup } from 'dns'
import { EventEmitter } from 'events'
import { existsSync, readdirSync, statSync } from 'fs'
import { createServer } from 'net'
import { homedir, totalmem, freemem } from 'os'
import { join } from 'path'
import { v4 as uuidv4 } from 'uuid'
import { db, Session } from './database'

export type { ServerConfig, DetectedProcess } from './server'
import type { ServerConfig, DetectedProcess } from './server'
import { detectModelConfigFromDir } from './model-config-registry'
import { getBundledPythonPath } from './vllm-manager'

/** Result of findVllmMlx: either bundled Python or a system binary */
type VllmMlxPath =
  | { type: 'bundled'; pythonPath: string }
  | { type: 'system'; binaryPath: string }

interface ManagedProcess {
  process: ChildProcess | null
  adoptedPid: number | null
  lastStderr?: string  // Last stderr line for error reporting
  exitCode?: number | null
}

/** Normalize model paths for consistent matching: resolve and strip trailing slashes */
function normalizePath(p: string): string {
  return p.replace(/\/+$/, '')
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
 */
export async function resolveUrl(url: string): Promise<string> {
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
      return resolved
    }
  } catch (e) {
    console.log(`[DNS] Failed to resolve ${url}:`, e)
  }
  return url
}

export class SessionManager extends EventEmitter {
  private processes = new Map<string, ManagedProcess>()
  private monitorInterval: ReturnType<typeof setInterval> | null = null
  private failCounts = new Map<string, number>()
  /** Per-session operation lock to prevent concurrent start/stop races */
  private operationLocks = new Map<string, Promise<void>>()
  // Allow up to 60 consecutive health check failures (5s * 60 = 5 min)
  // before marking session as down. Long prefill operations (e.g. 44k+
  // tokens) can block the server's event loop for 30+ seconds.
  private static readonly MAX_FAIL_COUNT = 60

  constructor() {
    super()
  }

  /**
   * Acquire a per-session operation lock. Serializes start/stop operations
   * for the same session to prevent race conditions (e.g. stop during start,
   * start during stop, rapid start/stop/start).
   */
  private async withSessionLock(sessionId: string, fn: () => Promise<void>): Promise<void> {
    // Wait for any pending operation on this session to finish
    const pending = this.operationLocks.get(sessionId)
    if (pending) {
      await pending.catch(() => {})
    }

    // Create a lock promise that resolves when our operation completes
    let unlock!: () => void
    const lock = new Promise<void>(r => { unlock = r })
    this.operationLocks.set(sessionId, lock)

    try {
      await fn()
    } finally {
      unlock()
      this.operationLocks.delete(sessionId)
    }
  }

  // ─── Process Detection (reused from ServerManager) ─────────────────

  async detect(): Promise<DetectedProcess[]> {
    const detected: DetectedProcess[] = []

    try {
      const output = execSync('ps aux', { encoding: 'utf-8', timeout: 5000 })
      const lines = output.split('\n')

      for (const line of lines) {
        if (line.includes('grep')) continue
        // Detect `vllm-mlx serve`, `python -m vllm_mlx.cli serve`, and `python -m vllm_mlx.server` processes
        const isCliServe = line.includes('vllm-mlx') && line.includes('serve')
        const isPythonModule = line.includes('vllm_mlx') && (line.includes('.cli') || line.includes('.server') || line.includes('--model'))
        if (!isCliServe && !isPythonModule) continue

        const parsed = this.parsePsLine(line)
        if (!parsed) continue

        let healthy = false
        let modelName: string | undefined
        try {
          const res = await fetch(
            `http://127.0.0.1:${parsed.port}/health`,
            { signal: AbortSignal.timeout(2000) }
          )
          if (res.ok) {
            const data = await res.json()
            healthy = true
            modelName = data.model_name
          }
        } catch (_) {}

        detected.push({
          pid: parsed.pid,
          port: parsed.port,
          modelPath: parsed.modelPath,
          healthy,
          modelName
        })
      }
    } catch (_) {}

    return detected
  }

  private parsePsLine(line: string): { pid: number; port: number; modelPath: string } | null {
    try {
      const parts = line.trim().split(/\s+/)
      const pid = parseInt(parts[1])
      if (isNaN(pid)) return null

      const cmdStart = parts.slice(10).join(' ')

      let modelPath = ''

      // Try `serve <model-path> --...` format first (vllm-mlx CLI)
      const serveIdx = cmdStart.indexOf('serve ')
      if (serveIdx !== -1) {
        const afterServe = cmdStart.substring(serveIdx + 6).trim()
        modelPath = afterServe.split(/\s+--/)[0].trim()
      }

      // Try `--model <path>` format (python -m vllm_mlx.server)
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
    // Normalize path to prevent trailing-slash mismatches
    modelPath = normalizePath(modelPath)

    // Check if session already exists for this model path
    const existing = db.getSessionByModelPath(modelPath)
    if (existing) {
      // Update config AND sync host/port columns
      const host = (config.host as string) || existing.host
      const port = (config.port as number) || existing.port
      db.updateSession(existing.id, {
        config: JSON.stringify({ ...config, modelPath, host, port }),
        host,
        port
      })
      return db.getSession(existing.id)!
    }

    const id = uuidv4()
    const host = config.host || '127.0.0.1'
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
    const port = parseInt(url.port) || (url.protocol === 'https:' ? 443 : 80)
    const now = Date.now()

    const session: Session = {
      id,
      modelPath,
      modelName: params.remoteModel,
      host,
      port,
      status: 'stopped',
      config: JSON.stringify({}),
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

    const vllmResult = this.findVllmMlx()
    if (!vllmResult) throw new Error('vllm-mlx not found. Please install it first.')
    if (!existsSync(config.modelPath)) throw new Error(`Model not found at: ${config.modelPath}`)

    // Validate model format: vllm-mlx only supports MLX (safetensors) models
    try {
      const files = readdirSync(config.modelPath)
      const hasGGUF = files.some(f => f.endsWith('.gguf') || f.endsWith('.gguf.part'))
      const hasSafetensors = files.some(f => f.endsWith('.safetensors'))
      const hasConfig = files.includes('config.json')

      if (hasGGUF && !hasSafetensors) {
        throw new Error(
          'This model is in GGUF format, which is not supported by vllm-mlx. ' +
          'Please download an MLX-format version (safetensors) from HuggingFace Hub.'
        )
      }
      if (!hasConfig) {
        throw new Error(
          'Model directory is missing config.json. This may not be a valid MLX model. ' +
          'vllm-mlx requires MLX-format models with config.json and .safetensors files.'
        )
      }
    } catch (e) {
      if ((e as Error).message.includes('GGUF format') || (e as Error).message.includes('config.json')) throw e
      // Ignore filesystem errors — let the server handle them
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

    let proc: ChildProcess
    if (vllmResult.type === 'bundled') {
      // Bundled Python: spawn python3 -m vllm_mlx.cli serve <model> --host ... --port ...
      // This avoids shebang path issues with relocatable Python
      const fullCmd = `${vllmResult.pythonPath} -m vllm_mlx.cli ${args.join(' ')}`
      this.emit('session:log', { sessionId, data: `$ ${fullCmd}\n` })
      proc = spawn(vllmResult.pythonPath, ['-m', 'vllm_mlx.cli', ...args], {
        env: spawnEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: true,  // Separate process group so we can kill entire group
      })
    } else {
      // System binary: spawn vllm-mlx directly
      const fullCmd = `${vllmResult.binaryPath} ${args.join(' ')}`
      this.emit('session:log', { sessionId, data: `$ ${fullCmd}\n` })
      proc = spawn(vllmResult.binaryPath, args, {
        env: spawnEnv,
        stdio: ['ignore', 'pipe', 'pipe'],
        detached: true,
      })
    }

    this.processes.set(sessionId, { process: proc, adoptedPid: null })

    proc.stdout?.on('data', (data) => {
      this.emit('session:log', { sessionId, data: data.toString() })
    })
    proc.stderr?.on('data', (data) => {
      const text = data.toString()
      this.emit('session:log', { sessionId, data: text })
      // Capture last meaningful stderr line for error reporting
      const managed = this.processes.get(sessionId)
      if (managed) {
        const lastLine = text.trim().split('\n').filter((l: string) => l.trim()).pop()
        if (lastLine) managed.lastStderr = lastLine
      }
    })
    proc.stdout?.on('error', () => {})
    proc.stderr?.on('error', () => {})

    proc.on('exit', (code, signal) => {
      const managed = this.processes.get(sessionId)
      const lastStderr = managed?.lastStderr
      this.processes.delete(sessionId)
      this.failCounts.delete(sessionId)
      const crashed = code !== null && code !== 0
      db.updateSession(sessionId, {
        status: crashed ? 'error' : 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now()
      })
      if (crashed) {
        const reason = lastStderr ? `Process exited with code ${code}: ${lastStderr}` : `Process exited with code ${code}`
        this.emit('session:error', { sessionId, error: reason })
      }
      // Store exit info for waitForReady to access
      this.processes.set(sessionId, { process: null, adoptedPid: null, exitCode: code, lastStderr })
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

    const baseUrl = session.remoteUrl!.replace(/\/+$/, '')
    const headers: Record<string, string> = {}
    if (session.remoteApiKey) headers['Authorization'] = `Bearer ${session.remoteApiKey}`
    if (session.remoteOrganization) headers['OpenAI-Organization'] = session.remoteOrganization

    const url = `${baseUrl}/v1/models`
    const resolvedUrl = await resolveUrl(url)
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

        console.log(`[SESSION] Remote connected: ${url} (attempt ${attempt})`)
        db.updateSession(session.id, { status: 'running' })
        this.emit('session:ready', { sessionId: session.id, port: session.port })
        return
      } catch (err) {
        lastErr = err as Error
        console.log(`[SESSION] Remote connect attempt ${attempt}/3 failed: ${lastErr.message}`)
        if (attempt < 3) await new Promise(r => setTimeout(r, attempt * 1000))
      }
    }

    db.updateSession(session.id, { status: 'error' })
    this.emit('session:error', { sessionId: session.id, error: `${lastErr!.message} (${url})` })
    throw new Error(`Cannot connect to remote endpoint ${url}: ${lastErr!.message}`)
  }

  async stopSession(sessionId: string): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    // Remote sessions just disconnect (no process to kill) — no lock needed
    if (session.type === 'remote') {
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
        await this.killChildProcess(managed.process)
        this.processes.delete(sessionId)
      } else if (managed?.adoptedPid) {
        this.killPid(managed.adoptedPid)
        await new Promise(r => setTimeout(r, 1500))
        try { process.kill(managed.adoptedPid, 0); this.killPid(managed.adoptedPid, 'SIGKILL') } catch (_) {}
        this.processes.delete(sessionId)
      } else if (session.pid) {
        // Fallback: kill by stored PID
        this.killPid(session.pid)
        await new Promise(r => setTimeout(r, 1500))
        try { process.kill(session.pid, 0); this.killPid(session.pid, 'SIGKILL') } catch (_) {}
      } else {
        // Last resort: kill by port
        await this.killByPort(session.port)
      }

      db.updateSession(sessionId, {
        status: 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now()
      })
      this.emit('session:stopped', { sessionId })
    })
  }

  async deleteSession(sessionId: string): Promise<void> {
    // Stop first if running
    const session = db.getSession(sessionId)
    if (session && (session.status === 'running' || session.status === 'loading')) {
      await this.stopSession(sessionId)
    }

    this.processes.delete(sessionId)
    this.failCounts.delete(sessionId)
    db.deleteSession(sessionId)
    this.emit('session:deleted', { sessionId })
  }

  async updateSessionConfig(sessionId: string, config: Partial<ServerConfig>): Promise<void> {
    const session = db.getSession(sessionId)
    if (!session) throw new Error(`Session ${sessionId} not found`)

    // Validate port if provided
    if (config.port !== undefined) {
      if (config.port < 1024 || config.port > 65535) {
        throw new Error(`Invalid port ${config.port}. Must be between 1024 and 65535.`)
      }
      // Check for port conflicts with ALL other sessions (not just running ones)
      const allSessions = db.getSessions()
      const conflicting = allSessions.find(s => s.port === config.port && s.id !== sessionId)
      if (conflicting) {
        throw new Error(`Port ${config.port} is already used by session "${conflicting.modelName || conflicting.modelPath}".`)
      }
    }

    let currentConfig: Record<string, unknown> = {}
    try {
      currentConfig = JSON.parse(session.config)
    } catch {
      // Corrupted config in DB — start fresh
    }
    const merged = { ...currentConfig, ...config }

    // Always keep host/port in sync between JSON blob and DB columns
    // Extract from merged to ensure JSON blob and DB columns always agree
    const host = (merged.host as string) || session.host
    const port = (merged.port as number) || session.port

    db.updateSession(sessionId, {
      config: JSON.stringify(merged),
      host,
      port
    })
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

      if (!session) {
        // Create a new session record for this detected process
        // Use full defaults so the settings page shows complete config
        const id = uuidv4()
        const now = Date.now()
        // Auto-detect model config for proper defaults (paged cache, parsers, etc.)
        const detected = detectModelConfigFromDir(proc.modelPath)
        const defaultConfig: ServerConfig = {
          modelPath: proc.modelPath,
          host: '127.0.0.1',
          port: proc.port,
          timeout: 300,
          maxNumSeqs: 256,
          prefillBatchSize: 512,
          completionBatchSize: 512,
          continuousBatching: false,
          enablePrefixCache: true,
          prefixCacheSize: 100,
          cacheMemoryMb: 0,
          cacheMemoryPercent: 20,
          noMemoryAwareCache: false,
          usePagedCache: detected.usePagedCache,
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
          host: '127.0.0.1',
          port: proc.port,
          pid: proc.pid,
          status: 'running',
          config: JSON.stringify(defaultConfig),
          createdAt: now,
          updatedAt: now,
          lastStartedAt: now,
          type: 'local'
        }
        db.createSession(session)
      } else {
        // Update existing session with live process info (also normalize stored path)
        db.updateSession(session.id, {
          status: 'running',
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
      if (s.status === 'running' || s.status === 'loading') {
        if (s.type === 'remote') {
          // Remote sessions have no persistent connection — if the app restarts,
          // they must be reconnected. Mark them stopped so the user can reconnect.
          console.log(`[SESSIONS] Resetting stale remote session "${s.modelName}" to stopped (was ${s.status})`)
          db.updateSession(s.id, { status: 'stopped' })
          this.emit('session:stopped', { sessionId: s.id })
        } else if (!adopted.find(a => a.id === s.id)) {
          db.updateSession(s.id, { status: 'stopped', pid: undefined })
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
            const resolvedHealthUrl = await resolveUrl(`${remoteBase}/v1/models`)
            const pingStart = Date.now()
            const res = await fetch(resolvedHealthUrl, {
              headers: remoteHeaders,
              signal: AbortSignal.timeout(10000)
            })
            const latencyMs = Date.now() - pingStart
            if (res.ok) {
              this.failCounts.delete(session.id)
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
            `http://${session.host}:${session.port}/health`,
            { signal: AbortSignal.timeout(10000) }
          )
          if (res.ok) {
            const data = await res.json()
            // Reset fail counter on success
            this.failCounts.delete(session.id)
            if (data.model_name && data.model_name !== session.modelName) {
              db.updateSession(session.id, { modelName: data.model_name })
            }
            if (session.status === 'loading') {
              db.updateSession(session.id, { status: 'running' })
              this.emit('session:ready', { sessionId: session.id, port: session.port })
            }
            this.emit('session:health', {
              sessionId: session.id,
              running: true,
              modelName: data.model_name,
              port: session.port
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
    }, 5000)
  }

  stopGlobalMonitor(): void {
    if (this.monitorInterval) {
      clearInterval(this.monitorInterval)
      this.monitorInterval = null
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
      } catch (_) {}
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
        // Remote sessions: no process to kill, but notify that active inference
        // should be aborted. The 'session:down' event is consumed by ipc/sessions.ts
        // to call abortByEndpoint before the session is marked stopped.
        console.log(`[SESSIONS] handleSessionDown: remote session ${sessionId} ("${session.modelName}") marked down`)
        this.emit('session:abortInference', { sessionId, host: session.host, port: session.port })
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
          this.killByPort(session.port).catch(() => {})
        }
        this.processes.delete(sessionId)
      }
      this.failCounts.delete(sessionId)
      db.updateSession(sessionId, {
        status: 'stopped',
        pid: undefined,
        lastStoppedAt: Date.now()
      })
      this.emit('session:stopped', { sessionId })
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
        try { managed.process.kill('SIGTERM') } catch (_) {}
      }
    }

    // Wait for graceful shutdown, then SIGKILL any survivors
    if (processes.length > 0 || this.processes.size > 0) {
      await new Promise(r => setTimeout(r, 3000))
      for (const proc of processes) {
        try { process.kill(proc.pid, 'SIGKILL') } catch (_) {}
      }
      for (const [, managed] of this.processes) {
        if (managed.process) {
          try { managed.process.kill('SIGKILL') } catch (_) {}
        }
      }
    }

    this.processes.clear()

    // Mark all sessions as stopped in DB
    const sessions = db.getSessions()
    for (const s of sessions) {
      if (s.status === 'running' || s.status === 'loading') {
        db.updateSession(s.id, { status: 'stopped', pid: undefined, lastStoppedAt: Date.now() })
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

    // Server settings — always pass explicitly
    args.push('--host', config.host)
    args.push('--port', config.port.toString())
    args.push('--timeout', (config.timeout != null && config.timeout > 0 ? config.timeout : 86400).toString())

    if (config.rateLimit && config.rateLimit > 0) args.push('--rate-limit', config.rateLimit.toString())

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
    if (config.continuousBatching) args.push('--continuous-batching')

    // Auto-detect tool/reasoning parser from model's config.json (authoritative) first,
    // then fall back to name-based regex matching. This prevents misdetection of fine-tunes
    // (e.g., a Qwen3 model named "Nemotron-Orchestrator" getting hybrid cache config).
    const detected = detectModelConfigFromDir(config.modelPath)

    // Parser resolution: detection ALWAYS wins when available.
    // This prevents stale session configs (from older registry versions) from overriding
    // correct auto-detected parsers (e.g., GLM-4.7 Flash stuck with deepseek_r1 instead of openai_gptoss).
    // Empty string "" = user explicitly chose "None" (disabled) — always respected.
    // If detection returns nothing (unknown model), fall back to saved session value.
    const userToolParser = config.toolCallParser
    const effectiveToolParser = userToolParser === ''
      ? undefined                     // User explicitly chose "None" — respect it
      : detected.toolParser           // Auto-detected wins when available
        || (userToolParser && userToolParser !== 'auto' ? userToolParser : undefined)  // Fallback for unknown models
    const effectiveAutoTool = config.enableAutoToolChoice ?? detected.enableAutoToolChoice
    const userReasoningParser = config.reasoningParser
    const effectiveReasoningParser = userReasoningParser === ''
      ? undefined                     // User explicitly chose "None" — respect it
      : detected.reasoningParser      // Auto-detected wins when available
        || (userReasoningParser && userReasoningParser !== 'auto' ? userReasoningParser : undefined)  // Fallback for unknown models

    // Log parser resolution with override warnings
    if (userReasoningParser && userReasoningParser !== 'auto' && detected.reasoningParser && userReasoningParser !== detected.reasoningParser) {
      console.log(`[SESSION] WARNING: Session had reasoningParser="${userReasoningParser}" but auto-detected "${detected.reasoningParser}" for ${detected.family}. Using detected value.`)
    }
    if (userToolParser && userToolParser !== 'auto' && detected.toolParser && userToolParser !== detected.toolParser) {
      console.log(`[SESSION] WARNING: Session had toolCallParser="${userToolParser}" but auto-detected "${detected.toolParser}" for ${detected.family}. Using detected value.`)
    }
    console.log(`[SESSION] Model family: ${detected.family} | tool: ${effectiveToolParser || 'none'} (user=${userToolParser}, detected=${detected.toolParser || 'none'}) | reasoning: ${effectiveReasoningParser || 'none'} (user=${userReasoningParser}, detected=${detected.reasoningParser || 'none'}) | autoTool: ${effectiveAutoTool}`)

    // Prefix cache — requires --continuous-batching to take effect in vllm-mlx
    // When MCP tools + auto-tool-choice are enabled, force prefix cache ON.
    // Tool follow-up requests share most of the prompt with the original request;
    // without prefix cache each follow-up re-processes the entire prompt (~16s).
    const toolsNeedCache = !!(effectiveAutoTool && config.mcpConfig)
    const prefixCacheOff = config.enablePrefixCache === false && !toolsNeedCache

    if (prefixCacheOff) {
      args.push('--disable-prefix-cache')
    } else {
      // Auto-enable continuous batching when prefix cache is on (required by vllm-mlx)
      if (!config.continuousBatching && !args.includes('--continuous-batching')) {
        args.push('--continuous-batching')
      }
      // Set safe prefill batch size to prevent Metal GPU crashes with large contexts
      if ((!config.prefillBatchSize || config.prefillBatchSize === 0) && !args.some(a => a === '--prefill-batch-size')) {
        args.push('--prefill-batch-size', '4096')
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
      }
    }

    // Paged cache (auto-detect from model family if not explicitly configured)
    // Paged cache is a prefix cache backend — skip when prefix cache is disabled
    if (!prefixCacheOff && (config.usePagedCache ?? detected.usePagedCache)) {
      args.push('--use-paged-cache')
      if (config.pagedCacheBlockSize && config.pagedCacheBlockSize > 0) {
        args.push('--paged-cache-block-size', config.pagedCacheBlockSize.toString())
      }
      if (config.maxCacheBlocks && config.maxCacheBlocks > 0) {
        args.push('--max-cache-blocks', config.maxCacheBlocks.toString())
      }
    }

    // KV cache quantization — only meaningful when prefix cache is active
    if (!prefixCacheOff && config.kvCacheQuantization && config.kvCacheQuantization !== 'none') {
      args.push('--kv-cache-quantization', config.kvCacheQuantization)
      if (config.kvCacheGroupSize && config.kvCacheGroupSize !== 64) {
        args.push('--kv-cache-group-size', config.kvCacheGroupSize.toString())
      }
    }

    // Disk cache (L2 persistent cache) — only meaningful with prefix cache
    if (!prefixCacheOff && config.enableDiskCache) {
      args.push('--enable-disk-cache')
      if (config.diskCacheDir) {
        args.push('--disk-cache-dir', config.diskCacheDir)
      }
      if (config.diskCacheMaxGb != null && config.diskCacheMaxGb >= 0) {
        args.push('--disk-cache-max-gb', config.diskCacheMaxGb.toString())
      }
    }

    // Block-level disk cache (L2 for paged cache blocks)
    if (config.usePagedCache && config.enableBlockDiskCache) {
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

    // Tool integration
    if (config.mcpConfig) args.push('--mcp-config', config.mcpConfig)
    if (effectiveAutoTool) {
      args.push('--enable-auto-tool-choice')
      if (effectiveToolParser) args.push('--tool-call-parser', effectiveToolParser)
    }

    // Reasoning parser
    if (effectiveReasoningParser) args.push('--reasoning-parser', effectiveReasoningParser)

    // Additional arguments
    if (config.additionalArgs) {
      args.push(...config.additionalArgs.trim().split(/\s+/))
    }

    return args
  }

  findVllmMlx(): VllmMlxPath | null {
    // Bundled Python: use python3 -m vllm_mlx.cli instead of vllm-mlx binary
    // This avoids shebang path issues in relocatable Python builds
    const bundledPython = getBundledPythonPath()
    if (bundledPython) {
      try {
        execSync(`"${bundledPython}" -c "import vllm_mlx"`, {
          encoding: 'utf-8',
          timeout: 10000
        })
        return { type: 'bundled', pythonPath: bundledPython }
      } catch (_) {
        console.log('[SESSIONS] Bundled Python found but vllm_mlx import failed, trying system')
      }
    }

    // System binary search
    const home = homedir()
    const locations = [
      join(home, '.local', 'bin', 'vllm-mlx'),     // uv tool / pip --user
      '/opt/homebrew/bin/vllm-mlx',                  // Homebrew (Apple Silicon)
      '/usr/local/bin/vllm-mlx',                     // Homebrew (Intel) / system pip
      '/usr/bin/vllm-mlx',                           // System pip
      join(home, 'miniforge3', 'bin', 'vllm-mlx'),  // Miniforge
      join(home, 'anaconda3', 'bin', 'vllm-mlx'),   // Anaconda
      join(home, 'miniconda3', 'bin', 'vllm-mlx'),  // Miniconda
    ]

    // Scan pyenv versions (common on macOS)
    const pyenvRoot = join(home, '.pyenv', 'versions')
    try {
      if (existsSync(pyenvRoot)) {
        for (const ver of readdirSync(pyenvRoot)) {
          locations.push(join(pyenvRoot, ver, 'bin', 'vllm-mlx'))
        }
      }
    } catch (_) {}

    for (const loc of locations) {
      if (existsSync(loc)) return { type: 'system', binaryPath: loc }
    }

    // Fallback: check PATH via login shell (picks up pyenv, nvm, etc.)
    for (const shell of ['/bin/zsh', '/bin/bash']) {
      try {
        const result = execSync(
          `${shell} -lc "which vllm-mlx"`,
          { encoding: 'utf-8', timeout: 5000 }
        ).trim()
        if (result && existsSync(result)) return { type: 'system', binaryPath: result }
      } catch (_) {}
    }

    // Last resort: plain which
    try {
      const result = execSync('which vllm-mlx', { encoding: 'utf-8', timeout: 3000 }).trim()
      if (result && existsSync(result)) return { type: 'system', binaryPath: result }
    } catch (_) {}
    return null
  }

  private async findAvailablePort(): Promise<number> {
    const sessions = db.getSessions()
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

  /** Check if a session's process is still alive via PID signal-0 probe. */
  private isProcessAlive(sessionId: string, dbPid?: number): boolean {
    // Check managed process first (spawned or adopted)
    const managed = this.processes.get(sessionId)
    const pid = managed?.adoptedPid ?? managed?.process?.pid ?? dbPid
    if (!pid) return false
    try {
      process.kill(pid, 0) // Signal 0: doesn't kill, just checks existence
      return true
    } catch (_) {
      return false
    }
  }

  private killPid(pid: number, signal: NodeJS.Signals = 'SIGTERM'): void {
    // Try process group kill first (negative PID kills entire group).
    // This ensures MCP subprocesses and uvicorn workers are also killed.
    // Falls back to single-PID kill if group kill fails (e.g., not a group leader).
    try { process.kill(-pid, signal) } catch (_) {
      try { process.kill(pid, signal) } catch (_) {}
    }
  }

  private async killByPort(port: number): Promise<void> {
    try {
      const output = execSync(`lsof -ti tcp:${port}`, { encoding: 'utf-8', timeout: 5000 }).trim()
      if (output) {
        for (const pidStr of output.split('\n')) {
          this.killPid(parseInt(pidStr))
        }
        await new Promise(r => setTimeout(r, 1500))
      }
    } catch (_) {}
  }

  private async killChildProcess(proc: ChildProcess): Promise<void> {
    const pid = proc.pid
    return new Promise((resolve) => {
      // Escalate to SIGKILL after 10s if SIGTERM doesn't work
      const killTimeout = setTimeout(() => {
        // Kill entire process group (detached spawn)
        if (pid) { try { process.kill(-pid, 'SIGKILL') } catch (_) {} }
        try { proc.kill('SIGKILL') } catch (_) {}
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
      if (pid) { try { process.kill(-pid, 'SIGTERM') } catch (_) {} }
      try { proc.kill('SIGTERM') } catch (_) {
        clearTimeout(killTimeout)
        clearTimeout(hardTimeout)
        resolve()
      }
    })
  }

  private async waitForReady(host: string, port: number, maxWait = 120000, sessionId?: string): Promise<void> {
    const startTime = Date.now()
    const healthUrl = `http://${host}:${port}/health`

    while (Date.now() - startTime < maxWait) {
      // Abort early if the process exited while we were waiting
      if (sessionId) {
        const managed = this.processes.get(sessionId)
        if (managed && !managed.process && !managed.adoptedPid) {
          // Process exited — include the reason if available
          const reason = managed.lastStderr || `exit code ${managed.exitCode ?? 'unknown'}`
          throw new Error(`Process exited before becoming ready: ${reason}`)
        }
        if (!managed) {
          throw new Error('Process exited before becoming ready')
        }
      }

      try {
        const response = await fetch(healthUrl, { signal: AbortSignal.timeout(1000) })
        if (response.ok) return
      } catch (_) {}
      await new Promise(resolve => setTimeout(resolve, 500))
    }

    throw new Error('Server failed to start within timeout period')
  }
}

export const sessionManager = new SessionManager()
