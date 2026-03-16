/**
 * ProcessManager — manages multiple vmlx-engine serve processes.
 *
 * Each model runs as a separate process on a unique port, providing:
 * - Crash isolation (VLM crash doesn't kill text LLM)
 * - Independent lifecycle management
 * - Health monitoring per process
 *
 * Integrates with existing SessionManager (sessions.ts) — this module
 * handles the multi-model orchestration layer on top.
 */

import { spawn, ChildProcess } from 'child_process'
import { randomUUID } from 'crypto'
import { EventEmitter } from 'events'
import { createServer } from 'net'

export interface ModelProcess {
  id: string
  model: string
  port: number
  pid: number | null
  status: 'starting' | 'running' | 'stopped' | 'error'
  gpuMemoryMB: number
  lastRequestTime: number
  startedAt: number
  pinned: boolean
  error?: string
}

interface ManagedModelProcess {
  id: string
  model: string
  port: number
  pid: number | null
  process: ChildProcess | null
  status: ModelProcess['status']
  gpuMemoryMB: number
  lastRequestTime: number
  startedAt: number
  pinned: boolean
  error?: string
  healthCheckTimer?: ReturnType<typeof setInterval>
  killing?: boolean  // True during intentional kill — suppresses error status in exit handler
}

/**
 * Find a free port by binding to port 0.
 */
async function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer()
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      if (addr && typeof addr === 'object') {
        const port = addr.port
        srv.close(() => resolve(port))
      } else {
        srv.close(() => reject(new Error('Could not determine port')))
      }
    })
    srv.on('error', reject)
  })
}

export class ProcessManager extends EventEmitter {
  private processes = new Map<string, ManagedModelProcess>()
  private spawning = new Set<string>()  // Guard against concurrent spawn of same model
  private maxMemoryGB: number
  private healthPollMs = 5000

  constructor(maxMemoryGB?: number) {
    super()
    // Default: system RAM minus 8 GB headroom
    const totalGB = Math.round(require('os').totalmem() / (1024 ** 3))
    this.maxMemoryGB = maxMemoryGB ?? Math.max(totalGB - 8, 8)
  }

  /**
   * Spawn a new model server process.
   */
  async spawn(
    model: string,
    opts: {
      port?: number
      pythonPath?: string
      binaryPath?: string
      extraArgs?: string[]
      pinned?: boolean
    } = {}
  ): Promise<ModelProcess> {
    // Check if already running or being spawned
    for (const [, mp] of this.processes) {
      if (mp.model === model && (mp.status === 'running' || mp.status === 'starting')) {
        return this.toPublic(mp)
      }
    }
    if (this.spawning.has(model)) {
      throw new Error(`Model ${model} is already being spawned`)
    }
    this.spawning.add(model)

    let port: number
    try {
      port = opts.port ?? await findFreePort()
    } catch (e) {
      this.spawning.delete(model)
      throw e
    }
    const id = `${model.split('/').pop()}-${port}-${randomUUID().slice(0, 8)}`

    const managed: ManagedModelProcess = {
      id,
      model,
      port,
      pid: null,
      process: null,
      status: 'starting',
      gpuMemoryMB: 0,
      lastRequestTime: 0,
      startedAt: Date.now(),
      pinned: opts.pinned ?? false,
    }

    // Build command
    let cmd: string
    let args: string[]

    if (opts.pythonPath) {
      cmd = opts.pythonPath
      args = [
        '-s', '-m', 'vmlx_engine.cli', 'serve',
        '--model', model,
        '--port', String(port),
        '--host', '0.0.0.0',
        ...(opts.extraArgs ?? []),
      ]
    } else if (opts.binaryPath) {
      cmd = opts.binaryPath
      args = [
        'serve',
        '--model', model,
        '--port', String(port),
        '--host', '0.0.0.0',
        ...(opts.extraArgs ?? []),
      ]
    } else {
      throw new Error('Either pythonPath or binaryPath must be provided')
    }

    const child = spawn(cmd, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONNOUSERSITE: '1', PYTHONPATH: '' },
    })

    managed.process = child
    managed.pid = child.pid ?? null

    child.stdout?.on('data', (data: Buffer) => {
      this.emit('log', { id, data: data.toString() })
    })

    child.stderr?.on('data', (data: Buffer) => {
      this.emit('log', { id, data: data.toString() })
    })

    child.on('exit', (code, signal) => {
      // Intentional kill (SIGTERM/SIGKILL from kill()) → treat as clean stop, not error
      const isClean = code === 0 || managed.killing
      managed.status = isClean ? 'stopped' : 'error'
      managed.error = !isClean ? `Process exited with code ${code} (${signal || 'no signal'})` : undefined
      managed.process = null
      if (managed.healthCheckTimer) {
        clearInterval(managed.healthCheckTimer)
        managed.healthCheckTimer = undefined
      }
      this.emit('process:exit', { id, model, code, signal })

      // Clean exit: remove immediately. Unexpected crash: keep 60s for tray visibility.
      if (isClean) {
        this.processes.delete(id)
      } else {
        setTimeout(() => this.processes.delete(id), 60_000)
      }
    })

    this.processes.set(id, managed)
    this.spawning.delete(model)

    // Start health checking
    this.startHealthCheck(managed)

    this.emit('process:spawn', this.toPublic(managed))
    return this.toPublic(managed)
  }

  /**
   * Kill a model process.
   */
  async kill(id: string): Promise<void> {
    const managed = this.processes.get(id)
    if (!managed) return

    if (managed.healthCheckTimer) {
      clearInterval(managed.healthCheckTimer)
      managed.healthCheckTimer = undefined
    }

    managed.killing = true

    if (managed.process) {
      managed.process.kill('SIGTERM')
      // Wait up to 5s for graceful shutdown, then force kill
      await new Promise<void>((resolve) => {
        const timeout = setTimeout(() => {
          if (managed.process) {
            managed.process.kill('SIGKILL')
          }
          resolve()
        }, 5000)

        if (managed.process) {
          managed.process.once('exit', () => {
            clearTimeout(timeout)
            resolve()
          })
        } else {
          clearTimeout(timeout)
          resolve()
        }
      })
    }

    managed.status = 'stopped'
    managed.process = null
    this.processes.delete(id)
    this.emit('process:killed', { id })
  }

  /**
   * Kill all processes.
   */
  async killAll(): Promise<void> {
    const ids = Array.from(this.processes.keys())
    await Promise.all(ids.map((id) => this.kill(id)))
  }

  /**
   * Get all process states.
   */
  list(): ModelProcess[] {
    return Array.from(this.processes.values()).map((mp) => this.toPublic(mp))
  }

  /**
   * Get total GPU memory usage across all processes.
   */
  totalMemoryMB(): number {
    let total = 0
    for (const mp of this.processes.values()) {
      if (mp.status === 'running') {
        total += mp.gpuMemoryMB
      }
    }
    return total
  }

  /**
   * Auto-evict LRU idle processes when over memory budget.
   */
  async autoEvict(): Promise<string[]> {
    const totalMB = this.totalMemoryMB()
    const maxMB = this.maxMemoryGB * 1024

    if (totalMB <= maxMB) return []

    const evicted: string[] = []
    const candidates = Array.from(this.processes.values())
      .filter((mp) => mp.status === 'running' && !mp.pinned)
      .sort((a, b) => a.lastRequestTime - b.lastRequestTime) // LRU first

    for (const mp of candidates) {
      if (this.totalMemoryMB() <= maxMB) break
      await this.kill(mp.id)
      evicted.push(mp.id)
      this.emit('process:evicted', { id: mp.id, model: mp.model, reason: 'memory' })
    }

    return evicted
  }

  /**
   * Set pinned state for a process (pinned processes are exempt from eviction).
   */
  setPinned(id: string, pinned: boolean): void {
    const managed = this.processes.get(id)
    if (managed) {
      managed.pinned = pinned
      this.emit('process:pinChanged', { id, pinned })
    }
  }

  /**
   * Set memory limit.
   */
  setMaxMemoryGB(gb: number): void {
    this.maxMemoryGB = gb
  }

  private startHealthCheck(managed: ManagedModelProcess): void {
    managed.healthCheckTimer = setInterval(async () => {
      if (managed.status !== 'starting' && managed.status !== 'running') return

      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 3000)
      try {
        const res = await fetch(`http://127.0.0.1:${managed.port}/health`, {
          signal: controller.signal,
        })

        if (res.ok) {
          const data = await res.json()
          const wasStarting = managed.status === 'starting'
          managed.status = 'running'
          managed.lastRequestTime = data.last_request_time ?? managed.lastRequestTime

          if (data.memory) {
            managed.gpuMemoryMB = data.memory.active_mb ?? 0
          }

          if (wasStarting) {
            this.emit('process:ready', this.toPublic(managed))
          }

          this.emit('process:health', this.toPublic(managed))
        }
      } catch {
        // Still starting or crashed — ignore
      } finally {
        clearTimeout(timeout)
      }
    }, this.healthPollMs)
  }

  private toPublic(mp: ManagedModelProcess): ModelProcess {
    return {
      id: mp.id,
      model: mp.model,
      port: mp.port,
      pid: mp.process?.pid ?? mp.pid,
      status: mp.status,
      gpuMemoryMB: mp.gpuMemoryMB,
      lastRequestTime: mp.lastRequestTime,
      startedAt: mp.startedAt,
      pinned: mp.pinned,
      error: mp.error,
    }
  }
}
