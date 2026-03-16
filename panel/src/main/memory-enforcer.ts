/**
 * Memory enforcer — monitors total GPU memory across all model processes
 * and evicts idle/LRU models when the budget is exceeded.
 *
 * Eviction cascade:
 * 1. Kill processes with expired TTL (idle > configured, not pinned)
 * 2. Kill LRU unpinned (oldest lastRequestTime)
 * 3. If all pinned → emit notification (no forced eviction)
 *
 * Integrates with ProcessManager for health data and tray for notifications.
 */

import { Notification } from 'electron'
import type { ProcessManager } from './process-manager'
import { db } from './database'

let enforceTimer: ReturnType<typeof setTimeout> | null = null

export interface MemoryEnforcerConfig {
  pollIntervalMs: number
  maxMemoryGB: number
  defaultTTLMinutes: number
}

const DEFAULT_CONFIG: MemoryEnforcerConfig = {
  pollIntervalMs: 5000,
  maxMemoryGB: 0, // 0 = auto (system RAM - 8GB)
  defaultTTLMinutes: 30,
}

/**
 * Get effective config from SQLite settings.
 */
function getConfig(): MemoryEnforcerConfig {
  const config = { ...DEFAULT_CONFIG }

  const maxMem = db.getSetting('memory_max_gb')
  if (maxMem) config.maxMemoryGB = parseFloat(maxMem)

  const ttl = db.getSetting('memory_default_ttl')
  if (ttl) config.defaultTTLMinutes = parseInt(ttl, 10)

  const poll = db.getSetting('memory_poll_interval')
  if (poll) config.pollIntervalMs = parseInt(poll, 10)

  // Auto-calculate if not set
  if (config.maxMemoryGB <= 0) {
    const totalGB = Math.round(require('os').totalmem() / (1024 ** 3))
    config.maxMemoryGB = Math.max(totalGB - 8, 8)
  }

  return config
}

/**
 * Run one enforcement cycle.
 */
async function enforce(pm: ProcessManager): Promise<void> {
  const config = getConfig()
  const maxMB = config.maxMemoryGB * 1024
  const totalMB = pm.totalMemoryMB()

  if (totalMB <= maxMB) return

  const processes = pm.list()
    .filter((p) => p.status === 'running')

  const now = Date.now() / 1000

  // Phase 1: Kill expired TTL (idle > configured, not pinned)
  const ttlSeconds = config.defaultTTLMinutes * 60
  const expired = processes
    .filter((p) => !p.pinned && p.lastRequestTime > 0 && (now - p.lastRequestTime) > ttlSeconds)
    .sort((a, b) => a.lastRequestTime - b.lastRequestTime) // Oldest first

  for (const proc of expired) {
    if (pm.totalMemoryMB() <= maxMB) break
    console.log(`[MemoryEnforcer] Evicting ${proc.model} (TTL expired, idle ${Math.round((now - proc.lastRequestTime) / 60)}m)`)
    await pm.kill(proc.id)
  }

  if (pm.totalMemoryMB() <= maxMB) return

  // Phase 2: Kill LRU unpinned
  const lru = processes
    .filter((p) => !p.pinned && p.status === 'running')
    .sort((a, b) => a.lastRequestTime - b.lastRequestTime)

  for (const proc of lru) {
    if (pm.totalMemoryMB() <= maxMB) break
    console.log(`[MemoryEnforcer] Evicting ${proc.model} (LRU, last used ${Math.round((now - proc.lastRequestTime) / 60)}m ago)`)
    await pm.kill(proc.id)
  }

  if (pm.totalMemoryMB() <= maxMB) return

  // Phase 3: All remaining are pinned — notify user
  const remaining = pm.totalMemoryMB()
  console.warn(`[MemoryEnforcer] Over budget (${(remaining / 1024).toFixed(1)} / ${config.maxMemoryGB} GB) but all models are pinned`)
  try {
    new Notification({
      title: 'vMLX Memory Warning',
      body: `Using ${(remaining / 1024).toFixed(1)} GB of ${config.maxMemoryGB} GB budget. All models are pinned — unpin or stop a model to free memory.`,
    }).show()
  } catch {
    // Notifications may not be available in all environments
  }
}

let currentPollMs = 0

/**
 * Start the memory enforcer polling loop.
 * Re-reads poll interval from config each cycle to pick up setting changes.
 */
export function startMemoryEnforcer(pm: ProcessManager): void {
  if (enforceTimer) return

  const config = getConfig()
  currentPollMs = config.pollIntervalMs
  console.log(`[MemoryEnforcer] Started (max ${config.maxMemoryGB} GB, poll ${config.pollIntervalMs}ms, TTL ${config.defaultTTLMinutes}m)`)

  const scheduleTick = (): void => {
    enforceTimer = setTimeout(() => {
      enforce(pm)
        .catch((err) => {
          console.error('[MemoryEnforcer] Error during enforcement:', err)
        })
        .finally(() => {
          if (enforceTimer === null) return  // stopMemoryEnforcer was called
          // Re-read interval in case user changed it in settings
          const newPollMs = getConfig().pollIntervalMs
          if (newPollMs !== currentPollMs) {
            console.log(`[MemoryEnforcer] Poll interval changed: ${currentPollMs}ms → ${newPollMs}ms`)
            currentPollMs = newPollMs
          }
          scheduleTick()
        })
    }, currentPollMs)
  }

  scheduleTick()
}

/**
 * Stop the memory enforcer.
 */
export function stopMemoryEnforcer(): void {
  if (enforceTimer) {
    clearTimeout(enforceTimer)
    enforceTimer = null
    console.log('[MemoryEnforcer] Stopped')
  }
}
