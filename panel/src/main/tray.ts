/**
 * Menu bar tray for vMLX.
 *
 * Shows model status, memory usage, and provides quick controls.
 * Window close keeps app alive in system tray when enabled.
 *
 * Tray icon states:
 * - Green: at least one model serving
 * - Yellow: starting up / loading model
 * - Gray: no models loaded
 */

import { app, Tray, Menu, nativeImage, BrowserWindow, clipboard } from 'electron'
import type { ProcessManager } from './process-manager'
import { db } from './database'
import { sessionManager, connectHost } from './sessions'

let tray: Tray | null = null
let boundProcessManager: ProcessManager | null = null
const boundListeners: Array<{ event: string; fn: (...args: any[]) => void }> = []

/** Track GPU memory per session (updated from session:health events) */
const sessionMemoryMB: Map<string, number> = new Map()

/**
 * Create a colored circle icon for tray.
 * Uses nativeImage.createFromDataURL with a simple SVG circle.
 */
function createTrayIcon(color: 'green' | 'yellow' | 'red' | 'gray'): Electron.NativeImage {
  // Generate a 44x44 PNG via Canvas-like pixel buffer.
  // macOS tray icons are 22x22 logical pixels; we provide @2x for retina.
  const s = 44
  const cx = s / 2, cy = s / 2, r = s / 2 - 4

  // RGBA pixel buffer
  const buf = Buffer.alloc(s * s * 4, 0)

  const colorValues: Record<string, [number, number, number]> = {
    green:  [34, 197, 94],
    yellow: [234, 179, 8],
    red:    [239, 68, 68],
    gray:   [255, 255, 255],
  }
  const [cr, cg, cb] = colorValues[color]

  for (let y = 0; y < s; y++) {
    for (let x = 0; x < s; x++) {
      const dx = x - cx, dy = y - cy
      const dist = Math.sqrt(dx * dx + dy * dy)
      const offset = (y * s + x) * 4

      if (color === 'gray') {
        // Ring outline for idle state
        const ringOuter = r, ringInner = r - 3
        if (dist <= ringOuter && dist >= ringInner) {
          // Anti-alias edges
          let alpha = 255
          if (dist > ringOuter - 1) alpha = Math.round((ringOuter - dist) * 255)
          else if (dist < ringInner + 1) alpha = Math.round((dist - ringInner) * 255)
          alpha = Math.max(0, Math.min(255, alpha))
          buf[offset] = cr; buf[offset + 1] = cg; buf[offset + 2] = cb; buf[offset + 3] = alpha
        }
      } else {
        // Filled circle for active states
        if (dist <= r) {
          let alpha = 255
          if (dist > r - 1) alpha = Math.round((r - dist) * 255)
          alpha = Math.max(0, Math.min(255, alpha))
          buf[offset] = cr; buf[offset + 1] = cg; buf[offset + 2] = cb; buf[offset + 3] = alpha
        }
      }
    }
  }

  const img = nativeImage.createFromBuffer(buf, { width: s, height: s })
  return img.resize({ width: 18, height: 18 })
}

/**
 * Build the tray context menu.
 */
function buildMenu(
  processManager: ProcessManager,
  getWindow: () => BrowserWindow | null,
): Electron.Menu {
  const processes = processManager.list()
  const running = processes.filter((p) => p.status === 'running')
  const totalGB = Math.round(require('os').totalmem() / (1024 ** 3))

  // Sum GPU memory from both ProcessManager and SessionManager health data
  let totalMemMB = processManager.totalMemoryMB()
  for (const mb of sessionMemoryMB.values()) {
    totalMemMB += mb
  }

  // Count ALL running models: ProcessManager + SessionManager (deduplicated)
  let runningSessions: any[] = []
  try {
    runningSessions = db.getSessions().filter((s: any) => s.status === 'running' || s.status === 'standby')
  } catch {}
  const sessionOnlyCount = runningSessions.filter((s: any) =>
    !processes.some(p => p.port === s.port)
  ).length
  const totalRunning = running.length + sessionOnlyCount

  const items: Electron.MenuItemConstructorOptions[] = [
    {
      label: `vMLX — ${totalRunning} model${totalRunning !== 1 ? 's' : ''} loaded`,
      enabled: false,
    },
    { type: 'separator' },
  ]

  // Per-model entries
  for (const proc of processes) {
    const statusIcon = proc.status === 'running' ? '●'
      : proc.status === 'starting' ? '◐'
        : proc.status === 'error' ? '✕' : '○'

    const memLabel = proc.gpuMemoryMB > 0
      ? ` — ${(proc.gpuMemoryMB / 1024).toFixed(1)} GB`
      : ''

    const modelName = proc.model.split('/').pop() || proc.model

    items.push({
      label: `${statusIcon} ${modelName} (:${proc.port})${memLabel}`,
      submenu: [
        {
          label: proc.status === 'running' ? '✓ Running' : proc.status,
          enabled: false,
        },
        {
          label: `Port: ${proc.port}`,
          enabled: false,
        },
        { type: 'separator' },
        {
          label: 'Copy API URL',
          click: () => {
            clipboard.writeText(`http://127.0.0.1:${proc.port}/v1`)
          },
        },
        {
          label: proc.pinned ? 'Unpin (allow eviction)' : 'Pin (prevent eviction)',
          click: () => {
            processManager.setPinned(proc.id, !proc.pinned)
            rebuildMenu(processManager, getWindow)
          },
        },
        { type: 'separator' },
        {
          label: 'Stop',
          click: async () => {
            try {
              await processManager.kill(proc.id)
              rebuildMenu(processManager, getWindow)
            } catch (err) {
              console.error(`[Tray] Failed to stop ${proc.id}:`, err)
            }
          },
        },
      ],
    })
  }

  // Add SessionManager sessions (includes image servers not in ProcessManager)
  try {
    const sessions = db.getSessions().filter(s => s.status === 'running' || s.status === 'standby')
    for (const s of sessions) {
      // Skip if already shown via ProcessManager
      const alreadyShown = processes.some(p => p.port === s.port)
      if (alreadyShown) continue

      let isImage = false
      try { isImage = JSON.parse(s.config || '{}').modelType === 'image' } catch {}
      const isSleeping = s.status === 'standby'
      const icon = isSleeping ? '💤' : isImage ? '🖼' : '●'
      const modelName = s.modelName || s.modelPath?.split('/').pop() || 'Unknown'
      const sessMem = sessionMemoryMB.get(s.id) || 0
      const sessMemLabel = sessMem > 0 ? ` — ${(sessMem / 1024).toFixed(1)} GB` : ''

      items.push({
        label: `${icon} ${modelName} (:${s.port})${sessMemLabel}`,
        submenu: [
          {
            label: 'Copy API URL',
            click: () => { clipboard.writeText(`http://${connectHost(s.host)}:${s.port}/v1`) },
          },
          { type: 'separator' },
          ...(isSleeping ? [{
            label: 'Wake',
            click: async () => {
              try {
                await sessionManager.wakeSession(s.id)
                rebuildMenu(processManager, getWindow)
              } catch (err) {
                console.error(`[Tray] Failed to wake session ${s.id}:`, err)
              }
            },
          }] : [{
            label: 'Sleep',
            click: async () => {
              try {
                await sessionManager.softSleep(s.id)
                rebuildMenu(processManager, getWindow)
              } catch (err) {
                console.error(`[Tray] Failed to sleep session ${s.id}:`, err)
              }
            },
          }]),
          { type: 'separator' as const },
          {
            label: 'Stop',
            click: async () => {
              try {
                await sessionManager.stopSession(s.id)
                rebuildMenu(processManager, getWindow)
              } catch (err) {
                console.error(`[Tray] Failed to stop session ${s.id}:`, err)
              }
            },
          },
        ],
      })
    }
  } catch (_) {}

  if (totalRunning === 0) {
    items.push({
      label: 'No models loaded',
      enabled: false,
    })
  }

  items.push(
    { type: 'separator' },
    {
      label: `Memory: ${(totalMemMB / 1024).toFixed(1)} / ${totalGB} GB`,
      enabled: false,
    },
    { type: 'separator' },
    {
      label: 'Open vMLX Window',
      click: () => {
        const win = getWindow()
        if (win && !win.isDestroyed()) {
          win.show()
          win.focus()
        } else {
          // Window was closed — recreate
          app.emit('activate')
        }
      },
    },
    {
      label: 'Quit vMLX',
      click: () => {
        app.quit()
      },
    },
  )

  return Menu.buildFromTemplate(items)
}

/**
 * Rebuild the tray menu and update icon (call after process state changes).
 */
export function rebuildMenu(
  processManager: ProcessManager,
  getWindow: () => BrowserWindow | null,
): void {
  if (!tray) return
  const processes = processManager.list()

  // Icon color: check BOTH ProcessManager and SessionManager for running/standby models
  let hasRunning = processes.some(p => p.status === 'running')
  let hasStarting = processes.some(p => p.status === 'starting')
  let hasStandby = false
  if (!hasRunning) {
    try {
      const sessions = db.getSessions()
      hasRunning = sessions.some((s: any) => s.status === 'running')
      hasStandby = sessions.some((s: any) => s.status === 'standby')
      hasStarting = hasStarting || sessions.some((s: any) => s.status === 'loading')
    } catch {}
  }
  // Green = running, Yellow = starting or standby (process alive, model sleeping), Gray = nothing
  const iconColor = hasRunning ? 'green' : (hasStarting || hasStandby) ? 'yellow' : 'gray'
  tray.setImage(createTrayIcon(iconColor))
  tray.setContextMenu(buildMenu(processManager, getWindow))

  // Tooltip: count all active models (running + standby)
  let totalRunning = 0
  let totalStandby = 0
  try {
    const sessions = db.getSessions()
    const runningSessions = sessions.filter((s: any) => s.status === 'running')
    const standbySessions = sessions.filter((s: any) => s.status === 'standby')
    const pmRunning = processes.filter(p => p.status === 'running')
    const sessionOnly = runningSessions.filter((s: any) => !pmRunning.some(p => p.port === s.port))
    totalRunning = pmRunning.length + sessionOnly.length
    totalStandby = standbySessions.length
  } catch {}
  const total = totalRunning + totalStandby
  tray.setToolTip(
    total > 0
      ? `vMLX — ${totalRunning} running${totalStandby > 0 ? `, ${totalStandby} sleeping` : ''}`
      : 'vMLX — No models loaded'
  )
}

/**
 * Create the system tray.
 */
export function createTray(
  processManager: ProcessManager,
  getWindow: () => BrowserWindow | null,
): Tray {
  if (tray) return tray

  tray = new Tray(createTrayIcon('gray'))
  tray.setToolTip('vMLX — No models loaded')
  tray.setContextMenu(buildMenu(processManager, getWindow))

  // macOS: clicking tray icon opens the context menu automatically (setContextMenu).
  // No 'click' handler — adding one causes BOTH the menu AND window to open on every click.
  // "Open vMLX Window" menu item handles showing the window explicitly.

  // Listen for process state changes to update tray (store refs for cleanup)
  boundProcessManager = processManager
  boundListeners.length = 0
  const events = ['process:spawn', 'process:ready', 'process:exit', 'process:killed', 'process:evicted', 'process:pinChanged']
  for (const event of events) {
    const fn = () => rebuildMenu(processManager, getWindow)
    processManager.on(event, fn)
    boundListeners.push({ event, fn })
  }

  // Also listen for SessionManager events (sessions started from Server/Image tabs)
  const sessionRebuild = () => rebuildMenu(processManager, getWindow)
  for (const event of ['session:created', 'session:starting', 'session:ready', 'session:stopped', 'session:error', 'session:deleted', 'session:standby']) {
    sessionManager.on(event, sessionRebuild)
    boundListeners.push({ event, fn: sessionRebuild })
  }

  // Track session memory from health events — only rebuild tray when memory changes
  const sessionHealthFn = (data: any) => {
    if (data.memory?.active_mb != null) {
      const prev = sessionMemoryMB.get(data.sessionId) || 0
      const curr = Math.round(data.memory.active_mb)
      if (Math.abs(curr - prev) >= 10) {  // Only rebuild if change >= 10 MB
        sessionMemoryMB.set(data.sessionId, curr)
        rebuildMenu(processManager, getWindow)
      } else if (prev === 0 && curr > 0) {
        sessionMemoryMB.set(data.sessionId, curr)
        rebuildMenu(processManager, getWindow)
      }
    }
  }
  sessionManager.on('session:health', sessionHealthFn)
  boundListeners.push({ event: 'session:health', fn: sessionHealthFn })

  // Clean up session memory tracking when sessions stop or are deleted
  const sessionCleanupFn = (data: any) => {
    if (data?.sessionId) sessionMemoryMB.delete(data.sessionId)
  }
  sessionManager.on('session:stopped', sessionCleanupFn)
  sessionManager.on('session:deleted', sessionCleanupFn)
  boundListeners.push({ event: 'session:stopped', fn: sessionCleanupFn })
  boundListeners.push({ event: 'session:deleted', fn: sessionCleanupFn })

  return tray
}

/**
 * Destroy the tray and clean up event listeners.
 */
export function destroyTray(): void {
  // Remove ProcessManager listeners to prevent leaks on recreate
  if (boundProcessManager) {
    for (const { event, fn } of boundListeners) {
      // Process events go to ProcessManager, session events go to SessionManager
      if (event.startsWith('session:')) {
        sessionManager.off(event, fn)
      } else {
        boundProcessManager.off(event, fn)
      }
    }
    boundListeners.length = 0
    boundProcessManager = null
  }
  sessionMemoryMB.clear()
  if (tray) {
    tray.destroy()
    tray = null
  }
}

/**
 * Check if tray exists.
 */
export function hasTray(): boolean {
  return tray !== null
}
