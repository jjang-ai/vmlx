import { app, BrowserWindow, ipcMain, dialog, shell, session } from 'electron'
import { join } from 'path'
import { readFileSync } from 'fs'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { registerSessionHandlers } from './ipc/sessions'
import { registerChatHandlers } from './ipc/chat'
import { registerModelHandlers, killActiveDownload } from './ipc/models'
import { registerEngineHandlers } from './ipc/engine'
import { registerAudioHandlers } from './ipc/audio'
import { registerCacheHandlers } from './ipc/cache'
import { registerBenchmarkHandlers } from './ipc/benchmark'
import { registerEmbeddingHandlers } from './ipc/embeddings'
import { registerExportHandlers } from './ipc/export'
import { registerPerformanceHandlers } from './ipc/performance'
import { registerDeveloperHandlers, killActiveOperation } from './ipc/developer'
import { registerCodingToolHandlers } from './ipc/coding-tools'
import { sessionManager } from './sessions'
import { apiGateway } from './api-gateway'
import { db } from './database'
import { checkEngineVersion, installEngineStreaming } from './engine-manager'
import { checkForUpdates } from './update-checker'
import { ProcessManager } from './process-manager'
import { createTray, destroyTray, hasTray } from './tray'
import { startMemoryEnforcer, stopMemoryEnforcer } from './memory-enforcer'
import { registerModelSettingsHandlers } from './db/model-settings'
import { registerImageHandlers } from './ipc/image'

let mainWindow: BrowserWindow | null = null
let downloadWindow: BrowserWindow | null = null
let handlersRegistered = false
let isQuitting = false
const processManager = new ProcessManager()

// Global crash handlers — prevent unhandled errors from silently crashing the app
process.on('uncaughtException', (error) => {
  console.error('[CRASH] Uncaught exception:', error)
  // Kill all Python processes to prevent orphans
  try { sessionManager.stopAll().catch(() => { }) } catch (_) { }
  try { processManager.killAll().catch(() => { }) } catch (_) { }
  try {
    dialog.showErrorBox(
      'Unexpected Error',
      `vMLX encountered an error:\n\n${error.message}\n\nThe app will now exit.`
    )
  } catch (_) { /* dialog may fail if app is in bad state */ }
  // Continuing after uncaught exception risks undefined behavior.
  process.exit(1)
})

process.on('unhandledRejection', (reason) => {
  console.error('[CRASH] Unhandled promise rejection:', reason)
})

// Prevent multiple instances — second instance would corrupt SQLite
const gotTheLock = app.requestSingleInstanceLock()
if (!gotTheLock) {
  app.quit()
} else {
  app.on('second-instance', () => {
    // Focus existing window when user tries to open a second instance
    if (mainWindow && !mainWindow.isDestroyed()) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
    }
  })
}

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    show: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false
    },
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 15, y: 15 }
  })

  // Register IPC handlers only once — pass getter to avoid stale window references on macOS recreation
  if (!handlersRegistered) {
    registerSessionHandlers(() => mainWindow)
    registerChatHandlers(() => mainWindow)
    registerModelHandlers()
    registerEngineHandlers(() => mainWindow)
    registerAudioHandlers()
    registerCacheHandlers()
    registerBenchmarkHandlers(() => mainWindow)
    registerEmbeddingHandlers()
    registerExportHandlers()
    registerPerformanceHandlers()
    registerDeveloperHandlers(() => mainWindow)
    registerModelSettingsHandlers()
    registerImageHandlers()
    registerCodingToolHandlers()

    // Download window — a real separate window for showing download progress
    ipcMain.handle('downloads:openWindow', () => {
      if (downloadWindow && !downloadWindow.isDestroyed()) {
        downloadWindow.focus()
        return
      }
      downloadWindow = new BrowserWindow({
        width: 500,
        height: 400,
        title: 'Downloads',
        parent: mainWindow || undefined,
        minimizable: true,
        maximizable: false,
        resizable: true,
        webPreferences: {
          preload: join(__dirname, '../preload/index.js'),
          sandbox: false,
          contextIsolation: true,
        },
      })
      // Load the same renderer with query param to show downloads view
      if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
        downloadWindow.loadURL(process.env['ELECTRON_RENDERER_URL'] + '?view=downloads')
      } else {
        downloadWindow.loadFile(join(__dirname, '../../out/renderer/index.html'), { query: { view: 'downloads' } })
      }
      downloadWindow.on('closed', () => { downloadWindow = null })
    })

    // Folder picker for built-in tools working directory
    ipcMain.handle('dialog:openDirectory', async () => {
      const result = await dialog.showOpenDialog({
        properties: ['openDirectory', 'createDirectory'],
        securityScopedBookmarks: true,
        title: 'Select Working Directory'
      })

      if (!result.canceled && result.filePaths.length > 0 && result.bookmarks && result.bookmarks.length > 0) {
        db.saveBookmark(result.filePaths[0], result.bookmarks[0])
      }

      return result
    })

    // Image picker for vision/multimodal chat — reads files and returns base64 data URLs
    ipcMain.handle('dialog:pickImages', async () => {
      const result = await dialog.showOpenDialog({
        properties: ['openFile', 'multiSelections'],
        title: 'Attach Images',
        filters: [
          { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff', 'tif', 'heic', 'heif', 'avif'] }
        ]
      })
      if (result.canceled || result.filePaths.length === 0) return []
      return result.filePaths.map(fp => {
        const ext = fp.split('.').pop()?.toLowerCase() || 'png'
        const mimeMap: Record<string, string> = {
          jpg: 'image/jpeg', jpeg: 'image/jpeg',
          gif: 'image/gif', webp: 'image/webp',
          bmp: 'image/bmp', tiff: 'image/tiff', tif: 'image/tiff',
          heic: 'image/heic', heif: 'image/heif',
          avif: 'image/avif',
        }
        const mime = mimeMap[ext] || 'image/png'
        const data = readFileSync(fp).toString('base64')
        const name = fp.split('/').pop() || 'image'
        return { dataUrl: `data:${mime};base64,${data}`, name }
      })
    })

    // App version
    ipcMain.handle('app:getVersion', () => app.getVersion())

    // App-level settings (API keys, preferences)
    ipcMain.handle('settings:get', (_e, key: string) => {
      return db.getSetting(key) ?? null
    })
    ipcMain.handle('settings:set', (_e, key: string, value: string) => {
      db.setSetting(key, value)
      return { success: true }
    })
    ipcMain.handle('settings:delete', (_e, key: string) => {
      db.deleteSetting(key)
      return { success: true }
    })

    // Auto-detect inference mode on first launch based on system RAM.
    // Casual (≤48GB) or Expert (>48GB). User can override anytime.
    const existingMode = db.getSetting('inference_mode')
    if (!existingMode) {
      const totalMemGB = Math.round(require('os').totalmem() / (1024 ** 3))
      const autoMode = totalMemGB <= 48 ? 'casual' : 'expert'
      db.setSetting('inference_mode', autoMode)
      console.log(`[MODE] Auto-detected: ${autoMode} (${totalMemGB}GB RAM)`)
    }

    // API Gateway
    ipcMain.handle('gateway:status', () => ({
      running: apiGateway.running,
      port: apiGateway.activePort,
    }))
    ipcMain.handle('gateway:start', async (_e, port?: number) => {
      await apiGateway.start(port)
      return { running: true, port: apiGateway.activePort }
    })
    ipcMain.handle('gateway:stop', async () => {
      await apiGateway.stop()
      return { running: false }
    })
    ipcMain.handle('gateway:restart', async (_e, port: number) => {
      db.setSetting('gateway_port', String(port))
      await apiGateway.restart(port)
      return { running: true, port }
    })

    // Prompt templates
    ipcMain.handle('templates:list', () => db.getPromptTemplates())
    ipcMain.handle('templates:save', (_e, template: { id: string; name: string; content: string; category: string }) => {
      db.savePromptTemplate(template)
      return { success: true }
    })
    ipcMain.handle('templates:delete', (_e, id: string) => {
      db.deletePromptTemplate(id)
      return { success: true }
    })

    handlersRegistered = true
  }

  // Content Security Policy — hardens renderer against XSS
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self';" +
          " script-src 'self';" +
          " style-src 'self' 'unsafe-inline' https://fonts.googleapis.com;" +
          " font-src 'self' https://fonts.gstatic.com;" +
          " img-src 'self' data: blob: http://127.0.0.1:* http://localhost:* https://huggingface.co https://*.huggingface.co https://raw.githubusercontent.com https://mlx.studio https://*.mlx.studio;" +
          " connect-src 'self' http://127.0.0.1:* http://localhost:* https://huggingface.co https://*.huggingface.co;" +
          " media-src 'self' blob:;"
        ]
      }
    })
  })

  // Close-to-tray: hide window instead of destroying when tray is active
  mainWindow.on('close', (e) => {
    if (isQuitting) return  // Let quit proceed
    const closeToTray = db.getSetting('tray_close_to_tray')
    if (hasTray() && closeToTray === '1') {
      e.preventDefault()
      mainWindow?.hide()
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    const { url } = details
    if (url.startsWith('https://') || url.startsWith('http://')) {
      shell.openExternal(url)
    }
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../../out/renderer/index.html'))
  }
}

// App ready
app.whenReady().then(async () => {
  electronApp.setAppUserModelId('net.vmlx.app')

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // Check if bundled engine needs updating BEFORE creating window —
  // prevents race where SetupScreen checks installation mid-pip-reinstall
  try {
    const versionInfo = checkEngineVersion()
    if (versionInfo.needsUpdate) {
      console.log(`[STARTUP] Engine update needed: ${versionInfo.current} -> ${versionInfo.bundled}`)
      await Promise.race([
        new Promise<void>((resolve) => {
          installEngineStreaming(
            'bundled-update',
            'install',
            undefined,
            (data) => console.log('[ENGINE UPDATE]', data.trimEnd()),
            (result) => {
              if (result.success) {
                console.log('[STARTUP] Engine updated successfully')
              } else {
                console.error('[STARTUP] Engine update failed:', result.error)
              }
              resolve()
            }
          )
        }),
        new Promise<void>((resolve) => {
          setTimeout(() => {
            console.error('[STARTUP] Engine update timed out after 30s, continuing with existing version')
            resolve()
          }, 30000)
        })
      ])
    }
  } catch (e) {
    console.error('[STARTUP] Error checking engine version:', e)
  }

  createWindow()

  // Restore macOS App Sandbox bookmarks for external directory access
  try {
    const bookmarks = db.getAllBookmarks()
    console.log(`[STARTUP] Restoring ${bookmarks.length} App Sandbox bookmarks`)
    for (const b of bookmarks) {
      try {
        app.startAccessingSecurityScopedResource(b.bookmark)
      } catch (e) {
        console.error(`[STARTUP] Failed to restore bookmark for ${b.path}`, e)
      }
    }
  } catch (err) {
    console.error('[STARTUP] Error fetching bookmarks from DB', err)
  }

  // Notify user if database was recovered from corruption
  if (db.recoveryBackupPath && mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.once('ready-to-show', () => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        dialog.showMessageBox(mainWindow, {
          type: 'warning',
          title: 'Database Recovered',
          message: 'Your chat database was corrupted and has been recreated.',
          detail: `Your previous data has been backed up to:\n${db.recoveryBackupPath}`,
          buttons: ['OK']
        }).catch(() => { })
      }
    })
  }

  // Check for app updates (non-blocking, fires after 5s delay)
  const appVersion = JSON.parse(readFileSync(join(__dirname, '../../package.json'), 'utf-8')).version
  checkForUpdates(() => mainWindow, appVersion)

  // Detect and adopt existing vmlx-engine processes on startup
  try {
    const adopted = await sessionManager.detectAndAdoptAll()
    if (adopted.length > 0) {
      console.log(`[STARTUP] Adopted ${adopted.length} vmlx-engine process(es):`)
      for (const s of adopted) {
        console.log(`  - ${s.modelName || s.modelPath} on port ${s.port} (PID ${s.pid})`)
      }
    } else {
      console.log('[STARTUP] No existing vmlx-engine processes found')
    }
  } catch (e) {
    console.error('[STARTUP] Error during process detection:', e)
  }

  // Start global health monitor for all sessions
  sessionManager.startGlobalMonitor()

  // Initialize tray if enabled in settings
  const trayEnabled = db.getSetting('tray_enabled')
  if (trayEnabled !== '0') {
    createTray(processManager, () => mainWindow)
  }

  // Start memory enforcer
  startMemoryEnforcer(processManager)

  // Start API gateway (single-port proxy for all models)
  const gatewayEnabled = db.getSetting('gateway_enabled') !== 'false'
  if (gatewayEnabled) {
    const gwPort = parseInt(db.getSetting('gateway_port') || '8080', 10)
    apiGateway.start(gwPort).catch(err => {
      console.error('[gateway] Failed to start:', err.message || err)
    })
  }

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    } else if (mainWindow && !mainWindow.isDestroyed()) {
      // Restore minimized or hidden window when dock icon is clicked
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.show()
      mainWindow.focus()
    }
  })
})

// Kill all vmlx-engine processes on quit — with timeout to prevent hanging
app.on('before-quit', async (e) => {
  if (isQuitting) return
  isQuitting = true
  e.preventDefault()
  try {
    stopMemoryEnforcer()
    destroyTray()
    apiGateway.stop().catch(() => {})
    sessionManager.stopGlobalMonitor()
    killActiveDownload()  // Kill any active download subprocess
    killActiveOperation()  // Kill any active developer tool subprocess
    // B15: Timeout stopAll to prevent app from hanging on quit
    await Promise.race([
      Promise.all([sessionManager.stopAll(), processManager.killAll()]),
      new Promise(resolve => setTimeout(resolve, 15000))
    ])
    console.log('[QUIT] All vmlx-engine processes stopped')
  } catch (err) {
    console.error('[QUIT] Error stopping processes:', err)
  }
  try {
    db.close()
    console.log('[QUIT] Database closed')
  } catch (err) {
    console.error('[QUIT] Error closing database:', err)
  }
  app.exit(0)
})

// Handle SIGTERM/SIGINT (e.g., macOS force-quit, killall) — trigger clean shutdown
for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.on(signal, () => {
    console.log(`[QUIT] Received ${signal} — triggering clean shutdown`)
    if (!isQuitting) app.quit()
  })
}

// Quit when all windows are closed, except on macOS or when tray is active
app.on('window-all-closed', () => {
  // On macOS, keep app alive (standard behavior)
  if (process.platform === 'darwin') return

  // If tray is active and close-to-tray is enabled, keep alive
  const closeToTray = db.getSetting('tray_close_to_tray')
  if (hasTray() && closeToTray !== '0') {
    console.log('[APP] Window closed, keeping alive in tray')
    return
  }

  app.quit()
})
