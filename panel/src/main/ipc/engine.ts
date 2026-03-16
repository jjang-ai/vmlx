import { ipcMain, BrowserWindow } from 'electron'
import {
  checkEngineInstallation,
  installEngineStreaming,
  cancelInstall,
  detectAvailableInstallers,
  checkEngineVersion
} from '../engine-manager'

export function registerEngineHandlers(getWindow: () => BrowserWindow | null): void {
  // Check if vmlx-engine is installed
  ipcMain.handle('engine:check-installation', async () => {
    return await checkEngineInstallation()
  })

  // Detect available installers (uv, pip)
  ipcMain.handle('engine:detect-installers', async () => {
    return await detectAvailableInstallers()
  })

  // Check engine version (bundled only)
  ipcMain.handle('engine:check-engine-version', async () => {
    return checkEngineVersion()
  })

  // Streaming install — sends log events to renderer
  ipcMain.handle('engine:install-streaming', async (_, method: 'uv' | 'pip' | 'bundled-update', action: 'install' | 'upgrade', installerPath?: string) => {
    return new Promise<{ success: boolean; error?: string }>((resolve) => {
      installEngineStreaming(
        method,
        action,
        installerPath,
        (data: string) => {
          try {
            const win = getWindow()
            if (win && !win.isDestroyed()) {
              win.webContents.send('engine:install-log', { data })
            }
          } catch (_) {}
        },
        (result) => {
          try {
            const win = getWindow()
            if (win && !win.isDestroyed()) {
              win.webContents.send('engine:install-complete', result)
            }
          } catch (_) {}
          resolve(result)
        }
      )
    })
  })

  // Cancel active install
  ipcMain.handle('engine:cancel-install', async () => {
    return { success: cancelInstall() }
  })
}
