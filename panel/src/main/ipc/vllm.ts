import { ipcMain, BrowserWindow } from 'electron'
import {
  checkVllmInstallation,
  installVllmStreaming,
  cancelInstall,
  detectAvailableInstallers,
  checkEngineVersion
} from '../vllm-manager'

export function registerVllmHandlers(getWindow: () => BrowserWindow | null): void {
  // Check if vmlx-engine is installed
  ipcMain.handle('vllm:check-installation', async () => {
    return await checkVllmInstallation()
  })

  // Detect available installers (uv, pip)
  ipcMain.handle('vllm:detect-installers', async () => {
    return await detectAvailableInstallers()
  })

  // Check engine version (bundled only)
  ipcMain.handle('vllm:check-engine-version', async () => {
    return checkEngineVersion()
  })

  // Streaming install — sends log events to renderer
  ipcMain.handle('vllm:install-streaming', async (_, method: 'uv' | 'pip' | 'bundled-update', action: 'install' | 'upgrade', installerPath?: string) => {
    return new Promise<{ success: boolean; error?: string }>((resolve) => {
      installVllmStreaming(
        method,
        action,
        installerPath,
        (data: string) => {
          try {
            const win = getWindow()
            if (win && !win.isDestroyed()) {
              win.webContents.send('vllm:install-log', { data })
            }
          } catch (_) {}
        },
        (result) => {
          try {
            const win = getWindow()
            if (win && !win.isDestroyed()) {
              win.webContents.send('vllm:install-complete', result)
            }
          } catch (_) {}
          resolve(result)
        }
      )
    })
  })

  // Cancel active install
  ipcMain.handle('vllm:cancel-install', async () => {
    return { success: cancelInstall() }
  })
}
