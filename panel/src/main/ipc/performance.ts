import { ipcMain } from 'electron'
import { resolveUrl } from '../sessions'

/**
 * Performance IPC handlers.
 * Proxies /health endpoint through main process for proper mDNS resolution.
 */

export function registerPerformanceHandlers(): void {
  ipcMain.handle('performance:health', async (_, endpoint: { host: string; port: number }) => {
    try {
      const baseUrl = await resolveUrl(`http://${endpoint.host}:${endpoint.port}`)
      const res = await fetch(`${baseUrl}/health`, {
        signal: AbortSignal.timeout(5000)
      })
      if (!res.ok) {
        throw new Error(`Health check failed: ${res.status}`)
      }
      return await res.json()
    } catch (err: any) {
      throw new Error(`Health endpoint unreachable: ${err.message || 'unknown error'}`)
    }
  })
}
