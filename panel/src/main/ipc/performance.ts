import { ipcMain } from 'electron'
import { resolveUrl, connectHost } from '../sessions'

/**
 * Performance IPC handlers.
 * Proxies /health endpoint through main process for proper mDNS resolution.
 */

function isExpectedPerformanceEndpointDisconnectError(err: unknown): boolean {
  const anyErr = err as any
  const code = anyErr?.code
  const message = String(anyErr?.message || anyErr || '').toLowerCase()
  const cause = anyErr?.cause
  const wrappedDisconnects = [
    cause,
    anyErr?.reason,
    anyErr?.error,
    anyErr?.detail,
  ].filter(Boolean)
  const nestedErrors = Array.isArray(anyErr?.errors) ? anyErr.errors : []

  return (
    code === 'EPIPE' ||
    code === 'ECONNRESET' ||
    code === 'ERR_STREAM_DESTROYED' ||
    code === 'ERR_STREAM_WRITE_AFTER_END' ||
    /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message) ||
    wrappedDisconnects.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested)) ||
    nestedErrors.some((nested) => isExpectedPerformanceEndpointDisconnectError(nested))
  )
}

export function registerPerformanceHandlers(): void {
  ipcMain.handle('performance:health', async (_, endpoint: { host: string; port: number }) => {
    try {
      const baseUrl = await resolveUrl(`http://${connectHost(endpoint.host)}:${endpoint.port}`)
      const res = await fetch(`${baseUrl}/health`, {
        signal: AbortSignal.timeout(30000)
      })
      if (!res.ok) {
        throw new Error(`Health check failed: ${res.status}`)
      }
      return await res.json()
    } catch (err: any) {
      if (isExpectedPerformanceEndpointDisconnectError(err)) {
        throw new Error('Performance health connection lost. The model server may have stopped or restarted; retry after the session is healthy.')
      }
      throw new Error(`Health endpoint unreachable: ${err.message || 'unknown error'}`)
    }
  })
}
