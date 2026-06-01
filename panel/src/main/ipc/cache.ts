import { ipcMain } from 'electron'
import { resolveBaseUrl, getAuthHeaders } from './utils'

/**
 * Cache management IPC handlers.
 * Proxies to the vmlx-engine server's /v1/cache/* endpoints.
 */

function isExpectedCacheEndpointDisconnectError(err: unknown): boolean {
  const anyErr = err as any
  const code = anyErr?.code
  const message = String(anyErr?.message || anyErr || "").toLowerCase()
  const cause = (err as any)?.cause
  const wrappedDisconnects = [
    cause,
    (err as any)?.reason,
    (err as any)?.error,
    (err as any)?.detail
  ].filter(Boolean)
  const nestedErrors = Array.isArray((err as any)?.errors)
    ? (err as any).errors
    : []

  return (
    code === "EPIPE" ||
    code === "ECONNRESET" ||
    code === "ERR_STREAM_DESTROYED" ||
    code === "ERR_STREAM_WRITE_AFTER_END" ||
    /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message) ||
    wrappedDisconnects.some((nested) => isExpectedCacheEndpointDisconnectError(nested)) ||
    nestedErrors.some((nested) => isExpectedCacheEndpointDisconnectError(nested))
  )
}

async function fetchCacheJson(label: string, url: string, init: RequestInit): Promise<any> {
  let res: Response
  try {
    res = await fetch(url, init)
  } catch (err) {
    if (isExpectedCacheEndpointDisconnectError(err)) {
      throw new Error(`${label} connection lost. The model server may have stopped or restarted; retry after the session is healthy.`)
    }
    throw err
  }
  if (!res.ok) throw new Error(`${label} failed: ${res.status}`)
  return await res.json()
}

export function registerCacheHandlers(): void {
  ipcMain.handle('cache:stats', async (_, endpoint?: { host: string; port: number }, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    return await fetchCacheJson('Cache stats', `${baseUrl}/v1/cache/stats`, {
      headers: authHeaders,
      signal: AbortSignal.timeout(30000)
    })
  })

  ipcMain.handle('cache:entries', async (_, endpoint?: { host: string; port: number }, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    return await fetchCacheJson('Cache entries', `${baseUrl}/v1/cache/entries`, {
      headers: authHeaders,
      signal: AbortSignal.timeout(30000)
    })
  })

  ipcMain.handle('cache:warm', async (_, prompts: string[], endpoint?: { host: string; port: number }, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    return await fetchCacheJson('Cache warm', `${baseUrl}/v1/cache/warm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders },
      body: JSON.stringify({ prompts }),
      signal: AbortSignal.timeout(60000)
    })
  })

  ipcMain.handle('cache:clear', async (_, cacheType: string, endpoint?: { host: string; port: number }, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    return await fetchCacheJson('Cache clear', `${baseUrl}/v1/cache?type=${encodeURIComponent(cacheType)}`, {
      method: 'DELETE',
      headers: authHeaders,
      signal: AbortSignal.timeout(10000)
    })
  })
}
