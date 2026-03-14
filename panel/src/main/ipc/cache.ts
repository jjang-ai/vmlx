import { ipcMain } from 'electron'
import { resolveBaseUrl, getAuthHeaders } from './utils'

/**
 * Cache management IPC handlers.
 * Proxies to the vmlx-engine server's /v1/cache/* endpoints.
 */

export function registerCacheHandlers(): void {
  ipcMain.handle('cache:stats', async (_, endpoint?: { host: string; port: number }, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    const res = await fetch(`${baseUrl}/v1/cache/stats`, { headers: authHeaders, signal: AbortSignal.timeout(30000) })
    if (!res.ok) throw new Error(`Cache stats failed: ${res.status}`)
    return await res.json()
  })

  ipcMain.handle('cache:entries', async (_, endpoint?: { host: string; port: number }, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    const res = await fetch(`${baseUrl}/v1/cache/entries`, { headers: authHeaders, signal: AbortSignal.timeout(30000) })
    if (!res.ok) throw new Error(`Cache entries failed: ${res.status}`)
    return await res.json()
  })

  ipcMain.handle('cache:warm', async (_, prompts: string[], endpoint?: { host: string; port: number }, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    const res = await fetch(`${baseUrl}/v1/cache/warm`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders },
      body: JSON.stringify({ prompts }),
      signal: AbortSignal.timeout(60000)
    })
    if (!res.ok) throw new Error(`Cache warm failed: ${res.status}`)
    return await res.json()
  })

  ipcMain.handle('cache:clear', async (_, cacheType: string, endpoint?: { host: string; port: number }, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    const res = await fetch(`${baseUrl}/v1/cache?type=${encodeURIComponent(cacheType)}`, {
      method: 'DELETE',
      headers: authHeaders,
      signal: AbortSignal.timeout(10000)
    })
    if (!res.ok) throw new Error(`Cache clear failed: ${res.status}`)
    return await res.json()
  })
}
