import { ipcMain } from 'electron'
import { resolveBaseUrl, getAuthHeaders } from './utils'

/**
 * Embeddings IPC handlers.
 * Proxies to the vllm-mlx server's /v1/embeddings endpoint.
 */

export function registerEmbeddingHandlers(): void {
  ipcMain.handle('embeddings:embed', async (_, texts: string[], endpoint: { host: string; port: number }, model?: string, sessionId?: string) => {
    const baseUrl = await resolveBaseUrl(endpoint)
    const authHeaders = getAuthHeaders(sessionId)
    const input = texts.length === 1 ? texts[0] : texts
    const res = await fetch(`${baseUrl}/v1/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders },
      body: JSON.stringify({ model: model || 'default', input }),
      signal: AbortSignal.timeout(60000)
    })
    if (!res.ok) {
      const body = await res.text().catch(() => '')
      throw new Error(`Embeddings failed: ${res.status} ${body}`)
    }
    return await res.json()
  })
}
