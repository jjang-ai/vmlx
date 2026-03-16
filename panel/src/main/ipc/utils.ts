import { sessionManager, resolveUrl, connectHost } from '../sessions'
import { db } from '../database'

/** Resolve the base URL for a session endpoint, with fallback to detection. */
export async function resolveBaseUrl(endpoint?: { host: string; port: number }): Promise<string> {
  if (endpoint) {
    return await resolveUrl(`http://${connectHost(endpoint.host)}:${endpoint.port}`)
  }
  const processes = await sessionManager.detect()
  const healthy = processes.find(p => p.healthy)
  if (healthy) return `http://127.0.0.1:${healthy.port}`
  return 'http://127.0.0.1:8000'
}

/** Build auth headers from session config (apiKey or remote credentials). */
export function getAuthHeaders(sessionId?: string): Record<string, string> {
  if (!sessionId) return {}
  try {
    const session = db.getSession(sessionId)
    if (!session) return {}
    const config = JSON.parse(session.config)
    if (session.type === 'remote' && session.remoteApiKey) {
      const h: Record<string, string> = { 'Authorization': `Bearer ${session.remoteApiKey}` }
      if (session.remoteOrganization) h['OpenAI-Organization'] = session.remoteOrganization
      return h
    } else if (config.apiKey) {
      return { 'Authorization': `Bearer ${config.apiKey}` }
    }
  } catch (_) {}
  return {}
}
