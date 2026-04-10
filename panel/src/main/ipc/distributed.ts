import { ipcMain, BrowserWindow } from 'electron'
import { sessionManager } from '../sessions'

let handlersRegistered = false

function findRunningSession(sessionId?: string) {
  if (sessionId) {
    return sessionManager.getSession(sessionId)
  }
  // Find first running session
  const sessions = sessionManager.getSessions()
  return sessions.find((s: any) => s.status === 'running' || s.status === 'ready')
}

export function registerDistributedHandlers(getWindow: () => BrowserWindow | null): void {
  if (handlersRegistered) return
  handlersRegistered = true

  // Discover nodes: trigger scan via cluster API
  ipcMain.handle('distributed:discover', async (_, sessionId?: string) => {
    try {
      const session = findRunningSession(sessionId)
      if (!session) return { success: false, error: 'No active session' }

      const port = session.port
      if (!port) return { success: false, error: 'Session has no port' }

      const resp = await fetch(`http://127.0.0.1:${port}/v1/cluster/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: AbortSignal.timeout(15000),
      })
      if (!resp.ok) return { success: false, error: `HTTP ${resp.status}` }
      const data = await resp.json()
      return { success: true, nodes: data.nodes || [] }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })

  // Get cluster status
  ipcMain.handle('distributed:status', async (_, sessionId?: string) => {
    try {
      const session = findRunningSession(sessionId)
      if (!session) return { success: false, error: 'No active session' }

      const port = session.port
      if (!port) return { success: false, error: 'Session has no port' }

      const resp = await fetch(`http://127.0.0.1:${port}/v1/cluster/status`, {
        signal: AbortSignal.timeout(5000),
      })
      if (!resp.ok) return { success: false, error: `HTTP ${resp.status}` }
      const data = await resp.json()
      return { success: true, ...data }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })

  // Get nodes list
  ipcMain.handle('distributed:nodes', async (_, sessionId?: string) => {
    try {
      const session = findRunningSession(sessionId)
      if (!session) return { success: false, error: 'No active session' }

      const port = session.port
      if (!port) return { success: false, error: 'Session has no port' }

      const resp = await fetch(`http://127.0.0.1:${port}/v1/cluster/nodes`, {
        signal: AbortSignal.timeout(5000),
      })
      if (!resp.ok) return { success: false, error: `HTTP ${resp.status}` }
      const data = await resp.json()
      return { success: true, nodes: data.nodes || [] }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })

  // Add node manually by IP:port
  ipcMain.handle('distributed:addNode', async (_, address: string, port?: number, sessionId?: string) => {
    try {
      const session = findRunningSession(sessionId)
      if (!session) return { success: false, error: 'No active session' }

      const sessionPort = session.port
      if (!sessionPort) return { success: false, error: 'Session has no port' }

      const resp = await fetch(`http://127.0.0.1:${sessionPort}/v1/cluster/nodes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address, port: port || 9100 }),
        signal: AbortSignal.timeout(15000),
      })
      if (!resp.ok) return { success: false, error: `HTTP ${resp.status}` }
      const data = await resp.json()
      return { success: true, node: data.node }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })

  // Remove a node
  ipcMain.handle('distributed:removeNode', async (_, nodeId: string, sessionId?: string) => {
    try {
      const session = findRunningSession(sessionId)
      if (!session) return { success: false, error: 'No active session' }

      const sessionPort = session.port
      if (!sessionPort) return { success: false, error: 'Session has no port' }

      const resp = await fetch(`http://127.0.0.1:${sessionPort}/v1/cluster/nodes/${nodeId}`, {
        method: 'DELETE',
        signal: AbortSignal.timeout(5000),
      })
      if (!resp.ok) return { success: false, error: `HTTP ${resp.status}` }
      return { success: true }
    } catch (error) {
      return { success: false, error: (error as Error).message }
    }
  })
}
