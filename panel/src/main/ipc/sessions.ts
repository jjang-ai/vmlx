import { ipcMain, BrowserWindow } from 'electron'
import { sessionManager } from '../sessions'
import type { ServerConfig } from '../server'
import { abortByEndpoint } from './chat'

const SESSION_EVENTS = [
  'session:created',
  'session:starting',
  'session:ready',
  'session:stopped',
  'session:error',
  'session:health',
  'session:log',
  'session:deleted'
]

let handlersRegistered = false

export function registerSessionHandlers(getWindow: () => BrowserWindow | null): void {
  if (!handlersRegistered) {
    ipcMain.handle('sessions:list', async () => {
      try {
        return sessionManager.getSessions()
      } catch (error) {
        console.error('[SESSION] Failed to list sessions:', error)
        return []
      }
    })

    ipcMain.handle('sessions:get', async (_, id: string) => {
      try {
        return sessionManager.getSession(id)
      } catch (error) {
        console.error('[SESSION] Failed to get session:', error)
        return null
      }
    })

    ipcMain.handle('sessions:create', async (_, modelPath: string, config: Partial<ServerConfig>) => {
      try {
        const session = await sessionManager.createSession(modelPath, config)
        return { success: true, session }
      } catch (error) {
        return { success: false, error: (error as Error).message }
      }
    })

    ipcMain.handle('sessions:start', async (_, sessionId: string) => {
      try {
        await sessionManager.startSession(sessionId)
        return { success: true }
      } catch (error) {
        return { success: false, error: (error as Error).message }
      }
    })

    ipcMain.handle('sessions:stop', async (_, sessionId: string) => {
      try {
        // Abort any active chat requests on this session's endpoint before killing the server
        const session = sessionManager.getSession(sessionId)
        if (session) {
          const aborted = abortByEndpoint(session.host, session.port)
          if (aborted > 0) console.log(`[SESSION] Aborted ${aborted} active chat(s) for session ${sessionId}`)
        }
        await sessionManager.stopSession(sessionId)
        return { success: true }
      } catch (error) {
        return { success: false, error: (error as Error).message }
      }
    })

    ipcMain.handle('sessions:delete', async (_, sessionId: string) => {
      try {
        await sessionManager.deleteSession(sessionId)
        return { success: true }
      } catch (error) {
        return { success: false, error: (error as Error).message }
      }
    })

    ipcMain.handle('sessions:createRemote', async (_, params: { remoteUrl: string; remoteApiKey?: string; remoteModel: string; remoteOrganization?: string }) => {
      try {
        const session = await sessionManager.createRemoteSession(params)
        return { success: true, session }
      } catch (error) {
        return { success: false, error: (error as Error).message }
      }
    })

    ipcMain.handle('sessions:detect', async () => {
      try {
        return await sessionManager.detectAndAdoptAll()
      } catch (error) {
        console.error('[SESSION] Failed to detect sessions:', error)
        return []
      }
    })

    ipcMain.handle('sessions:update', async (_, sessionId: string, config: Partial<ServerConfig>) => {
      try {
        const result = await sessionManager.updateSessionConfig(sessionId, config)
        return { success: true, ...result }
      } catch (error) {
        return { success: false, error: (error as Error).message }
      }
    })

    ipcMain.handle('sessions:getLogs', async (_, sessionId: string) => {
      try {
        return sessionManager.getLogs(sessionId)
      } catch (error) {
        console.error('[SESSION] Failed to get logs:', error)
        return []
      }
    })

    ipcMain.handle('sessions:clearLogs', async (_, sessionId: string) => {
      sessionManager.clearLogs(sessionId)
      return { success: true }
    })

    handlersRegistered = true
  }

  // Remove old event listeners to prevent accumulation on window recreation
  for (const eventName of SESSION_EVENTS) {
    sessionManager.removeAllListeners(eventName)
  }
  sessionManager.removeAllListeners('session:abortInference')

  // Forward session events to renderer
  for (const eventName of SESSION_EVENTS) {
    sessionManager.on(eventName, (data: any) => {
      try {
        const win = getWindow()
        if (win && !win.isDestroyed()) {
          win.webContents.send(eventName, data)
        }
      } catch (_) {}
    })
  }

  // When a session goes down (health monitor), abort any active inference on that endpoint.
  // This prevents orphaned SSE streams that block new requests after reconnect.
  sessionManager.on('session:abortInference', (data: { sessionId: string; host: string; port: number }) => {
    const aborted = abortByEndpoint(data.host, data.port)
    if (aborted > 0) console.log(`[SESSION] Aborted ${aborted} active chat(s) for downed session ${data.sessionId}`)
  })
}
