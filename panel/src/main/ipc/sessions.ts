import { ipcMain, BrowserWindow, dialog, app as electronApp } from 'electron'
import { readFileSync } from 'fs'
import { sessionManager } from '../sessions'
import type { ServerConfig } from '../server'
import { abortByEndpoint } from './chat'
import { validateMcpConfigText } from '../../shared/mcpConfigValidation'
import { defaultManagedMcpConfigDir, importMcpConfigFile } from '../mcp-config-store'

function connectHost(host: string): string {
  return host === '0.0.0.0' ? '127.0.0.1' : host
}

function sessionAuthHeaders(config: Partial<ServerConfig>): Record<string, string> {
  const key = String(config.apiKey || '').trim()
  return key ? { Authorization: `Bearer ${key}` } : {}
}

function isExpectedSessionLifecycleDisconnectError(error: unknown): boolean {
  const err = error as NodeJS.ErrnoException | undefined
  const code = String(err?.code || '')
  const message = String((err as Error)?.message || error || '')
  const cause = (err as any)?.cause
  const wrappedDisconnects = [
    cause,
    (err as any)?.reason,
    (err as any)?.error,
    (err as any)?.detail,
  ].filter(Boolean)
  const nestedErrors = Array.isArray((err as any)?.errors) ? (err as any).errors : []
  return (
    code === 'EPIPE' ||
    code === 'ECONNRESET' ||
    code === 'ERR_STREAM_DESTROYED' ||
    code === 'ERR_STREAM_WRITE_AFTER_END' ||
    /EPIPE|write EPIPE|broken pipe|socket hang up|connection reset|premature close|stream.*destroyed|write after end/i.test(message) ||
    wrappedDisconnects.some((nested) => isExpectedSessionLifecycleDisconnectError(nested)) ||
    nestedErrors.some((nested) => isExpectedSessionLifecycleDisconnectError(nested))
  )
}

function formatSessionLifecycleError(error: unknown): string {
  if (isExpectedSessionLifecycleDisconnectError(error)) {
    return 'Server connection lost. The model server may have stopped or restarted. Try restarting the session.'
  }
  return String((error as Error)?.message || error || 'Unknown error')
}

async function fetchSessionJson(sessionId: string, path: string): Promise<any> {
  const session = sessionManager.getSession(sessionId)
  if (!session) throw new Error(`Session ${sessionId} not found`)
  let config: Partial<ServerConfig> = {}
  try {
    config = JSON.parse(session.config || '{}')
  } catch {
    config = {}
  }
  const url = `http://${connectHost(session.host)}:${session.port}${path}`
  const response = await fetch(url, {
    headers: sessionAuthHeaders(config),
    signal: AbortSignal.timeout(5000),
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`${path} failed with ${response.status}${text ? `: ${text.slice(0, 200)}` : ''}`)
  }
  return response.json()
}

function validateMcpConfigFile(filePath: string): any {
  if (!filePath || !filePath.trim()) throw new Error('MCP config path is empty')
  const raw = readFileSync(filePath, 'utf8')
  return validateMcpConfigText(raw, filePath)
}

async function pickMcpConfigFile(): Promise<string | null> {
  const result = await dialog.showOpenDialog({
    title: 'Select mcp.json',
    properties: ['openFile'],
    filters: [
      { name: 'MCP config', extensions: ['json', 'jsonc', 'yaml', 'yml'] },
      { name: 'All files', extensions: ['*'] },
    ],
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
}

const SESSION_EVENTS = [
  'session:created',
  'session:starting',
  'session:ready',
  'session:stopped',
  'session:error',
  'session:health',
  'session:log',
  'session:deleted',
  'session:standby',
  'session:loadProgress'
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
        return { success: false, error: formatSessionLifecycleError(error) }
      }
    })

    ipcMain.handle('sessions:start', async (_, sessionId: string) => {
      try {
        await sessionManager.startSession(sessionId)
        return { success: true }
      } catch (error) {
        return { success: false, error: formatSessionLifecycleError(error) }
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
        return { success: false, error: formatSessionLifecycleError(error) }
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

    ipcMain.handle('sessions:createRemote', async (_, params: { remoteUrl: string; remoteApiKey?: string; remoteModel: string; remoteOrganization?: string; capabilityModelPath?: string }) => {
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

    ipcMain.handle('sessions:browseMcpConfig', async () => {
      const filePath = await pickMcpConfigFile()
      return {
        canceled: !filePath,
        filePath: filePath || undefined,
      }
    })

    ipcMain.handle('sessions:importMcpConfig', async (_, filePath?: string) => {
      try {
        const sourcePath = filePath?.trim() || await pickMcpConfigFile()
        if (!sourcePath) return { success: false, canceled: true, servers: [] }
        return importMcpConfigFile(sourcePath, {
          storeDir: defaultManagedMcpConfigDir(electronApp.getPath('userData')),
        })
      } catch (error) {
        return { success: false, error: (error as Error).message, servers: [] }
      }
    })

    ipcMain.handle('sessions:validateMcpConfig', async (_, filePath: string) => {
      try {
        return validateMcpConfigFile(filePath)
      } catch (error) {
        return { success: false, error: (error as Error).message, servers: [] }
      }
    })

    ipcMain.handle('sessions:softSleep', async (_, sessionId: string) => {
      try {
        return await sessionManager.softSleep(sessionId)
      } catch (error) {
        return { success: false, error: formatSessionLifecycleError(error) }
      }
    })

    ipcMain.handle('sessions:deepSleep', async (_, sessionId: string) => {
      try {
        return await sessionManager.deepSleep(sessionId)
      } catch (error) {
        return { success: false, error: formatSessionLifecycleError(error) }
      }
    })

    ipcMain.handle('sessions:wake', async (_, sessionId: string) => {
      try {
        return await sessionManager.wakeSession(sessionId)
      } catch (error) {
        return { success: false, error: formatSessionLifecycleError(error) }
      }
    })

    ipcMain.handle('sessions:touch', async (_, sessionId: string) => {
      sessionManager.touchSession(sessionId)
      return { success: true }
    })

    ipcMain.handle('sessions:mcpStatus', async (_, sessionId: string) => {
      try {
        const [tools, servers] = await Promise.all([
          fetchSessionJson(sessionId, '/v1/mcp/tools'),
          fetchSessionJson(sessionId, '/v1/mcp/servers'),
        ])
        return {
          success: true,
          tools: Array.isArray(tools?.tools) ? tools.tools : [],
          servers: Array.isArray(servers?.servers) ? servers.servers : [],
          count: typeof tools?.count === 'number' ? tools.count : 0,
        }
      } catch (error) {
        return { success: false, error: (error as Error).message, tools: [], servers: [] }
      }
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
          const payload = eventName === 'session:error'
            ? { ...data, error: formatSessionLifecycleError(data.error) }
            : data
          win.webContents.send(eventName, payload)
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
