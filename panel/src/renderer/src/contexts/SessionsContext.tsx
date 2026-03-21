import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'

export interface SessionSummary {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  standbyDepth?: 'soft' | 'deep' | null
  type?: 'local' | 'remote'
  remoteUrl?: string
  config?: string // JSON blob — includes modelType, imageMode, etc.
}

export interface LoadProgress {
  label: string
  progress: number
}

interface SessionsContextValue {
  sessions: SessionSummary[]
  loadingSessions: Set<string>
  loadProgress: Map<string, LoadProgress>
  ensureSessionRunning: (modelPath: string) => Promise<SessionSummary>
  refreshSessions: () => Promise<void>
}

const SessionsContext = createContext<SessionsContextValue>(null!)

export function useSessionsContext() {
  return useContext(SessionsContext)
}

export function SessionsProvider({ children }: { children: React.ReactNode }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [loadingSessions, setLoadingSessions] = useState<Set<string>>(new Set())
  const [loadProgress, setLoadProgress] = useState<Map<string, LoadProgress>>(new Map())
  const sessionsRef = useRef(sessions)
  sessionsRef.current = sessions

  const refreshSessions = useCallback(async () => {
    try {
      const list = await window.api.sessions.list()
      setSessions(list)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    refreshSessions()

    const unsubs = [
      window.api.sessions.onCreated(() => refreshSessions()),
      window.api.sessions.onDeleted(() => refreshSessions()),
      window.api.sessions.onStarting((data: any) => {
        setSessions(prev => prev.map(s => s.id === data.sessionId ? { ...s, status: 'loading' as const } : s))
        setLoadProgress(prev => { const next = new Map(prev); next.delete(data.sessionId); return next })
      }),
      window.api.sessions.onReady((data: any) => {
        setSessions(prev => prev.map(s =>
          s.id === data.sessionId
            ? { ...s, status: 'running' as const, ...(data.pid ? { pid: data.pid } : {}), ...(data.port ? { port: data.port } : {}) }
            : s
        ))
        setLoadingSessions(prev => {
          const next = new Set(prev)
          const session = sessionsRef.current.find(s => s.id === data.sessionId)
          if (session) next.delete(session.modelPath)
          return next
        })
        setLoadProgress(prev => { const next = new Map(prev); next.set(data.sessionId, { label: 'Ready', progress: 100 }); return next })
      }),
      window.api.sessions.onStopped((data: any) => {
        setSessions(prev => prev.map(s => s.id === data.sessionId ? { ...s, status: 'stopped' as const } : s))
        setLoadProgress(prev => { const next = new Map(prev); next.delete(data.sessionId); return next })
      }),
      window.api.sessions.onError((data: any) => {
        setSessions(prev => prev.map(s => s.id === data.sessionId ? { ...s, status: 'error' as const } : s))
        setLoadingSessions(prev => {
          const next = new Set(prev)
          const session = sessionsRef.current.find(s => s.id === data.sessionId)
          if (session) next.delete(session.modelPath)
          return next
        })
        setLoadProgress(prev => { const next = new Map(prev); next.delete(data.sessionId); return next })
      }),
      // Loading progress — real-time phase tracking from engine log parsing
      ...(window.api.sessions.onLoadProgress ? [window.api.sessions.onLoadProgress((data: any) => {
        setLoadProgress(prev => {
          const next = new Map(prev)
          next.set(data.sessionId, { label: data.label, progress: data.progress })
          return next
        })
      })] : []),
      window.api.sessions.onHealth((data: any) => {
        // Only set 'running' when the model is actually loaded (data.running === true)
        // The health monitor sends running=false when server is up but model still loading
        if (data.running) {
          setSessions(prev => prev.map(s =>
            s.id === data.sessionId ? { ...s, status: 'running' as const, ...(data.modelName ? { modelName: data.modelName } : {}) } : s
          ))
        }
      }),
      ...(window.api.sessions.onStandby ? [window.api.sessions.onStandby((data: any) => {
        setSessions(prev => prev.map(s =>
          s.id === data.sessionId ? { ...s, status: 'standby' as const, standbyDepth: data.depth || 'soft' } : s
        ))
      })] : []),
    ]

    return () => unsubs.forEach(fn => fn())
  }, [])

  const ensureSessionRunning = useCallback(async (modelPath: string): Promise<SessionSummary> => {
    const current = sessionsRef.current

    // Check if already running
    const existing = current.find(s => s.modelPath === modelPath && s.status === 'running')
    if (existing) return existing

    // Standby sessions have a live process — wake them instead of starting fresh
    const standby = current.find(s => s.modelPath === modelPath && s.status === 'standby')
    if (standby) {
      await window.api.sessions.wake?.(standby.id)
      // Return immediately — JIT middleware on the server handles the rest
      return { ...standby, status: 'running' }
    }

    // Check if loading
    const loading = current.find(s => s.modelPath === modelPath && s.status === 'loading')
    if (loading) {
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => { unsubReady(); unsubErr(); reject(new Error('Session start timed out')) }, 300000) // 5 min — JANG/large models need time
        const unsubReady = window.api.sessions.onReady((data: any) => {
          if (data.sessionId === loading.id) {
            clearTimeout(timeout)
            unsubReady()
            unsubErr()
            refreshSessions().then(() => {
              resolve({ ...loading, status: 'running', ...(data.port ? { port: data.port } : {}) })
            })
          }
        })
        const unsubErr = window.api.sessions.onError((data: any) => {
          if (data.sessionId === loading.id) {
            clearTimeout(timeout)
            unsubReady()
            unsubErr()
            reject(new Error(data.error || 'Session failed to start'))
          }
        })
      })
    }

    // Find stopped session or create new
    setLoadingSessions(prev => new Set(prev).add(modelPath))

    let session = current.find(s => s.modelPath === modelPath)
    if (!session) {
      const result = await window.api.sessions.create(modelPath, {})
      if (!result.success) {
        setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
        throw new Error(result.error || 'Failed to create session')
      }
      await refreshSessions()
      session = sessionsRef.current.find(s => s.modelPath === modelPath)
      if (!session) {
        setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
        throw new Error('Session created but not found')
      }
    }

    // Start the session
    if (session.status !== 'running' && session.status !== 'loading') {
      const result = await window.api.sessions.start(session.id)
      if (!result.success) {
        setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
        throw new Error(result.error || 'Failed to start session')
      }
    }

    // Wait for ready
    const sessionId = session.id
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        unsubReady()
        unsubErr()
        setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
        reject(new Error('Session start timed out after 60s'))
      }, 60000)
      const unsubReady = window.api.sessions.onReady((data: any) => {
        if (data.sessionId === sessionId) {
          clearTimeout(timeout)
          unsubReady()
          unsubErr()
          setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
          refreshSessions().then(() => {
            resolve({ ...session!, status: 'running', ...(data.port ? { port: data.port } : {}) })
          })
        }
      })
      const unsubErr = window.api.sessions.onError((data: any) => {
        if (data.sessionId === sessionId) {
          clearTimeout(timeout)
          unsubReady()
          unsubErr()
          setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
          reject(new Error(data.error || 'Session failed to start'))
        }
      })
    })
  }, [refreshSessions])

  return (
    <SessionsContext.Provider value={{ sessions, loadingSessions, loadProgress, ensureSessionRunning, refreshSessions }}>
      {children}
    </SessionsContext.Provider>
  )
}
