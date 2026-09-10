import { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'
import { useTranslation } from '../i18n'

export interface SessionSummary {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  standbyDepth?: 'soft' | 'deep' | null
  type?: 'local' | 'remote'
  remoteUrl?: string
  config?: string // JSON blob — includes modelType, imageMode, etc.
  modelPathMissing: boolean
  usableTwinId?: string
}

export interface LoadProgress {
  /** English text from the main process, used as the i18n fallback. */
  label: string
  /**
   * i18n key for the SAME text. The main process derives these labels from
   * engine log lines and has no locale catalog of its own, so it ships both and
   * the renderer resolves the key. Optional so an older main process still
   * renders (the label is then used verbatim, exactly as before).
   */
  labelKey?: string
  /**
   * Values interpolated into `labelKey`. The resident-RAM phases embed GB
   * figures, so the numbers travel separately and the sentence around them
   * stays translatable instead of being baked into an English template.
   */
  labelParams?: Record<string, string | number>
  progress: number
  /** Engine lifecycle generation — stale-event guard across load/wake attempts. */
  progressGeneration?: number
  /** True while the current phase has no measured denominator — render an
   *  animated bar with no numeric percentage instead of an invented one. */
  indeterminate?: boolean
  modelBytes?: number
  expectedResidentBytes?: number
  lazyResident?: boolean
  residentMb?: number
  residentPercent?: number
  peakMb?: number
  cacheMb?: number
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
  const { t } = useTranslation()
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

    // Hydrate the current load/wake progress from the main process: a
    // renderer (re)mounted mid-load never saw the events emitted before it
    // opened, and a bar that depends only on those events starts blank.
    window.api.sessions.getLoadProgress?.().then((snapshot: Record<string, any>) => {
      if (!snapshot || typeof snapshot !== 'object') return
      setLoadProgress(prev => {
        const next = new Map(prev)
        for (const [sessionId, entry] of Object.entries(snapshot)) {
          if (!next.has(sessionId) && entry && typeof entry === 'object') {
            next.set(sessionId, entry as LoadProgress)
          }
        }
        return next
      })
    }).catch(() => {})

    const unsubs = [
      window.api.sessions.onCreated(() => refreshSessions()),
      window.api.sessions.onDeleted(() => refreshSessions()),
      window.api.sessions.onUpdated(() => refreshSessions()),
      window.api.sessions.onStarting((data: any) => {
        setSessions(prev => prev.map(s => s.id === data.sessionId ? { ...s, status: 'loading' as const } : s))
        setLoadProgress(prev => { const next = new Map(prev); next.delete(data.sessionId); return next })
      }),
      window.api.sessions.onReady((data: any) => {
        // Restart promotes pending launch settings after session:updated. The
        // pre-restart list still carries the old active config; status/PID alone
        // cannot refresh the config consumed by the chat toolbar and drawers.
        void refreshSessions()
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
        // Deliberately NOT forcing progress to 100 here: the main process
        // owns the terminal 100% (engine-authoritative ready) on every path;
        // fabricating it here ended the bar before the model was ready.
      }),
      window.api.sessions.onStopped((data: any) => {
        setSessions(prev => prev.map(s => s.id === data.sessionId ? { ...s, status: 'stopped' as const, pid: undefined } : s))
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
          // Lifecycle generation guard: an event from an older load/wake
          // attempt (stale after Stop/restart/PID replacement) must never
          // repaint the bar over the current attempt's state.
          const existing = prev.get(data.sessionId) as any
          if (
            data.progressGeneration != null &&
            existing?.progressGeneration != null &&
            data.progressGeneration < existing.progressGeneration
          ) {
            return prev
          }
          const next = new Map(prev)
          next.set(data.sessionId, {
            ...(next.get(data.sessionId) || {}),
            ...(data.progressGeneration != null ? { progressGeneration: data.progressGeneration } : {}),
            indeterminate: data.indeterminate === true,
            label: data.label,
            // Assigned unconditionally, not spread in only when present: the
            // previous entry is spread above, so a conditional copy would leave
            // the LAST phase's key paired with this phase's English label and
            // render the wrong translated string. Dropping it here was what
            // made the i18n keys dead — SessionCard saw labelKey undefined and
            // fell back to English in every locale.
            labelKey: data.labelKey,
            labelParams: data.labelParams,
            progress: data.progress,
            ...(data.modelBytes != null ? { modelBytes: data.modelBytes } : {}),
            ...(data.expectedResidentBytes != null ? { expectedResidentBytes: data.expectedResidentBytes } : {}),
            ...(data.lazyResident != null ? { lazyResident: data.lazyResident } : {}),
            ...(data.residentMb != null ? { residentMb: data.residentMb } : {}),
            ...(data.residentPercent != null ? { residentPercent: data.residentPercent } : {}),
            ...(data.peakMb != null ? { peakMb: data.peakMb } : {}),
            ...(data.cacheMb != null ? { cacheMb: data.cacheMb } : {}),
          })
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
        const timeout = setTimeout(() => { unsubReady(); unsubErr(); reject(new Error(t('sessions.context.startTimedOut'))) }, 300000) // 5 min — JANG/large models need time
        const unsubReady = window.api.sessions.onReady((data: any) => {
          if (data.sessionId === loading.id) {
            clearTimeout(timeout)
            unsubReady()
            unsubErr()
            refreshSessions().then(() => {
              resolve({
                ...loading,
                status: 'running',
                ...(data.pid ? { pid: data.pid } : {}),
                ...(data.port ? { port: data.port } : {})
              })
            })
          }
        })
        const unsubErr = window.api.sessions.onError((data: any) => {
          if (data.sessionId === loading.id) {
            clearTimeout(timeout)
            unsubReady()
            unsubErr()
            reject(new Error(data.error || t('sessions.context.sessionFailedToStart')))
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
        throw new Error(result.error || t('sessions.context.createFailed'))
      }
      await refreshSessions()
      session = sessionsRef.current.find(s => s.modelPath === modelPath)
      if (!session) {
        setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
        throw new Error(t('sessions.context.createdNotFound'))
      }
    }

    // Start the session
    if (session.status !== 'running' && session.status !== 'loading') {
      const result = await window.api.sessions.start(session.id)
      if (!result.success) {
        setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
        throw new Error(result.error || t('sessions.dashboard.toast.startFailed'))
      }
    }

    // Wait for ready
    const sessionId = session.id
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        unsubReady()
        unsubErr()
        setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
        reject(new Error(t('sessions.context.startTimedOut5m')))
      }, 300000)
      const unsubReady = window.api.sessions.onReady((data: any) => {
        if (data.sessionId === sessionId) {
          clearTimeout(timeout)
          unsubReady()
          unsubErr()
          setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
          refreshSessions().then(() => {
            resolve({
              ...session!,
              status: 'running',
              ...(data.pid ? { pid: data.pid } : {}),
              ...(data.port ? { port: data.port } : {})
            })
          })
        }
      })
      const unsubErr = window.api.sessions.onError((data: any) => {
        if (data.sessionId === sessionId) {
          clearTimeout(timeout)
          unsubReady()
          unsubErr()
          setLoadingSessions(prev => { const next = new Set(prev); next.delete(modelPath); return next })
          reject(new Error(data.error || t('sessions.context.sessionFailedToStart')))
        }
      })
    })
  }, [refreshSessions, t])

  return (
    <SessionsContext.Provider value={{ sessions, loadingSessions, loadProgress, ensureSessionRunning, refreshSessions }}>
      {children}
    </SessionsContext.Provider>
  )
}
