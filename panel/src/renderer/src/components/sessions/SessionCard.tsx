import { useState, useEffect } from 'react'
import { Settings, ScrollText, Moon, Sun } from 'lucide-react'
import { useSessionsContext, type LoadProgress } from '../../contexts/SessionsContext'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  standbyDepth?: 'soft' | 'deep' | null
  config: string
  createdAt: number
  updatedAt: number
  lastStartedAt?: number
  lastStoppedAt?: number
  type?: 'local' | 'remote'
  remoteUrl?: string
  remoteModel?: string
}

interface SessionCardProps {
  session: Session
  onOpen: (sessionId: string) => void
  onConfigure: (sessionId: string) => void
  onStart: (sessionId: string) => void
  onStop: (sessionId: string) => void
  onDelete: (sessionId: string) => void
  onSleep?: (sessionId: string) => void
  onWake?: (sessionId: string) => void
}

const statusColors: Record<string, string> = {
  running: 'bg-green-500',
  stopped: 'bg-muted-foreground',
  error: 'bg-destructive',
  loading: 'bg-yellow-500 animate-pulse',
  standby: 'bg-blue-400'
}

function formatElapsed(secs: number): string {
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${m}m ${s}s`
}

const statusLabels: Record<string, string> = {
  running: 'Running',
  stopped: 'Stopped',
  error: 'Error',
  loading: 'Loading...',
  standby: 'Sleeping'
}

export function SessionCard({ session, onOpen, onConfigure, onStart, onStop, onDelete, onSleep, onWake }: SessionCardProps) {
  const isRemote = session.type === 'remote'
  const isImage = (() => { try { return JSON.parse(session.config || '{}').modelType === 'image' } catch { return false } })()
  const shortName = session.modelName || session.modelPath.split('/').pop() || session.modelPath
  const [jangLabel, setJangLabel] = useState<string | undefined>(undefined)
  const [loadingElapsed, setLoadingElapsed] = useState(0)
  const { loadProgress } = useSessionsContext()
  const progress = loadProgress.get(session.id)

  // Elapsed time counter when model is loading
  useEffect(() => {
    if (session.status !== 'loading') {
      setLoadingElapsed(0)
      return
    }
    const interval = setInterval(() => {
      setLoadingElapsed(prev => prev + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [session.status])

  useEffect(() => {
    if (isRemote) return
    // Check if model directory name contains JANG/MXQ indicators.
    // NOTE: JANG detection happens server-side (via jang_config.json parsing in the engine).
    // Pre-start model info would require reading jang_config.json from the frontend,
    // which is not currently implemented. This heuristic uses directory name patterns only.
    const name = session.modelPath.toLowerCase()
    if (name.includes('jang') || name.includes('mxq') || name.includes('mlxq')) {
      // Extract bits from patterns: JANG_4K, JANG_2S, JANG_2L, JANG-3.99-bit
      const match = name.match(/jang[_-](\d+\.?\d*)/i)
      setJangLabel(match ? `JANG ${match[1]}-bit` : 'JANG')
    }
  }, [session.modelPath, isRemote])

  return (
    <div className="bg-card border border-border rounded-lg p-4 hover:border-primary/50 transition-colors">
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1 min-w-0 mr-2">
          <div className="flex items-center gap-1.5">
            {isRemote && <span className="text-xs bg-primary/20 text-primary px-1.5 py-0.5 rounded flex-shrink-0">Remote</span>}
            {isImage && <span className="text-xs bg-violet-500/15 text-violet-400 px-1.5 py-0.5 rounded flex-shrink-0">Image</span>}
            <h3 className="font-semibold text-sm truncate" title={session.modelPath}>
              {shortName}
            </h3>
            {jangLabel && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-400 font-medium flex-shrink-0">
                {jangLabel}
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground truncate mt-0.5" title={isRemote ? session.remoteUrl : session.modelPath}>
            {isRemote ? session.remoteUrl : session.modelPath}
          </p>
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {session.status === 'standby' ? (
            <Moon className={`h-3 w-3 ${session.standbyDepth === 'deep' ? 'text-indigo-400' : 'text-blue-400'}`} />
          ) : (
            <span className={`w-2 h-2 rounded-full ${statusColors[session.status]}`} />
          )}
          <span className="text-xs text-muted-foreground">
            {session.status === 'loading'
              ? `Loading... ${formatElapsed(loadingElapsed)}`
              : session.status === 'standby'
                ? (session.standbyDepth === 'deep' ? 'Deep Sleep' : 'Light Sleep')
                : statusLabels[session.status]}
          </span>
        </div>
      </div>

      {/* Loading progress bar */}
      {session.status === 'loading' && (
        <div className="mb-3">
          <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-yellow-500 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress?.progress ?? 2}%` }}
            />
          </div>
          {progress && (
            <p className="text-[10px] text-muted-foreground mt-1">
              {progress.label} ({progress.progress}%)
            </p>
          )}
        </div>
      )}

      {/* Info */}
      <div className="flex gap-4 text-xs text-muted-foreground mb-3">
        {isRemote
          ? <span>{session.remoteUrl ? new URL(session.remoteUrl).host : session.host}</span>
          : <span>{session.host}:{session.port}</span>
        }
        {!isRemote && session.pid && <span>PID {session.pid}</span>}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        {session.status === 'running' && (
          <button
            onClick={() => onOpen(session.id)}
            className="flex-1 px-3 py-1.5 bg-primary text-primary-foreground text-sm rounded hover:bg-primary/90"
          >
            Open
          </button>
        )}

        {session.status === 'standby' && onWake && (
          <button
            onClick={() => onWake(session.id)}
            className="flex-1 px-3 py-1.5 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 flex items-center justify-center gap-1.5"
            title={session.standbyDepth === 'deep' ? 'Reload model and resume' : 'Resume from light sleep'}
          >
            <Sun className="h-3.5 w-3.5" />
            Wake
          </button>
        )}

        {session.status === 'loading' && (
          <>
            <button
              onClick={() => onOpen(session.id)}
              className="flex-1 px-3 py-1.5 bg-warning/20 text-warning text-sm rounded hover:bg-warning/30 flex items-center justify-center gap-1.5"
            >
              <ScrollText className="h-3.5 w-3.5" />
              Logs
            </button>
          </>
        )}

        {session.status === 'stopped' || session.status === 'error' ? (
          <button
            onClick={() => onStart(session.id)}
            className="flex-1 px-3 py-1.5 bg-success text-success-foreground text-sm rounded hover:bg-success/90"
          >
            {isRemote ? 'Connect' : 'Start'}
          </button>
        ) : null}

        {!isRemote && (
          <button
            onClick={() => onConfigure(session.id)}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-accent"
            title="Configure session settings"
          >
            <Settings className="h-4 w-4" />
          </button>
        )}

        {session.status === 'running' && !isRemote && onSleep && (
          <button
            onClick={() => onSleep(session.id)}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-blue-500/10 hover:text-blue-400 hover:border-blue-500/30"
            title="Put model to sleep (frees memory)"
          >
            <Moon className="h-4 w-4" />
          </button>
        )}

        {(session.status === 'running' || session.status === 'loading') && (
          <button
            onClick={() => onStop(session.id)}
            className="px-3 py-1.5 bg-destructive text-destructive-foreground text-sm rounded hover:bg-destructive/90"
          >
            {isRemote ? 'Disconnect' : 'Stop'}
          </button>
        )}

        {(session.status === 'standby') && (
          <button
            onClick={() => onStop(session.id)}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-destructive hover:text-destructive-foreground hover:border-destructive"
            title="Stop server process completely"
          >
            Stop
          </button>
        )}

        {session.status === 'error' && (
          <button
            onClick={() => onOpen(session.id)}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-accent"
            title="View crash logs"
          >
            <ScrollText className="h-4 w-4" />
          </button>
        )}

        {(session.status === 'stopped' || session.status === 'error') && (
          <button
            onClick={() => onDelete(session.id)}
            className="px-3 py-1.5 text-sm rounded border border-border text-muted-foreground hover:bg-destructive hover:text-destructive-foreground hover:border-destructive"
          >
            Delete
          </button>
        )}
      </div>
    </div>
  )
}
