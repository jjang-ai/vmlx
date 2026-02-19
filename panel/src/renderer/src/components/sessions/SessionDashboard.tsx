import { useState, useEffect } from 'react'
import { SessionCard } from './SessionCard'
import { useToast } from '../Toast'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading'
  config: string
  createdAt: number
  updatedAt: number
  lastStartedAt?: number
  lastStoppedAt?: number
}

interface SessionDashboardProps {
  onOpenSession: (sessionId: string) => void
  onConfigureSession: (sessionId: string) => void
  onCreateSession: () => void
}

export function SessionDashboard({ onOpenSession, onConfigureSession, onCreateSession }: SessionDashboardProps) {
  const { showToast } = useToast()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const [showDirManager, setShowDirManager] = useState(false)
  const [userDirs, setUserDirs] = useState<string[]>([])
  const [builtinDirs, setBuiltinDirs] = useState<string[]>([])
  const [dirError, setDirError] = useState<string | null>(null)
  const [manualPath, setManualPath] = useState('')

  const loadSessions = async () => {
    try {
      const list = await window.api.sessions.list()
      setSessions(list)
    } catch (err) {
      console.error('Failed to load sessions:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadSessions()

    // Listen for session events to refresh
    const unsubs = [
      window.api.sessions.onStarting(() => loadSessions()),
      window.api.sessions.onReady(() => loadSessions()),
      window.api.sessions.onStopped(() => loadSessions()),
      window.api.sessions.onError(() => loadSessions()),
      window.api.sessions.onHealth((data: any) => {
        // Targeted update — only patch the specific session's status fields
        // instead of reloading the entire list every 5 seconds
        if (data?.sessionId) {
          setSessions(prev => prev.map(s =>
            s.id === data.sessionId
              ? { ...s, status: 'running' as const, modelName: data.modelName || s.modelName }
              : s
          ))
        }
      }),
      window.api.sessions.onCreated(() => loadSessions()),
      window.api.sessions.onDeleted(() => loadSessions()),
    ]

    return () => {
      unsubs.forEach(unsub => unsub())
    }
  }, [])

  const handleStart = async (sessionId: string) => {
    // Optimistic update
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, status: 'loading' as const } : s))
    const result = await window.api.sessions.start(sessionId)
    if (!result.success) {
      showToast('error', 'Failed to start session', result.error)
    }
    loadSessions()
  }

  const handleStop = async (sessionId: string) => {
    setSessions(prev => prev.map(s => s.id === sessionId ? { ...s, status: 'stopped' as const } : s))
    const result = await window.api.sessions.stop(sessionId)
    if (!result.success) {
      showToast('error', 'Failed to stop session', result.error)
    }
    loadSessions()
  }

  const handleDelete = async (sessionId: string) => {
    if (!confirm('Delete this session? Chat history for this model will be preserved.')) return
    const result = await window.api.sessions.delete(sessionId)
    if (!result.success) {
      showToast('error', 'Failed to delete session', result.error)
    }
    loadSessions()
  }

  const handleDetect = async () => {
    setLoading(true)
    try {
      await window.api.sessions.detect()
      await loadSessions()
    } catch (err) {
      console.error('Failed to detect processes:', err)
    } finally {
      setLoading(false)
    }
  }

  const loadDirectories = async () => {
    try {
      const result = await window.api.models.getDirectories()
      setUserDirs(result.userDirectories)
      setBuiltinDirs(result.builtinDirectories)
    } catch (err) {
      console.error('Failed to load directories:', err)
    }
  }

  const handleBrowseDirectory = async () => {
    setDirError(null)
    const result = await window.api.models.browseDirectory()
    if (result.canceled || !result.path) return
    await addDirectory(result.path)
  }

  const handleAddManualPath = async () => {
    if (!manualPath.trim()) return
    setDirError(null)
    await addDirectory(manualPath.trim())
    setManualPath('')
  }

  const addDirectory = async (dirPath: string) => {
    const result = await window.api.models.addDirectory(dirPath)
    if (result.success) {
      await loadDirectories()
    } else {
      setDirError(result.error || 'Failed to add directory')
    }
  }

  const handleRemoveDirectory = async (dirPath: string) => {
    await window.api.models.removeDirectory(dirPath)
    await loadDirectories()
  }

  const runningSessions = sessions.filter(s => s.status === 'running' || s.status === 'loading')
  const stoppedSessions = sessions.filter(s => s.status === 'stopped' || s.status === 'error')

  if (loading && sessions.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Loading sessions...</p>
      </div>
    )
  }

  return (
    <div className="p-6 overflow-auto h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Sessions</h1>
        <div className="flex gap-2">
          <button
            onClick={() => { setShowDirManager(!showDirManager); if (!showDirManager) loadDirectories() }}
            className={`px-3 py-1.5 text-sm border rounded ${
              showDirManager ? 'border-primary bg-primary/10 text-primary' : 'border-border hover:bg-accent'
            }`}
            title="Configure model scan directories"
          >
            📁 Directories
          </button>
          <button
            onClick={handleDetect}
            className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent"
          >
            Detect Processes
          </button>
          <button
            onClick={onCreateSession}
            className="px-4 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
          >
            + New Session
          </button>
        </div>
      </div>

      {/* Directory Manager Panel */}
      {showDirManager && (
        <div className="mb-6 p-4 bg-card border border-border rounded-lg">
          <h3 className="text-sm font-semibold mb-3">Model Scan Directories</h3>
          <p className="text-xs text-muted-foreground mb-3">
            vMLX scans these directories for models when creating a new session.
          </p>

          {/* Built-in directories */}
          {builtinDirs.length > 0 && (
            <div className="mb-3">
              <span className="text-xs text-muted-foreground uppercase tracking-wider">Default</span>
              {builtinDirs.map(dir => (
                <div key={dir} className="flex items-center gap-2 mt-1 px-2 py-1.5 bg-muted/50 rounded text-xs text-muted-foreground">
                  <span className="truncate flex-1" title={dir}>{dir}</span>
                  <span className="text-xs opacity-50 flex-shrink-0">built-in</span>
                </div>
              ))}
            </div>
          )}

          {/* User directories */}
          {userDirs.length > 0 && (
            <div className="mb-3">
              <span className="text-xs text-muted-foreground uppercase tracking-wider">Custom</span>
              {userDirs.map(dir => (
                <div key={dir} className="flex items-center gap-2 mt-1 px-2 py-1.5 bg-muted/50 rounded text-xs">
                  <span className="truncate flex-1" title={dir}>{dir}</span>
                  <button
                    onClick={() => handleRemoveDirectory(dir)}
                    className="text-destructive hover:text-destructive/80 flex-shrink-0"
                    title="Remove directory"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Add directory */}
          <div className="flex items-center gap-2 mt-2">
            <input
              type="text"
              placeholder="Enter path or browse..."
              value={manualPath}
              onChange={(e) => { setManualPath(e.target.value); setDirError(null) }}
              onKeyDown={(e) => { if (e.key === 'Enter') handleAddManualPath() }}
              className="flex-1 px-2 py-1.5 bg-background border border-input rounded text-xs"
            />
            <button
              onClick={handleAddManualPath}
              disabled={!manualPath.trim()}
              className="px-2 py-1.5 text-xs border border-border rounded hover:bg-accent disabled:opacity-50"
            >
              Add
            </button>
            <button
              onClick={handleBrowseDirectory}
              className="px-2 py-1.5 text-xs border border-border rounded hover:bg-accent"
            >
              Browse...
            </button>
          </div>
          {dirError && (
            <p className="text-xs text-destructive mt-1">{dirError}</p>
          )}
        </div>
      )}

      {sessions.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="text-6xl mb-4">🚀</div>
          <h2 className="text-xl font-semibold mb-2">No Sessions Yet</h2>
          <p className="text-muted-foreground text-center mb-6 max-w-md">
            Create a session to load a model and start an inference server.
            Or click "Detect Processes" to detect running servers.
          </p>
          <div className="flex gap-3">
            <button
              onClick={onCreateSession}
              className="px-6 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90"
            >
              Create Session
            </button>
            <button
              onClick={handleDetect}
              className="px-6 py-2 border border-border rounded hover:bg-accent"
            >
              Detect Running
            </button>
          </div>
        </div>
      ) : (
        <>
          {/* Running Sessions */}
          {runningSessions.length > 0 && (
            <div className="mb-6">
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Active ({runningSessions.length})
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {runningSessions.map(session => (
                  <SessionCard
                    key={session.id}
                    session={session}
                    onOpen={onOpenSession}
                    onConfigure={onConfigureSession}
                    onStart={handleStart}
                    onStop={handleStop}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Stopped Sessions */}
          {stoppedSessions.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                Inactive ({stoppedSessions.length})
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {stoppedSessions.map(session => (
                  <SessionCard
                    key={session.id}
                    session={session}
                    onOpen={onOpenSession}
                    onConfigure={onConfigureSession}
                    onStart={handleStart}
                    onStop={handleStop}
                    onDelete={handleDelete}
                  />
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
