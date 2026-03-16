import { useState, useEffect } from 'react'
import { FolderOpen, Cpu } from 'lucide-react'
import { SessionCard } from './SessionCard'
import { DirectoryManager } from './DirectoryManager'
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
  /** Custom title for the dashboard (default: "Model Sessions") */
  title?: string
  /** Custom subtitle */
  subtitle?: string
  /** Filter session list by type */
  filterType?: 'text' | 'image'
}

export function SessionDashboard({ onOpenSession, onConfigureSession, onCreateSession, title, subtitle, filterType }: SessionDashboardProps) {
  const { showToast } = useToast()
  const [sessions, setSessions] = useState<Session[]>([])
  const [loading, setLoading] = useState(true)
  const [showDirManager, setShowDirManager] = useState(false)
  const [userDirs, setUserDirs] = useState<string[]>([])
  const [builtinDirs, setBuiltinDirs] = useState<string[]>([])
  const [dirError, setDirError] = useState<string | null>(null)

  const loadSessions = async () => {
    try {
      let list = await window.api.sessions.list()
      // Filter by model type using file-based detection (not name matching)
      // Image models: have model_index.json (diffusers format) — detected via IPC
      // Text models: have config.json with model_type (transformers format)
      if (filterType === 'image' || filterType === 'text') {
        try {
          const modelTypes = await window.api.models.detectTypes(list.map((s: any) => s.modelPath))
          list = list.filter((s: any, i: number) => {
            const type = modelTypes[i] // 'image' | 'text' | 'unknown'
            if (filterType === 'image') return type === 'image'
            if (filterType === 'text') return type !== 'image'
            return true
          })
        } catch {
          // If detection fails, show all sessions
        }
      }
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

  const handleAddManualPath = async (path: string) => {
    if (!path) return
    setDirError(null)
    await addDirectory(path)
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
        <div>
          <h1 className="text-2xl font-bold">{title || 'Sessions'}</h1>
          {subtitle && <p className="text-sm text-muted-foreground mt-0.5">{subtitle}</p>}
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => { setShowDirManager(!showDirManager); if (!showDirManager) loadDirectories() }}
            className={`px-3 py-1.5 text-sm border rounded ${
              showDirManager ? 'border-primary bg-primary/10 text-primary' : 'border-border hover:bg-accent'
            }`}
            title="Configure model scan directories"
          >
            <FolderOpen className="h-4 w-4" /> Directories
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
          <DirectoryManager
            userDirs={userDirs}
            builtinDirs={builtinDirs}
            dirError={dirError}
            onAdd={handleAddManualPath}
            onRemove={handleRemoveDirectory}
            onBrowse={handleBrowseDirectory}
            onClearError={() => setDirError(null)}
            description="vMLX scans these directories for models when creating a new session."
          />
        </div>
      )}

      {sessions.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mb-4">
            <Cpu className="h-7 w-7 text-primary" />
          </div>
          <h2 className="text-xl font-semibold mb-2">No sessions yet</h2>
          <p className="text-sm text-muted-foreground text-center mb-6 max-w-sm">
            Create a session to load a model and start chatting, or detect a running server.
          </p>
          <div className="flex gap-3">
            <button
              onClick={onCreateSession}
              className="px-5 py-2.5 bg-primary text-primary-foreground rounded-xl hover:bg-primary/90 font-medium text-sm transition-colors"
            >
              Create Session
            </button>
            <button
              onClick={handleDetect}
              className="px-5 py-2.5 border border-border rounded-xl hover:bg-accent text-sm transition-colors"
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
