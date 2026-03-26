import { useState, useEffect } from 'react'
import { ArrowLeft, Menu, Settings, X, ImageIcon, AlertTriangle } from 'lucide-react'
import { ChatInterface } from '../chat/ChatInterface'
import { ChatList } from '../chat/ChatList'
import { ChatSettings } from '../chat/ChatSettings'
import { ServerSettingsDrawer } from './ServerSettingsDrawer'
import { CachePanel } from './CachePanel'
import { BenchmarkPanel } from './BenchmarkPanel'
import { EmbeddingsPanel } from './EmbeddingsPanel'
import { PerformancePanel } from './PerformancePanel'
import { LogsPanel } from './LogsPanel'
import { useToast } from '../Toast'
import { useAppState } from '../../contexts/AppStateContext'
import { useSessionsContext } from '../../contexts/SessionsContext'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  config: string
  createdAt: number
  updatedAt: number
  type?: 'local' | 'remote'
  remoteUrl?: string
  remoteModel?: string
  latencyMs?: number
}

interface SessionViewProps {
  sessionId: string
  onBack: () => void
}

export function SessionView({ sessionId, onBack }: SessionViewProps) {
  const { showToast } = useToast()
  const { setMode } = useAppState()
  const [session, setSession] = useState<Session | null>(null)
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [showChatList, setShowChatList] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [showServerSettings, setShowServerSettings] = useState(false)
  const [showCache, setShowCache] = useState(false)
  const [showBenchmark, setShowBenchmark] = useState(false)
  const [showEmbeddings, setShowEmbeddings] = useState(false)
  const [showPerformance, setShowPerformance] = useState(false)
  const [showLogs, setShowLogs] = useState(false)
  const [overridesVersion, setOverridesVersion] = useState(0)
  const [effectiveReasoningParser, setEffectiveReasoningParser] = useState<string | undefined>(undefined)
  const [jangLabel, setJangLabel] = useState<string | undefined>(undefined)
  const [jangNoticeDismissed, setJangNoticeDismissed] = useState(true) // hidden by default until checked

  // Check JANG redownload notice dismiss state
  useEffect(() => {
    try {
      const val = window.api.settings?.get('jang_redownload_dismissed')
      if (val instanceof Promise) {
        val.then((v: any) => { if (v !== '2026-03-19') setJangNoticeDismissed(false) })
           .catch(() => setJangNoticeDismissed(false))
      } else {
        if (val !== '2026-03-19') setJangNoticeDismissed(false)
      }
    } catch { setJangNoticeDismissed(false) }
  }, [])

  // Load session and its chats
  useEffect(() => {
    const loadSession = async () => {
      try {
        const s = await window.api.sessions.get(sessionId)
        setSession(s)
        // Auto-open logs panel if session is in error state
        if (s?.status === 'error') setShowLogs(true)

        if (s) {
          // Skip chat loading for image model sessions — they don't use chat
          const cfg = s.config ? (() => { try { return JSON.parse(s.config) } catch { return {} } })() : {}
          if (cfg.modelType !== 'image') {
            // Load chats for this model, auto-create first chat if none exist
            const chats = await window.api.chat.getByModel(s.modelPath)
            if (chats.length > 0) {
              setCurrentChatId(chats[0].id)
            } else {
              try {
                const title = `Chat ${new Date().toLocaleString()}`
                const chat = await window.api.chat.create(title, 'default', undefined, s.modelPath)
                setCurrentChatId(chat.id)
              } catch (_) { /* will show empty state */ }
            }
          }
          // Detect effective reasoning parser for chat settings UI
          // Skip filesystem detection for remote sessions (remote:// paths don't exist on disk)
          try {
            const cfg = s.config ? JSON.parse(s.config) : {}
            if (cfg.reasoningParser && cfg.reasoningParser !== 'auto') {
              setEffectiveReasoningParser(cfg.reasoningParser)
            } else if (!s.modelPath.startsWith('remote://')) {
              const detected = await window.api.models.detectConfig(s.modelPath)
              setEffectiveReasoningParser(detected?.reasoningParser || undefined)
            }
          } catch (_) { /* ignore detection errors */ }

          // Detect JANG format: scan model list for matching path
          if (!s.modelPath.startsWith('remote://')) {
            try {
              const models = await window.api.models.scan()
              const match = models.find((m: any) => m.path === s.modelPath)
              if (match?.quantization && match.quantization.startsWith('JANG')) {
                setJangLabel(match.quantization)
              }
            } catch (_) { /* ignore scan errors */ }
          }
        }
      } catch (err) {
        console.error('Failed to load session:', err)
      }
    }
    loadSession()

    // Listen for session health updates
    const handleHealth = (data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? {
          ...prev,
          status: 'running',
          ...(data.modelName ? { modelName: data.modelName } : {}),
          ...(data.port ? { port: data.port } : {}),
          ...(data.latencyMs != null ? { latencyMs: data.latencyMs } : {})
        } : prev)
      }
    }
    const handleStarting = (data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'loading' } : prev)
      }
    }
    const handleReady = (data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? {
          ...prev,
          status: 'running',
          ...(data.pid ? { pid: data.pid } : {}),
          ...(data.port ? { port: data.port } : {})
        } : prev)
      }
    }
    const handleStopped = (data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'stopped', pid: undefined, latencyMs: undefined } : prev)
      }
    }
    const handleError = (data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'error', latencyMs: undefined } : prev)
      }
    }

    const unsubHealth = window.api.sessions.onHealth(handleHealth)
    const unsubStarting = window.api.sessions.onStarting(handleStarting)
    const unsubReady = window.api.sessions.onReady(handleReady)
    const unsubStopped = window.api.sessions.onStopped(handleStopped)
    const unsubError = window.api.sessions.onError(handleError)
    const unsubStandby = window.api.sessions.onStandby?.((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'standby' as const } : prev)
      }
    })

    return () => {
      unsubHealth()
      unsubStarting()
      unsubReady()
      unsubStopped()
      unsubError()
      unsubStandby?.()
    }
  }, [sessionId])

  const handleNewChat = async () => {
    if (!session) return
    try {
      const title = `Chat ${new Date().toLocaleString()}`
      const chat = await window.api.chat.create(title, 'default', undefined, session.modelPath)
      setCurrentChatId(chat.id)
      setShowChatList(false)
    } catch (error) {
      showToast('error', 'Failed to create chat', (error as Error).message)
    }
  }

  const handleChatSelect = (chatId: string) => {
    setCurrentChatId(chatId)
    setShowChatList(false)
  }

  const handleStop = async () => {
    if (!session) return
    const result = await window.api.sessions.stop(session.id)
    if (!result.success) {
      showToast('error', 'Failed to stop server', result.error)
    }
  }

  const handleStart = async () => {
    if (!session) return
    setSession(prev => prev ? { ...prev, status: 'loading' } : prev)
    const result = await window.api.sessions.start(session.id)
    if (!result.success) {
      setSession(prev => prev ? { ...prev, status: 'error' } : prev)
      showToast('error', 'Failed to start server', result.error)
    }
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Loading session...</p>
      </div>
    )
  }

  const isRemote = session.type === 'remote'
  const shortName = session.modelName || session.modelPath.split('/').pop() || session.modelPath
  const statusColor = session.status === 'running'
    ? (isRemote ? 'bg-success' : 'bg-primary')
    : session.status === 'standby' ? 'bg-blue-400'
    : session.status === 'loading' ? 'bg-warning'
    : session.status === 'error' ? 'bg-destructive'
    : 'bg-muted-foreground'
  const sessionConfig = (() => { try { return JSON.parse(session.config || '{}') } catch { return {} } })()
  const modelType = sessionConfig.modelType
  const isImage = modelType === 'image'
  // Read imageMode directly from config — no regex guessing
  const isImageEdit = isImage && sessionConfig.imageMode === 'edit'

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Session Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-border bg-card/50 flex-shrink-0">
        <button onClick={onBack} className="text-muted-foreground hover:text-foreground text-sm flex items-center gap-1">
          <ArrowLeft className="h-3.5 w-3.5" /> Sessions
        </button>
        <div className="w-px h-4 bg-border" />

        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusColor}`} />
          <span className="font-medium text-sm truncate" title={session.modelPath}>
            {shortName}
          </span>
          {isImage && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium flex-shrink-0 ${
              isImageEdit ? 'bg-violet-500/15 text-violet-400' : 'bg-blue-500/15 text-blue-400'
            }`}>
              {isImageEdit ? 'Image Edit' : 'Image Gen'}
            </span>
          )}
          {jangLabel && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-500/15 text-violet-400 font-medium flex-shrink-0">
              {jangLabel}
            </span>
          )}
          <span className="text-xs text-muted-foreground flex-shrink-0">
            {isRemote ? (session.remoteUrl || session.host) : `${session.host}:${session.port}`}
          </span>
          {!isRemote && session.pid && (
            <span className="text-xs text-muted-foreground flex-shrink-0">
              PID {session.pid}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          {!isImage && (
            <>
              <button
                onClick={() => setShowChatList(!showChatList)}
                className="text-sm hover:bg-accent px-2 py-1 rounded"
              >
                {showChatList ? <ArrowLeft className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
              </button>
              <button
                onClick={handleNewChat}
                className="text-sm hover:bg-accent px-2 py-1 rounded"
              >
                + Chat
              </button>
              <button
                onClick={() => { setShowSettings(!showSettings); if (!showSettings) { setShowServerSettings(false); setShowCache(false); setShowBenchmark(false); setShowEmbeddings(false); setShowPerformance(false); setShowLogs(false) } }}
                className={`text-sm px-2 py-1 rounded flex items-center gap-1 ${showSettings ? 'bg-accent' : 'hover:bg-accent'}`}
                title="Chat inference settings"
              >
                <Settings className="h-3.5 w-3.5" /> Chat
              </button>
            </>
          )}
          {isImage && (
            <button
              onClick={() => setMode('image')}
              className="text-sm hover:bg-accent px-2 py-1 rounded flex items-center gap-1"
              title={isImageEdit ? 'Switch to Image Editor' : 'Switch to Image Generator'}
            >
              <ImageIcon className="h-3.5 w-3.5" /> {isImageEdit ? 'Image Editor' : 'Image Gen'}
            </button>
          )}
          <button
            onClick={() => { setShowServerSettings(!showServerSettings); if (!showServerSettings) { setShowSettings(false); setShowCache(false); setShowBenchmark(false); setShowEmbeddings(false); setShowPerformance(false); setShowLogs(false) } }}
            className={`text-sm px-2 py-1 rounded flex items-center gap-1 ${showServerSettings ? 'bg-accent' : 'hover:bg-accent'}`}
            title={isRemote ? 'Connection Settings' : 'Server Settings'}
          >
            <Settings className="h-3.5 w-3.5" /> {isRemote ? 'Connection' : 'Server'}
          </button>
          {!isRemote && !isImage && session.status === 'running' && (
            <>
              <button
                onClick={() => { setShowCache(!showCache); if (!showCache) { setShowSettings(false); setShowServerSettings(false); setShowBenchmark(false); setShowEmbeddings(false); setShowPerformance(false); setShowLogs(false) } }}
                className={`text-sm px-2 py-1 rounded ${showCache ? 'bg-accent' : 'hover:bg-accent'}`}
                title="Cache Management"
              >
                Cache
              </button>
              <button
                onClick={() => { setShowBenchmark(!showBenchmark); if (!showBenchmark) { setShowSettings(false); setShowServerSettings(false); setShowCache(false); setShowEmbeddings(false); setShowPerformance(false); setShowLogs(false) } }}
                className={`text-sm px-2 py-1 rounded ${showBenchmark ? 'bg-accent' : 'hover:bg-accent'}`}
                title="Run Benchmark"
              >
                Bench
              </button>
              <button
                onClick={() => { setShowEmbeddings(!showEmbeddings); if (!showEmbeddings) { setShowSettings(false); setShowServerSettings(false); setShowCache(false); setShowBenchmark(false); setShowPerformance(false); setShowLogs(false) } }}
                className={`text-sm px-2 py-1 rounded ${showEmbeddings ? 'bg-accent' : 'hover:bg-accent'}`}
                title="Embeddings"
              >
                Embed
              </button>
              <button
                onClick={() => { setShowPerformance(!showPerformance); if (!showPerformance) { setShowSettings(false); setShowServerSettings(false); setShowCache(false); setShowBenchmark(false); setShowEmbeddings(false); setShowLogs(false) } }}
                className={`text-sm px-2 py-1 rounded ${showPerformance ? 'bg-accent' : 'hover:bg-accent'}`}
                title="Performance Monitor"
              >
                Perf
              </button>
            </>
          )}
          <button
            onClick={() => { setShowLogs(!showLogs); if (!showLogs) { setShowSettings(false); setShowServerSettings(false); setShowCache(false); setShowBenchmark(false); setShowEmbeddings(false); setShowPerformance(false) } }}
            className={`text-sm px-2 py-1 rounded ${showLogs ? 'bg-accent' : 'hover:bg-accent'}`}
            title={isRemote ? 'Connection Logs' : 'Server Logs'}
          >
            Logs
          </button>
          {session.status === 'running' && (
            <>
              {isRemote && (
                <span className="text-xs text-success font-medium px-1">
                  Connected{session.latencyMs != null ? ` · ${session.latencyMs}ms` : ''}
                </span>
              )}
              <button
                onClick={handleStop}
                className="text-xs px-2 py-1 bg-destructive text-destructive-foreground rounded hover:bg-destructive/90"
              >
                {isRemote ? 'Disconnect' : 'Stop'}
              </button>
            </>
          )}
          {session.status === 'standby' && (
            <>
              <span className="text-xs text-blue-400 font-medium px-1">
                {(session as any).standbyDepth === 'deep' ? 'Deep Sleep' : 'Light Sleep'}
              </span>
              <button
                onClick={async () => {
                  try { await window.api.sessions.wake?.(session.id) } catch {}
                }}
                className="text-xs px-2 py-1 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Wake
              </button>
              <button
                onClick={handleStop}
                className="text-xs px-2 py-1 border border-border text-muted-foreground rounded hover:bg-destructive hover:text-destructive-foreground"
              >
                Stop
              </button>
            </>
          )}
          {(session.status === 'stopped' || session.status === 'error') && (
            <>
              {isRemote && (
                <span className="text-xs text-destructive font-medium px-1">Disconnected</span>
              )}
              <button
                onClick={handleStart}
                className="text-xs px-2 py-1 bg-success text-success-foreground rounded hover:bg-success/90"
              >
                {isRemote ? (session.status === 'error' ? 'Reconnect' : 'Connect') : 'Start'}
              </button>
            </>
          )}
          {session.status === 'loading' && (
            <>
              <span className="text-xs text-muted-foreground px-2 py-1 animate-pulse">
                {isRemote ? 'Connecting...' : 'Starting...'}
              </span>
              <button
                onClick={handleStop}
                className="text-xs px-2 py-1 border border-destructive/40 text-destructive rounded hover:bg-destructive/10"
              >
                Cancel
              </button>
            </>
          )}
        </div>
      </div>

      {/* Loading Progress Bar */}
      {session.status === 'loading' && <SessionViewLoadBar sessionId={session.id} />}

      {/* JANG Redownload Notice */}
      {jangLabel && !jangNoticeDismissed && (
        <div className="flex items-start gap-2 px-4 py-2 bg-amber-500/10 border-b border-amber-500/20 flex-shrink-0">
          <AlertTriangle className="h-3.5 w-3.5 text-amber-400 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-amber-300/90 flex-1">
            <span className="font-semibold">JANG Model Update Required:</span>{' '}
            As of 3/19/2026, all JANG models must be re-downloaded from HuggingFace (JANGQ-AI) to include proper chat templates.
            Without updated models, thinking on/off may not work correctly and some models may loop.
          </p>
          <button
            onClick={() => {
              setJangNoticeDismissed(true)
              try { window.api.settings?.set('jang_redownload_dismissed', '2026-03-19') } catch {}
            }}
            className="text-amber-400/60 hover:text-amber-300 flex-shrink-0"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Chat List Overlay */}
      {showChatList && (
        <div className="fixed inset-0 bg-background/50 z-10" onClick={() => setShowChatList(false)}>
          <div className="w-80 h-full bg-card border-r border-border" onClick={e => e.stopPropagation()}>
            <ChatList
              currentChatId={currentChatId}
              onChatSelect={handleChatSelect}
              onNewChat={handleNewChat}
              modelPath={session.modelPath}
            />
          </div>
        </div>
      )}

      {/* Chat Interface + Settings Drawer (or Image placeholder) */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          {isImage ? (
            <div className="flex flex-col items-center justify-center h-full text-center px-8">
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 ${
                isImageEdit ? 'bg-violet-500/10' : 'bg-blue-500/10'
              }`}>
                <ImageIcon className={`h-8 w-8 ${isImageEdit ? 'text-violet-400' : 'text-blue-400'}`} />
              </div>
              <h2 className="text-lg font-semibold mb-2">
                {isImageEdit ? 'Image Edit Server' : 'Image Generation Server'}
              </h2>
              <p className="text-sm text-muted-foreground mb-6 max-w-sm">
                {isImageEdit
                  ? 'This is an image editing model. Use the Image tab to upload images and edit them with prompts.'
                  : 'This is an image generation model. Use the Image tab to generate images from text prompts.'
                }
              </p>
              <button
                onClick={() => setMode('image')}
                className={`px-4 py-2 text-white text-sm rounded-md transition-colors flex items-center gap-2 ${
                  isImageEdit ? 'bg-violet-600 hover:bg-violet-500' : 'bg-blue-600 hover:bg-blue-500'
                }`}
              >
                <ImageIcon className="h-4 w-4" />
                {isImageEdit ? 'Open Image Editor' : 'Open Image Generator'}
              </button>
            </div>
          ) : (
            <ChatInterface
              chatId={currentChatId}
              onNewChat={handleNewChat}
              sessionEndpoint={(session.status === 'running' || session.status === 'standby') ? { host: session.host, port: session.port } : undefined}
              sessionId={session.id}
              sessionStatus={session.status}
              overridesVersion={overridesVersion}
            />
          )}
        </div>
        {showSettings && currentChatId && (
          <ChatSettings
            chatId={currentChatId}
            session={{
              modelName: session.modelName,
              modelPath: session.modelPath,
              host: session.host,
              port: session.port,
              status: session.status,
              pid: session.pid,
              type: session.type,
              remoteUrl: session.remoteUrl,
              modelType: (() => { try { const c = JSON.parse(session.config || '{}'); return c.modelType } catch { return undefined } })()
            }}
            reasoningParser={effectiveReasoningParser}
            onClose={() => setShowSettings(false)}
            onOverridesChanged={() => setOverridesVersion(v => v + 1)}
          />
        )}
        {showServerSettings && (
          <ServerSettingsDrawer
            session={session}
            isRemote={isRemote}
            onClose={() => setShowServerSettings(false)}
            onSessionUpdate={async () => {
              const s = await window.api.sessions.get(sessionId)
              if (s) setSession(s)
            }}
          />
        )}
        {showCache && (
          <div className="w-80 border-l border-border bg-card overflow-y-auto flex-shrink-0">
            <div className="flex items-center justify-between p-3 border-b border-border">
              <h3 className="text-sm font-medium">Cache Management</h3>
              <button onClick={() => setShowCache(false)} className="text-muted-foreground hover:text-foreground text-sm"><X className="h-3.5 w-3.5" /></button>
            </div>
            <div className="p-3">
              <CachePanel endpoint={{ host: session.host, port: session.port }} sessionStatus={session.status} sessionId={session.id} />
            </div>
          </div>
        )}
        {showBenchmark && (
          <div className="w-96 border-l border-border bg-card overflow-y-auto flex-shrink-0">
            <div className="flex items-center justify-between p-3 border-b border-border">
              <h3 className="text-sm font-medium">Benchmark</h3>
              <button onClick={() => setShowBenchmark(false)} className="text-muted-foreground hover:text-foreground text-sm"><X className="h-3.5 w-3.5" /></button>
            </div>
            <div className="p-3">
              <BenchmarkPanel
                sessionId={session.id}
                endpoint={{ host: session.host, port: session.port }}
                modelPath={session.modelPath}
                modelName={session.modelName}
                sessionStatus={session.status}
              />
            </div>
          </div>
        )}
        {showEmbeddings && (
          <div className="w-80 border-l border-border bg-card overflow-y-auto flex-shrink-0">
            <div className="flex items-center justify-between p-3 border-b border-border">
              <h3 className="text-sm font-medium">Embeddings</h3>
              <button onClick={() => setShowEmbeddings(false)} className="text-muted-foreground hover:text-foreground text-sm"><X className="h-3.5 w-3.5" /></button>
            </div>
            <div className="p-3">
              <EmbeddingsPanel
                endpoint={{ host: session.host, port: session.port }}
                sessionStatus={session.status}
                sessionId={session.id}
              />
            </div>
          </div>
        )}
        {showPerformance && (
          <div className="w-80 border-l border-border bg-card overflow-y-auto flex-shrink-0">
            <div className="flex items-center justify-between p-3 border-b border-border">
              <h3 className="text-sm font-medium">Performance</h3>
              <button onClick={() => setShowPerformance(false)} className="text-muted-foreground hover:text-foreground text-sm"><X className="h-3.5 w-3.5" /></button>
            </div>
            <div className="p-3">
              <PerformancePanel
                endpoint={{ host: session.host, port: session.port }}
                sessionStatus={session.status}
              />
            </div>
          </div>
        )}
        {showLogs && (
          <div className="w-96 border-l border-border bg-card flex-shrink-0 flex flex-col">
            <div className="flex items-center justify-between p-3 border-b border-border flex-shrink-0">
              <h3 className="text-sm font-medium">{isRemote ? 'Connection Logs' : 'Server Logs'}</h3>
              <button onClick={() => setShowLogs(false)} className="text-muted-foreground hover:text-foreground text-sm"><X className="h-3.5 w-3.5" /></button>
            </div>
            <LogsPanel sessionId={session.id} sessionStatus={session.status} isRemote={isRemote} />
          </div>
        )}
      </div>
    </div>
  )
}

/** Extracted component so useSessionsContext() is called from a proper hook scope */
function SessionViewLoadBar({ sessionId }: { sessionId: string }) {
  const { loadProgress } = useSessionsContext()
  const progress = loadProgress.get(sessionId)
  return (
    <div className="px-4 py-1.5 border-b border-border bg-card/30 flex-shrink-0">
      <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          className="h-full bg-warning rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progress?.progress ?? 2}%` }}
        />
      </div>
      <p className="text-[10px] text-muted-foreground mt-1">
        {progress?.label ?? 'Starting server...'} {progress ? `(${progress.progress}%)` : ''}
      </p>
    </div>
  )
}
