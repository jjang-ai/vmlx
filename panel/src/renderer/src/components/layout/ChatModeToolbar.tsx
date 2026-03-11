import { useState, useEffect, useRef } from 'react'
import { Settings, Server, Play, Square, ChevronDown, Loader2 } from 'lucide-react'
import { ChatSettings } from '../chat/ChatSettings'
import { ServerSettingsDrawer } from '../sessions/ServerSettingsDrawer'
import { useSessionsContext, type SessionSummary } from '../../contexts/SessionsContext'

interface ChatModeToolbarProps {
  activeChatId: string | null
  activeSessionId: string | null
  onSessionChange: (sessionId: string) => void
}

interface SessionDetail {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading'
  config: string
  type?: 'local' | 'remote'
  remoteUrl?: string
  remoteModel?: string
}

export function ChatModeToolbar({ activeChatId, activeSessionId, onSessionChange }: ChatModeToolbarProps) {
  const { sessions, refreshSessions } = useSessionsContext()
  const [showChatSettings, setShowChatSettings] = useState(false)
  const [showServerSettings, setShowServerSettings] = useState(false)
  const [showModelPicker, setShowModelPicker] = useState(false)
  const [sessionDetail, setSessionDetail] = useState<SessionDetail | null>(null)
  const [effectiveReasoningParser, setEffectiveReasoningParser] = useState<string | undefined>(undefined)
  const pickerRef = useRef<HTMLDivElement>(null)

  // Load full session detail when active session changes
  useEffect(() => {
    if (!activeSessionId) {
      setSessionDetail(null)
      return
    }
    window.api.sessions.get(activeSessionId).then((s: SessionDetail | null) => {
      setSessionDetail(s)
      if (s) {
        try {
          const cfg = s.config ? JSON.parse(s.config) : {}
          if (cfg.reasoningParser && cfg.reasoningParser !== 'auto') {
            setEffectiveReasoningParser(cfg.reasoningParser)
          } else if (!s.modelPath.startsWith('remote://')) {
            window.api.models.detectConfig(s.modelPath).then((detected: any) => {
              setEffectiveReasoningParser(detected?.reasoningParser || undefined)
            }).catch(() => {})
          }
        } catch { /* ignore */ }
      }
    }).catch(() => {})
  }, [activeSessionId])

  // Close model picker on outside click
  useEffect(() => {
    if (!showModelPicker) return
    const handler = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setShowModelPicker(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showModelPicker])

  // Keep session status in sync via context
  const contextSession = sessions.find(s => s.id === activeSessionId)
  const displaySession = sessionDetail
    ? { ...sessionDetail, status: contextSession?.status || sessionDetail.status, port: contextSession?.port || sessionDetail.port }
    : null

  const isRemote = displaySession?.type === 'remote'
  const shortName = displaySession
    ? (displaySession.modelName || displaySession.modelPath.split('/').pop() || 'Model')
    : 'No model selected'
  const isRunning = displaySession?.status === 'running'
  const isLoading = displaySession?.status === 'loading'
  const isStopped = displaySession?.status === 'stopped' || displaySession?.status === 'error'

  const handleStart = async () => {
    if (!activeSessionId) return
    await window.api.sessions.start(activeSessionId)
  }

  const handleStop = async () => {
    if (!activeSessionId) return
    await window.api.sessions.stop(activeSessionId)
  }

  const handleSelectSession = (session: SessionSummary) => {
    onSessionChange(session.id)
    setShowModelPicker(false)
  }

  // No chat selected — don't render toolbar
  if (!activeChatId) return null

  return (
    <>
      {/* Compact toolbar */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border bg-card/50 flex-shrink-0">
        {/* Model selector */}
        <div className="relative" ref={pickerRef}>
          <button
            onClick={() => setShowModelPicker(!showModelPicker)}
            className="flex items-center gap-1.5 text-xs px-2 py-1 rounded border border-border hover:bg-accent transition-colors min-w-0 max-w-[280px]"
          >
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              isRunning ? 'bg-success' : isLoading ? 'bg-warning animate-pulse' : 'bg-muted-foreground'
            }`} />
            <span className="font-medium truncate">{shortName}</span>
            {isRemote && (
              <span className="text-[10px] bg-primary/15 text-primary px-1 py-0.5 rounded flex-shrink-0">R</span>
            )}
            <ChevronDown className="h-3 w-3 text-muted-foreground flex-shrink-0" />
          </button>

          {/* Model picker dropdown */}
          {showModelPicker && sessions.length > 0 && (
            <div className="absolute left-0 top-full mt-1 w-72 bg-popover border border-border rounded-md shadow-lg z-30 py-1 max-h-64 overflow-y-auto">
              {sessions.map(s => {
                const name = s.modelName || s.modelPath.split('/').pop() || s.modelPath
                const isActive = s.id === activeSessionId
                return (
                  <button
                    key={s.id}
                    onClick={() => handleSelectSession(s)}
                    className={`w-full flex items-center gap-2 px-3 py-2 text-left text-xs hover:bg-accent transition-colors ${
                      isActive ? 'bg-accent/50' : ''
                    }`}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                      s.status === 'running' ? 'bg-success'
                        : s.status === 'loading' ? 'bg-warning animate-pulse'
                        : 'bg-muted-foreground'
                    }`} />
                    <span className="flex-1 truncate font-medium">{name}</span>
                    {s.type === 'remote' && (
                      <span className="text-[10px] bg-primary/15 text-primary px-1 py-0.5 rounded">Remote</span>
                    )}
                    <span className="text-[10px] text-muted-foreground">
                      {s.status === 'running' ? 'Running' : s.status === 'loading' ? 'Loading' : 'Stopped'}
                    </span>
                  </button>
                )
              })}
            </div>
          )}
        </div>

        {/* Start / Stop controls */}
        {displaySession && (
          <div className="flex items-center gap-1">
            {isStopped && (
              <button
                onClick={handleStart}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-success text-success-foreground hover:bg-success/90 transition-colors"
                title={isRemote ? 'Connect' : 'Start model'}
              >
                <Play className="h-3 w-3" />
                {isRemote ? 'Connect' : 'Start'}
              </button>
            )}
            {isLoading && (
              <span className="flex items-center gap-1 text-xs px-2 py-1 text-warning">
                <Loader2 className="h-3 w-3 animate-spin" />
                {isRemote ? 'Connecting...' : 'Starting...'}
              </span>
            )}
            {isRunning && (
              <button
                onClick={handleStop}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
                title={isRemote ? 'Disconnect' : 'Stop model'}
              >
                <Square className="h-3 w-3" />
                {isRemote ? 'Disconnect' : 'Stop'}
              </button>
            )}
          </div>
        )}

        {/* Spacer */}
        <div className="flex-1" />

        {/* Settings buttons */}
        {displaySession && (
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={() => { setShowChatSettings(!showChatSettings); setShowServerSettings(false) }}
              className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${
                showChatSettings ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
              title="Chat inference settings (temperature, system prompt, tools, etc.)"
            >
              <Settings className="h-3 w-3" />
              Chat
            </button>
            <button
              onClick={() => { setShowServerSettings(!showServerSettings); setShowChatSettings(false) }}
              className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${
                showServerSettings ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
              title={isRemote ? 'Connection settings' : 'Server settings'}
            >
              <Server className="h-3 w-3" />
              {isRemote ? 'Connection' : 'Server'}
            </button>
          </div>
        )}
      </div>

      {/* Settings drawers */}
      {showChatSettings && activeChatId && displaySession && (
        <div className="absolute right-0 top-0 bottom-0 z-20">
          <ChatSettings
            chatId={activeChatId}
            session={{
              modelName: displaySession.modelName,
              modelPath: displaySession.modelPath,
              host: displaySession.host,
              port: displaySession.port,
              status: displaySession.status,
              pid: displaySession.pid,
              type: displaySession.type,
              remoteUrl: displaySession.remoteUrl,
            }}
            reasoningParser={effectiveReasoningParser}
            onClose={() => setShowChatSettings(false)}
          />
        </div>
      )}
      {showServerSettings && displaySession && (
        <div className="absolute right-0 top-0 bottom-0 z-20">
          <ServerSettingsDrawer
            session={displaySession}
            isRemote={isRemote}
            onClose={() => setShowServerSettings(false)}
            onSessionUpdate={async () => {
              const s = await window.api.sessions.get(activeSessionId!)
              if (s) setSessionDetail(s)
            }}
          />
        </div>
      )}
    </>
  )
}
