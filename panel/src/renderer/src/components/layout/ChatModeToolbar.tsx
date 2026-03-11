import { useState, useEffect } from 'react'
import { Settings, Server, X, Info } from 'lucide-react'
import { ChatSettings } from '../chat/ChatSettings'
import { ServerSettingsDrawer } from '../sessions/ServerSettingsDrawer'
import { useSessionsContext } from '../../contexts/SessionsContext'

interface ChatModeToolbarProps {
  activeChatId: string | null
  activeSessionId: string | null
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

export function ChatModeToolbar({ activeChatId, activeSessionId }: ChatModeToolbarProps) {
  const { sessions } = useSessionsContext()
  const [showChatSettings, setShowChatSettings] = useState(false)
  const [showServerSettings, setShowServerSettings] = useState(false)
  const [sessionDetail, setSessionDetail] = useState<SessionDetail | null>(null)
  const [effectiveReasoningParser, setEffectiveReasoningParser] = useState<string | undefined>(undefined)

  // Load full session detail when active session changes
  useEffect(() => {
    if (!activeSessionId) {
      setSessionDetail(null)
      return
    }
    window.api.sessions.get(activeSessionId).then((s: SessionDetail | null) => {
      setSessionDetail(s)
      // Detect reasoning parser
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

  // Keep session status in sync via context
  const contextSession = sessions.find(s => s.id === activeSessionId)
  const displaySession = sessionDetail
    ? { ...sessionDetail, status: contextSession?.status || sessionDetail.status, port: contextSession?.port || sessionDetail.port }
    : null

  if (!activeChatId || !displaySession) return null

  const isRemote = displaySession.type === 'remote'
  const shortName = displaySession.modelName || displaySession.modelPath.split('/').pop() || 'Model'
  const isRunning = displaySession.status === 'running'
  const isLoading = displaySession.status === 'loading'

  return (
    <>
      {/* Compact toolbar */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-border bg-card/50 flex-shrink-0">
        {/* Model info */}
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
            isRunning ? 'bg-success' : isLoading ? 'bg-warning animate-pulse' : 'bg-muted-foreground'
          }`} />
          <span className="text-xs font-medium truncate">{shortName}</span>
          {isRemote && (
            <span className="text-[10px] bg-primary/15 text-primary px-1.5 py-0.5 rounded flex-shrink-0">Remote</span>
          )}
          <span className="text-[10px] text-muted-foreground flex-shrink-0">
            {isRemote ? displaySession.remoteUrl : `${displaySession.host}:${displaySession.port}`}
          </span>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            onClick={() => { setShowChatSettings(!showChatSettings); setShowServerSettings(false) }}
            className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${
              showChatSettings ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent hover:text-foreground'
            }`}
            title="Chat inference settings (temperature, system prompt, tools, etc.)"
          >
            <Settings className="h-3 w-3" />
            <span className="hidden sm:inline">Chat</span>
          </button>
          <button
            onClick={() => { setShowServerSettings(!showServerSettings); setShowChatSettings(false) }}
            className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${
              showServerSettings ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent hover:text-foreground'
            }`}
            title={isRemote ? 'Connection settings' : 'Server settings'}
          >
            <Server className="h-3 w-3" />
            <span className="hidden sm:inline">{isRemote ? 'Connection' : 'Server'}</span>
          </button>
        </div>
      </div>

      {/* Settings drawers — rendered as siblings, positioned in parent flex */}
      {showChatSettings && (
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
      {showServerSettings && (
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
