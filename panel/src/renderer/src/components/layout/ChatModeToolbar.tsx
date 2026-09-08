import { useState, useEffect, useRef, useMemo } from 'react'
import { Settings, Server, Play, Square, ChevronDown, Loader2, Plus, Globe, X, RotateCw } from 'lucide-react'
import { ChatSettings } from '../chat/ChatSettings'
import { ServerSettingsDrawer } from '../sessions/ServerSettingsDrawer'
import { useSessionsContext, type SessionSummary } from '../../contexts/SessionsContext'
import { isImageSession } from '../../../../shared/sessionUtils'
import {
  reasoningParserIsEnabled,
  resolveEffectiveReasoningParser,
} from '../../../../shared/reasoningParserAliases'
import { useTranslation } from '../../i18n'

interface ChatModeToolbarProps {
  activeChatId: string | null
  activeSessionId: string | null
  onSessionChange: (sessionId: string) => void
  onOverridesChanged?: () => void
}

interface SessionDetail {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  config: string
  type?: 'local' | 'remote'
  remoteUrl?: string
  remoteModel?: string
}

export function ChatModeToolbar({ activeChatId, activeSessionId, onSessionChange, onOverridesChanged }: ChatModeToolbarProps) {
  const { t } = useTranslation()
  const { sessions: allSessions } = useSessionsContext()

  // Filter out image sessions — they belong in the Image tab, not chat
  const sessions = useMemo(() => allSessions.filter(s => !isImageSession(s)), [allSessions])

  const [showChatSettings, setShowChatSettings] = useState(false)
  const [showServerSettings, setShowServerSettings] = useState(false)
  const [showModelPicker, setShowModelPicker] = useState(false)
  const [showRemoteForm, setShowRemoteForm] = useState(false)
  const [sessionDetail, setSessionDetail] = useState<SessionDetail | null>(null)
  const [effectiveReasoningParser, setEffectiveReasoningParser] = useState<string | undefined>(undefined)
  const pickerRef = useRef<HTMLDivElement>(null)

  // Remote endpoint quick-connect state
  const [remoteUrl, setRemoteUrl] = useState('')
  const [remoteModel, setRemoteModel] = useState('')
  const [remoteApiKey, setRemoteApiKey] = useState('')
  const [remoteConnecting, setRemoteConnecting] = useState(false)
  const [remoteError, setRemoteError] = useState<string | null>(null)

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
          if (!s.modelPath.startsWith('remote://')) {
            window.api.models.detectConfig(s.modelPath).then((detected: any) => {
              const effective = resolveEffectiveReasoningParser({
                configuredParser: cfg.reasoningParser,
                detectedParser: detected?.reasoningParser,
                supportsThinking: detected?.supportsThinking,
              })
              setEffectiveReasoningParser(
                reasoningParserIsEnabled(effective) ? effective : undefined,
              )
            }).catch((err) => console.error('Failed to load session info:', err))
          } else {
            const effective = resolveEffectiveReasoningParser({
              configuredParser: cfg.reasoningParser,
            })
            setEffectiveReasoningParser(
              reasoningParserIsEnabled(effective) ? effective : undefined,
            )
          }
        } catch { /* ignore */ }
      }
    }).catch((err) => console.error('Failed to load session info:', err))
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
    ? {
        ...sessionDetail,
        status: contextSession?.status ?? sessionDetail.status,
        modelPath: contextSession?.modelPath ?? sessionDetail.modelPath,
        modelName: contextSession?.modelName ?? sessionDetail.modelName,
        host: contextSession?.host ?? sessionDetail.host,
        port: contextSession?.port ?? sessionDetail.port,
        // When a context row exists its PID is authoritative even when it is
        // undefined: session:stopped deliberately clears the old process ID.
        pid: contextSession ? contextSession.pid : sessionDetail.pid,
        type: contextSession?.type ?? sessionDetail.type,
        remoteUrl: contextSession?.remoteUrl ?? sessionDetail.remoteUrl,
        config: contextSession?.config ?? sessionDetail.config,
      }
    : null

  const isRemote = displaySession?.type === 'remote'
  const shortName = displaySession
    ? (displaySession.modelName || displaySession.modelPath.split('/').pop() || t('layout.chatToolbar.modelFallback'))
    : t('layout.chatToolbar.noModelSelected')
  const isRunning = displaySession?.status === 'running'
  const isStandby = displaySession?.status === 'standby'
  const isLoading = displaySession?.status === 'loading'
  const isError = displaySession?.status === 'error'
  const isStopped = displaySession?.status === 'stopped' || isError

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

  const handleConnectRemote = async () => {
    if (!remoteUrl.trim() || !remoteModel.trim()) return
    setRemoteError(null)
    setRemoteConnecting(true)

    try {
      const remoteResult = await window.api.sessions.createRemote({
        remoteUrl: remoteUrl.trim(),
        remoteApiKey: remoteApiKey.trim() || undefined,
        remoteModel: remoteModel.trim(),
      })
      if (!remoteResult.success) {
        throw new Error(remoteResult.error || t('layout.chatToolbar.remoteCreateFailed'))
      }
      const session = remoteResult.session

      const result = await window.api.sessions.start(session.id)
      if (result.success) {
        onSessionChange(session.id)
        setShowRemoteForm(false)
        setShowModelPicker(false)
        setRemoteUrl('')
        setRemoteModel('')
        setRemoteApiKey('')
      } else {
        setRemoteError(result.error || t('layout.chatToolbar.remoteConnectFailed'))
      }
    } catch (error) {
      setRemoteError((error as Error).message)
    } finally {
      setRemoteConnecting(false)
    }
  }

  // No chat selected — don't render toolbar
  if (!activeChatId) return null

  // No sessions at all — show a prompt with both local and remote options
  if (!displaySession && sessions.length === 0) {
    return (
      <div className="flex flex-col border-b border-border bg-card/50 flex-shrink-0">
        <div className="flex items-center gap-2 px-3 py-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground flex-shrink-0" />
          <span className="text-xs text-muted-foreground">{t('layout.chatToolbar.noModels')}</span>
          <button
            onClick={() => {
              const event = new CustomEvent('vmlx:navigate', { detail: { mode: 'chat', panel: 'create', modelPath: null } })
              window.dispatchEvent(event)
            }}
            className="text-xs text-primary hover:text-primary/80 font-medium"
          >
            {t('layout.chatToolbar.addLocalModel')}
          </button>
          <span className="text-xs text-muted-foreground">{t('common.or')}</span>
          <button
            onClick={() => { setShowRemoteForm(!showRemoteForm); setRemoteError(null) }}
            className="text-xs text-primary hover:text-primary/80 font-medium"
          >
            {t('layout.chatToolbar.connectRemote')}
          </button>
        </div>
        {showRemoteForm && (
          <div className="px-3 pb-2 space-y-2">
            <input
              autoFocus
              value={remoteUrl}
              onChange={e => setRemoteUrl(e.target.value)}
              placeholder={t('layout.chatToolbar.apiUrlPlaceholder')}
              className="w-full px-2 py-1.5 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
            />
            <div className="flex gap-2">
              <input
                value={remoteModel}
                onChange={e => setRemoteModel(e.target.value)}
                placeholder={t('layout.chatToolbar.modelNamePlaceholder')}
                className="flex-1 px-2 py-1.5 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <input
                type="password"
                value={remoteApiKey}
                onChange={e => setRemoteApiKey(e.target.value)}
                placeholder={t('layout.chatToolbar.apiKeyPlaceholder')}
                className="flex-1 px-2 py-1.5 bg-background border border-input rounded text-xs font-mono focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>
            {remoteError && (
              <p className="text-[10px] text-destructive">{remoteError}</p>
            )}
            <div className="flex gap-2">
              <button
                onClick={handleConnectRemote}
                disabled={remoteConnecting || !remoteUrl.trim() || !remoteModel.trim()}
                className="flex items-center gap-1.5 px-3 py-1.5 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 transition-colors"
              >
                {remoteConnecting ? (
                  <><Loader2 className="h-3 w-3 animate-spin" /> {t('common.connecting')}</>
                ) : (
                  <><Globe className="h-3 w-3" /> {t('common.connect')}</>
                )}
              </button>
              <button
                onClick={() => setShowRemoteForm(false)}
                className="text-xs text-muted-foreground hover:text-foreground px-2 py-1.5"
              >
                {t('common.cancel')}
              </button>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <>
      {/* Compact toolbar */}
      <div className="flex flex-wrap items-center gap-2 px-3 py-1.5 border-b border-border bg-card/50 flex-shrink-0">
        {/* Model selector */}
        <div className="relative flex-1 min-w-0" ref={pickerRef}>
          <button
            onClick={() => setShowModelPicker(!showModelPicker)}
            className="flex w-full items-center gap-1.5 text-xs px-2 py-1 rounded border border-border hover:bg-accent transition-colors min-w-0 max-w-[280px]"
          >
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
              isRunning ? 'bg-success' : isStandby ? 'bg-blue-400' : isLoading ? 'bg-warning animate-pulse' : 'bg-muted-foreground'
            }`} />
            <span className="font-medium truncate">{shortName}</span>
            {isRemote && (
              <span className="text-[10px] bg-primary/15 text-primary px-1 py-0.5 rounded flex-shrink-0">R</span>
            )}
            <ChevronDown className="h-3 w-3 text-muted-foreground flex-shrink-0" />
          </button>

          {/* Model picker dropdown */}
          {showModelPicker && (
            <div className="absolute left-0 top-full mt-1 w-80 bg-popover border border-border rounded-md shadow-lg z-30 py-1 max-h-96 overflow-y-auto">
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
                        : s.status === 'standby' ? 'bg-blue-400'
                        : s.status === 'loading' ? 'bg-warning animate-pulse'
                        : 'bg-muted-foreground'
                    }`} />
                    <span className="flex-1 truncate font-medium">{name}</span>
                    {s.type === 'remote' && (
                      <span className="text-[10px] bg-primary/15 text-primary px-1 py-0.5 rounded">{t('layout.chatToolbar.remoteBadge')}</span>
                    )}
                    <span className="text-[10px] text-muted-foreground">
                      {s.status === 'running' ? t('status.running') : s.status === 'standby' ? t('status.sleeping') : s.status === 'loading' ? t('status.loading') : t('status.stopped')}
                    </span>
                  </button>
                )
              })}

              {/* Divider + add options */}
              <div className="border-t border-border my-1" />
              <button
                onClick={() => {
                  setShowModelPicker(false)
                  const event = new CustomEvent('vmlx:navigate', { detail: { mode: 'chat', panel: 'create', modelPath: null } })
                  window.dispatchEvent(event)
                }}
                className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              >
                <Plus className="h-3 w-3" />
                {t('layout.chatToolbar.addLocalModel')}
              </button>
              <button
                onClick={() => { setShowRemoteForm(true); setRemoteError(null) }}
                className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              >
                <Globe className="h-3 w-3" />
                {t('layout.chatToolbar.connectRemoteEndpoint')}
              </button>

              {/* Inline remote form */}
              {showRemoteForm && (
                <div className="border-t border-border mt-1 p-3 space-y-2" onClick={e => e.stopPropagation()}>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium">{t('layout.chatToolbar.connectRemote')}</span>
                    <button onClick={() => setShowRemoteForm(false)} className="text-muted-foreground hover:text-foreground" aria-label={t('common.close')} title={t('common.close')}>
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                  <input
                    autoFocus
                    value={remoteUrl}
                    onChange={e => setRemoteUrl(e.target.value)}
                    placeholder={t('layout.chatToolbar.apiUrlPlaceholder')}
                    className="w-full px-2 py-1.5 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
                  />
                  <input
                    value={remoteModel}
                    onChange={e => setRemoteModel(e.target.value)}
                    placeholder={t('layout.chatToolbar.modelNamePlaceholder')}
                    className="w-full px-2 py-1.5 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
                  />
                  <input
                    type="password"
                    value={remoteApiKey}
                    onChange={e => setRemoteApiKey(e.target.value)}
                    placeholder={t('layout.chatToolbar.apiKeyOptionalPlaceholder')}
                    className="w-full px-2 py-1.5 bg-background border border-input rounded text-xs font-mono focus:outline-none focus:ring-1 focus:ring-ring"
                  />
                  {remoteError && (
                    <p className="text-[10px] text-destructive">{remoteError}</p>
                  )}
                  <button
                    onClick={handleConnectRemote}
                    disabled={remoteConnecting || !remoteUrl.trim() || !remoteModel.trim()}
                    className="w-full flex items-center justify-center gap-1.5 px-2 py-1.5 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50 transition-colors"
                  >
                    {remoteConnecting ? (
                      <><Loader2 className="h-3 w-3 animate-spin" /> {t('common.connecting')}</>
                    ) : (
                      <><Globe className="h-3 w-3" /> {t('common.connect')}</>
                    )}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Start / Stop / Restart controls */}
        {displaySession && (
          <div className="flex items-center gap-1 flex-shrink-0">
            {isError && (
              <button
                onClick={handleStart}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-destructive/10 text-destructive hover:bg-destructive/20 border border-destructive/30 transition-colors"
                title={isRemote ? t('common.reconnect') : t('sessions.toolbar.restartModel')}
              >
                <RotateCw className="h-3 w-3" />
                {isRemote ? t('common.reconnect') : t('sessions.toolbar.restart')}
              </button>
            )}
            {isStopped && !isError && (
              <button
                onClick={handleStart}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-success text-success-foreground hover:bg-success/90 transition-colors"
                title={isRemote ? t('common.connect') : t('sessions.toolbar.startModel')}
              >
                <Play className="h-3 w-3" />
                {isRemote ? t('common.connect') : t('common.start')}
              </button>
            )}
            {isLoading && (
              <>
                <span className="flex items-center gap-1 text-xs px-2 py-1 text-warning">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  {isRemote ? t('common.connecting') : t('common.loading')}
                </span>
                <button
                  onClick={handleStop}
                  className="text-xs px-1.5 py-0.5 rounded text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                  title={t('common.cancel')}
                >
                  <X className="h-3 w-3" />
                </button>
              </>
            )}
            {isRunning && (
              <button
                onClick={handleStop}
                className="flex items-center gap-1 text-xs px-2 py-1 rounded text-muted-foreground hover:bg-destructive/10 hover:text-destructive transition-colors"
                title={isRemote ? t('common.disconnect') : t('sessions.toolbar.stopModel')}
              >
                <Square className="h-3 w-3" />
                {isRemote ? t('common.disconnect') : t('common.stop')}
              </button>
            )}
          </div>
        )}

        {/* Spacer */}
        <div className="flex-1 max-[800px]:hidden" />

        {/* Settings buttons */}
        {displaySession && (
          <div className="flex items-center gap-1 flex-shrink-0 max-[800px]:basis-full max-[800px]:justify-end">
            <button
              data-vmlx-control="chat-settings"
              onClick={() => { setShowChatSettings(!showChatSettings); setShowServerSettings(false) }}
              className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${
                showChatSettings ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
              title={t('sessions.view.chatSettingsTitle')}
            >
              <Settings className="h-3 w-3" />
              {t('sessions.view.chatLabel')}
            </button>
            <button
              data-vmlx-control="server-settings"
              aria-pressed={showServerSettings}
              onClick={() => { setShowServerSettings(!showServerSettings); setShowChatSettings(false) }}
              className={`flex items-center gap-1 text-xs px-2 py-1 rounded transition-colors ${
                showServerSettings ? 'bg-accent text-foreground' : 'text-muted-foreground hover:bg-accent hover:text-foreground'
              }`}
              title={isRemote ? t('sessions.view.connectionTitle') : t('sessions.view.serverSettingsTitle')}
            >
              <Server className="h-3 w-3" />
              {isRemote ? t('sessions.view.connection') : t('sessions.view.server')}
            </button>
          </div>
        )}
      </div>

      {/* Settings drawers */}
      {showChatSettings && activeChatId && displaySession && (
        <div className="absolute right-0 top-0 bottom-0 z-20 flex w-full max-w-80 justify-end">
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
              modelType: (() => { try { return JSON.parse(displaySession.config || '{}').modelType } catch { return undefined } })(),
              config: displaySession.config,
            }}
            reasoningParser={effectiveReasoningParser}
            onClose={() => setShowChatSettings(false)}
            onOverridesChanged={onOverridesChanged}
          />
        </div>
      )}
      {showServerSettings && displaySession && (
        <div className="absolute right-0 top-0 bottom-0 z-20 flex w-full max-w-96 justify-end">
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
