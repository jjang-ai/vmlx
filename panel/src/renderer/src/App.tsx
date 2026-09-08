import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { MessageSquare, ArrowLeft, Terminal } from 'lucide-react'
import { TitleBar } from './components/layout/TitleBar'
import { ConsoleSubnavigation } from './components/layout/ConsoleSubnavigation'
import { DownloadTab } from './components/sessions/DownloadTab'
import { ConsoleSidebar } from './components/layout/ConsoleSidebar'
import { SessionDashboard } from './components/sessions/SessionDashboard'
import { CreateSession } from './components/sessions/CreateSession'
import { SessionView } from './components/sessions/SessionView'
import { SessionSettings } from './components/sessions/SessionSettings'
import { ChatInterface } from './components/chat/ChatInterface'
import { SetupScreen } from './components/setup/SetupScreen'
import { ToastProvider } from './components/Toast'
import { DownloadStatusBar } from './components/DownloadStatusBar'
import { UpdateBanner } from './components/UpdateBanner'
import { MtpComponentUpdatePrompt } from './components/MtpComponentUpdatePrompt'
import { useAppState } from './contexts/AppStateContext'
import { useSessionsContext } from './contexts/SessionsContext'
import { ChatModeToolbar } from './components/layout/ChatModeToolbar'
import { ToolsDashboard } from './components/tools/ToolsDashboard'
import { ModelInspector } from './components/tools/ModelInspector'
import { ModelDoctor } from './components/tools/ModelDoctor'
import { ModelConverter } from './components/tools/ModelConverter'
import { ApiDashboard } from './components/api/ApiDashboard'
import { ImageTab } from './components/image/ImageTab'
import { isImageSession, sessionMatchesModelPath } from '../../shared/sessionUtils'
import { useTranslation, LOCALES, LOCALE_NAMES, LOCALE_FLAGS } from './i18n'

function App() {
  const { t } = useTranslation()
  const [setupDone, setSetupDone] = useState(false)
  const [checkingSetup, setCheckingSetup] = useState(true)
  const { state, dispatch, setMode, openChat } = useAppState()
  const { sessions: allSessions } = useSessionsContext()

  // For chat mode, exclude image sessions — they belong in the Image tab
  const sessions = useMemo(() => allSessions.filter(s => !isImageSession(s)), [allSessions])

  // Check if engine is already installed (skip setup screen if so)
  useEffect(() => {
    window.api.engine.checkInstallation()
      .then((result: any) => {
        if (result.installed) setSetupDone(true)
      })
      .catch((err) => console.error('Installation check failed:', err))
      .finally(() => setCheckingSetup(false))
  }, [])

  // Clear stale chat locks on mount
  useEffect(() => {
    window.api.chat.clearAllLocks().catch((err) => console.error('Failed to clear chat locks:', err))
  }, [])

  // Listen for navigation events from child components (e.g. toolbar "Add a model")
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      if (detail?.mode) setMode(detail.mode)
      if (detail?.panel) {
        if (detail.mode === 'tools') {
          dispatch({ type: 'SET_TOOLS_PANEL', panel: detail.panel, modelPath: detail.modelPath })
        } else {
          dispatch({ type: 'SET_SERVER_PANEL', panel: detail.panel, sessionId: detail.sessionId, modelPath: detail.modelPath })
        }
      }
    }
    window.addEventListener('vmlx:navigate', handler)
    return () => window.removeEventListener('vmlx:navigate', handler)
  }, [setMode, dispatch])

  // If the active session was deleted, fall back to another session
  useEffect(() => {
    if (state.activeSessionId && sessions.length > 0 && !sessions.find(s => s.id === state.activeSessionId)) {
      // Active session no longer exists — switch to first available
      const fallback = sessions.find(s => s.status === 'running') || sessions[0]
      if (fallback && state.activeChatId) {
        openChat(state.activeChatId, fallback.id)
      }
    }
  }, [sessions, state.activeSessionId, state.activeChatId, openChat])

  // Resolve the endpoint for the active session. A chat can stay pinned to a
  // stopped DUPLICATE session of the same model (identity split across path
  // prefixes: ~/models symlink vs /Volumes real dir spawns twin session rows)
  // while the usable twin runs — prefer the usable same-identity session so
  // the composer/banner reflect reality instead of the stale pin.
  const pinnedSession = sessions.find(s => s.id === state.activeSessionId)
  const sessionUsable = (s: typeof sessions[number] | undefined) =>
    !!s && (s.status === 'running' || s.status === 'loading' || s.status === 'standby')
  const activeSession = (pinnedSession && !sessionUsable(pinnedSession))
    ? sessions.find(s => sessionMatchesModelPath(s.modelPath, pinnedSession.modelPath) && sessionUsable(s)) || pinnedSession
    : pinnedSession
  // Standby sessions still have a live process on their port — JIT middleware
  // auto-wakes. Loading sessions expose their endpoint too: a message sent
  // mid-load queues exactly once in the main process (visible load progress,
  // Stop cancels) instead of being rejected at the input box.
  const sessionEndpoint = (activeSession?.status === 'running' || activeSession?.status === 'standby' || activeSession?.status === 'loading')
    ? { host: activeSession.host, port: activeSession.port }
    : undefined

  const handleChatSelect = useCallback((chatId: string, modelPath: string) => {
    // Empty chatId means deselect (e.g. after deleting the active chat)
    if (!chatId) {
      dispatch({ type: 'CLOSE_CHAT' })
      return
    }

    // Match the chat's model to a session by real model IDENTITY, not raw path
    // equality. The same model is stored under different path prefixes — a
    // ~/.mlxstudio/models/X symlink vs the real /Volumes/…/org/X, or an HF repo
    // id vs its resolved local dir — so `s.modelPath === modelPath` misses the
    // right session and reverts the chat to an arbitrary session of a DIFFERENT
    // model (disabling the composer). `sessionMatchesModelPath` mirrors the
    // backend send-path resolver in ipc/chat.ts.
    const sameModel = (s: typeof sessions[number]) => sessionMatchesModelPath(s.modelPath, modelPath)
    const usable = (s: typeof sessions[number]) =>
      s.status === 'running' || s.status === 'loading' || s.status === 'standby'
    // Prefer this model's ready session, then this model in any state, then any
    // ready session, then anything — never silently jump to a different model's
    // session when this model's own session exists.
    const session =
      sessions.find(s => sameModel(s) && usable(s)) ||
      sessions.find(s => sameModel(s)) ||
      sessions.find(s => usable(s)) ||
      sessions[0]

    if (session) {
      openChat(chatId, session.id)
    } else {
      // Truly no sessions at all — open chat without one, toolbar will handle
      dispatch({ type: 'OPEN_CHAT', chatId, sessionId: '' })
    }
  }, [sessions, openChat, dispatch])

  const handleNewChat = useCallback(async () => {
    // mlxstudio #60: when the user has explicitly switched to session A in
    // the sidebar, "+ New Chat" must create a chat against session A — not
    // whichever running session happens to be first in the array. Earlier
    // logic always picked `sessions.find(s => running)` which was a coin
    // flip after loading a second model. Now we honor `activeSessionId`
    // (set by `openChat` / `handleSessionChange` whenever the user
    // navigates to a chat) and only fall back to "first running" or
    // "first session" when nothing is active yet.
    const explicit = state.activeSessionId
      ? sessions.find(s => s.id === state.activeSessionId)
      : null
    const running = sessions.find(s => s.status === 'running')
    // A pinned session that is no longer running must not win over a session
    // that is: the new chat would open against a stopped server and sit with
    // a disabled composer while another model is up (seen live after
    // stopping one session and starting another from the Server tab).
    const explicitUsable = explicit && (explicit.status === 'running' || explicit.status === 'loading' || !running)
    const target = (explicitUsable ? explicit : null) || running || explicit || sessions[0]

    if (!target) {
      // No sessions — switch to server mode to create one
      setMode('server')
      dispatch({ type: 'SET_SERVER_PANEL', panel: 'create' })
      return
    }

    const modelName = target.modelName || target.modelPath.split('/').pop() || t('chat.interface.newChat')
    const result = await window.api.chat.create(
      t('chat.quickStart.chatWithModel', { model: modelName }),
      target.modelPath,
      undefined,
      target.modelPath
    )
    if (result?.id) {
      openChat(result.id, target.id)
    }
  }, [sessions, state.activeSessionId, setMode, dispatch, openChat, t])

  const handleSessionChange = useCallback(async (sessionId: string) => {
    if (!state.activeChatId) return
    // Find the new session to get its modelPath
    const newSession = sessions.find(s => s.id === sessionId)
    if (newSession) {
      // Update the chat's model_path in DB so it persists across reloads
      await window.api.chat.update(state.activeChatId, {
        modelId: newSession.modelPath,
        modelPath: newSession.modelPath
      } as any).catch((err) => console.error('Failed to update chat model:', err))
    }
    dispatch({ type: 'OPEN_CHAT', chatId: state.activeChatId, sessionId })
  }, [dispatch, state.activeChatId, sessions])

  // Setup screen
  if (checkingSetup) return null
  if (!setupDone) {
    return (
      <ToastProvider>
        <div className="flex flex-col h-screen bg-background text-foreground">
          <SetupScreen onReady={() => setSetupDone(true)} />
        </div>
      </ToastProvider>
    )
  }

  return (
    <ToastProvider>
      <div className="flex flex-col h-screen bg-background text-foreground">
        <TitleBar />
        <UpdateBanner />
        <MtpComponentUpdatePrompt />
        <DownloadStatusBar />

        <div className="flex flex-1 overflow-hidden">
          <ConsoleSidebar onChatSelect={handleChatSelect} onNewChat={handleNewChat} />

          {/* Main content area */}
          <main className="flex-1 min-w-0 min-h-0 overflow-hidden flex flex-col">
            <ConsoleSubnavigation />
            <div className="flex-1 min-h-0 overflow-hidden">
            {state.mode === 'code' && (
              <div className="flex flex-col items-center justify-center h-full text-center px-8">
                <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 flex items-center justify-center mb-4">
                  <Terminal className="h-8 w-8 text-emerald-500" />
                </div>
                <h2 className="text-lg font-semibold mb-2">{t('app.code.title')}</h2>
                <p className="text-sm text-muted-foreground max-w-sm">
                  {t('app.code.description')}
                </p>
                <span className="mt-4 px-3 py-1 text-xs font-medium bg-emerald-500/10 text-emerald-500 rounded-full">
                  {t('app.code.comingSoon')}
                </span>
              </div>
            )}

            {state.mode === 'chat' && (
              <ChatModeContent
                activeChatId={state.activeChatId}
                sessionEndpoint={sessionEndpoint}
                sessionStatus={activeSession?.status}
                activeSessionId={state.activeSessionId}
                onNewChat={handleNewChat}
                onSessionChange={handleSessionChange}
              />
            )}

            {state.mode === 'server' && (
              <ServerModeContent />
            )}

            {state.mode === 'tools' && (
              <ToolsModeContent />
            )}
            {state.mode === 'models' && (
              <div className="h-full overflow-auto p-4">
                <DownloadTab onDownloadComplete={() => {
                  // Downloads remain main-process jobs. No automatic load or
                  // route switch when a background transfer finishes.
                  window.dispatchEvent(new CustomEvent('vmlx:models-changed'))
                }} />
              </div>
            )}
            {state.mode === 'image' && (
              <ImageTab />
            )}
            {state.mode === 'api' && (
              <ApiDashboard />
            )}
            </div>
          </main>
        </div>
      </div>
    </ToastProvider>
  )
}

// ─── Chat Mode Content ──────────────────────────────────────────────────────

function ChatModeContent({ activeChatId, sessionEndpoint, sessionStatus, activeSessionId, onNewChat, onSessionChange }: {
  activeChatId: string | null
  sessionEndpoint?: { host: string; port: number }
  sessionStatus?: string
  activeSessionId: string | null
  onNewChat: () => void
  onSessionChange: (sessionId: string) => void
}) {
  const [overridesVersion, setOverridesVersion] = useState(0)

  if (!activeChatId) {
    return <ChatEmptyState onNewChat={onNewChat} />
  }

  return (
    <div className="flex flex-col h-full relative">
      <ChatModeToolbar
        activeChatId={activeChatId}
        activeSessionId={activeSessionId}
        onSessionChange={onSessionChange}
        onOverridesChanged={() => setOverridesVersion(v => v + 1)}
      />
      <div className="flex-1 overflow-hidden">
        <ChatInterface
          chatId={activeChatId}
          onNewChat={onNewChat}
          sessionEndpoint={sessionEndpoint}
          sessionId={activeSessionId || undefined}
          sessionStatus={sessionStatus}
          overridesVersion={overridesVersion}
        />
      </div>
    </div>
  )
}

// vMLX / mlxstudio — authored by Jinho Jang
function ChatEmptyState({ onNewChat }: { onNewChat: () => void }) {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8 overflow-auto" data-mlx-studio="jinhojang">
      <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4 flex-shrink-0">
        <MessageSquare className="h-8 w-8 text-primary" />
      </div>
      <h2 className="text-lg font-semibold mb-2">{t('chat.interface.emptyStateTitle')}</h2>
      <p className="text-sm text-muted-foreground mb-4 max-w-md">
        {t('chat.quickStart.emptyBody')}
      </p>
      <button
        onClick={onNewChat}
        className="px-4 py-2 bg-primary text-primary-foreground text-sm rounded-md hover:bg-primary/90 transition-colors mb-6"
      >
        {t('chat.interface.newChat')}
      </button>

      <div className="text-left max-w-lg space-y-3 text-xs text-muted-foreground border-t border-border pt-4">
        <p className="font-semibold text-foreground text-sm">{t('chat.quickStart.title')}</p>

        <div>
          <p className="font-medium text-foreground">{t('chat.quickStart.textModelsTitle')}</p>
          <p>{t('chat.quickStart.textModelsBody')}</p>
          <p className="text-muted-foreground/70 mt-0.5">
            {t('chat.quickStart.textModelsRecommended')}
          </p>
        </div>

        <div>
          <p className="font-medium text-foreground">{t('chat.quickStart.visionModelsTitle')}</p>
          <p>{t('chat.quickStart.visionModelsBody')}</p>
          <p className="text-muted-foreground/70 mt-0.5">
            {t('chat.quickStart.visionModelsRecommended')}
          </p>
        </div>

        <div>
          <p className="font-medium text-foreground">{t('chat.quickStart.tipsTitle')}</p>
          <ul className="list-disc list-inside space-y-0.5 text-muted-foreground/80">
            <li>{t('chat.quickStart.tipLooping')}</li>
            <li>{t('chat.quickStart.tipFirstResponse')}</li>
            <li>{t('chat.quickStart.tipJang')}</li>
            <li>{t('chat.quickStart.tipSettings')}</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

// ─── Server Mode Content ────────────────────────────────────────────────────

function ServerModeContent() {
  const { state, dispatch } = useAppState()
  const { serverPanel, serverSessionId, serverInitialModelPath } = state
  const { t, locale, setLocale } = useTranslation()

  return (
    <>
      {serverPanel === 'dashboard' && (
        <SessionDashboard
          onOpenSession={(sessionId) => dispatch({ type: 'SET_SERVER_PANEL', panel: 'session', sessionId })}
          onConfigureSession={(sessionId) => dispatch({ type: 'SET_SERVER_PANEL', panel: 'settings', sessionId })}
          onCreateSession={() => dispatch({ type: 'SET_SERVER_PANEL', panel: 'create' })}
        />
      )}

      {serverPanel === 'create' && (
        <CreateSession
          initialModelPath={serverInitialModelPath}
          onBack={() => dispatch({ type: 'SET_SERVER_PANEL', panel: 'dashboard' })}
          onCreated={(sessionId) => dispatch({ type: 'SET_SERVER_PANEL', panel: 'session', sessionId })}
        />
      )}

      {serverPanel === 'session' && serverSessionId && (
        <SessionView
          sessionId={serverSessionId}
          onBack={() => dispatch({ type: 'SET_SERVER_PANEL', panel: 'dashboard' })}
        />
      )}

      {serverPanel === 'settings' && serverSessionId && (
        <SessionSettings
          sessionId={serverSessionId}
          onBack={() => dispatch({ type: 'SET_SERVER_PANEL', panel: 'dashboard' })}
        />
      )}

      {serverPanel === 'about' && (
        <div className="p-8 overflow-auto h-full">
          <div className="max-w-3xl mx-auto space-y-6">
            <button
              onClick={() => dispatch({ type: 'SET_SERVER_PANEL', panel: 'dashboard' })}
              className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
            >
              <ArrowLeft className="h-3 w-3" />
              {t('common.back')}
            </button>
            <h2 className="text-2xl font-bold">{t('app.about.title')}</h2>
            <p className="text-sm text-muted-foreground">
              {t('app.about.desc')}
            </p>
            <p className="text-xs text-muted-foreground/70">
              {t('app.about.creator')}
            </p>
            <AppVersion />
            <div className="flex gap-4 text-xs">
              <a href="https://mlx.studio" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">mlx.studio</a>
              <a href="https://github.com/jjang-ai/vmlx" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">GitHub</a>
              <a href="https://jangq.ai" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">JANG</a>
              <a href="https://ko-fi.com/jinhojang" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Ko-fi</a>
            </div>
            <div className="border border-border rounded-lg p-5">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-4">{t('app.about.language')}</h3>
              <div className="flex gap-2">
                {LOCALES.map((l) => (
                  <button
                    key={l}
                    onClick={() => setLocale(l)}
                    className={`px-3 py-1.5 text-sm rounded-md border transition-colors ${
                      locale === l
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'border-border hover:bg-accent'
                    }`}
                  >
                    {LOCALE_FLAGS[l]} {LOCALE_NAMES[l]}
                  </button>
                ))}
              </div>
            </div>
            <ApiKeysSection />
          </div>
        </div>
      )}
    </>
  )
}

// ─── Tools Mode Content ─────────────────────────────────────────────────────

function ToolsModeContent() {
  const { state, dispatch } = useAppState()
  const { toolsPanel, toolsModelPath } = state
  const [scannedModels, setScannedModels] = useState<Array<{ name: string; path: string }>>([])

  // Single model scan shared by all sub-panels
  useEffect(() => {
    window.api.models.scan().then(setScannedModels).catch((err) => console.error('Failed to scan models:', err))
  }, [])

  const navigateTo = (panel: 'dashboard' | 'inspector' | 'doctor' | 'converter', modelPath?: string) => {
    dispatch({ type: 'SET_TOOLS_PANEL', panel, modelPath: modelPath !== undefined ? (modelPath || null) : undefined })
  }

  const handleServe = (modelPath?: string) => {
    window.dispatchEvent(new CustomEvent('vmlx:navigate', {
      detail: { mode: 'server', panel: 'create', modelPath }
    }))
  }

  return (
    <>
      {toolsPanel === 'dashboard' && (
        <ToolsDashboard
          onInspect={(path) => navigateTo('inspector', path)}
          onDiagnose={(path) => navigateTo('doctor', path)}
          onConvert={(path) => navigateTo('converter', path)}
          onServe={(path) => handleServe(path)}
        />
      )}

      {toolsPanel === 'inspector' && (
        <ModelInspector
          initialModelPath={toolsModelPath}
          onBack={() => navigateTo('dashboard')}
          models={scannedModels}
        />
      )}

      {toolsPanel === 'doctor' && (
        <ModelDoctor
          initialModelPath={toolsModelPath}
          onBack={() => navigateTo('dashboard')}
          models={scannedModels}
        />
      )}

      {toolsPanel === 'converter' && (
        <ModelConverter
          initialModelPath={toolsModelPath}
          onBack={() => navigateTo('dashboard')}
          onServe={(path) => handleServe(path)}
          models={scannedModels}
        />
      )}
    </>
  )
}

// ─── Shared Components ──────────────────────────────────────────────────────

function AppVersion() {
  const { t } = useTranslation()
  const [version, setVersion] = useState('...')
  useEffect(() => {
    window.api.app.getVersion().then((v: string) => setVersion(v)).catch(() => setVersion('unknown'))
  }, [])
  return (
    <div className="text-xs text-muted-foreground space-y-1">
      <p>{t('app.about.version', { version })}</p>
      <p>{t('app.about.copyright', { year: new Date().getFullYear() })}</p>
    </div>
  )
}

function ApiKeysSection() {
  const { t } = useTranslation()
  const [braveKey, setBraveKey] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [saved, setSaved] = useState(false)
  const [hfSaved, setHfSaved] = useState(false)
  const [hasSavedBraveKey, setHasSavedBraveKey] = useState(false)
  const [hasSavedHfToken, setHasSavedHfToken] = useState(false)
  const [showKey, setShowKey] = useState(false)
  const [showHfKey, setShowHfKey] = useState(false)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    window.api.settings.has('braveApiKey').then((exists) => {
      if (mountedRef.current) setHasSavedBraveKey(exists)
    })
    window.api.settings.has('hf_api_key').then((exists) => {
      if (mountedRef.current) setHasSavedHfToken(exists)
    })
    return () => { mountedRef.current = false }
  }, [])

  const handleSave = async () => {
    const trimmed = braveKey.trim()
    if (trimmed) {
      await window.api.settings.set('braveApiKey', trimmed)
    } else {
      await window.api.settings.delete('braveApiKey')
    }
    setHasSavedBraveKey(!!trimmed)
    setSaved(true)
    setTimeout(() => { if (mountedRef.current) setSaved(false) }, 2000)
  }

  const handleHfSave = async () => {
    const trimmed = hfToken.trim()
    if (trimmed) {
      await window.api.settings.set('hf_api_key', trimmed)
    } else {
      await window.api.settings.delete('hf_api_key')
    }
    setHasSavedHfToken(!!trimmed)
    setHfSaved(true)
    setTimeout(() => { if (mountedRef.current) setHfSaved(false) }, 2000)
  }

  return (
    <div className="border border-border rounded-lg p-5">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-4">{t('app.about.apiKeysTitle')}</h3>
      <div className="space-y-3">
        <div>
          <label className="text-sm font-medium">{t('app.about.braveKey')}</label>
          <p className="text-xs text-muted-foreground mt-0.5 mb-2">
            {t('app.about.braveRequired')}{' '}
            <a
              href="https://brave.com/search/api/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              {t('app.about.getFreeKey')}
            </a>
          </p>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={braveKey}
                onChange={e => { setBraveKey(e.target.value); setSaved(false) }}
                placeholder={hasSavedBraveKey ? t('app.about.savedKeyPlaceholder') : 'BSA...'}
                className="w-full px-3 py-2 bg-background border border-input rounded text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ring pr-10"
              />
              <button
                onClick={() => setShowKey(!showKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground text-xs"
                title={showKey ? t('app.about.hide') : t('app.about.show')}
              >
                {showKey ? t('app.about.hide') : t('app.about.show')}
              </button>
            </div>
            <button
              onClick={handleSave}
              className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
            >
              {saved ? t('app.about.saved') : t('common.save')}
            </button>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-border">
          <label className="text-sm font-medium">{t('app.about.hfToken')}</label>
          <p className="text-xs text-muted-foreground mt-0.5 mb-2">
            {t('app.about.hfRequired')}{' '}
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              {t('app.about.getToken')}
            </a>
          </p>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type={showHfKey ? 'text' : 'password'}
                value={hfToken}
                onChange={e => { setHfToken(e.target.value); setHfSaved(false) }}
                placeholder={hasSavedHfToken ? t('app.about.savedTokenPlaceholder') : 'hf_...'}
                className="w-full px-3 py-2 bg-background border border-input rounded text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ring pr-10"
              />
              <button
                onClick={() => setShowHfKey(!showHfKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground text-xs"
                title={showHfKey ? t('app.about.hide') : t('app.about.show')}
              >
                {showHfKey ? t('app.about.hide') : t('app.about.show')}
              </button>
            </div>
            <button
              onClick={handleHfSave}
              className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
            >
              {hfSaved ? t('app.about.saved') : t('common.save')}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
