import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { MessageSquare, ArrowLeft, Terminal } from 'lucide-react'
import { TitleBar } from './components/layout/TitleBar'
import { Sidebar } from './components/layout/Sidebar'
import { SessionDashboard } from './components/sessions/SessionDashboard'
import { CreateSession } from './components/sessions/CreateSession'
import { SessionView } from './components/sessions/SessionView'
import { SessionSettings } from './components/sessions/SessionSettings'
import { ChatInterface } from './components/chat/ChatInterface'
import { SetupScreen } from './components/setup/SetupScreen'
import { ToastProvider } from './components/Toast'
import { DownloadStatusBar } from './components/DownloadStatusBar'
import { UpdateBanner } from './components/UpdateBanner'
import { UpdateNotice } from './components/UpdateNotice'
import { useAppState } from './contexts/AppStateContext'
import { useSessionsContext } from './contexts/SessionsContext'
import { ChatModeToolbar } from './components/layout/ChatModeToolbar'
import { ToolsDashboard } from './components/tools/ToolsDashboard'
import { ModelInspector } from './components/tools/ModelInspector'
import { ModelDoctor } from './components/tools/ModelDoctor'
import { ModelConverter } from './components/tools/ModelConverter'
import { ApiDashboard } from './components/api/ApiDashboard'
import { ImageTab } from './components/image/ImageTab'
import { isImageSession } from '../../shared/sessionUtils'

function App() {
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
          dispatch({ type: 'SET_SERVER_PANEL', panel: detail.panel, modelPath: detail.modelPath })
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

  // Resolve the endpoint for the active session
  const activeSession = sessions.find(s => s.id === state.activeSessionId)
  const sessionEndpoint = activeSession?.status === 'running'
    ? { host: activeSession.host, port: activeSession.port }
    : undefined

  const handleChatSelect = useCallback((chatId: string, modelPath: string) => {
    // Empty chatId means deselect (e.g. after deleting the active chat)
    if (!chatId) {
      dispatch({ type: 'CLOSE_CHAT' })
      return
    }

    // Find the session for this model — prefer running, then any matching, then first available
    const exactRunning = sessions.find(s => s.modelPath === modelPath && s.status === 'running')
    const exactAny = sessions.find(s => s.modelPath === modelPath)
    const fallback = sessions.find(s => s.status === 'running') || sessions[0]
    const session = exactRunning || exactAny || fallback

    if (session) {
      openChat(chatId, session.id)
    } else {
      // Truly no sessions at all — open chat without one, toolbar will handle
      dispatch({ type: 'OPEN_CHAT', chatId, sessionId: '' })
    }
  }, [sessions, openChat, dispatch])

  const handleNewChat = useCallback(async () => {
    // Find the first running session, or any session
    const running = sessions.find(s => s.status === 'running')
    const target = running || sessions[0]

    if (!target) {
      // No sessions — switch to server mode to create one
      setMode('server')
      dispatch({ type: 'SET_SERVER_PANEL', panel: 'create' })
      return
    }

    const modelName = target.modelName || target.modelPath.split('/').pop() || 'New Chat'
    const result = await window.api.chat.create(
      `Chat with ${modelName}`,
      target.modelPath,
      undefined,
      target.modelPath
    )
    if (result?.id) {
      openChat(result.id, target.id)
    }
  }, [sessions, setMode, dispatch, openChat])

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
        <DownloadStatusBar />
        <UpdateNotice />

        <div className="flex flex-1 overflow-hidden">
          {/* Sidebar — chat mode only */}
          {state.mode === 'chat' && (
            <Sidebar
              collapsed={state.sidebarCollapsed}
              currentChatId={state.activeChatId}
              onChatSelect={handleChatSelect}
              onNewChat={handleNewChat}
            />
          )}

          {/* Main content area */}
          <main className="flex-1 overflow-hidden">
            {state.mode === 'code' && (
              <div className="flex flex-col items-center justify-center h-full text-center px-8">
                <div className="w-16 h-16 rounded-2xl bg-emerald-500/10 flex items-center justify-center mb-4">
                  <Terminal className="h-8 w-8 text-emerald-500" />
                </div>
                <h2 className="text-lg font-semibold mb-2">Code</h2>
                <p className="text-sm text-muted-foreground max-w-sm">
                  IDE-like coding environment with AI agent, file browser, and integrated terminal.
                </p>
                <span className="mt-4 px-3 py-1 text-xs font-medium bg-emerald-500/10 text-emerald-500 rounded-full">
                  Coming Soon
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
            {state.mode === 'image' && (
              <ImageTab />
            )}
            {state.mode === 'api' && (
              <ApiDashboard />
            )}
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

// MLX Studio — eric@mlx.studio
function ChatEmptyState({ onNewChat }: { onNewChat: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8" data-mlx-studio="jinhojang">
      <div className="w-16 h-16 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
        <MessageSquare className="h-8 w-8 text-primary" />
      </div>
      <h2 className="text-lg font-semibold mb-2">Start a conversation</h2>
      <p className="text-sm text-muted-foreground mb-6 max-w-sm">
        Select a chat from the sidebar or create a new one to begin.
      </p>
      <button
        onClick={onNewChat}
        className="px-4 py-2 bg-primary text-primary-foreground text-sm rounded-md hover:bg-primary/90 transition-colors"
      >
        New Chat
      </button>
    </div>
  )
}

// ─── Server Mode Content ────────────────────────────────────────────────────

function ServerModeContent() {
  const { state, dispatch } = useAppState()
  const { serverPanel, serverSessionId, serverInitialModelPath } = state

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
              Back
            </button>
            <h2 className="text-2xl font-bold">About MLX Studio</h2>
            <p className="text-sm text-muted-foreground">
              A native macOS application for running local AI models on Apple Silicon.
            </p>
            <p className="text-xs text-muted-foreground/70">
              Created by Jinho Jang &middot; eric@mlx.studio
            </p>
            <AppVersion />
            <div className="flex gap-4 text-xs">
              <a href="https://mlx.studio" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">mlx.studio</a>
              <a href="https://github.com/jjang-ai/vmlx" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">GitHub</a>
              <a href="https://jangq.ai" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">JANG</a>
              <a href="https://ko-fi.com/jinhojang" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Ko-fi</a>
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
  const [version, setVersion] = useState('...')
  useEffect(() => {
    window.api.app.getVersion().then((v: string) => setVersion(v)).catch(() => setVersion('unknown'))
  }, [])
  return (
    <div className="text-xs text-muted-foreground space-y-1">
      <p>Version {version}</p>
      <p>&copy; {new Date().getFullYear()} Eric Jang. All rights reserved.</p>
    </div>
  )
}

function ApiKeysSection() {
  const [braveKey, setBraveKey] = useState('')
  const [hfToken, setHfToken] = useState('')
  const [saved, setSaved] = useState(false)
  const [hfSaved, setHfSaved] = useState(false)
  const [showKey, setShowKey] = useState(false)
  const [showHfKey, setShowHfKey] = useState(false)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    window.api.settings.get('braveApiKey').then((val) => {
      if (mountedRef.current && val) setBraveKey(val)
    })
    window.api.settings.get('hf_api_key').then((val) => {
      if (mountedRef.current && val) setHfToken(val)
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
    setHfSaved(true)
    setTimeout(() => { if (mountedRef.current) setHfSaved(false) }, 2000)
  }

  return (
    <div className="border border-border rounded-lg p-5">
      <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-4">API Keys</h3>
      <div className="space-y-3">
        <div>
          <label className="text-sm font-medium">Brave Search API Key</label>
          <p className="text-xs text-muted-foreground mt-0.5 mb-2">
            Required for the <code className="text-xs bg-muted px-1 py-0.5 rounded">web_search</code> tool when built-in tools are enabled.{' '}
            <a
              href="https://brave.com/search/api/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              Get a free key
            </a>
          </p>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={braveKey}
                onChange={e => { setBraveKey(e.target.value); setSaved(false) }}
                placeholder="BSA..."
                className="w-full px-3 py-2 bg-background border border-input rounded text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ring pr-10"
              />
              <button
                onClick={() => setShowKey(!showKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground text-xs"
                title={showKey ? 'Hide' : 'Show'}
              >
                {showKey ? 'Hide' : 'Show'}
              </button>
            </div>
            <button
              onClick={handleSave}
              className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
            >
              {saved ? 'Saved' : 'Save'}
            </button>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-border">
          <label className="text-sm font-medium">HuggingFace Token</label>
          <p className="text-xs text-muted-foreground mt-0.5 mb-2">
            Required for downloading gated models (Flux, Llama, etc.).{' '}
            <a
              href="https://huggingface.co/settings/tokens"
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline"
            >
              Get your token
            </a>
          </p>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <input
                type={showHfKey ? 'text' : 'password'}
                value={hfToken}
                onChange={e => { setHfToken(e.target.value); setHfSaved(false) }}
                placeholder="hf_..."
                className="w-full px-3 py-2 bg-background border border-input rounded text-sm font-mono focus:outline-none focus:ring-1 focus:ring-ring pr-10"
              />
              <button
                onClick={() => setShowHfKey(!showHfKey)}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground text-xs"
                title={showHfKey ? 'Hide' : 'Show'}
              >
                {showHfKey ? 'Hide' : 'Show'}
              </button>
            </div>
            <button
              onClick={handleHfSave}
              className="px-4 py-2 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90"
            >
              {hfSaved ? 'Saved' : 'Save'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
