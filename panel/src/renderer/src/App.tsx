import { useState, useEffect, useRef, useCallback } from 'react'
import { MessageSquare, ArrowLeft } from 'lucide-react'
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
import { useAppState } from './contexts/AppStateContext'
import { useSessionsContext } from './contexts/SessionsContext'
import { ChatModeToolbar } from './components/layout/ChatModeToolbar'

function App() {
  const [setupDone, setSetupDone] = useState(false)
  const [checkingSetup, setCheckingSetup] = useState(true)
  const { state, dispatch, setMode, openChat } = useAppState()
  const { sessions } = useSessionsContext()

  // Check if engine is already installed (skip setup screen if so)
  useEffect(() => {
    window.api.vllm.checkInstallation()
      .then((result: any) => {
        if (result.installed) setSetupDone(true)
      })
      .catch(() => {})
      .finally(() => setCheckingSetup(false))
  }, [])

  // Clear stale chat locks on mount
  useEffect(() => {
    window.api.chat.clearAllLocks().catch(() => {})
  }, [])

  // Listen for navigation events from child components (e.g. toolbar "Add a model")
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail
      if (detail?.mode) setMode(detail.mode)
      if (detail?.panel) dispatch({ type: 'SET_SERVER_PANEL', panel: detail.panel })
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
      } as any).catch(() => {})
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
            {state.mode === 'chat' && (
              <ChatModeContent
                activeChatId={state.activeChatId}
                sessionEndpoint={sessionEndpoint}
                activeSessionId={state.activeSessionId}
                onNewChat={handleNewChat}
                onSessionChange={handleSessionChange}
              />
            )}

            {state.mode === 'server' && (
              <ServerModeContent />
            )}
          </main>
        </div>
      </div>
    </ToastProvider>
  )
}

// ─── Chat Mode Content ──────────────────────────────────────────────────────

function ChatModeContent({ activeChatId, sessionEndpoint, activeSessionId, onNewChat, onSessionChange }: {
  activeChatId: string | null
  sessionEndpoint?: { host: string; port: number }
  activeSessionId: string | null
  onNewChat: () => void
  onSessionChange: (sessionId: string) => void
}) {
  if (!activeChatId) {
    return <ChatEmptyState onNewChat={onNewChat} />
  }

  return (
    <div className="flex flex-col h-full relative">
      <ChatModeToolbar
        activeChatId={activeChatId}
        activeSessionId={activeSessionId}
        onSessionChange={onSessionChange}
      />
      <div className="flex-1 overflow-hidden">
        <ChatInterface
          chatId={activeChatId}
          onNewChat={onNewChat}
          sessionEndpoint={sessionEndpoint}
          sessionId={activeSessionId || undefined}
        />
      </div>
    </div>
  )
}

function ChatEmptyState({ onNewChat }: { onNewChat: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-8">
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
  const { serverPanel, serverSessionId } = state

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
            <h2 className="text-2xl font-bold">About vMLX</h2>
            <p className="text-sm text-muted-foreground">
              A native macOS application for running local AI models on Apple Silicon.
            </p>
            <AppVersion />
            <div className="flex gap-4 text-xs">
              <a href="https://vmlx.net" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Website</a>
              <a href="https://github.com/vmlxllm/vmlx" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">GitHub</a>
            </div>
            <ApiKeysSection />
          </div>
        </div>
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
  const [saved, setSaved] = useState(false)
  const [showKey, setShowKey] = useState(false)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    window.api.settings.get('braveApiKey').then((val) => {
      if (mountedRef.current && val) setBraveKey(val)
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
      </div>
    </div>
  )
}

export default App
