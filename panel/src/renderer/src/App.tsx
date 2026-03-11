import { useState, useEffect, useRef } from 'react'
import { ArrowLeft } from 'lucide-react'
import { ThemeToggle } from './components/ui/theme-toggle'
import { SessionDashboard } from './components/sessions/SessionDashboard'
import { CreateSession } from './components/sessions/CreateSession'
import { SessionView } from './components/sessions/SessionView'
import { SessionSettings } from './components/sessions/SessionSettings'
import { SetupScreen } from './components/setup/SetupScreen'
import { ToastProvider } from './components/Toast'
import { DownloadStatusBar } from './components/DownloadStatusBar'
import { UpdateBanner } from './components/UpdateBanner'

type View =
  | { type: 'setup' }
  | { type: 'dashboard' }
  | { type: 'create' }
  | { type: 'session'; sessionId: string }
  | { type: 'sessionSettings'; sessionId: string }
  | { type: 'about' }

function App() {
  const [view, setView] = useState<View>({ type: 'setup' })

  // Clear any stale chat locks on app mount (handles window reload recovery)
  useEffect(() => {
    window.api.chat.clearAllLocks().catch(() => { })
  }, [])

  return (
    <ToastProvider>
      <div className="flex flex-col h-screen bg-background text-foreground">
        {/* Window drag bar */}
        <div className="flex items-center h-8 bg-card border-b border-border flex-shrink-0" style={{ WebkitAppRegion: 'drag' } as React.CSSProperties}>
          {/* Back button + title in drag bar (no-drag region) — pl-[72px] clears macOS traffic lights */}
          <div className="flex items-center gap-2 pl-[72px] pr-3" style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties}>
            {view.type !== 'dashboard' && view.type !== 'setup' && (
              <button
                onClick={() => setView({ type: 'dashboard' })}
                className="text-xs text-muted-foreground hover:text-foreground inline-flex items-center gap-1"
              >
                <ArrowLeft className="h-3 w-3" />
                Home
              </button>
            )}
          </div>
          <span className="text-xs text-muted-foreground flex-1 text-center font-medium tracking-wide">vMLX</span>
          {/* Theme toggle + About button */}
          <div style={{ WebkitAppRegion: 'no-drag' } as React.CSSProperties} className="flex items-center gap-1 px-3">
            <ThemeToggle />
            <button
              onClick={() => setView({ type: 'about' })}
              className={`text-xs px-2 py-0.5 rounded ${view.type === 'about' ? 'bg-accent' : 'hover:bg-accent'}`}
            >
              About
            </button>
          </div>
        </div>

        {/* Update notification + download status */}
        <UpdateBanner />
        <DownloadStatusBar />

        {/* Main content */}
        <main className="flex-1 overflow-hidden">
          {view.type === 'setup' && (
            <SetupScreen onReady={() => setView({ type: 'dashboard' })} />
          )}

          {view.type === 'dashboard' && (
            <SessionDashboard
              onOpenSession={(sessionId) => setView({ type: 'session', sessionId })}
              onConfigureSession={(sessionId) => setView({ type: 'sessionSettings', sessionId })}
              onCreateSession={() => setView({ type: 'create' })}
            />
          )}

          {view.type === 'create' && (
            <CreateSession
              onBack={() => setView({ type: 'dashboard' })}
              onCreated={(sessionId) => setView({ type: 'session', sessionId })}
            />
          )}

          {view.type === 'session' && (
            <SessionView
              sessionId={view.sessionId}
              onBack={() => setView({ type: 'dashboard' })}
            />
          )}

          {view.type === 'sessionSettings' && (
            <SessionSettings
              sessionId={view.sessionId}
              onBack={() => setView({ type: 'dashboard' })}
            />
          )}

          {view.type === 'about' && (
            <div className="p-8 overflow-auto h-full">
              <div className="max-w-3xl mx-auto space-y-6">
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
        </main>
      </div>
    </ToastProvider>
  )
}

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

// ─── API Keys Section (About page) ─────────────────────────────────────────

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
