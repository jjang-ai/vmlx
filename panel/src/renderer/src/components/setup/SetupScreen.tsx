import { useState, useEffect, useRef } from 'react'

interface AvailableInstaller {
  method: 'uv' | 'pip'
  path: string
  label: string
}

interface SetupScreenProps {
  onReady: () => void
}

export function SetupScreen({ onReady }: SetupScreenProps) {
  const [checking, setChecking] = useState(true)
  const [installers, setInstallers] = useState<AvailableInstaller[]>([])
  const [selectedMethod, setSelectedMethod] = useState<'uv' | 'pip' | null>(null)
  const [installing, setInstalling] = useState(false)
  const [installError, setInstallError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const logEndRef = useRef<HTMLDivElement>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    checkAndProceed()
    return () => { mountedRef.current = false }
  }, [])

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const checkAndProceed = async () => {
    setChecking(true)
    try {
      const result = await window.api.vllm.checkInstallation()
      if (result.installed) {
        onReady()
        return
      }

      // Not installed — detect available installers
      const available = await window.api.vllm.detectInstallers()
      setInstallers(available)
      if (available.length > 0) {
        setSelectedMethod(available[0].method)
      }
    } catch (err) {
      console.error('Setup check failed:', err)
    } finally {
      setChecking(false)
    }
  }

  const handleInstall = async () => {
    if (!selectedMethod || installing) return

    const installer = installers.find(i => i.method === selectedMethod)
    if (!installer) return

    setInstalling(true)
    setInstallError(null)
    setLogs([])

    const unsubLog = window.api.vllm.onInstallLog((data: any) => {
      if (mountedRef.current) setLogs(prev => [...prev.slice(-500), data.data])
    })

    try {
      const result = await window.api.vllm.installStreaming(
        selectedMethod,
        'install',
        installer.path
      )

      if (!mountedRef.current) return

      if (result.success) {
        setLogs(prev => [...prev, '\nInference engine installed successfully!'])
        // Brief pause so user sees success, then proceed
        setTimeout(() => { if (mountedRef.current) onReady() }, 1000)
      } else {
        setInstallError(result.error || 'Installation failed')
        setInstalling(false)
      }
    } catch (error) {
      if (mountedRef.current) {
        setInstallError((error as Error).message)
        setInstalling(false)
      }
    } finally {
      unsubLog()
    }
  }

  const handleCancel = async () => {
    await window.api.vllm.cancelInstall()
    setInstalling(false)
    setLogs(prev => [...prev, '\nInstallation cancelled.'])
  }

  if (checking) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Initializing inference engine...</p>
      </div>
    )
  }

  return (
    <div className="flex items-center justify-center h-full p-6">
      <div className="max-w-lg w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold mb-2">Welcome to vMLX</h1>
          <p className="text-muted-foreground">
            First-time setup: vMLX needs to install its inference engine.
            {installers.length > 0
              ? ' Click below to install it automatically.'
              : ' Please install it manually to continue.'}
          </p>
        </div>

        {/* Install log (shown during/after install) */}
        {logs.length > 0 && (
          <div className="mb-4 bg-background/80 text-primary font-mono text-xs p-4 rounded-lg max-h-[40vh] overflow-auto border border-border">
            {logs.map((line, i) => (
              <div key={i} className={`whitespace-pre-wrap ${line.includes('ERROR') || line.includes('error') ? 'text-destructive' : ''}`}>
                {line}
              </div>
            ))}
            {installing && <div className="animate-pulse">...</div>}
            <div ref={logEndRef} />
          </div>
        )}

        {/* Error banner */}
        {installError && (
          <div className="mb-4 p-3 bg-destructive/10 border border-destructive/30 rounded-lg">
            <p className="text-sm text-destructive">{installError}</p>
          </div>
        )}

        {installers.length > 0 ? (
          <div className="space-y-4">
            {/* Method picker */}
            {!installing && installers.length > 1 && (
              <div className="flex gap-2">
                {installers.map(inst => (
                  <button
                    key={inst.method}
                    onClick={() => setSelectedMethod(inst.method)}
                    className={`flex-1 p-3 rounded border text-sm font-medium transition-colors ${
                      selectedMethod === inst.method
                        ? 'border-primary bg-primary/10 text-primary'
                        : 'border-border hover:border-primary/50'
                    }`}
                  >
                    {inst.label}
                  </button>
                ))}
              </div>
            )}

            {/* Install / Cancel buttons */}
            <div className="flex gap-3 justify-center">
              {installing ? (
                <button
                  onClick={handleCancel}
                  className="px-6 py-2.5 text-sm border border-border rounded hover:bg-accent"
                >
                  Cancel
                </button>
              ) : (
                <>
                  <button
                    onClick={handleInstall}
                    disabled={!selectedMethod}
                    className="px-8 py-2.5 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium disabled:opacity-50"
                  >
                    Install Engine
                  </button>
                  {installError && (
                    <button
                      onClick={checkAndProceed}
                      className="px-4 py-2.5 text-sm border border-border rounded hover:bg-accent"
                    >
                      Check Again
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        ) : (
          /* No installers found — manual instructions */
          <div className="space-y-4">
            <div className="p-4 bg-card border border-border rounded-lg">
              <p className="text-sm text-muted-foreground mb-3">
                No compatible package manager found (uv or pip with Python 3.10+).
                Install manually:
              </p>
              <div className="space-y-2">
                <div className="p-2 bg-muted rounded font-mono text-xs">
                  <span className="text-muted-foreground"># Install uv first (recommended):</span>
                  <br />
                  curl -LsSf https://astral.sh/uv/install.sh | sh
                </div>
                <p className="text-xs text-muted-foreground">
                  Then restart vMLX — it will complete setup automatically.
                </p>
              </div>
            </div>
            <div className="flex justify-center">
              <button
                onClick={checkAndProceed}
                className="px-6 py-2.5 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium"
              >
                Check Again
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
