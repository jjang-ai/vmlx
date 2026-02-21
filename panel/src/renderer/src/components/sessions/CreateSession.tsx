import { useState, useEffect, useRef } from 'react'
import { SessionConfigForm, SessionConfig, DEFAULT_CONFIG } from './SessionConfigForm'
import { DownloadTab } from './DownloadTab'

interface ModelInfo {
  path: string
  name: string
  size?: string
  quantization?: string
}

interface CreateSessionProps {
  onBack: () => void
  onCreated: (sessionId: string) => void
}

export function CreateSession({ onBack, onCreated }: CreateSessionProps) {
  const [sessionType, setSessionType] = useState<'local' | 'remote' | 'download'>('local')
  const [step, setStep] = useState<1 | 2>(1)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [modelFilter, setModelFilter] = useState('')
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_CONFIG)
  const [detectedCacheType, setDetectedCacheType] = useState<string | undefined>()
  const [launching, setLaunching] = useState(false)
  const [launchError, setLaunchError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [scanLoading, setScanLoading] = useState(true)
  const [showDirManager, setShowDirManager] = useState(false)
  const [userDirs, setUserDirs] = useState<string[]>([])
  const [builtinDirs, setBuiltinDirs] = useState<string[]>([])
  const [dirError, setDirError] = useState<string | null>(null)
  const [manualPath, setManualPath] = useState('')
  const logEndRef = useRef<HTMLDivElement>(null)
  const launchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Remote session fields
  const [remoteUrl, setRemoteUrl] = useState('')
  const [remoteApiKey, setRemoteApiKey] = useState('')
  const [remoteModel, setRemoteModel] = useState('')
  const [remoteOrganization, setRemoteOrganization] = useState('')
  const [remoteConnecting, setRemoteConnecting] = useState(false)

  // Cleanup launch timer on unmount
  useEffect(() => {
    return () => {
      if (launchTimerRef.current) clearTimeout(launchTimerRef.current)
    }
  }, [])

  const scanModels = async () => {
    setScanLoading(true)
    try {
      const scanned = await window.api.models.scan()
      setModels(scanned)
    } catch (err) {
      console.error('Failed to scan models:', err)
    } finally {
      setScanLoading(false)
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

  useEffect(() => {
    scanModels()
    loadDirectories()
  }, [])

  // Auto-assign next available port
  useEffect(() => {
    const assignPort = async () => {
      try {
        const sessions = await window.api.sessions.list()
        const usedPorts = new Set(sessions.map((s: any) => s.port))
        let port = 8000
        while (usedPorts.has(port)) port++
        setConfig(prev => ({ ...prev, port }))
      } catch (_) { }
    }
    assignPort()
  }, [])

  const handleChange = <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const handleReset = async () => {
    const base = { ...DEFAULT_CONFIG, port: config.port }
    // Re-run model detection to get proper defaults for this model
    if (selectedModel) {
      try {
        const detected = await window.api.models.detectConfig(selectedModel)
        if (detected && detected.family !== 'unknown') {
          base.enableAutoToolChoice = detected.enableAutoToolChoice
          base.usePagedCache = detected.usePagedCache
        }
      } catch (_) { }
    }
    setConfig(base)
  }

  // Clean up log listener when launching state changes or component unmounts
  useEffect(() => {
    if (!launching) return

    const unsubLog = window.api.sessions.onLog((data: any) => {
      setLogs(prev => [...prev.slice(-200), data.data])
    })

    // Also listen for errors during launch
    const unsubError = window.api.sessions.onError((data: any) => {
      setLogs(prev => [...prev, `ERROR: ${data.error}`])
      setLaunchError(data.error)
    })

    return () => {
      unsubLog()
      unsubError()
    }
  }, [launching])

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleLaunch = async () => {
    if (!selectedModel) return

    setLaunchError(null)

    // Pre-validate before entering launch state
    try {
      // Check vllm-mlx installation
      const installation = await window.api.vllm.checkInstallation()
      if (!installation?.installed) {
        setLaunchError(
          'Inference engine not found. Restart vMLX to run first-time setup, ' +
          'or install manually:\n\n' +
          '  uv tool install vllm-mlx\n' +
          '  pip3 install vllm-mlx'
        )
        return
      }
    } catch (_) {
      // If the check itself fails, proceed anyway — startSession will catch it
    }

    setLaunching(true)
    setLogs(['Creating session...'])

    try {
      const session = await window.api.sessions.create(selectedModel, config)
      setLogs(prev => [...prev, `Session created: ${session.id}`, 'Starting server...'])

      const result = await window.api.sessions.start(session.id)
      if (result.success) {
        setLogs(prev => [...prev, 'Server is ready!'])
        launchTimerRef.current = setTimeout(() => onCreated(session.id), 500)
      } else {
        const errorMsg = result.error || 'Unknown error'
        setLogs(prev => [...prev, `\nERROR: ${errorMsg}`])
        setLaunchError(errorMsg)
        setLaunching(false)
      }
    } catch (error) {
      const errorMsg = (error as Error).message
      setLogs(prev => [...prev, `\nERROR: ${errorMsg}`])
      setLaunchError(errorMsg)
      setLaunching(false)
    }
  }

  const handleLaunchRemote = async () => {
    if (!remoteUrl.trim() || !remoteModel.trim()) return
    setLaunchError(null)
    setRemoteConnecting(true)

    try {
      const session = await window.api.sessions.createRemote({
        remoteUrl: remoteUrl.trim(),
        remoteApiKey: remoteApiKey.trim() || undefined,
        remoteModel: remoteModel.trim(),
        remoteOrganization: remoteOrganization.trim() || undefined
      })

      const result = await window.api.sessions.start(session.id)
      if (result.success) {
        onCreated(session.id)
      } else {
        setLaunchError(result.error || 'Failed to connect to remote endpoint')
        setRemoteConnecting(false)
      }
    } catch (error) {
      setLaunchError((error as Error).message)
      setRemoteConnecting(false)
    }
  }

  const handleBrowseDirectory = async () => {
    setDirError(null)
    const result = await window.api.models.browseDirectory()
    if (result.canceled || !result.path) return
    await addDirectory(result.path)
  }

  const handleAddManualPath = async () => {
    if (!manualPath.trim()) return
    setDirError(null)
    await addDirectory(manualPath.trim())
    setManualPath('')
  }

  const addDirectory = async (dirPath: string) => {
    const result = await window.api.models.addDirectory(dirPath)
    if (result.success) {
      await loadDirectories()
      // Rescan with the new directory
      await scanModels()
    } else {
      setDirError(result.error || 'Failed to add directory')
    }
  }

  const handleRemoveDirectory = async (dirPath: string) => {
    await window.api.models.removeDirectory(dirPath)
    await loadDirectories()
    await scanModels()
  }

  const filteredModels = models.filter(m =>
    m.name.toLowerCase().includes(modelFilter.toLowerCase()) ||
    m.path.toLowerCase().includes(modelFilter.toLowerCase())
  )

  // Step 1: Model Selection
  if (step === 1) {
    return (
      <div className="p-6 overflow-auto h-full">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-center gap-3 mb-6">
            <button onClick={onBack} className="text-muted-foreground hover:text-foreground">
              ← Back
            </button>
            <h1 className="text-2xl font-bold">Create Session</h1>
          </div>

          {/* Session Type Selector */}
          <div className="flex gap-1 bg-background rounded border border-border p-0.5 mb-4">
            <button
              onClick={() => setSessionType('local')}
              className={`flex-1 px-3 py-1.5 text-sm rounded transition-colors ${sessionType === 'local'
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-accent text-muted-foreground'
                }`}
            >
              Local Model
            </button>
            <button
              onClick={() => setSessionType('download')}
              className={`flex-1 px-3 py-1.5 text-sm rounded transition-colors ${sessionType === 'download'
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-accent text-muted-foreground'
                }`}
            >
              Download
            </button>
            <button
              onClick={() => setSessionType('remote')}
              className={`flex-1 px-3 py-1.5 text-sm rounded transition-colors ${sessionType === 'remote'
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-accent text-muted-foreground'
                }`}
            >
              Remote Endpoint
            </button>
          </div>

          {/* Download Tab */}
          {sessionType === 'download' ? (
            <DownloadTab onDownloadComplete={() => { setSessionType('local'); scanModels() }} />
          ) : sessionType === 'remote' ? (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                Connect to any OpenAI-compatible API endpoint.
              </p>

              <div>
                <label className="text-sm font-medium block mb-1">API Base URL *</label>
                <input
                  type="url"
                  placeholder="https://api.openai.com"
                  value={remoteUrl}
                  onChange={(e) => setRemoteUrl(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Base URL of the OpenAI-compatible API (e.g., https://api.openai.com, http://localhost:8000)
                </p>
              </div>

              <div>
                <label className="text-sm font-medium block mb-1">Model Name *</label>
                <input
                  type="text"
                  placeholder="gpt-4o, llama-3.1-8b, etc."
                  value={remoteModel}
                  onChange={(e) => setRemoteModel(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                />
              </div>

              <div>
                <label className="text-sm font-medium block mb-1">API Key</label>
                <input
                  type="password"
                  placeholder="sk-..."
                  value={remoteApiKey}
                  onChange={(e) => setRemoteApiKey(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                />
              </div>

              <div>
                <label className="text-sm font-medium block mb-1">Organization</label>
                <input
                  type="text"
                  placeholder="Optional"
                  value={remoteOrganization}
                  onChange={(e) => setRemoteOrganization(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                />
              </div>

              {launchError && (
                <div className="p-3 bg-destructive/10 border border-destructive/30 rounded">
                  <p className="text-sm text-destructive">{launchError}</p>
                </div>
              )}

              <button
                onClick={handleLaunchRemote}
                disabled={remoteConnecting || !remoteUrl.trim() || !remoteModel.trim()}
                className="px-6 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {remoteConnecting ? 'Connecting...' : 'Connect'}
              </button>
            </div>
          ) : (
            <>
              <span className="text-sm text-muted-foreground block mb-4">Select a local MLX model to serve</span>

              {/* Search + Actions Row */}
              <div className="flex items-center gap-2 mb-4">
                <input
                  type="text"
                  placeholder="Filter models..."
                  value={modelFilter}
                  onChange={(e) => setModelFilter(e.target.value)}
                  className="flex-1 px-3 py-2 bg-background border border-input rounded text-sm"
                />
                <button
                  onClick={scanModels}
                  disabled={scanLoading}
                  className="px-3 py-2 text-sm border border-border rounded hover:bg-accent disabled:opacity-50 whitespace-nowrap"
                >
                  {scanLoading ? 'Scanning...' : 'Rescan'}
                </button>
                <button
                  onClick={() => setShowDirManager(!showDirManager)}
                  className={`px-3 py-2 text-sm border rounded whitespace-nowrap ${showDirManager ? 'border-primary bg-primary/10 text-primary' : 'border-border hover:bg-accent'
                    }`}
                >
                  Directories
                </button>
              </div>

              {/* Directory Manager Panel */}
              {showDirManager && (
                <div className="mb-4 p-4 bg-card border border-border rounded-lg">
                  <h3 className="text-sm font-semibold mb-3">Model Scan Directories</h3>

                  {/* Built-in directories */}
                  {builtinDirs.length > 0 && (
                    <div className="mb-3">
                      <span className="text-xs text-muted-foreground uppercase tracking-wider">Default</span>
                      {builtinDirs.map(dir => (
                        <div key={dir} className="flex items-center gap-2 mt-1 px-2 py-1.5 bg-muted/50 rounded text-xs text-muted-foreground">
                          <span className="truncate flex-1" title={dir}>{dir}</span>
                          <span className="text-xs opacity-50 flex-shrink-0">built-in</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* User directories */}
                  {userDirs.length > 0 && (
                    <div className="mb-3">
                      <span className="text-xs text-muted-foreground uppercase tracking-wider">Custom</span>
                      {userDirs.map(dir => (
                        <div key={dir} className="flex items-center gap-2 mt-1 px-2 py-1.5 bg-muted/50 rounded text-xs">
                          <span className="truncate flex-1" title={dir}>{dir}</span>
                          <button
                            onClick={() => handleRemoveDirectory(dir)}
                            className="text-destructive hover:text-destructive/80 flex-shrink-0"
                            title="Remove directory"
                          >
                            ✕
                          </button>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Add directory */}
                  <div className="flex items-center gap-2 mt-2">
                    <input
                      type="text"
                      placeholder="Enter path or browse..."
                      value={manualPath}
                      onChange={(e) => { setManualPath(e.target.value); setDirError(null) }}
                      onKeyDown={(e) => { if (e.key === 'Enter') handleAddManualPath() }}
                      className="flex-1 px-2 py-1.5 bg-background border border-input rounded text-xs"
                    />
                    <button
                      onClick={handleAddManualPath}
                      disabled={!manualPath.trim()}
                      className="px-2 py-1.5 text-xs border border-border rounded hover:bg-accent disabled:opacity-50"
                    >
                      Add
                    </button>
                    <button
                      onClick={handleBrowseDirectory}
                      className="px-2 py-1.5 text-xs border border-border rounded hover:bg-accent"
                    >
                      Browse...
                    </button>
                  </div>
                  {dirError && (
                    <p className="text-xs text-destructive mt-1">{dirError}</p>
                  )}
                </div>
              )}

              {scanLoading ? (
                <p className="text-muted-foreground">Scanning for models...</p>
              ) : filteredModels.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground mb-2">No models found</p>
                  <p className="text-xs text-muted-foreground mb-4">
                    Click "Directories" above to add your model folders,
                    or place models in ~/.lmstudio/models/ or ~/.cache/huggingface/hub/
                  </p>
                  <div className="mb-4 p-3 bg-card border border-border rounded-lg text-left">
                    <p className="text-xs text-muted-foreground mb-2">To download a model, run in Terminal:</p>
                    <div className="p-2 bg-muted rounded font-mono text-[11px] text-foreground select-all">
                      huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir ~/.cache/huggingface/hub/mlx-community/Llama-3.2-3B-Instruct-4bit
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-2">
                      Browse more MLX models at{' '}
                      <span className="text-primary select-all">huggingface.co/mlx-community</span>
                    </p>
                  </div>
                  <button
                    onClick={() => setShowDirManager(true)}
                    className="px-4 py-2 text-sm border border-border rounded hover:bg-accent"
                  >
                    Manage Directories
                  </button>
                </div>
              ) : (
                <div className="space-y-1">
                  {filteredModels.map(model => (
                    <button
                      key={model.path}
                      onClick={async () => {
                        setSelectedModel(model.path)
                        // Pre-populate from existing session config if this model was launched before
                        try {
                          const sessions = await window.api.sessions.list()
                          const normalized = model.path.replace(/\/+$/, '')
                          const existing = sessions.find((s: any) =>
                            (s.modelPath || '').replace(/\/+$/, '') === normalized
                          )
                          if (existing?.config) {
                            try {
                              const stored = JSON.parse(existing.config)
                              setConfig(prev => ({ ...prev, ...stored, port: prev.port }))
                              // Still detect cache type for UI gating (Mamba vs KV, VLM)
                              try {
                                const det = await window.api.models.detectConfig(model.path)
                                if (det?.cacheType) setDetectedCacheType(det.cacheType)
                              } catch (_) { }
                              setStep(2)
                              return // skip auto-detect for config — existing config already has everything
                            } catch (_) { }
                          }
                        } catch (_) { }
                        // Fallback: auto-detect model config for fresh sessions
                        try {
                          const detected = await window.api.models.detectConfig(model.path)
                          if (detected && detected.family !== 'unknown') {
                            setConfig(prev => ({
                              ...prev,
                              enableAutoToolChoice: detected.enableAutoToolChoice,
                              toolCallParser: 'auto',
                              reasoningParser: 'auto',
                              usePagedCache: detected.usePagedCache,
                            }))
                            setDetectedCacheType(detected.cacheType || 'kv')
                          }
                        } catch (_) {
                          // Auto-detect failed — user can configure manually
                        }
                        setStep(2)
                      }}
                      className={`w-full text-left p-3 rounded border transition-colors ${selectedModel === model.path
                        ? 'border-primary bg-primary/10'
                        : 'border-border hover:border-primary/50 hover:bg-accent'
                        }`}
                    >
                      <div className="font-medium text-sm">{model.name}</div>
                      <div className="text-xs text-muted-foreground truncate">{model.path}</div>
                      {model.size && (
                        <div className="text-xs text-muted-foreground mt-1">
                          {model.size}
                          {model.quantization && ` · ${model.quantization}`}
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    )
  }

  // Launching state
  if (launching) {
    return (
      <div className="p-6 overflow-auto h-full">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-2xl font-bold mb-2">Loading Model</h1>
          <p className="text-muted-foreground text-sm mb-4">
            {selectedModel.split('/').pop()}
          </p>

          <div className="bg-background/80 text-primary font-mono text-xs p-4 rounded-lg max-h-[60vh] overflow-auto border border-border">
            {logs.map((line, i) => (
              <div key={i} className={`whitespace-pre-wrap ${line.startsWith('ERROR') ? 'text-destructive font-bold' : ''}`}>{line}</div>
            ))}
            {!launchError && <div className="animate-pulse">▌</div>}
            <div ref={logEndRef} />
          </div>

          {launchError && (
            <div className="mt-4 p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
              <h3 className="text-sm font-bold text-destructive mb-2">Launch Failed</h3>
              <p className="text-sm text-destructive/90 whitespace-pre-wrap">{launchError}</p>
              <div className="flex gap-2 mt-3">
                <button
                  onClick={() => {
                    setLaunching(false)
                    setLaunchError(null)
                    setLogs([])
                  }}
                  className="px-4 py-1.5 text-sm bg-destructive text-destructive-foreground rounded hover:bg-destructive/90"
                >
                  Back to Config
                </button>
                <button
                  onClick={() => {
                    setLaunchError(null)
                    setLogs([])
                    handleLaunch()
                  }}
                  className="px-4 py-1.5 text-sm border border-border rounded hover:bg-accent"
                >
                  Retry
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  // Step 2: Configuration
  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center gap-3 mb-6">
          <button onClick={() => setStep(1)} className="text-muted-foreground hover:text-foreground">
            ← Back
          </button>
          <h1 className="text-2xl font-bold">Create Session</h1>
          <span className="text-sm text-muted-foreground">Step 2: Configure</span>
        </div>

        {/* Pre-launch error banner */}
        {launchError && (
          <div className="mb-4 p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
            <p className="text-sm text-destructive whitespace-pre-wrap">{launchError}</p>
          </div>
        )}

        {/* Selected model */}
        <div className="mb-4 p-3 bg-card border border-border rounded">
          <span className="text-xs text-muted-foreground">Model</span>
          <p className="font-medium text-sm truncate">{selectedModel}</p>
        </div>

        {/* Config Form */}
        <SessionConfigForm config={config} onChange={handleChange} onReset={handleReset} detectedCacheType={detectedCacheType} />

        {/* Launch */}
        <div className="flex gap-3 mt-6 pb-6">
          <button onClick={() => setStep(1)} className="px-4 py-2 border border-border rounded hover:bg-accent">
            Back
          </button>
          <button
            onClick={handleLaunch}
            disabled={launching || !selectedModel}
            className="px-6 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Launch Session
          </button>
        </div>
      </div>
    </div>
  )
}
