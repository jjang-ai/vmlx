import { useState, useEffect, useCallback, useRef } from 'react'
import { ImageModelPicker } from './ImageModelPicker'
import { ImagePromptBar } from './ImagePromptBar'
import { ImageGallery } from './ImageGallery'
import { ImageHistory } from './ImageHistory'
import { ImageTopBar } from './ImageTopBar'
import { ImageSettings } from './ImageSettings'

export interface ImageSessionInfo {
  id: string
  modelName: string
  createdAt: number
  updatedAt: number
}

export interface ImageGenerationInfo {
  id: string
  sessionId: string
  prompt: string
  negativePrompt?: string
  modelName: string
  width: number
  height: number
  steps: number
  guidance: number
  seed?: number
  elapsedSeconds?: number
  imagePath: string
  createdAt: number
}

type ServerStatus = 'stopped' | 'starting' | 'running' | 'error'

export function ImageTab() {
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ImageSessionInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [serverStatus, setServerStatus] = useState<ServerStatus>('stopped')
  const [serverPort, setServerPort] = useState<number | null>(null)
  const [serverSessionId, _setServerSessionId] = useState<string | null>(null)
  const serverSessionIdRef = useRef<string | null>(null)
  const setServerSessionId = (id: string | null) => { serverSessionIdRef.current = id; _setServerSessionId(id) }
  const [showSettings, setShowSettings] = useState(false)
  const [showModelPicker, setShowModelPicker] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [generations, setGenerations] = useState<ImageGenerationInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [quantize, setQuantize] = useState<number>(4)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Image generation settings (quick settings + full settings)
  const [settings, setSettings] = useState({
    steps: 4,
    width: 1024,
    height: 1024,
    guidance: 3.5,
    negativePrompt: '',
    seed: undefined as number | undefined,
    count: 1,
    quantize: 4
  })

  // Load image sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  // Check if an image server is already running
  useEffect(() => {
    window.api.image.getRunningServer().then((server: any) => {
      if (server) {
        setSelectedModel(server.modelName)
        setServerStatus('running')
        setServerPort(server.port)
        setServerSessionId(server.sessionId)
        setShowModelPicker(false)
      }
    }).catch(() => {})
  }, [])

  // Listen for session events to detect when server becomes ready
  useEffect(() => {
    const unsubReady = window.api.sessions.onReady((data: any) => {
      if (data.sessionId === serverSessionIdRef.current) {
        setServerStatus('running')
        // Fetch the session to get port
        window.api.sessions.get(data.sessionId).then((s: any) => {
          if (s) setServerPort(s.port)
        }).catch(() => {})
      }
    })
    const unsubStopped = window.api.sessions.onStopped((data: any) => {
      if (data.sessionId === serverSessionIdRef.current) {
        setServerStatus('stopped')
        setServerPort(null)
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === serverSessionIdRef.current) {
        setServerStatus('error')
        setError(data.error || 'Server error')
      }
    })
    return () => {
      unsubReady()
      unsubStopped()
      unsubError()
    }
  }, []) // Uses ref, not state — no dependency needed

  // Poll for server health when starting
  useEffect(() => {
    if (serverStatus === 'starting' && serverPort) {
      pollRef.current = setInterval(async () => {
        try {
          const resp = await fetch(`http://127.0.0.1:${serverPort}/health`)
          if (resp.ok) {
            setServerStatus('running')
            if (pollRef.current) clearInterval(pollRef.current)
          }
        } catch (_) {
          // Still starting
        }
      }, 1000)
      return () => {
        if (pollRef.current) clearInterval(pollRef.current)
      }
    }
  }, [serverStatus, serverPort])

  // Load generations when session changes
  useEffect(() => {
    if (currentSessionId) {
      loadGenerations(currentSessionId)
    } else {
      setGenerations([])
    }
  }, [currentSessionId])

  const loadSessions = useCallback(async () => {
    const result = await window.api.image.getSessions()
    setSessions(result || [])
  }, [])

  const loadGenerations = useCallback(async (sessionId: string) => {
    const result = await window.api.image.getGenerations(sessionId)
    setGenerations(result || [])
  }, [])

  const handleModelSelect = useCallback(async (modelId: string, modelQuantize?: number) => {
    setSelectedModel(modelId)
    setShowModelPicker(false)
    setError(null)
    const q = modelQuantize ?? 4
    setQuantize(q)

    // Update default steps based on model
    const defaultSteps = getDefaultSteps(modelId)
    setSettings(prev => ({ ...prev, steps: defaultSteps, quantize: q }))

    // Auto-start server
    setServerStatus('starting')
    try {
      const result = await window.api.image.startServer(modelId, q)
      if (result.success) {
        setServerSessionId(result.sessionId)
        setServerPort(result.port)
        // Status will transition to 'running' via polling or session events
      } else {
        setServerStatus('error')
        setError(result.error || 'Failed to start server')
      }
    } catch (err) {
      setServerStatus('error')
      setError((err as Error).message)
    }
  }, [])

  const handleGenerate = useCallback(async (prompt: string) => {
    if (!serverPort || serverStatus !== 'running' || !selectedModel) return

    setGenerating(true)
    setError(null)

    try {
      // Create image session if we don't have one
      let sessionId = currentSessionId
      if (!sessionId) {
        const result = await window.api.image.createSession(selectedModel)
        if (result.success && result.session) {
          sessionId = result.session.id
          setCurrentSessionId(sessionId)
          await loadSessions()
        } else {
          throw new Error('Failed to create image session')
        }
      }

      const result = await window.api.image.generate({
        sessionId: sessionId!,
        prompt,
        negativePrompt: settings.negativePrompt || undefined,
        model: selectedModel,
        width: settings.width,
        height: settings.height,
        steps: settings.steps,
        guidance: settings.guidance,
        seed: settings.seed,
        count: settings.count,
        quantize: settings.quantize,
        serverPort
      })

      if (result.success && result.generations) {
        setGenerations(prev => [...prev, ...result.generations])
        await loadSessions() // Refresh session list (updatedAt changed)
      } else {
        setError(result.error || 'Generation failed')
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setGenerating(false)
    }
  }, [serverPort, serverStatus, selectedModel, currentSessionId, settings, loadSessions])

  const handleStop = useCallback(async () => {
    try {
      await window.api.image.stopServer()
      setServerStatus('stopped')
      setServerPort(null)
      setServerSessionId(null)
    } catch (err) {
      console.error('Failed to stop image server:', err)
    }
  }, [])

  const handleChangeModel = useCallback(async () => {
    // Stop the running server before switching models
    if (serverStatus === 'running' || serverStatus === 'starting') {
      await handleStop()
    }
    setSelectedModel(null)
    setShowModelPicker(true)
  }, [serverStatus, handleStop])

  const handleNewSession = useCallback(() => {
    setCurrentSessionId(null)
    setGenerations([])
  }, [])

  const handleSelectSession = useCallback(async (sessionId: string) => {
    setCurrentSessionId(sessionId)
  }, [])

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    await window.api.image.deleteSession(sessionId)
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null)
      setGenerations([])
    }
    await loadSessions()
  }, [currentSessionId, loadSessions])

  const handleSettingsChange = useCallback((newSettings: typeof settings) => {
    setSettings(newSettings)
  }, [])

  // Show model picker if no model selected
  if (showModelPicker && !selectedModel) {
    return (
      <div className="h-full flex flex-col">
        <ImageModelPicker onSelect={handleModelSelect} />
      </div>
    )
  }

  return (
    <div className="h-full flex">
      {/* History Sidebar */}
      {!sidebarCollapsed && (
        <ImageHistory
          sessions={sessions}
          currentId={currentSessionId}
          onSelect={handleSelectSession}
          onNew={handleNewSession}
          onDelete={handleDeleteSession}
          onCollapse={() => setSidebarCollapsed(true)}
        />
      )}

      {/* Main Area */}
      <div className="flex-1 flex flex-col min-w-0">
        <ImageTopBar
          model={selectedModel}
          quantize={quantize}
          status={serverStatus}
          port={serverPort}
          onSettings={() => setShowSettings(!showSettings)}
          onStop={handleStop}
          onChangeModel={handleChangeModel}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        />

        {showSettings && (
          <ImageSettings
            settings={settings}
            onChange={handleSettingsChange}
            model={selectedModel}
          />
        )}

        {error && (
          <div className="mx-4 mt-2 px-3 py-2 bg-destructive/10 border border-destructive/20 rounded-md text-sm text-destructive">
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-xs underline">Dismiss</button>
          </div>
        )}

        <div className="flex-1 overflow-hidden">
          <ImageGallery
            generations={generations}
            generating={generating}
          />
        </div>

        <ImagePromptBar
          onGenerate={handleGenerate}
          disabled={serverStatus !== 'running'}
          generating={generating}
          settings={settings}
          onSettingsChange={handleSettingsChange}
          model={selectedModel}
        />
      </div>
    </div>
  )
}

function getDefaultSteps(modelId: string): number {
  const defaults: Record<string, number> = {
    'schnell': 4,
    'dev': 20,
    'z-image-turbo': 4,
    'flux2-klein-4b': 20,
    'flux2-klein-9b': 20
  }
  return defaults[modelId] || 4
}
