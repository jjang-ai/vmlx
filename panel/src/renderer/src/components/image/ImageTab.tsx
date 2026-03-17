import { useState, useEffect, useCallback, useRef } from 'react'
import { ImageModelPicker } from './ImageModelPicker'
import { ImagePromptBar } from './ImagePromptBar'
import { ImageGallery } from './ImageGallery'
import { ImageHistory } from './ImageHistory'
import { ImageTopBar } from './ImageTopBar'
import { ImageSettings } from './ImageSettings'
import { LogsPanel } from '../sessions/LogsPanel'
import { getDefaultSteps } from '../../../../shared/imageModels'

export interface ImageSessionInfo {
  id: string
  modelName: string
  sessionType?: 'generate' | 'edit'
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
  strength?: number
  elapsedSeconds?: number
  imagePath: string
  sourceImagePath?: string
  createdAt: number
}

type ServerStatus = 'stopped' | 'starting' | 'running' | 'error'

interface ImageSettings {
  steps: number
  width: number
  height: number
  guidance: number
  negativePrompt: string
  seed?: number
  count: number
  quantize: number
  strength: number
}

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
  const [showLogs, setShowLogs] = useState(false)
  const [showModelPicker, setShowModelPicker] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [generations, setGenerations] = useState<ImageGenerationInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  const [quantize, setQuantize] = useState<number>(4)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Edit mode state
  const [sessionMode, setSessionMode] = useState<'generate' | 'edit'>('generate')
  const [sourceImage, setSourceImage] = useState<{ dataUrl: string; name: string } | null>(null)

  // Image generation settings (quick settings + full settings)
  const [settings, setSettings] = useState<ImageSettings>({
    steps: 4,
    width: 1024,
    height: 1024,
    guidance: 3.5,
    negativePrompt: '',
    seed: undefined,
    count: 1,
    quantize: 4,
    strength: 0.8
  })

  // Load image sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  // Check if an image server is already running
  useEffect(() => {
    window.api.image.getRunningServer().then((server: any) => {
      if (server) {
        const name = server.modelName
        setSelectedModel(name)
        setServerStatus(server.status === 'loading' ? 'starting' : 'running')
        setServerPort(server.port)
        setServerSessionId(server.sessionId)
        setShowModelPicker(false)

        // Read imageMode from session config — no guessing from name
        const mode = server.imageMode || 'generate'
        setSessionMode(mode)

        // Restore quantize from server config
        const q = server.quantize ?? 0
        setQuantize(q)

        // Restore proper default steps for this model
        const defaultSteps = getDefaultSteps(name)
        setSettings(prev => ({ ...prev, steps: defaultSteps, quantize: q }))
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
        const errMsg = data.error || 'Server error'
        // Detect gated/auth errors and show helpful message
        const isGated = /40[13]|gated|access.*denied|authentication|authorized|forbidden/i.test(errMsg)
        if (isGated) {
          setError(
            'Model download failed — authentication required. ' +
            'Go to the Server tab > Download section and add your HuggingFace token, ' +
            'then accept the model license on huggingface.co. ' +
            'Original error: ' + errMsg.slice(0, 200)
          )
        } else {
          setError(errMsg)
        }
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
    if (serverStatus !== 'starting' || !serverPort) return
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

  const handleModelSelect = useCallback(async (modelId: string, modelQuantize?: number, category?: 'generate' | 'edit') => {
    const mode = category || 'generate'

    // Stop any currently running/starting server before switching models
    if (serverStatus === 'running' || serverStatus === 'starting') {
      try {
        await window.api.image.stopServer()
        setServerPort(null)
        setServerSessionId(null)
      } catch (err) {
        console.error('Failed to stop previous image server:', err)
      }
    }

    setSelectedModel(modelId)
    setShowModelPicker(false)
    setError(null)
    const q = modelQuantize ?? 4
    setQuantize(q)

    // Use the explicit category from model picker — no guessing
    setSessionMode(mode)
    setSourceImage(null)

    // Reset ALL settings to defaults for the new model (not just steps/quantize).
    // Without this, guidance, strength, width, height, count, seed, negativePrompt
    // from the previous model would persist and confuse users.
    const defaultSteps = getDefaultSteps(modelId)
    setSettings({
      steps: defaultSteps,
      width: 1024,
      height: 1024,
      guidance: 3.5,
      negativePrompt: '',
      seed: undefined,
      count: 1,
      quantize: q,
      strength: 0.8
    })

    // Auto-start server, passing imageMode so it's stored in session config
    setServerStatus('starting')
    try {
      const result = await window.api.image.startServer(modelId, q, mode)
      if (result.success) {
        setServerSessionId(result.sessionId ?? null)
        setServerPort(result.port ?? null)
        setShowLogs(true) // Auto-show logs during startup so user can see loading progress
        // Status will transition to 'running' via polling or session events
      } else {
        setServerStatus('error')
        setError(result.error || 'Failed to start server')
      }
    } catch (err) {
      setServerStatus('error')
      setError((err as Error).message)
    }
  }, [serverStatus])

  const handleSubmit = useCallback(async (prompt: string) => {
    if (!serverPort || serverStatus !== 'running' || !selectedModel) return

    // Edit mode requires a source image
    if (sessionMode === 'edit' && !sourceImage) {
      setError('Upload a source image before editing.')
      return
    }

    setGenerating(true)
    setError(null)

    try {
      // Create image session if we don't have one
      let sessionId = currentSessionId
      if (!sessionId) {
        const result = await window.api.image.createSession(selectedModel, sessionMode)
        if (result.success && result.session) {
          sessionId = result.session.id
          setCurrentSessionId(sessionId)
          await loadSessions()
        } else {
          throw new Error('Failed to create image session')
        }
      }

      let result: any
      if (sessionMode === 'edit') {
        result = await window.api.image.edit({
          sessionId: sessionId!,
          prompt,
          negativePrompt: settings.negativePrompt || undefined,
          model: selectedModel,
          imageBase64: sourceImage!.dataUrl,
          width: settings.width,
          height: settings.height,
          steps: settings.steps,
          guidance: settings.guidance,
          strength: settings.strength,
          seed: settings.seed,
          serverPort
        })
      } else {
        result = await window.api.image.generate({
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
      }

      if (result.success && result.generations) {
        setGenerations(prev => [...prev, ...result.generations])
        await loadSessions() // Refresh session list (updatedAt changed)
      } else {
        setError(result.error || (sessionMode === 'edit' ? 'Edit failed' : 'Generation failed'))
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setGenerating(false)
    }
  }, [serverPort, serverStatus, selectedModel, currentSessionId, settings, sessionMode, sourceImage, loadSessions])

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
    // Stop the running server before switching models (including error state cleanup)
    if (serverStatus === 'running' || serverStatus === 'starting' || serverStatus === 'error') {
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
    // Restore sessionMode from the selected session's type
    const session = sessions.find(s => s.id === sessionId)
    if (session?.sessionType) {
      setSessionMode(session.sessionType)
    }
  }, [sessions])

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    await window.api.image.deleteSession(sessionId)
    if (currentSessionId === sessionId) {
      setCurrentSessionId(null)
      setGenerations([])
    }
    await loadSessions()
  }, [currentSessionId, loadSessions])

  const handleSettingsChange = useCallback((newSettings: ImageSettings) => {
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
          mode={sessionMode}
          onSettings={() => setShowSettings(!showSettings)}
          onLogs={() => setShowLogs(!showLogs)}
          onStop={handleStop}
          onChangeModel={handleChangeModel}
          onSelectModel={(modelId, category) => {
            // Quick switch: stop current, start new model
            handleModelSelect(modelId, undefined, category)
          }}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        />

        {showSettings && (
          <ImageSettings
            settings={settings}
            onChange={handleSettingsChange}
            model={selectedModel}
            mode={sessionMode}
          />
        )}

        {showLogs && (
          <div className="h-48 border-b border-border flex-shrink-0">
            {serverSessionId ? (
              <LogsPanel
                sessionId={serverSessionId}
                sessionStatus={serverStatus}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-sm text-muted-foreground">
                Start an image model to view server logs
              </div>
            )}
          </div>
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
            mode={sessionMode}
          />
        </div>

        <ImagePromptBar
          onGenerate={handleSubmit}
          disabled={serverStatus !== 'running'}
          generating={generating}
          settings={settings}
          onSettingsChange={handleSettingsChange}
          mode={sessionMode}
          sourceImage={sourceImage}
          onSourceImageChange={setSourceImage}
        />
      </div>
    </div>
  )
}

// getDefaultSteps is now imported from shared/imageModels.ts
