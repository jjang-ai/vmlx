import { useState, useEffect, useCallback, useRef } from 'react'
import { ImageModelPicker } from './ImageModelPicker'
import { ImagePromptBar } from './ImagePromptBar'
import { ImageGallery } from './ImageGallery'
import { ImageHistory } from './ImageHistory'
import { ImageTopBar } from './ImageTopBar'
import { ImageSettings } from './ImageSettings'
import { LogsPanel } from '../sessions/LogsPanel'
import { getDefaultSteps, getImageModel } from '../../../../shared/imageModels'
import type { ImageServerSettings } from './ImageModelPicker'

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

  // Load saved image settings on mount
  useEffect(() => {
    window.api.settings.get('image_settings').then((saved: string | null) => {
      if (saved) {
        try {
          const parsed = JSON.parse(saved)
          setSettings(prev => ({ ...prev, ...parsed, seed: undefined })) // Never restore seed
        } catch {}
      }
    })
  }, [])

  // Save settings when they change (debounced via the settings object reference)
  const settingsRef = useRef(settings)
  settingsRef.current = settings
  useEffect(() => {
    const timer = setTimeout(() => {
      const { seed, quantize, ...toSave } = settingsRef.current // Don't persist seed or quantize
      window.api.settings.set('image_settings', JSON.stringify(toSave)).catch(() => {})
    }, 500)
    return () => clearTimeout(timer)
  }, [settings.steps, settings.width, settings.height, settings.guidance, settings.negativePrompt, settings.count, settings.strength])

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

  const handleModelSelect = useCallback(async (modelId: string, modelQuantize?: number, category?: 'generate' | 'edit', serverSettings?: ImageServerSettings) => {
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
    // Use the provided quantize, or fall back to the model's first supported quantize option
    const modelDef = getImageModel(modelId)
    const q = modelQuantize ?? modelDef?.quantizeOptions[0] ?? 4
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
      const result = await window.api.image.startServer(modelId, q, mode, serverSettings)
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

  const handleSubmit = useCallback(async (prompt: string, overrideSettings?: Partial<ImageSettings>) => {
    if (!serverPort || serverStatus !== 'running' || !selectedModel) return

    // Merge override settings (used by reiteration to bypass React batching)
    const s = overrideSettings ? { ...settings, ...overrideSettings } : settings

    // Edit mode requires a source image (gen mode allows optional source for img2img)
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
          negativePrompt: s.negativePrompt || undefined,
          model: selectedModel,
          imageBase64: sourceImage!.dataUrl,
          width: s.width,
          height: s.height,
          steps: s.steps,
          guidance: s.guidance,
          strength: s.strength,
          seed: s.seed,
          serverPort
        })
      } else {
        const genParams: any = {
          sessionId: sessionId!,
          prompt,
          negativePrompt: s.negativePrompt || undefined,
          model: selectedModel,
          width: s.width,
          height: s.height,
          steps: s.steps,
          guidance: s.guidance,
          seed: s.seed,
          count: s.count,
          quantize: s.quantize,
          serverPort
        }
        // img2img: pass source image + strength when user uploaded an image in gen mode
        if (sourceImage) {
          genParams.imageBase64 = sourceImage.dataUrl
          genParams.strength = s.strength
        }
        result = await window.api.image.generate(genParams)
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
    setSourceImage(null)
    setError(null)
    // Reset mode to match the currently running model's category
    if (selectedModel) {
      const modelDef = getImageModel(selectedModel)
      if (modelDef) setSessionMode(modelDef.category)
    }
  }, [selectedModel])

  const handleSelectSession = useCallback(async (sessionId: string) => {
    setCurrentSessionId(sessionId)
    setSourceImage(null) // Clear source image when switching sessions
    setError(null)
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
      setSourceImage(null)
      setError(null)
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
          generating={generating}
          onSettings={() => setShowSettings(!showSettings)}
          onLogs={() => setShowLogs(!showLogs)}
          onStop={handleStop}
          onChangeModel={handleChangeModel}
          onSelectModel={(modelId, modelQuantize, category) => {
            // Quick switch: stop current, start new model with correct quantize
            handleModelSelect(modelId, modelQuantize, category)
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
            onRegenerate={async (gen) => {
              // Iterate: set the output image as source for img2img
              // Read the generated image and set as source
              try {
                const dataUrl = await window.api.image.readFile(gen.imagePath)
                if (dataUrl) {
                  setSourceImage({ dataUrl, name: `iterate-${gen.id.slice(0, 8)}.png` })
                  // Restore settings from this generation
                  setSettings(prev => ({
                    ...prev,
                    steps: gen.steps,
                    width: gen.width,
                    height: gen.height,
                    guidance: gen.guidance,
                    strength: gen.strength ?? 0.7,
                    negativePrompt: gen.negativePrompt || '',
                    seed: undefined,
                  }))
                }
              } catch (err) {
                console.error('Failed to load image for iteration:', err)
              }
            }}
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
