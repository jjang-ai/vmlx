import { useState, useEffect, useCallback, useRef } from 'react'
import { useTranslation } from '../../i18n'
import { ImageModelPicker } from './ImageModelPicker'
import { ImagePromptBar } from './ImagePromptBar'
import { ImageGallery } from './ImageGallery'
import { ImageHistory } from './ImageHistory'
import { ImageTopBar } from './ImageTopBar'
import { ImageSettings } from './ImageSettings'
import { LogsPanel } from '../sessions/LogsPanel'
import { getDefaultSteps, getDefaultGuidance, getImageModel, resolveImageModelFromDirectoryName } from '../../../../shared/imageModels'
import { fetchImageCapabilities, type ImageCapabilities } from '../../../../shared/imageCapabilities'
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

interface ImageGenerationStatus {
  generating: boolean
  startTime: number | null
  sessionId: string | null
}

export function ImageTab() {
  const { t } = useTranslation()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [sessions, setSessions] = useState<ImageSessionInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [selectedModelDisplayName, setSelectedModelDisplayName] = useState<string | null>(null)
  const [serverStatus, setServerStatus] = useState<ServerStatus>('stopped')
  const [serverPort, setServerPort] = useState<number | null>(null)
  // What the loaded model actually accepts (from /health.image); null until known.
  const [capabilities, setCapabilities] = useState<ImageCapabilities | null>(null)
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
  const [maskBase64, setMaskBase64] = useState<string | null>(null)
  // Iterate: pre-fill prompt in the prompt bar (counter forces re-trigger for same prompt)
  const [iteratePrompt, setIteratePrompt] = useState<string | null>(null)
  const [iterateCounter, setIterateCounter] = useState(0)

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
        const canonical = server.canonicalModelId || resolveImageModelFromDirectoryName(name)?.id || name
        setSelectedModel(canonical)
        setSelectedModelDisplayName(server.displayModelName || name)
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

        // Restore proper default steps + guidance for this model when the
        // saved settings don't already carry values (generation_config-like
        // recommendations from `shared/imageModels.ts`). User-saved values
        // always win via the `prev` spread below.
        const defaultSteps = getDefaultSteps(name)
        const defaultGuidance = getDefaultGuidance(name)
        setSettings(prev => ({
          ...prev,
          steps: prev.steps || defaultSteps,
          guidance: prev.guidance ?? defaultGuidance,
          quantize: q,
        }))
      }
    }).catch(() => {})
  }, [])

  // Check if image generation is in-flight (persists across tab switches)
  // Also reload gallery if generation completed while we were on another tab
  useEffect(() => {
    window.api.image.isGenerating().then((status: ImageGenerationStatus) => {
      if (status.generating) {
        setGenerating(true)
        if (status.sessionId) setCurrentSessionId(status.sessionId)
      } else if (status.sessionId || currentSessionId) {
        // Generation may have completed while we were away — reload gallery
        const sessionIdToRefresh = status.sessionId || currentSessionId!
        setCurrentSessionId(sessionIdToRefresh)
        loadGenerations(sessionIdToRefresh)
        loadSessions()
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
        setGenerating(false)  // Server stopped — cancel any in-flight generation
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === serverSessionIdRef.current) {
        setServerStatus('error')
        const errMsg = data.error || t('image.tab.serverError')
        // Detect gated/auth errors and show helpful message
        const isGated = /40[13]|gated|access.*denied|authentication|authorized|forbidden/i.test(errMsg)
        if (isGated) {
          setError(
            t('image.tab.gatedDownloadError', { error: errMsg.slice(0, 200) })
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

  // Read the loaded model's real capabilities once the server runs; forget
  // them when it stops so stale limits never apply to the next model.
  useEffect(() => {
    if (serverStatus === 'running' && serverPort) {
      let cancelled = false
      fetchImageCapabilities(serverPort).then(caps => { if (!cancelled) setCapabilities(caps) })
      return () => { cancelled = true }
    }
    if (serverStatus === 'stopped' || serverStatus === 'error') setCapabilities(null)
    return undefined
  }, [serverStatus, serverPort])

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

  const syncGenerationStatus = useCallback(async () => {
    const status: ImageGenerationStatus = await window.api.image.isGenerating()
    if (status.generating) {
      setGenerating(true)
      if (status.sessionId && status.sessionId !== currentSessionId) {
        setCurrentSessionId(status.sessionId)
      }
      return
    }

    setGenerating(false)
    const sessionIdToRefresh = status.sessionId || currentSessionId
    if (sessionIdToRefresh) {
      setCurrentSessionId(sessionIdToRefresh)
      await loadGenerations(sessionIdToRefresh)
      await loadSessions()
    }
  }, [currentSessionId, loadGenerations, loadSessions])

  useEffect(() => {
    if (!generating) return
    const timer = setInterval(syncGenerationStatus, 1500)
    syncGenerationStatus().catch(() => {})
    return () => clearInterval(timer)
  }, [generating, syncGenerationStatus])

  const handleSourceImageChange = useCallback((img: { dataUrl: string; name: string } | null) => {
    setSourceImage(img)
    setMaskBase64(null)
  }, [])

  // Main-process start failures carry a stable code plus parameters; translate
  // those here and fall back to the English message the process sent.
  const describeStartError = useCallback((result: { error?: string; errorCode?: string; errorParams?: Record<string, string> }): string => {
    if (result.errorCode) {
      return t(`image.server.errors.${result.errorCode}`, { ...(result.errorParams || {}), defaultValue: result.error || t('sessions.view.toast.failedToStartServer') })
    }
    return result.error || t('sessions.view.toast.failedToStartServer')
  }, [t])

  const handleModelSelect = useCallback(async (modelId: string, modelQuantize?: number, category?: 'generate' | 'edit', serverSettings?: ImageServerSettings) => {
    const mode = category || 'generate'

    // The main process validates the requested folder BEFORE it stops the
    // running server, so a bad path (unplugged drive, typo, folder of
    // variants) leaves the current server, and this selection, untouched.
    // Snapshot the selection so it can be restored when that happens.
    const previous = {
      model: selectedModel,
      displayName: selectedModelDisplayName,
      status: serverStatus,
      quantize,
      mode: sessionMode,
      settings,
    }
    const serverWasLive = serverStatus === 'running' || serverStatus === 'starting'

    setSelectedModel(modelId)
    setSelectedModelDisplayName(null)
    setShowModelPicker(false)
    setError(null)
    // A custom path is a directory, not a registry id: resolve its basename to
    // the registry entry (FLUX.1-dev-mflux-8bit -> dev) so quantize options,
    // default steps and guidance follow the model instead of the generic 4/3.5.
    const modelDef = getImageModel(modelId) || resolveImageModelFromDirectoryName(modelId.split('/').filter(Boolean).pop() || modelId)
    const resolvedModelId = modelDef?.id ?? modelId
    const q = modelQuantize ?? modelDef?.quantizeOptions[0] ?? 4
    setQuantize(q)

    // Use the explicit category from model picker — no guessing
    setSessionMode(mode)
    setSourceImage(null)
    setMaskBase64(null)

    // Reset ALL settings to defaults for the new model (not just steps/quantize).
    // Without this, guidance, strength, width, height, count, seed, negativePrompt
    // from the previous model would persist and confuse users.
    // Steps + guidance come from the model's canonical defaults in
    // `shared/imageModels.ts` (e.g. Schnell=4/0, Dev=20/3.5, Qwen=20/4,
    // Fill=20/30) — mirrors the Python engine's DEFAULT_STEPS table so the
    // UI doesn't silently clobber model-recommended values.
    const defaultSteps = getDefaultSteps(resolvedModelId)
    const defaultGuidance = getDefaultGuidance(resolvedModelId)
    setSettings({
      steps: defaultSteps,
      width: 1024,
      height: 1024,
      guidance: defaultGuidance,
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
        // A local folder runs at its own precision whatever the picker said;
        // show and persist the effective value the main process resolved.
        if (typeof result.quantize === 'number' && result.quantize !== q) {
          setQuantize(result.quantize)
          setSettings(prev => ({ ...prev, quantize: result.quantize as number }))
        }
        setShowLogs(true) // Auto-show logs during startup so user can see loading progress
        // Status will transition to 'running' via polling or session events
      } else {
        const message = describeStartError(result)
        if (result.serverKept && serverWasLive) {
          // Nothing was stopped: put the previous selection back and surface why.
          setSelectedModel(previous.model)
          setSelectedModelDisplayName(previous.displayName)
          setServerStatus(previous.status)
          setQuantize(previous.quantize)
          setSessionMode(previous.mode)
          setSettings(previous.settings)
          setShowModelPicker(false)
        } else {
          setServerStatus('error')
        }
        setError(message)
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
      setError(t('image.tab.uploadSourceFirst'))
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
          throw new Error(t('image.tab.createSessionFailed'))
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
          maskBase64: maskBase64 || undefined,
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
        setError(result.error || (sessionMode === 'edit' ? t('image.tab.editFailed') : t('image.tab.generationFailed')))
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setGenerating(false)
    }
  }, [serverPort, serverStatus, selectedModel, currentSessionId, settings, sessionMode, sourceImage, maskBase64, loadSessions])

  const handleStop = useCallback(async () => {
    try {
      await window.api.image.stopServer()
      setServerStatus('stopped')
      setServerPort(null)
      setServerSessionId(null)
      setGenerating(false)  // Reset generating state when server stops
    } catch (err) {
      console.error('Failed to stop image server:', err)
    }
  }, [])

  const handleChangeModel = useCallback(async () => {
    if (serverStatus === 'running' || serverStatus === 'starting') {
      // Keep the server up: the picker opens over it, and the main process
      // only stops it once the replacement folder has been validated. "Keep
      // current" in the picker returns here with nothing changed.
      setShowModelPicker(true)
      return
    }
    if (serverStatus === 'error') {
      await handleStop()
    }
    setSelectedModel(null)
    setSelectedModelDisplayName(null)
    setShowModelPicker(true)
  }, [serverStatus, handleStop])

  const handleNewSession = useCallback(() => {
    setCurrentSessionId(null)
    setGenerations([])
    setSourceImage(null)
    setMaskBase64(null)
    setError(null)
    setIteratePrompt(null)
    // Reset mode to match the currently running model's category
    if (selectedModel) {
      const modelDef = getImageModel(selectedModel) || resolveImageModelFromDirectoryName(selectedModel.split('/').filter(Boolean).pop() || selectedModel)
      if (modelDef) setSessionMode(modelDef.category)
    }
  }, [selectedModel])

  const handleSelectSession = useCallback(async (sessionId: string) => {
    setCurrentSessionId(sessionId)
    setSourceImage(null)
    setMaskBase64(null)
    setError(null)
    setIteratePrompt(null)
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
      handleSourceImageChange(null)
      setError(null)
    }
    await loadSessions()
  }, [currentSessionId, handleSourceImageChange, loadSessions])

  const handleSettingsChange = useCallback((newSettings: ImageSettings) => {
    setSettings(newSettings)
  }, [])

  // Show model picker if no model selected, or when switching while a server runs
  const serverLive = serverStatus === 'running' || serverStatus === 'starting'
  if (showModelPicker && (!selectedModel || serverLive)) {
    return (
      <div className="h-full flex flex-col">
        <ImageModelPicker
          onSelect={handleModelSelect}
          currentModel={selectedModel && serverLive ? (selectedModelDisplayName || selectedModel) : null}
          onKeepCurrent={selectedModel && serverLive ? () => setShowModelPicker(false) : undefined}
        />
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
          displayModelName={selectedModelDisplayName}
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
            capabilities={capabilities}
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
                {t('image.tab.startModelForLogs')}
              </div>
            )}
          </div>
        )}

        {error && (
          <div className="mx-4 mt-2 px-3 py-2 bg-destructive/10 border border-destructive/20 rounded-md text-sm text-destructive">
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-xs underline">{t('image.tab.dismissError')}</button>
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
                if (!dataUrl) {
                  setError(t('image.tab.iterateLoadFailedDeleted'))
                  return
                }
                handleSourceImageChange({ dataUrl, name: `iterate-${gen.id.slice(0, 8)}.png` })
                // Restore settings from this generation
                setSettings(prev => ({
                  ...prev,
                  steps: gen.steps,
                  width: gen.width,
                  height: gen.height,
                  guidance: gen.guidance,
                  strength: 0.85, // High default so iterate changes are visible
                  negativePrompt: gen.negativePrompt || '',
                  seed: undefined,
                }))
                // Pre-fill the prompt bar with the original prompt so user can modify
                setIteratePrompt(gen.prompt)
                setIterateCounter(c => c + 1) // force re-trigger even if same prompt
              } catch (err) {
                console.error('Failed to load image for iteration:', err)
                setError(t('image.tab.iterateLoadFailed'))
              }
            }}
            onDelete={async (gen) => {
              // ms#61: delete this image from the gallery. Unlinks the
              // file on disk (only if inside ~/.mlxstudio) and removes
              // the DB row.
              try {
                const r = await window.api.image.deleteGeneration(gen.id)
                if (!r.success) {
                  setError(t('image.tab.deleteFailed', { error: r.error || 'unknown' }))
                  return
                }
                // Drop from local state — no need to refetch the whole list.
                setGenerations(prev => prev.filter(g => g.id !== gen.id))
              } catch (err) {
                console.error('Failed to delete image:', err)
                setError(t('image.tab.deleteFailedPlain'))
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
          modelName={selectedModel}
          capabilities={capabilities}
          sourceImage={sourceImage}
          onSourceImageChange={handleSourceImageChange}
          maskBase64={maskBase64}
          onMaskChange={setMaskBase64}
          iteratePrompt={iteratePrompt}
          iterateCounter={iterateCounter}
          onClearIterate={() => { setIteratePrompt(null); handleSourceImageChange(null) }}
        />
      </div>
    </div>
  )
}

// getDefaultSteps is now imported from shared/imageModels.ts
