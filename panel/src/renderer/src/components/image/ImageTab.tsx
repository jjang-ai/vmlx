import { useState, useEffect, useCallback, useRef, useSyncExternalStore } from 'react'
import { useTranslation } from '../../i18n'
import { ImageModelPicker } from './ImageModelPicker'
import { ImagePromptBar } from './ImagePromptBar'
import { ImageGallery } from './ImageGallery'
import { ImageHistory } from './ImageHistory'
import { ImageTopBar } from './ImageTopBar'
import { ImageSettings } from './ImageSettings'
import { LogsPanel } from '../sessions/LogsPanel'
import { getDefaultSteps, getDefaultGuidance, getImageModel, resolveImageModelFromDirectoryName } from '../../../../shared/imageModels'
import { imageRuntimeSnapshot, type ImageCapabilities, type ImageServerStatus } from '../../../../shared/imageCapabilities'
import { ImageSubmissionGuard } from '../../../../shared/imageSubmissionGuard'
import { defaultImageRuntimeSettings } from '../../../../shared/imageRuntimeSettings'
import { ImageDraftStore } from '../../../../shared/imageDrafts'
import type { ImageServerSettings } from './ImageModelPicker'
import type { ImageJobProgress } from '../../../../shared/imageJobProgress'

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

type ServerStatus = ImageServerStatus

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
  progress?: ImageJobProgress | null
  generating: boolean
  cancelling?: boolean
  startTime: number | null
  sessionId: string | null
}

// Survives page unmount, not application restart. Media stays out of preferences.
const imageDrafts = new ImageDraftStore()

export function ImageTab() {
  const { t } = useTranslation()
  const draftSnapshot = useSyncExternalStore(imageDrafts.subscribe, imageDrafts.getSnapshot)
  const { currentSessionId, draft } = draftSnapshot
  const { prompt, sourceImage, maskBase64, iteratePrompt, iterateCounter } = draft
  const setPrompt = (prompt: string) => { imageDrafts.update(draftSnapshot, { prompt }) }
  const setMaskBase64 = (maskBase64: string | null) => { imageDrafts.update(draftSnapshot, { maskBase64 }) }
  const [sessions, setSessions] = useState<ImageSessionInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [selectedModelDisplayName, setSelectedModelDisplayName] = useState<string | null>(null)
  const [serverStatus, setServerStatus] = useState<ServerStatus>('stopped')
  const [serverPort, setServerPort] = useState<number | null>(null)
  // What the loaded model actually accepts (from /health.image); null until known.
  const [capabilities, setCapabilities] = useState<ImageCapabilities | null>(null)
  const [serverSessionId, _setServerSessionId] = useState<string | null>(null)
  const serverSessionIdRef = useRef<string | null>(null)
  const healthRevision = useRef(0)
  const wakePending = useRef(false)
  const setServerSessionId = (id: string | null) => {
    serverSessionIdRef.current = id
    _setServerSessionId(id)
    if (id) imageDrafts.activate(id)
  }
  const [showSettings, setShowSettings] = useState(false)
  const [showLogs, setShowLogs] = useState(false)
  const [showModelPicker, setShowModelPicker] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [jobProgress, setJobProgress] = useState<ImageJobProgress | null>(null)
  const [jobStartTime, setJobStartTime] = useState<number | null>(null)
  useEffect(() => { if (!generating) { setJobProgress(null); setJobStartTime(null) } }, [generating])
  const [cancelling, setCancelling] = useState(false)
  useEffect(() => { if (!generating) setCancelling(false) }, [generating])
  const submissionGuard = useRef(new ImageSubmissionGuard())
  const [generations, setGenerations] = useState<ImageGenerationInfo[]>([])
  const [error, setError] = useState<string | null>(null)
  // Non-fatal advisory from the start handler (e.g. low-precision edit variant) with an optional alternative to start instead.
  const [warning, setWarning] = useState<{ text: string; tone: 'warning' | 'info'; alternativePath?: string; alternativeName?: string; alternativeBits?: number } | null>(null)
  const [quantize, setQuantize] = useState<number>(4)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Edit mode state
  const [sessionMode, setSessionMode] = useState<'generate' | 'edit'>('generate')

  // Image generation settings (quick settings + full settings)
  const [settings, setSettings] = useState<ImageSettings>(() => defaultImageRuntimeSettings('', 0))
  const [settingsOwner, setSettingsOwner] = useState<string | null>(null)
  const settingsRevision = useRef(0)
  const settingsEdited = useRef(false)
  const hydrateSettings = useCallback(async (id: string, adoptLegacy: boolean, revision: number) => {
    const restored = await window.api.image.getRuntimeSettings(id, adoptLegacy)
    if (settingsRevision.current !== revision) return
    setSettings(restored)
    settingsEdited.current = false
    setSettingsOwner(id)
  }, [])

  // The server settings (host, port, api key, log level, mflux class) the
  // current server was started with; a sibling variant offered by the
  // low-precision warning starts with the same ones.
  const serverSettingsRef = useRef<ImageServerSettings | undefined>(undefined)
  useEffect(() => {
    if (!settingsOwner || !settingsEdited.current) return
    // Capture the owner and values together; a later model switch must not
    // save old controls into a new model's session.
    // Do not defer this behind a timer cancelled by page navigation. Main
    // serializes the small per-session record update synchronously.
    window.api.image.saveRuntimeSettings(settingsOwner, settings).catch(error => setError((error as Error).message))
  }, [settings, settingsOwner])

  // Load image sessions on mount
  useEffect(() => {
    loadSessions()
  }, [])

  // Check if an image server is already running
  useEffect(() => {
    const revision = ++settingsRevision.current
    window.api.image.getRunningServer().then(async (server: any) => {
      if (revision !== settingsRevision.current) return
      if (server) {
        const name = server.modelName
        const canonical = server.canonicalModelId || resolveImageModelFromDirectoryName(name)?.id || name
        setSelectedModel(canonical)
        setSelectedModelDisplayName(server.displayModelName || name)
        // Discovery identifies the session, not loaded model readiness.
        setServerStatus(server.status === 'standby' ? 'standby' : 'starting')
        setServerPort(server.port)
        setServerSessionId(server.sessionId)
        setShowModelPicker(false)

        // Read imageMode from session config — no guessing from name
        const mode = server.imageMode || 'generate'
        setSessionMode(mode)

        // Restore quantize from server config
        const q = server.quantize ?? 0
        setQuantize(q)

        await hydrateSettings(server.sessionId, true, revision)
      }
    }).catch(error => { if (revision === settingsRevision.current) setError((error as Error).message) })
    return () => { if (revision === settingsRevision.current) settingsRevision.current++ }
  }, [])

  // Check if image generation is in-flight (persists across tab switches)
  // Also reload gallery if generation completed while we were on another tab
  useEffect(() => {
    const snapshot = submissionGuard.current.snapshot()
    window.api.image.isGenerating().then((status: ImageGenerationStatus) => {
      if (!submissionGuard.current.canApply(snapshot)) return
      if (status.generating) {
        setGenerating(true)
        setCancelling(status.cancelling === true)
        setJobProgress(status.progress ?? null)
        setJobStartTime(status.startTime)
      } else if (status.sessionId || currentSessionId) {
        // Generation may have completed while we were away — reload gallery
        const sessionIdToRefresh = currentSessionId || status.sessionId!
        loadGenerations(sessionIdToRefresh)
        loadSessions()
      }
    }).catch(() => {})
  }, [])

  // Listen for session events to detect when server becomes ready
  useEffect(() => {
    const unsubReady = window.api.sessions.onReady((data: any) => {
      if (data.sessionId === serverSessionIdRef.current) {
        healthRevision.current++
        setServerStatus('starting')
        // Fetch the session to get port
        window.api.sessions.get(data.sessionId).then((s: any) => {
          if (s && data.sessionId === serverSessionIdRef.current) setServerPort(s.port)
        }).catch(() => {})
      }
    })
    const unsubStopped = window.api.sessions.onStopped((data: any) => {
      if (data.sessionId === serverSessionIdRef.current) {
        healthRevision.current++
        setServerStatus('stopped')
        setServerPort(null)
        setGenerating(false)  // Server stopped — cancel any in-flight generation
      }
    })
    const unsubStandby = window.api.sessions.onStandby((data: any) => {
      if (data.sessionId === serverSessionIdRef.current) {
        healthRevision.current++
        setServerStatus('standby')
        setCapabilities(null)
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === serverSessionIdRef.current) {
        healthRevision.current++
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
      unsubStandby()
      unsubError()
    }
  }, []) // Uses ref, not state — no dependency needed

  // Observe model readiness throughout its lifetime, not just process startup.
  // One in-flight poll, cancelled on session/port change, prevents a late old
  // health response from enabling the replacement model's composer.
  useEffect(() => {
    setCapabilities(null)
    if (!serverPort || !serverSessionId) return
    let cancelled = false
    let inFlight = false
    let controller: AbortController | null = null
    const poll = async () => {
      if (inFlight || cancelled) return
      inFlight = true
      const revision = healthRevision.current
      controller = new AbortController()
      const timeout = setTimeout(() => controller?.abort(), 5000)
      try {
        const resp = await fetch(`http://127.0.0.1:${serverPort}/health`, { signal: controller.signal })
        if (!resp.ok) throw new Error(`Image health HTTP ${resp.status}`)
        const snapshot = imageRuntimeSnapshot(await resp.json())
        if (cancelled || serverSessionIdRef.current !== serverSessionId || revision !== healthRevision.current) return
        setServerStatus(wakePending.current && snapshot.status === 'standby' ? 'starting' : snapshot.status)
        setCapabilities(snapshot.capabilities)
      } catch (_) {
        if (!cancelled && serverSessionIdRef.current === serverSessionId && revision === healthRevision.current) {
          setCapabilities(null)
          setServerStatus(previous => previous === 'starting' ? 'starting' : 'error')
        }
      } finally {
        clearTimeout(timeout)
        inFlight = false
      }
    }
    void poll()
    pollRef.current = setInterval(poll, 2000)
    return () => {
      cancelled = true
      controller?.abort()
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [serverSessionId, serverPort])

  // Load generations when session changes
  useEffect(() => {
    if (currentSessionId) {
      loadGenerations(currentSessionId)
    } else {
      setGenerations([])
    }
  }, [currentSessionId, draftSnapshot.owner, draftSnapshot.epoch])

  const loadSessions = useCallback(async () => {
    const result = await window.api.image.getSessions()
    setSessions(result || [])
  }, [])

  const loadGenerations = useCallback(async (sessionId: string) => {
    const token = imageDrafts.getSnapshot()
    const result = await window.api.image.getGenerations(sessionId)
    if (imageDrafts.isCurrent(token) && token.currentSessionId === sessionId) setGenerations(result || [])
  }, [])

  const syncGenerationStatus = useCallback(async () => {
    const snapshot = submissionGuard.current.snapshot()
    if (!submissionGuard.current.isCurrent(snapshot)) return
    const status: ImageGenerationStatus = await window.api.image.isGenerating()
    if (!submissionGuard.current.isCurrent(snapshot)) return
    if (status.generating) {
      setGenerating(true)
      setCancelling(status.cancelling === true)
      setJobProgress(status.sessionId === currentSessionId ? status.progress ?? null : null)
      setJobStartTime(status.startTime)
      return
    }

    // Progress from the active request is useful during submission. An idle
    // response is different: it cannot release locally owned preprocessing.
    if (!submissionGuard.current.canApply(snapshot)) return
    setGenerating(false)
    const sessionIdToRefresh = currentSessionId
    if (sessionIdToRefresh) {
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
    imageDrafts.update(draftSnapshot, { sourceImage: img, maskBase64: null })
  }, [draftSnapshot])

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
      settingsOwner,
    }
    const serverWasLive = serverStatus === 'running' || serverStatus === 'starting' || serverStatus === 'standby'
    const revision = ++settingsRevision.current
    if (settingsOwner && settingsEdited.current) {
      try { await window.api.image.saveRuntimeSettings(settingsOwner, settings) }
      catch (error) { setError((error as Error).message); return }
      if (revision !== settingsRevision.current) return
    }
    settingsEdited.current = false
    setSettingsOwner(null)

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
    // Keep the current draft until validation succeeds and a new server owner
    // is activated. A rejected folder must not consume an unsent edit.

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
      const result = await window.api.image.startServer(modelId, modelQuantize ?? 0, category, serverSettings)
      if (result.success) {
        if (!result.sessionId) throw new Error('Image server started without a session identity')
        // The launch resolver owns local-folder identity (including q8-style
        // subfolders). Use its defaults, not a second basename-only guess.
        if (result.imageMode) setSessionMode(result.imageMode)
        if (revision !== settingsRevision.current) return
        serverSettingsRef.current = serverSettings
        setServerSessionId(result.sessionId ?? null)
        setServerPort(result.port ?? null)
        await hydrateSettings(result.sessionId, false, revision)
        if (revision !== settingsRevision.current) return
        // Switching over a live server: the previous session's stopped event
        // arrived while this tab still tracked that session and left the
        // status at 'stopped', and the new session's ready event fired before
        // the tab learned its id. Re-arm readiness on the new session: the
        // health poll (or a later ready event) promotes it to 'running'.
        // Seen live: q4 -> "Use q8" left the tab at Stopped with a disabled
        // composer while the q8 server answered /health.
        setServerStatus('starting')
        if (result.warningCode) {
          const p = result.warningParams || {}
          setWarning({
            text: t(`image.server.warnings.${result.warningCode}`, { ...p, defaultValue: result.warningCode }),
            tone: result.warningCode === 'editLowPrecisionUntested' ? 'info' : 'warning',
            alternativePath: p.alternativePath || undefined,
            alternativeName: p.alternative || undefined,
            alternativeBits: p.alternativeBits ? Number(p.alternativeBits) : undefined,
          })
        } else {
          setWarning(null)
        }
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
          setSettingsOwner(previous.settingsOwner)
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
  }, [serverStatus, selectedModel, selectedModelDisplayName, quantize, sessionMode, settings, settingsOwner, hydrateSettings, describeStartError, t])

  const handleSubmit = useCallback(async (prompt: string, overrideSettings?: Partial<ImageSettings>) => {
    if (!serverPort || serverStatus !== 'running' || !selectedModel || !settingsOwner) return

    // Merge override settings (used by reiteration to bypass React batching)
    const s = overrideSettings ? { ...settings, ...overrideSettings } : settings

    // Edit mode requires a source image (gen mode allows optional source for img2img)
    if (sessionMode === 'edit' && !sourceImage) {
      setError(t('image.tab.uploadSourceFirst'))
      return
    }

    // Session creation and IPC preprocessing precede backend busy state. An
    // idle status response must not clear this submission or permit a duplicate.
    const owner = submissionGuard.current.begin()
    if (owner === null) return
    setGenerating(true)
    setError(null)
    const draftToken = imageDrafts.getSnapshot()

    try {
      // Create image session if we don't have one
      let sessionId = currentSessionId
      if (!sessionId) {
        const result = await window.api.image.createSession(selectedModel, sessionMode)
        if (result.success && result.session) {
          sessionId = result.session.id
          imageDrafts.promote(draftToken, sessionId!)
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
        await loadGenerations(sessionId!)
        await loadSessions() // Refresh session list (updatedAt changed)
      } else {
        if (!result.cancelled) setError(result.error || (sessionMode === 'edit' ? t('image.tab.editFailed') : t('image.tab.generationFailed')))
      }
    } catch (err) {
      setError((err as Error).message)
    } finally {
      if (submissionGuard.current.finish(owner)) setGenerating(false)
    }
  }, [serverPort, serverStatus, selectedModel, currentSessionId, settings, settingsOwner, sessionMode, sourceImage, maskBase64, loadSessions, loadGenerations])

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
    if (serverStatus === 'running' || serverStatus === 'starting' || serverStatus === 'standby') {
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

  const handleWake = useCallback(async () => {
    const id = serverSessionIdRef.current
    if (!id || serverStatus !== 'standby') return
    healthRevision.current++
    wakePending.current = true
    setServerStatus('starting')
    setError(null)
    try {
      const result = await window.api.sessions.wake(id)
      if (result?.success === false) throw new Error(result.error || t('image.tab.serverError'))
      // Loaded health, not IPC success alone, enables generation.
    } catch (error) {
      if (serverSessionIdRef.current === id) {
        setServerStatus('standby')
        setError((error as Error).message)
      }
    } finally {
      wakePending.current = false
    }
  }, [serverStatus, t])

  const handleNewSession = useCallback(() => {
    imageDrafts.newConversation()
    setGenerations([])
    setError(null)
    // Reset mode to match the currently running model's category
    if (selectedModel) {
      const modelDef = getImageModel(selectedModel) || resolveImageModelFromDirectoryName(selectedModel.split('/').filter(Boolean).pop() || selectedModel)
      if (modelDef) setSessionMode(modelDef.category)
    }
  }, [selectedModel])

  const handleSelectSession = useCallback(async (sessionId: string) => {
    imageDrafts.select(sessionId)
    setError(null)
    // Restore sessionMode from the selected session's type
    const session = sessions.find(s => s.id === sessionId)
    if (session?.sessionType) {
      setSessionMode(session.sessionType)
    }
  }, [sessions])

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    const result = await window.api.image.deleteSession(sessionId)
    if (!result.success) {
      setError(result.error || t('image.tab.deleteFailedPlain'))
      return
    }
    imageDrafts.remove(sessionId)
    if (currentSessionId === sessionId) {
      setGenerations([])
      setError(null)
    }
    await loadSessions()
  }, [currentSessionId, loadSessions, t])

  const handleSettingsChange = useCallback((newSettings: ImageSettings) => {
    settingsEdited.current = true
    setSettings(newSettings)
  }, [])

  // Show model picker if no model selected, or when switching while a server runs
  const serverLive = serverStatus === 'running' || serverStatus === 'starting' || serverStatus === 'standby'
  if (showModelPicker && (!selectedModel || serverLive)) {
    return (
      <div className="h-full min-h-0 min-w-0 flex flex-col">
        <ImageModelPicker
          onSelect={handleModelSelect}
          currentModel={selectedModel && serverLive ? (selectedModelDisplayName || selectedModel) : null}
          onKeepCurrent={selectedModel && serverLive ? () => setShowModelPicker(false) : undefined}
        />
      </div>
    )
  }

  return (
    <div className="h-full min-h-0 min-w-0 flex">
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
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
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
          onWake={handleWake}
          onChangeModel={handleChangeModel}
          sidebarCollapsed={sidebarCollapsed}
          onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
        />

        {showSettings && settingsOwner && (
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

        {warning && (
          <div role="alert" data-vmlx-tone={warning.tone} data-vmlx-control="image-warning" className={`mx-4 mt-2 px-3 py-2 rounded-md text-sm flex items-center gap-3 ${warning.tone === 'info' ? 'bg-primary/10 border border-primary/30 text-foreground' : 'bg-warning/10 border border-warning/30 text-warning'}`}>
            <span className="flex-1">{warning.text}</span>
            {warning.alternativePath && (
              <button
                type="button"
                data-vmlx-control="image-use-alternative"
                onClick={() => { const alt = warning; setWarning(null); handleModelSelect(alt.alternativePath!, alt.alternativeBits, 'edit', serverSettingsRef.current) }}
                className="px-2 py-1 rounded border border-warning/40 hover:bg-warning/20 text-xs"
              >
                {t('image.server.warnings.useAlternative', { name: warning.alternativeName || '' })}
              </button>
            )}
            <button onClick={() => setWarning(null)} className="text-xs underline">{t('image.tab.dismissError')}</button>
          </div>
        )}
        {error && (
          <div role="alert" data-vmlx-tone="error" className="mx-4 mt-2 px-3 py-2 bg-destructive/10 border border-destructive/20 rounded-md text-sm text-destructive">
            {error}
            <button onClick={() => setError(null)} className="ml-2 text-xs underline">{t('image.tab.dismissError')}</button>
          </div>
        )}

        <div className="flex-1 min-h-0 overflow-hidden">
          <ImageGallery
            generations={generations}
            generating={generating}
            progress={jobProgress}
            startTime={jobStartTime}
            cancelling={cancelling}
            mode={sessionMode}
            onRegenerate={async (gen) => {
              // Iterate: set the output image as source for img2img
              // Read the generated image and set as source
              try {
                const token = imageDrafts.getSnapshot()
                const dataUrl = await window.api.image.readFile(gen.imagePath)
                if (!imageDrafts.isCurrent(token)) return
                if (!dataUrl) {
                  setError(t('image.tab.iterateLoadFailedDeleted'))
                  return
                }
                imageDrafts.update(token, {
                  sourceImage: { dataUrl, name: `iterate-${gen.id.slice(0, 8)}.png` },
                  maskBase64: null, prompt: '', iteratePrompt: gen.prompt,
                  iterateCounter: token.draft.iterateCounter + 1,
                })
                // Restore settings from this generation
                settingsEdited.current = true
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
          cancelling={cancelling}
          key={`${draftSnapshot.owner}:${currentSessionId}:${draftSnapshot.epoch}`}
          prompt={prompt}
          onPromptChange={setPrompt}
          onGenerate={handleSubmit}
          disabled={serverStatus !== 'running' || !settingsOwner}
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
          onClearIterate={() => { imageDrafts.update(draftSnapshot, { iteratePrompt: null, sourceImage: null, maskBase64: null }) }}
        />
      </div>
    </div>
  )
}

// getDefaultSteps is now imported from shared/imageModels.ts
