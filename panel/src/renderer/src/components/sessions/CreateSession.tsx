import { useState, useEffect, useRef } from 'react'
import { ArrowLeft } from 'lucide-react'
import {
  SessionConfigForm,
  SessionConfig,
  DEFAULT_CONFIG,
  DSV4_MAX_CACHE_BLOCKS,
  DSV4_PAGED_CACHE_BLOCK_SIZE,
} from './SessionConfigForm'
import { DownloadTab } from './DownloadTab'
import { DirectoryManager } from './DirectoryManager'
import { useTranslation } from '../../i18n'
import { useSessionsContext } from '../../contexts/SessionsContext'
import { formatModelBytes, formatResidentLoad } from './loadProgressFormat'
import {
  applyBundleDsv4PoolQuantToSessionConfig,
  applyBundleGenerationDefaultsToSessionConfig,
} from '../../../../shared/sessionGenerationDefaults'
import { usesExactTypedPromptDiskCache } from '../../../../shared/detectedFamilyNames'

interface ModelInfo {
  path: string
  name: string
  size?: string
  quantization?: string
}

interface CreateSessionProps {
  initialModelPath?: string | null
  onBack: () => void
  onCreated: (sessionId: string) => void
  /** Filter to show only image or text models */
  filterType?: 'text' | 'image'
}

export function CreateSession({ initialModelPath, onBack, onCreated, filterType: filterTypeProp }: CreateSessionProps) {
  const { t } = useTranslation()
  const { loadProgress } = useSessionsContext()
  const [sessionType, setSessionType] = useState<'local' | 'remote' | 'download'>('local')
  const [step, setStep] = useState<1 | 2>(initialModelPath ? 2 : 1)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState<string>(initialModelPath || '')
  const [autoDetectedType, setAutoDetectedType] = useState<'text' | 'image' | undefined>(filterTypeProp)
  const [modelFilter, setModelFilter] = useState('')
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_CONFIG)
  const [detectedCacheType, setDetectedCacheType] = useState<string | undefined>()
  const [detectedUsePagedCache, setDetectedUsePagedCache] = useState<boolean | undefined>(undefined)
  const [detectedCacheSubtype, setDetectedCacheSubtype] = useState<string | undefined>()
  const [detectedFamily, setDetectedFamily] = useState<string | undefined>()
  const [detectedArchitectureHints, setDetectedArchitectureHints] = useState<Record<string, string | number | boolean> | undefined>()
  const [detectedToolParser, setDetectedToolParser] = useState<string | undefined>()
  const [detectedReasoningParser, setDetectedReasoningParser] = useState<string | undefined>()
  const [detectedEnableAutoToolChoice, setDetectedEnableAutoToolChoice] = useState<boolean | undefined>()
  const [detectedIsTurboQuant, setDetectedIsTurboQuant] = useState<boolean>(false)
  const [detectedIsMultimodal, setDetectedIsMultimodal] = useState<boolean>(false)
  const [detectedForceTextOnly, setDetectedForceTextOnly] = useState<boolean>(false)
  const [detectedRuntimeModalities, setDetectedRuntimeModalities] = useState<string[] | undefined>(undefined)
  const [detectedMaxContext, setDetectedMaxContext] = useState<number | undefined>()
  const [detectedNativeMtp, setDetectedNativeMtp] = useState<any>(undefined)
  const [launching, setLaunching] = useState(false)
  const [launchError, setLaunchError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [scanLoading, setScanLoading] = useState(true)
  const [showDirManager, setShowDirManager] = useState(false)
  const [userDirs, setUserDirs] = useState<string[]>([])
  const [builtinDirs, setBuiltinDirs] = useState<string[]>([])
  const [dirError, setDirError] = useState<string | null>(null)
  const logEndRef = useRef<HTMLDivElement>(null)
  const launchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const launchSessionIdRef = useRef<string | null>(null)
  const mountedRef = useRef(true)
  const modelDefaultsRequestRef = useRef(0)

  // Remote session fields
  const [remoteUrl, setRemoteUrl] = useState('')
  const [remoteApiKey, setRemoteApiKey] = useState('')
  const [remoteModel, setRemoteModel] = useState('')
  const [remoteOrganization, setRemoteOrganization] = useState('')
  const [remoteConnecting, setRemoteConnecting] = useState(false)

  const applyModelDefaults = async (modelPath: string) => {
    const requestId = ++modelDefaultsRequestRef.current
    const [detected, gen] = await Promise.all([
      window.api.models.detectConfig(modelPath).catch(() => null),
      window.api.models.getGenerationDefaults(modelPath).catch(() => null),
    ]) as [any, any]
    if (!mountedRef.current || modelDefaultsRequestRef.current !== requestId) return
    setConfig(prev => {
      const next: SessionConfig = {
        ...prev,
        // Auto remains undefined; detection is displayed separately and launch
        // resolves the effective value. Materializing detection here made a
        // model-derived default indistinguishable from an explicit user choice.
        enableAutoToolChoice: undefined,
        toolCallParser: 'auto',
        reasoningParser: 'auto',
        dsv4PrefixCache: detected?.family === 'deepseek-v4' ? true : prev.dsv4PrefixCache,
        dsv4PoolQuant: detected?.family === 'deepseek-v4'
          ? (typeof detected?.dsv4PoolQuantDefault === 'boolean'
              ? detected.dsv4PoolQuantDefault
              : undefined)
          : prev.dsv4PoolQuant,
        enablePrefixCache: detected?.family === 'openpangu_v2' || detected?.family === 'deepseek-v4' ? true : prev.enablePrefixCache,
        // Paged RAM is retired product-wide. Detection remains informational;
        // it must not materialize a stale per-family capability into the form.
        usePagedCache: false,
        enableDiskCache: detected?.family === 'openpangu_v2',
        enableBlockDiskCache: detected?.family !== 'openpangu_v2',
        kvCacheQuantization: 'auto',
        pagedCacheBlockSize: detected?.family === 'deepseek-v4' ? DSV4_PAGED_CACHE_BLOCK_SIZE : prev.pagedCacheBlockSize,
        maxCacheBlocks: detected?.family === 'deepseek-v4' ? DSV4_MAX_CACHE_BLOCKS : prev.maxCacheBlocks,
        // Mirror the main-process family timeout bump (sessions.ts normalizer)
        // so the form displays the value launch will actually use.
        timeout: ['deepseek-v4', 'minimax_m3', 'openpangu_v2'].includes(detected?.family) &&
          (prev.timeout == null || prev.timeout === 300)
          ? 900
          : prev.timeout,
      }
      return applyBundleGenerationDefaultsToSessionConfig(next, gen)
    })
    if (detected?.cacheType) setDetectedCacheType(detected.cacheType)
    else setDetectedCacheType('kv')
    setDetectedUsePagedCache(detected?.usePagedCache)
    setDetectedCacheSubtype(detected?.cacheSubtype)
    setDetectedArchitectureHints(detected?.architectureHints)
    if (detected?.family && detected.family !== 'unknown') setDetectedFamily(detected.family)
    else setDetectedFamily(undefined)
    setDetectedToolParser(detected?.toolParser)
    setDetectedReasoningParser(detected?.reasoningParser)
    setDetectedEnableAutoToolChoice(detected?.enableAutoToolChoice)
    setDetectedIsTurboQuant(!!detected?.isTurboQuant)
    setDetectedIsMultimodal(!!detected?.isMultimodal)
    setDetectedForceTextOnly(!!detected?.forceTextOnly)
    setDetectedRuntimeModalities(Array.isArray(detected?.runtimeModalities) ? detected.runtimeModalities : undefined)
    setDetectedNativeMtp(detected?.nativeMtp)
    if (detected?.maxContextLength) setDetectedMaxContext(detected.maxContextLength)
  }

  // Auto-detect image model type on mount when initialModelPath is provided
  useEffect(() => {
    if (initialModelPath && !filterTypeProp) {
      window.api.models.detectTypes([initialModelPath]).then((types: any) => {
        if (types?.[initialModelPath] === 'image') setAutoDetectedType('image')
      }).catch(() => {})
    }
  }, [initialModelPath, filterTypeProp])

  // Cleanup on unmount. The effect body must re-arm mountedRef: under React 18
  // StrictMode (dev) effects run mount -> cleanup -> mount on the same instance,
  // so without this the cleanup leaves mountedRef=false forever and handleLaunch
  // silently bails right after sessions:create — the launch UI then stalls at
  // "Creating session..." even though the session row was created.
  useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
      if (launchTimerRef.current) clearTimeout(launchTimerRef.current)
    }
  }, [])

  const scanModels = async () => {
    setScanLoading(true)
    try {
      const scanned = await window.api.models.scan(filterTypeProp)
      setModels(scanned)
    } catch (err) {
      console.error('Failed to scan models:', err)
    } finally {
      setScanLoading(false)
    }
  }

  const loadDirectories = async () => {
    try {
      const result = await window.api.models.getDirectories(filterTypeProp)
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
        // The main process checks both persisted session ports and real OS
        // listeners with net.createServer. Renderer-only DB filtering can pick
        // a port owned by launchd or another app and fail at model load time.
        const port = await window.api.sessions.availablePort()
        setConfig(prev => ({ ...prev, port }))
      } catch (_) { }
    }
    assignPort()
  }, [])

  // Auto-detect config when arriving with a pre-selected model (e.g. from Developer Tools "Serve")
  useEffect(() => {
    if (!initialModelPath) return
    const detect = async () => {
      try {
        await applyModelDefaults(initialModelPath)
      } catch (_) { }
    }
    detect()
  }, [initialModelPath])

  const handleChange = <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const handleReset = async () => {
    const requestId = ++modelDefaultsRequestRef.current
    const base = { ...DEFAULT_CONFIG, port: config.port }
    // Re-run model detection to get proper defaults for this model
    if (selectedModel) {
      try {
        const [detected, gen] = await Promise.all([
          window.api.models.detectConfig(selectedModel).catch(() => null),
          window.api.models.getGenerationDefaults(selectedModel).catch(() => null),
        ]) as [any, any]
        if (!mountedRef.current || modelDefaultsRequestRef.current !== requestId) return
        if (detected && detected.family !== 'unknown') {
          base.enableAutoToolChoice = undefined
          if (['deepseek-v4', 'minimax_m3', 'openpangu_v2'].includes(detected.family)) {
            base.timeout = 900
          }
          if (detected.family === 'deepseek-v4') {
            base.dsv4PrefixCache = true
            base.dsv4PoolQuant = typeof detected.dsv4PoolQuantDefault === 'boolean'
              ? detected.dsv4PoolQuantDefault
              : undefined
            base.enablePrefixCache = true
            base.usePagedCache = false
            base.enableDiskCache = false
            base.enableBlockDiskCache = true
            base.kvCacheQuantization = 'auto'
            base.pagedCacheBlockSize = DSV4_PAGED_CACHE_BLOCK_SIZE
            base.maxCacheBlocks = DSV4_MAX_CACHE_BLOCKS
          } else if (usesExactTypedPromptDiskCache(detected.family)) {
            base.enablePrefixCache = true
            base.usePagedCache = false
            base.enableDiskCache = true
            base.enableBlockDiskCache = false
            base.noMemoryAwareCache = false
            base.kvCacheQuantization = 'auto'
          } else {
            base.usePagedCache = false
            base.enableDiskCache = false
            base.enableBlockDiskCache = true
          }
          setDetectedFamily(detected.family)
          setDetectedToolParser(detected.toolParser)
          setDetectedReasoningParser(detected.reasoningParser)
          setDetectedEnableAutoToolChoice(detected.enableAutoToolChoice)
          setDetectedCacheSubtype(detected.cacheSubtype)
          setDetectedArchitectureHints(detected.architectureHints)
          setDetectedIsTurboQuant(!!detected.isTurboQuant)
          setDetectedIsMultimodal(!!detected.isMultimodal)
          setDetectedForceTextOnly(!!detected.forceTextOnly)
          setDetectedNativeMtp(detected.nativeMtp)
        } else {
          setDetectedFamily(undefined)
          setDetectedToolParser(undefined)
          setDetectedReasoningParser(undefined)
          setDetectedEnableAutoToolChoice(undefined)
          setDetectedCacheSubtype(undefined)
          setDetectedArchitectureHints(undefined)
          setDetectedIsTurboQuant(false)
          setDetectedIsMultimodal(false)
          setDetectedForceTextOnly(false)
          setDetectedNativeMtp(undefined)
        }
        Object.assign(base, applyBundleGenerationDefaultsToSessionConfig(base, gen))
      } catch (_) {
        if (!mountedRef.current || modelDefaultsRequestRef.current !== requestId) return
        setDetectedFamily(undefined)
        setDetectedToolParser(undefined)
        setDetectedReasoningParser(undefined)
        setDetectedEnableAutoToolChoice(undefined)
        setDetectedCacheSubtype(undefined)
        setDetectedArchitectureHints(undefined)
        setDetectedIsTurboQuant(false)
        setDetectedIsMultimodal(false)
        setDetectedForceTextOnly(false)
        setDetectedNativeMtp(undefined)
      }
    }
    setConfig(base)
  }

  // Clean up log listener when launching state changes or component unmounts
  useEffect(() => {
    if (!launching) return

    const unsubLog = window.api.sessions.onLog((data: any) => {
      // Only show logs for the session being launched (not other running sessions)
      if (launchSessionIdRef.current && data.sessionId !== launchSessionIdRef.current) return
      setLogs(prev => [...prev.slice(-200), data.data])
    })

    // Also listen for errors during launch
    const unsubError = window.api.sessions.onError((data: any) => {
      if (launchSessionIdRef.current && data.sessionId !== launchSessionIdRef.current) return
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
      // Check vmlx-engine installation
      const installation = await window.api.engine.checkInstallation()
      if (!installation?.installed) {
        setLaunchError(
          t('sessions.create.engineNotFound') +
          '\n\n' +
          '  uv tool install vmlx\n' +
          '  pip3 install vmlx'
        )
        return
      }
    } catch (_) {
      // If the check itself fails, proceed anyway — startSession will catch it
    }

    setLaunching(true)
    setLogs([t('sessions.create.creatingSession')])

    try {
      // Set model type so buildArgs skips text-specific flags for image models.
      // Preserve explicit cache toggles exactly; shared launch policy resolves
      // architecture prerequisites without silently undoing a user L2 opt-out.
      const normalizedCacheConfig = config
      const launchConfig = (autoDetectedType || filterTypeProp) === 'image'
        ? { ...normalizedCacheConfig, modelType: 'image' as const }
        : normalizedCacheConfig
      const createResult = await window.api.sessions.create(selectedModel, launchConfig)
      if (!mountedRef.current) return
      if (!createResult.success) {
        throw new Error(createResult.error || t('sessions.context.createFailed'))
      }
      const session = createResult.session
      launchSessionIdRef.current = session.id
      setLogs(prev => [...prev, t('sessions.create.sessionCreated', { id: session.id }), t('main.loadProgress.startingServer')])

      const result = await window.api.sessions.start(session.id)
      if (!mountedRef.current) return
      if (result.success) {
        setLogs(prev => [...prev, t('sessions.create.serverReady')])
        launchTimerRef.current = setTimeout(() => onCreated(session.id), 500)
      } else {
        const errorMsg = result.error || t('chat.interface.toast.unknownError')
        setLogs(prev => [...prev, `\nERROR: ${errorMsg}`])
        setLaunchError(errorMsg)
        setLaunching(false)
      }
    } catch (error) {
      if (!mountedRef.current) return
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
      const remoteResult = await window.api.sessions.createRemote({
        remoteUrl: remoteUrl.trim(),
        remoteApiKey: remoteApiKey.trim() || undefined,
        remoteModel: remoteModel.trim(),
        remoteOrganization: remoteOrganization.trim() || undefined
      })
      if (!remoteResult.success) {
        throw new Error(remoteResult.error || t('layout.chatToolbar.remoteCreateFailed'))
      }
      const session = remoteResult.session

      const result = await window.api.sessions.start(session.id)
      if (result.success) {
        onCreated(session.id)
      } else {
        setLaunchError(result.error || t('sessions.create.connectRemoteFailed'))
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

  const handleAddManualPath = async (path: string) => {
    if (!path) return
    setDirError(null)
    await addDirectory(path)
  }

  const addDirectory = async (dirPath: string) => {
    const result = await window.api.models.addDirectory(dirPath, filterTypeProp)
    if (result.success) {
      await loadDirectories()
      // Rescan with the new directory
      await scanModels()
    } else {
      setDirError(result.error || t('sessions.create.addDirectoryFailed'))
    }
  }

  const handleRemoveDirectory = async (dirPath: string) => {
    await window.api.models.removeDirectory(dirPath, filterTypeProp)
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
            <button onClick={onBack} className="text-muted-foreground hover:text-foreground flex items-center gap-1">
              <ArrowLeft className="h-3.5 w-3.5" /> {t('common.back')}
            </button>
            <h1 className="text-2xl font-bold">{t('sessions.create.title')}</h1>
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
              {t('sessions.create.localModel')}
            </button>
            <button
              onClick={() => setSessionType('download')}
              className={`flex-1 px-3 py-1.5 text-sm rounded transition-colors ${sessionType === 'download'
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-accent text-muted-foreground'
                }`}
            >
              {t('sessions.create.download')}
            </button>
            <button
              onClick={() => setSessionType('remote')}
              className={`flex-1 px-3 py-1.5 text-sm rounded transition-colors ${sessionType === 'remote'
                ? 'bg-primary text-primary-foreground'
                : 'hover:bg-accent text-muted-foreground'
                }`}
            >
              {t('sessions.create.remoteEndpoint')}
            </button>
          </div>

          {/* Download Tab */}
          {sessionType === 'download' ? (
            <DownloadTab onDownloadComplete={() => { setSessionType('local'); scanModels() }} />
          ) : sessionType === 'remote' ? (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                {t('sessions.create.remoteHint')}
              </p>

              <div>
                <label className="text-sm font-medium block mb-1">{t('sessions.create.apiBaseUrl')}</label>
                <input
                  type="url"
                  placeholder={t('sessions.create.apiBaseUrlPlaceholder')}
                  value={remoteUrl}
                  onChange={(e) => setRemoteUrl(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  {t('sessions.create.apiBaseUrlHelp')}
                </p>
              </div>

              <div>
                <label className="text-sm font-medium block mb-1">{t('sessions.create.modelName')}</label>
                <input
                  type="text"
                  placeholder={t('sessions.create.modelNamePlaceholder')}
                  value={remoteModel}
                  onChange={(e) => setRemoteModel(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                />
              </div>

              <div>
                <label className="text-sm font-medium block mb-1">{t('sessions.create.apiKey')}</label>
                <input
                  type="password"
                  placeholder={t('sessions.create.apiKeyPlaceholder')}
                  value={remoteApiKey}
                  onChange={(e) => setRemoteApiKey(e.target.value)}
                  className="w-full px-3 py-2 bg-background border border-input rounded text-sm"
                />
              </div>

              <div>
                <label className="text-sm font-medium block mb-1">{t('sessions.create.organization')}</label>
                <input
                  type="text"
                  placeholder={t('sessions.create.organizationPlaceholder')}
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
                {remoteConnecting ? t('common.connecting') : t('common.connect')}
              </button>
            </div>
          ) : (
            <>
              <span className="text-sm text-muted-foreground block mb-4">{t('sessions.create.localHint')}</span>

              {/* Search + Actions Row */}
              <div className="flex items-center gap-2 mb-4">
                <input
                  type="text"
                  placeholder={t('sessions.create.filterPlaceholder')}
                  value={modelFilter}
                  onChange={(e) => setModelFilter(e.target.value)}
                  className="flex-1 px-3 py-2 bg-background border border-input rounded text-sm"
                />
                <button
                  onClick={scanModels}
                  disabled={scanLoading}
                  className="px-3 py-2 text-sm border border-border rounded hover:bg-accent disabled:opacity-50 whitespace-nowrap"
                >
                  {scanLoading ? t('sessions.create.scanning') : t('sessions.create.rescan')}
                </button>
                <button
                  onClick={() => setShowDirManager(!showDirManager)}
                  className={`px-3 py-2 text-sm border rounded whitespace-nowrap ${showDirManager ? 'border-primary bg-primary/10 text-primary' : 'border-border hover:bg-accent'
                    }`}
                >
                  {t('sessions.create.directories')}
                </button>
              </div>

              {/* Directory Manager Panel */}
              {showDirManager && (
                <div className="mb-4 p-4 bg-card border border-border rounded-lg">
                  <DirectoryManager
                    userDirs={userDirs}
                    builtinDirs={builtinDirs}
                    dirError={dirError}
                    onAdd={handleAddManualPath}
                    onRemove={handleRemoveDirectory}
                    onBrowse={handleBrowseDirectory}
                    onClearError={() => setDirError(null)}
                  />
                </div>
              )}

              {scanLoading ? (
                <p className="text-muted-foreground">{t('sessions.create.scanning')}</p>
              ) : filteredModels.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-muted-foreground mb-2">{t('sessions.create.noModels')}</p>
                  <p className="text-xs text-muted-foreground mb-4">
                    {t('sessions.create.noModelsHelp')}
                  </p>
                  <div className="mb-4 p-3 bg-card border border-border rounded-lg text-left">
                    <p className="text-xs text-muted-foreground mb-2">{t('sessions.create.downloadHint')}</p>
                    <div className="p-2 bg-muted rounded font-mono text-[11px] text-foreground select-all">
                      huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir ~/.cache/huggingface/hub/mlx-community/Llama-3.2-3B-Instruct-4bit
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-2">
                      {t('sessions.create.browseModels')}{' '}
                      <span className="text-primary select-all">huggingface.co/mlx-community</span>
                    </p>
                  </div>
                  <button
                    onClick={() => setShowDirManager(true)}
                    className="px-4 py-2 text-sm border border-border rounded hover:bg-accent"
                  >
                    {t('sessions.create.manageDirectories')}
                  </button>
                </div>
              ) : (
                <div className="space-y-1">
                  {filteredModels.map(model => (
                    <div key={model.path} className="group relative">
                    <button
                      onClick={async () => {
                        const selectionRequestId = ++modelDefaultsRequestRef.current
                        const selectionStillCurrent = () => (
                          mountedRef.current &&
                          modelDefaultsRequestRef.current === selectionRequestId
                        )
                        setSelectedModel(model.path)
                        // Pre-populate from existing session config if this model was launched before
                        try {
                          const sessions = await window.api.sessions.list()
                          if (!selectionStillCurrent()) return
                          const normalized = model.path.replace(/\/+$/, '')
                          const existing = sessions.find((s: any) =>
                            (s.modelPath || '').replace(/\/+$/, '') === normalized
                          )
                          if (existing?.config) {
                            try {
                              const stored = JSON.parse(existing.config)
                              // Preserve explicit tri-state tool policy. Legacy
                              // migration belongs in the versioned DB/session
                              // migration, never in this renderer reload path.
                              try {
                                const gen = await window.api.models.getGenerationDefaults(model.path) as any
                                if (!selectionStillCurrent()) return
                                Object.assign(
                                  stored,
                                  applyBundleGenerationDefaultsToSessionConfig(stored, gen),
                                )
                              } catch (_) { }
	                              setConfig(prev => ({ ...prev, ...stored, port: prev.port }))
                              // Still detect cache type for UI gating (Mamba vs KV, VLM)
                              try {
                                const det = await window.api.models.detectConfig(model.path) as any
                                if (!selectionStillCurrent()) return
                                setConfig(current => applyBundleDsv4PoolQuantToSessionConfig(current, det))
                                if (det?.cacheType) setDetectedCacheType(det.cacheType)
                                setDetectedUsePagedCache(det?.usePagedCache)
                                setDetectedCacheSubtype(det?.cacheSubtype)
                                setDetectedArchitectureHints(det?.architectureHints)
                                if (det?.family && det.family !== 'unknown') setDetectedFamily(det.family)
                                else setDetectedFamily(undefined)
                                setDetectedToolParser(det?.toolParser)
                                setDetectedReasoningParser(det?.reasoningParser)
                                setDetectedEnableAutoToolChoice(det?.enableAutoToolChoice)
                                setDetectedIsTurboQuant(!!det?.isTurboQuant)
                                setDetectedIsMultimodal(!!det?.isMultimodal)
                                setDetectedForceTextOnly(!!det?.forceTextOnly)
                                setDetectedNativeMtp(det?.nativeMtp)
                                if (det?.maxContextLength) setDetectedMaxContext(det.maxContextLength)
                              } catch (_) { }
                              // Auto-detect image model
                              try {
                                const types = await window.api.models.detectTypes([model.path])
                                if (!selectionStillCurrent()) return
                                if (types?.[model.path] === 'image') setAutoDetectedType('image')
                                else setAutoDetectedType('text')
                              } catch (_) { setAutoDetectedType(undefined) }
                              setStep(2)
                              return // skip auto-detect for config — existing config already has everything
                            } catch (_) { }
                          }
                        } catch (_) { }
	                        // Fallback: auto-detect model config for fresh sessions
	                        try {
	                          await applyModelDefaults(model.path)
	                        } catch (_) {
	                          // Auto-detect failed — user can configure manually
	                        }
                        // Auto-detect image model
                        try {
                          const types = await window.api.models.detectTypes([model.path])
                          if (types?.[model.path] === 'image') setAutoDetectedType('image')
                          else setAutoDetectedType('text')
                        } catch (_) { setAutoDetectedType(undefined) }
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
                    {/* vmlx#57: delete a local model from disk. Hover-reveal
                        trash icon in the top-right of each row. Confirmation
                        dialog quotes the path so the user sees exactly what
                        will be rm'd. */}
                    <button
                      onClick={async (e) => {
                        e.stopPropagation()
                        const label = model.size ? `${model.name} (${model.size})` : model.name
                        if (!confirm(
                          t('sessions.create.deleteModelConfirm', { label, path: model.path })
                        )) return
                        try {
                          const r = await window.api.models.deleteLocal(model.path)
                          if (!r?.success) {
                            alert(t('sessions.create.deleteFailed', { error: r?.error || t('chat.interface.toast.unknownError') }))
                            return
                          }
                          // If the selected model was just nuked, drop the selection
                          if (selectedModel === model.path) {
                            setSelectedModel('')
                            setStep(1)
                          }
                          // Rescan so the row disappears
                          try {
                            const scanned = await window.api.models.scan(filterTypeProp)
                            setModels(scanned)
                          } catch (_) { /* non-fatal */ }
                        } catch (err) {
                          alert(t('sessions.create.deleteFailed', { error: (err as Error).message }))
                        }
                      }}
                      className="absolute top-2 right-2 opacity-60 group-hover:opacity-100 transition-opacity p-1.5 rounded hover:bg-red-500/15 hover:text-red-400 text-muted-foreground text-xs"
                      title={t('sessions.create.deleteModelTitle', { name: model.name })}
                      aria-label={t('sessions.create.deleteModelTitle', { name: model.name })}
                    >
                      {/* Inline Trash2 SVG — avoid adding another lucide import to this file */}
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="14" height="14"
                        viewBox="0 0 24 24"
                        fill="none" stroke="currentColor"
                        strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                      >
                        <path d="M3 6h18" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                        <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        <line x1="10" x2="10" y1="11" y2="17" />
                        <line x1="14" x2="14" y1="11" y2="17" />
                      </svg>
                    </button>
                    </div>
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
    const launchSessionId = launchSessionIdRef.current
    const progress = launchSessionId ? loadProgress.get(launchSessionId) : undefined
    const residentLoad = formatResidentLoad(progress)
    return (
      <div className="p-6 overflow-auto h-full">
        <div className="max-w-3xl mx-auto">
          <h1 className="text-2xl font-bold mb-2">{t('sessions.create.loadingModel')}</h1>
          <p className="text-muted-foreground text-sm mb-4">
            {selectedModel.split('/').pop()}
          </p>

          {progress && (
            <div className="mb-4" data-vmlx-create-load-session-id={launchSessionId || ''}>
              <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className={`h-full bg-warning rounded-full transition-all duration-500 ease-out ${progress.indeterminate !== false ? 'animate-pulse' : ''}`}
                  style={{ width: progress.indeterminate === false ? `${progress.progress}%` : '100%' }}
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={progress.progress}
                />
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {progress.labelKey
                  ? t(progress.labelKey, { defaultValue: progress.label, ...(progress.labelParams || {}) })
                  : progress.label} {progress.indeterminate === false ? `(${progress.progress}%)` : ''}
              </p>
              {formatModelBytes(progress.modelBytes) && (
                <p className="text-xs text-muted-foreground/80 mt-0.5">
                  {t('sessions.card.modelFiles')} {formatModelBytes(progress.modelBytes)}
                </p>
              )}
              {residentLoad && (
                <p className="text-xs text-muted-foreground/80 mt-0.5">
                  {t('sessions.card.residentRam')} {residentLoad}
                </p>
              )}
            </div>
          )}

          <div className="bg-background/80 text-primary font-mono text-xs p-4 rounded-lg max-h-[60vh] overflow-auto border border-border">
            {logs.map((line, i) => (
              <div key={i} className={`whitespace-pre-wrap ${line.startsWith('ERROR') ? 'text-destructive font-bold' : ''}`}>{line}</div>
            ))}
            {!launchError && <div className="animate-pulse">▌</div>}
            <div ref={logEndRef} />
          </div>

          {launchError && (
            <div className="mt-4 p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
              <h3 className="text-sm font-bold text-destructive mb-2">{t('sessions.create.launchFailed')}</h3>
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
                  {t('sessions.create.backToConfig')}
                </button>
                <button
                  onClick={() => {
                    setLaunchError(null)
                    setLogs([])
                    handleLaunch()
                  }}
                  className="px-4 py-1.5 text-sm border border-border rounded hover:bg-accent"
                >
                  {t('common.retry')}
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
          <button onClick={() => setStep(1)} className="text-muted-foreground hover:text-foreground flex items-center gap-1">
            <ArrowLeft className="h-3.5 w-3.5" /> {t('common.back')}
          </button>
          <h1 className="text-2xl font-bold">{t('sessions.create.title')}</h1>
          <span className="text-sm text-muted-foreground">{t('sessions.create.step2')}</span>
        </div>

        {/* Pre-launch error banner */}
        {launchError && (
          <div className="mb-4 p-4 bg-destructive/10 border border-destructive/30 rounded-lg">
            <p className="text-sm text-destructive whitespace-pre-wrap">{launchError}</p>
          </div>
        )}

        {/* Selected model */}
        <div className="mb-4 p-3 bg-card border border-border rounded">
          <span className="text-xs text-muted-foreground">{t('sessions.create.modelLabel')}</span>
          <p className="font-medium text-sm truncate">{selectedModel}</p>
        </div>

        {/* Config Form — image models get simplified settings */}
        {(autoDetectedType || filterTypeProp) === 'image' ? (
          <div className="space-y-4 border border-border rounded p-4">
            <h3 className="text-sm font-medium">{t('sessions.create.imageServerSettings')}</h3>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">{t('sessions.create.imageMode')}</label>
                <select value={config.imageMode || 'generate'} onChange={e => handleChange('imageMode', e.target.value)} className="w-full px-2 py-1.5 bg-background border border-input rounded text-sm">
                  <option value="generate">{t('sessions.create.imageModeGen')}</option>
                  <option value="edit">{t('sessions.create.imageModeEdit')}</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">{t('sessions.create.quantization')}</label>
                <select value={config.imageQuantize ?? 0} onChange={e => handleChange('imageQuantize', parseInt(e.target.value))} className="w-full px-2 py-1.5 bg-background border border-input rounded text-sm">
                  <option value={0}>{t('sessions.create.quantFullPrecision')}</option>
                  <option value={4}>4-bit</option>
                  <option value={8}>8-bit</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">{t('sessions.create.host')}</label>
                <input type="text" value={config.host} onChange={e => handleChange('host', e.target.value)} className="w-full px-2 py-1.5 bg-background border border-input rounded text-sm" />
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">{t('sessions.create.port')}</label>
                <input type="number" value={config.port} onChange={e => handleChange('port', parseInt(e.target.value) || 8000)} className="w-full px-2 py-1.5 bg-background border border-input rounded text-sm" />
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">{t('sessions.create.apiKeyOptional')}</label>
                <input type="password" value={config.apiKey} onChange={e => handleChange('apiKey', e.target.value)} placeholder={t('sessions.config.apiKeyPlaceholder')} className="w-full px-2 py-1.5 bg-background border border-input rounded text-sm" />
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">{t('sessions.create.logLevel')}</label>
                <select value={config.logLevel || 'INFO'} onChange={e => handleChange('logLevel', e.target.value)} className="w-full px-2 py-1.5 bg-background border border-input rounded text-sm">
                  <option value="DEBUG">DEBUG</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                </select>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">
              {(config.imageMode === 'edit')
                ? t('sessions.create.imageServerEditHint')
                : t('sessions.create.imageServerGenHint')
              }
            </p>
          </div>
        ) : (
          <SessionConfigForm config={config} onChange={handleChange} onReset={handleReset} detectedCacheType={detectedCacheType} detectedUsePagedCache={detectedUsePagedCache} detectedCacheSubtype={detectedCacheSubtype} detectedFamily={detectedFamily} detectedArchitectureHints={detectedArchitectureHints} detectedToolParser={detectedToolParser} detectedReasoningParser={detectedReasoningParser} detectedEnableAutoToolChoice={detectedEnableAutoToolChoice} detectedIsTurboQuant={detectedIsTurboQuant} detectedIsMultimodal={detectedIsMultimodal} detectedForceTextOnly={detectedForceTextOnly} detectedRuntimeModalities={detectedRuntimeModalities} detectedMaxContext={detectedMaxContext} detectedNativeMtp={detectedNativeMtp} modelIdentity={selectedModel} />
        )}

        {/* Launch */}
        <div className="flex gap-3 mt-6 pb-6">
          <button onClick={() => setStep(1)} className="px-4 py-2 border border-border rounded hover:bg-accent">
            {t('common.back')}
          </button>
          <button
            onClick={handleLaunch}
            disabled={launching || !selectedModel}
            className="px-6 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {t('sessions.create.launchSession')}
          </button>
        </div>
      </div>
    </div>
  )
}
