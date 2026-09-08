import { useState, useEffect, useRef } from 'react'
import { X } from 'lucide-react'
import {
  SessionConfigForm,
  SessionConfig,
  DEFAULT_CONFIG,
  SliderField,
  DSV4_MAX_CACHE_BLOCKS,
  DSV4_PAGED_CACHE_BLOCK_SIZE,
  commitActiveSettingsInput,
} from './SessionConfigForm'
import { useTranslation } from '../../i18n'
import {
  applyBundleDsv4PoolQuantToSessionConfig,
  applyBundleGenerationDefaultsToSessionConfig,
} from '../../../../shared/sessionGenerationDefaults'
import { usesExactTypedPromptDiskCache } from '../../../../shared/detectedFamilyNames'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading' | 'standby'
  config: string
}

interface ServerSettingsDrawerProps {
  session: Session
  isRemote?: boolean
  onClose: () => void
  onSessionUpdate?: () => void
}

export function ServerSettingsDrawer({ session, isRemote, onClose, onSessionUpdate }: ServerSettingsDrawerProps) {
  const { t } = useTranslation()
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_CONFIG)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [detectedCacheType, setDetectedCacheType] = useState<string>('kv')
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
  const [singleModelMode, setSingleModelMode] = useState(false)
  const restartingRef = useRef(false)
  const sessionIdRef = useRef(session.id)
  const resetRequestRef = useRef(0)
  restartingRef.current = restarting
  sessionIdRef.current = session.id

  useEffect(() => {
    let active = true
    resetRequestRef.current += 1
    // A replacement session must never inherit the previous model's async
    // detection while the new lookup is in flight. In particular, stale
    // Laguna architecture hints would incorrectly disable JIT controls for an
    // unrelated model.
    setDetectedCacheType('kv')
    setDetectedUsePagedCache(undefined)
    setDetectedCacheSubtype(undefined)
    setDetectedFamily(undefined)
    setDetectedArchitectureHints(undefined)
    setDetectedToolParser(undefined)
    setDetectedReasoningParser(undefined)
    setDetectedEnableAutoToolChoice(undefined)
    setDetectedIsTurboQuant(false)
    setDetectedIsMultimodal(false)
    setDetectedForceTextOnly(false)
    setDetectedMaxContext(undefined)
    setDetectedNativeMtp(undefined)
    const load = async () => {
      let base: SessionConfig
      try {
        const stored = JSON.parse(session.config)
        // Always use DB columns as canonical source for host/port to prevent mismatch
        base = { ...DEFAULT_CONFIG, ...stored, host: session.host, port: session.port }
      } catch {
        base = { ...DEFAULT_CONFIG, host: session.host, port: session.port }
      }

      // Install the persisted session synchronously before the bundle lookup.
      // The later lookup only owns default* metadata, so merging it into the
      // latest state cannot erase a user edit made while the request is in
      // flight.
      setConfig(base)
      setDirty(false)
      setMessage(null)
      const generationDefaults = session.modelPath
        ? await window.api.models.getGenerationDefaults(session.modelPath).catch(() => null)
        : null
      if (!active) return
      setConfig(current => applyBundleGenerationDefaultsToSessionConfig(current, generationDefaults))

      // Detect model cache type for feature gating.
      if (!session.modelPath) return
      try {
        const det: any = await window.api.models.detectConfig(session.modelPath)
        if (!active) return
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
        setDetectedRuntimeModalities(Array.isArray(det?.runtimeModalities) ? det.runtimeModalities : undefined)
        setDetectedNativeMtp(det?.nativeMtp)
        if (det?.maxContextLength) setDetectedMaxContext(det.maxContextLength)
      } catch (err) {
        if (active) {
          console.error('Failed to detect model config:', err)
          setDetectedFamily(undefined)
          setDetectedToolParser(undefined)
          setDetectedReasoningParser(undefined)
          setDetectedEnableAutoToolChoice(undefined)
          setDetectedCacheSubtype(undefined)
          setDetectedArchitectureHints(undefined)
        }
      }
    }
    load()
    return () => { active = false }
  }, [session.id, session.config, session.host, session.port, session.modelPath])

  useEffect(() => {
    window.api.gateway?.getStatus?.()
      .then((status: any) => setSingleModelMode(!!status?.singleModelMode))
      .catch(() => {})
  }, [])

  useEffect(() => {
    const unsubscribe = window.api.gateway?.onSingleModelModeChanged?.((data: { singleModelMode: boolean }) => {
      setSingleModelMode(!!data?.singleModelMode)
    })
    return () => { unsubscribe?.() }
  }, [])

  // Listen for restart completion
  useEffect(() => {
    const unsubReady = window.api.sessions.onReady((data: any) => {
      if (data.sessionId === session.id) {
        setRestarting(false)
        setMessage({ type: 'success', text: t('sessions.drawer.restartedToast') })
        onSessionUpdate?.()
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === session.id && restartingRef.current) {
        setRestarting(false)
        setMessage({ type: 'error', text: t('sessions.settings.restartFailed', { error: data.error }) })
      }
    })
    return () => {
      unsubReady()
      unsubError()
    }
  }, [session.id])

  const handleChange = <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    setDirty(true)
    setMessage(null)
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const result = await window.api.sessions.update(session.id, config)
      if (result.success) {
        setDirty(false)
        setMessage({
          type: 'success',
          text: isRemote ? t('sessions.drawer.savedRemote') : (
            result.restartRequired ? t('sessions.drawer.savedRestartToApply', { keys: result.changedKeys?.join(', ') ?? '' }) : t('sessions.settings.savedToast')
          )
        })
        onSessionUpdate?.()
      } else {
        setMessage({ type: 'error', text: result.error || t('sessions.settings.saveFailed') })
      }
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
    } finally {
      setSaving(false)
    }
  }

  const handleSaveAndRestart = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const saveResult = await window.api.sessions.update(session.id, config)
      if (!saveResult.success) {
        setMessage({ type: 'error', text: saveResult.error || t('sessions.settings.saveFailed') })
        setSaving(false)
        return
      }
      setDirty(false)
      setRestarting(true)
      setMessage({ type: 'success', text: t('sessions.drawer.stopping') })

      // ONE atomic main-process operation. Orchestrating stop + start as two
      // IPC calls let an explicit user Stop land between them; the queued
      // start then captured the post-Stop epoch and spawned an engine the UI
      // no longer tracked. restartSession carries the restart generation so a
      // later Stop always wins.
      const restartResult = await window.api.sessions.restart(session.id)
      if (!restartResult.success) {
        setMessage({ type: 'error', text: t('sessions.settings.restartFailed', { error: restartResult.error ?? '' }) })
        setRestarting(false)
      }
      onSessionUpdate?.()
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
      setRestarting(false)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    const resetSessionId = session.id
    const resetRequest = ++resetRequestRef.current
    const resetStillCurrent = () => (
      sessionIdRef.current === resetSessionId
      && resetRequestRef.current === resetRequest
    )
    const base = { ...DEFAULT_CONFIG, host: config.host, port: config.port }
    // Re-run model detection to get proper defaults for this model
    if (session.modelPath) {
      try {
        const detected = await window.api.models.detectConfig(session.modelPath)
        if (!resetStillCurrent()) return
        if (detected && detected.family !== 'unknown') {
          base.enableAutoToolChoice = undefined
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
        }
      } catch (_) {
        if (!resetStillCurrent()) return
        setDetectedFamily(undefined)
        setDetectedToolParser(undefined)
        setDetectedReasoningParser(undefined)
        setDetectedEnableAutoToolChoice(undefined)
        setDetectedCacheSubtype(undefined)
        setDetectedArchitectureHints(undefined)
        setDetectedIsTurboQuant(false)
        setDetectedIsMultimodal(false)
        setDetectedForceTextOnly(false)
      }
      // Bundle generation defaults are independent of family/cache detection.
      // Keep hydrating them even if detectConfig cannot classify the model.
      const generationDefaults = await window.api.models.getGenerationDefaults(session.modelPath).catch(() => null)
      if (!resetStillCurrent()) return
      Object.assign(base, applyBundleGenerationDefaultsToSessionConfig(base, generationDefaults))
    }
    if (!resetStillCurrent()) return
    setConfig(base)
    setDirty(true)
    setMessage(null)
  }

  const handleGatewaySingleModelModeToggle = async () => {
    const next = !singleModelMode
    setSingleModelMode(next)
    try {
      const status = await window.api.gateway?.setSingleModelMode?.(next)
      if (typeof status?.singleModelMode === 'boolean') {
        setSingleModelMode(status.singleModelMode)
      }
    } catch (e) {
      setSingleModelMode(!next)
      setMessage({ type: 'error', text: (e as Error).message })
    }
  }

  const isRunning = session.status === 'running' || session.status === 'loading'

  // Keyboard parity with the modal: Escape closes the drawer while it is
  // mounted, wherever focus sits. After a keyboard save the Save button
  // disables itself and focus falls to the body, so a handler scoped to the
  // drawer element never saw the key (live: drawer stayed open). A native
  // <select> consumes Escape for its own list first (defaultPrevented).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !e.defaultPrevented) onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      data-vmlx-surface="server-settings"
      className="w-full max-w-96 h-full border-l border-border bg-card flex flex-col overflow-hidden flex-shrink-0"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border flex-shrink-0">
        <span className="font-medium text-sm">{isRemote ? t('sessions.view.connectionTitle') : t('sessions.view.serverSettingsTitle')}</span>
        <button onClick={onClose} aria-label={t('common.close')} title={t('common.close')} className="text-muted-foreground hover:text-foreground text-sm px-1">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {/* Status message */}
        {message && (
          <div className={`p-2 rounded text-xs ${message.type === 'success'
              ? 'bg-primary/10 border border-primary/30 text-primary'
              : 'bg-destructive/10 border border-destructive/30 text-destructive'
            }`}>
            {message.text}
          </div>
        )}

        {isRunning && !restarting && !isRemote && (
          <div className="p-2 bg-warning/10 border border-warning/30 rounded text-xs text-warning">
            {t('sessions.settings.runningWarning')}
          </div>
        )}

        {!isRemote && (
          <div className="p-3 rounded border border-border bg-background/60 space-y-2">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-medium">{t('main.tray.singleModelMode')}</div>
                <div className="text-xs text-muted-foreground">
                  {singleModelMode ? t('api.singleModelModeOn') : t('api.singleModelModeOff')}
                </div>
              </div>
              <button
                data-vmlx-control="gateway-single-model-mode"
                aria-pressed={singleModelMode}
                onClick={handleGatewaySingleModelModeToggle}
                aria-label={t('main.tray.singleModelMode')}
                className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors flex-shrink-0 ${singleModelMode ? 'bg-primary' : 'bg-muted'}`}
              >
                <span
                  className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${singleModelMode ? 'translate-x-[18px]' : 'translate-x-0.5'}`}
                />
              </button>
            </div>
          </div>
        )}

        {/* Config Form — remote sessions only show timeout */}
        {isRemote ? (
          <div className="space-y-3">
            <SliderField
              label={t('sessions.config.timeout')}
              tooltip={t('sessions.drawer.timeoutTooltip')}
              value={config.timeout}
              onChange={v => handleChange('timeout', v)}
              min={10}
              max={3600}
              step={10}
              defaultValue={DEFAULT_CONFIG.timeout}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel={t('sessions.config.timeoutNoLimit')}
            />
          </div>
        ) : (
          <SessionConfigForm config={config} onChange={handleChange} detectedCacheType={detectedCacheType} detectedUsePagedCache={detectedUsePagedCache} detectedCacheSubtype={detectedCacheSubtype} detectedFamily={detectedFamily} detectedArchitectureHints={detectedArchitectureHints} detectedToolParser={detectedToolParser} detectedReasoningParser={detectedReasoningParser} detectedEnableAutoToolChoice={detectedEnableAutoToolChoice} detectedIsTurboQuant={detectedIsTurboQuant} detectedIsMultimodal={detectedIsMultimodal} detectedForceTextOnly={detectedForceTextOnly} detectedRuntimeModalities={detectedRuntimeModalities} detectedMaxContext={detectedMaxContext} detectedNativeMtp={detectedNativeMtp} modelType={(() => { try { return JSON.parse(session.config || '{}').modelType } catch { return undefined } })()} imageMode={(() => { try { return JSON.parse(session.config || '{}').imageMode } catch { return undefined } })()} sessionId={session.id} modelIdentity={`${session.modelName || ''} ${session.modelPath}`} />
        )}
      </div>

      {/* Footer Actions */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3 border-t border-border flex-shrink-0">
        {/* data-testid on all three: the CDP proof harness text-matches
            buttons, and "Save & Restart" also appears in an info div — a
            text mclick hit the div and reported success with no handler
            fired (#191 lead 6). Stable selectors end that trap. */}
        <button
          data-testid="server-settings-save"
          data-vmlx-control="server-settings-save"
          onPointerDown={commitActiveSettingsInput}
          onClick={handleSave}
          disabled={!dirty || saving || restarting}
          className="flex-1 px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40"
        >
          {saving && !restarting
            ? t('common.saving')
            : t(isRunning ? 'sessions.settings.saveForNextRestart' : 'common.save')}
        </button>
        {isRunning && !isRemote && (
          <button
            data-testid="server-settings-save-restart"
            data-vmlx-control="server-settings-save-restart"
            onPointerDown={commitActiveSettingsInput}
            onClick={handleSaveAndRestart}
            disabled={saving || restarting}
            className="flex-1 px-3 py-1.5 text-sm bg-success text-success-foreground rounded hover:bg-success/90 disabled:opacity-40"
          >
            {restarting ? t('sessions.settings.restarting') : t('sessions.settings.saveAndRestart')}
          </button>
        )}
        <button
          data-testid="server-settings-reset"
          data-vmlx-control="server-settings-reset"
          onClick={handleReset}
          disabled={restarting}
          className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent disabled:opacity-40"
        >
          {t('common.reset')}
        </button>
      </div>
    </div>
  )
}
