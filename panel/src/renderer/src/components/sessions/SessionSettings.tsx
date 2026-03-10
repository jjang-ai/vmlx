import { useState, useEffect } from 'react'
import { SessionConfigForm, SessionConfig, DEFAULT_CONFIG } from './SessionConfigForm'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading'
  config: string
  createdAt: number
  updatedAt: number
}

interface SessionSettingsProps {
  sessionId: string
  onBack: () => void
}

/**
 * Build a preview of the CLI command from config.
 * This MUST mirror the logic in sessions.ts buildArgs() exactly.
 * When editing buildArgs(), update this function too (and vice versa).
 */
function buildCommandPreview(
  modelPath: string,
  config: SessionConfig,
  detected?: { toolParser?: string; reasoningParser?: string; isMultimodal?: boolean; usePagedCache?: boolean; enableAutoToolChoice?: boolean; cacheType?: string } | null
): string {
  const parts = ['vmlx-engine serve', modelPath]
  // Manual config takes priority over auto-detect for VLM mode
  const isVLM = config.isMultimodal ?? !!detected?.isMultimodal

  // Server settings
  parts.push('--host', config.host)
  parts.push('--port', config.port.toString())
  parts.push('--timeout', (config.timeout != null && config.timeout > 0 ? config.timeout : 86400).toString())

  if (config.apiKey) parts.push('# VLLM_API_KEY=*** (env var)')
  if (config.rateLimit && config.rateLimit > 0) parts.push('--rate-limit', config.rateLimit.toString())

  // Concurrent processing
  if (config.maxNumSeqs && config.maxNumSeqs > 0) parts.push('--max-num-seqs', config.maxNumSeqs.toString())
  if (config.prefillBatchSize && config.prefillBatchSize > 0) parts.push('--prefill-batch-size', config.prefillBatchSize.toString())
  if (config.completionBatchSize && config.completionBatchSize > 0) parts.push('--completion-batch-size', config.completionBatchSize.toString())

  if (isVLM) parts.push('--is-mllm')
  if (config.continuousBatching) parts.push('--continuous-batching')

  // Parser resolution: User explicit choice -> Detected config -> Fallback
  // (mirrors buildArgs: user choice wins over detection)
  const effectiveToolParser = config.toolCallParser === ''
    ? undefined
    : (config.toolCallParser && config.toolCallParser !== 'auto' ? config.toolCallParser
      : detected?.toolParser)
  const effectiveAutoTool = config.enableAutoToolChoice ?? detected?.enableAutoToolChoice
  const effectiveReasoningParser = config.reasoningParser === ''
    ? undefined
    : (config.reasoningParser && config.reasoningParser !== 'auto' ? config.reasoningParser
      : detected?.reasoningParser)

  // Prefix cache (mirrors buildArgs lines 1077-1114)
  const toolsNeedCache = !!(effectiveAutoTool && config.mcpConfig)
  const prefixCacheOff = config.enablePrefixCache === false && !toolsNeedCache

  if (prefixCacheOff) {
    parts.push('--disable-prefix-cache')
  } else {
    // Auto-enable continuous batching when prefix cache is on (required by vmlx-engine)
    if (!config.continuousBatching && !parts.includes('--continuous-batching')) {
      parts.push('--continuous-batching')
    }
    if (config.noMemoryAwareCache) {
      parts.push('--no-memory-aware-cache')
      if (config.prefixCacheSize && config.prefixCacheSize > 0) parts.push('--prefix-cache-size', config.prefixCacheSize.toString())
    } else {
      if (config.cacheMemoryMb && config.cacheMemoryMb > 0) parts.push('--cache-memory-mb', config.cacheMemoryMb.toString())
      if (config.cacheMemoryPercent && config.cacheMemoryPercent > 0) parts.push('--cache-memory-percent', (config.cacheMemoryPercent / 100).toString())
      if (config.cacheTtlMinutes && config.cacheTtlMinutes > 0 && !(config.usePagedCache ?? detected?.usePagedCache)) parts.push('--cache-ttl-minutes', config.cacheTtlMinutes.toString())
    }
  }

  // Paged cache — requires prefix cache ON (works for both LLM and VLM)
  if (!prefixCacheOff && (config.usePagedCache ?? detected?.usePagedCache)) {
    parts.push('--use-paged-cache')
    if (config.pagedCacheBlockSize && config.pagedCacheBlockSize > 0) parts.push('--paged-cache-block-size', config.pagedCacheBlockSize.toString())
    if (config.maxCacheBlocks && config.maxCacheBlocks > 0) parts.push('--max-cache-blocks', config.maxCacheBlocks.toString())
  }

  // KV cache quantization — requires prefix cache ON (works for both LLM and VLM)
  // Hybrid/Mamba models allowed — Python scheduler only quantizes KVCache layers
  if (!prefixCacheOff && config.kvCacheQuantization && config.kvCacheQuantization !== 'none') {
    parts.push('--kv-cache-quantization', config.kvCacheQuantization)
    if (config.kvCacheGroupSize && config.kvCacheGroupSize !== 64) {
      parts.push('--kv-cache-group-size', config.kvCacheGroupSize.toString())
    }
  }

  // Disk cache (mirrors buildArgs) — requires prefix cache ON, incompatible with paged cache
  if (!prefixCacheOff && config.enableDiskCache && !(config.usePagedCache ?? detected?.usePagedCache)) {
    parts.push('--enable-disk-cache')
    if (config.diskCacheDir) parts.push('--disk-cache-dir', config.diskCacheDir)
    if (config.diskCacheMaxGb != null && config.diskCacheMaxGb >= 0) parts.push('--disk-cache-max-gb', config.diskCacheMaxGb.toString())
  }

  // Block disk cache — requires prefix cache ON + paged cache ON (works for both LLM and VLM)
  if (!prefixCacheOff && (config.usePagedCache ?? detected?.usePagedCache) && config.enableBlockDiskCache) {
    parts.push('--enable-block-disk-cache')
    if (config.blockDiskCacheDir) parts.push('--block-disk-cache-dir', config.blockDiskCacheDir)
    if (config.blockDiskCacheMaxGb != null && config.blockDiskCacheMaxGb >= 0) parts.push('--block-disk-cache-max-gb', config.blockDiskCacheMaxGb.toString())
  }

  // Performance
  if (config.streamInterval && config.streamInterval > 0) parts.push('--stream-interval', config.streamInterval.toString())
  if (config.maxTokens && config.maxTokens > 0) {
    parts.push('--max-tokens', config.maxTokens.toString())
  } else {
    parts.push('--max-tokens', '1000000')
  }

  // Tool integration — mirrors buildArgs lines 1136-1147
  if (effectiveToolParser) {
    parts.push('--tool-call-parser', effectiveToolParser)
    if (effectiveAutoTool || config.enableAutoToolChoice === undefined) {
      parts.push('--enable-auto-tool-choice')
    }
  } else if (effectiveAutoTool) {
    parts.push('--enable-auto-tool-choice')
  }
  if (effectiveReasoningParser) parts.push('--reasoning-parser', effectiveReasoningParser)

  if (config.mcpConfig) parts.push('--mcp-config', config.mcpConfig)

  // Served model name
  if (config.servedModelName) parts.push('--served-model-name', config.servedModelName)

  // Speculative decoding
  if (config.speculativeModel) {
    parts.push('--speculative-model', config.speculativeModel)
    if (config.numDraftTokens && config.numDraftTokens !== 3) {
      parts.push('--num-draft-tokens', config.numDraftTokens.toString())
    }
  }

  // Generation defaults
  if (config.defaultTemperature && config.defaultTemperature > 0) {
    parts.push('--default-temperature', (config.defaultTemperature / 100).toFixed(2))
  }
  if (config.defaultTopP && config.defaultTopP > 0) {
    parts.push('--default-top-p', (config.defaultTopP / 100).toFixed(2))
  }

  // Embedding model
  if (config.embeddingModel) parts.push('--embedding-model', config.embeddingModel)

  if (config.additionalArgs && config.additionalArgs.trim()) parts.push(config.additionalArgs.trim())

  return parts.join(' \\\n  ')
}

export function SessionSettings({ sessionId, onBack }: SessionSettingsProps) {
  const [session, setSession] = useState<Session | null>(null)
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_CONFIG)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [detectedConfig, setDetectedConfig] = useState<{ toolParser?: string; reasoningParser?: string; cacheType?: string; isMultimodal?: boolean; usePagedCache?: boolean; enableAutoToolChoice?: boolean; family?: string } | null>(null)

  useEffect(() => {
    const load = async () => {
      const s = await window.api.sessions.get(sessionId)
      if (s) {
        setSession(s)
        // Parse stored config JSON, merge with defaults
        try {
          const stored = JSON.parse(s.config)
          setConfig({ ...DEFAULT_CONFIG, ...stored })
        } catch {
          setConfig(DEFAULT_CONFIG)
          setMessage({ type: 'error', text: 'Stored configuration was corrupted and has been reset to defaults. Save to persist.' })
          setDirty(true)
        }
        // Load auto-detected config for preview resolution
        try {
          const det = await window.api.models.detectConfig(s.modelPath)
          if (det && det.family !== 'unknown') setDetectedConfig(det)
        } catch (_) { }
      }
    }
    load()
  }, [sessionId])

  // Listen for session status changes
  useEffect(() => {
    const unsubStopped = window.api.sessions.onStopped((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'stopped', pid: undefined } : prev)
        setRestarting(false)
      }
    })
    const unsubReady = window.api.sessions.onReady((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'running' } : prev)
        setRestarting(false)
        setMessage({ type: 'success', text: 'Session restarted with new settings.' })
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === sessionId) {
        setSession(prev => prev ? { ...prev, status: 'error' } : prev)
        setRestarting(false)
        setMessage({ type: 'error', text: `Restart failed: ${data.error}` })
      }
    })
    return () => {
      unsubStopped()
      unsubReady()
      unsubError()
    }
  }, [sessionId])

  const handleChange = <K extends keyof SessionConfig>(key: K, value: SessionConfig[K]) => {
    setConfig(prev => ({ ...prev, [key]: value }))
    setDirty(true)
    setMessage(null)
  }

  const handleSave = async () => {
    setSaving(true)
    setMessage(null)
    try {
      const result = await window.api.sessions.update(sessionId, config)
      if (result.success) {
        setDirty(false)
        setMessage({
          type: 'success',
          text: result.restartRequired
            ? `Settings saved. Restart the session for changes to take effect (${result.changedKeys?.join(', ')}).`
            : 'Settings saved.'
        })
        // Refresh session data
        const s = await window.api.sessions.get(sessionId)
        if (s) setSession(s)
      } else {
        setMessage({ type: 'error', text: result.error || 'Failed to save' })
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
      // Save first
      const saveResult = await window.api.sessions.update(sessionId, config)
      if (!saveResult.success) {
        setMessage({ type: 'error', text: saveResult.error || 'Failed to save' })
        setSaving(false)
        return
      }
      setDirty(false)

      // Stop and wait for the process to actually exit
      setRestarting(true)
      setMessage({ type: 'success', text: 'Stopping session...' })
      const stopResult = await window.api.sessions.stop(sessionId)
      if (!stopResult.success) {
        setMessage({ type: 'error', text: `Failed to stop: ${stopResult.error}` })
        setRestarting(false)
        setSaving(false)
        return
      }

      // Wait for port to free (backend uses up to 10s SIGKILL timeout)
      await new Promise(r => setTimeout(r, 2500))

      // Start with new config
      setMessage({ type: 'success', text: 'Starting session with new settings...' })
      const startResult = await window.api.sessions.start(sessionId)
      if (!startResult.success) {
        setMessage({ type: 'error', text: `Failed to start: ${startResult.error}` })
        setRestarting(false)
      }
      // Success/failure will be handled by event listeners above

      // Refresh session data
      const s = await window.api.sessions.get(sessionId)
      if (s) setSession(s)
    } catch (e) {
      setMessage({ type: 'error', text: (e as Error).message })
      setRestarting(false)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    const base = { ...DEFAULT_CONFIG, host: config.host, port: config.port }
    // Re-run model detection to get proper defaults for this model
    if (session?.modelPath) {
      try {
        const detected = await window.api.models.detectConfig(session.modelPath)
        if (detected && detected.family !== 'unknown') {
          base.enableAutoToolChoice = detected.enableAutoToolChoice
          base.usePagedCache = detected.usePagedCache
          // VLM models: set isMultimodal flag (all cache features now supported)
          if (detected.isMultimodal) {
            base.isMultimodal = true
          }
        }
      } catch (_) { }
    }
    setConfig(base)
    setDirty(true)
    setMessage(null)
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-muted-foreground">Loading session...</p>
      </div>
    )
  }

  const shortName = session.modelName || session.modelPath.split('/').pop() || session.modelPath
  const isRunning = session.status === 'running' || session.status === 'loading'

  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3 mb-6">
          <button onClick={onBack} className="text-muted-foreground hover:text-foreground">
            ← Back
          </button>
          <h1 className="text-2xl font-bold">Session Settings</h1>
        </div>

        {/* Session Info */}
        <div className="mb-4 p-3 bg-card border border-border rounded">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-muted-foreground">Model</span>
              <p className="font-medium text-sm truncate" title={session.modelPath}>{shortName}</p>
              <p className="text-xs text-muted-foreground truncate">{session.modelPath}</p>
            </div>
            <div className="text-right text-sm">
              <span className={`inline-flex items-center gap-1.5 ${isRunning ? 'text-primary' : 'text-muted-foreground'}`}>
                <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-primary' : 'bg-muted-foreground'}`} />
                {restarting ? 'restarting...' : session.status}
              </span>
              <p className="text-xs text-muted-foreground">{config.host}:{config.port}</p>
            </div>
          </div>
        </div>

        {/* Running warning */}
        {isRunning && !restarting && (
          <div className="mb-4 p-3 bg-warning/10 border border-warning/30 rounded-lg text-sm text-warning">
            Session is running. Save changes and use "Save & Restart" to apply them.
          </div>
        )}

        {/* Status message */}
        {message && (
          <div className={`mb-4 p-3 rounded-lg text-sm ${message.type === 'success'
            ? 'bg-primary/10 border border-primary/30 text-primary'
            : 'bg-destructive/10 border border-destructive/30 text-destructive'
            }`}>
            {message.text}
          </div>
        )}

        {/* Config Form */}
        <SessionConfigForm config={config} onChange={handleChange} onReset={handleReset} detectedCacheType={detectedConfig?.cacheType} />

        {/* Command Preview */}
        <div className="mt-4">
          <button
            onClick={() => setShowPreview(!showPreview)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            {showPreview ? '▼' : '▶'} CLI Command Preview
          </button>
          {showPreview && (
            <pre className="mt-2 p-3 bg-background/80 text-primary text-xs font-mono rounded-lg overflow-x-auto whitespace-pre-wrap">
              {buildCommandPreview(session.modelPath, config, detectedConfig)}
            </pre>
          )}
        </div>

        {/* Actions */}
        <div className="flex flex-wrap gap-3 mt-6 pb-6">
          <button onClick={onBack} className="px-4 py-2 border border-border rounded hover:bg-accent">
            Back
          </button>
          <button
            onClick={handleSave}
            disabled={!dirty || saving || restarting}
            className="px-6 py-2 bg-primary text-primary-foreground rounded hover:bg-primary/90 font-medium disabled:opacity-40"
          >
            {saving && !restarting ? 'Saving...' : 'Save Settings'}
          </button>
          {isRunning && (
            <button
              onClick={handleSaveAndRestart}
              disabled={saving || restarting}
              className="px-6 py-2 bg-emerald-600 text-white rounded hover:bg-emerald-700 font-medium disabled:opacity-40"
            >
              {restarting ? 'Restarting...' : 'Save & Restart'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
