import { useState, useEffect } from 'react'
import { SessionConfigForm, SessionConfig, DEFAULT_CONFIG, SliderField } from './SessionConfigForm'

interface Session {
  id: string
  modelPath: string
  modelName?: string
  host: string
  port: number
  pid?: number
  status: 'running' | 'stopped' | 'error' | 'loading'
  config: string
}

interface ServerSettingsDrawerProps {
  session: Session
  isRemote?: boolean
  onClose: () => void
  onSessionUpdate?: () => void
}

export function ServerSettingsDrawer({ session, isRemote, onClose, onSessionUpdate }: ServerSettingsDrawerProps) {
  const [config, setConfig] = useState<SessionConfig>(DEFAULT_CONFIG)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)
  const [detectedCacheType, setDetectedCacheType] = useState<string>('kv')
  const [detectedMaxContext, setDetectedMaxContext] = useState<number | undefined>()

  useEffect(() => {
    try {
      const stored = JSON.parse(session.config)
      // Always use DB columns as canonical source for host/port to prevent mismatch
      setConfig({ ...DEFAULT_CONFIG, ...stored, host: session.host, port: session.port })
    } catch {
      setConfig({ ...DEFAULT_CONFIG, host: session.host, port: session.port })
    }
    setDirty(false)
    setMessage(null)
    // Detect model cache type for feature gating
    if (session.modelPath) {
      window.api.models.detectConfig(session.modelPath)
        .then((det: any) => {
          if (det?.cacheType) setDetectedCacheType(det.cacheType)
          if (det?.maxContextLength) setDetectedMaxContext(det.maxContextLength)
        })
        .catch(() => { })
    }
  }, [session.id, session.config, session.host, session.port])

  // Listen for restart completion
  useEffect(() => {
    const unsubReady = window.api.sessions.onReady((data: any) => {
      if (data.sessionId === session.id) {
        setRestarting(false)
        setMessage({ type: 'success', text: 'Restarted with new settings.' })
        onSessionUpdate?.()
      }
    })
    const unsubError = window.api.sessions.onError((data: any) => {
      if (data.sessionId === session.id && restarting) {
        setRestarting(false)
        setMessage({ type: 'error', text: `Restart failed: ${data.error}` })
      }
    })
    return () => {
      unsubReady()
      unsubError()
    }
  }, [session.id, restarting])

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
        const isRunning = session.status === 'running' || session.status === 'loading'
        setMessage({
          type: 'success',
          text: isRemote ? 'Saved. Applies to next request.' : (isRunning ? 'Saved. Restart to apply.' : 'Settings saved.')
        })
        onSessionUpdate?.()
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
      const saveResult = await window.api.sessions.update(session.id, config)
      if (!saveResult.success) {
        setMessage({ type: 'error', text: saveResult.error || 'Failed to save' })
        setSaving(false)
        return
      }
      setDirty(false)
      setRestarting(true)
      setMessage({ type: 'success', text: 'Stopping...' })

      const stopResult = await window.api.sessions.stop(session.id)
      if (!stopResult.success) {
        setMessage({ type: 'error', text: `Failed to stop: ${stopResult.error}` })
        setRestarting(false)
        setSaving(false)
        return
      }

      await new Promise(r => setTimeout(r, 2500))
      setMessage({ type: 'success', text: 'Starting with new settings...' })
      const startResult = await window.api.sessions.start(session.id)
      if (!startResult.success) {
        setMessage({ type: 'error', text: `Failed to start: ${startResult.error}` })
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
    const base = { ...DEFAULT_CONFIG, host: config.host, port: config.port }
    // Re-run model detection to get proper defaults for this model
    if (session.modelPath) {
      try {
        const detected = await window.api.models.detectConfig(session.modelPath)
        if (detected && detected.family !== 'unknown') {
          base.enableAutoToolChoice = detected.enableAutoToolChoice
          base.usePagedCache = detected.usePagedCache
        }
      } catch (_) { }
    }
    setConfig(base)
    setDirty(true)
    setMessage(null)
  }

  const isRunning = session.status === 'running' || session.status === 'loading'

  return (
    <div className="w-96 h-full border-l border-border bg-card flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border flex-shrink-0">
        <span className="font-medium text-sm">{isRemote ? 'Connection Settings' : 'Server Settings'}</span>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-sm px-1">
          ✕
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
            Session is running. Save & Restart to apply changes.
          </div>
        )}

        {/* Config Form — remote sessions only show timeout */}
        {isRemote ? (
          <div className="space-y-3">
            <SliderField
              label="Request Timeout (seconds)"
              tooltip="Maximum time to wait for a response from the remote server before timing out. Increase this for slow models, long generations, or high-latency connections. Default 300s (5 minutes)."
              value={config.timeout}
              onChange={v => handleChange('timeout', v)}
              min={10}
              max={3600}
              step={10}
              defaultValue={DEFAULT_CONFIG.timeout}
              allowUnlimited
              unlimitedValue={0}
              unlimitedLabel="No limit"
            />
          </div>
        ) : (
          <SessionConfigForm config={config} onChange={handleChange} detectedCacheType={detectedCacheType} detectedMaxContext={detectedMaxContext} />
        )}
      </div>

      {/* Footer Actions */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3 border-t border-border flex-shrink-0">
        <button
          onClick={handleSave}
          disabled={!dirty || saving || restarting}
          className="flex-1 px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-40"
        >
          {saving && !restarting ? 'Saving...' : 'Save'}
        </button>
        {isRunning && !isRemote && (
          <button
            onClick={handleSaveAndRestart}
            disabled={saving || restarting}
            className="flex-1 px-3 py-1.5 text-sm bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-40"
          >
            {restarting ? 'Restarting...' : 'Save & Restart'}
          </button>
        )}
        <button
          onClick={handleReset}
          disabled={restarting}
          className="px-3 py-1.5 text-sm border border-border rounded hover:bg-accent disabled:opacity-40"
        >
          Reset
        </button>
      </div>
    </div>
  )
}
