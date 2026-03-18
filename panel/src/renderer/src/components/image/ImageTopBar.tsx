import { useState, useRef, useEffect } from 'react'
import { Settings, Square, RefreshCw, PanelLeftOpen, ChevronDown, Download, Loader2, ScrollText } from 'lucide-react'

function formatElapsed(secs: number): string {
  if (secs < 60) return `${secs}s`
  const m = Math.floor(secs / 60)
  const s = secs % 60
  return `${m}m ${s}s`
}

import { IMAGE_MODELS } from '../../../../shared/imageModels'

type ServerStatus = 'stopped' | 'starting' | 'running' | 'error'

interface ImageModel {
  id: string
  name: string
  category: 'generate' | 'edit'
  quantizeOptions: number[]
}

// Derive AVAILABLE_MODELS from the shared registry
const AVAILABLE_MODELS: ImageModel[] = IMAGE_MODELS.map(m => ({
  id: m.id,
  name: m.name,
  category: m.category,
  quantizeOptions: m.quantizeOptions,
}))

interface ImageTopBarProps {
  model: string | null
  quantize: number
  status: ServerStatus
  port: number | null
  generating?: boolean
  mode: 'generate' | 'edit'
  onSettings: () => void
  onLogs: () => void
  onStop: () => void
  onChangeModel: () => void
  onSelectModel: (modelId: string, quantize: number | undefined, category: 'generate' | 'edit') => void
  sidebarCollapsed: boolean
  onToggleSidebar: () => void
}

export function ImageTopBar({
  model,
  quantize,
  status,
  port,
  mode,
  generating,
  onSettings,
  onLogs,
  onStop,
  onChangeModel,
  onSelectModel,
  sidebarCollapsed,
  onToggleSidebar
}: ImageTopBarProps) {
  const quantizeLabel = quantize === 0 ? 'Full' : `${quantize}-bit`
  const [showPicker, setShowPicker] = useState(false)
  const [loadingElapsed, setLoadingElapsed] = useState(0)

  // Elapsed time counter when model is loading
  useEffect(() => {
    if (status !== 'starting') {
      setLoadingElapsed(0)
      return
    }
    const interval = setInterval(() => {
      setLoadingElapsed(prev => prev + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [status])
  const [availability, setAvailability] = useState<Record<string, boolean>>({})
  const [checkingAvail, setCheckingAvail] = useState(false)
  const pickerRef = useRef<HTMLDivElement>(null)

  // Close dropdown on click outside
  useEffect(() => {
    if (!showPicker) return
    const handleClick = (e: MouseEvent) => {
      if (pickerRef.current && !pickerRef.current.contains(e.target as Node)) {
        setShowPicker(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [showPicker])

  // Check model availability when dropdown opens — check ALL quantize options per model
  useEffect(() => {
    if (!showPicker) return
    setCheckingAvail(true)
    const checks = AVAILABLE_MODELS.map(async (m) => {
      // Check all quantize options, report available if ANY is downloaded
      const results = await Promise.all(
        m.quantizeOptions.map(async (q) => {
          try {
            const result = await window.api.models.checkImageModel(m.id, q)
            return result.available
          } catch { return false }
        })
      )
      return { id: m.id, available: results.some(Boolean) }
    })
    Promise.all(checks).then(results => {
      const avail: Record<string, boolean> = {}
      results.forEach(r => { avail[r.id] = r.available })
      setAvailability(avail)
      setCheckingAvail(false)
    })
  }, [showPicker])

  const displayName = model ? (model.includes('/') ? model.split('/').pop() : model) : 'No model selected'

  const genModels = AVAILABLE_MODELS.filter(m => m.category === 'generate')
  const editModels = AVAILABLE_MODELS.filter(m => m.category === 'edit')

  const isActive = (id: string) => model === id
  const isDownloaded = (id: string) => availability[id] === true

  return (
    <div className="h-11 border-b border-border flex items-center justify-between px-3 bg-background flex-shrink-0">
      <div className="flex items-center gap-2">
        {sidebarCollapsed && (
          <button
            onClick={onToggleSidebar}
            className="p-1 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors mr-1"
            title="Show history"
          >
            <PanelLeftOpen className="h-4 w-4" />
          </button>
        )}

        {/* Model selector dropdown */}
        <div className="relative" ref={pickerRef}>
          <button
            onClick={() => !generating && setShowPicker(!showPicker)}
            disabled={generating}
            className={`flex items-center gap-1 text-sm font-medium transition-colors ${generating ? 'opacity-50 cursor-not-allowed' : 'hover:text-primary'}`}
            title={generating ? 'Cannot switch while generating' : 'Switch model'}
          >
            <span className="truncate max-w-[200px]">{displayName}</span>
            <ChevronDown className={`h-3 w-3 text-muted-foreground transition-transform flex-shrink-0 ${showPicker ? 'rotate-180' : ''}`} />
          </button>

          {showPicker && (
            <div className="absolute top-full left-0 mt-1 w-72 bg-card border border-border rounded-lg shadow-lg z-50 py-1 max-h-96 overflow-auto">
              {/* Generation models */}
              <div className="px-3 py-1.5">
                <span className="text-[10px] font-semibold text-blue-400 uppercase tracking-wider">Image Generation</span>
              </div>
              {genModels.map(m => (
                <ModelRow
                  key={m.id}
                  model={m}
                  active={isActive(m.id)}
                  downloaded={isDownloaded(m.id)}
                  checking={checkingAvail}
                  running={isActive(m.id) && status === 'running'}
                  onSelect={() => { setShowPicker(false); onSelectModel(m.id, undefined, m.category) }}
                />
              ))}

              <div className="border-t border-border my-1" />

              {/* Edit models */}
              <div className="px-3 py-1.5">
                <span className="text-[10px] font-semibold text-violet-400 uppercase tracking-wider">Image Editing</span>
              </div>
              {editModels.map(m => (
                <ModelRow
                  key={m.id}
                  model={m}
                  active={isActive(m.id)}
                  downloaded={isDownloaded(m.id)}
                  checking={checkingAvail}
                  running={isActive(m.id) && status === 'running'}
                  onSelect={() => { setShowPicker(false); onSelectModel(m.id, undefined, m.category) }}
                />
              ))}

              <div className="border-t border-border my-1" />

              {/* Custom / Browse */}
              <button
                onClick={() => { setShowPicker(false); onChangeModel() }}
                className="w-full text-left px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
              >
                Browse custom model...
              </button>
            </div>
          )}
        </div>

        {/* Mode badge */}
        {model && mode === 'edit' && (
          <span className="text-[10px] px-1.5 py-0.5 bg-violet-500/15 text-violet-400 rounded-full font-medium">Image Edit</span>
        )}
        {model && mode === 'generate' && (
          <span className="text-[10px] px-1.5 py-0.5 bg-blue-500/15 text-blue-400 rounded-full font-medium">Image Gen</span>
        )}
        {model && (
          <span className="text-[10px] px-1.5 py-0.5 bg-muted rounded-full text-muted-foreground">
            {quantizeLabel}
          </span>
        )}

        {/* Status indicator */}
        <div className="flex items-center gap-1.5 ml-2">
          <div className={`w-2 h-2 rounded-full ${
            status === 'running' ? 'bg-green-500' :
            status === 'starting' ? 'bg-yellow-500 animate-pulse' :
            status === 'error' ? 'bg-red-500' :
            'bg-gray-400'
          }`} />
          <span className="text-xs text-muted-foreground">
            {status === 'running' && port ? `Running on :${port}` :
             status === 'starting' ? `Loading model... ${formatElapsed(loadingElapsed)}` :
             status === 'error' ? 'Error' :
             'Stopped'}
          </span>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-1">
        <button
          onClick={onLogs}
          className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="Logs"
        >
          <ScrollText className="h-4 w-4" />
        </button>
        <button
          onClick={onSettings}
          className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
          title="Settings"
        >
          <Settings className="h-4 w-4" />
        </button>
        {(status === 'running' || status === 'starting') && (
          <button
            onClick={onStop}
            className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-destructive transition-colors"
            title={status === 'starting' ? 'Cancel loading' : 'Stop server'}
          >
            <Square className="h-4 w-4" />
          </button>
        )}
        {status === 'error' && (
          <button
            onClick={onChangeModel}
            className="p-1.5 rounded hover:bg-accent text-muted-foreground hover:text-foreground transition-colors"
            title="Retry"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        )}
      </div>
    </div>
  )
}

function ModelRow({ model, active, downloaded, checking, running, onSelect }: {
  model: ImageModel
  active: boolean
  downloaded: boolean
  checking: boolean
  running: boolean
  onSelect: () => void
}) {
  const isEdit = model.category === 'edit'
  const dotColor = isEdit ? 'bg-violet-500' : 'bg-blue-500'

  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-3 py-2 text-sm hover:bg-accent transition-colors flex items-center gap-2 ${
        active ? (isEdit ? 'bg-violet-500/5' : 'bg-blue-500/5') : ''
      }`}
    >
      {/* Status dot */}
      {running ? (
        <span className="w-2 h-2 rounded-full bg-green-500 flex-shrink-0" />
      ) : (
        <span className={`w-2 h-2 rounded-full ${dotColor} opacity-30 flex-shrink-0`} />
      )}

      {/* Model name */}
      <span className={active ? (isEdit ? 'text-violet-400 font-medium' : 'text-blue-400 font-medium') : 'text-foreground'}>
        {model.name}
      </span>

      {/* Right side: status badges */}
      <div className="ml-auto flex items-center gap-1.5">
        {running && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-green-500/15 text-green-500">running</span>
        )}
        {!running && active && (
          <span className="text-[9px] px-1 py-0.5 rounded bg-yellow-500/15 text-yellow-500">loading</span>
        )}
        {checking ? (
          <Loader2 className="h-3 w-3 text-muted-foreground animate-spin" />
        ) : downloaded ? (
          <span className="text-[9px] text-green-500">ready</span>
        ) : (
          <span className="text-[9px] text-muted-foreground flex items-center gap-0.5">
            <Download className="h-2.5 w-2.5" />
            download
          </span>
        )}
      </div>
    </button>
  )
}
