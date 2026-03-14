import { useState, useEffect, useRef } from 'react'
import { Search, Stethoscope, ArrowRightLeft, Eye, Play, HardDrive, RefreshCw } from 'lucide-react'

interface LocalModel {
  id: string
  name: string
  path: string
  size?: string
  format?: 'mlx' | 'gguf' | 'unknown'
  quantization?: string
}

interface ToolsDashboardProps {
  onInspect: (modelPath: string) => void
  onDiagnose: (modelPath: string) => void
  onConvert: (modelPath: string) => void
  onServe: (modelPath: string) => void
}

export function ToolsDashboard({ onInspect, onDiagnose, onConvert, onServe }: ToolsDashboardProps) {
  const [models, setModels] = useState<LocalModel[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  const loadModels = async () => {
    setLoading(true)
    try {
      const scanned = await window.api.models.scan()
      if (mountedRef.current) setModels(scanned)
    } catch (err) {
      console.error('Failed to scan models:', err)
    } finally {
      if (mountedRef.current) setLoading(false)
    }
  }

  useEffect(() => { loadModels() }, [])

  const mlxModels = models.filter(m =>
    m.format === 'mlx' && (
      m.name.toLowerCase().includes(filter.toLowerCase()) ||
      m.path.toLowerCase().includes(filter.toLowerCase())
    )
  )

  return (
    <div className="p-6 overflow-auto h-full">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold">Developer Tools</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Inspect, diagnose, and convert models
            </p>
          </div>
          <button
            onClick={loadModels}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground border border-border rounded-md hover:bg-accent transition-colors"
          >
            <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search models..."
            value={filter}
            onChange={e => setFilter(e.target.value)}
            className="w-full pl-9 pr-4 py-2 bg-background border border-input rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>

        {/* Quick actions */}
        <div className="grid grid-cols-3 gap-3">
          <button
            onClick={() => onConvert('')}
            className="flex items-center gap-3 p-4 border border-border rounded-lg hover:bg-accent transition-colors text-left"
          >
            <div className="w-9 h-9 rounded-lg bg-blue-500/10 flex items-center justify-center flex-shrink-0">
              <ArrowRightLeft className="h-4 w-4 text-blue-500" />
            </div>
            <div>
              <p className="text-sm font-medium">Convert Model</p>
              <p className="text-xs text-muted-foreground">HF to quantized MLX</p>
            </div>
          </button>
          <button
            onClick={() => onDiagnose('')}
            className="flex items-center gap-3 p-4 border border-border rounded-lg hover:bg-accent transition-colors text-left"
          >
            <div className="w-9 h-9 rounded-lg bg-green-500/10 flex items-center justify-center flex-shrink-0">
              <Stethoscope className="h-4 w-4 text-green-500" />
            </div>
            <div>
              <p className="text-sm font-medium">Diagnose Model</p>
              <p className="text-xs text-muted-foreground">Run health checks</p>
            </div>
          </button>
          <button
            onClick={() => onInspect('')}
            className="flex items-center gap-3 p-4 border border-border rounded-lg hover:bg-accent transition-colors text-left"
          >
            <div className="w-9 h-9 rounded-lg bg-purple-500/10 flex items-center justify-center flex-shrink-0">
              <Eye className="h-4 w-4 text-purple-500" />
            </div>
            <div>
              <p className="text-sm font-medium">Inspect Model</p>
              <p className="text-xs text-muted-foreground">View metadata</p>
            </div>
          </button>
        </div>

        {/* Loading */}
        {loading && (
          <div className="text-center py-12 text-muted-foreground text-sm">
            Scanning for models...
          </div>
        )}

        {/* MLX Models */}
        {!loading && mlxModels.length > 0 && (
          <div className="space-y-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              MLX Models ({mlxModels.length})
            </h3>
            <div className="grid gap-2">
              {mlxModels.map(model => (
                <ModelCard
                  key={model.path}
                  model={model}
                  onInspect={() => onInspect(model.path)}
                  onDiagnose={() => onDiagnose(model.path)}
                  onConvert={() => onConvert(model.path)}
                  onServe={() => onServe(model.path)}
                />
              ))}
            </div>
          </div>
        )}

        {/* Empty state */}
        {!loading && mlxModels.length === 0 && (
          <div className="text-center py-12">
            <HardDrive className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
            <p className="text-sm text-muted-foreground">
              {filter ? 'No models match your search' : 'No local models found'}
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Download models from the Server tab, or use Convert to quantize a HuggingFace model
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

function ModelCard({ model, onInspect, onDiagnose, onConvert, onServe }: {
  model: LocalModel
  onInspect: () => void
  onDiagnose: () => void
  onConvert: () => void
  onServe: () => void
}) {
  return (
    <div className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-accent/50 transition-colors group">
      <div className="flex items-center gap-3 min-w-0 flex-1">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <p className="text-sm font-medium truncate">{model.name}</p>
            {model.quantization && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary flex-shrink-0">
                {model.quantization}
              </span>
            )}
            {model.size && (
              <span className="text-[10px] text-muted-foreground flex-shrink-0">
                {model.size}
              </span>
            )}
          </div>
          <p className="text-xs text-muted-foreground truncate mt-0.5">{model.path}</p>
        </div>
      </div>

      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0 ml-3">
        <button
          onClick={onInspect}
          className="p-1.5 text-muted-foreground hover:text-foreground rounded hover:bg-background transition-colors"
          title="Inspect"
        >
          <Eye className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={onDiagnose}
          className="p-1.5 text-muted-foreground hover:text-foreground rounded hover:bg-background transition-colors"
          title="Diagnose"
        >
          <Stethoscope className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={onConvert}
          className="p-1.5 text-muted-foreground hover:text-foreground rounded hover:bg-background transition-colors"
          title="Convert"
        >
          <ArrowRightLeft className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={onServe}
          className="p-1.5 text-muted-foreground hover:text-primary rounded hover:bg-background transition-colors"
          title="Serve"
        >
          <Play className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )
}
