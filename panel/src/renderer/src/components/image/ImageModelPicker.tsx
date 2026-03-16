import { useState } from 'react'
import { Zap, Sparkles, Gauge, Box, Layers, FolderOpen, Play, Download } from 'lucide-react'

const NAMED_MODELS = [
  { id: 'schnell', name: 'Flux Schnell', desc: 'Fast generation (4 steps)', size: '~12 GB', steps: 4, icon: Zap },
  { id: 'dev', name: 'Flux Dev', desc: 'High quality (20 steps)', size: '~24 GB', steps: 20, icon: Sparkles },
  { id: 'z-image-turbo', name: 'Z-Image Turbo', desc: 'xcreates turbo (4 steps)', size: '~12 GB', steps: 4, icon: Gauge },
  { id: 'flux2-klein-4b', name: 'Flux Klein 4B', desc: 'Compact model (20 steps)', size: '~8 GB', steps: 20, icon: Box },
  { id: 'flux2-klein-9b', name: 'Flux Klein 9B', desc: 'Mid-size model (20 steps)', size: '~16 GB', steps: 20, icon: Layers },
]

const QUANTIZE_OPTIONS = [
  { value: 4, label: '4-bit', desc: 'Fastest, lowest memory' },
  { value: 8, label: '8-bit', desc: 'Better quality' },
  { value: 0, label: 'Full', desc: 'Best quality, most memory' },
]

interface ImageModelPickerProps {
  onSelect: (modelId: string, quantize?: number) => void
}

export function ImageModelPicker({ onSelect }: ImageModelPickerProps) {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [selectedQuantize, setSelectedQuantize] = useState<number>(4)
  const [customPath, setCustomPath] = useState('')
  const [showCustom, setShowCustom] = useState(false)

  const handleStart = () => {
    if (showCustom && customPath.trim()) {
      onSelect(customPath.trim(), selectedQuantize)
    } else if (selectedModel) {
      onSelect(selectedModel, selectedQuantize)
    }
  }

  const handleBrowse = async () => {
    try {
      const result = await window.api.models.browseDirectory()
      if (result?.path) setCustomPath(result.path)
    } catch {}
  }

  return (
    <div className="h-full flex items-center justify-center p-8 overflow-auto">
      <div className="max-w-3xl w-full space-y-6">
        {/* Header */}
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">Start Image Server</h2>
          <p className="text-sm text-muted-foreground">
            Choose a model below or browse for a local model. The server will download the model on first use if needed.
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            <strong>Schnell</strong> is fastest (4 steps). <strong>Dev</strong> has highest quality (20 steps).
            <strong>4-bit</strong> uses least memory (~6GB). <strong>Full</strong> precision needs ~24GB.
            For Macs with 16GB RAM, use Schnell or Klein 4B at 4-bit.
          </p>
        </div>

        {/* Preset Models */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {NAMED_MODELS.map((model) => {
            const Icon = model.icon
            const isSelected = selectedModel === model.id && !showCustom
            return (
              <button
                key={model.id}
                onClick={() => { setSelectedModel(model.id); setShowCustom(false) }}
                className={`text-left p-4 border rounded-lg transition-all ${
                  isSelected
                    ? 'border-primary bg-primary/5 ring-1 ring-primary/30'
                    : 'border-border hover:border-primary/40 hover:bg-accent/30'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
                    isSelected ? 'bg-primary/20' : 'bg-muted'
                  }`}>
                    <Icon className={`h-4 w-4 ${isSelected ? 'text-primary' : 'text-muted-foreground'}`} />
                  </div>
                  <div className="min-w-0">
                    <h3 className="font-semibold text-sm">{model.name}</h3>
                    <p className="text-[11px] text-muted-foreground mt-0.5">{model.desc}</p>
                    <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
                      <span>{model.steps} steps</span>
                      <span>·</span>
                      <span>{model.size}</span>
                    </div>
                  </div>
                </div>
              </button>
            )
          })}
        </div>

        {/* Custom Model */}
        <div className="border border-border rounded-lg p-4">
          <button
            onClick={() => { setShowCustom(!showCustom); setSelectedModel(null) }}
            className={`flex items-center gap-2 text-sm font-medium ${
              showCustom ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <FolderOpen className="h-4 w-4" />
            Use custom model (HuggingFace ID or local path)
          </button>
          {showCustom && (
            <div className="mt-3 flex gap-2">
              <input
                type="text"
                value={customPath}
                onChange={e => setCustomPath(e.target.value)}
                placeholder="e.g., black-forest-labs/FLUX.1-schnell or /path/to/model"
                className="flex-1 px-3 py-2 text-sm bg-background border border-input rounded"
              />
              <button
                onClick={handleBrowse}
                className="px-3 py-2 text-sm border border-input rounded hover:bg-accent"
                title="Browse folders"
              >
                <FolderOpen className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>

        {/* Quantize + Start */}
        {(selectedModel || (showCustom && customPath.trim())) && (
          <div className="flex items-center gap-4 p-4 bg-card border border-border rounded-lg">
            <div className="flex-1">
              <label className="text-xs text-muted-foreground block mb-1.5">Quantization</label>
              <div className="flex gap-1.5">
                {QUANTIZE_OPTIONS.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => setSelectedQuantize(opt.value)}
                    className={`flex-1 px-3 py-1.5 text-xs rounded transition-colors ${
                      selectedQuantize === opt.value
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted text-muted-foreground hover:bg-muted/80'
                    }`}
                  >
                    <span className="font-medium">{opt.label}</span>
                    <span className="block text-[10px] opacity-75 mt-0.5">{opt.desc}</span>
                  </button>
                ))}
              </div>
            </div>
            <button
              onClick={handleStart}
              className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2 font-medium text-sm"
            >
              <Play className="h-4 w-4" />
              Start Server
            </button>
          </div>
        )}

        {/* Info */}
        <p className="text-[11px] text-muted-foreground text-center">
          Named models are downloaded automatically via mflux on first use (~12-24 GB depending on model).
          You need to accept the <a href="https://huggingface.co/black-forest-labs/FLUX.1-schnell" className="underline" target="_blank" rel="noopener">Flux license on HuggingFace</a> first.
        </p>
      </div>
    </div>
  )
}
