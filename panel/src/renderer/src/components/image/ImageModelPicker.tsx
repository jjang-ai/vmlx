import { useState, useEffect } from 'react'
import { Zap, Sparkles, Gauge, Box, Layers, FolderOpen, Play, Download, AlertCircle, CheckCircle, Loader2, Pencil } from 'lucide-react'
import { IMAGE_MODELS } from '../../../../shared/imageModels'

// Map model IDs to icons (icons are React components, can't live in the shared registry)
const MODEL_ICONS: Record<string, typeof Zap> = {
  'schnell': Zap,
  'flux2-klein-4b': Box,
  'z-image-turbo': Gauge,
  'flux2-klein-9b': Layers,
  'dev': Sparkles,
  'qwen-image-edit': Pencil,
}

// Build NAMED_MODELS from the shared registry + icons
const NAMED_MODELS = IMAGE_MODELS.map(m => ({
  ...m,
  icon: MODEL_ICONS[m.id] || Zap,
}))

const QUANTIZE_OPTIONS = [
  { value: 4, label: '4-bit', desc: 'Fastest, lowest memory' },
  { value: 8, label: '8-bit', desc: 'Better quality' },
  { value: 0, label: 'Full', desc: 'Best quality, most memory' },
]

interface ImageModelPickerProps {
  onSelect: (modelId: string, quantize?: number, category?: 'generate' | 'edit') => void
}

type DownloadState = 'idle' | 'checking' | 'downloading' | 'ready' | 'error'

export function ImageModelPicker({ onSelect }: ImageModelPickerProps) {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [selectedQuantize, setSelectedQuantize] = useState<number>(4)
  const [customPath, setCustomPath] = useState('')
  const [customCategory, setCustomCategory] = useState<'generate' | 'edit'>('generate')
  const [showCustom, setShowCustom] = useState(false)

  // Download state
  const [downloadState, setDownloadState] = useState<DownloadState>('idle')
  const [downloadProgress, setDownloadProgress] = useState<any>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [modelAvailability, setModelAvailability] = useState<Record<string, boolean>>({})
  const [modelMissing, setModelMissing] = useState<Record<string, string[]>>({})
  const [hasHfToken, setHasHfToken] = useState(false)

  // Check HF token and in-progress downloads on mount
  useEffect(() => {
    window.api.settings.get('hf_api_key').then((val: string | null) => {
      setHasHfToken(!!val)
    })
    // Recover state if a download is already in progress (e.g., user navigated away and back)
    window.api.models.getDownloadStatus().then((status: any) => {
      if (status.active) {
        setDownloadState('downloading')
        if (status.active.progress) setDownloadProgress(status.active.progress)
      }
    }).catch(() => {})
  }, [])

  // Check model availability when selection or quantize changes
  useEffect(() => {
    if (!selectedModel || showCustom) return
    const key = `${selectedModel}-${selectedQuantize}`
    if (modelAvailability[key] !== undefined) return

    setDownloadState('checking')
    window.api.models.checkImageModel(selectedModel, selectedQuantize)
      .then((result: any) => {
        setModelAvailability(prev => ({ ...prev, [key]: result.available }))
        if (result.missing && result.missing.length > 0) {
          setModelMissing(prev => ({ ...prev, [key]: result.missing }))
        }
        setDownloadState(result.available ? 'ready' : 'idle')
      })
      .catch(() => {
        setDownloadState('idle')
      })
  }, [selectedModel, selectedQuantize, showCustom])

  // Listen for download progress
  useEffect(() => {
    const unsubProgress = window.api.models.onDownloadProgress((data: any) => {
      if (downloadState === 'downloading') {
        setDownloadProgress(data.progress)
      }
    })
    const unsubComplete = window.api.models.onDownloadComplete((data: any) => {
      if (downloadState === 'downloading') {
        if (data.status === 'complete') {
          setDownloadState('ready')
          setDownloadProgress(null)
          // Update availability cache
          if (selectedModel) {
            const key = `${selectedModel}-${selectedQuantize}`
            setModelAvailability(prev => ({ ...prev, [key]: true }))
          }
        }
      }
    })
    const unsubError = window.api.models.onDownloadError((data: any) => {
      if (downloadState === 'downloading') {
        const errMsg = data.error || 'Download failed'
        const isGated = data.gated
        if (isGated) {
          setDownloadError(
            'This model requires a HuggingFace token. Go to the Server tab > Download section and add your HF token, ' +
            'then accept the model license at huggingface.co.'
          )
        } else {
          setDownloadError(errMsg)
        }
        setDownloadState('error')
        setDownloadProgress(null)
      }
    })
    return () => {
      unsubProgress()
      unsubComplete()
      unsubError()
    }
  }, [downloadState, selectedModel, selectedQuantize])

  const handleDownload = async () => {
    if (!selectedModel) return
    setDownloadState('downloading')
    setDownloadError(null)
    setDownloadProgress(null)

    try {
      const result = await window.api.models.downloadImageModel(selectedModel, selectedQuantize)
      if (result.status === 'already_downloaded') {
        setDownloadState('ready')
        const key = `${selectedModel}-${selectedQuantize}`
        setModelAvailability(prev => ({ ...prev, [key]: true }))
      }
      // Otherwise, download events will update state
    } catch (err) {
      setDownloadError((err as Error).message)
      setDownloadState('error')
    }
  }

  const handleStart = () => {
    if (showCustom && customPath.trim()) {
      onSelect(customPath.trim(), selectedQuantize, customCategory)
    } else if (selectedModel) {
      const modelInfo = NAMED_MODELS.find(m => m.id === selectedModel)
      onSelect(selectedModel, selectedQuantize, modelInfo?.category || 'generate')
    }
  }

  const handleBrowse = async () => {
    try {
      const result = await window.api.models.browseDirectory()
      if (result?.path) setCustomPath(result.path)
    } catch {}
  }

  // Get allowed quantize options for the selected model
  const selectedModelInfo = NAMED_MODELS.find(m => m.id === selectedModel)
  const allowedQuantize = selectedModelInfo?.quantizeOptions || [4, 8, 0]
  const filteredQuantizeOptions = QUANTIZE_OPTIONS.filter(opt => allowedQuantize.includes(opt.value))

  const isModelAvailable = selectedModel
    ? modelAvailability[`${selectedModel}-${selectedQuantize}`]
    : false

  const missingComponents = selectedModel
    ? modelMissing[`${selectedModel}-${selectedQuantize}`]
    : undefined

  return (
    <div className="h-full flex items-center justify-center p-8 overflow-auto">
      <div className="max-w-3xl w-full space-y-6">
        {/* Header */}
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">Start Image Server</h2>
          <p className="text-sm text-muted-foreground">
            Choose a model below or browse for a local model.
            {!hasHfToken && (
              <span className="text-warning"> You haven't set an HF token — some models may require authentication.</span>
            )}
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            <strong>Generation:</strong> Schnell is fastest (4 steps), Dev is highest quality. 4-bit uses ~6GB, Full ~24GB.
            For 16GB RAM, use Schnell or Klein 4B at 4-bit.
            <br />
            <strong>Editing:</strong> Full precision only — quantized edit models produce unusable results.
            Requires 24-54 GB depending on model.
          </p>
        </div>

        {/* Generation Models */}
        <div>
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Image Generation</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {NAMED_MODELS.filter(m => m.category === 'generate').map((model) => {
            const Icon = model.icon
            const isSelected = selectedModel === model.id && !showCustom
            const key = `${model.id}-${selectedQuantize}`
            const available = modelAvailability[key]
            return (
              <button
                key={model.id}
                onClick={() => { setSelectedModel(model.id); setShowCustom(false); setDownloadState('idle'); setDownloadError(null); if (!model.quantizeOptions.includes(selectedQuantize)) setSelectedQuantize(model.quantizeOptions[0]) }}
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
                    <div className="flex items-center gap-1.5">
                      <h3 className="font-semibold text-sm">{model.name}</h3>
                      {available === true && (
                        <CheckCircle className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                      )}
                    </div>
                    <p className="text-[11px] text-muted-foreground mt-0.5">{model.desc}</p>
                    <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
                      <span>{model.steps} steps</span>
                      <span>·</span>
                      <span>{model.size}</span>
                      {available === true && <span className="text-green-500">· Downloaded</span>}
                    </div>
                  </div>
                </div>
              </button>
            )
          })}
          </div>
        </div>

        {/* Image Editing Models */}
        <div>
          <h3 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Image Editing</h3>
          <p className="text-[11px] text-muted-foreground mb-2">Submit a photo + text prompt to edit, inpaint, or transform images. Full precision only for best quality. More editing models coming soon.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {NAMED_MODELS.filter(m => m.category === 'edit').map((model) => {
            const Icon = model.icon
            const isSelected = selectedModel === model.id && !showCustom
            const key = `${model.id}-${selectedQuantize}`
            const available = modelAvailability[key]
            return (
              <button
                key={model.id}
                onClick={() => { setSelectedModel(model.id); setShowCustom(false); setDownloadState('idle'); setDownloadError(null); if (!model.quantizeOptions.includes(selectedQuantize)) setSelectedQuantize(model.quantizeOptions[0]) }}
                className={`text-left p-4 border rounded-lg transition-all ${
                  isSelected
                    ? 'border-violet-500 bg-violet-500/5 ring-1 ring-violet-500/30'
                    : 'border-border hover:border-violet-500/40 hover:bg-accent/30'
                }`}
              >
                <div className="flex items-start gap-3">
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 ${
                    isSelected ? 'bg-violet-500/20' : 'bg-muted'
                  }`}>
                    <Icon className={`h-4 w-4 ${isSelected ? 'text-violet-400' : 'text-muted-foreground'}`} />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <h3 className="font-semibold text-sm">{model.name}</h3>
                      {available === true && (
                        <CheckCircle className="h-3.5 w-3.5 text-green-500 flex-shrink-0" />
                      )}
                    </div>
                    <p className="text-[11px] text-muted-foreground mt-0.5">{model.desc}</p>
                    <div className="flex items-center gap-2 mt-1 text-[10px] text-muted-foreground">
                      <span>{model.steps} steps</span>
                      <span>·</span>
                      <span>{model.size}</span>
                      {available === true && <span className="text-green-500">· Downloaded</span>}
                    </div>
                  </div>
                </div>
              </button>
            )
          })}
          </div>
        </div>

        {/* Custom Model */}
        <div className="border border-border rounded-lg p-4">
          <button
            onClick={() => { setShowCustom(!showCustom); setSelectedModel(null); setDownloadState('idle'); setDownloadError(null) }}
            className={`flex items-center gap-2 text-sm font-medium ${
              showCustom ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <FolderOpen className="h-4 w-4" />
            Use custom model (HuggingFace ID or local path)
          </button>
          {showCustom && (
            <div className="mt-3 space-y-2">
              <div className="flex gap-2">
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
              <div className="flex items-center gap-2">
                <label className="text-xs text-muted-foreground">Mode:</label>
                <select
                  value={customCategory}
                  onChange={e => setCustomCategory(e.target.value as 'generate' | 'edit')}
                  className="px-2 py-1 text-xs bg-background border border-input rounded"
                >
                  <option value="generate">Image Generation</option>
                  <option value="edit">Image Editing</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Download Progress / Error */}
        {downloadState === 'downloading' && downloadProgress && (
          <div className="p-3 bg-primary/5 border border-primary/20 rounded-lg">
            <div className="flex items-center gap-2 text-sm">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              <span className="font-medium">Downloading model...</span>
              {downloadProgress.percent != null && (
                <span className="text-muted-foreground">{downloadProgress.percent}%</span>
              )}
              {downloadProgress.speed && (
                <span className="text-muted-foreground">{downloadProgress.speed}</span>
              )}
              {downloadProgress.eta && (
                <span className="text-muted-foreground">ETA {downloadProgress.eta}</span>
              )}
            </div>
            {downloadProgress.percent != null && (
              <div className="mt-2 h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(downloadProgress.percent, 100)}%` }}
                />
              </div>
            )}
            {downloadProgress.currentFile && (
              <p className="text-[10px] text-muted-foreground mt-1 truncate">{downloadProgress.currentFile}</p>
            )}
          </div>
        )}

        {downloadState === 'downloading' && !downloadProgress && (
          <div className="p-3 bg-primary/5 border border-primary/20 rounded-lg flex items-center gap-2 text-sm">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            <span>Starting download...</span>
          </div>
        )}

        {downloadError && (
          <div className="p-3 bg-destructive/10 border border-destructive/30 rounded-lg">
            <div className="flex items-start gap-2 text-sm text-destructive">
              <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Download failed</p>
                <p className="text-xs mt-1">{downloadError}</p>
                {!hasHfToken && (
                  <p className="text-xs mt-2">
                    Add your HuggingFace token in the <strong>Server tab &gt; Download</strong> section.{' '}
                    <a
                      href="https://huggingface.co/settings/tokens"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline"
                    >
                      Get a token here
                    </a>
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Incomplete model warning — model has some weights but is missing required components */}
        {!isModelAvailable && missingComponents && missingComponents.length > 0 && downloadState !== 'downloading' && (
          <div className="p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
            <div className="flex items-start gap-2 text-sm text-yellow-600 dark:text-yellow-400">
              <AlertCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-medium">Incomplete download</p>
                <p className="text-xs mt-1">
                  This model has weights but is missing required components: <strong>{missingComponents.join(', ')}</strong>.
                  Re-download the model to get all files, or the server will hang trying to fetch them.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Quantize + Action Buttons */}
        {(selectedModel || (showCustom && customPath.trim())) && (
          <div className="flex items-center gap-4 p-4 bg-card border border-border rounded-lg">
            <div className="flex-1">
              <label className="text-xs text-muted-foreground block mb-1.5">Quantization</label>
              <div className="flex gap-1.5">
                {filteredQuantizeOptions.map(opt => (
                  <button
                    key={opt.value}
                    onClick={() => { setSelectedQuantize(opt.value); setDownloadState('idle'); setDownloadError(null) }}
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

            {/* Show Download or Start button based on availability */}
            {showCustom ? (
              <button
                onClick={handleStart}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2 font-medium text-sm"
              >
                <Play className="h-4 w-4" />
                Start Server
              </button>
            ) : isModelAvailable || downloadState === 'ready' ? (
              <button
                onClick={handleStart}
                className="px-6 py-3 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 flex items-center gap-2 font-medium text-sm"
              >
                <Play className="h-4 w-4" />
                Start Server
              </button>
            ) : downloadState === 'downloading' ? (
              <button
                disabled
                className="px-6 py-3 bg-muted text-muted-foreground rounded-lg flex items-center gap-2 font-medium text-sm opacity-60"
              >
                <Loader2 className="h-4 w-4 animate-spin" />
                Downloading...
              </button>
            ) : downloadState === 'checking' ? (
              <button
                disabled
                className="px-6 py-3 bg-muted text-muted-foreground rounded-lg flex items-center gap-2 font-medium text-sm opacity-60"
              >
                <Loader2 className="h-4 w-4 animate-spin" />
                Checking...
              </button>
            ) : (
              <div className="flex gap-2">
                <button
                  onClick={handleDownload}
                  className="px-5 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2 font-medium text-sm"
                >
                  <Download className="h-4 w-4" />
                  Download First
                </button>
                <button
                  onClick={handleStart}
                  className="px-5 py-3 border border-border text-foreground rounded-lg hover:bg-accent flex items-center gap-2 font-medium text-sm"
                  title="Start anyway — the server will download the model (no progress bar)"
                >
                  <Play className="h-4 w-4" />
                  Start
                </button>
              </div>
            )}
          </div>
        )}

        {/* View Downloads + Info */}
        <div className="flex justify-center mb-2">
          <button
            onClick={() => window.dispatchEvent(new Event('open-download-popup'))}
            className="text-[11px] px-2 py-1 border border-border rounded hover:bg-accent text-muted-foreground hover:text-foreground"
          >
            View Downloads
          </button>
        </div>
        <p className="text-[11px] text-muted-foreground text-center">
          Models are downloaded from HuggingFace (~6-24 GB depending on model and quantization).
          {!hasHfToken && (
            <span>
              {' '}You may need to <a href="https://huggingface.co/settings/tokens" className="underline" target="_blank" rel="noopener">set an HF token</a> and accept the{' '}
              <a href="https://huggingface.co/black-forest-labs/FLUX.1-schnell" className="underline" target="_blank" rel="noopener">Flux license</a>.
            </span>
          )}
        </p>
      </div>
    </div>
  )
}
