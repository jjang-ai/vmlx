import { useState } from 'react'
import { ChevronDown, ChevronRight, HelpCircle } from 'lucide-react'

interface ImageSettingsData {
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

interface ImageSettingsProps {
  settings: ImageSettingsData
  onChange: (settings: ImageSettingsData) => void
  model: string | null
  mode: 'generate' | 'edit'
}

const SIZE_PRESETS = [
  { label: '512x512', width: 512, height: 512 },
  { label: '768x768', width: 768, height: 768 },
  { label: '1024x1024', width: 1024, height: 1024 },
  { label: '1024x768 (Landscape)', width: 1024, height: 768 },
  { label: '768x1024 (Portrait)', width: 768, height: 1024 },
  { label: '1280x720 (16:9)', width: 1280, height: 720 },
]

export function ImageSettings({ settings, onChange, model, mode }: ImageSettingsProps) {
  const isEdit = mode === 'edit'
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [showServer, setShowServer] = useState(false)
  const [showNegativeHelp, setShowNegativeHelp] = useState(false)

  const currentSize = SIZE_PRESETS.find(p => p.width === settings.width && p.height === settings.height)

  const update = (key: keyof ImageSettingsData, value: any) => {
    onChange({ ...settings, [key]: value })
  }

  return (
    <div className="border-b border-border bg-muted/30 px-4 py-3 max-h-[50vh] overflow-auto">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
        {isEdit ? 'Edit Settings' : 'Generation Settings'}
      </h3>

      {/* Standard Settings */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3">
        {/* Steps */}
        <div>
          <label className="text-xs text-muted-foreground block mb-1" title="Number of denoising iterations. More steps = better quality but slower. Schnell works well at 4, Dev needs 20+.">Steps &#9432;</label>
          <input
            type="number"
            value={settings.steps}
            onChange={(e) => update('steps', Math.max(1, Math.min(100, parseInt(e.target.value) || 1)))}
            className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
            min={1}
            max={100}
          />
        </div>

        {/* Size */}
        <div>
          <label className="text-xs text-muted-foreground block mb-1" title="Output image dimensions. Larger = more detail but slower and more memory. 1024x1024 is standard.">Size &#9432;</label>
          <select
            value={currentSize?.label || 'custom'}
            onChange={(e) => {
              const preset = SIZE_PRESETS.find(p => p.label === e.target.value)
              if (preset) onChange({ ...settings, width: preset.width, height: preset.height })
            }}
            className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          >
            {SIZE_PRESETS.map((p) => (
              <option key={p.label} value={p.label}>{p.label}</option>
            ))}
          </select>
        </div>

        {/* Guidance */}
        <div>
          <label className="text-xs text-muted-foreground block mb-1" title="How closely the image follows your prompt. Higher = more literal, lower = more creative. Default 3.5 works for most prompts.">Guidance &#9432;</label>
          <input
            type="number"
            value={settings.guidance}
            onChange={(e) => update('guidance', Math.max(0, Math.min(20, parseFloat(e.target.value) || 0)))}
            className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
            min={0}
            max={20}
            step={0.5}
          />
        </div>

        {/* Seed */}
        <div>
          <label className="text-xs text-muted-foreground block mb-1" title="Random seed for reproducibility. Same seed + same prompt = same image. Leave empty for random.">Seed &#9432;</label>
          <input
            type="number"
            value={settings.seed ?? ''}
            onChange={(e) => {
              const val = e.target.value.trim()
              update('seed', val ? parseInt(val) : undefined)
            }}
            placeholder="Random"
            className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>

        {/* Strength (edit mode only) */}
        {isEdit && (
          <div>
            <label className="text-xs text-muted-foreground block mb-1" title="How much to change the source image. 0.0 = no change, 1.0 = completely redrawn. 0.7-0.85 works well for most edits.">Strength &#9432;</label>
            <input
              type="number"
              value={settings.strength}
              onChange={(e) => update('strength', Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)))}
              className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
              min={0}
              max={1}
              step={0.05}
            />
          </div>
        )}

        {/* Count (generate mode only — edit always returns 1) */}
        {!isEdit && (
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Number of Images</label>
            <input
              type="number"
              value={settings.count}
              onChange={(e) => update('count', Math.max(1, Math.min(4, parseInt(e.target.value) || 1)))}
              className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
              min={1}
              max={4}
            />
          </div>
        )}

        {/* Quantize (read-only — set at server start) */}
        <div>
          <label className="text-xs text-muted-foreground block mb-1" title="Model precision. Set when starting the server and cannot be changed while running.">Quantize &#9432;</label>
          <div className="w-full px-2 py-1 bg-muted/50 border border-input rounded text-xs text-muted-foreground">
            {settings.quantize === 0 ? 'Full Precision' : `${settings.quantize}-bit`}
          </div>
        </div>
      </div>

      {/* Negative Prompt */}
      <div className="mb-3">
        <div className="flex items-center gap-1 mb-1">
          <label className="text-xs text-muted-foreground">Negative Prompt</label>
          <button
            type="button"
            onClick={() => setShowNegativeHelp(p => !p)}
            className="text-muted-foreground hover:text-foreground"
          >
            <HelpCircle className="h-3 w-3" />
          </button>
        </div>
        {showNegativeHelp && (
          <p className="text-[10px] text-muted-foreground bg-muted/50 rounded px-2 py-1.5 mb-1.5">
            Describe what you <strong>don't</strong> want in the image. Useful for avoiding common artifacts.
            Example: <em>"blurry, low quality, text, watermark, deformed hands"</em>
          </p>
        )}
        <input
          type="text"
          value={settings.negativePrompt}
          onChange={(e) => update('negativePrompt', e.target.value)}
          placeholder="Things to avoid in the image..."
          className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
        />
      </div>

      {/* Advanced Section (collapsed) */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors mb-2"
      >
        {showAdvanced ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        Advanced
      </button>
      {showAdvanced && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3 pl-4 border-l border-border">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Width (custom)</label>
            <input
              type="number"
              value={settings.width}
              onChange={(e) => update('width', Math.max(64, Math.min(2048, parseInt(e.target.value) || 512)))}
              className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
              min={64}
              max={2048}
              step={64}
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Height (custom)</label>
            <input
              type="number"
              value={settings.height}
              onChange={(e) => update('height', Math.max(64, Math.min(2048, parseInt(e.target.value) || 512)))}
              className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
              min={64}
              max={2048}
              step={64}
            />
          </div>
        </div>
      )}

      {/* Server Section (collapsed) */}
      <button
        onClick={() => setShowServer(!showServer)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors mb-2"
      >
        {showServer ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        Server
      </button>
      {showServer && (
        <div className="pl-4 border-l border-border text-xs text-muted-foreground space-y-1">
          <p>Host: 0.0.0.0 (all interfaces)</p>
          <p>Port: auto-assigned</p>
          <p>Model: {model || 'none'}</p>
          <p className="text-[10px] mt-2 opacity-70">
            Server settings are managed automatically for image models.
          </p>
        </div>
      )}
    </div>
  )
}
