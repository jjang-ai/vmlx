import { useState, useRef, useCallback, KeyboardEvent } from 'react'
import { Send, Loader2 } from 'lucide-react'

interface ImagePromptBarSettings {
  steps: number
  width: number
  height: number
  guidance: number
  negativePrompt: string
  seed?: number
  count: number
  quantize: number
}

const SIZE_PRESETS = [
  { label: '512x512', width: 512, height: 512 },
  { label: '768x768', width: 768, height: 768 },
  { label: '1024x1024', width: 1024, height: 1024 },
  { label: '1024x768 (Landscape)', width: 1024, height: 768 },
  { label: '768x1024 (Portrait)', width: 768, height: 1024 },
  { label: '1280x720 (16:9)', width: 1280, height: 720 },
]

interface ImagePromptBarProps {
  onGenerate: (prompt: string) => void
  disabled: boolean
  generating: boolean
  settings: ImagePromptBarSettings
  onSettingsChange: (settings: ImagePromptBarSettings) => void
  model: string | null
}

export function ImagePromptBar({ onGenerate, disabled, generating, settings, onSettingsChange, model }: ImagePromptBarProps) {
  const [prompt, setPrompt] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const handleSubmit = useCallback(() => {
    const trimmed = prompt.trim()
    if (!trimmed || disabled || generating) return
    onGenerate(trimmed)
    // Don't clear prompt -- user might want to iterate
  }, [prompt, disabled, generating, onGenerate])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }, [handleSubmit])

  const currentSize = SIZE_PRESETS.find(p => p.width === settings.width && p.height === settings.height)
  const sizeLabel = currentSize?.label || `${settings.width}x${settings.height}`

  return (
    <div className="border-t border-border bg-background px-4 py-3">
      {/* Quick settings row */}
      <div className="flex items-center gap-4 mb-2 text-xs">
        <div className="flex items-center gap-1.5">
          <label className="text-muted-foreground">Steps</label>
          <input
            type="number"
            value={settings.steps}
            onChange={(e) => onSettingsChange({ ...settings, steps: Math.max(1, Math.min(100, parseInt(e.target.value) || 1)) })}
            className="w-14 px-1.5 py-0.5 bg-muted border border-input rounded text-xs text-center focus:outline-none focus:ring-1 focus:ring-ring"
            min={1}
            max={100}
          />
        </div>

        <div className="flex items-center gap-1.5">
          <label className="text-muted-foreground">Size</label>
          <select
            value={sizeLabel}
            onChange={(e) => {
              const preset = SIZE_PRESETS.find(p => p.label === e.target.value)
              if (preset) onSettingsChange({ ...settings, width: preset.width, height: preset.height })
            }}
            className="px-1.5 py-0.5 bg-muted border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          >
            {SIZE_PRESETS.map((p) => (
              <option key={p.label} value={p.label}>{p.label}</option>
            ))}
          </select>
        </div>

        <div className="flex items-center gap-1.5">
          <label className="text-muted-foreground">Guidance</label>
          <input
            type="number"
            value={settings.guidance}
            onChange={(e) => onSettingsChange({ ...settings, guidance: Math.max(0, Math.min(20, parseFloat(e.target.value) || 0)) })}
            className="w-14 px-1.5 py-0.5 bg-muted border border-input rounded text-xs text-center focus:outline-none focus:ring-1 focus:ring-ring"
            min={0}
            max={20}
            step={0.5}
          />
        </div>

        {settings.count > 1 && (
          <span className="text-muted-foreground">x{settings.count} images</span>
        )}
      </div>

      {/* Prompt input + generate button */}
      <div className="flex gap-2">
        <textarea
          ref={textareaRef}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={disabled ? 'Waiting for server to start...' : 'Describe the image you want to generate...'}
          disabled={disabled || generating}
          rows={2}
          className="flex-1 px-3 py-2 bg-muted border border-input rounded-md text-sm resize-none focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50 placeholder:text-muted-foreground/60"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || generating || !prompt.trim()}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 self-end"
        >
          {generating ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              <span className="text-sm">Generating</span>
            </>
          ) : (
            <>
              <Send className="h-4 w-4" />
              <span className="text-sm">Generate</span>
            </>
          )}
        </button>
      </div>
    </div>
  )
}
