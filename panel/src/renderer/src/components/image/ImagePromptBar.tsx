import { useState, useEffect, useRef, useCallback, KeyboardEvent, DragEvent, ClipboardEvent } from 'react'
import { Send, ImagePlus, X, Pencil, RefreshCw } from 'lucide-react'

interface ImagePromptBarSettings {
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
  mode: 'generate' | 'edit'
  sourceImage: { dataUrl: string; name: string } | null
  onSourceImageChange: (img: { dataUrl: string; name: string } | null) => void
  /** Original prompt when iterating (shown as base, user adds modifications) */
  iteratePrompt?: string | null
  /** Counter to force re-trigger even with same prompt */
  iterateCounter?: number
  /** Called when user clears the iterate state */
  onClearIterate?: () => void
}

export function ImagePromptBar({ onGenerate, disabled, generating, settings, onSettingsChange, mode, sourceImage, onSourceImageChange, iteratePrompt, iterateCounter, onClearIterate }: ImagePromptBarProps) {
  const [prompt, setPrompt] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Iterate mode: source image is from a previous generation
  const isIterating = !!(iteratePrompt && sourceImage)
  const isEdit = mode === 'edit'

  // When Iterate button is clicked, focus the modification input
  useEffect(() => {
    if (iteratePrompt != null) {
      setPrompt('') // Clear the modification field — user types what to change
      setTimeout(() => {
        textareaRef.current?.focus()
      }, 100)
    }
  }, [iteratePrompt, iterateCounter])

  // Build the actual prompt sent to the engine
  const buildFinalPrompt = useCallback((): string => {
    if (isIterating && prompt.trim()) {
      // Combine: original prompt + user's modification
      return `${iteratePrompt}, ${prompt.trim()}`
    }
    if (isIterating && !prompt.trim()) {
      // No modification — use original prompt as-is (variation with new seed)
      return iteratePrompt!
    }
    return prompt.trim()
  }, [prompt, iteratePrompt, isIterating])

  const canSubmit = !disabled && !generating && (isIterating ? true : prompt.trim()) && (!isEdit || !!sourceImage)

  const handleSubmit = useCallback(() => {
    if (!canSubmit) return
    const finalPrompt = buildFinalPrompt()
    if (finalPrompt) onGenerate(finalPrompt)
  }, [canSubmit, buildFinalPrompt, onGenerate])

  const handleKeyDown = useCallback((e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }, [handleSubmit])

  const handlePickImage = async () => {
    try {
      const picked = await window.api.chat.pickImages()
      if (picked?.[0]) onSourceImageChange(picked[0])
    } catch {}
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = () => onSourceImageChange({ dataUrl: reader.result as string, name: file.name })
    reader.readAsDataURL(file)
  }

  const handlePaste = (e: ClipboardEvent) => {
    // Allow paste for both edit and gen (img2img) modes
    const items = e.clipboardData.items
    for (const item of Array.from(items)) {
      if (item.type.startsWith('image/')) {
        const file = item.getAsFile()
        if (!file) continue
        const reader = new FileReader()
        reader.onload = () => onSourceImageChange({ dataUrl: reader.result as string, name: 'pasted-image.png' })
        reader.readAsDataURL(file)
        break
      }
    }
  }

  const handleClearIterate = () => {
    onSourceImageChange(null)
    setPrompt('')
    onClearIterate?.()
  }

  const currentSize = SIZE_PRESETS.find(p => p.width === settings.width && p.height === settings.height)
  const sizeLabel = currentSize?.label || `${settings.width}x${settings.height}`

  return (
    <div className="border-t border-border bg-background px-4 py-3">
      {/* Source image zone */}
      {isIterating ? (
        // Iterate mode: show source image + original prompt as context
        <div className="mb-2 p-2 bg-primary/5 border border-primary/20 rounded-md">
          <div className="flex items-center gap-3">
            <img
              src={sourceImage.dataUrl}
              alt="Source"
              className="h-14 w-14 rounded object-cover flex-shrink-0 border border-border"
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5 mb-0.5">
                <RefreshCw className="h-3 w-3 text-primary flex-shrink-0" />
                <span className="text-[10px] font-medium text-primary uppercase tracking-wider">Iterating</span>
              </div>
              <p className="text-xs text-foreground line-clamp-2" title={iteratePrompt!}>
                {iteratePrompt}
              </p>
            </div>
            <button
              onClick={handleClearIterate}
              className="p-1 text-muted-foreground hover:text-destructive rounded flex-shrink-0"
              title="Cancel iteration"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      ) : (
        // Normal mode: upload zone
        <div className="mb-2">
          {sourceImage ? (
            <div className="flex items-center gap-3 p-2 bg-muted/50 rounded-md">
              <img
                src={sourceImage.dataUrl}
                alt="Source"
                className="h-12 w-12 rounded object-cover flex-shrink-0"
              />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium truncate">{sourceImage.name}</p>
                <p className="text-[10px] text-muted-foreground">
                  {isEdit ? 'Source image for editing' : 'Source for img2img (strength controls how much changes)'}
                </p>
              </div>
              <button
                onClick={() => onSourceImageChange(null)}
                className="p-1 text-muted-foreground hover:text-destructive rounded"
                title="Remove source image"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
          ) : (
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
              onClick={handlePickImage}
              className={`flex items-center justify-center gap-2 p-2 border-2 border-dashed rounded-md cursor-pointer transition-colors ${
                dragOver ? 'border-primary bg-primary/5' : 'border-border/50 hover:border-primary/40 hover:bg-accent/30'
              }`}
            >
              <ImagePlus className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-[11px] text-muted-foreground">
                {isEdit ? 'Upload source image (required)' : 'Drop image to iterate or restyle (optional)'}
              </span>
            </div>
          )}
        </div>
      )}

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

        {/* Strength: show when source image is present (iterate or edit) */}
        {(isEdit || sourceImage) && (
          <div className="flex items-center gap-1.5">
            <label className="text-muted-foreground">Strength</label>
            <input
              type="number"
              value={settings.strength}
              onChange={(e) => onSettingsChange({ ...settings, strength: Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)) })}
              className="w-14 px-1.5 py-0.5 bg-muted border border-input rounded text-xs text-center focus:outline-none focus:ring-1 focus:ring-ring"
              min={0}
              max={1}
              step={0.05}
            />
          </div>
        )}

        {!isEdit && !isIterating && settings.count > 1 && (
          <span className="text-muted-foreground">x{settings.count} images</span>
        )}
      </div>

      {/* Prompt input + submit button */}
      <div className="flex gap-2">
        <textarea
          ref={textareaRef}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          placeholder={
            disabled ? 'Waiting for server to start...'
              : isIterating ? 'Add modifications (or leave empty for variation with new seed)...'
              : isEdit && !sourceImage ? 'Upload a source image above, then describe your edit...'
              : isEdit ? 'Describe the edit to apply...'
              : 'Describe the image you want to generate...'
          }
          disabled={disabled || generating}
          rows={2}
          className="flex-1 px-3 py-2 bg-muted border border-input rounded-md text-sm resize-none focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50 placeholder:text-muted-foreground/60"
        />
        {generating ? (
          <button
            onClick={() => window.api.image.cancelGeneration()}
            className="px-4 py-2 rounded-md bg-destructive text-destructive-foreground hover:bg-destructive/90 flex items-center gap-2 self-end"
          >
            <X className="h-4 w-4" />
            <span className="text-sm">Cancel</span>
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            className={`px-4 py-2 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 self-end ${
              isIterating
                ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                : isEdit
                ? 'bg-violet-600 text-white hover:bg-violet-700'
                : 'bg-primary text-primary-foreground hover:bg-primary/90'
            }`}
          >
            {isIterating ? <RefreshCw className="h-4 w-4" /> : isEdit ? <Pencil className="h-4 w-4" /> : <Send className="h-4 w-4" />}
            <span className="text-sm">{isIterating ? 'Iterate' : isEdit ? 'Edit' : 'Generate'}</span>
          </button>
        )}
      </div>

      {/* Combined prompt preview when iterating */}
      {isIterating && prompt.trim() && (
        <p className="mt-1.5 text-[10px] text-muted-foreground truncate">
          Prompt: <span className="text-foreground/70">{iteratePrompt}, {prompt.trim()}</span>
        </p>
      )}
    </div>
  )
}
