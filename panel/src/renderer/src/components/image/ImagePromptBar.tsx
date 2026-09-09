import { useState, useEffect, useRef, useCallback, KeyboardEvent, DragEvent, ClipboardEvent } from 'react'
import { Send, ImagePlus, X, Pencil, RefreshCw, HelpCircle, Paintbrush } from 'lucide-react'
import { MaskPainter } from './MaskPainter'
import { useTranslation } from '../../i18n'
import { imageGuidanceFromInput, imageVariationPreservesSource, type ImageCapabilities } from '../../../../shared/imageCapabilities'

/** Inline help icon with tooltip */
function Help({ tip }: { tip: string }) {
  return <span title={tip} className="inline-flex cursor-help flex-shrink-0"><HelpCircle className="h-3 w-3 text-muted-foreground/50 hover:text-muted-foreground" /></span>
}

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
  prompt: string
  onPromptChange: (prompt: string) => void
  onGenerate: (prompt: string) => void
  disabled: boolean
  generating: boolean
  cancelling?: boolean
  settings: ImagePromptBarSettings
  onSettingsChange: (settings: ImagePromptBarSettings) => void
  mode: 'generate' | 'edit'
  modelName?: string | null
  sourceImage: { dataUrl: string; name: string } | null
  onSourceImageChange: (img: { dataUrl: string; name: string } | null) => void
  /** Mask data URL for Fill inpainting (only for Fill model) */
  maskBase64?: string | null
  /** Called when mask is created/cleared */
  onMaskChange?: (mask: string | null) => void
  /** Original prompt when iterating (shown as base, user adds modifications) */
  iteratePrompt?: string | null
  /** Counter to force re-trigger even with same prompt */
  iterateCounter?: number
  /** Called when user clears the iterate state */
  onClearIterate?: () => void
  /** From /health.image of the running server; null while unknown. */
  capabilities?: ImageCapabilities | null
}

export function ImagePromptBar({ prompt, onPromptChange: setPrompt, capabilities, onGenerate, disabled, generating, cancelling, settings, onSettingsChange, mode, sourceImage, onSourceImageChange, maskBase64, onMaskChange, iteratePrompt, iterateCounter, onClearIterate }: ImagePromptBarProps) {
  const { t } = useTranslation()
  const [dragOver, setDragOver] = useState(false)
  const [showMaskPainter, setShowMaskPainter] = useState(false)
  const [cancelRequested, setCancelRequested] = useState(false)
  useEffect(() => { if (!generating) setCancelRequested(false) }, [generating])
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // The loaded adapter owns the mask contract. Local folders and served
  // aliases may be renamed; a name containing "fill" proves nothing.
  const isFillModel = mode === 'edit' && capabilities?.loaded === true && capabilities.mask === 'required'

  const isEdit = mode === 'edit'
  // Variation mode: gen model with source image from Iterate button
  const isVariation = !isEdit && !!(iteratePrompt && sourceImage)
  const preservesSource = isVariation && imageVariationPreservesSource(capabilities)

  // Auto-open mask painter when Fill model gets a source image (any method)
  useEffect(() => {
    if (isFillModel && sourceImage && !maskBase64) {
      setShowMaskPainter(true)
    }
  }, [isFillModel, sourceImage, maskBase64])

  // When Iterate button is clicked, focus the input
  useEffect(() => {
    if (iteratePrompt != null) {
      const timer = setTimeout(() => textareaRef.current?.focus(), 100)
      return () => clearTimeout(timer)
    }
    return undefined
  }, [iteratePrompt, iterateCounter])

  // Build the actual prompt sent to the engine
  const buildFinalPrompt = useCallback((): string => {
    if (isVariation && prompt.trim()) {
      return `${iteratePrompt}, ${prompt.trim()}`
    }
    if (isVariation && !prompt.trim()) {
      return iteratePrompt!
    }
    return prompt.trim()
  }, [prompt, iteratePrompt, isVariation])

  // Fill model requires both source image AND mask
  const needsMask = isFillModel && isEdit
  const canSubmit = !disabled && !generating && (isVariation ? true : prompt.trim()) && (!isEdit || !!sourceImage) && (!needsMask || !!maskBase64)

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
      if (picked?.[0]) {
        onSourceImageChange(picked[0])
        // Auto-open mask painter for Fill model
        if (isFillModel) setTimeout(() => setShowMaskPainter(true), 100)
      }
    } catch {}
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = () => {
      onSourceImageChange({ dataUrl: reader.result as string, name: file.name })
      if (isFillModel) setTimeout(() => setShowMaskPainter(true), 100)
    }
    reader.readAsDataURL(file)
  }

  const handlePaste = (e: ClipboardEvent) => {
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

  const currentSize = SIZE_PRESETS.find(p => p.width === settings.width && p.height === settings.height)
  const sizeLabel = currentSize?.label || `${settings.width}x${settings.height}`

  return (
    <div className="border-t border-border bg-background px-4 py-3">
      {/* Source image zone */}
      {isVariation ? (
        // Variation mode: show source + original prompt
        <div className="mb-2 p-2 bg-emerald-500/5 border border-emerald-500/20 rounded-md">
          <div className="flex items-center gap-3">
            <img src={sourceImage!.dataUrl} alt={t('image.gallery.sourceAlt')} className="h-14 w-14 rounded object-cover flex-shrink-0 border border-border" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5 mb-0.5">
                <RefreshCw className="h-3 w-3 text-emerald-500 flex-shrink-0" />
                <span className="text-[10px] font-medium text-emerald-500 uppercase tracking-wider">{t('image.prompt.variationLabel')}</span>
                <span className="text-[10px] text-muted-foreground ml-1">{t('image.prompt.strengthValue', { value: settings.strength })}</span>
              </div>
              <p className="text-xs text-foreground line-clamp-2" title={iteratePrompt!}>{iteratePrompt}</p>
            </div>
            <button onClick={() => { onSourceImageChange(null); onClearIterate?.() }}
              className="p-1 text-muted-foreground hover:text-destructive rounded flex-shrink-0" title={t('image.prompt.cancelTitle')}>
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      ) : isEdit ? (
        // Edit mode: source required
        <div className="mb-2">
          {showMaskPainter && sourceImage ? (
            <MaskPainter
              imageDataUrl={sourceImage.dataUrl}
              initialMaskDataUrl={maskBase64}
              onConfirm={(mask) => { onMaskChange?.(mask); setShowMaskPainter(false) }}
              onCancel={() => setShowMaskPainter(false)}
            />
          ) : sourceImage ? (
            <div className="flex items-center gap-3 p-2 bg-violet-500/5 border border-violet-500/20 rounded-md">
              <img src={sourceImage.dataUrl} alt={t('image.gallery.sourceAlt')} className="h-12 w-12 rounded object-cover flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-medium truncate">{sourceImage.name}</p>
                <p className="text-[10px] text-muted-foreground">
                  {isFillModel ? t('image.prompt.sourceInpainting') : t('image.prompt.sourceInstructionEdit')}
                  {isFillModel && maskBase64 && <span className="text-violet-400 ml-1">{t('image.prompt.maskApplied')}</span>}
                </p>
              </div>
              <div className="flex items-center gap-1 flex-shrink-0">
                {isFillModel && (
                  <button
                    onClick={() => setShowMaskPainter(true)}
                    className={`px-2 py-1 text-xs rounded flex items-center gap-1 transition-colors ${
                      maskBase64
                        ? 'bg-violet-500/15 text-violet-400 hover:bg-violet-500/25'
                        : 'bg-primary/10 text-primary hover:bg-primary/20'
                    }`}
                    title={t('image.prompt.maskTitle')}
                  >
                    <Paintbrush className="h-3 w-3" />
                    {maskBase64 ? t('image.prompt.editMask') : t('image.prompt.paintMask')}
                  </button>
                )}
                <button onClick={() => onSourceImageChange(null)} className="p-1 text-muted-foreground hover:text-destructive rounded" title={t('image.prompt.removeSourceTitle')}>
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          ) : (
            <div onDragOver={(e) => { e.preventDefault(); setDragOver(true) }} onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop} onClick={handlePickImage}
              className={`flex items-center justify-center gap-2 p-3 border-2 border-dashed rounded-md cursor-pointer transition-colors ${
                dragOver ? 'border-violet-500 bg-violet-500/5' : 'border-border hover:border-violet-500/40'
              }`}>
              <ImagePlus className="h-4 w-4 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">{t('image.promptBar.uploadSourceImage')}</span>
            </div>
          )}
        </div>
      ) : null}

      {/* Quick settings row */}
      <div className="flex items-center gap-4 mb-2 text-xs">
        <div className="flex items-center gap-1">
          <label className="text-muted-foreground">{t('image.prompt.steps')}</label>
          <Help tip={t('image.prompt.stepsTip')} />
          <input type="number" value={settings.steps}
            onChange={(e) => onSettingsChange({ ...settings, steps: Math.max(1, Math.min(100, parseInt(e.target.value) || 1)) })}
            className="w-14 px-1.5 py-0.5 bg-muted border border-input rounded text-xs text-center focus:outline-none focus:ring-1 focus:ring-ring"
            min={1} max={100} />
        </div>
        <div className="flex items-center gap-1">
          <label className="text-muted-foreground">{t('image.prompt.size')}</label>
          <Help tip={t('image.prompt.sizeTip')} />
          <select value={sizeLabel}
            onChange={(e) => { const preset = SIZE_PRESETS.find(p => p.label === e.target.value); if (preset) onSettingsChange({ ...settings, width: preset.width, height: preset.height }) }}
            className="px-1.5 py-0.5 bg-muted border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring">
            {SIZE_PRESETS.map((p) => (<option key={p.label} value={p.label}>{p.label}</option>))}
          </select>
        </div>
        <div className="flex items-center gap-1">
          <label className="text-muted-foreground">{t('image.prompt.guidance')}</label>
          <Help tip={t('image.prompt.guidanceTip')} />
          <input type="number" value={settings.guidance} data-vmlx-control="image-guidance-quick"
            disabled={capabilities?.guidance === false}
            title={capabilities?.guidance === false ? t('image.settings.guidanceUnused') : undefined}
            onChange={(e) => onSettingsChange({ ...settings, guidance: imageGuidanceFromInput(e.target.value) })}
            className="w-14 px-1.5 py-0.5 bg-muted border border-input rounded text-xs text-center focus:outline-none focus:ring-1 focus:ring-ring"
            min={0} step={0.5} />
        </div>
        {/* Strength: edit mode / variation mode, and only when the loaded model takes it */}
        {((isEdit && capabilities?.edit_strength !== false) || (isVariation && capabilities?.variation_strength !== false)) && (
          <div className="flex items-center gap-1">
            <label className="text-muted-foreground">{t(preservesSource ? 'image.prompt.sourcePreservation' : 'image.prompt.strength')}</label>
            <Help tip={preservesSource ? t('image.prompt.sourcePreservationTip') : isEdit
              ? t('image.prompt.strengthEditTip')
              : t('image.prompt.strengthVariationTip')
            } />
            <input type="number" value={settings.strength}
              onChange={(e) => onSettingsChange({ ...settings, strength: Math.max(0, Math.min(1, parseFloat(e.target.value) || 0)) })}
              className="w-14 px-1.5 py-0.5 bg-muted border border-input rounded text-xs text-center focus:outline-none focus:ring-1 focus:ring-ring"
              min={0} max={1} step={0.05} />
          </div>
        )}
        {!isEdit && !isVariation && settings.count > 1 && (
          <span className="text-muted-foreground">{t('image.prompt.imageCount', { n: settings.count })}</span>
        )}
      </div>

      {/* Prompt input + submit */}
      <div className="flex gap-2">
        <textarea ref={textareaRef} value={prompt}
          onChange={(e) => setPrompt(e.target.value)} onKeyDown={handleKeyDown} onPaste={handlePaste}
          placeholder={
            disabled ? t('image.prompt.placeholderWaiting')
              : isVariation ? t('image.prompt.placeholderVariation')
              : isEdit && !sourceImage ? t('image.prompt.placeholderNeedSource')
              : needsMask && !maskBase64 ? t('image.prompt.placeholderNeedMask')
              : isEdit ? t('image.prompt.placeholderEdit')
              : t('image.prompt.placeholderGenerate')
          }
          disabled={disabled || generating} rows={2}
          className="flex-1 px-3 py-2 bg-muted border border-input rounded-md text-sm resize-none focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50 placeholder:text-muted-foreground/60"
        />
        {generating ? (
          <button disabled={cancelling || cancelRequested} onClick={async () => {
            setCancelRequested(true)
            try { await window.api.image.cancelGeneration() } catch { setCancelRequested(false) }
          }} data-vmlx-control="image-cancel" data-vmlx-state={cancelling || cancelRequested ? 'cancelling' : 'running'}
            className="px-4 py-2 rounded-md bg-destructive text-destructive-foreground hover:bg-destructive/90 flex items-center gap-2 self-end">
            <X className="h-4 w-4" /><span className="text-sm">{t(cancelling || cancelRequested ? 'image.prompt.cancelling' : 'image.prompt.cancel')}</span>
          </button>
        ) : (
          <button onClick={handleSubmit} disabled={!canSubmit} data-vmlx-control="image-generate"
            className={`px-4 py-2 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 self-end ${
              isVariation ? 'bg-emerald-600 text-white hover:bg-emerald-700'
                : isEdit ? 'bg-violet-600 text-white hover:bg-violet-700'
                : 'bg-primary text-primary-foreground hover:bg-primary/90'
            }`}>
            {isVariation ? <RefreshCw className="h-4 w-4" /> : isEdit ? <Pencil className="h-4 w-4" /> : <Send className="h-4 w-4" />}
            <span className="text-sm">{isVariation ? t('image.prompt.vary') : isEdit ? t('image.prompt.edit') : t('image.prompt.generate')}</span>
          </button>
        )}
      </div>

      {/* Prompt preview for variation */}
      {isVariation && (
        <p className="mt-1.5 text-[10px] text-muted-foreground truncate">
          {prompt.trim()
            ? <>{t('image.prompt.promptLabel')} <span className="text-foreground/70">{iteratePrompt}, {prompt.trim()}</span></>
            : <>{t('image.promptBar.variationOf')} <span className="text-foreground/70">{iteratePrompt}</span> {t('image.prompt.variationNewSeed')}</>
          }
        </p>
      )}
    </div>
  )
}
