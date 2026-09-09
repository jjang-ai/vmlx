import { useState, useEffect } from 'react'
import { FolderOpen, Play, Download } from 'lucide-react'
import { canLaunchInspectedImageFolder } from '../../../../shared/imageFolderLaunch'
import type { inspectLocalImageModel } from '../../../../shared/imageLocalModel'
import { IMAGE_MODEL_DISCOVERY_NAVIGATION } from '../../../../shared/modelDiscoveryNavigation'
import { useTranslation } from '../../i18n'

export interface ImageServerSettings {
  host: string
  port: number
  apiKey: string
  mfluxClass?: string
  logLevel: string
}

interface ImageModelPickerProps {
  onSelect: (modelId: string, quantize?: number, category?: 'generate' | 'edit', serverSettings?: ImageServerSettings) => void
  /** Display name of the model whose server is still running while this picker is open. */
  currentModel?: string | null
  /** Close the picker and keep that running server. */
  onKeepCurrent?: () => void
}

export function ImageModelPicker({ onSelect, currentModel, onKeepCurrent }: ImageModelPickerProps) {
  const { t } = useTranslation()
  const [customPath, setCustomPath] = useState('')
  const [customCategory, setCustomCategory] = useState<'' | 'generate' | 'edit'>('')
  const [customMfluxClass, setCustomMfluxClass] = useState('')
  const [localPreview, setLocalPreview] = useState<ReturnType<typeof inspectLocalImageModel> | null>(null)
  const [inspecting, setInspecting] = useState(false)
  const [previewInput, setPreviewInput] = useState('')

  const chooseCustomPath = (path: string) => {
    setCustomPath(path)
    setLocalPreview(null)
    setPreviewInput('')
    setInspecting(!!path.trim())
    // Overrides describe one folder, never the next selection.
    setCustomCategory('')
    setCustomMfluxClass('')
  }
  const canLoadFolder = canLaunchInspectedImageFolder({
    input: customPath, inspectedInput: previewInput, inspecting,
    success: !!localPreview?.success,
    detected: !!(localPreview?.success && localPreview.model),
    explicitClass: customMfluxClass, explicitTask: customCategory,
  })

  useEffect(() => {
    setLocalPreview(null)
    if (!customPath.trim()) { setInspecting(false); return }
    let cancelled = false
    setInspecting(true)
    const timer = setTimeout(() => {
      window.api.image.inspectLocalModel(customPath.trim()).then(result => {
        if (!cancelled) { setLocalPreview(result); setPreviewInput(customPath.trim()) }
      }).catch(error => {
        if (!cancelled) setLocalPreview({ success: false, error: String(error) })
      }).finally(() => { if (!cancelled) setInspecting(false) })
    }, 250)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [customPath])

  // Server settings (same as Server tab CreateSession simplified config)
  const [serverHost, setServerHost] = useState('127.0.0.1')
  const [serverPort, setServerPort] = useState(0) // 0 = auto
  const [serverApiKey, setServerApiKey] = useState('')
  const [serverLogLevel, setServerLogLevel] = useState('INFO')


  const handleStart = () => {
    if (!canLoadFolder || !localPreview?.success) return
    const settings: ImageServerSettings = {
      host: serverHost, port: serverPort, apiKey: serverApiKey,
      logLevel: serverLogLevel, mfluxClass: customMfluxClass || undefined,
    }
    onSelect(localPreview.path, localPreview.quantize, customCategory || localPreview.model?.category, settings)
  }

  const handleBrowse = async () => {
    try {
      const result = await window.api.models.browseDirectory()
      if (result?.path) chooseCustomPath(result.path)
    } catch {}
  }

  return (
    <div data-vmlx-section="image-model-picker" className="h-full min-h-0 min-w-0 overflow-auto p-4 sm:p-8">
      <div className="max-w-3xl w-full mx-auto space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">{t('image.picker.chooseFolderTitle')}</h2>
          <p className="text-sm text-muted-foreground">{t('image.picker.folderFirstIntro')}</p>
          {currentModel && onKeepCurrent && (
            <button type="button" onClick={onKeepCurrent} data-vmlx-control="image-keep-current-model"
              className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 border border-border text-sm hover:bg-accent">
              {t('image.picker.keepCurrent', { model: currentModel })}
            </button>
          )}
          <p data-vmlx-status="mflux-compatibility" className="mt-3 text-xs text-muted-foreground">
            {t('image.picker.mfluxCompatibility')}
          </p>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 border border-border p-4">
          <p className="text-sm text-muted-foreground">{t('image.picker.findMfluxHelp')}</p>
          <button type="button" data-vmlx-control="image-source-catalog"
            onClick={() => window.dispatchEvent(new CustomEvent('vmlx:navigate', { detail: IMAGE_MODEL_DISCOVERY_NAVIGATION }))}
            className="inline-flex shrink-0 items-center gap-2 px-3 py-2 border border-border text-sm hover:bg-accent">
            <Download className="h-4 w-4" />{t('image.picker.findMfluxModels')}
          </button>
        </div>
        {/* Custom Model */}
        <div className="border border-border p-4 min-w-0" data-vmlx-section="image-folder-selection">
          <h3 className="text-sm font-medium">{t('image.picker.folderTab')}</h3>
            <div className="mt-3 space-y-2">
              <div className="flex gap-2">
                <input
                  type="text"
                  value={customPath}
                  onChange={e => chooseCustomPath(e.target.value)}
                  data-vmlx-control="image-folder-path"
                  placeholder={t('image.picker.customPathPlaceholder')}
                  className="min-w-0 flex-1 px-3 py-2 text-sm bg-background border border-input"
                />
                <button
                  onClick={handleBrowse}
                  data-vmlx-control="image-browse-folder"
                  className="px-3 py-2 text-sm border border-input rounded hover:bg-accent"
                  title={t('image.picker.browseTitle')}
                >
                  <FolderOpen className="h-4 w-4" />
                </button>
              </div>
              <details className="border-t border-border pt-3">
                <summary className="cursor-pointer text-xs text-muted-foreground">{t('image.picker.adapterOverride')}</summary>
                <p className="text-xs text-muted-foreground my-2">{t('image.picker.overrideHelp')}</p>
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground">{t('image.picker.modeLabel')}</label>
                  <select
                    value={customCategory}
                    onChange={e => setCustomCategory(e.target.value as '' | 'generate' | 'edit')}
                    className="px-2 py-1 text-xs bg-background border border-input rounded"
                  >
                    <option value="">{t('image.picker.automatic')}</option>
                    <option value="generate">{t('image.picker.imageGeneration')}</option>
                    <option value="edit">{t('image.picker.imageEditing')}</option>
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-muted-foreground" title={t('image.picker.classTitle')}>{t('image.picker.classLabel')}</label>
                  <select
                    value={customMfluxClass}
                    onChange={e => setCustomMfluxClass(e.target.value)}
                    className="px-2 py-1 text-xs bg-background border border-input rounded"
                  >
                    <option value="">{t('image.picker.automatic')}</option>
                    <option value="Flux1">{t('image.picker.classFlux1')}</option>
                    <option value="ZImage">{t('image.picker.classZImage')}</option>
                    <option value="Flux2Klein">{t('image.picker.classFlux2Klein')}</option>
                    <option value="QwenImage">{t('image.picker.classQwenImage')}</option>
                    <option value="QwenImageEdit">{t('image.picker.classQwenImageEdit')}</option>
                    <option value="Flux1Kontext">{t('image.picker.classFlux1Kontext')}</option>
                    <option value="Flux1Fill">{t('image.picker.classFlux1Fill')}</option>
                    <option value="Flux2KleinEdit">{t('image.picker.classFlux2KleinEdit')}</option>
                    <option value="FIBO">{t('image.picker.classFIBO')}</option>
                    <option value="SeedVR2">{t('image.picker.classSeedVR2')}</option>
                  </select>
                </div>
              </div>
              </details>
              <div role="status" data-vmlx-control="image-folder-detection" className="text-xs text-muted-foreground break-words">
                {inspecting ? t('image.picker.inspectingFolder') : localPreview?.success ? (
                  <>
                    <p>{localPreview.model ? `${localPreview.model.name} · ${localPreview.model.category} · ${localPreview.model.mfluxClass}` : t('image.picker.unknownArchitecture')}</p>
                    <p>{t('image.picker.folderPrecision')}: {localPreview.quantize ? `${localPreview.quantize}-bit` : t('image.topbar.quantFull')} ({localPreview.quantizeSource || t('image.picker.notDeclared')})</p>
                  </>
                ) : localPreview?.error}
              </div>
            </div>
        </div>


        {customPath.trim() && (
          <div className="flex flex-wrap items-end gap-4 p-4 bg-card border border-border">
            {/* Server Settings */}
            <div className="flex-1 min-w-[200px]">
              <label className="text-xs text-muted-foreground block mb-1.5">{t('image.picker.serverSettings')}</label>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-[10px] text-muted-foreground">{t('sessions.create.host')}</label>
                  <input type="text" value={serverHost} onChange={e => setServerHost(e.target.value)}
                    className="w-full px-2 py-1 bg-muted border border-input rounded text-xs" />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">{t('image.picker.portAuto')}</label>
                  <input type="number" value={serverPort} onChange={e => setServerPort(parseInt(e.target.value) || 0)}
                    className="w-full px-2 py-1 bg-muted border border-input rounded text-xs" min={0} max={65535} />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">{t('image.picker.apiKey')}</label>
                  <input type="password" value={serverApiKey} onChange={e => setServerApiKey(e.target.value)}
                    placeholder={t('image.picker.apiKeyPlaceholder')} className="w-full px-2 py-1 bg-muted border border-input rounded text-xs" />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground">{t('image.picker.logLevel')}</label>
                  <select value={serverLogLevel} onChange={e => setServerLogLevel(e.target.value)}
                    className="w-full px-2 py-1 bg-muted border border-input rounded text-xs">
                    <option value="DEBUG">DEBUG</option>
                    <option value="INFO">INFO</option>
                    <option value="WARNING">WARNING</option>
                    <option value="ERROR">ERROR</option>
                  </select>
                </div>
              </div>
            </div>


            <button onClick={handleStart} disabled={!canLoadFolder} data-vmlx-control="image-load-folder"
              className="px-6 py-3 bg-primary text-primary-foreground hover:bg-primary/90 flex items-center gap-2 font-medium text-sm disabled:opacity-50">
              <Play className="h-4 w-4" />{t('image.picker.loadFolder')}
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
