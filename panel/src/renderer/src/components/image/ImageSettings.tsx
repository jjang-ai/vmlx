import { useState } from 'react'
import { ChevronDown, ChevronRight, HelpCircle } from 'lucide-react'
import { useTranslation } from '../../i18n'
import type { ImageCapabilities } from '../../../../shared/imageCapabilities'

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
  /** From /health.image of the running server; null while unknown. */
  capabilities?: ImageCapabilities | null
}

const SIZE_PRESETS = [
  { label: '512x512', width: 512, height: 512 },
  { label: '768x768', width: 768, height: 768 },
  { label: '1024x1024', width: 1024, height: 1024 },
  { label: '1024x768 (Landscape)', width: 1024, height: 768 },
  { label: '768x1024 (Portrait)', width: 768, height: 1024 },
  { label: '1280x720 (16:9)', width: 1280, height: 720 },
]

export function ImageSettings({ settings, onChange, model, mode, capabilities }: ImageSettingsProps) {
  const { t } = useTranslation()
  const isEdit = mode === 'edit'
  // Only claim a control is effective when the loaded model takes it.
  const strengthUnused = isEdit && capabilities?.edit_strength === false
  const negativeUnused = capabilities?.negative_prompt === false
  const capModel = capabilities?.mflux_class || model || ''
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
        {isEdit ? t('image.settings.editSettings') : t('image.settings.generationSettings')}
      </h3>

      {/* Standard Settings */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3">
        {/* Steps */}
        <div>
          <label className="text-xs text-muted-foreground block mb-1" title={t('image.settings.stepsTitle')}>{t('image.settings.steps')} &#9432;</label>
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
          <label className="text-xs text-muted-foreground block mb-1" title={t('image.settings.sizeTitle')}>{t('image.settings.size')} &#9432;</label>
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
          <label className="text-xs text-muted-foreground block mb-1" title={t('image.settings.guidanceTitle')}>{t('image.settings.guidance')} &#9432;</label>
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
          <label className="text-xs text-muted-foreground block mb-1" title={t('image.settings.seedTitle')}>{t('image.settings.seed')} &#9432;</label>
          <input
            type="number"
            value={settings.seed ?? ''}
            onChange={(e) => {
              const val = e.target.value.trim()
              update('seed', val ? parseInt(val) : undefined)
            }}
            placeholder={t('image.settings.seedPlaceholder')}
            className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring"
          />
        </div>

        {/* Strength (edit mode only; hidden with a note when the loaded edit class never uses it) */}
        {strengthUnused && (
          <div data-vmlx-image-cap="strength-unused" className="text-[10px] text-muted-foreground self-end pb-1">
            {t('image.settings.strengthNotUsed', { model: capModel })}
          </div>
        )}
        {isEdit && !strengthUnused && (
          <div>
            <label className="text-xs text-muted-foreground block mb-1" title={t('image.settings.strengthTitle')}>{t('image.settings.strength')} &#9432;</label>
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
            <label className="text-xs text-muted-foreground block mb-1" title={t('image.settings.countTitle')}>{t('image.settings.count')} &#9432;</label>
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
          <label className="text-xs text-muted-foreground block mb-1" title={t('image.settings.quantizeTitle')}>{t('image.settings.quantize')} &#9432;</label>
          <div className="w-full px-2 py-1 bg-muted/50 border border-input rounded text-xs text-muted-foreground">
            {settings.quantize === 0 ? t('sessions.create.quantFullPrecision') : `${settings.quantize}-bit`}
          </div>
        </div>
      </div>

      {/* Negative Prompt */}
      <div className="mb-3">
        <div className="flex items-center gap-1 mb-1">
          <label className="text-xs text-muted-foreground">{t('image.settings.negativePrompt')}</label>
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
            {t('image.settings.negativeHelpBody')}{' '}
            <em>{t('image.settings.negativeHelpExample')}</em>
          </p>
        )}
        <input
          type="text"
          value={settings.negativePrompt}
          onChange={(e) => update('negativePrompt', e.target.value)}
          placeholder={negativeUnused ? t('image.settings.negativeNotUsed', { model: capModel }) : t('image.settings.negativePlaceholder')}
          disabled={negativeUnused}
          data-vmlx-image-cap={negativeUnused ? 'negative-unused' : 'negative'}
          className="w-full px-2 py-1 bg-background border border-input rounded text-xs focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
        />
        {negativeUnused && (
          <p className="text-[10px] text-muted-foreground mt-1">{t('image.settings.negativeNotUsed', { model: capModel })}</p>
        )}
      </div>

      {/* Advanced Section (collapsed) */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors mb-2"
      >
        {showAdvanced ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        {t('image.settings.advanced')}
      </button>
      {showAdvanced && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-3 pl-4 border-l border-border">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">{t('image.settings.widthCustom')}</label>
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
            <label className="text-xs text-muted-foreground block mb-1">{t('image.settings.heightCustom')}</label>
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
        {t('image.settings.server')}
      </button>
      {showServer && (
        <div className="pl-4 border-l border-border text-xs text-muted-foreground space-y-1">
          <p>{t('image.settings.hostLocalhost')}</p>
          <p>{t('image.settings.portAutoAssigned')}</p>
          <p>{t('image.settings.modelLine', { model: model || 'none' })}</p>
          {capabilities && (
            <div data-vmlx-image-cap="summary" className="mt-1 space-y-0.5">
              <p>{t('image.settings.capClass', { cls: capabilities.mflux_class || '?', mode: capabilities.mode || '?' })}</p>
              <p>{t('image.settings.capNegativePrompt')}: {capabilities.negative_prompt ? t('image.settings.capYes') : t('image.settings.capNo')}</p>
              <p>{t('image.settings.capStrength')}: {capabilities.mode === 'edit' ? (capabilities.edit_strength ? t('image.settings.capYes') : t('image.settings.capNo')) : (capabilities.variation_strength ? t('image.settings.capVariationOnly') : t('image.settings.capNo'))}</p>
              <p>{t('image.settings.capMask')}: {capabilities.mask === 'required' ? t('image.settings.capRequired') : t('image.settings.capNo')}</p>
              <p>{t('image.settings.capCount')}: {capabilities.count ? t('image.settings.capYes') : t('image.settings.capSingle')}</p>
            </div>
          )}
          <p className="text-[10px] mt-2 opacity-70">
            {t('image.settings.serverManagedAutomatically')}
          </p>
        </div>
      )}
    </div>
  )
}
