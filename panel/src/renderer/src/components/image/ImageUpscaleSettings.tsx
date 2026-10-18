import { useTranslation } from '../../i18n'

interface UpscaleSettings {
  resolution?: number | string
  softness?: number
}

/** SeedVR2's native controls. Geometry is computed from the source aspect ratio. */
export function ImageUpscaleSettings<T extends UpscaleSettings>({ settings, onChange }: {
  settings: T; onChange: (settings: T) => void
}) {
  const { t } = useTranslation()
  return <div className="mb-2 space-y-2" data-vmlx-control="image-upscale-settings">
    <p className="text-xs text-muted-foreground">{t('image.settings.upscaleHelp')}</p>
    <div className="flex flex-wrap gap-4 text-xs">
      <label className="flex items-center gap-2">
        {t('image.settings.upscaleResolution')}
        <input type="text" value={settings.resolution ?? 384} data-vmlx-control="image-upscale-resolution"
          onChange={e => onChange({ ...settings, resolution: e.target.value })}
          className="w-24 px-2 py-1 bg-background border border-input rounded" />
      </label>
      <label className="flex items-center gap-2">
        {t('image.settings.upscaleSoftness')}
        <input type="number" min={0} max={1} step={0.05} value={settings.softness ?? 0} data-vmlx-control="image-upscale-softness"
          onChange={e => onChange({ ...settings, softness: Number(e.target.value) })}
          className="w-20 px-2 py-1 bg-background border border-input rounded" />
      </label>
    </div>
  </div>
}
