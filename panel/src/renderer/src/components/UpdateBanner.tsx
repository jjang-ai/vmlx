import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import { useTranslation } from '../i18n'

interface UpdateInfo {
  currentVersion: string
  latestVersion: string
  url: string
  notes?: string
}

export function UpdateBanner() {
  const { t } = useTranslation()
  const [update, setUpdate] = useState<UpdateInfo | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const unsub = window.api.app.onUpdateAvailable((data: UpdateInfo) => {
      const prev = localStorage.getItem('vmlx-dismissed-update')
      if (prev !== data.latestVersion) {
        setUpdate(data)
        setDismissed(false)
      }
    })
    return unsub
  }, [])

  const handleDismiss = () => {
    setDismissed(true)
    if (update) localStorage.setItem('vmlx-dismissed-update', update.latestVersion)
  }

  if (!update || dismissed) return null

  return (
    <div role="status" aria-live="polite" data-vmlx-section="release-update"
      className="shrink-0 flex flex-wrap items-center gap-x-3 gap-y-1 px-4 py-1.5 bg-accent/50 border-b border-border text-xs">
      <span className="text-foreground min-w-0 flex-1 basis-full sm:basis-auto [overflow-wrap:anywhere]">
        <strong>{t('update.banner.versionLabel', { version: update.latestVersion })}</strong>{' '}
        {t('update.banner.available')}
        {update.notes && <span className="text-muted-foreground ml-1">— {update.notes}</span>}
      </span>
      <a
        data-vmlx-control="release-update-download"
        href="#"
        onClick={(e) => {
          e.preventDefault()
          window.open(update.url)
        }}
        className="shrink-0 text-primary hover:text-primary/80 font-medium"
      >
        {t('update.banner.download')}
      </a>
      <span className="text-muted-foreground">|</span>
      <a
        href="#"
        onClick={(e) => {
          e.preventDefault()
          window.open('https://github.com/jjang-ai/mlxstudio')
        }}
        className="shrink-0 text-muted-foreground hover:text-foreground"
      >
        {t('update.banner.starOnGitHub')}
      </a>
      <button
        type="button" data-vmlx-control="release-update-dismiss"
        onClick={handleDismiss}
        className="ml-auto shrink-0 text-muted-foreground hover:text-foreground"
        title={t('update.banner.dismissTitle')}
        aria-label={t('update.banner.dismissTitle')}
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}
