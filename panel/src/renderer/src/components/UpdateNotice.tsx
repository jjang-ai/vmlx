import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import { useTranslation } from '../i18n'

const CURRENT_NOTICE_VERSION = '1.5.45'

export function UpdateNotice() {
  const [visible, setVisible] = useState(false)
  const { t } = useTranslation()

  const noticeSections = [
    {
      heading: t('update.notice.section1Heading'),
      lines: [t('update.notice.section1BodyA'), t('update.notice.section1BodyB')],
    },
    {
      heading: t('update.notice.section2Heading'),
      lines: [t('update.notice.section2BodyA'), t('update.notice.section2BodyB')],
    },
    {
      heading: t('update.notice.section3Heading'),
      lines: [t('update.notice.section3BodyA'), t('update.notice.section3BodyB')],
    },
    {
      heading: t('update.notice.section4Heading'),
      lines: [t('update.notice.section4BodyA'), t('update.notice.section4BodyB')],
    },
  ]

  useEffect(() => {
    const dismissed = window.api.settings?.get('notice_dismissed_version')
    if (dismissed instanceof Promise) {
      dismissed.then((v: any) => {
        if (v !== CURRENT_NOTICE_VERSION) setVisible(true)
      }).catch(() => setVisible(true))
    } else {
      if (dismissed !== CURRENT_NOTICE_VERSION) setVisible(true)
    }
  }, [])

  const dismiss = () => {
    setVisible(false)
    try {
      window.api.settings?.set('notice_dismissed_version', CURRENT_NOTICE_VERSION)
    } catch {}
  }

  if (!visible) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
      <div className="bg-card border border-border rounded-xl shadow-2xl max-w-md w-full mx-4 p-5">
        <div className="flex items-start justify-between mb-3">
          <h2 className="text-sm font-bold">{t('update.notice.title')}</h2>
          <button onClick={dismiss} className="p-0.5 text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="space-y-3 text-xs text-muted-foreground leading-relaxed max-h-[60vh] overflow-y-auto">
          {noticeSections.map((section, si) => (
            <div key={si}>
              <p className="text-[11px] font-semibold text-foreground/80 mb-1">{section.heading}</p>
              {section.lines.map((line, li) => (
                <p key={li} className="ml-2">{line}</p>
              ))}
            </div>
          ))}
        </div>
        <p className="text-[10px] text-muted-foreground/60 mt-3 italic">{t('update.notice.footer')}</p>
        <button
          onClick={dismiss}
          className="mt-3 w-full py-1.5 bg-primary text-primary-foreground text-xs font-medium rounded-lg hover:bg-primary/90"
        >
          {t('update.notice.dismiss')}
        </button>
      </div>
    </div>
  )
}
