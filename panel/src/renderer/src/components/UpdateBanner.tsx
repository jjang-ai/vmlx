import { useState, useEffect } from 'react'
import { X } from 'lucide-react'

interface UpdateInfo {
  currentVersion: string
  latestVersion: string
  url: string
  notes?: string
}

export function UpdateBanner() {
  const [update, setUpdate] = useState<UpdateInfo | null>(null)
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    const unsub = window.api.app.onUpdateAvailable((data: UpdateInfo) => {
      setUpdate(data)
    })
    return unsub
  }, [])

  if (!update || dismissed) return null

  return (
    <div className="flex items-center gap-3 px-4 py-1.5 bg-accent/50 border-b border-border text-xs">
      <span className="text-foreground">
        <strong>vMLX {update.latestVersion}</strong> is available
        {update.notes && <span className="text-muted-foreground ml-1">— {update.notes}</span>}
      </span>
      <a
        href="#"
        onClick={(e) => {
          e.preventDefault()
          window.open(update.url)
        }}
        className="text-primary hover:text-primary/80 font-medium"
      >
        Download
      </a>
      <span className="text-muted-foreground">|</span>
      <a
        href="#"
        onClick={(e) => {
          e.preventDefault()
          window.open('https://github.com/jjang-ai/mlxstudio')
        }}
        className="text-muted-foreground hover:text-foreground"
      >
        Star on GitHub
      </a>
      <button
        onClick={() => setDismissed(true)}
        className="ml-auto text-muted-foreground hover:text-foreground"
        title="Dismiss"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </div>
  )
}
