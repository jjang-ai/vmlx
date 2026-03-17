import { useState, useEffect, useRef } from 'react'
import { Maximize2, Download, Loader2 } from 'lucide-react'

interface DownloadProgress {
  percent: number
  speed: string
  downloaded: string
  total: string
  eta: string
  currentFile: string
  filesProgress: string
  raw: string
}

interface ActiveDownload {
  jobId: string
  repoId: string
  progress?: DownloadProgress
  error?: string
}

interface DownloadStatusBarProps {
  onComplete?: () => void
}

export function DownloadStatusBar({ onComplete }: DownloadStatusBarProps) {
  const [active, setActive] = useState<ActiveDownload | null>(null)
  const [queue, setQueue] = useState<Array<{ jobId: string; repoId: string }>>([])
  const [_expanded, _setExpanded] = useState(false) // kept for event compat
  const onCompleteRef = useRef(onComplete)
  onCompleteRef.current = onComplete

  // Allow other components to open the download window via custom event
  useEffect(() => {
    const handler = () => window.api.models.openDownloadWindow()
    window.addEventListener('open-download-popup', handler)
    return () => window.removeEventListener('open-download-popup', handler)
  }, [])

  useEffect(() => {
    // Poll initial status
    window.api.models.getDownloadStatus().then((status: any) => {
      if (status.active) setActive(status.active)
      setQueue(status.queue || [])
    }).catch(() => {})

    const unsubProgress = window.api.models.onDownloadProgress((data: any) => {
      setActive(prev => prev && prev.jobId === data.jobId ? { ...prev, progress: data.progress } : prev)
    })
    const unsubComplete = window.api.models.onDownloadComplete((data: any) => {
      if (data.status === 'complete') onCompleteRef.current?.()
      setActive(prev => prev?.jobId === data.jobId ? null : prev)
      window.api.models.getDownloadStatus().then((status: any) => {
        if (status.active) setActive(status.active)
        else setActive(null)
        setQueue(status.queue || [])
      }).catch(() => {})
    })
    const unsubError = window.api.models.onDownloadError((data: any) => {
      setActive(prev => {
        if (prev && prev.jobId === data.jobId) {
          return { ...prev, error: data.error || 'Download failed', progress: undefined }
        }
        return prev
      })
      setTimeout(() => {
        setActive(prev => prev?.jobId === data.jobId ? null : prev)
        window.api.models.getDownloadStatus().then((status: any) => {
          if (status.active) setActive(status.active)
          else setActive(null)
          setQueue(status.queue || [])
        }).catch(() => {})
      }, 5000)
    })
    const unsubStart = window.api.models.onDownloadStarted?.((data: any) => {
      setActive({ jobId: data.jobId, repoId: data.repoId })
      // Auto-open the downloads window when a download starts
      window.api.models.openDownloadWindow()
    })

    return () => {
      unsubProgress()
      unsubComplete()
      unsubError()
      unsubStart?.()
    }
  }, [])

  if (!active && queue.length === 0) return null

  const shortName = (repoId: string) => repoId.includes('/') ? repoId.split('/').pop() : repoId
  const p = active?.progress

  // Inline bar (always visible when downloading)
  const inlineBar = (
    <div className="bg-card border-b border-border px-3 py-1.5 flex-shrink-0">
      <div className="flex items-center gap-2 text-xs">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {active?.error ? (
              <span className="w-1.5 h-1.5 bg-destructive rounded-full flex-shrink-0" />
            ) : (
              <Loader2 className="h-3 w-3 text-primary animate-spin flex-shrink-0" />
            )}
            <span className={`truncate font-medium ${active?.error ? 'text-destructive' : ''}`}>
              {active?.error ? `Failed: ${shortName(active.repoId)}` : `Downloading ${shortName(active?.repoId || '')}`}
            </span>
            {p?.percent != null && <span className="text-muted-foreground">{p.percent}%</span>}
            {p?.speed && <span className="text-muted-foreground">{p.speed}</span>}
            {p?.eta && <span className="text-muted-foreground">ETA {p.eta}</span>}
            {!p && !active?.error && <span className="text-muted-foreground">Starting...</span>}
            {queue.length > 0 && <span className="text-muted-foreground">+{queue.length} queued</span>}
          </div>
          {p?.percent != null && (
            <div className="mt-0.5 h-1 bg-muted rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full transition-all duration-300" style={{ width: `${Math.min(p.percent, 100)}%` }} />
            </div>
          )}
        </div>
        <button onClick={() => window.api.models.openDownloadWindow()} className="p-1 text-muted-foreground hover:text-foreground" title="Open Downloads window">
          <Maximize2 className="h-3 w-3" />
        </button>
        {active && !active.error && (
          <button onClick={() => window.api.models.cancelDownload(active.jobId)} className="text-[10px] text-destructive hover:text-destructive/80 px-1.5 py-0.5 border border-destructive/30 rounded flex-shrink-0">
            Cancel
          </button>
        )}
      </div>
    </div>
  )

  return inlineBar
}
