import { useState, useEffect, useRef } from 'react'
import { Maximize2, Loader2 } from 'lucide-react'

interface ActiveDownload {
  jobId: string
  repoId: string
  progress?: { percent?: number; speed?: string; eta?: string; downloaded?: string; total?: string; filesProgress?: string }
  error?: string
}

interface DownloadStatusBarProps {
  onComplete?: () => void
}

export function DownloadStatusBar({ onComplete }: DownloadStatusBarProps) {
  const [activeDownloads, setActiveDownloads] = useState<ActiveDownload[]>([])
  const [queueCount, setQueueCount] = useState(0)
  const onCompleteRef = useRef(onComplete)
  onCompleteRef.current = onComplete

  // Open download window via custom event
  useEffect(() => {
    const handler = () => window.api.models.openDownloadWindow()
    window.addEventListener('open-download-popup', handler)
    return () => window.removeEventListener('open-download-popup', handler)
  }, [])

  const refreshStatus = () => {
    window.api.models.getDownloadStatus().then((status: any) => {
      if (status.activeAll) {
        setActiveDownloads(status.activeAll)
      } else if (status.active) {
        setActiveDownloads([status.active])
      } else {
        setActiveDownloads([])
      }
      setQueueCount(status.queue?.length || 0)
    }).catch(() => {})
  }

  useEffect(() => {
    refreshStatus()

    const unsubProgress = window.api.models.onDownloadProgress((data: any) => {
      setActiveDownloads(prev => prev.map(d =>
        d.jobId === data.jobId ? { ...d, progress: data.progress } : d
      ))
    })
    const unsubComplete = window.api.models.onDownloadComplete((data: any) => {
      if (data.status === 'complete') onCompleteRef.current?.()
      setActiveDownloads(prev => prev.filter(d => d.jobId !== data.jobId))
      setTimeout(refreshStatus, 500)
    })
    const unsubError = window.api.models.onDownloadError((data: any) => {
      setActiveDownloads(prev => prev.map(d =>
        d.jobId === data.jobId ? { ...d, error: data.error } : d
      ))
      setTimeout(() => {
        setActiveDownloads(prev => prev.filter(d => d.jobId !== data.jobId))
        refreshStatus()
      }, 5000)
    })
    const unsubStart = window.api.models.onDownloadStarted?.((data: any) => {
      setActiveDownloads(prev => {
        if (prev.some(d => d.jobId === data.jobId)) return prev
        return [...prev, { jobId: data.jobId, repoId: data.repoId }]
      })
      // Download moved from queue to active — decrement queue count
      setQueueCount(prev => Math.max(0, prev - 1))
      window.api.models.openDownloadWindow()
    })
    const unsubQueued = window.api.models.onDownloadQueued?.((data: any) => {
      setQueueCount(prev => prev + 1)
    })

    return () => {
      unsubProgress()
      unsubComplete()
      unsubError()
      unsubStart?.()
      unsubQueued?.()
    }
  }, [])

  if (activeDownloads.length === 0 && queueCount === 0) return null

  const shortName = (repoId: string) => repoId.includes('/') ? repoId.split('/').pop() : repoId
  // Show the first active download in the inline bar
  const primary = activeDownloads[0]
  const p = primary?.progress

  return (
    <div className="bg-card border-b border-border px-3 py-1.5 flex-shrink-0">
      <div className="flex items-center gap-2 text-xs">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {primary?.error ? (
              <span className="w-1.5 h-1.5 bg-destructive rounded-full flex-shrink-0" />
            ) : (
              <Loader2 className="h-3 w-3 text-primary animate-spin flex-shrink-0" />
            )}
            <span className={`truncate font-medium ${primary?.error ? 'text-destructive' : ''}`}>
              {primary?.error
                ? `Failed: ${shortName(primary.repoId)}`
                : `Downloading ${shortName(primary?.repoId || '')}`}
            </span>
            {p?.percent != null && <span className="text-muted-foreground">{p.percent}%</span>}
            {p?.speed && <span className="text-muted-foreground">{p.speed}</span>}
            {p?.eta && <span className="text-muted-foreground">{p.eta}</span>}
            {!p && !primary?.error && <span className="text-muted-foreground">Starting...</span>}
            {activeDownloads.length > 1 && <span className="text-muted-foreground">+{activeDownloads.length - 1} more</span>}
            {queueCount > 0 && <span className="text-muted-foreground">+{queueCount} queued</span>}
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
      </div>
    </div>
  )
}
