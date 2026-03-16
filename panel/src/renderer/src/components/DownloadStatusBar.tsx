import { useState, useEffect, useRef } from 'react'
import { X } from 'lucide-react'

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
}

interface QueuedDownload {
  jobId: string
  repoId: string
}

export function DownloadStatusBar({ onComplete }: { onComplete?: () => void }) {
  const [active, setActive] = useState<ActiveDownload | null>(null)
  const [queue, setQueue] = useState<QueuedDownload[]>([])
  const [collapsed, setCollapsed] = useState(false)
  const onCompleteRef = useRef(onComplete)
  onCompleteRef.current = onComplete

  // Poll initial status on mount
  useEffect(() => {
    window.api.models.getDownloadStatus().then((status: any) => {
      if (status.active) setActive(status.active)
      if (status.queue) setQueue(status.queue)
    }).catch((err) => console.error('Failed to get download status:', err))
  }, [])

  // Listen for progress updates
  useEffect(() => {
    const unsubProgress = window.api.models.onDownloadProgress((data: any) => {
      setActive(prev => prev?.jobId === data.jobId || !prev
        ? { jobId: data.jobId, repoId: data.repoId, progress: data.progress }
        : prev
      )
    })

    const unsubStarted = window.api.models.onDownloadStarted((data: any) => {
      setActive({ jobId: data.jobId, repoId: data.repoId })
      setQueue(prev => prev.filter(q => q.jobId !== data.jobId))
    })

    const unsubComplete = window.api.models.onDownloadComplete((data: any) => {
      if (data.status === 'complete') onCompleteRef.current?.()
      setActive(prev => prev?.jobId === data.jobId ? null : prev)
      // Re-poll to get updated queue
      window.api.models.getDownloadStatus().then((status: any) => {
        if (status.active) setActive(status.active)
        else setActive(null)
        setQueue(status.queue || [])
      }).catch((err) => console.error('Failed to refresh download status:', err))
    })

    const unsubError = window.api.models.onDownloadError((data: any) => {
      setActive(prev => prev?.jobId === data.jobId ? null : prev)
      window.api.models.getDownloadStatus().then((status: any) => {
        if (status.active) setActive(status.active)
        else setActive(null)
        setQueue(status.queue || [])
      }).catch((err) => console.error('Failed to refresh download status:', err))
    })

    return () => {
      unsubProgress()
      unsubStarted()
      unsubComplete()
      unsubError()
    }
  }, [])

  // Don't render if nothing is happening
  if (!active && queue.length === 0) return null

  const shortName = (repoId: string) => repoId.includes('/') ? repoId.split('/').pop() : repoId
  const p = active?.progress

  return (
    <div className="bg-card border-b border-border px-3 py-1.5 flex-shrink-0">
      {active && (
        <div className="flex items-center gap-2 text-xs">
          {/* Progress bar */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-primary rounded-full animate-pulse flex-shrink-0" />
              <span className="truncate font-medium" title={active.repoId}>
                {shortName(active.repoId)}
              </span>
              {p?.percent != null ? (
                <>
                  <span className="text-muted-foreground">{p.percent}%</span>
                  {p?.speed && <span className="text-muted-foreground">{p.speed}</span>}
                  {p?.eta && <span className="text-muted-foreground">ETA {p.eta}</span>}
                </>
              ) : p?.raw ? (
                <span className="text-muted-foreground truncate max-w-[200px]" title={p.raw}>
                  {p.raw.slice(0, 60)}
                </span>
              ) : (
                <span className="text-muted-foreground">Starting...</span>
              )}
              {queue.length > 0 && (
                <span className="text-muted-foreground">+{queue.length} queued</span>
              )}
            </div>
            {!collapsed && p?.percent != null && (
              <div className="mt-0.5 h-1 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(p.percent, 100)}%` }}
                />
              </div>
            )}
          </div>

          {/* Details toggle */}
          {p?.currentFile && !collapsed && (
            <span className="text-[10px] text-muted-foreground truncate max-w-[150px]" title={p.currentFile}>
              {p.currentFile}
            </span>
          )}

          {/* Collapse toggle */}
          <button
            onClick={() => setCollapsed(c => !c)}
            className="text-[10px] text-muted-foreground hover:text-foreground px-1 flex-shrink-0"
            title={collapsed ? 'Show details' : 'Hide details'}
          >
            {collapsed ? '\u25B8' : '\u25BE'}
          </button>
          {/* Cancel button */}
          <button
            onClick={() => window.api.models.cancelDownload(active.jobId)}
            className="text-[10px] text-destructive hover:text-destructive/80 px-1.5 py-0.5 border border-destructive/30 rounded flex-shrink-0"
          >
            Cancel
          </button>
        </div>
      )}

      {/* Queued items (show when no active, or expanded) */}
      {!active && queue.length > 0 && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="w-1.5 h-1.5 bg-warning rounded-full flex-shrink-0" />
          <span>{queue.length} download{queue.length > 1 ? 's' : ''} queued</span>
          {queue.map(q => (
            <span key={q.jobId} className="flex items-center gap-1">
              <span className="truncate max-w-[100px]">{shortName(q.repoId)}</span>
              <button
                onClick={() => window.api.models.cancelDownload(q.jobId)}
                className="text-destructive hover:text-destructive/80"
                title="Cancel"
              >
                <X className="h-3 w-3" />
              </button>
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
