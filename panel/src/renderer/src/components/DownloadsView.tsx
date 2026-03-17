import { useState, useEffect, useRef } from 'react'
import { Download, Loader2, X, CheckCircle, AlertCircle } from 'lucide-react'

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

interface CompletedDownload {
  jobId: string
  repoId: string
  status: 'complete' | 'cancelled'
  time: number
}

export function DownloadsView() {
  const [active, setActive] = useState<ActiveDownload | null>(null)
  const [queue, setQueue] = useState<Array<{ jobId: string; repoId: string }>>([])
  const [completed, setCompleted] = useState<CompletedDownload[]>([])

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
      if (data.status === 'complete' || data.status === 'cancelled') {
        setCompleted(prev => [{ jobId: data.jobId, repoId: data.repoId, status: data.status, time: Date.now() }, ...prev])
      }
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
    })

    return () => {
      unsubProgress()
      unsubComplete()
      unsubError()
      unsubStart?.()
    }
  }, [])

  const shortName = (repoId: string) => repoId.includes('/') ? repoId.split('/').pop() : repoId
  const p = active?.progress

  return (
    <div className="h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-card" style={{ WebkitAppRegion: 'drag' } as any}>
        <Download className="h-5 w-5 text-primary" />
        <h1 className="text-sm font-semibold">Downloads</h1>
        {active && !active.error && (
          <span className="ml-auto text-xs text-muted-foreground">
            {p?.percent != null ? `${p.percent}%` : 'Starting...'}
          </span>
        )}
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {/* Active download */}
        {active && (
          <div className={`p-4 rounded-lg border ${active.error ? 'border-destructive/30 bg-destructive/5' : 'border-primary/20 bg-primary/5'}`}>
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                {active.error ? (
                  <AlertCircle className="h-4 w-4 text-destructive" />
                ) : (
                  <Loader2 className="h-4 w-4 text-primary animate-spin" />
                )}
                <span className="text-sm font-medium">{active.repoId}</span>
              </div>
              {!active.error && (
                <button
                  onClick={() => window.api.models.cancelDownload(active.jobId)}
                  className="text-xs text-destructive hover:text-destructive/80 px-2 py-1 border border-destructive/30 rounded"
                >
                  Cancel
                </button>
              )}
            </div>
            {active.error && (
              <p className="text-xs text-destructive mb-2">{active.error}</p>
            )}
            {p && (
              <div className="space-y-2">
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(p.percent || 0, 100)}%` }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span className="font-medium">{p.percent != null ? `${p.percent}%` : 'Starting...'}</span>
                  <div className="flex gap-3">
                    {p.downloaded && p.total && <span>{p.downloaded} / {p.total}</span>}
                    {p.speed && <span>{p.speed}</span>}
                    {p.eta && <span>ETA {p.eta}</span>}
                  </div>
                </div>
                {p.currentFile && (
                  <p className="text-[11px] text-muted-foreground truncate">
                    {p.filesProgress && <span className="font-medium">[{p.filesProgress}] </span>}
                    {p.currentFile}
                  </p>
                )}
              </div>
            )}
            {!p && !active.error && (
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span>Preparing download...</span>
              </div>
            )}
          </div>
        )}

        {/* Queued downloads */}
        {queue.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Queued ({queue.length})
            </h3>
            {queue.map((item, i) => (
              <div key={item.jobId || i} className="flex items-center justify-between py-2 px-3 border border-border rounded mb-1">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-yellow-500 rounded-full" />
                  <span className="text-sm">{item.repoId}</span>
                </div>
                <button
                  onClick={() => window.api.models.cancelDownload(item.jobId)}
                  className="p-1 text-muted-foreground hover:text-destructive"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Completed downloads (this session) */}
        {completed.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">
              Completed
            </h3>
            {completed.map((item) => (
              <div key={item.jobId} className="flex items-center gap-2 py-2 px-3 border border-border rounded mb-1">
                {item.status === 'complete' ? (
                  <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                ) : (
                  <X className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                )}
                <span className="text-sm">{shortName(item.repoId)}</span>
                <span className="text-xs text-muted-foreground ml-auto">
                  {item.status === 'complete' ? 'Done' : 'Cancelled'}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Empty state */}
        {!active && queue.length === 0 && completed.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Download className="h-10 w-10 mb-3 opacity-30" />
            <p className="text-sm">No active downloads</p>
            <p className="text-xs mt-1">Downloads started from the Image tab will appear here</p>
          </div>
        )}
      </div>
    </div>
  )
}
