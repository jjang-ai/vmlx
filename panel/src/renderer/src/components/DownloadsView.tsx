import { useState, useEffect } from 'react'
import { Download, Loader2, X, CheckCircle, AlertCircle, Pause, Play } from 'lucide-react'

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
  const [activeDownloads, setActiveDownloads] = useState<ActiveDownload[]>([])
  const [queue, setQueue] = useState<Array<{ jobId: string; repoId: string }>>([])
  const [paused, setPaused] = useState<Array<{ jobId: string; repoId: string }>>([])
  const [completed, setCompleted] = useState<CompletedDownload[]>([])

  const refreshStatus = () => {
    window.api.models.getDownloadStatus().then((status: any) => {
      if (status.activeAll) {
        setActiveDownloads(status.activeAll.map((a: any) => ({ jobId: a.jobId, repoId: a.repoId, progress: a.progress })))
      } else if (status.active) {
        setActiveDownloads([{ jobId: status.active.jobId, repoId: status.active.repoId, progress: status.active.progress }])
      } else {
        setActiveDownloads([])
      }
      setQueue(status.queue || [])
      // Restore completed downloads so re-opening window shows history
      if (status.completed?.length) {
        setCompleted(prev => {
          const existing = new Set(prev.map(c => c.jobId))
          const newItems = status.completed
            .filter((c: any) => !existing.has(c.jobId))
            .map((c: any) => ({ jobId: c.jobId, repoId: c.repoId, status: c.status as 'complete' | 'cancelled', time: Date.now() }))
          return [...newItems, ...prev]
        })
      }
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
      if (data.status === 'complete' || data.status === 'cancelled') {
        setCompleted(prev => [{ jobId: data.jobId, repoId: data.repoId, status: data.status, time: Date.now() }, ...prev])
      }
      setActiveDownloads(prev => prev.filter(d => d.jobId !== data.jobId))
      // Refresh to pick up next queued item
      setTimeout(refreshStatus, 500)
    })
    const unsubError = window.api.models.onDownloadError((data: any) => {
      // Show error on active card briefly, then move to completed as failed
      setActiveDownloads(prev => prev.map(d =>
        d.jobId === data.jobId ? { ...d, error: data.error || 'Download failed', progress: undefined } : d
      ))
      setTimeout(() => {
        setActiveDownloads(prev => prev.filter(d => d.jobId !== data.jobId))
        setCompleted(prev => [{ jobId: data.jobId, repoId: data.repoId, status: 'cancelled' as const, time: Date.now() }, ...prev])
        refreshStatus()
      }, 3000)
    })
    const unsubStart = window.api.models.onDownloadStarted?.((data: any) => {
      setActiveDownloads(prev => {
        if (prev.some(d => d.jobId === data.jobId)) return prev
        return [...prev, { jobId: data.jobId, repoId: data.repoId }]
      })
      // Remove from queue display
      setQueue(prev => prev.filter(q => q.jobId !== data.jobId))
    })
    const unsubQueued = window.api.models.onDownloadQueued?.((data: any) => {
      setQueue(prev => {
        if (prev.some(q => q.jobId === data.jobId)) return prev
        return [...prev, { jobId: data.jobId, repoId: data.repoId }]
      })
    })
    const unsubPaused = window.api.models.onDownloadPaused?.((data: any) => {
      setActiveDownloads(prev => prev.filter(d => d.jobId !== data.jobId))
      setPaused(prev => {
        if (prev.some(p => p.jobId === data.jobId)) return prev
        return [...prev, { jobId: data.jobId, repoId: data.repoId }]
      })
    })

    return () => {
      unsubProgress()
      unsubComplete()
      unsubError()
      unsubStart?.()
      unsubQueued?.()
      unsubPaused?.()
    }
  }, [])

  const shortName = (repoId: string) => repoId.includes('/') ? repoId.split('/').pop() : repoId

  return (
    <div className="h-screen bg-background text-foreground flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border bg-card" style={{ WebkitAppRegion: 'drag' } as any}>
        <Download className="h-5 w-5 text-primary" />
        <h1 className="text-sm font-semibold">Downloads</h1>
        <span className="ml-auto text-xs text-muted-foreground">
          {activeDownloads.length > 0 ? `${activeDownloads.length} active` : 'No active downloads'}
          {queue.length > 0 && ` · ${queue.length} queued`}
        </span>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-3">
        {/* Active downloads */}
        {activeDownloads.map(dl => {
          const p = dl.progress
          return (
            <div key={dl.jobId} className={`p-4 rounded-lg border ${dl.error ? 'border-destructive/30 bg-destructive/5' : 'border-primary/20 bg-primary/5'}`}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2 min-w-0">
                  {dl.error ? (
                    <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0" />
                  ) : (
                    <Loader2 className="h-4 w-4 text-primary animate-spin flex-shrink-0" />
                  )}
                  <span className="text-sm font-medium truncate">{dl.repoId}</span>
                </div>
                {!dl.error && (
                  <div className="flex gap-1 flex-shrink-0 ml-2">
                    <button
                      onClick={() => window.api.models.pauseDownload(dl.jobId)}
                      className="text-xs text-yellow-600 hover:text-yellow-500 px-2 py-1 border border-yellow-500/30 rounded"
                      title="Pause download"
                    >
                      <Pause className="h-3 w-3" />
                    </button>
                    <button
                      onClick={() => window.api.models.cancelDownload(dl.jobId)}
                      className="text-xs text-destructive hover:text-destructive/80 px-2 py-1 border border-destructive/30 rounded"
                    >
                      Cancel
                    </button>
                  </div>
                )}
              </div>
              {dl.error && <p className="text-xs text-destructive mb-2">{dl.error}</p>}
              {p && (
                <div className="space-y-2">
                  <div className="h-3 bg-muted rounded-full overflow-hidden">
                    <div className="h-full bg-primary rounded-full transition-all duration-500" style={{ width: `${Math.min(p.percent || 0, 100)}%` }} />
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="font-medium">{p.percent != null ? `${p.percent}%` : 'Starting...'}</span>
                    <div className="flex gap-3">
                      {p.downloaded && p.total && <span>{p.downloaded} / {p.total}</span>}
                      {p.speed && <span>{p.speed}</span>}
                      {p.eta && <span>{p.eta}</span>}
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
              {!p && !dl.error && (
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span>Preparing download...</span>
                </div>
              )}
            </div>
          )
        })}

        {/* Paused */}
        {paused.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">Paused ({paused.length})</h3>
            {paused.map((item) => (
              <div key={item.jobId} className="flex items-center justify-between py-2 px-3 border border-yellow-500/30 bg-yellow-500/5 rounded mb-1">
                <div className="flex items-center gap-2">
                  <Pause className="h-3.5 w-3.5 text-yellow-500" />
                  <span className="text-sm truncate">{item.repoId}</span>
                </div>
                <div className="flex gap-1">
                  <button
                    onClick={() => {
                      window.api.models.resumeDownload(item.jobId)
                      setPaused(prev => prev.filter(p => p.jobId !== item.jobId))
                    }}
                    className="text-xs text-green-600 hover:text-green-500 px-2 py-1 border border-green-500/30 rounded flex items-center gap-1"
                  >
                    <Play className="h-3 w-3" /> Resume
                  </button>
                  <button
                    onClick={() => {
                      window.api.models.cancelDownload(item.jobId)
                      setPaused(prev => prev.filter(p => p.jobId !== item.jobId))
                    }}
                    className="p-1 text-muted-foreground hover:text-destructive"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Queued */}
        {queue.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">Queued ({queue.length})</h3>
            {queue.map((item, i) => (
              <div key={item.jobId || i} className="flex items-center justify-between py-2 px-3 border border-border rounded mb-1">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-yellow-500 rounded-full" />
                  <span className="text-sm truncate">{item.repoId}</span>
                </div>
                <button onClick={() => window.api.models.cancelDownload(item.jobId)} className="p-1 text-muted-foreground hover:text-destructive">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Completed */}
        {completed.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">Completed</h3>
            {completed.map((item) => (
              <div key={item.jobId} className="flex items-center gap-2 py-2 px-3 border border-border rounded mb-1">
                {item.status === 'complete' ? (
                  <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                ) : (
                  <X className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                )}
                <span className="text-sm truncate">{shortName(item.repoId)}</span>
                <span className="text-xs text-muted-foreground ml-auto">{item.status === 'complete' ? 'Done' : 'Cancelled'}</span>
              </div>
            ))}
          </div>
        )}

        {/* Empty */}
        {activeDownloads.length === 0 && queue.length === 0 && completed.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
            <Download className="h-10 w-10 mb-3 opacity-30" />
            <p className="text-sm">No active downloads</p>
            <p className="text-xs mt-1">Downloads from the Image or Server tab appear here</p>
          </div>
        )}
      </div>
    </div>
  )
}
