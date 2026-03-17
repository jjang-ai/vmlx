import { useState, useEffect, useRef } from 'react'
import { X, Maximize2, Minimize2, Download, Loader2 } from 'lucide-react'

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
  const [expanded, setExpanded] = useState(false)
  const onCompleteRef = useRef(onComplete)
  onCompleteRef.current = onComplete

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
      setExpanded(true) // Auto-expand when download starts
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
        <button onClick={() => setExpanded(!expanded)} className="p-1 text-muted-foreground hover:text-foreground" title={expanded ? 'Collapse' : 'Expand downloads'}>
          {expanded ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
        </button>
        {active && !active.error && (
          <button onClick={() => window.api.models.cancelDownload(active.jobId)} className="text-[10px] text-destructive hover:text-destructive/80 px-1.5 py-0.5 border border-destructive/30 rounded flex-shrink-0">
            Cancel
          </button>
        )}
      </div>
    </div>
  )

  // Expanded popup panel
  const expandedPanel = expanded ? (
    <div className="fixed inset-x-0 top-20 mx-auto w-[500px] max-h-[400px] bg-card border border-border rounded-lg shadow-2xl z-50 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
        <div className="flex items-center gap-2">
          <Download className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-medium">Downloads</h3>
        </div>
        <button onClick={() => setExpanded(false)} className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-accent">
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="p-4 space-y-4 overflow-auto max-h-[340px]">
        {/* Active download */}
        {active && (
          <div className={`p-3 rounded-lg border ${active.error ? 'border-destructive/30 bg-destructive/5' : 'border-primary/20 bg-primary/5'}`}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                {active.error ? (
                  <span className="w-2 h-2 bg-destructive rounded-full" />
                ) : (
                  <Loader2 className="h-4 w-4 text-primary animate-spin" />
                )}
                <span className="text-sm font-medium">{active.repoId}</span>
              </div>
              {!active.error && (
                <button onClick={() => window.api.models.cancelDownload(active.jobId)} className="text-xs text-destructive hover:text-destructive/80 px-2 py-1 border border-destructive/30 rounded">
                  Cancel
                </button>
              )}
            </div>
            {active.error && (
              <p className="text-xs text-destructive mb-2">{active.error}</p>
            )}
            {p && (
              <div className="space-y-1.5">
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div className="h-full bg-primary rounded-full transition-all duration-300" style={{ width: `${Math.min(p.percent || 0, 100)}%` }} />
                </div>
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>{p.percent != null ? `${p.percent}%` : 'Starting...'}</span>
                  <div className="flex gap-3">
                    {p.downloaded && p.total && <span>{p.downloaded} / {p.total}</span>}
                    {p.speed && <span>{p.speed}</span>}
                    {p.eta && <span>ETA {p.eta}</span>}
                  </div>
                </div>
                {p.currentFile && (
                  <p className="text-[10px] text-muted-foreground truncate" title={p.currentFile}>{p.currentFile}</p>
                )}
                {p.filesProgress && (
                  <p className="text-[10px] text-muted-foreground">{p.filesProgress}</p>
                )}
              </div>
            )}
            {!p && !active.error && (
              <p className="text-xs text-muted-foreground">Preparing download...</p>
            )}
          </div>
        )}

        {/* Queued downloads */}
        {queue.length > 0 && (
          <div>
            <h4 className="text-xs font-medium text-muted-foreground mb-2">Queued ({queue.length})</h4>
            {queue.map((item, i) => (
              <div key={item.jobId || i} className="flex items-center justify-between py-1.5 px-2 text-xs">
                <div className="flex items-center gap-2">
                  <span className="w-1.5 h-1.5 bg-warning rounded-full" />
                  <span className="text-muted-foreground">{item.repoId}</span>
                </div>
                <button onClick={() => window.api.models.cancelDownload(item.jobId)} className="text-[10px] text-muted-foreground hover:text-destructive">
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* No active downloads */}
        {!active && queue.length === 0 && (
          <p className="text-sm text-muted-foreground text-center py-4">No active downloads</p>
        )}
      </div>
    </div>
  ) : null

  return (
    <>
      {inlineBar}
      {expandedPanel}
    </>
  )
}
