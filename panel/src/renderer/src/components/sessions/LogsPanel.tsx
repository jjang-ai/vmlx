import { useState, useEffect, useRef, useCallback } from 'react'
import { Trash2, Download, Pause, Play } from 'lucide-react'

interface LogsPanelProps {
  sessionId: string
  sessionStatus: string
  isRemote?: boolean
}

/** Classify a log line for severity coloring */
function getLineClass(line: string): string {
  if (line.includes('ERROR') || line.includes('Traceback') || line.includes('Exception'))
    return 'text-destructive'
  if (line.includes('WARNING') || line.includes('warn'))
    return 'text-warning'
  if (line.includes('[INFO]'))
    return 'text-muted-foreground'
  return 'text-foreground/80'
}

export function LogsPanel({ sessionId, sessionStatus, isRemote }: LogsPanelProps) {
  const [lines, setLines] = useState<string[]>([])
  const [paused, setPaused] = useState(false)
  const [filter, setFilter] = useState('')
  const scrollRef = useRef<HTMLDivElement>(null)
  const autoScrollRef = useRef(true)

  // Pull full buffer on mount
  useEffect(() => {
    window.api.sessions.getLogs(sessionId).then((logs: string[]) => {
      setLines(logs)
    })
  }, [sessionId])

  // Subscribe to live log events (only when not paused)
  useEffect(() => {
    if (paused) return

    const unsub = window.api.sessions.onLog((data: any) => {
      if (data.sessionId !== sessionId) return
      const timestamp = new Date().toISOString().slice(11, 23)
      const newLines = data.data.split('\n').filter((l: string) => l)
        .map((l: string) => `[${timestamp}] ${l}`)
      setLines(prev => {
        const next = [...prev, ...newLines]
        return next.length > 2000 ? next.slice(next.length - 2000) : next
      })
    })

    return unsub
  }, [sessionId, paused])

  // Auto-scroll to bottom when new lines arrive
  useEffect(() => {
    if (autoScrollRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [lines])

  const handleScroll = useCallback(() => {
    if (!scrollRef.current) return
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current
    autoScrollRef.current = scrollHeight - scrollTop - clientHeight < 50
  }, [])

  const handleClear = async () => {
    await window.api.sessions.clearLogs(sessionId)
    setLines([])
  }

  const handleExport = () => {
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `vmlx-logs-${sessionId.slice(0, 8)}-${new Date().toISOString().slice(0, 10)}.log`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Filter lines by search term
  const displayLines = filter
    ? lines.filter(l => l.toLowerCase().includes(filter.toLowerCase()))
    : lines

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border flex-shrink-0">
        <input
          type="text"
          placeholder="Filter logs..."
          value={filter}
          onChange={e => setFilter(e.target.value)}
          className="flex-1 px-2 py-1 text-xs bg-background border border-input rounded focus:outline-none focus:ring-1 focus:ring-ring"
        />
        <button
          onClick={() => setPaused(!paused)}
          className={`p-1 rounded text-xs ${paused ? 'text-warning' : 'text-muted-foreground hover:text-foreground'}`}
          title={paused ? 'Resume live logs' : 'Pause live logs'}
        >
          {paused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
        </button>
        <button
          onClick={handleExport}
          className="p-1 rounded text-muted-foreground hover:text-foreground text-xs"
          title="Export logs"
        >
          <Download className="h-3.5 w-3.5" />
        </button>
        <button
          onClick={handleClear}
          className="p-1 rounded text-muted-foreground hover:text-destructive text-xs"
          title="Clear logs"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Status bar */}
      <div className="flex items-center gap-2 px-3 py-1 text-[10px] text-muted-foreground border-b border-border flex-shrink-0">
        <span>{displayLines.length} lines</span>
        {filter && <span>({lines.length} total)</span>}
        {paused && <span className="text-warning font-medium">PAUSED</span>}
        {sessionStatus !== 'running' && <span className="text-destructive">{isRemote ? 'Not connected' : 'Server not running'}</span>}
      </div>

      {/* Log output */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto overflow-x-auto bg-[#0d1117] p-2 font-mono text-[11px] leading-relaxed"
      >
        {displayLines.length === 0 ? (
          <div className="text-muted-foreground/50 text-center py-8">
            {sessionStatus === 'running'
              ? 'No log output yet...'
              : isRemote ? 'Connect to see connection logs' : 'Start the server to see logs'}
          </div>
        ) : (
          displayLines.map((line, i) => (
            <div key={i} className={`whitespace-pre-wrap break-all ${getLineClass(line)}`}>
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  )
}
