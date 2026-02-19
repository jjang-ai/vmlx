import { useState, useEffect } from 'react'

interface BenchmarkPanelProps {
  sessionId: string
  endpoint: { host: string; port: number }
  modelPath: string
  modelName?: string
  sessionStatus: string
}

interface PromptResult {
  label: string
  ttft: number
  tps: number
  promptTokens: number
  completionTokens: number
  totalTime: number
  ppSpeed: number
}

interface BenchmarkRun {
  id: string
  sessionId: string
  modelPath: string
  modelName?: string
  results: PromptResult[]
  createdAt: number
}

export function BenchmarkPanel({ sessionId, endpoint, modelPath, modelName, sessionStatus }: BenchmarkPanelProps) {
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<{ current: number; total: number; label: string } | null>(null)
  const [currentResults, setCurrentResults] = useState<PromptResult[] | null>(null)
  const [history, setHistory] = useState<BenchmarkRun[]>([])
  const [showHistory, setShowHistory] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [flushCache, setFlushCache] = useState(true)

  // Load history on mount
  useEffect(() => {
    loadHistory()
  }, [modelPath])

  // Listen for progress events
  useEffect(() => {
    const unsub = window.api.benchmark.onProgress((data: any) => {
      if (data.sessionId === sessionId) {
        setProgress({ current: data.current, total: data.total, label: data.label })
      }
    })
    return unsub
  }, [sessionId])

  const loadHistory = async () => {
    try {
      const h = await window.api.benchmark.history(modelPath)
      setHistory(h)
    } catch { /* ignore */ }
  }

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    setCurrentResults(null)
    setProgress(null)

    try {
      const result = await window.api.benchmark.run(sessionId, endpoint, modelPath, modelName, { flushCache })
      setCurrentResults(result.results)
      await loadHistory()
    } catch (err: any) {
      setError(err.message || 'Benchmark failed')
    } finally {
      setRunning(false)
      setProgress(null)
    }
  }

  const handleDelete = async (id: string) => {
    await window.api.benchmark.delete(id)
    setHistory(prev => prev.filter(h => h.id !== id))
  }

  if (sessionStatus !== 'running') {
    return (
      <div className="text-sm text-muted-foreground p-4">
        Session must be running to benchmark.
      </div>
    )
  }

  const avgTps = (results: PromptResult[]) => {
    const valid = results.filter(r => r.tps > 0)
    return valid.length > 0 ? valid.reduce((sum, r) => sum + r.tps, 0) / valid.length : 0
  }

  const avgTtft = (results: PromptResult[]) => {
    const valid = results.filter(r => r.ttft > 0)
    return valid.length > 0 ? valid.reduce((sum, r) => sum + r.ttft, 0) / valid.length : 0
  }

  return (
    <div className="space-y-4">
      {error && (
        <div className="text-xs text-destructive bg-destructive/10 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {/* Run Benchmark */}
      <div>
        <div className="flex items-center gap-3">
          <button
            onClick={handleRun}
            disabled={running}
            className="px-3 py-1.5 text-xs bg-primary text-primary-foreground rounded hover:bg-primary/90 disabled:opacity-50"
          >
            {running ? 'Running...' : 'Run Benchmark'}
          </button>
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer">
            <input
              type="checkbox"
              checked={flushCache}
              onChange={e => setFlushCache(e.target.checked)}
              className="rounded border-border"
            />
            Flush cache first
          </label>
        </div>
        {progress && (
          <div className="mt-2 text-xs text-muted-foreground">
            {progress.label} ({progress.current}/{progress.total})
            <div className="w-full h-1.5 bg-muted rounded mt-1">
              <div
                className="h-full bg-primary rounded transition-all"
                style={{ width: `${(progress.current / progress.total) * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Current Results */}
      {currentResults && (
        <ResultsTable results={currentResults} />
      )}

      {/* History */}
      {history.length > 0 && (
        <div>
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="text-xs text-muted-foreground hover:text-foreground"
          >
            {showHistory ? 'Hide' : 'Show'} History ({history.length})
          </button>
          {showHistory && (
            <div className="mt-2 space-y-3 max-h-64 overflow-auto">
              {history.map(run => (
                <div key={run.id} className="bg-background border border-border rounded p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs text-muted-foreground">
                      {new Date(run.createdAt).toLocaleString()}
                    </span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono">
                        Avg: {avgTps(run.results).toFixed(1)} t/s | TTFT: {(avgTtft(run.results) * 1000).toFixed(0)}ms
                      </span>
                      <button
                        onClick={() => handleDelete(run.id)}
                        className="text-xs text-destructive hover:text-destructive/80"
                      >
                        x
                      </button>
                    </div>
                  </div>
                  <ResultsTable results={run.results} compact />
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function ResultsTable({ results, compact }: { results: PromptResult[]; compact?: boolean }) {
  return (
    <div className="overflow-auto">
      <table className={`w-full text-xs ${compact ? '' : 'mt-2'}`}>
        <thead>
          <tr className="text-muted-foreground border-b border-border">
            <th className="text-left py-1 pr-2">Test</th>
            <th className="text-right py-1 px-1">TTFT</th>
            <th className="text-right py-1 px-1">TPS</th>
            <th className="text-right py-1 px-1">PP t/s</th>
            <th className="text-right py-1 pl-1">Time</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => (
            <tr key={i} className="border-b border-border/50">
              <td className="py-1 pr-2 truncate max-w-[120px]" title={r.label}>{r.label}</td>
              <td className="text-right py-1 px-1 font-mono">
                {r.ttft > 0 ? `${(r.ttft * 1000).toFixed(0)}ms` : '—'}
              </td>
              <td className="text-right py-1 px-1 font-mono font-medium">
                {r.tps > 0 ? r.tps.toFixed(1) : '—'}
              </td>
              <td className="text-right py-1 px-1 font-mono">
                {r.ppSpeed > 0 ? r.ppSpeed.toFixed(0) : '—'}
              </td>
              <td className="text-right py-1 pl-1 font-mono">
                {r.totalTime > 0 ? `${r.totalTime.toFixed(1)}s` : '—'}
              </td>
            </tr>
          ))}
          {/* Summary row */}
          <tr className="font-medium">
            <td className="py-1 pr-2">Average</td>
            <td className="text-right py-1 px-1 font-mono">
              {(() => {
                const valid = results.filter(r => r.ttft > 0)
                return valid.length > 0
                  ? `${(valid.reduce((s, r) => s + r.ttft, 0) / valid.length * 1000).toFixed(0)}ms`
                  : '—'
              })()}
            </td>
            <td className="text-right py-1 px-1 font-mono">
              {(() => {
                const valid = results.filter(r => r.tps > 0)
                return valid.length > 0
                  ? (valid.reduce((s, r) => s + r.tps, 0) / valid.length).toFixed(1)
                  : '—'
              })()}
            </td>
            <td className="text-right py-1 px-1 font-mono">
              {(() => {
                const valid = results.filter(r => r.ppSpeed > 0)
                return valid.length > 0
                  ? (valid.reduce((s, r) => s + r.ppSpeed, 0) / valid.length).toFixed(0)
                  : '—'
              })()}
            </td>
            <td className="text-right py-1 pl-1 font-mono">
              {results.reduce((s, r) => s + r.totalTime, 0).toFixed(1)}s
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  )
}
