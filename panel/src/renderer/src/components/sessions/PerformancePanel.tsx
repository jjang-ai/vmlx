import { useState, useEffect, useRef } from 'react'

interface PerformancePanelProps {
  endpoint: { host: string; port: number }
  sessionStatus: string
}

interface HealthData {
  status: string
  model_loaded: boolean
  model_name?: string
  model_type?: string
  engine_type?: string
  memory?: {
    active_mb: number
    peak_mb: number
    cache_mb: number
  }
  kv_cache_quantization?: {
    enabled: boolean
    bits?: number
    group_size?: number
  }
}

export function PerformancePanel({ endpoint, sessionStatus }: PerformancePanelProps) {
  const [health, setHealth] = useState<HealthData | null>(null)
  const [history, setHistory] = useState<Array<{ time: number; active: number; peak: number }>>([])
  const [error, setError] = useState<string | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (sessionStatus !== 'running') {
      setHealth(null)
      setHistory([])
      return
    }

    const poll = async () => {
      try {
        const data = await window.api.performance.health(endpoint)
        setHealth(data)
        setError(null)

        if (data.memory) {
          setHistory(prev => {
            const next = [...prev, { time: Date.now(), active: data.memory.active_mb, peak: data.memory.peak_mb }]
            return next.slice(-60) // Keep last 60 samples (5 minutes at 5s interval)
          })
        }
      } catch (err: any) {
        setError(err.message)
      }
    }

    poll()
    intervalRef.current = setInterval(poll, 5000)
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [endpoint.host, endpoint.port, sessionStatus])

  if (sessionStatus !== 'running') {
    return (
      <div className="text-sm text-muted-foreground p-4">
        Session must be running to monitor performance.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {error && (
        <div className="text-xs text-destructive bg-destructive/10 px-3 py-2 rounded">{error}</div>
      )}

      {/* Engine Info */}
      {health && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Engine</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <InfoCard label="Status" value={health.status} />
            <InfoCard label="Engine" value={health.engine_type || 'unknown'} />
            <InfoCard label="Model Type" value={health.model_type || '-'} />
            {health.kv_cache_quantization?.enabled && (
              <InfoCard label="KV Quant" value={`${health.kv_cache_quantization.bits}-bit`} />
            )}
          </div>
        </div>
      )}

      {/* Memory */}
      {health?.memory && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">GPU Memory (Metal)</h4>
          <div className="grid grid-cols-3 gap-2">
            <MemoryCard label="Active" value={health.memory.active_mb} />
            <MemoryCard label="Peak" value={health.memory.peak_mb} />
            <MemoryCard label="Cache" value={health.memory.cache_mb} />
          </div>

          {/* Memory Graph */}
          {history.length > 1 && (
            <div className="mt-3">
              <div className="text-xs text-muted-foreground mb-1">Memory over time</div>
              <MiniGraph data={history} />
            </div>
          )}
        </div>
      )}

      {!health && !error && (
        <div className="text-sm text-muted-foreground">Loading health data...</div>
      )}
    </div>
  )
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-background px-2 py-1.5 rounded border border-border">
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="font-mono text-xs">{value}</div>
    </div>
  )
}

function MemoryCard({ label, value }: { label: string; value: number }) {
  const formatted = value >= 1024 ? `${(value / 1024).toFixed(1)} GB` : `${value.toFixed(0)} MB`
  return (
    <div className="bg-background px-2 py-1.5 rounded border border-border text-center">
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="font-mono text-sm">{formatted}</div>
    </div>
  )
}

function MiniGraph({ data }: { data: Array<{ time: number; active: number; peak: number }> }) {
  const maxVal = Math.max(...data.map(d => d.peak), 1)
  const h = 60
  const w = 240
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * w
    const y = h - (d.active / maxVal) * h
    return `${x},${y}`
  }).join(' ')

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-[60px] border border-border rounded bg-background">
      <polyline
        points={points}
        fill="none"
        stroke="hsl(var(--primary))"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      {/* Peak line */}
      <line
        x1="0" y1={h - (data[data.length - 1].peak / maxVal) * h}
        x2={w} y2={h - (data[data.length - 1].peak / maxVal) * h}
        stroke="hsl(var(--destructive))"
        strokeWidth="0.5"
        strokeDasharray="4 2"
        opacity="0.5"
      />
      {/* Label */}
      <text x={w - 2} y={10} textAnchor="end" fontSize="8" fill="hsl(var(--muted-foreground))">
        {(maxVal >= 1024 ? (maxVal / 1024).toFixed(1) + ' GB' : maxVal.toFixed(0) + ' MB')}
      </text>
    </svg>
  )
}
