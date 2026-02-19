import { useState, useEffect, useRef } from 'react'

interface CachePanelProps {
  endpoint: { host: string; port: number }
  sessionStatus: string
}

export function CachePanel({ endpoint, sessionStatus }: CachePanelProps) {
  const [stats, setStats] = useState<any>(null)
  const [entries, setEntries] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showEntries, setShowEntries] = useState(false)
  const [warming, setWarming] = useState(false)
  const [clearing, setClearing] = useState(false)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const fetchStats = async () => {
    if (sessionStatus !== 'running') return
    try {
      const s = await window.api.cache.stats(endpoint)
      setStats(s)
      setError(null)
    } catch (err: any) {
      setError(err.message || 'Failed to fetch cache stats')
    }
  }

  // Poll stats every 5 seconds
  useEffect(() => {
    if (sessionStatus === 'running') {
      fetchStats()
      intervalRef.current = setInterval(fetchStats, 5000)
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current)
    }
  }, [endpoint.host, endpoint.port, sessionStatus])

  const handleFetchEntries = async () => {
    setLoading(true)
    try {
      const e = await window.api.cache.entries(endpoint)
      setEntries(e)
      setShowEntries(true)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleWarm = async () => {
    const prompt = window.prompt('Enter system prompt to warm cache with:')
    if (!prompt) return
    setWarming(true)
    try {
      await window.api.cache.warm([prompt], endpoint)
      await fetchStats()
    } catch (err: any) {
      setError(err.message)
    } finally {
      setWarming(false)
    }
  }

  const handleClear = async (type: string) => {
    setClearing(true)
    try {
      await window.api.cache.clear(type, endpoint)
      await fetchStats()
      setEntries(null)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setClearing(false)
    }
  }

  if (sessionStatus !== 'running') {
    return (
      <div className="text-sm text-muted-foreground p-4">
        Session must be running to view cache stats.
      </div>
    )
  }

  const schedulerCache = stats?.scheduler_cache
  const schedulerStats = stats?.scheduler_stats
  const diskCache = stats?.disk_cache
  const kvQuant = stats?.kv_cache_quantization

  return (
    <div className="space-y-4">
      {error && (
        <div className="text-xs text-destructive bg-destructive/10 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {/* Cache Stats Overview */}
      {schedulerCache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Prefix Cache</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {schedulerCache.hit_rate != null && (
              <StatCard label="Hit Rate" value={`${(schedulerCache.hit_rate * 100).toFixed(1)}%`} />
            )}
            {schedulerCache.entries != null && (
              <StatCard label="Entries" value={String(schedulerCache.entries)} />
            )}
            {schedulerCache.memory_mb != null && (
              <StatCard label="Memory" value={`${schedulerCache.memory_mb.toFixed(1)} MB`} />
            )}
            {schedulerCache.hits != null && (
              <StatCard label="Hits / Misses" value={`${schedulerCache.hits} / ${schedulerCache.misses || 0}`} />
            )}
            {schedulerCache.total_cached_tokens != null && (
              <StatCard label="Cached Tokens" value={schedulerCache.total_cached_tokens.toLocaleString()} />
            )}
            {schedulerCache.evictions != null && (
              <StatCard label="Evictions" value={String(schedulerCache.evictions)} />
            )}
          </div>
        </div>
      )}

      {/* Scheduler Stats */}
      {schedulerStats && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Scheduler</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <StatCard label="Requests" value={String(schedulerStats.num_requests_processed || 0)} />
            <StatCard label="Running" value={String(schedulerStats.num_running || 0)} />
            <StatCard label="Prompt Tokens" value={(schedulerStats.total_prompt_tokens || 0).toLocaleString()} />
            <StatCard label="Completion Tokens" value={(schedulerStats.total_completion_tokens || 0).toLocaleString()} />
          </div>
        </div>
      )}

      {/* KV Quantization Info */}
      {kvQuant && (
        <div className="text-xs text-muted-foreground">
          KV Cache: {kvQuant.bits}-bit quantization (group size {kvQuant.group_size})
        </div>
      )}

      {/* Disk Cache */}
      {diskCache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Disk Cache (L2)</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {diskCache.entries != null && <StatCard label="Entries" value={String(diskCache.entries)} />}
            {diskCache.size_mb != null && <StatCard label="Size" value={`${diskCache.size_mb.toFixed(1)} MB`} />}
            {diskCache.hit_rate != null && <StatCard label="Hit Rate" value={`${(diskCache.hit_rate * 100).toFixed(1)}%`} />}
          </div>
        </div>
      )}

      {!schedulerCache && !schedulerStats && !stats?.error && (
        <div className="text-sm text-muted-foreground">Loading cache stats...</div>
      )}

      {stats?.error && (
        <div className="text-sm text-muted-foreground">{stats.error}</div>
      )}

      {/* Cache Entries */}
      {showEntries && entries && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            Cache Entries ({entries.count || 0}) — {entries.cache_type}
          </h4>
          <div className="max-h-48 overflow-auto space-y-1">
            {entries.entries?.map((entry: any, i: number) => (
              <div key={i} className="text-xs bg-background px-2 py-1 rounded border border-border flex justify-between">
                <span>{entry.tokens_count} tokens</span>
                {entry.memory_mb && <span className="text-muted-foreground">{entry.memory_mb} MB</span>}
                {entry.ref_count != null && <span className="text-muted-foreground">refs: {entry.ref_count}</span>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={handleFetchEntries}
          disabled={loading}
          className="px-3 py-1.5 text-xs border border-border rounded hover:bg-accent disabled:opacity-50"
        >
          {loading ? 'Loading...' : showEntries ? 'Refresh Entries' : 'Show Entries'}
        </button>
        <button
          onClick={handleWarm}
          disabled={warming}
          className="px-3 py-1.5 text-xs border border-border rounded hover:bg-accent disabled:opacity-50"
        >
          {warming ? 'Warming...' : 'Warm Cache'}
        </button>
        <button
          onClick={() => handleClear('prefix')}
          disabled={clearing}
          className="px-3 py-1.5 text-xs border border-destructive/50 text-destructive rounded hover:bg-destructive/10 disabled:opacity-50"
        >
          Clear Prefix
        </button>
        <button
          onClick={() => handleClear('all')}
          disabled={clearing}
          className="px-3 py-1.5 text-xs border border-destructive/50 text-destructive rounded hover:bg-destructive/10 disabled:opacity-50"
        >
          Clear All
        </button>
      </div>
    </div>
  )
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-background px-3 py-2 rounded border border-border">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="font-mono text-sm">{value}</div>
    </div>
  )
}
