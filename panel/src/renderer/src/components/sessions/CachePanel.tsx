import { useState, useEffect, useRef } from 'react'

interface CachePanelProps {
  endpoint: { host: string; port: number }
  sessionStatus: string
  sessionId?: string
}

export function CachePanel({ endpoint, sessionStatus, sessionId }: CachePanelProps) {
  const [stats, setStats] = useState<any>(null)
  const [entries, setEntries] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showEntries, setShowEntries] = useState(false)
  const [warming, setWarming] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [warmInput, setWarmInput] = useState('')
  const [showWarmInput, setShowWarmInput] = useState(false)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)

  const fetchStats = async () => {
    if (sessionStatus !== 'running' && sessionStatus !== 'standby') return
    try {
      const s = await window.api.cache.stats(endpoint, sessionId)
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
  }, [endpoint.host, endpoint.port, sessionStatus, sessionId])

  const handleFetchEntries = async () => {
    setLoading(true)
    try {
      const e = await window.api.cache.entries(endpoint, sessionId)
      setEntries(e)
      setShowEntries(true)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleWarm = async () => {
    if (!warmInput.trim()) {
      setShowWarmInput(true)
      return
    }
    setWarming(true)
    try {
      await window.api.cache.warm([warmInput.trim()], endpoint, sessionId)
      await fetchStats()
      setWarmInput('')
      setShowWarmInput(false)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setWarming(false)
    }
  }

  const handleClear = async (type: string) => {
    setClearing(true)
    try {
      await window.api.cache.clear(type, endpoint, sessionId)
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
  const nativeCache = stats?.native_cache
  const turboQuantKv = stats?.turboquant_kv_cache
  const cacheTotals = stats?.cache_totals
  const attentionKvStorage =
    nativeCache?.attention_kv_storage_quantization ??
    nativeCache?.storage_quantization

  return (
    <div className="space-y-4">
      {error && (
        <div className="text-xs text-destructive bg-destructive/10 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {cacheTotals && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Cache Totals</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {cacheTotals.ram_tokens_cached != null && (
              <StatCard label="RAM Cached Tokens" value={(cacheTotals.ram_tokens_cached || 0).toLocaleString()} />
            )}
            {cacheTotals.l2_tokens_on_disk != null && (
              <StatCard label="L2 Tokens on Disk" value={(cacheTotals.l2_tokens_on_disk || 0).toLocaleString()} />
            )}
            {cacheTotals.l2_prompt_tokens_on_disk != null && (
              <StatCard label="Prompt L2 Tokens" value={(cacheTotals.l2_prompt_tokens_on_disk || 0).toLocaleString()} />
            )}
            {cacheTotals.l2_block_tokens_on_disk != null && (
              <StatCard label="Block L2 Tokens" value={(cacheTotals.l2_block_tokens_on_disk || 0).toLocaleString()} />
            )}
            {(() => {
              const ssmL2Tokens = cacheTotals.l2_ssm_tokens_on_disk ?? cacheTotals.ssm_tokens_on_disk
              return ssmL2Tokens != null && ssmL2Tokens > 0 ? (
                <StatCard label="SSM L2 Tokens" value={(ssmL2Tokens || 0).toLocaleString()} />
              ) : null
            })()}
          </div>
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
            {(schedulerCache.entry_count ?? schedulerCache.entries) != null && (
              <StatCard label="Entries" value={String(schedulerCache.entry_count ?? schedulerCache.entries)} />
            )}
            {(schedulerCache.current_memory_mb ?? schedulerCache.memory_mb) != null && (
              <StatCard label="Memory" value={`${(schedulerCache.current_memory_mb ?? schedulerCache.memory_mb).toFixed(1)} MB`} />
            )}
            {schedulerCache.hits != null && (
              <StatCard label="Hits / Misses" value={`${schedulerCache.hits} / ${schedulerCache.misses || 0}`} />
            )}
            {(schedulerCache.tokens_saved ?? schedulerCache.total_cached_tokens) != null && (
              <StatCard label="Cached Tokens" value={(schedulerCache.tokens_saved ?? schedulerCache.total_cached_tokens).toLocaleString()} />
            )}
            {schedulerCache.evictions != null && (
              <StatCard label="Evictions" value={String(schedulerCache.evictions)} />
            )}
            {schedulerCache.block_size != null && (
              <StatCard label="Block Size" value={`${schedulerCache.block_size} tokens`} />
            )}
            {schedulerCache.allocated_blocks != null && (
              <StatCard label="Blocks" value={`${schedulerCache.allocated_blocks} / ${schedulerCache.max_blocks} (${schedulerCache.shared_blocks ?? 0} shared)`} />
            )}
            {schedulerCache.utilization != null && (
              <StatCard label="Utilization" value={`${(schedulerCache.utilization * 100).toFixed(1)}%`} />
            )}
            {schedulerCache.disk_hits != null && (schedulerCache.disk_hits > 0 || schedulerCache.disk_misses > 0) && (
              <StatCard label="L2 Disk Hits" value={`${schedulerCache.disk_hits} / ${schedulerCache.disk_misses} miss`} />
            )}
            {schedulerCache.cow_copies != null && schedulerCache.cow_copies > 0 && (
              <StatCard label="COW Copies" value={String(schedulerCache.cow_copies)} />
            )}
            {/* F4 (audit 2026-04-08): Agent 1's PrefixCacheManager
                 cache_type LRU exposes per-type byte / entry counts.
                 Display them when present so users can see system /
                 user / assistant priority pinning at a glance. */}
            {schedulerCache.max_bytes != null && schedulerCache.max_bytes > 0 && schedulerCache.nbytes != null && (
              <StatCard
                label="Cache Bytes"
                value={`${(schedulerCache.nbytes / (1024 * 1024)).toFixed(1)} / ${(schedulerCache.max_bytes / (1024 * 1024)).toFixed(0)} MB`}
              />
            )}
          </div>
          {schedulerCache.entries_by_type && (
            <div className="grid grid-cols-3 gap-2 text-xs mt-2">
              {(['system', 'user', 'assistant'] as const).map((t) => {
                const n = schedulerCache.entries_by_type?.[t] ?? 0
                const b = schedulerCache.nbytes_by_type?.[t] ?? 0
                if (n === 0 && b === 0) return null
                return (
                  <div key={t} className="bg-background px-2 py-1.5 rounded border border-border">
                    <div className="text-[10px] uppercase text-muted-foreground">{t}</div>
                    <div className="font-mono">{n} entries</div>
                    <div className="font-mono text-[10px] text-muted-foreground">{(b / (1024 * 1024)).toFixed(1)} MB</div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      {/* SSM Companion Cache (hybrid models only).
           A3→A1-001 (audit 2026-04-08): also surface nbytes_mb so users
           can see the real cache memory cost on hybrid models — Nemotron
           120B can silently consume ~32 GB of SSM state beyond the
           prefix-cache budget. Field is provided either via the legacy
           top-level `stats.ssm_companion` shape or the new
           `schedulerCache.ssm_companion_cache` shape (preferred). */}
      {(stats?.ssm_companion || (schedulerCache as any)?.ssm_companion_cache) && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">SSM Companion</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {(() => {
              const ssm =
                (schedulerCache as any)?.ssm_companion_cache ||
                stats?.ssm_companion ||
                {}
              return (
                <>
                  <StatCard
                    label="Entries"
                    value={`${ssm.entries ?? 0} / ${ssm.max_entries ?? 0}`}
                  />
                  {ssm.nbytes_mb != null && ssm.nbytes_mb > 0 && (
                    <StatCard
                      label="SSM Bytes"
                      value={`${ssm.nbytes_mb.toFixed(1)} MB`}
                    />
                  )}
                  {ssm.disk?.total_tokens_on_disk != null && (
                    <StatCard
                      label="SSM Tokens on Disk"
                      value={(ssm.disk.total_tokens_on_disk || 0).toLocaleString()}
                    />
                  )}
                  {ssm.disk?.hits != null && (
                    <StatCard
                      label="SSM L2 Hits / Misses"
                      value={`${ssm.disk.hits || 0} / ${ssm.disk.misses || 0}`}
                    />
                  )}
                </>
              )
            })()}
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
            <StatCard label="Waiting" value={String(schedulerStats.num_waiting || 0)} />
            {schedulerStats.ewma_ttft_seconds != null && (
              <StatCard label="TTFT EWMA" value={`${Number(schedulerStats.ewma_ttft_seconds || 0).toFixed(3)} s`} />
            )}
            <StatCard label="Prompt Tokens" value={(schedulerStats.total_prompt_tokens || 0).toLocaleString()} />
            <StatCard label="Completion Tokens" value={(schedulerStats.total_completion_tokens || 0).toLocaleString()} />
            {schedulerStats.cache_hit_tokens != null && (
              <StatCard label="Cache Hit Tokens" value={(schedulerStats.cache_hit_tokens || 0).toLocaleString()} />
            )}
            {schedulerStats.cache_hit_requests != null && (
              <StatCard label="Cache Hit Requests" value={(schedulerStats.cache_hit_requests || 0).toLocaleString()} />
            )}
            {schedulerStats.hybrid_kv_without_ssm_hits != null && (
              <StatCard label="Hybrid KV-Only Misses" value={(schedulerStats.hybrid_kv_without_ssm_hits || 0).toLocaleString()} />
            )}
            {schedulerStats.hybrid_kv_without_ssm_tokens != null && schedulerStats.hybrid_kv_without_ssm_tokens > 0 && (
              <StatCard label="KV-Only Tokens" value={(schedulerStats.hybrid_kv_without_ssm_tokens || 0).toLocaleString()} />
            )}
            {schedulerStats.cache_reuse_skips != null && (
              <StatCard label="Cache Reuse Skips" value={String(schedulerStats.cache_reuse_skips || 0)} />
            )}
            {schedulerStats.cache_reuse_skip_tokens != null && schedulerStats.cache_reuse_skip_tokens > 0 && (
              <StatCard label="Skipped Hit Tokens" value={(schedulerStats.cache_reuse_skip_tokens || 0).toLocaleString()} />
            )}
            {schedulerStats.cache_reuse_partial_downgrades != null && (
              <StatCard label="Partial Reuse" value={String(schedulerStats.cache_reuse_partial_downgrades || 0)} />
            )}
            {schedulerStats.cache_reuse_partial_tokens != null && schedulerStats.cache_reuse_partial_tokens > 0 && (
              <StatCard label="Partial Hit Tokens" value={(schedulerStats.cache_reuse_partial_tokens || 0).toLocaleString()} />
            )}
          </div>
          {schedulerStats.last_cache_reuse_partial && (
            <div className="mt-2 text-xs bg-accent/10 border border-accent/30 text-foreground px-3 py-2 rounded">
              Cache reuse was memory-fit: used {(schedulerStats.last_cache_reuse_partial.used_cached_tokens ?? 0).toLocaleString()} of {(schedulerStats.last_cache_reuse_partial.original_cached_tokens ?? 0).toLocaleString()} cached tokens,
              estimated merge {schedulerStats.last_cache_reuse_partial.used_needed_mb ?? '?'} MB within {schedulerStats.last_cache_reuse_partial.budget_mb ?? schedulerStats.last_cache_reuse_partial.available_mb ?? '?'} MB budgeted,
              prefilling {(schedulerStats.last_cache_reuse_partial.tail_tokens ?? 0).toLocaleString()} tail tokens.
              {schedulerStats.last_cache_reuse_partial.cache_format && (
                <> Format {schedulerStats.last_cache_reuse_partial.cache_format}.</>
              )}
            </div>
          )}
          {schedulerStats.cache_hit_tokens_by_detail && Object.keys(schedulerStats.cache_hit_tokens_by_detail).length > 0 && (
            <div className="mt-2">
              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">Hit Tokens by Detail</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {Object.entries(schedulerStats.cache_hit_tokens_by_detail).map(([detail, tokens]: [string, any]) => (
                  <StatCard key={detail} label={detail} value={(Number(tokens) || 0).toLocaleString()} />
                ))}
              </div>
            </div>
          )}
          {schedulerStats.last_hybrid_kv_without_ssm && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning-foreground px-3 py-2 rounded">
              Hybrid cache full-prefilled: KV had {(schedulerStats.last_hybrid_kv_without_ssm.cached_tokens ?? 0).toLocaleString()} reusable tokens,
              but no matching SSM companion state ({schedulerStats.last_hybrid_kv_without_ssm.reason || 'missing_ssm'}).
              {schedulerStats.last_hybrid_kv_without_ssm.checkpoint_tokens != null && (
                <> Checkpoint {(schedulerStats.last_hybrid_kv_without_ssm.checkpoint_tokens ?? 0).toLocaleString()} tokens.</>
              )}
            </div>
          )}
          {schedulerStats.last_cache_reuse_skip && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning-foreground px-3 py-2 rounded">
              Cache reuse skipped after partial reuse failed: needed {schedulerStats.last_cache_reuse_skip.needed_mb ?? '?'} MB,
              budget {schedulerStats.last_cache_reuse_skip.budget_mb ?? schedulerStats.last_cache_reuse_skip.available_mb ?? '?'} MB,
              available {schedulerStats.last_cache_reuse_skip.available_mb ?? '?'} MB,
              dropped {(schedulerStats.last_cache_reuse_skip.dropped_cached_tokens ?? schedulerStats.last_cache_reuse_skip.cached_tokens ?? 0).toLocaleString()} cached tokens,
              full-prefilling {(schedulerStats.last_cache_reuse_skip.full_prefill_tokens ?? schedulerStats.last_cache_reuse_skip.prompt_tokens ?? 0).toLocaleString()} tokens.
              {schedulerStats.last_cache_reuse_skip.cache_contract && (
                <> Contract {schedulerStats.last_cache_reuse_skip.cache_contract}.</>
              )}
              {schedulerStats.last_cache_reuse_skip.cache_format && (
                <> Format {schedulerStats.last_cache_reuse_skip.cache_format}.</>
              )}
              {schedulerStats.last_cache_reuse_skip.partial_reuse_unavailable_reason && (
                <> Partial reason: {schedulerStats.last_cache_reuse_skip.partial_reuse_unavailable_reason}.</>
              )}
            </div>
          )}
        </div>
      )}

      {/* KV Quantization Info */}
      {(kvQuant || turboQuantKv || nativeCache) && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Cache Contract</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {nativeCache?.cache_type && (
              <StatCard label="Native Cache" value={nativeCache.cache_type} />
            )}
            {nativeCache?.schema && (
              <StatCard label="Schema" value={nativeCache.schema} />
            )}
            {turboQuantKv && (
              <StatCard
                label="TurboQuant KV"
                value={
                  turboQuantKv.enabled
                    ? turboQuantKv.single_sequence_only
                      ? 'enabled (single-seq)'
                      : 'enabled'
                    : 'disabled'
                }
              />
            )}
            {turboQuantKv?.single_sequence_only && (
              <StatCard
                label="TQ Batch"
                value={`${turboQuantKv.effective_max_num_seqs ?? 1} seq / prefill ${turboQuantKv.effective_prefill_batch_size ?? 1} / decode ${turboQuantKv.effective_completion_batch_size ?? 1}`}
              />
            )}
            {nativeCache?.generic_turboquant_kv && (
              <StatCard
                label="Generic TQ-KV"
                value={
                  nativeCache.generic_turboquant_kv.enabled
                    ? 'enabled'
                    : `off (${nativeCache.generic_turboquant_kv.reason || 'native'})`
                }
              />
            )}
            {attentionKvStorage && (
              <StatCard
                label="Attention KV L2"
                value={
                  attentionKvStorage.enabled
                    ? `q${attentionKvStorage.bits} / group ${attentionKvStorage.group_size ?? 64}`
                    : 'disabled'
                }
              />
            )}
            {attentionKvStorage?.ssm_policy && (
              <StatCard
                label="SSM Policy"
                value={`${attentionKvStorage.ssm_policy}${attentionKvStorage.rederive ? ' + rederive' : ''}`}
              />
            )}
            {kvQuant && (
              <StatCard
                label="Stored KV Quant"
                value={
                  kvQuant?.enabled
                    ? `${kvQuant.bits}-bit / group ${kvQuant.group_size}`
                    : 'disabled'
                }
              />
            )}
            {nativeCache?.components?.length > 0 && (
              <StatCard
                label="Components"
                value={nativeCache.components.join(', ')}
              />
            )}
          </div>
        </div>
      )}

      {/* Disk Cache (L2 prompt-level) */}
      {diskCache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Disk Cache (L2)</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {diskCache.entries != null && <StatCard label="Entries" value={String(diskCache.entries)} />}
            {(diskCache.total_size_mb ?? diskCache.size_mb) != null && <StatCard label="Size" value={`${(diskCache.total_size_mb ?? diskCache.size_mb ?? 0).toFixed(1)} MB`} />}
            {diskCache.total_tokens_on_disk != null && <StatCard label="Tokens on Disk" value={(diskCache.total_tokens_on_disk || 0).toLocaleString()} />}
            {diskCache.hit_rate != null && <StatCard label="Hit Rate" value={`${(diskCache.hit_rate * 100).toFixed(1)}%`} />}
            {diskCache.hits != null && <StatCard label="Hits / Misses" value={`${diskCache.hits} / ${diskCache.misses ?? 0}`} />}
            {diskCache.stores != null && <StatCard label="Stores" value={String(diskCache.stores)} />}
            {diskCache.tq_native_stores != null && diskCache.tq_native_stores > 0 && <StatCard label="TQ-Native Stores" value={String(diskCache.tq_native_stores)} />}
            {diskCache.tq_native_hits != null && diskCache.tq_native_hits > 0 && <StatCard label="TQ-Native Hits" value={String(diskCache.tq_native_hits)} />}
            {diskCache.pending_writes != null && diskCache.pending_writes > 0 && <StatCard label="Pending Writes" value={String(diskCache.pending_writes)} />}
          </div>
        </div>
      )}

      {/* Block Disk Cache (L2 paged blocks) */}
      {stats?.block_disk_cache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Block Disk Cache (L2 Paged)</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <StatCard label="Blocks on Disk" value={String(stats.block_disk_cache.blocks_on_disk ?? 0)} />
            <StatCard label="Disk Size" value={`${(stats.block_disk_cache.disk_size_gb ?? 0).toFixed(2)} GB`} />
            {stats.block_disk_cache.total_tokens_on_disk != null && <StatCard label="Tokens on Disk" value={(stats.block_disk_cache.total_tokens_on_disk || 0).toLocaleString()} />}
            <StatCard label="Disk Hits / Misses" value={`${stats.block_disk_cache.disk_hits ?? 0} / ${stats.block_disk_cache.disk_misses ?? 0}`} />
            <StatCard label="Disk Writes" value={String(stats.block_disk_cache.disk_writes ?? 0)} />
            {(stats.block_disk_cache.disk_evictions ?? 0) > 0 && <StatCard label="Disk Evictions" value={String(stats.block_disk_cache.disk_evictions)} />}
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

      {/* Warm Cache Input */}
      {showWarmInput && (
        <div className="flex gap-2 items-center">
          <input
            type="text"
            value={warmInput}
            onChange={e => setWarmInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && warmInput.trim()) handleWarm() }}
            placeholder="Enter system prompt to warm cache with..."
            autoFocus
            className="flex-1 px-2 py-1.5 text-xs bg-background border border-input rounded focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <button
            onClick={() => { setShowWarmInput(false); setWarmInput('') }}
            className="px-2 py-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            Cancel
          </button>
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
