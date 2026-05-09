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
  scheduler?: {
    num_waiting?: number
    num_running?: number
    ewma_ttft_seconds?: number
    cache_hit_requests?: number
    cache_hit_tokens?: number
    cache_hit_tokens_by_detail?: Record<string, number>
    hybrid_kv_without_ssm_hits?: number
    hybrid_kv_without_ssm_tokens?: number
    last_hybrid_kv_without_ssm?: {
      reason?: string
      cached_tokens?: number
      checkpoint_tokens?: number
    } | null
    cache_reuse_skips?: number
    cache_reuse_skip_tokens?: number
    last_cache_reuse_skip?: {
      reason?: string
      action?: string
      needed_mb?: number
      budget_mb?: number
      available_mb?: number
      cache_mb?: number
      budget_fraction?: number
      cached_tokens?: number
      dropped_cached_tokens?: number
      full_prefill_tokens?: number
      prompt_tokens?: number
      cache_contract?: string
      partial_reuse_unavailable_reason?: string
    } | null
    cache_reuse_partial_downgrades?: number
    cache_reuse_partial_tokens?: number
    last_cache_reuse_partial?: {
      reason?: string
      original_needed_mb?: number
      budget_mb?: number
      available_mb?: number
      original_cache_mb?: number
      used_cache_mb?: number
      used_needed_mb?: number
      budget_fraction?: number
      original_cached_tokens?: number
      used_cached_tokens?: number
      dropped_cached_tokens?: number
      tail_tokens?: number
      cache_contract?: string
    } | null
  }
  cache?: {
    scheduler_cache?: {
      total_tokens_cached?: number
      tokens_saved?: number
      allocated_blocks?: number
    }
    disk_cache?: {
      entries?: number
      total_tokens_on_disk?: number
      hits?: number
      misses?: number
    }
    block_disk_cache?: {
      blocks_on_disk?: number
      total_tokens_on_disk?: number
      disk_hits?: number
      disk_misses?: number
    }
    ssm_companion?: {
      entries?: number
      max_entries?: number
      disk?: {
        entries?: number
        total_tokens_on_disk?: number
        hits?: number
        misses?: number
      }
    }
    totals?: {
      ram_tokens_cached?: number
      l2_prompt_tokens_on_disk?: number
      l2_block_tokens_on_disk?: number
      ssm_tokens_on_disk?: number
      l2_tokens_on_disk?: number
      l2_tokens_on_disk_store_sum?: number
      l2_tokens_on_disk_note?: string
    }
  }
  kv_cache_quantization?: {
    enabled: boolean
    bits?: number
    group_size?: number
  }
  turboquant_kv_cache?: {
    enabled: boolean
    single_sequence_only?: boolean
    effective_max_num_seqs?: number
    effective_prefill_batch_size?: number
    effective_completion_batch_size?: number
  }
  native_cache?: {
    family?: string
    schema?: string
    cache_type?: string
    generic_turboquant_kv?: {
      enabled?: boolean
      reason?: string
    }
  }
  quantization_format?: {
    type: string
    target_bits?: number
    actual_bits?: number
    block_size?: number
  }
  quantization?: {
    codec?: string
    weight_format?: string
    backend?: string
    profile?: string
    mxtq_bits?: number
    mxtq_bits_by_role?: Record<string, number>
    routed_expert_bits?: number
    routed_expert_bits_by_projection?: Record<string, number>
    routed_expert_bits_label?: string
    target_bits?: number
    actual_bits?: number
    config_bits?: number
    passthrough_bit_widths_used?: number[]
    passthrough_tensor_count?: number
    compat_warnings?: string[]
    sidecar?: {
      jang_config?: boolean
      jangtq_runtime?: boolean
      prestacked_bundle?: boolean
    }
  }
  acceleration?: {
    kernel_type?: string
    metal_na_capable?: boolean
    metal_na_active_on_host?: boolean
    reason?: string
    metal_na_symbols?: {
      available?: boolean
      nax_symbols?: number
      naxtile_symbols?: number
    }
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
            {health.quantization_format && (
              <InfoCard
                label="Weight Quant"
                value={`${(health.quantization_format.type || 'JANG').toUpperCase()} ${health.quantization_format.actual_bits ?? health.quantization_format.target_bits}-bit`}
              />
            )}
            {health.quantization?.codec && (
              <InfoCard
                label="Weight Codec"
                value={
                  health.quantization.codec === 'turboquant_codebook'
                    ? health.quantization.routed_expert_bits_label
                      ? `JANGTQ ${health.quantization.routed_expert_bits_label}`
                      : `JANGTQ ${health.quantization.routed_expert_bits ?? health.quantization.mxtq_bits ?? health.quantization.actual_bits ?? health.quantization.target_bits ?? '-'}-bit`
                    : health.quantization.codec
                }
              />
            )}
            {health.quantization?.codec === 'turboquant_codebook' && (
              <InfoCard
                label="JANGTQ Layout"
                value={
                  health.quantization.sidecar?.prestacked_bundle
                    ? 'pre-stacked bundle'
                    : health.quantization.sidecar?.jangtq_runtime
                      ? 'runtime sidecar'
                      : health.quantization.sidecar?.jang_config
                        ? 'config only'
                        : 'unknown'
                }
              />
            )}
            {health.quantization?.passthrough_tensor_count ? (
              <InfoCard
                label="F16 Passthrough"
                value={`${health.quantization.passthrough_tensor_count} tensors (${(health.quantization.passthrough_bit_widths_used || []).join('/') || 16}-bit)`}
              />
            ) : null}
            {health.acceleration?.kernel_type && (
              <InfoCard
                label="Metal NA"
                value={
                  health.acceleration.metal_na_active_on_host
                    ? 'active'
                    : health.acceleration.kernel_type === 'turboquant_codebook'
                      ? 'not used by JANGTQ'
                      : health.acceleration.metal_na_capable
                        ? 'unavailable'
                        : 'not applicable'
                }
              />
            )}
            {health.kv_cache_quantization?.enabled && (
              <InfoCard label="KV Quant" value={`${health.kv_cache_quantization.bits}-bit`} />
            )}
            {health.native_cache?.cache_type && (
              <InfoCard label="Native Cache" value={health.native_cache.cache_type} />
            )}
            {health.turboquant_kv_cache && (
              <InfoCard
                label="TQ-KV"
                value={
                  health.turboquant_kv_cache.enabled
                    ? health.turboquant_kv_cache.single_sequence_only
                      ? 'enabled (single-seq)'
                      : 'enabled'
                    : 'disabled'
                }
              />
            )}
            {health.turboquant_kv_cache?.single_sequence_only && (
              <InfoCard
                label="TQ Batch"
                value={`${health.turboquant_kv_cache.effective_max_num_seqs ?? 1} seq / prefill ${health.turboquant_kv_cache.effective_prefill_batch_size ?? 1} / decode ${health.turboquant_kv_cache.effective_completion_batch_size ?? 1}`}
              />
            )}
            {health.native_cache?.generic_turboquant_kv && (
              <InfoCard
                label="Generic TQ-KV"
                value={
                  health.native_cache.generic_turboquant_kv.enabled
                    ? 'enabled'
                    : `off: ${health.native_cache.generic_turboquant_kv.reason || 'native'}`
                }
              />
            )}
          </div>
          {health.quantization?.compat_warnings?.length ? (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning px-3 py-2 rounded space-y-1">
              {health.quantization.compat_warnings.map((warning, index) => (
                <div key={index}>{warning}</div>
              ))}
            </div>
          ) : null}
        </div>
      )}

      {/* Scheduler */}
      {health?.scheduler && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Scheduler</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <InfoCard
              label="Queue"
              value={`${health.scheduler.num_running ?? 0} running / ${health.scheduler.num_waiting ?? 0} waiting`}
            />
            {health.scheduler.ewma_ttft_seconds != null && (
              <InfoCard
                label="TTFT EWMA"
                value={`${Number(health.scheduler.ewma_ttft_seconds || 0).toFixed(3)} s`}
              />
            )}
            {health.scheduler.cache_hit_tokens != null && (
              <InfoCard
                label="Cache Hit Tokens"
                value={(health.scheduler.cache_hit_tokens || 0).toLocaleString()}
              />
            )}
            {health.scheduler.cache_hit_requests != null && (
              <InfoCard
                label="Cache Hit Requests"
                value={(health.scheduler.cache_hit_requests || 0).toLocaleString()}
              />
            )}
            {health.scheduler.hybrid_kv_without_ssm_hits != null && (
              <InfoCard
                label="Hybrid KV-Only Misses"
                value={(health.scheduler.hybrid_kv_without_ssm_hits || 0).toLocaleString()}
              />
            )}
            {health.scheduler.hybrid_kv_without_ssm_tokens != null && health.scheduler.hybrid_kv_without_ssm_tokens > 0 && (
              <InfoCard
                label="KV-Only Tokens"
                value={(health.scheduler.hybrid_kv_without_ssm_tokens || 0).toLocaleString()}
              />
            )}
            {health.scheduler.cache_reuse_skips != null && (
              <InfoCard label="Cache Skips" value={String(health.scheduler.cache_reuse_skips || 0)} />
            )}
            {health.scheduler.cache_reuse_skip_tokens != null && health.scheduler.cache_reuse_skip_tokens > 0 && (
              <InfoCard
                label="Skipped Tokens"
                value={(health.scheduler.cache_reuse_skip_tokens || 0).toLocaleString()}
              />
            )}
            {health.scheduler.cache_reuse_partial_downgrades != null && (
              <InfoCard
                label="Partial Reuse"
                value={String(health.scheduler.cache_reuse_partial_downgrades || 0)}
              />
            )}
            {health.scheduler.cache_reuse_partial_tokens != null && health.scheduler.cache_reuse_partial_tokens > 0 && (
              <InfoCard
                label="Partial Hit Tokens"
                value={(health.scheduler.cache_reuse_partial_tokens || 0).toLocaleString()}
              />
            )}
          </div>
          {health.scheduler.last_cache_reuse_partial && (
            <div className="mt-2 text-xs bg-accent/10 border border-accent/30 text-foreground px-3 py-2 rounded">
              Cache reuse was memory-fit: used {(health.scheduler.last_cache_reuse_partial.used_cached_tokens ?? 0).toLocaleString()} of {(health.scheduler.last_cache_reuse_partial.original_cached_tokens ?? 0).toLocaleString()} cached tokens,
              estimated merge {health.scheduler.last_cache_reuse_partial.used_needed_mb ?? '?'} MB within {health.scheduler.last_cache_reuse_partial.budget_mb ?? health.scheduler.last_cache_reuse_partial.available_mb ?? '?'} MB budgeted,
              prefilling {(health.scheduler.last_cache_reuse_partial.tail_tokens ?? 0).toLocaleString()} tail tokens.
            </div>
          )}
          {health.scheduler.last_hybrid_kv_without_ssm && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning px-3 py-2 rounded">
              Hybrid cache full-prefilled: KV had {(health.scheduler.last_hybrid_kv_without_ssm.cached_tokens ?? 0).toLocaleString()} reusable tokens,
              but no matching SSM companion state ({health.scheduler.last_hybrid_kv_without_ssm.reason || 'missing_ssm'}).
              {health.scheduler.last_hybrid_kv_without_ssm.checkpoint_tokens != null && (
                <> Checkpoint {(health.scheduler.last_hybrid_kv_without_ssm.checkpoint_tokens ?? 0).toLocaleString()} tokens.</>
              )}
            </div>
          )}
          {health.scheduler.last_cache_reuse_skip && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning px-3 py-2 rounded">
              Cache reuse skipped after partial reuse failed: needed {health.scheduler.last_cache_reuse_skip.needed_mb ?? '?'} MB,
              budget {health.scheduler.last_cache_reuse_skip.budget_mb ?? health.scheduler.last_cache_reuse_skip.available_mb ?? '?'} MB,
              available {health.scheduler.last_cache_reuse_skip.available_mb ?? '?'} MB,
              dropped {(health.scheduler.last_cache_reuse_skip.dropped_cached_tokens ?? health.scheduler.last_cache_reuse_skip.cached_tokens ?? 0).toLocaleString()} cached tokens,
              full-prefilling {(health.scheduler.last_cache_reuse_skip.full_prefill_tokens ?? health.scheduler.last_cache_reuse_skip.prompt_tokens ?? 0).toLocaleString()} tokens.
              {health.scheduler.last_cache_reuse_skip.cache_contract && (
                <> Contract {health.scheduler.last_cache_reuse_skip.cache_contract}.</>
              )}
              {health.scheduler.last_cache_reuse_skip.partial_reuse_unavailable_reason && (
                <> Partial reason: {health.scheduler.last_cache_reuse_skip.partial_reuse_unavailable_reason}.</>
              )}
            </div>
          )}
        </div>
      )}

      {/* Cache */}
      {health?.cache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Cache</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {health.cache.totals?.ram_tokens_cached != null && (
              <InfoCard
                label="RAM Cached Tokens"
                value={(health.cache.totals.ram_tokens_cached || 0).toLocaleString()}
              />
            )}
            {(health.cache.totals?.l2_tokens_on_disk_store_sum ?? health.cache.totals?.l2_tokens_on_disk) != null && (
              <InfoCard
                label="L2 Token Entries"
                value={`${(health.cache.totals?.l2_tokens_on_disk_store_sum ?? health.cache.totals?.l2_tokens_on_disk ?? 0).toLocaleString()} store-sum`}
              />
            )}
            {health.cache.disk_cache?.entries != null && (
              <InfoCard
                label="Prompt L2"
                value={`${health.cache.disk_cache.entries || 0} entries / ${(health.cache.disk_cache.total_tokens_on_disk || 0).toLocaleString()} tokens`}
              />
            )}
            {health.cache.block_disk_cache?.blocks_on_disk != null && (
              <InfoCard
                label="Paged L2"
                value={`${health.cache.block_disk_cache.blocks_on_disk || 0} blocks / ${(health.cache.block_disk_cache.total_tokens_on_disk || 0).toLocaleString()} tokens`}
              />
            )}
            {health.cache.ssm_companion?.disk?.entries != null && (
              <InfoCard
                label="SSM L2"
                value={`${health.cache.ssm_companion.disk.entries || 0} entries / ${(health.cache.ssm_companion.disk.total_tokens_on_disk || 0).toLocaleString()} tokens`}
              />
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
        stroke="rgb(var(--primary))"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      {/* Peak line */}
      <line
        x1="0" y1={h - (data[data.length - 1].peak / maxVal) * h}
        x2={w} y2={h - (data[data.length - 1].peak / maxVal) * h}
        stroke="rgb(var(--destructive))"
        strokeWidth="0.5"
        strokeDasharray="4 2"
        opacity="0.5"
      />
      {/* Label */}
      <text x={w - 2} y={10} textAnchor="end" fontSize="8" fill="rgb(var(--muted-foreground))">
        {(maxVal >= 1024 ? (maxVal / 1024).toFixed(1) + ' GB' : maxVal.toFixed(0) + ' MB')}
      </text>
    </svg>
  )
}
