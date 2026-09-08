import { useState, useEffect, useRef, useCallback } from 'react'
import { describeDsv4ActivationQat } from './dsv4QatStatus'
import { useTranslation } from '../../i18n'

export interface CachePanelRequestToken {
  identityGeneration: number
  requestGeneration: number
}

export function formatCacheStorageBytes(rawBytes: unknown): string {
  const bytes = typeof rawBytes === 'number' && Number.isFinite(rawBytes)
    ? Math.max(0, rawBytes)
    : 0
  if (bytes < 1024) return `${Math.round(bytes)} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

/**
 * Keeps cache-panel async work scoped to the current session lifecycle and
 * makes cache-stat refreshes last-request-wins.
 */
export class CachePanelRequestGuard {
  private identityGeneration = 0
  private requestGeneration = 0
  private activeActionRequestGeneration: number | null = null

  captureIdentity(): number {
    return this.identityGeneration
  }

  resetIdentity(): void {
    this.identityGeneration += 1
    this.requestGeneration += 1
    this.activeActionRequestGeneration = null
  }

  isIdentityCurrent(identityGeneration: number): boolean {
    return identityGeneration === this.identityGeneration
  }

  beginLatest(identityGeneration = this.identityGeneration): CachePanelRequestToken | null {
    if (
      !this.isIdentityCurrent(identityGeneration)
      || this.activeActionRequestGeneration !== null
    ) return null
    this.requestGeneration += 1
    return {
      identityGeneration,
      requestGeneration: this.requestGeneration,
    }
  }

  beginAction(identityGeneration = this.identityGeneration): CachePanelRequestToken | null {
    if (
      !this.isIdentityCurrent(identityGeneration)
      || this.activeActionRequestGeneration !== null
    ) return null
    this.requestGeneration += 1
    this.activeActionRequestGeneration = this.requestGeneration
    return {
      identityGeneration,
      requestGeneration: this.requestGeneration,
    }
  }

  finishAction(token: CachePanelRequestToken): void {
    if (
      this.isCurrent(token)
      && this.activeActionRequestGeneration === token.requestGeneration
    ) {
      this.activeActionRequestGeneration = null
    }
  }

  invalidateRequests(): void {
    this.requestGeneration += 1
    this.activeActionRequestGeneration = null
  }

  isCurrent(token: CachePanelRequestToken): boolean {
    return (
      token.identityGeneration === this.identityGeneration
      && token.requestGeneration === this.requestGeneration
    )
  }
}

interface CachePanelProps {
  endpoint: { host: string; port: number }
  sessionStatus: string
  sessionId?: string
}

export function CachePanel({ endpoint, sessionStatus, sessionId }: CachePanelProps) {
  const { t } = useTranslation()
  const [stats, setStats] = useState<any>(null)
  const [entries, setEntries] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showEntries, setShowEntries] = useState(false)
  const [warming, setWarming] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [warmInput, setWarmInput] = useState('')
  const [showWarmInput, setShowWarmInput] = useState(false)
  const requestGuardRef = useRef<CachePanelRequestGuard | null>(null)
  const warmInputGenerationRef = useRef(0)
  if (!requestGuardRef.current) {
    requestGuardRef.current = new CachePanelRequestGuard()
  }
  const requestGuard = requestGuardRef.current
  const identityKey = `${endpoint.host}:${endpoint.port}:${sessionId ?? ''}:${sessionStatus}`
  const identityKeyRef = useRef(identityKey)
  // Invalidate old async work during render, before passive effects run. This
  // closes the commit-to-effect window where a completion from the previous
  // session could otherwise update the newly-rendered panel.
  if (identityKeyRef.current !== identityKey) {
    requestGuard.resetIdentity()
    identityKeyRef.current = identityKey
  }

  const fetchStatsForToken = useCallback(async (requestToken: CachePanelRequestToken) => {
    if (sessionStatus !== 'running' && sessionStatus !== 'standby') return
    try {
      const s = await window.api.cache.stats(endpoint, sessionId)
      if (!requestGuard.isCurrent(requestToken)) return
      setStats(s)
      setError(null)
    } catch (err: any) {
      if (!requestGuard.isCurrent(requestToken)) return
      setError(err.message || t('sessions.cache.fetchStatsFailed'))
    }
  }, [endpoint.host, endpoint.port, requestGuard, sessionId, sessionStatus, t])

  const fetchStats = useCallback(async (
    expectedIdentity = requestGuard.captureIdentity(),
  ) => {
    const requestToken = requestGuard.beginLatest(expectedIdentity)
    if (!requestToken) return
    await fetchStatsForToken(requestToken)
  }, [fetchStatsForToken, requestGuard])

  // A reused SessionView keeps this component instance alive when its session
  // changes. Reset all panel state and invalidate every old async completion.
  useEffect(() => {
    setStats(null)
    setEntries(null)
    setLoading(false)
    setError(null)
    setShowEntries(false)
    setWarming(false)
    setClearing(false)
    setWarmInput('')
    setShowWarmInput(false)

    return () => {
      requestGuard.resetIdentity()
    }
  }, [endpoint.host, endpoint.port, requestGuard, sessionId, sessionStatus])

  // Poll stats every 5 seconds
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null
    if (sessionStatus === 'running') {
      fetchStats()
      interval = setInterval(fetchStats, 5000)
    }
    return () => {
      if (interval) clearInterval(interval)
      requestGuard.invalidateRequests()
    }
  }, [fetchStats, requestGuard, sessionStatus])

  const handleFetchEntries = async () => {
    const identity = requestGuard.captureIdentity()
    const actionToken = requestGuard.beginAction(identity)
    if (!actionToken) return
    setLoading(true)
    try {
      const e = await window.api.cache.entries(endpoint, sessionId)
      if (!requestGuard.isCurrent(actionToken)) return
      setEntries(e)
      setShowEntries(true)
    } catch (err: any) {
      if (!requestGuard.isCurrent(actionToken)) return
      setError(err.message)
    } finally {
      if (requestGuard.isCurrent(actionToken)) setLoading(false)
      requestGuard.finishAction(actionToken)
    }
  }

  const handleWarm = async () => {
    if (!warmInput.trim()) {
      setShowWarmInput(true)
      return
    }
    const identity = requestGuard.captureIdentity()
    const actionToken = requestGuard.beginAction(identity)
    if (!actionToken) return
    const submittedInput = warmInput.trim()
    const submittedInputGeneration = warmInputGenerationRef.current
    setWarming(true)
    try {
      await window.api.cache.warm([submittedInput], endpoint, sessionId)
      if (!requestGuard.isCurrent(actionToken)) return
      await fetchStatsForToken(actionToken)
      if (!requestGuard.isCurrent(actionToken)) return
      if (warmInputGenerationRef.current === submittedInputGeneration) {
        setWarmInput('')
        setShowWarmInput(false)
      }
    } catch (err: any) {
      if (!requestGuard.isCurrent(actionToken)) return
      setError(err.message)
    } finally {
      if (requestGuard.isCurrent(actionToken)) setWarming(false)
      requestGuard.finishAction(actionToken)
    }
  }

  const handleClear = async (type: string) => {
    const identity = requestGuard.captureIdentity()
    const actionToken = requestGuard.beginAction(identity)
    if (!actionToken) return
    setClearing(true)
    try {
      const res: any = await window.api.cache.clear(type, endpoint, sessionId)
      if (!requestGuard.isCurrent(actionToken)) return
      // The engine answers with what it cleared AND what it refused to touch:
      // `paged_prefix:blocks_in_use` means a live request still holds those
      // blocks and nothing was freed there, and a `busy` status means nothing
      // was cleared at all. Discarding the response rendered both as success —
      // a spinner, a refresh, and a cache the user believes is gone.
      const skipped: string[] = Array.isArray(res?.skipped) ? res.skipped : []
      if (res?.status === 'busy' || skipped.length > 0) {
        setError(t('sessions.cachePanel.clearSkipped', { skipped: skipped.join(', ') || res?.detail || '' }))
      }
      await fetchStatsForToken(actionToken)
      if (!requestGuard.isCurrent(actionToken)) return
      setEntries(null)
    } catch (err: any) {
      if (!requestGuard.isCurrent(actionToken)) return
      setError(err.message)
    } finally {
      if (requestGuard.isCurrent(actionToken)) setClearing(false)
      requestGuard.finishAction(actionToken)
    }
  }

  if (sessionStatus !== 'running') {
    return (
      <div className="text-sm text-muted-foreground p-4">
        {t('sessions.cachePanel.sessionNotRunning')}
      </div>
    )
  }

  const schedulerCache = stats?.scheduler_cache
  const schedulerStats = stats?.scheduler_stats
  const visionMemoryCache =
    schedulerStats?.vision_cache ??
    schedulerCache?.vision_cache ??
    (schedulerCache?.pixel_cache_size != null || schedulerCache?.pixel_cache_hits != null
      ? schedulerCache
      : null)
  const lastCacheExecution =
    schedulerStats?.last_cache_execution ??
    schedulerStats?.batch_generator?.last_cache_execution
  // the last completed generation's terminal fence (request-exact; engine field last_durability)
  const lastDurability = (schedulerStats?.batch_generator?.last_durability ?? schedulerStats?.last_durability) as
    | { request_id?: string; wait_ms?: number; waited?: boolean; cache_outcome?: string; retained_tokens?: number | null; detail?: string; at?: number }
    | null
    | undefined
  const lastCacheSelection =
    lastCacheExecution?.selection ??
    schedulerStats?.last_cache_selection
  const diskCache = stats?.disk_cache
  const kvQuant = stats?.kv_cache_quantization
  const nativeCache = stats?.native_cache
  const dsv4ActivationQatDisplay = nativeCache?.activation_qat
    ? describeDsv4ActivationQat(nativeCache.activation_qat)
    : null
  const turboQuantKv = stats?.turboquant_kv_cache
  const cacheTotals = stats?.cache_totals
  const blockDiskCache = stats?.block_disk_cache
  const globalBlockDiskBudget = blockDiskCache?.global_budget
  const actionBusy = loading || warming || clearing
  const attentionKvStorage =
    nativeCache?.attention_kv_storage_quantization ??
    nativeCache?.storage_quantization
  const tqKeyBits = turboQuantKv?.key_bits_values?.length
    ? turboQuantKv.key_bits_values.map((bits: number) => `q${bits}`).join('/')
    : `q${turboQuantKv?.storage_key_bits ?? '?'}`
  const tqValueBits = turboQuantKv?.value_bits_values?.length
    ? turboQuantKv.value_bits_values.map((bits: number) => `q${bits}`).join('/')
    : `q${turboQuantKv?.storage_value_bits ?? '?'}`
  const tqStoredPrefix = turboQuantKv?.storage_encode_enabled
    ? `${turboQuantKv.stored_prefix_quantization ?? 'TurboQuant'} (K ${tqKeyBits} / V ${tqValueBits})`
    : null
  const runtimeCacheObjects = nativeCache?.runtime_cache_effective_class_counts
    ? Object.entries(nativeCache.runtime_cache_effective_class_counts)
        .map(([className, count]) => `${className} × ${Number(count)}`)
        .join(', ')
    : ''
  const runtimeCacheOwners = nativeCache?.runtime_cache_owner_component_class_counts
    ? Object.entries(nativeCache.runtime_cache_owner_component_class_counts)
        .map(([className, count]) => `${className} × ${Number(count)}`)
        .join(', ')
    : ''
  const dtypeHarmonization = nativeCache?.parameter_dtype_harmonization
  const formatDtypeCounts = (counts: any): string => counts && typeof counts === 'object'
    ? Object.entries(counts)
        .map(([dtype, count]) => `${dtype} × ${Number(count)}`)
        .join(', ')
    : ''
  const storedAttentionDtypes = formatDtypeCounts(
    blockDiskCache?.latest_payload?.original_attention_kv_dtype_counts,
  )
  const physicalBlockDtypes = formatDtypeCounts(
    blockDiskCache?.latest_payload?.physical_tensor_dtype_counts,
  )
  const displayComponents = Array.isArray(nativeCache?.components)
    ? nativeCache.components.map((component: string) => (
        component === 'ssm_companion_state'
          ? 'recurrent_companion_state'
          : component
      ))
    : []

  return (
    <div className="space-y-4">
      {error && (
        <div className="text-xs text-destructive bg-destructive/10 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {cacheTotals && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.cachePanel.cacheTotals')}</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {cacheTotals.retained_cache_bytes != null && (
              <StatCard
                label={t('sessions.cache.retainedCacheRam')}
                value={formatCacheStorageBytes(cacheTotals.retained_cache_bytes)}
              />
            )}
            {cacheTotals.retained_cache_ram_enabled != null && (
              <StatCard
                label={t('sessions.cache.retainedRamPolicy')}
                value={cacheTotals.retained_cache_ram_enabled
                  ? t('sessions.cache.statusEnabled')
                  : t('sessions.cache.statusDisabled')}
              />
            )}
            {visionMemoryCache?.enabled != null && (
              <StatCard
                label={t('sessions.cache.mediaRamTier')}
                value={visionMemoryCache.enabled
                  ? t('sessions.cache.statusEnabled')
                  : t('sessions.cache.statusDisabled')}
              />
            )}
            {visionMemoryCache?.retained_bytes != null && (
              <StatCard
                label={t('sessions.cache.mediaRamBytes')}
                value={formatCacheStorageBytes(visionMemoryCache.retained_bytes)}
              />
            )}
            {visionMemoryCache?.pixel_cache_size != null && (
              <StatCard
                label={t('sessions.cache.mediaRamEntries')}
                value={`${Number(visionMemoryCache.pixel_cache_size || 0).toLocaleString()} / ${Number(visionMemoryCache.max_entries || 0).toLocaleString()}`}
              />
            )}
            {cacheTotals.ram_tokens_cached != null && (
              <StatCard label={t('sessions.cache.ramResidentTokens')} value={(cacheTotals.ram_tokens_cached || 0).toLocaleString()} />
            )}
            {cacheTotals.l1_indexed_tokens != null && (
              <StatCard label={t('sessions.cache.l1IndexedTokens')} value={(cacheTotals.l1_indexed_tokens || 0).toLocaleString()} />
            )}
            {cacheTotals.l1_resident_bytes_mb != null && (
              <StatCard
                label={t('sessions.cache.l1ResidentMemory')}
                value={`${(cacheTotals.l1_resident_bytes_mb || 0).toFixed(1)} / ${(cacheTotals.l1_max_resident_bytes_mb || 0).toFixed(1)} MB`}
              />
            )}
            {cacheTotals.l1_evictions != null && (
              <StatCard label={t('sessions.cache.l1Evictions')} value={(cacheTotals.l1_evictions || 0).toLocaleString()} />
            )}
            {cacheTotals.l2_tokens_on_disk != null && (
              <StatCard label={t('sessions.cache.l2TokensOnDisk')} value={(cacheTotals.l2_tokens_on_disk || 0).toLocaleString()} />
            )}
            {cacheTotals.l2_prompt_tokens_on_disk != null && (
              <StatCard label={t('sessions.cache.promptL2Tokens')} value={(cacheTotals.l2_prompt_tokens_on_disk || 0).toLocaleString()} />
            )}
            {cacheTotals.l2_block_tokens_on_disk != null && (
              <StatCard label={t('sessions.cache.blockL2Tokens')} value={(cacheTotals.l2_block_tokens_on_disk || 0).toLocaleString()} />
            )}
            {(() => {
              const ssmL2Tokens = cacheTotals.l2_ssm_tokens_on_disk ?? cacheTotals.ssm_tokens_on_disk
              return ssmL2Tokens != null && ssmL2Tokens > 0 ? (
                <StatCard label={t('sessions.cache.ssmL2Tokens')} value={(ssmL2Tokens || 0).toLocaleString()} />
              ) : null
            })()}
          </div>
        </div>
      )}

      {/* Cache Stats Overview */}
      {schedulerCache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.cachePanel.prefixCache')}</h4>
          {/* The old copy promised "longest continuous causal prefix" without
              qualification, which overpromises for path-dependent families:
              MEASURED, a prompt shortened by ~130 tokens — well inside one 256
              block — reused NOTHING, while a same-length divergent tail
              correctly reused 1792/1878. Also the only untranslated string on
              this panel. */}
          <p className="mb-2 text-xs text-muted-foreground">
            {t('sessions.cachePanel.reuseExplainer')}
          </p>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {schedulerCache.hit_rate != null && (
              <StatCard label={t('sessions.cache.hitRate')} value={`${(schedulerCache.hit_rate * 100).toFixed(1)}%`} />
            )}
            {(schedulerCache.entry_count ?? schedulerCache.entries) != null && (
              <StatCard label={t('sessions.cache.entries')} value={String(schedulerCache.entry_count ?? schedulerCache.entries)} />
            )}
            {(schedulerCache.current_memory_mb ?? schedulerCache.memory_mb) != null && (
              <StatCard label={t('sessions.cache.memory')} value={`${(schedulerCache.current_memory_mb ?? schedulerCache.memory_mb).toFixed(1)} MB`} />
            )}
            {schedulerCache.hits != null && (
              <StatCard label={t('sessions.cache.hitsMisses')} value={`${schedulerCache.hits} / ${schedulerCache.misses || 0}`} />
            )}
            {/* Paged/native backends never drive allocated_blocks / utilization /
                tokens_saved-as-residency (allocator counters stay pinned at the
                null block; see PagedCacheManager._live_cached_blocks). Prefer the
                live-derived fields and fall back to legacy ones only when the
                paged fields are absent. */}
            {(schedulerCache.total_tokens_cached ?? schedulerCache.tokens_saved) != null && (
              <StatCard
                label={t('sessions.cache.cachedTokens')}
                value={(schedulerCache.total_tokens_cached ?? schedulerCache.tokens_saved).toLocaleString()}
              />
            )}
            {schedulerCache.total_tokens_cached != null && schedulerCache.tokens_saved != null && (
              <StatCard label={t('sessions.cache.tokensSaved')} value={schedulerCache.tokens_saved.toLocaleString()} />
            )}
            {schedulerCache.evictions != null && (
              <StatCard label={t('sessions.cache.evictions')} value={String(schedulerCache.evictions)} />
            )}
            {schedulerCache.block_size != null && (
              <StatCard label={t('sessions.cache.blockSize')} value={t('chat.bubble.tokensSuffix', { n: schedulerCache.block_size })} />
            )}
            {(schedulerCache.cached_blocks ?? schedulerCache.allocated_blocks) != null && (
              <StatCard
                label={t('sessions.cache.blocks')}
                value={t('sessions.cache.blocksValue', {
                  cached: schedulerCache.cached_blocks ?? schedulerCache.allocated_blocks,
                  max: schedulerCache.max_blocks,
                  shared: schedulerCache.shared_blocks ?? 0,
                })}
              />
            )}
            {(schedulerCache.cache_occupancy ?? schedulerCache.utilization) != null && (
              <StatCard
                label={t('sessions.cache.utilization')}
                value={`${((schedulerCache.cache_occupancy ?? schedulerCache.utilization) * 100).toFixed(1)}%`}
              />
            )}
            {!blockDiskCache && schedulerCache.disk_hits != null && (schedulerCache.disk_hits > 0 || schedulerCache.disk_misses > 0) && (
              <StatCard
                label={t('sessions.cache.l2DiskHits')}
                value={t('sessions.cache.l2DiskHitsValue', { hits: schedulerCache.disk_hits, misses: schedulerCache.disk_misses ?? 0 })}
              />
            )}
            {schedulerCache.cow_copies != null && schedulerCache.cow_copies > 0 && (
              <StatCard label={t('sessions.cache.cowCopies')} value={String(schedulerCache.cow_copies)} />
            )}
            {schedulerCache.reconstruct_memo_allowed != null && (
              <StatCard
                label={t('sessions.cache.reconstructMemo')}
                value={schedulerCache.reconstruct_memo_allowed
                  ? (schedulerCache.reconstruct_memo_resident
                      ? t('sessions.cache.reconstructMemoResident')
                      : t('sessions.cache.reconstructMemoEmpty'))
                  : t('sessions.cache.reconstructMemoDisabledSsd')}
              />
            )}
            {/* F4 (audit 2026-04-08): Agent 1's PrefixCacheManager
                 cache_type LRU exposes per-type byte / entry counts.
                 Display them when present so users can see system /
                 user / assistant priority pinning at a glance. */}
            {schedulerCache.max_bytes != null && schedulerCache.max_bytes > 0 && schedulerCache.nbytes != null && (
              <StatCard
                label={t('sessions.cache.cacheBytes')}
                value={`${(schedulerCache.nbytes / (1024 * 1024)).toFixed(1)} / ${(schedulerCache.max_bytes / (1024 * 1024)).toFixed(0)} MB`}
              />
            )}
          </div>
          {schedulerCache.entries_by_type && (
            <div className="grid grid-cols-3 gap-2 text-xs mt-2">
              {(['system', 'user', 'assistant'] as const).map((entryType) => {
                const n = schedulerCache.entries_by_type?.[entryType] ?? 0
                const b = schedulerCache.nbytes_by_type?.[entryType] ?? 0
                if (n === 0 && b === 0) return null
                return (
                  <div key={entryType} className="bg-background px-2 py-1.5 rounded border border-border">
                    <div className="text-[10px] uppercase text-muted-foreground">{t(`sessions.cache.entryType.${entryType}`)}</div>
                    <div className="font-mono">{t('sessions.cache.entriesCount', { n })}</div>
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
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.cachePanel.ssmCompanion')}</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {(() => {
              const ssm =
                (schedulerCache as any)?.ssm_companion_cache ||
                stats?.ssm_companion ||
                {}
              return (
                <>
                  <StatCard
                    label={t('sessions.cache.entries')}
                    value={`${ssm.entries ?? 0} / ${ssm.max_entries ?? 0}`}
                  />
                  <StatCard
                    label={t('sessions.cache.ssmRamTier')}
                    value={ssm.ram_enabled
                      ? t('sessions.cache.statusEnabled')
                      : t('sessions.cache.statusDisabled')}
                  />
                  <StatCard
                    label={t('sessions.cache.ssmBytes')}
                    value={ssm.max_bytes != null && ssm.max_bytes > 0
                      ? `${formatCacheStorageBytes(ssm.nbytes)} / ${formatCacheStorageBytes(ssm.max_bytes)}`
                      : formatCacheStorageBytes(ssm.nbytes)}
                  />
                  {ssm.evictions != null && (
                    <StatCard
                      label={t('sessions.cache.ssmEvictions')}
                      value={ssm.evicted_bytes_mb != null && ssm.evicted_bytes_mb > 0
                        ? `${ssm.evictions} / ${ssm.evicted_bytes_mb.toFixed(1)} MB`
                        : String(ssm.evictions)}
                    />
                  )}
                  {ssm.disk?.total_tokens_on_disk != null && (
                    <StatCard
                      label={t('sessions.cache.ssmTokensOnDisk')}
                      value={(ssm.disk.total_tokens_on_disk || 0).toLocaleString()}
                    />
                  )}
                  {ssm.disk?.hits != null && (
                    <StatCard
                      label={t('sessions.cache.ssmL2HitsMisses')}
                      value={`${ssm.disk.hits || 0} / ${ssm.disk.misses || 0}`}
                    />
                  )}
                  {ssm.disk?.latest_payload?.physical_tensor_dtype_counts && (
                    <StatCard
                      label={t('sessions.cache.companionDtypes')}
                      value={formatDtypeCounts(ssm.disk.latest_payload.physical_tensor_dtype_counts)}
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
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.cache.scheduler')}</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <StatCard label={t('sessions.cache.requests')} value={String(schedulerStats.num_requests_processed || 0)} />
            <StatCard label={t('status.running')} value={String(schedulerStats.num_running || 0)} />
            <StatCard label={t('sessions.cache.waiting')} value={String(schedulerStats.num_waiting || 0)} />
            {schedulerStats.ewma_ttft_seconds != null && (
              <StatCard label={t('sessions.cache.ttftEwma')} value={`${Number(schedulerStats.ewma_ttft_seconds || 0).toFixed(3)} s`} />
            )}
            <StatCard label={t('sessions.cache.promptTokens')} value={(schedulerStats.total_prompt_tokens || 0).toLocaleString()} />
            <StatCard label={t('sessions.cache.completionTokens')} value={(schedulerStats.total_completion_tokens || 0).toLocaleString()} />
            {schedulerStats.cache_hit_tokens != null && (
              <StatCard label={t('sessions.cache.cacheHitTokens')} value={(schedulerStats.cache_hit_tokens || 0).toLocaleString()} />
            )}
            {schedulerStats.cache_hit_requests != null && (
              <StatCard label={t('sessions.cache.cacheHitRequests')} value={(schedulerStats.cache_hit_requests || 0).toLocaleString()} />
            )}
            {schedulerStats.hybrid_kv_without_ssm_hits != null && (
              <StatCard label={t('sessions.cache.hybridKvOnlyMisses')} value={(schedulerStats.hybrid_kv_without_ssm_hits || 0).toLocaleString()} />
            )}
            {schedulerStats.hybrid_kv_without_ssm_tokens != null && schedulerStats.hybrid_kv_without_ssm_tokens > 0 && (
              <StatCard label={t('sessions.cache.kvOnlyTokens')} value={(schedulerStats.hybrid_kv_without_ssm_tokens || 0).toLocaleString()} />
            )}
            {schedulerStats.cache_reuse_skips != null && (
              <StatCard label={t('sessions.cache.cacheReuseSkips')} value={String(schedulerStats.cache_reuse_skips || 0)} />
            )}
            {schedulerStats.cache_reuse_skip_tokens != null && schedulerStats.cache_reuse_skip_tokens > 0 && (
              <StatCard label={t('sessions.cache.skippedHitTokens')} value={(schedulerStats.cache_reuse_skip_tokens || 0).toLocaleString()} />
            )}
            {schedulerStats.cache_reuse_partial_downgrades != null && (
              <StatCard label={t('sessions.cache.partialReuse')} value={String(schedulerStats.cache_reuse_partial_downgrades || 0)} />
            )}
            {schedulerStats.cache_reuse_partial_tokens != null && schedulerStats.cache_reuse_partial_tokens > 0 && (
              <StatCard label={t('sessions.cache.partialHitTokens')} value={(schedulerStats.cache_reuse_partial_tokens || 0).toLocaleString()} />
            )}
          </div>
          {schedulerStats.last_cache_reuse_partial && (
            <div className="mt-2 text-xs bg-accent/10 border border-accent/30 text-foreground px-3 py-2 rounded">
              {t('sessions.cache.reusePartialMessage', {
                used: (schedulerStats.last_cache_reuse_partial.used_cached_tokens ?? 0).toLocaleString(),
                original: (schedulerStats.last_cache_reuse_partial.original_cached_tokens ?? 0).toLocaleString(),
                neededMb: schedulerStats.last_cache_reuse_partial.used_needed_mb ?? '?',
                budgetMb: schedulerStats.last_cache_reuse_partial.budget_mb ?? schedulerStats.last_cache_reuse_partial.available_mb ?? '?',
                tailTokens: (schedulerStats.last_cache_reuse_partial.tail_tokens ?? 0).toLocaleString(),
              })}
              {schedulerStats.last_cache_reuse_partial.cache_format && (
                <> {t('sessions.cache.formatSentence', { format: schedulerStats.last_cache_reuse_partial.cache_format })}</>
              )}
            </div>
          )}
          {schedulerStats.cache_hit_tokens_by_detail && Object.keys(schedulerStats.cache_hit_tokens_by_detail).length > 0 && (
            <div className="mt-2">
              <div className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-1">{t('sessions.cachePanel.hitTokensByDetail')}</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                {Object.entries(schedulerStats.cache_hit_tokens_by_detail).map(([detail, tokens]: [string, any]) => (
                  <StatCard key={detail} label={detail} value={(Number(tokens) || 0).toLocaleString()} />
                ))}
              </div>
            </div>
          )}
          {schedulerStats.last_hybrid_kv_without_ssm && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning-foreground px-3 py-2 rounded">
              {t('sessions.cache.hybridFullPrefillMessage', {
                cachedTokens: (schedulerStats.last_hybrid_kv_without_ssm.cached_tokens ?? 0).toLocaleString(),
                reason: schedulerStats.last_hybrid_kv_without_ssm.reason || 'missing_ssm',
              })}
              {schedulerStats.last_hybrid_kv_without_ssm.checkpoint_tokens != null && (
                <> {t('sessions.cache.checkpointSentence', { tokens: (schedulerStats.last_hybrid_kv_without_ssm.checkpoint_tokens ?? 0).toLocaleString() })}</>
              )}
            </div>
          )}
          {schedulerStats.last_cache_reuse_skip && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning-foreground px-3 py-2 rounded">
              {t('sessions.cache.reuseSkipMessage', {
                neededMb: schedulerStats.last_cache_reuse_skip.needed_mb ?? '?',
                budgetMb: schedulerStats.last_cache_reuse_skip.budget_mb ?? schedulerStats.last_cache_reuse_skip.available_mb ?? '?',
                availableMb: schedulerStats.last_cache_reuse_skip.available_mb ?? '?',
                droppedTokens: (schedulerStats.last_cache_reuse_skip.dropped_cached_tokens ?? schedulerStats.last_cache_reuse_skip.cached_tokens ?? 0).toLocaleString(),
                prefillTokens: (schedulerStats.last_cache_reuse_skip.full_prefill_tokens ?? schedulerStats.last_cache_reuse_skip.prompt_tokens ?? 0).toLocaleString(),
              })}
              {schedulerStats.last_cache_reuse_skip.cache_contract && (
                <> {t('sessions.cache.contractSentence', { contract: schedulerStats.last_cache_reuse_skip.cache_contract })}</>
              )}
              {schedulerStats.last_cache_reuse_skip.cache_format && (
                <> {t('sessions.cache.formatSentence', { format: schedulerStats.last_cache_reuse_skip.cache_format })}</>
              )}
              {schedulerStats.last_cache_reuse_skip.partial_reuse_unavailable_reason && (
                <> {t('sessions.cache.partialReasonSentence', { reason: schedulerStats.last_cache_reuse_skip.partial_reuse_unavailable_reason })}</>
              )}
            </div>
          )}
        </div>
      )}

      {lastCacheSelection && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            {t('sessions.cache.selection')}
          </h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {lastCacheSelection?.selected && (
              <StatCard
                label={t('sessions.cache.selection')}
                value={`${String(lastCacheSelection.selected)}${lastCacheSelection?.rejected ? ` ← ${String(lastCacheSelection.rejected)}` : ''}`}
              />
            )}
            {lastCacheSelection?.reason && (
              <StatCard
                label={t('sessions.cache.selectionReason')}
                value={String(lastCacheSelection.reason)}
              />
            )}
            {lastCacheSelection?.paged_cached_tokens != null && (
              <StatCard
                label={t('sessions.cache.ssdCandidate')}
                value={Number(lastCacheSelection.paged_cached_tokens || 0).toLocaleString()}
              />
            )}
            {lastCacheSelection?.cost_history_comparable != null && (
              <StatCard
                label={t('sessions.cache.costComparable')}
                value={lastCacheSelection.cost_history_comparable ? t('common.yes') : t('common.no')}
              />
            )}
            {lastCacheSelection?.cost_history_comparable === true
              && lastCacheSelection?.estimated_disk_seconds != null && (
              <StatCard
                label={t('sessions.cache.estimatedSsd')}
                value={`${(Number(lastCacheSelection.estimated_disk_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
            {lastCacheSelection?.cost_history_comparable === true
              && lastCacheSelection?.estimated_prefill_seconds != null && (
              <StatCard
                label={t('sessions.cache.estimatedPrefill')}
                value={`${(Number(lastCacheSelection.estimated_prefill_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
          </div>
        </div>
      )}

      {lastDurability && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            {t('sessions.cachePanel.lastDurability')}
          </h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <StatCard label={t('sessions.cache.requestId')} value={String(lastDurability.request_id || '—').slice(-12)} />
            <StatCard label={t('sessions.cache.durabilityWait')} value={typeof lastDurability.wait_ms === 'number' ? `${lastDurability.wait_ms.toFixed(1)} ms${lastDurability.waited ? '' : ` (${t('sessions.cache.durabilityAlreadyDurable')})`}` : '—'} />
            {lastDurability.cache_outcome && (
              <StatCard label={t('sessions.cache.durabilityOutcome')} value={String(lastDurability.cache_outcome)} />
            )}
            {lastDurability.retained_tokens != null && (
              <StatCard label={t('sessions.cache.durabilityRetained')} value={Number(lastDurability.retained_tokens).toLocaleString()} />
            )}
            {lastDurability.detail && (
              <StatCard label={t('sessions.cache.durabilityDetail')} value={String(lastDurability.detail)} />
            )}
          </div>
        </div>
      )}

      {lastCacheExecution && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            {t('sessions.cachePanel.lastCacheExecution')}
          </h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {lastCacheExecution.request_id && (
              <StatCard label={t('sessions.cache.requestId')} value={String(lastCacheExecution.request_id).slice(-12)} />
            )}
            {lastCacheExecution.cache_detail && (
              <StatCard label={t('sessions.cache.cacheDetail')} value={String(lastCacheExecution.cache_detail)} />
            )}
            {lastCacheExecution.cache_reuse_applied != null && (
              <StatCard
                label={t('sessions.cache.reuseApplied')}
                value={lastCacheExecution.cache_reuse_applied ? t('common.yes') : t('common.no')}
              />
            )}
            {lastCacheExecution.cache_outcome && (
              <StatCard label={t('sessions.cache.outcome')} value={String(lastCacheExecution.cache_outcome)} />
            )}
            {lastCacheExecution.prompt_tokens != null && (
              <StatCard
                label={t('sessions.cache.promptTokens')}
                value={Number(lastCacheExecution.prompt_tokens || 0).toLocaleString()}
              />
            )}
            {lastCacheExecution.cached_tokens != null && (
              <StatCard
                label={t('sessions.cache.cachedPrefix')}
                value={Number(lastCacheExecution.cached_tokens || 0).toLocaleString()}
              />
            )}
            {lastCacheExecution.attempted_cached_tokens != null && (
              <StatCard
                label={t('sessions.cache.attemptedPrefix')}
                value={Number(lastCacheExecution.attempted_cached_tokens || 0).toLocaleString()}
              />
            )}
            {lastCacheExecution.uncached_prompt_tokens != null && (
              <StatCard
                label={t('sessions.cache.uncachedTail')}
                value={Number(lastCacheExecution.uncached_prompt_tokens || 0).toLocaleString()}
              />
            )}
            {lastCacheExecution.prefill_tokens != null && (
              <StatCard
                label={t('sessions.cache.forwardedPrefill')}
                value={Number(lastCacheExecution.prefill_tokens || 0).toLocaleString()}
              />
            )}
            {lastCacheExecution.generation_prompt_suffix_tokens != null && (
              <StatCard
                label={t('sessions.cache.generationSuffix')}
                value={Number(lastCacheExecution.generation_prompt_suffix_tokens || 0).toLocaleString()}
              />
            )}
            {lastCacheExecution.media_cache_scope?.mode && (
              <StatCard
                label={t('sessions.cache.mediaScope')}
                value={String(lastCacheExecution.media_cache_scope.mode)}
              />
            )}
            {Array.isArray(lastCacheExecution.media_cache_scope?.boundaries) && (
              <StatCard
                label={t('sessions.cache.mediaBoundaries')}
                value={lastCacheExecution.media_cache_scope.boundaries.length
                  ? lastCacheExecution.media_cache_scope.boundaries.join(', ')
                  : '0'}
              />
            )}
            {lastCacheExecution.blocks != null && (
              <StatCard label={t('sessions.cache.matchedBlocks')} value={Number(lastCacheExecution.blocks || 0).toLocaleString()} />
            )}
            {lastCacheExecution.disk_blocks != null && (
              <StatCard label={t('sessions.cache.ssdBlocksRead')} value={Number(lastCacheExecution.disk_blocks || 0).toLocaleString()} />
            )}
            {lastCacheExecution.candidate_lookup_seconds != null && (
              <StatCard
                label={t('sessions.cache.candidateLookup')}
                value={`${(Number(lastCacheExecution.candidate_lookup_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
            {lastCacheExecution.ssd_retrieval_seconds != null && (
              <StatCard
                label={t('sessions.cache.ssdRetrievalTotal')}
                value={`${(Number(lastCacheExecution.ssd_retrieval_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
            {lastCacheExecution.admission_first_token_seconds != null && (
              <StatCard
                label={t('sessions.cache.admissionFirstToken')}
                value={`${(Number(lastCacheExecution.admission_first_token_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
            {lastCacheExecution.reconstruction_seconds != null && (
              <StatCard
                label={t('sessions.cache.reconstruction')}
                value={`${(Number(lastCacheExecution.reconstruction_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
            {lastCacheExecution.dequantization_seconds != null && (
              <StatCard
                label={t('sessions.cache.dequantization')}
                value={`${(Number(lastCacheExecution.dequantization_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
            {lastCacheExecution.tq_rewrap_seconds != null && (
              <StatCard
                label={t('sessions.cache.tqRewrap')}
                value={`${(Number(lastCacheExecution.tq_rewrap_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
            {lastCacheExecution.total_worker_cache_seconds != null && (
              <StatCard
                label={t('sessions.cache.workerCacheTime')}
                value={`${(Number(lastCacheExecution.total_worker_cache_seconds || 0) * 1000).toFixed(2)} ms`}
              />
            )}
          </div>
          {lastCacheExecution.fallback_reason && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning-foreground px-3 py-2 rounded">
              {t('sessions.cache.fallbackMessage', { reason: String(lastCacheExecution.fallback_reason) })}
            </div>
          )}
        </div>
      )}

      {/* KV Quantization Info */}
      {(kvQuant || turboQuantKv || nativeCache) && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.cachePanel.cacheContract')}</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {nativeCache?.cache_type && (
              <StatCard label={t('sessions.cache.nativeCache')} value={nativeCache.cache_type} />
            )}
            {nativeCache?.schema && (
              <StatCard label={t('sessions.cache.schema')} value={nativeCache.schema} />
            )}
            {turboQuantKv && (
              <StatCard
                label={t('sessions.cache.turboquantKv')}
                value={
                  turboQuantKv.enabled
                    ? turboQuantKv.single_sequence_only
                      ? t('sessions.cache.statusEnabledSingleSeq')
                      : t('sessions.cache.statusEnabled')
                    : t('sessions.cache.statusDisabled')
                }
              />
            )}
            {turboQuantKv?.single_sequence_only && (
              <StatCard
                label={t('sessions.cache.tqBatch')}
                value={t('sessions.cache.tqBatchValue', {
                  seqs: turboQuantKv.effective_max_num_seqs ?? 1,
                  prefill: turboQuantKv.effective_prefill_batch_size ?? 1,
                  decode: turboQuantKv.effective_completion_batch_size ?? 1,
                })}
              />
            )}
            {nativeCache?.generic_turboquant_kv && (
              <StatCard
                label={
                  nativeCache.generic_turboquant_kv.reason === 'hybrid_attention_kv_only'
                    ? t('sessions.cache.selectiveTqKv')
                    : t('sessions.cache.genericTqKv')
                }
                value={
                  nativeCache.generic_turboquant_kv.enabled
                    ? t('sessions.cache.statusEnabled')
                    : t('sessions.cache.offWithReason', { reason: nativeCache.generic_turboquant_kv.reason || 'native' })
                }
              />
            )}
            {dsv4ActivationQatDisplay && (
              <>
                <StatCard label={t('sessions.cache.dsv4QatRequested')} value={dsv4ActivationQatDisplay.requestedEffective} />
                <StatCard label={t('sessions.cache.dsv4QatObserved')} value={dsv4ActivationQatDisplay.observedAttestation} />
                <StatCard label={t('sessions.cache.dsv4QatPaths')} value={dsv4ActivationQatDisplay.paths} />
                <StatCard label={t('sessions.cache.dsv4QatKernels')} value={dsv4ActivationQatDisplay.fusedKernels} />
              </>
            )}
            {attentionKvStorage && (
              <StatCard
                label={t('sessions.cache.attentionKvL2')}
                value={
                  tqStoredPrefix ?? (attentionKvStorage.enabled
                    ? t('sessions.cache.attentionKvL2Value', { bits: attentionKvStorage.bits, groupSize: attentionKvStorage.group_size ?? 64 })
                    : t('sessions.cache.attentionKvFullPrecision'))
                }
              />
            )}
            {attentionKvStorage?.ssm_policy && (
              <StatCard
                label={t('sessions.cache.ssmPolicy')}
                value={`${attentionKvStorage.ssm_policy}${attentionKvStorage.rederive ? ' + rederive' : ''}`}
              />
            )}
            {kvQuant && !tqStoredPrefix && (
              <StatCard
                label={t('sessions.cache.storedKvQuant')}
                value={
                  kvQuant?.enabled
                    ? t('sessions.cache.storedKvQuantValue', { bits: kvQuant.bits, groupSize: kvQuant.group_size })
                    : t('sessions.cache.statusDisabled')
                }
              />
            )}
            {displayComponents.length > 0 && (
              <StatCard
                label={t('sessions.cache.components')}
                value={displayComponents.join(', ')}
              />
            )}
            {runtimeCacheObjects && (
              <StatCard
                label={t('sessions.cache.runtimeCacheObjects')}
                value={runtimeCacheObjects}
              />
            )}
            {runtimeCacheOwners && (
              <StatCard
                label={t('sessions.cache.runtimeCacheOwners')}
                value={runtimeCacheOwners}
              />
            )}
            {dtypeHarmonization?.enabled && (
              <StatCard
                label={t('sessions.cache.dtypeHarmonization')}
                value={`${dtypeHarmonization.cast ?? 0} F16→BF16; ${dtypeHarmonization.preserved_f16 ?? 0} F16 preserved`}
              />
            )}
            {storedAttentionDtypes && (
              <StatCard
                label={t('sessions.cache.storedKvDtypes')}
                value={storedAttentionDtypes}
              />
            )}
            {physicalBlockDtypes && (
              <StatCard
                label={t('sessions.cache.physicalKvDtypes')}
                value={physicalBlockDtypes}
              />
            )}
            {Array.isArray(nativeCache?.kv_layer_indices) && nativeCache.cache_layer_count > 0 && (
              <StatCard
                label={t('sessions.cache.attentionLayers')}
                value={`${nativeCache.kv_layer_indices.length} / ${nativeCache.cache_layer_count}`}
              />
            )}
            {nativeCache?.companion_layer_count != null && nativeCache.cache_layer_count > 0 && (
              <StatCard
                label={t('sessions.cache.companionLayers')}
                value={`${nativeCache.companion_layer_count} / ${nativeCache.cache_layer_count}`}
              />
            )}
            {Array.isArray(nativeCache?.kv_layer_indices) && nativeCache.kv_layer_indices.length > 0 && (
              <StatCard
                label={t('sessions.cache.attentionLayerIds')}
                value={nativeCache.kv_layer_indices.join(', ')}
              />
            )}
            {Array.isArray(nativeCache?.full_attention_layer_indices) && nativeCache.full_attention_layer_indices.length > 0 && (
              <StatCard
                label={t('sessions.cache.fullAttentionLayerIds')}
                value={nativeCache.full_attention_layer_indices.join(', ')}
              />
            )}
            {Array.isArray(nativeCache?.sliding_attention_layer_indices) && nativeCache.sliding_attention_layer_indices.length > 0 && (
              <StatCard
                label={t('sessions.cache.slidingAttentionLayerIds')}
                value={nativeCache.sliding_attention_layer_indices.join(', ')}
              />
            )}
            {Array.isArray(nativeCache?.runtime_cache_unknown_layer_indices) && nativeCache.runtime_cache_unknown_layer_indices.length > 0 && (
              <StatCard
                label={t('sessions.cache.unclassifiedLayerIds')}
                value={nativeCache.runtime_cache_unknown_layer_indices.join(', ')}
              />
            )}
            {nativeCache?.kv_layer_indices_source && (
              <StatCard
                label={t('sessions.cache.layoutEvidence')}
                value={['instantiated_make_cache', 'instantiated_runtime_cache_factory'].includes(nativeCache.kv_layer_indices_source)
                  ? t('sessions.cache.instantiatedRuntimeCache')
                  : nativeCache.kv_layer_indices_source}
              />
            )}
          </div>
        </div>
      )}

      {/* Disk Cache (L2 prompt-level) */}
      {diskCache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.cachePanel.diskCacheL2')}</h4>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {diskCache.entries != null && <StatCard label={t('sessions.cache.entries')} value={String(diskCache.entries)} />}
            {(diskCache.total_size_mb ?? diskCache.size_mb) != null && <StatCard label={t('image.prompt.size')} value={`${(diskCache.total_size_mb ?? diskCache.size_mb ?? 0).toFixed(1)} MB`} />}
            {diskCache.total_tokens_on_disk != null && <StatCard label={t('sessions.cache.tokensOnDisk')} value={(diskCache.total_tokens_on_disk || 0).toLocaleString()} />}
            {diskCache.hit_rate != null && <StatCard label={t('sessions.cache.hitRate')} value={`${(diskCache.hit_rate * 100).toFixed(1)}%`} />}
            {diskCache.hits != null && <StatCard label={t('sessions.cache.hitsMisses')} value={`${diskCache.hits} / ${diskCache.misses ?? 0}`} />}
            {diskCache.stores != null && <StatCard label={t('sessions.cache.stores')} value={String(diskCache.stores)} />}
            {diskCache.tq_native_stores != null && diskCache.tq_native_stores > 0 && <StatCard label={t('sessions.cache.tqNativeStores')} value={String(diskCache.tq_native_stores)} />}
            {diskCache.tq_native_hits != null && diskCache.tq_native_hits > 0 && <StatCard label={t('sessions.cache.tqNativeHits')} value={String(diskCache.tq_native_hits)} />}
            {diskCache.pending_writes != null && diskCache.pending_writes > 0 && <StatCard label={t('sessions.cache.pendingWrites')} value={String(diskCache.pending_writes)} />}
          </div>
        </div>
      )}

      {/* Block Disk Cache (SSD / L2 content-addressed blocks) */}
      {blockDiskCache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.cachePanel.blockDiskCache')}</h4>
          <p className="mb-2 text-xs text-muted-foreground">
            {t('sessions.cache.blockDiskExplainer')}
          </p>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <StatCard label={t('sessions.cache.namespaceBlocks')} value={String(blockDiskCache.blocks_on_disk ?? 0)} />
            <StatCard
              label={t('sessions.cache.namespaceSize')}
              value={blockDiskCache.disk_size_bytes != null
                ? formatCacheStorageBytes(blockDiskCache.disk_size_bytes)
                : `${(blockDiskCache.disk_size_gb ?? 0).toFixed(2)} GB`}
            />
            {globalBlockDiskBudget && (
              <StatCard
                label={t('sessions.cache.managedRootSize')}
                value={globalBlockDiskBudget.accounted === true && globalBlockDiskBudget.bytes_after != null
                  ? `${(globalBlockDiskBudget.bytes_after / 1024 ** 3).toFixed(2)} GB`
                  : t('sessions.cache.reconciliationPending')}
              />
            )}
            {globalBlockDiskBudget?.max_size_bytes != null && (
              <StatCard
                label={t('sessions.cache.managedRootLimit')}
                value={globalBlockDiskBudget.max_size_bytes > 0
                  ? `${(globalBlockDiskBudget.max_size_bytes / 1024 ** 3).toFixed(2)} GB`
                  : t('sessions.cache.unlimited')}
              />
            )}
            {globalBlockDiskBudget && (
              <StatCard
                label={t('sessions.cache.managedRootStatus')}
                value={globalBlockDiskBudget.accounted !== true
                  ? t('sessions.cache.reconciliationPending')
                  : globalBlockDiskBudget.compliant
                    ? t('sessions.cache.withinLimit')
                    : t('sessions.cache.overLimit')}
              />
            )}
            {blockDiskCache.total_tokens_on_disk != null && <StatCard label={t('sessions.cache.namespaceTokens')} value={(blockDiskCache.total_tokens_on_disk || 0).toLocaleString()} />}
            <StatCard label={t('sessions.cache.persistedBlockReads')} value={String(blockDiskCache.total_accesses ?? 0)} />
            <StatCard label={t('sessions.cache.thisEngineReads')} value={`${blockDiskCache.disk_hits ?? 0} / ${blockDiskCache.disk_misses ?? 0}`} />
            <StatCard label={t('sessions.cache.thisEngineWrites')} value={String(blockDiskCache.disk_writes ?? 0)} />
            <StatCard label={t('sessions.cache.thisEngineEvictions')} value={String(blockDiskCache.disk_evictions ?? 0)} />
            {blockDiskCache.write_pipeline && (
              <StatCard
                label={t('sessions.cache.writerPendingInFlight')}
                value={`${blockDiskCache.write_pipeline.pending_items ?? 0} / ${blockDiskCache.write_pipeline.inflight ?? 0}`}
              />
            )}
            {blockDiskCache.write_pipeline && (
              <StatCard
                label={t('sessions.cache.offThreadWrites')}
                value={`${blockDiskCache.write_pipeline.offthread_serializations_queued ?? 0} / ${blockDiskCache.write_pipeline.offthread_serializations_completed ?? 0} / ${blockDiskCache.write_pipeline.offthread_serialization_failures ?? 0}`}
              />
            )}
            {globalBlockDiskBudget && (
              <StatCard
                label={t('sessions.cache.lastReconciliationTrim')}
                value={t('sessions.cache.entriesGbValue', {
                  entries: globalBlockDiskBudget.evicted_entries ?? 0,
                  gb: ((globalBlockDiskBudget.evicted_bytes ?? 0) / 1024 ** 3).toFixed(2),
                })}
              />
            )}
          </div>
        </div>
      )}

      {!schedulerCache && !schedulerStats && !stats?.error && (
        <div className="text-sm text-muted-foreground">{t('sessions.cachePanel.loading')}</div>
      )}

      {stats?.error && (
        <div className="text-sm text-muted-foreground">{stats.error}</div>
      )}

      {/* Cache Entries */}
      {showEntries && entries && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">
            {t('sessions.cache.cacheEntriesHeader', { count: entries.count || 0, type: entries.cache_type ?? '' })}
          </h4>
          <div className="max-h-48 overflow-auto space-y-1">
            {entries.entries?.map((entry: any, i: number) => (
              <div key={i} className="text-xs bg-background px-2 py-1 rounded border border-border flex justify-between">
                <span>{t('chat.bubble.tokensSuffix', { n: entry.tokens_count })}</span>
                {entry.memory_mb && <span className="text-muted-foreground">{entry.memory_mb} MB</span>}
                {entry.ref_count != null && <span className="text-muted-foreground">{t('sessions.cache.refsCount', { count: entry.ref_count })}</span>}
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
            onChange={e => {
              warmInputGenerationRef.current += 1
              setWarmInput(e.target.value)
            }}
            onKeyDown={e => { if (e.key === 'Enter' && warmInput.trim()) handleWarm() }}
            disabled={actionBusy}
            placeholder={t('sessions.cache.warmPlaceholder')}
            autoFocus
            className="flex-1 px-2 py-1.5 text-xs bg-background border border-input rounded focus:outline-none focus:ring-1 focus:ring-ring"
          />
          <button
            onClick={() => {
              warmInputGenerationRef.current += 1
              setShowWarmInput(false)
              setWarmInput('')
            }}
            disabled={actionBusy}
            className="px-2 py-1.5 text-xs text-muted-foreground hover:text-foreground"
          >
            {t('common.cancel')}
          </button>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={handleFetchEntries}
          disabled={actionBusy}
          className="px-3 py-1.5 text-xs border border-border rounded hover:bg-accent disabled:opacity-50"
        >
          {loading ? t('common.loading') : showEntries ? t('sessions.cache.refreshEntries') : t('sessions.cache.showEntries')}
        </button>
        <button
          onClick={handleWarm}
          disabled={actionBusy}
          className="px-3 py-1.5 text-xs border border-border rounded hover:bg-accent disabled:opacity-50"
        >
          {warming ? t('sessions.cache.warming') : t('sessions.cache.warmCache')}
        </button>
        <button
          onClick={() => handleClear('ram')}
          disabled={actionBusy}
          title={t('sessions.cache.clearRamTitle')}
          className="px-3 py-1.5 text-xs border border-destructive/50 text-destructive rounded hover:bg-destructive/10 disabled:opacity-50"
        >
          {t('sessions.cachePanel.clearRam')}
        </button>
        <button
          onClick={() => handleClear('prefix')}
          disabled={actionBusy}
          title={t('sessions.cache.clearPrefixTitle')}
          className="px-3 py-1.5 text-xs border border-destructive/50 text-destructive rounded hover:bg-destructive/10 disabled:opacity-50"
        >
          {t('sessions.cachePanel.clearPrefixL2')}
        </button>
        <button
          onClick={() => handleClear('all')}
          disabled={actionBusy}
          className="px-3 py-1.5 text-xs border border-destructive/50 text-destructive rounded hover:bg-destructive/10 disabled:opacity-50"
        >
          {t('sessions.cachePanel.clearAll')}
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
