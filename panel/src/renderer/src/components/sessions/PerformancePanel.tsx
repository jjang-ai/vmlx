import { useState, useEffect, useRef } from 'react'
import { useTranslation } from '../../i18n'
import {
  describeDsv4ActivationQat,
  type Dsv4ActivationQatStatus,
} from './dsv4QatStatus'

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
    batch_generator?: {
      last_native_mtp?: {
        request_id?: string
        finish_reason?: string
        final_depth?: number
        cycles?: number
        accepted_tokens?: number
        drafted_tokens?: number
        acceptance_rate?: number | null
        depth_acceptance_rates?: Record<string, number | null>
        forwards?: {
          seed_main?: number
          verify_main?: number
          replay_main?: number
          mtp?: number
        }
        timings_ms?: {
          total?: number
          avg_cycle?: number
          verify?: number
          draft?: number
          replay?: number
        }
        fallback_reason?: string | null
        policy?: string | null
        configured_depth?: number | null
        at?: number | null
      } | null
      last_native_mtp_skip?: {
        request_id?: string
        reason?: string
      } | null
    }
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
      cache_format?: string
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
      cache_format?: string
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
      nbytes?: number
      nbytes_mb?: number
      max_bytes?: number
      max_bytes_mb?: number
      evictions?: number
      evicted_bytes?: number
      evicted_bytes_mb?: number
      disk?: {
        entries?: number
        total_tokens_on_disk?: number
        hits?: number
        misses?: number
      }
    }
    totals?: {
      ram_tokens_cached?: number
      l1_indexed_tokens?: number
      l1_resident_bytes?: number
      l1_resident_bytes_mb?: number
      l1_max_resident_bytes?: number
      l1_max_resident_bytes_mb?: number
      l1_evictions?: number
      l2_prompt_tokens_on_disk?: number
      l2_block_tokens_on_disk?: number
      l2_ssm_tokens_on_disk?: number
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
    storage_encode_enabled?: boolean
    stored_prefix_quantization?: string
    storage_key_bits?: number
    storage_value_bits?: number
    key_bits_values?: number[]
    value_bits_values?: number[]
  }
  native_cache?: {
    family?: string
    schema?: string
    cache_type?: string
    components?: string[]
    prefix?: boolean
    paged?: boolean
    block_disk_l2?: boolean
    ssm_entries?: number | null
    kv_layer_indices?: number[]
    generic_turboquant_kv?: {
      enabled?: boolean
      reason?: string
    }
    pool_quant?: {
      requested?: boolean
      enabled?: boolean
      observed?: boolean | null
      matches_request?: boolean
      env?: string
      error?: string | null
    }
    activation_qat?: Dsv4ActivationQatStatus
    attention_kv_storage_quantization?: NativeStorageQuantization
    storage_quantization?: NativeStorageQuantization
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
    group_size?: number
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
    jangtq_acceleration?: {
      mode?: 'auto' | 'off' | 'on'
      requested?: boolean
      available?: boolean
      active?: boolean
      reason?: string | null
    }
    metal_na_symbols?: {
      available?: boolean
      nax_symbols?: number
      naxtile_symbols?: number
    }
  }
  mtp?: {
    config_num_nextn_predict_layers?: number | null
    jang_drop_mtp?: boolean | null
    index_has_mtp_tensors?: boolean
    artifact_available?: boolean
    runtime_available?: boolean
    runtime_supported?: boolean
    runtime_active?: boolean
    effective_depth?: number | null
    effective_depth_source?: string | null
    runtime_reason?: string
    runtime_scope?: string
    vl_runtime_available?: boolean
    request_policy?: string
    request_gate?: string
    family?: string | null
    mtp_tensor_count?: number
    vision_tensor_count?: number
    status?: string
    issues?: string[]
  }
}

type NativeStorageQuantization = {
  enabled?: boolean
  mode?: string
  bits?: number | null
  group_size?: number | null
  applies_to?: string
  ssm_policy?: string
  rederive?: string
  metadata_policy?: string
}

export function PerformancePanel({ endpoint, sessionStatus }: PerformancePanelProps) {
  const { t } = useTranslation()
  const [health, setHealth] = useState<HealthData | null>(null)
  const [history, setHistory] = useState<Array<{ time: number; active: number; peak: number }>>([])
  const [error, setError] = useState<string | null>(null)
  const intervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastNativeMtp = health?.scheduler?.batch_generator?.last_native_mtp
  const attentionKvStorage =
    health?.native_cache?.attention_kv_storage_quantization ??
    health?.native_cache?.storage_quantization
  const dsv4ActivationQat = health?.native_cache?.activation_qat
  const dsv4ActivationQatDisplay = dsv4ActivationQat
    ? describeDsv4ActivationQat(dsv4ActivationQat)
    : null
  const tqKeyBits = health?.turboquant_kv_cache?.key_bits_values?.length
    ? health.turboquant_kv_cache.key_bits_values.map(bits => `q${bits}`).join('/')
    : `q${health?.turboquant_kv_cache?.storage_key_bits ?? '?'}`
  const tqValueBits = health?.turboquant_kv_cache?.value_bits_values?.length
    ? health.turboquant_kv_cache.value_bits_values.map(bits => `q${bits}`).join('/')
    : `q${health?.turboquant_kv_cache?.storage_value_bits ?? '?'}`
  const tqStoredPrefix = health?.turboquant_kv_cache?.storage_encode_enabled
    ? `${health.turboquant_kv_cache.stored_prefix_quantization ?? 'TurboQuant'} (K ${tqKeyBits} / V ${tqValueBits})`
    : null

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
        {t('sessions.performance.sessionMustBeRunning')}
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
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.perf.engine')}</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <InfoCard label={t('chat.settings.status')} value={health.status} />
            <InfoCard label={t('sessions.perf.engine')} value={health.engine_type || t('sessions.performance.unknown')} />
            <InfoCard label={t('sessions.performance.modelType')} value={health.model_type || '-'} />
            {(health.quantization_format || health.quantization) && (
              <InfoCard
                label={t('sessions.performance.weightQuant')}
                value={formatWeightQuant(health, t)}
              />
            )}
            {health.quantization?.codec && (
              <InfoCard
                label={t('sessions.performance.weightCodec')}
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
                label={t('sessions.performance.jangtqLayout')}
                value={
                  health.quantization.sidecar?.prestacked_bundle
                    ? t('sessions.performance.layoutPrestackedBundle')
                    : health.quantization.sidecar?.jangtq_runtime
                      ? t('sessions.performance.layoutRuntimeSidecar')
                      : health.quantization.sidecar?.jang_config
                        ? t('sessions.performance.layoutConfigOnly')
                        : t('sessions.performance.unknown')
                }
              />
            )}
            {health.quantization?.passthrough_tensor_count ? (
              <InfoCard
                label={t('sessions.performance.f16Passthrough')}
                value={t('sessions.performance.f16PassthroughValue', {
                  count: health.quantization.passthrough_tensor_count,
                  bits: (health.quantization.passthrough_bit_widths_used || []).join('/') || 16,
                })}
              />
            ) : null}
            {health.acceleration?.kernel_type && (
              <InfoCard
                label={t('sessions.performance.metalNa')}
                value={
                  health.acceleration.metal_na_active_on_host
                    ? t('sessions.performance.statusActive')
                    : health.acceleration.kernel_type === 'turboquant_codebook'
                      ? t('sessions.performance.notUsedByJangtq')
                      : health.acceleration.metal_na_capable
                        ? t('sessions.performance.statusUnavailable')
                        : t('sessions.performance.notApplicable')
                }
              />
            )}
            {health.mtp && health.mtp.status && health.mtp.status !== 'not_configured' && (
              <InfoCard
                label={t('sessions.performance.mtp')}
                value={
                  health.mtp.runtime_active
                    ? `${t('sessions.performance.statusActive')}${health.mtp.effective_depth ? ` D${health.mtp.effective_depth}` : ''}${health.mtp.runtime_scope ? ` (${health.mtp.runtime_scope})` : ''}`
                    : health.mtp.runtime_available
                      ? t('sessions.performance.weightsPresentRuntimeReady')
                    : health.mtp.artifact_available
                      ? t('sessions.performance.weightsPresentRuntimeUnwired')
                      : health.mtp.status.replace(/_/g, ' ')
                }
              />
            )}
            {health.mtp?.effective_depth && health.mtp.runtime_available && (
              <InfoCard
                label={t('sessions.performance.mtpDepth')}
                value={health.mtp.effective_depth_source === 'default'
                  ? t('sessions.performance.mtpDepthDefaultValue', { depth: health.mtp.effective_depth })
                  : `D${health.mtp.effective_depth}`}
              />
            )}
            {health.mtp?.runtime_scope && health.mtp.runtime_available && (
              <InfoCard
                label={t('sessions.performance.mtpRuntimeScope')}
                value={health.mtp.vl_runtime_available
                  ? health.mtp.runtime_scope
                  : t('sessions.performance.scopeOnlyValue', { scope: health.mtp.runtime_scope })}
              />
            )}
            {health.mtp?.request_policy && health.mtp.runtime_available && (
              <InfoCard
                label={t('sessions.performance.mtpPolicy')}
                value={health.mtp.request_policy === 'greedy-only'
                  ? t('sessions.performance.greedyOnly')
                  : health.mtp.request_policy === 'deterministic-defaults'
                    ? t('sessions.performance.deterministicDefaults')
                    : health.mtp.request_policy}
              />
            )}
            {health.mtp?.request_gate && health.mtp.runtime_available && (
              <InfoCard
                label={t('sessions.performance.mtpGate')}
                value={health.mtp.request_gate.replace(',', ', ')}
              />
            )}
            {(health.mtp?.mtp_tensor_count != null || health.mtp?.vision_tensor_count != null) && (
              <InfoCard
                label={t('sessions.performance.mtpTensors')}
                value={t('sessions.performance.mtpTensorsValue', {
                  mtp: health.mtp?.mtp_tensor_count ?? 0,
                  vision: health.mtp?.vision_tensor_count ?? 0,
                })}
              />
            )}
            {lastNativeMtp && (
              <InfoCard
                label={t('sessions.performance.mtpScope')}
                value={formatMtpScope(lastNativeMtp)}
              />
            )}
            {lastNativeMtp && (
              <InfoCard
                label={t('sessions.performance.mtpAccept')}
                value={`${reportedCount(lastNativeMtp.accepted_tokens)}/${reportedCount(lastNativeMtp.drafted_tokens)} (${formatPercent(lastNativeMtp.acceptance_rate)})`}
              />
            )}
            {lastNativeMtp?.depth_acceptance_rates && (
              <InfoCard
                label={t('sessions.performance.mtpDepthRates')}
                value={formatMtpDepthRates(lastNativeMtp.depth_acceptance_rates, lastNativeMtp.final_depth)}
              />
            )}
            {lastNativeMtp?.forwards && (
              <InfoCard
                label={t('sessions.performance.mtpForwards')}
                value={`v${reportedCount(lastNativeMtp.forwards.verify_main)} / r${reportedCount(lastNativeMtp.forwards.replay_main)} / m${reportedCount(lastNativeMtp.forwards.mtp)}`}
              />
            )}
            {lastNativeMtp?.timings_ms && (
              <InfoCard
                label={t('sessions.performance.mtpTiming')}
                value={typeof lastNativeMtp.timings_ms.avg_cycle === 'number' ? `${lastNativeMtp.timings_ms.avg_cycle.toFixed(1)} ms/cyc` : t('sessions.performance.notReported')}
              />
            )}
            {health.kv_cache_quantization?.enabled && (
              <InfoCard label={t('sessions.performance.kvQuant')} value={`${health.kv_cache_quantization.bits}-bit`} />
            )}
            {health.native_cache?.cache_type && (
              <InfoCard label={t('sessions.cache.nativeCache')} value={health.native_cache.cache_type} />
            )}
            {health.native_cache && (health.native_cache.paged != null || health.native_cache.block_disk_l2 != null) && (
              <InfoCard
                label={t('sessions.performance.cacheStack')}
                value={`${health.native_cache.paged ? t('sessions.performance.cacheStackRamPaged') : health.native_cache.prefix ? t('sessions.performance.cacheStackPrefix') : t('sessions.performance.cacheStackNoPrefix')}${health.native_cache.block_disk_l2 ? ' + SSD L2' : ''}`}
              />
            )}
            {health.native_cache?.components?.length ? (
              <InfoCard
                label={t('sessions.performance.cacheComponents')}
                value={health.native_cache.components.join(', ')}
              />
            ) : null}
            {health.native_cache?.ssm_entries != null && (
              <InfoCard label={t('sessions.performance.ssmEntries')} value={String(health.native_cache.ssm_entries || 0)} />
            )}
            {health.turboquant_kv_cache && (
              <InfoCard
                label={t('sessions.performance.tqKv')}
                value={
                  health.turboquant_kv_cache.enabled
                    ? health.turboquant_kv_cache.single_sequence_only
                      ? t('sessions.cache.statusEnabledSingleSeq')
                      : t('sessions.cache.statusEnabled')
                    : t('sessions.cache.statusDisabled')
                }
              />
            )}
            {health.turboquant_kv_cache?.single_sequence_only && (
              <InfoCard
                label={t('sessions.cache.tqBatch')}
                value={t('sessions.cache.tqBatchValue', {
                  seqs: health.turboquant_kv_cache.effective_max_num_seqs ?? 1,
                  prefill: health.turboquant_kv_cache.effective_prefill_batch_size ?? 1,
                  decode: health.turboquant_kv_cache.effective_completion_batch_size ?? 1,
                })}
              />
            )}
            {health.native_cache?.generic_turboquant_kv && (
              <InfoCard
                label={
                  health.native_cache.generic_turboquant_kv.reason === 'hybrid_attention_kv_only'
                    ? t('sessions.cache.selectiveTqKv')
                    : t('sessions.cache.genericTqKv')
                }
                value={
                  health.native_cache.generic_turboquant_kv.enabled
                    ? t('sessions.cache.statusEnabled')
                    : t('sessions.performance.offWithReasonColon', { reason: health.native_cache.generic_turboquant_kv.reason || 'native' })
                }
              />
            )}
            {health.native_cache?.pool_quant && (
              <InfoCard
                label={t('sessions.performance.dsv4PoolQuant')}
                value={
                  health.native_cache.pool_quant.error
                    ? t('sessions.performance.poolQuantErrorValue', { error: health.native_cache.pool_quant.error })
                    : health.native_cache.pool_quant.matches_request === false
                      ? t('sessions.performance.poolQuantMismatchValue', {
                          requested: health.native_cache.pool_quant.requested ? t('sessions.performance.stateOn') : t('sessions.performance.stateOff'),
                          observed: health.native_cache.pool_quant.observed ? t('sessions.performance.stateOn') : t('sessions.performance.stateOff'),
                        })
                      : health.native_cache.pool_quant.enabled ? t('sessions.cache.statusEnabled') : t('sessions.cache.statusDisabled')
                }
              />
            )}
            {dsv4ActivationQatDisplay && (
              <>
                <InfoCard label={t('sessions.cache.dsv4QatRequested')} value={dsv4ActivationQatDisplay.requestedEffective} />
                <InfoCard label={t('sessions.cache.dsv4QatObserved')} value={dsv4ActivationQatDisplay.observedAttestation} />
                <InfoCard label={t('sessions.cache.dsv4QatPaths')} value={dsv4ActivationQatDisplay.paths} />
                <InfoCard label={t('sessions.cache.dsv4QatKernels')} value={dsv4ActivationQatDisplay.fusedKernels} />
              </>
            )}
            {attentionKvStorage && (
              <InfoCard
                label={t('sessions.cache.attentionKvL2')}
                value={
                  tqStoredPrefix ?? (attentionKvStorage.enabled
                    ? `q${attentionKvStorage.bits} / g${attentionKvStorage.group_size ?? 64}`
                    : t('sessions.cache.statusDisabled'))
                }
              />
            )}
            {attentionKvStorage?.ssm_policy && (
              <InfoCard
                label={t('sessions.cache.ssmPolicy')}
                value={`${attentionKvStorage.ssm_policy}${attentionKvStorage.rederive ? ' + rederive' : ''}`}
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
          {health.mtp?.issues?.length ? (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning px-3 py-2 rounded space-y-1">
              {health.mtp.issues.map((issue, index) => (
                <div key={index}>{issue}</div>
              ))}
            </div>
          ) : null}
          {lastNativeMtp?.fallback_reason ? (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning px-3 py-2 rounded">
              {t('sessions.performance.nativeMtpFallback', { reason: lastNativeMtp.fallback_reason })}
            </div>
          ) : null}
        </div>
      )}

      {/* Scheduler */}
      {health?.scheduler && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.cache.scheduler')}</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <InfoCard
              label={t('sessions.performance.queue')}
              value={t('sessions.performance.queueValue', {
                running: health.scheduler.num_running ?? 0,
                waiting: health.scheduler.num_waiting ?? 0,
              })}
            />
            {health.scheduler.ewma_ttft_seconds != null && (
              <InfoCard
                label={t('sessions.cache.ttftEwma')}
                value={`${Number(health.scheduler.ewma_ttft_seconds || 0).toFixed(3)} s`}
              />
            )}
            {health.scheduler.cache_hit_tokens != null && (
              <InfoCard
                label={t('sessions.cache.cacheHitTokens')}
                value={(health.scheduler.cache_hit_tokens || 0).toLocaleString()}
              />
            )}
            {health.scheduler.cache_hit_requests != null && (
              <InfoCard
                label={t('sessions.cache.cacheHitRequests')}
                value={(health.scheduler.cache_hit_requests || 0).toLocaleString()}
              />
            )}
            {health.scheduler.hybrid_kv_without_ssm_hits != null && (
              <InfoCard
                label={t('sessions.cache.hybridKvOnlyMisses')}
                value={(health.scheduler.hybrid_kv_without_ssm_hits || 0).toLocaleString()}
              />
            )}
            {health.scheduler.hybrid_kv_without_ssm_tokens != null && health.scheduler.hybrid_kv_without_ssm_tokens > 0 && (
              <InfoCard
                label={t('sessions.cache.kvOnlyTokens')}
                value={(health.scheduler.hybrid_kv_without_ssm_tokens || 0).toLocaleString()}
              />
            )}
            {health.scheduler.cache_reuse_skips != null && (
              <InfoCard label={t('sessions.performance.cacheSkips')} value={String(health.scheduler.cache_reuse_skips || 0)} />
            )}
            {health.scheduler.cache_reuse_skip_tokens != null && health.scheduler.cache_reuse_skip_tokens > 0 && (
              <InfoCard
                label={t('sessions.performance.skippedTokens')}
                value={(health.scheduler.cache_reuse_skip_tokens || 0).toLocaleString()}
              />
            )}
            {health.scheduler.cache_reuse_partial_downgrades != null && (
              <InfoCard
                label={t('sessions.cache.partialReuse')}
                value={String(health.scheduler.cache_reuse_partial_downgrades || 0)}
              />
            )}
            {health.scheduler.cache_reuse_partial_tokens != null && health.scheduler.cache_reuse_partial_tokens > 0 && (
              <InfoCard
                label={t('sessions.cache.partialHitTokens')}
                value={(health.scheduler.cache_reuse_partial_tokens || 0).toLocaleString()}
              />
            )}
            {health.scheduler.batch_generator?.last_native_mtp && (
              <InfoCard
                label={t('sessions.performance.mtpLast')}
                value={`D${health.scheduler.batch_generator.last_native_mtp.final_depth ?? '?'}${
                  health.scheduler.batch_generator.last_native_mtp.acceptance_rate != null
                    ? ` ${t('sessions.performance.percentAcceptValue', { percent: Math.round((health.scheduler.batch_generator.last_native_mtp.acceptance_rate || 0) * 100) })}`
                    : ''
                }`}
              />
            )}
            {health.scheduler.batch_generator?.last_native_mtp_skip && (
              <InfoCard
                label={t('sessions.performance.mtpSkip')}
                value={health.scheduler.batch_generator.last_native_mtp_skip.reason || t('sessions.performance.skipped')}
              />
            )}
          </div>
          {health.scheduler.last_cache_reuse_partial && (
            <div className="mt-2 text-xs bg-accent/10 border border-accent/30 text-foreground px-3 py-2 rounded">
              {t('sessions.cache.reusePartialMessage', {
                used: (health.scheduler.last_cache_reuse_partial.used_cached_tokens ?? 0).toLocaleString(),
                original: (health.scheduler.last_cache_reuse_partial.original_cached_tokens ?? 0).toLocaleString(),
                neededMb: health.scheduler.last_cache_reuse_partial.used_needed_mb ?? '?',
                budgetMb: health.scheduler.last_cache_reuse_partial.budget_mb ?? health.scheduler.last_cache_reuse_partial.available_mb ?? '?',
                tailTokens: (health.scheduler.last_cache_reuse_partial.tail_tokens ?? 0).toLocaleString(),
              })}
              {health.scheduler.last_cache_reuse_partial.cache_format && (
                <> {t('sessions.cache.formatSentence', { format: health.scheduler.last_cache_reuse_partial.cache_format })}</>
              )}
            </div>
          )}
          {health.scheduler.last_hybrid_kv_without_ssm && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning px-3 py-2 rounded">
              {t('sessions.cache.hybridFullPrefillMessage', {
                cachedTokens: (health.scheduler.last_hybrid_kv_without_ssm.cached_tokens ?? 0).toLocaleString(),
                reason: health.scheduler.last_hybrid_kv_without_ssm.reason || 'missing_ssm',
              })}
              {health.scheduler.last_hybrid_kv_without_ssm.checkpoint_tokens != null && (
                <> {t('sessions.cache.checkpointSentence', { tokens: (health.scheduler.last_hybrid_kv_without_ssm.checkpoint_tokens ?? 0).toLocaleString() })}</>
              )}
            </div>
          )}
          {health.scheduler.last_cache_reuse_skip && (
            <div className="mt-2 text-xs bg-warning/10 border border-warning/30 text-warning px-3 py-2 rounded">
              {t('sessions.cache.reuseSkipMessage', {
                neededMb: health.scheduler.last_cache_reuse_skip.needed_mb ?? '?',
                budgetMb: health.scheduler.last_cache_reuse_skip.budget_mb ?? health.scheduler.last_cache_reuse_skip.available_mb ?? '?',
                availableMb: health.scheduler.last_cache_reuse_skip.available_mb ?? '?',
                droppedTokens: (health.scheduler.last_cache_reuse_skip.dropped_cached_tokens ?? health.scheduler.last_cache_reuse_skip.cached_tokens ?? 0).toLocaleString(),
                prefillTokens: (health.scheduler.last_cache_reuse_skip.full_prefill_tokens ?? health.scheduler.last_cache_reuse_skip.prompt_tokens ?? 0).toLocaleString(),
              })}
              {health.scheduler.last_cache_reuse_skip.cache_contract && (
                <> {t('sessions.cache.contractSentence', { contract: health.scheduler.last_cache_reuse_skip.cache_contract })}</>
              )}
              {health.scheduler.last_cache_reuse_skip.cache_format && (
                <> {t('sessions.cache.formatSentence', { format: health.scheduler.last_cache_reuse_skip.cache_format })}</>
              )}
              {health.scheduler.last_cache_reuse_skip.partial_reuse_unavailable_reason && (
                <> {t('sessions.cache.partialReasonSentence', { reason: health.scheduler.last_cache_reuse_skip.partial_reuse_unavailable_reason })}</>
              )}
            </div>
          )}
        </div>
      )}

      {/* Cache */}
      {health?.cache && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.view.cache')}</h4>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {health.cache.totals?.ram_tokens_cached != null && (
              <InfoCard
                label={t('sessions.cache.ramResidentTokens')}
                value={(health.cache.totals.ram_tokens_cached || 0).toLocaleString()}
              />
            )}
            {health.cache.totals?.l1_indexed_tokens != null && (
              <InfoCard
                label={t('sessions.cache.l1IndexedTokens')}
                value={(health.cache.totals.l1_indexed_tokens || 0).toLocaleString()}
              />
            )}
            {health.cache.totals?.l1_resident_bytes_mb != null && (
              <InfoCard
                label={t('sessions.cache.l1ResidentMemory')}
                value={`${(health.cache.totals.l1_resident_bytes_mb || 0).toFixed(1)} / ${(health.cache.totals.l1_max_resident_bytes_mb || 0).toFixed(1)} MB`}
              />
            )}
            {health.cache.totals?.l1_evictions != null && (
              <InfoCard
                label={t('sessions.cache.l1Evictions')}
                value={(health.cache.totals.l1_evictions || 0).toLocaleString()}
              />
            )}
            {(health.cache.totals?.l2_tokens_on_disk_store_sum ?? health.cache.totals?.l2_tokens_on_disk) != null && (
              <InfoCard
                label={t('sessions.performance.l2TokenEntries')}
                value={t('sessions.performance.storeSumValue', {
                  count: (health.cache.totals?.l2_tokens_on_disk_store_sum ?? health.cache.totals?.l2_tokens_on_disk ?? 0).toLocaleString(),
                })}
              />
            )}
            {health.cache.disk_cache?.entries != null && (
              <InfoCard
                label={t('sessions.performance.promptL2')}
                value={t('sessions.performance.entriesTokensValue', {
                  entries: health.cache.disk_cache.entries || 0,
                  tokens: (health.cache.disk_cache.total_tokens_on_disk || 0).toLocaleString(),
                })}
              />
            )}
            {health.cache.block_disk_cache?.blocks_on_disk != null && (
              <InfoCard
                label={t('sessions.performance.blockDiskL2Ssd')}
                value={t('sessions.performance.blocksTokensValue', {
                  blocks: health.cache.block_disk_cache.blocks_on_disk || 0,
                  tokens: (health.cache.block_disk_cache.total_tokens_on_disk || 0).toLocaleString(),
                })}
              />
            )}
            {health.cache.ssm_companion?.disk?.entries != null && (
              <InfoCard
                label={t('sessions.performance.ssmL2')}
                value={t('sessions.performance.entriesTokensValue', {
                  entries: health.cache.ssm_companion.disk.entries || 0,
                  tokens: (health.cache.ssm_companion.disk.total_tokens_on_disk || 0).toLocaleString(),
                })}
              />
            )}
            {health.cache.ssm_companion?.nbytes_mb != null && (
              <InfoCard
                label={t('sessions.performance.ssmResidentMemory')}
                value={health.cache.ssm_companion.max_bytes_mb != null
                  ? `${health.cache.ssm_companion.nbytes_mb.toFixed(1)} / ${health.cache.ssm_companion.max_bytes_mb.toFixed(1)} MB`
                  : `${health.cache.ssm_companion.nbytes_mb.toFixed(1)} MB`}
              />
            )}
            {health.cache.ssm_companion?.evictions != null && (
              <InfoCard
                label={t('sessions.cache.ssmEvictions')}
                value={health.cache.ssm_companion.evicted_bytes_mb != null && health.cache.ssm_companion.evicted_bytes_mb > 0
                  ? `${health.cache.ssm_companion.evictions} / ${health.cache.ssm_companion.evicted_bytes_mb.toFixed(1)} MB`
                  : String(health.cache.ssm_companion.evictions)}
              />
            )}
          </div>
        </div>
      )}

      {/* Memory */}
      {health?.memory && (
        <div>
          <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">{t('sessions.performance.gpuMemoryMetal')}</h4>
          <div className="grid grid-cols-3 gap-2">
            <MemoryCard label={t('sessions.performance.memActive')} value={health.memory.active_mb} />
            <MemoryCard label={t('sessions.performance.memPeak')} value={health.memory.peak_mb} />
            <MemoryCard label={t('sessions.view.cache')} value={health.memory.cache_mb} />
          </div>

          {/* Memory Graph */}
          {history.length > 1 && (
            <div className="mt-3">
              <div className="text-xs text-muted-foreground mb-1">{t('sessions.performance.memoryOverTime')}</div>
              <MiniGraph data={history} />
            </div>
          )}
        </div>
      )}

      {!health && !error && (
        <div className="text-sm text-muted-foreground">{t('sessions.performance.loadingHealthData')}</div>
      )}
    </div>
  )
}

function formatWeightQuant(
  health: HealthData,
  t: (key: string, params?: Record<string, string | number>) => string,
): string {
  const q = health.quantization
  const qf = health.quantization_format
  const bits =
    q?.actual_bits ??
    q?.target_bits ??
    q?.config_bits ??
    q?.routed_expert_bits ??
    q?.mxtq_bits ??
    qf?.actual_bits ??
    qf?.target_bits
  const group = q?.group_size ?? qf?.block_size

  if (q?.profile) return `${q.profile}${bits != null ? ` ${bits}-bit` : ''}${group != null ? ` g${group}` : ''}`
  if (q?.weight_format) return `${q.weight_format.toUpperCase()}${bits != null ? ` ${bits}-bit` : ''}${group != null ? ` g${group}` : ''}`
  if (q?.codec === 'turboquant_codebook') {
    return q.routed_expert_bits_label || `JANGTQ${bits != null ? ` ${bits}-bit` : ''}${group != null ? ` g${group}` : ''}`
  }
  if (qf?.type) return `${qf.type.toUpperCase()}${bits != null ? ` ${bits}-bit` : ''}${group != null ? ` g${group}` : ''}`
  return bits != null ? `${bits}-bit` : t('sessions.performance.unknown')
}

// A missing MTP measurement is UNKNOWN, never zero: a sparse family payload
// must not read as "0 accepted / 0 drafted" (Cache/Perf display audit,
// 2026-09-07). An explicit 0 from the engine still renders as 0.
export function reportedCount(v: number | null | undefined): string {
  return typeof v === 'number' && Number.isFinite(v) ? String(v) : '—'
}

// The MTP cards describe ONE request; say which, with its final state and
// depth, so a previous generation's statistics are never read as the active
// tool step or the whole outer turn.
export function formatMtpScope(m: {
  request_id?: string
  finish_reason?: string
  final_depth?: number
  policy?: string | null
  configured_depth?: number | null
}): string {
  const id = m.request_id ? m.request_id.slice(-12) : '—'
  const finish = m.finish_reason || 'last completed'
  const configured = typeof m.configured_depth === 'number' ? `D${m.configured_depth}` : null
  const final = typeof m.final_depth === 'number' ? `D${m.final_depth}` : null
  const depth = configured && final && configured !== final ? `${configured}→${final}` : (final || configured || '—')
  const policy = m.policy ? ` ${m.policy}` : ''
  return `${id} · ${finish} · ${depth}${policy}`
}

function formatPercent(value?: number | null): string {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  return `${(Number(value) * 100).toFixed(1)}%`
}

function formatMtpDepthRates(
  rates: Record<string, number | null>,
  finalDepth?: number,
): string {
  const depth = finalDepth ? ` D${finalDepth}` : ''
  return `D1 ${formatPercent(rates.d1)} / D2 ${formatPercent(rates.d2)} / D3 ${formatPercent(rates.d3)}${depth}`
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-background px-2 py-1.5 rounded border border-border">
      <div className="text-[10px] text-muted-foreground">{label}</div>
      <div className="font-mono text-xs break-words">{value}</div>
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
