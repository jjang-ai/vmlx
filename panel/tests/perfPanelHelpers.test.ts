import { describe, expect, it } from 'vitest'
import { formatMtpScope, reportedCount, formatWeightQuant } from '../src/renderer/src/components/sessions/PerformancePanel'

// Cache/Perf display audit (2026-09-07): an absent MTP measurement must read as
// unknown, never as zero, and the MTP cards must say which request they describe.
describe('Performance panel MTP helpers', () => {
  it('uses text runtime ceiling and policy, and names AR rather than D0', () => {
    expect(formatMtpScope({ request_id: '27', finish_reason: 'stop', depth_ceiling: 3, final_depth: 1, depth_policy: 'adaptive' })).toBe('27 · stop · D3→D1 adaptive')
    expect(formatMtpScope({ request_id: '28', finish_reason: 'fallback_to_ar', depth_ceiling: 2, configured_depth: 3, final_depth: 0, depth_policy: 'fixed' })).toBe('28 · fallback_to_ar · D2→AR fixed')
    expect(formatMtpScope({ final_depth: Number.NaN, depth_ceiling: Number.NaN })).toBe('— · last completed · —')
  })
  it('reports unknown counts as a dash and explicit zero as 0', () => {
    expect(reportedCount(undefined)).toBe('—')
    expect(reportedCount(null)).toBe('—')
    expect(reportedCount(Number.NaN)).toBe('—')
    expect(reportedCount(0)).toBe('0')
    expect(reportedCount(52)).toBe('52')
  })

  it('binds the MTP cards to a request, its state and configured→effective depth', () => {
    expect(formatMtpScope({ request_id: 'chatcmpl-d59f0421', finish_reason: 'stop', final_depth: 1, configured_depth: 3, policy: 'fixed' })).toBe('mpl-d59f0421 · stop · D3→D1 fixed')
    expect(formatMtpScope({ request_id: 'resp_2333ab852daa', finish_reason: 'stop', final_depth: 3, configured_depth: 3 })).toBe('2333ab852daa · stop · D3')
    expect(formatMtpScope({})).toBe('— · last completed · —')
  })
})


describe('Performance panel mixed weight labels', () => {
  const t = (key: string) => key
  it('never labels a declared mixed bundle by its fallback module bits', () => {
    const health = {quantization:{profile:'JANG_2L',mixed_precision:true,config_bits:8,group_size:64}} as Parameters<typeof formatWeightQuant>[0]
    expect(formatWeightQuant(health,t)).toBe('JANG_2L mixed')
    health.quantization!.actual_bits = 2.73
    expect(formatWeightQuant(health,t)).toBe('JANG_2L mixed (2.73 bpw)')
  })
  it('retains ordinary uniform labels', () => {
    const health = {quantization:{weight_format:'mxfp8',config_bits:8,group_size:32}} as Parameters<typeof formatWeightQuant>[0]
    expect(formatWeightQuant(health,t)).toBe('MXFP8 8-bit g32')
  })
})


describe('JANGH component labels', () => {
  it('prefers the component name while preserving legacy serialized format', () => {
    const health = {quantization:{runtime_component_label:'JANGH',weight_format:'jangtq2',mixed_precision:true,config_bits:8}} as Parameters<typeof formatWeightQuant>[0]
    expect(formatWeightQuant(health, key => key)).toBe('JANGH mixed')
    expect(health.quantization!.weight_format).toBe('jangtq2')
  })
})
