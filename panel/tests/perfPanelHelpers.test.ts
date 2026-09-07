import { describe, expect, it } from 'vitest'
import { formatMtpScope, reportedCount } from '../src/renderer/src/components/sessions/PerformancePanel'

// Cache/Perf display audit (2026-09-07): an absent MTP measurement must read as
// unknown, never as zero, and the MTP cards must say which request they describe.
describe('Performance panel MTP helpers', () => {
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
