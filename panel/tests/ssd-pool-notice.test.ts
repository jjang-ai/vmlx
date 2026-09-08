import { describe, expect, it } from 'vitest'
import { readSsdPoolSnapshot, ssdPoolNoticeKind } from '../src/renderer/src/components/sessions/ssdPoolNoticeState'

const snapshot = { root: '/cache', used: 950, cap: 1000, evicted: 0 }
describe('SSD pool capacity notice', () => {
  it('uses actual effective bytes at arbitrary GB or percent-derived caps', () => {
    for (const cap of [1073741824, 399625232056, 7654321]) {
      expect(ssdPoolNoticeKind({ ...snapshot, cap, used: cap * .96 }, null)).toBe('capacity')
      expect(ssdPoolNoticeKind({ ...snapshot, cap, used: cap * .90 }, null)).toBeNull()
    }
  })
  it('does not invent a zero cap or usage from missing or unaccounted telemetry', () => {
    expect(readSsdPoolSnapshot({})).toBeNull()
    expect(readSsdPoolSnapshot({ block_disk_cache: { global_budget: { accounted: false } } })).toBeNull()
    expect(ssdPoolNoticeKind({ ...snapshot, cap: 0 }, null)).toBeNull()
  })
  it('reports only observed new eviction, not historical totals or a different pool', () => {
    expect(ssdPoolNoticeKind({ ...snapshot, used: 10, evicted: 99 }, null)).toBeNull()
    expect(ssdPoolNoticeKind({ ...snapshot, used: 10, evicted: 99 }, snapshot)).toBe('evicted')
    expect(ssdPoolNoticeKind({ ...snapshot, root: '/other', used: 10, evicted: 99 }, snapshot)).toBeNull()
  })
  it('accepts valid shared telemetry without substituting per-model usage', () => {
    expect(readSsdPoolSnapshot({ block_disk_cache: { disk_size_bytes: 1, global_budget: {
      root: '/cache', bytes_after: 950, max_size_bytes: 1000, evicted_entries_total: 0, accounted: true,
    } } })).toEqual(snapshot)
  })
})
