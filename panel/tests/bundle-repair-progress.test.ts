import { describe, expect, it } from 'vitest'
import { createBundleRepairProgressReporter } from '../src/main/bundle-repair-progress'

describe('bundle repair startup progress', () => {
  const line = (stage: string, extra = {}) => '[BUNDLE-ALIGNMENT] ' + JSON.stringify({ stage, shard: '/models/a/model-00002-of-00026.safetensors', ...extra })
  it('announces once and publishes every actual stage without an overall percentage', () => {
    const events: any[] = []
    const report = createBundleRepairProgressReporter((message, notice) => events.push({ ...message, notice }))
    for (const stage of ['MISALIGNED_DETECTED', 'COPYING', 'VALIDATED', 'TRANSACTION_COMMITTED', 'REPAIRED_ON_DISK']) report(line(stage))
    expect(events).toHaveLength(6)
    expect(events.filter(e => e.notice)).toHaveLength(1)
    expect(events[0].label).toContain('current files')
    expect(events[1].label).toContain('model-00002-of-00026.safetensors')
    expect(events[5].label).toContain('continuing bundle check')
    expect(events.every(e => !('progress' in e))).toBe(true)
  })
  it('uses measured per-shard copied bytes only', () => {
    const events: any[] = []
    const report = createBundleRepairProgressReporter(m => events.push(m))
    report(line('COPYING', { copied_bytes: 1048576, payload_bytes: 4194304 }))
    expect(events.at(-1).label).toContain('1.0 / 4.0 MiB')
    report(line('COPYING', { copied_bytes: 9, payload_bytes: 2 }))
    expect(events.at(-1).label).not.toContain('MiB')
  })
  it('ignores malformed, unrelated and unknown input', () => {
    const events: any[] = []
    const report = createBundleRepairProgressReporter(m => events.push(m))
    for (const value of ['[BUNDLE-ALIGNMENT] {', 'unrelated', '[BUNDLE-ALIGNMENT] null', line('toString'), line('unknown')]) report(value)
    expect(events).toEqual([])
    report(line('REPAIR_FAILED'))
    expect(events.at(-1).labelKey).toBe('main.loadProgress.bundleRepairFailed')
  })
  it('does not share notice state between session starts', () => {
    for (let i = 0; i < 2; i++) {
      const notices: boolean[] = []
      createBundleRepairProgressReporter((_m, n) => notices.push(n))(line('COPYING'))
      expect(notices).toEqual([true, false])
    }
  })
})
