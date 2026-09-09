import { describe, expect, it } from 'vitest'
import { ImageSubmissionGuard } from '../src/shared/imageSubmissionGuard'

describe('image submission busy-state ownership', () => {
  it('allows current progress during submission without allowing idle to clear busy', () => {
    const guard = new ImageSubmissionGuard()
    const owner = guard.begin()!
    const poll = guard.snapshot()
    expect(guard.isCurrent(poll)).toBe(true)
    expect(guard.canApply(poll)).toBe(false)
    guard.finish(owner)
    expect(guard.isCurrent(poll)).toBe(false)
    guard.begin()
    expect(guard.isCurrent(poll)).toBe(false)
  })
  it('rejects an idle poll during history creation and prevents duplicate submission', () => {
    const guard = new ImageSubmissionGuard()
    const earlyPoll = guard.snapshot()
    const owner = guard.begin()!
    expect(guard.canApply(earlyPoll)).toBe(false)
    expect(guard.canApply(guard.snapshot())).toBe(false)
    expect(guard.begin()).toBeNull()
    expect(guard.finish(owner)).toBe(true)
    expect(guard.canApply(earlyPoll)).toBe(false)
    expect(guard.canApply(guard.snapshot())).toBe(true)
  })

  it('does not let an older request clear a newer submission', () => {
    const guard = new ImageSubmissionGuard()
    const first = guard.begin()!
    guard.finish(first)
    const second = guard.begin()!
    expect(guard.finish(first)).toBe(false)
    expect(guard.canApply(guard.snapshot())).toBe(false)
    expect(guard.finish(second)).toBe(true)
  })

  it('ignores a delayed active poll even if a local job has already completed', () => {
    const guard = new ImageSubmissionGuard()
    const poll = guard.snapshot()
    const owner = guard.begin()!
    guard.finish(owner)
    expect(guard.canApply(poll)).toBe(false)
  })
})
