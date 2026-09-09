import { describe, expect, it, vi } from 'vitest'
import { observeImageServerDiscovery } from '../src/shared/imageServerDiscovery'
const flush = () => new Promise(resolve => setImmediate(resolve))

describe('late image server discovery', () => {
  it('subscribes before snapshot and retries a ready event racing a null response', async () => {
    let ready!: () => void
    let resolveFirst!: (value: null) => void
    const first = new Promise<null>(resolve => { resolveFirst = resolve })
    const read = vi.fn().mockImplementationOnce(() => first).mockResolvedValue({ sessionId: 'image' })
    const accept = vi.fn()
    const stop = observeImageServerDiscovery({
      read, accept, isCurrent: () => true, onError: error => { throw error },
      onReady: callback => { ready = callback; expect(read).not.toHaveBeenCalled(); return () => {} },
    })
    ready(); ready()
    resolveFirst(null)
    await flush()
    expect(read).toHaveBeenCalledTimes(2)
    expect(accept).toHaveBeenCalledWith({ sessionId: 'image' })
    ready()
    await flush()
    expect(accept).toHaveBeenCalledTimes(1)
    stop()
  })

  it('accepts a later image-ready event after the initial empty snapshot', async () => {
    let ready!: () => void
    const read = vi.fn().mockResolvedValueOnce(null).mockResolvedValueOnce(null).mockResolvedValue({ sessionId: 'image' })
    const accept = vi.fn()
    const stop = observeImageServerDiscovery({
      read, accept, isCurrent: () => true, onError: vi.fn(),
      onReady: callback => { ready = callback; return () => {} },
    })
    await flush()
    ready(); await flush() // A text server became ready: still no image owner.
    expect(accept).not.toHaveBeenCalled()
    ready(); await flush()
    expect(accept).toHaveBeenCalledWith({ sessionId: 'image' })
    stop()
  })

  it.each(['unmount', 'manual selection'])('ignores a late snapshot after %s', async kind => {
    let current = true
    let resolve!: (value: {sessionId: string}) => void
    const read = vi.fn(() => new Promise<{sessionId: string}>(r => { resolve = r }))
    const accept = vi.fn(), unsubscribe = vi.fn()
    const stop = observeImageServerDiscovery({
      read, accept, isCurrent: () => current, onError: vi.fn(), onReady: () => unsubscribe,
    })
    if (kind === 'unmount') stop()
    else current = false
    resolve({ sessionId: 'stale-image' })
    await flush()
    expect(accept).not.toHaveBeenCalled()
    if (kind === 'unmount') expect(unsubscribe).toHaveBeenCalledOnce()
    else stop()
  })
})
