import { expect, it, vi } from 'vitest'
import { withImageServerStartProgress, type ImageServerStartProgress } from '../src/shared/imageServerStartProgress'

it('subscribes before launch and isolates overlapping requests', async () => {
  let receive: (e: ImageServerStartProgress) => void = () => {}
  const remove = vi.fn(), accept = vi.fn()
  const subscribe = vi.fn(callback => { receive = callback; return remove })
  const event = { requestId: 'own', label: 'repair', labelKey: 'key', notice: false }
  const invoke = vi.fn(async () => { receive({...event, requestId:'other'}); receive(event); return 17 })
  await expect(withImageServerStartProgress('own', subscribe, invoke, accept)).resolves.toBe(17)
  expect(subscribe.mock.invocationCallOrder[0]).toBeLessThan(invoke.mock.invocationCallOrder[0])
  expect(accept).toHaveBeenCalledExactlyOnceWith(event)
  expect(remove).toHaveBeenCalledOnce()
})

it('removes the listener after failed IPC without hiding the error', async () => {
  const remove = vi.fn()
  await expect(withImageServerStartProgress('own', () => remove, async () => { throw Error('broken shard') }))
    .rejects.toThrow('broken shard')
  expect(remove).toHaveBeenCalledOnce()
})
