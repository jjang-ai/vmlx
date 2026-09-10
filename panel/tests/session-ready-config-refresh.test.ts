import { beforeEach, describe, expect, it, vi } from 'vitest'

const harness = vi.hoisted(() => ({ effects: [] as Array<() => unknown>, setters: [] as Array<ReturnType<typeof vi.fn>> }))
vi.mock('react', () => ({
  createContext: () => ({ Provider: () => null }),
  useContext: vi.fn(),
  useState: (initial: unknown) => { const setter = vi.fn(); harness.setters.push(setter); return [initial, setter] },
  useEffect: (effect: () => unknown) => { harness.effects.push(effect) },
  useCallback: (fn: unknown) => fn,
  useRef: (current: unknown) => ({ current }),
}))
vi.mock('../src/renderer/src/i18n', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
import { SessionsProvider } from '../src/renderer/src/contexts/SessionsContext'

describe('ready event refreshes promoted session config', () => {
  beforeEach(() => { harness.effects.length = 0; harness.setters.length = 0 })
  it('replaces the old active config after restart promotes pending settings', async () => {
    const handlers: Record<string, (data?: unknown) => void> = {}
    const old = { id: 'spark', status: 'running', config: '{"additionalArgs":"old"}' }
    const promoted = { ...old, pid: 42, config: '{"additionalArgs":"new"}' }
    const list = vi.fn().mockResolvedValueOnce([old]).mockResolvedValueOnce([promoted])
    const sessions: Record<string, unknown> = { list }
    for (const name of ['onCreated', 'onDeleted', 'onUpdated', 'onStarting', 'onReady', 'onStopped', 'onError', 'onHealth']) {
      sessions[name] = (callback: (data?: unknown) => void) => { handlers[name] = callback; return () => {} }
    }
    vi.stubGlobal('window', { api: { sessions } })
    try {
      SessionsProvider({ children: null })
      const cleanup = harness.effects[0]() as () => void
      await Promise.resolve()
      expect(list).toHaveBeenCalledTimes(1)
      handlers.onReady({ sessionId: 'spark', pid: 42, port: 8019 })
      await Promise.resolve()
      expect(list).toHaveBeenCalledTimes(2)
      expect(harness.setters[0]).toHaveBeenLastCalledWith([promoted])
      cleanup()
    } finally { vi.unstubAllGlobals() }
  })
})
