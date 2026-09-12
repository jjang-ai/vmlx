import { describe, expect, it, vi, beforeEach } from 'vitest'
import { apiCapabilityKey, apiCapabilityLabel, sessionCapabilityModalities } from '../src/shared/apiModelCapabilities'

const mocks = vi.hoisted(() => ({ handlers: new Map<string, Function>(), getSession: vi.fn() }))
vi.mock('electron', () => ({ ipcMain: { handle: (name: string, fn: Function) => mocks.handlers.set(name, fn) } }))
vi.mock('../src/main/database', () => ({ db: { getSession: mocks.getSession } }))
vi.mock('../src/main/sessions', () => ({ resolveUrl: async (url: string) => url, connectHost: (host: string) => host }))
import { registerPerformanceHandlers } from '../src/main/ipc/performance'

describe('live API model capabilities', () => {
  const session = { id: 's', host: '127.0.0.1', port: 8001, modelPath: '/model', pid: 123, status: 'running', type: 'local', config: '{}' }
  beforeEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); mocks.getSession.mockReset(); mocks.handlers.clear(); registerPerformanceHandlers() })
  it('shows actual modalities, not the MLLM lane or stale configuration', () => {
    expect(apiCapabilityLabel({ modalities: ['text', 'vision', 'video'] })).toBe('text · vision · video')
    expect(apiCapabilityLabel({ model_type: 'mllm' })).toBeNull()
    expect(apiCapabilityLabel({ modalities: ['vision'], media: { runtime_modalities: ['text'] } })).toBe('text')
    expect(apiCapabilityLabel({ modalities: ['vision'], media: { runtime_modalities: [] } })).toBeNull()
  })
  it('keys change on restart, switch and stop', () => {
    for (const change of [{ pid: 124 }, { modelPath: '/other' }, { status: 'stopped' }, { port: 8002 }]) {
      expect(apiCapabilityKey({ ...session, ...change })).not.toBe(apiCapabilityKey(session))
    }
  })
  it('binds drawer modalities only to the exact running session', () => {
    const reply = { key: apiCapabilityKey(session), capabilities: { modalities: ['text', 'vision', 'video'] } }
    expect(sessionCapabilityModalities(session, reply)).toEqual(['text', 'vision', 'video'])
    for (const change of [{ id: 'other' }, { pid: 124 }, { modelPath: '/other' }, { status: 'stopped' }, { status: 'standby' }, { port: 8002 }, { host: 'other' }]) {
      expect(sessionCapabilityModalities({ ...session, ...change }, reply)).toBeUndefined()
    }
    expect(sessionCapabilityModalities({ ...session, status: 'stopped' }, {
      ...reply, key: apiCapabilityKey({ ...session, status: 'stopped' }),
    })).toBeUndefined()
  })
  it('distinguishes unknown drawer modalities from live text-only metadata', () => {
    for (const capabilities of [null, {}, { model_type: 'mllm' }, { modalities: [] }, { modalities: 'video' }]) {
      expect(sessionCapabilityModalities(session, { key: apiCapabilityKey(session), capabilities })).toBeUndefined()
    }
    expect(sessionCapabilityModalities(session, null)).toBeUndefined()
    expect(sessionCapabilityModalities(session, {
      key: apiCapabilityKey(session),
      capabilities: { modalities: ['vision', 'video'], media: { runtime_modalities: ['text'] } },
    })).toEqual(['text'])
  })
  it.each(['standby', 'stopped', 'loading'])('does not wake a %s session', async status => {
    mocks.getSession.mockReturnValue({ ...session, status })
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch)
    expect(await mocks.handlers.get('performance:capabilities')!(null, 's')).toBeNull()
    expect(fetch).not.toHaveBeenCalled()
  })
  it('does not probe a remote provider', async () => {
    mocks.getSession.mockReturnValue({ ...session, type: 'remote' })
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch)
    expect(await mocks.handlers.get('performance:capabilities')!(null, 's')).toBeNull()
    expect(fetch).not.toHaveBeenCalled()
  })
  it('rejects metadata from a process replaced during the fetch', async () => {
    mocks.getSession.mockReturnValueOnce(session).mockReturnValueOnce({ ...session, pid: 124 })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ modalities: ['vision'] }) }))
    expect(await mocks.handlers.get('performance:capabilities')!(null, 's')).toBeNull()
  })
  it('binds an authenticated live result without returning credentials', async () => {
    mocks.getSession.mockReturnValue({ ...session, config: '{"apiKey":"test-only"}' })
    const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ modalities: ['text'] }) }); vi.stubGlobal('fetch', fetch)
    const result = await mocks.handlers.get('performance:capabilities')!(null, 's')
    expect(result).toEqual({ key: apiCapabilityKey(session), capabilities: { modalities: ['text'] } })
    expect(fetch.mock.calls[0][1].headers).toEqual({ Authorization: 'Bearer test-only' })
  })
  it('leaves failed capability requests unknown', async () => {
    mocks.getSession.mockReturnValue(session)
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new Error('unavailable')))
    expect(await mocks.handlers.get('performance:capabilities')!(null, 's')).toBeNull()
  })
})
