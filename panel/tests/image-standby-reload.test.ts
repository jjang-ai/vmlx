import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({
  handlers: new Map<string, Function>(),
  rows: [] as any[],
  invalid: false,
  unknownArchitecture: false,
  storedPath: null as string | null,
  registerPath: vi.fn(),
  stop: vi.fn(),
  create: vi.fn(),
  start: vi.fn(),
  preflight: vi.fn(),
}))
vi.mock('electron', () => ({ ipcMain: { handle: (name: string, fn: Function) => state.handlers.set(name, fn) } }))
vi.mock('../src/main/sessions', () => ({ sessionManager: {
  getSession: (id: string) => state.rows.find(row => row.id === id),
  stopSession: (...args: any[]) => state.stop(...args),
  createSession: (...args: any[]) => state.create(...args),
  startSession: (...args: any[]) => state.start(...args),
  preflightImageModelPath: (...args: any[]) => state.preflight(...args),
} }))
vi.mock('../src/main/database', () => ({ db: {
  getSessions: () => state.rows,
  getImageModelPath: () => state.storedPath ? { localPath: state.storedPath } : null,
  setImageModelPath: (...args: any[]) => state.registerPath(...args),
} }))
vi.mock('../src/shared/imageLocalModel', async original => ({
  ...await original<object>(),
  resolveLocalImageModelDirectory: (path: string) => !path.startsWith('/') ? null : state.invalid
    ? { kind: 'missing', path }
    : { kind: 'model', path, quantize: 8, quantizeSource: 'header' },
  localImageModelError: () => ({ code: 'missing', message: 'Missing folder' }),
  resolveImageModelForLocalDirectory: () => state.unknownArchitecture ? undefined : ({
    id: 'qwen-image-edit', name: 'Qwen Image Edit',
    mfluxClass: 'QwenImageEdit', mfluxName: 'qwen-image-edit', category: 'edit',
  }),
}))

import { registerImageHandlers } from '../src/main/ipc/image'

describe('explicit folder load replaces its own untracked standby session', () => {
  beforeAll(() => registerImageHandlers())
  beforeEach(async () => {
    state.rows = []
    // Reset activeImageSessionId through the real discovery handler.
    await state.handlers.get('image:getRunningServer')!({})
    state.invalid = false
    state.unknownArchitecture = false
    state.storedPath = null
    state.registerPath.mockReset()
    state.rows = [{
      id: 'image-owned', type: 'local', status: 'standby', port: 8000,
      modelPath: '/models/edit/q8',
      config: JSON.stringify({ modelType: 'image', imageMode: 'edit', imageQuantize: 8, mfluxClass: 'QwenImageEdit' }),
    }]
    state.stop.mockReset().mockImplementation(async (id: string) => {
      state.rows.find(row => row.id === id).status = 'stopped'
    })
    state.start.mockReset().mockResolvedValue(undefined)
    state.preflight.mockReset().mockResolvedValue(undefined)
    state.create.mockReset().mockImplementation(async (path: string) => {
      const existing = state.rows.find(row => row.modelPath === path)
      if (existing && ['standby', 'running', 'loading'].includes(existing.status))
        throw new Error('This model already has an active session')
      return existing || { id: 'replacement', port: 8001 }
    })
  })

  it('stops the same-folder standby owner before creation, without weakening the manager guard', async () => {
    const result = await state.handlers.get('image:startServer')!({}, '/models/edit/q8', 8, 'edit')
    expect(result.success).toBe(true)
    expect(state.stop).toHaveBeenCalledWith('image-owned')
    expect(state.stop.mock.invocationCallOrder[0]).toBeLessThan(state.create.mock.invocationCallOrder[0])
    expect(state.start).toHaveBeenCalledWith('image-owned')
    expect(state.preflight).toHaveBeenCalledWith('/models/edit/q8', expect.any(Function))
    expect(state.preflight.mock.invocationCallOrder[0]).toBeLessThan(state.stop.mock.invocationCallOrder[0])
  })

  it('rejects a broken checkpoint before cancelling, stopping, registering or creating a replacement', async () => {
    state.preflight.mockRejectedValue(new Error('referenced shard missing.safetensors is missing'))
    const result = await state.handlers.get('image:startServer')!({}, '/models/edit/q8', 8, 'edit')
    expect(result).toMatchObject({ success: false, serverKept: true, error: expect.stringContaining('missing.safetensors') })
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.create).not.toHaveBeenCalled()
    expect(state.start).not.toHaveBeenCalled()
    expect(state.registerPath).not.toHaveBeenCalled()
  })

  it('sends real repair progress to the requesting window before replacement', async () => {
    const send = vi.fn()
    state.preflight.mockImplementation(async (_path, progress) => {
      progress('[BUNDLE-ALIGNMENT] {"stage":"COPYING","shard":"transformer/0.safetensors","copied_bytes":1048576,"payload_bytes":2097152}')
      expect(state.stop).not.toHaveBeenCalled()
    })
    await state.handlers.get('image:startServer')!({sender:{send,isDestroyed:()=>false}}, '/models/edit/q8', 8, 'edit', undefined, 'own-load')
    expect(send).toHaveBeenCalledTimes(2)
    expect(send).toHaveBeenLastCalledWith('image:serverStartProgress', expect.objectContaining({
      requestId:'own-load', notice:false, label:expect.stringContaining('1.0 / 2.0 MiB'),
    }))
  })

  it('rediscovers a standby image session for the page without claiming it is running', async () => {
    const result = await state.handlers.get('image:getRunningServer')!({})
    expect(result).toMatchObject({ sessionId: 'image-owned', status: 'standby', imageMode: 'edit', quantize: 8 })
  })

  it('prefers a running image over a different sleeper when no session is selected', async () => {
    state.rows.push({ ...state.rows[0], id: 'running', modelPath: '/other/model', status: 'running' })
    const result = await state.handlers.get('image:getRunningServer')!({})
    expect(result).toMatchObject({ sessionId: 'running', status: 'running' })
    expect(state.rows[0].id).toBe('image-owned')
  })

  it('does not stop a different folder sharing the same basename', async () => {
    const result = await state.handlers.get('image:startServer')!({}, '/other/edit/q8', 8, 'edit')
    expect(result.success).toBe(true)
    expect(state.stop).not.toHaveBeenCalled()
  })

  it('validates the new folder before stopping any session', async () => {
    state.invalid = true
    const result = await state.handlers.get('image:startServer')!({}, '/models/edit/q8', 8, 'edit')
    expect(result).toMatchObject({ success: false, serverKept: true })
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.create).not.toHaveBeenCalled()
  })

  it('does not allow an explicit adapter override to bypass a rejected folder', async () => {
    state.invalid = true
    const result = await state.handlers.get('image:startServer')!({}, '/models/edit/q8', 8, 'edit', { mfluxClass: 'QwenImageEdit' })
    expect(result).toMatchObject({ success: false, serverKept: true })
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.create).not.toHaveBeenCalled()
  })

  it('does not revive an unresolved local architecture from a familiar folder name', async () => {
    state.unknownArchitecture = true
    const result = await state.handlers.get('image:startServer')!({}, '/models/FLUX.1-schnell-mflux-4bit', 4)
    expect(result).toMatchObject({ success: false, serverKept: true })
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.create).not.toHaveBeenCalled()
    expect(state.registerPath).not.toHaveBeenCalled()
  })

  it('does not create or relaunch if the owning stop fails', async () => {
    state.stop.mockRejectedValue(new Error('Stop refused'))
    const result = await state.handlers.get('image:startServer')!({}, '/models/edit/q8', 8, 'edit')
    expect(result).toMatchObject({ success: false, error: 'Stop refused' })
    expect(state.create).not.toHaveBeenCalled()
    expect(state.start).not.toHaveBeenCalled()
  })

  it('uses actual folder adapter and precision for a stale registered model id', async () => {
    state.storedPath = __dirname
    const result = await state.handlers.get('image:startServer')!({}, 'schnell', 4)
    expect(result).toMatchObject({ success: true, quantize: 8, modelId: 'qwen-image-edit', imageMode: 'edit' })
    expect(state.create).toHaveBeenCalledWith(__dirname, expect.objectContaining({
      mfluxClass: 'QwenImageEdit', imageQuantize: 8, servedModelName: 'qwen-image-edit', imageMode: 'edit',
    }))
  })

  it('rejects unresolved metadata reached through a registry id without replacing the owner', async () => {
    state.storedPath = __dirname
    state.unknownArchitecture = true
    const result = await state.handlers.get('image:startServer')!({}, 'schnell', 4)
    expect(result).toMatchObject({ success: false, serverKept: true })
    expect(state.stop).not.toHaveBeenCalled()
    expect(state.create).not.toHaveBeenCalled()
    expect(state.registerPath).not.toHaveBeenCalled()
  })
})
