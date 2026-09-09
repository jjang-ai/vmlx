import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

const state = vi.hoisted(() => ({
  handlers: new Map<string, Function>(),
  rows: [] as any[],
  invalid: false,
  stop: vi.fn(),
  create: vi.fn(),
  start: vi.fn(),
}))
vi.mock('electron', () => ({ ipcMain: { handle: (name: string, fn: Function) => state.handlers.set(name, fn) } }))
vi.mock('../src/main/sessions', () => ({ sessionManager: {
  getSession: (id: string) => state.rows.find(row => row.id === id),
  stopSession: (...args: any[]) => state.stop(...args),
  createSession: (...args: any[]) => state.create(...args),
  startSession: (...args: any[]) => state.start(...args),
} }))
vi.mock('../src/main/database', () => ({ db: {
  getSessions: () => state.rows,
  setImageModelPath: vi.fn(),
} }))
vi.mock('../src/shared/imageLocalModel', async original => ({
  ...await original<object>(),
  resolveLocalImageModelDirectory: (path: string) => state.invalid
    ? { kind: 'missing', path }
    : { kind: 'model', path, quantize: 8, quantizeSource: 'header' },
  localImageModelError: () => ({ code: 'missing', message: 'Missing folder' }),
  resolveImageModelForLocalDirectory: () => ({
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
    state.rows = [{
      id: 'image-owned', type: 'local', status: 'standby', port: 8000,
      modelPath: '/models/edit/q8',
      config: JSON.stringify({ modelType: 'image', imageMode: 'edit', imageQuantize: 8, mfluxClass: 'QwenImageEdit' }),
    }]
    state.stop.mockReset().mockImplementation(async (id: string) => {
      state.rows.find(row => row.id === id).status = 'stopped'
    })
    state.start.mockReset().mockResolvedValue(undefined)
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

  it('does not create or relaunch if the owning stop fails', async () => {
    state.stop.mockRejectedValue(new Error('Stop refused'))
    const result = await state.handlers.get('image:startServer')!({}, '/models/edit/q8', 8, 'edit')
    expect(result).toMatchObject({ success: false, error: 'Stop refused' })
    expect(state.create).not.toHaveBeenCalled()
    expect(state.start).not.toHaveBeenCalled()
  })
})
