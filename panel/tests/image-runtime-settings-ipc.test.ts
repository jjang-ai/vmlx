import { beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'
const state = vi.hoisted(() => ({
  handlers: new Map<string, Function>(), values: new Map<string, string>(), rows: [] as any[],
}))
vi.mock('electron', () => ({ ipcMain: { handle: (key: string, fn: Function) => state.handlers.set(key, fn) } }))
vi.mock('../src/main/sessions', () => ({ sessionManager: {
  getSession: (id: string) => state.rows.find(row => row.id === id),
} }))
vi.mock('../src/main/database', () => ({ db: {
  getSetting: (key: string) => state.values.get(key),
  setSetting: (key: string, value: string) => state.values.set(key, value),
} }))
import { registerImageHandlers } from '../src/main/ipc/image'
describe('image settings IPC uses the actual server session', () => {
  beforeAll(() => registerImageHandlers())
  beforeEach(() => {
    state.values.clear()
    state.rows = [{ id: 'owned-q8', type: 'local', modelPath: '/not-mounted/q8',
      config: JSON.stringify({ modelType: 'image', imageQuantize: 8, servedModelName: 'qwen-image-edit' }) }]
  })
  it('resolves canonical defaults despite a precision-only folder name', () => {
    expect(state.handlers.get('image:getRuntimeSettings')!({}, 'owned-q8'))
      .toMatchObject({ steps: 28, guidance: 4, quantize: 8 })
  })
  it('round-trips values under the real owner without accepting client precision', () => {
    state.handlers.get('image:saveRuntimeSettings')!({}, 'owned-q8', { steps: 6, guidance: 0, quantize: 4, seed: 5 })
    expect(state.handlers.get('image:getRuntimeSettings')!({}, 'owned-q8'))
      .toMatchObject({ steps: 6, guidance: 0, quantize: 8, seed: undefined })
  })
  it('rejects missing, remote, and text sessions without writing settings', () => {
    state.rows.push({ id: 'remote', type: 'remote', config: '{}' }, { id: 'text', type: 'local', config: '{"modelType":"text"}' })
    for (const id of ['missing', 'remote', 'text']) {
      expect(() => state.handlers.get('image:saveRuntimeSettings')!({}, id, { steps: 5 })).toThrow()
    }
    expect(state.values.size).toBe(0)
  })
})

