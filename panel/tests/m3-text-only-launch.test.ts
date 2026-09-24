import { describe, expect, it, vi } from 'vitest'

vi.mock('../src/main/database', () => ({ db: {
  getSessions: () => [], getSetting: () => undefined,
} }))
vi.mock('electron', () => ({
  app: { getAppPath: () => process.cwd(), getPath: () => '/tmp', isPackaged: false },
  powerSaveBlocker: { isStarted: () => false, start: () => 1, stop: () => undefined },
}))
vi.mock('../src/main/model-config-registry', () => ({
  detectModelConfigFromDir: () => ({
    family: 'minimax_m3', isMultimodal: true, m3VlRoute: true,
    toolParser: 'minimax_m3', reasoningParser: 'minimax_m3',
  }),
}))

import { SessionManager } from '../src/main/sessions'

describe('M3 production launch arguments', () => {
  it.each([undefined, true, false])('respects explicit media mode %s', (isMultimodal) => {
    const args = SessionManager.prototype.buildArgs({
      modelPath: '/fixture/m3', host: '127.0.0.1', port: 8034,
      isMultimodal,
    } as any)
    expect(args.includes('--is-mllm')).toBe(false)
    expect(args.includes('--text-only')).toBe(isMultimodal === false)
  })
})
