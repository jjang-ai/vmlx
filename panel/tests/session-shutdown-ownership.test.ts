import { afterEach, describe, expect, it, vi } from 'vitest'
import { spawn } from 'node:child_process'
import { once } from 'node:events'

const { db, state } = vi.hoisted(() => {
  const state = { sessions: [] as any[] }
  return { state, db: {
    getSessions: vi.fn(() => state.sessions),
    updateSession: vi.fn((id: string, patch: any) => {
      Object.assign(state.sessions.find(s => s.id === id), patch)
    }),
    getSetting: vi.fn(),
  } }
})
vi.mock('../src/main/database', () => ({ db }))
vi.mock('electron', () => ({
  app: { getAppPath: () => process.cwd(), getPath: () => '/tmp', isPackaged: false },
  powerSaveBlocker: { isStarted: () => false, start: () => 1, stop: () => undefined },
}))
import { SessionManager } from '../src/main/sessions'

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.clearAllMocks()
  state.sessions = []
})

describe('app shutdown engine ownership', () => {
  it('leaves a real independent worker alive while its owned worker exits', async () => {
    const owned = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
    const foreign = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
    try {
      await Promise.all([once(owned, 'spawn'), once(foreign, 'spawn')])
      const manager = new SessionManager()
      vi.spyOn(manager, 'detect').mockResolvedValue([
        { pid: foreign.pid!, port: 8002, modelPath: '/foreign/image', healthy: true },
      ])
      ;(manager as any).processes.set('owned', { process: owned, adoptedPid: null })
      state.sessions = [{ id: 'owned', status: 'running', type: 'local', pid: owned.pid }]
      await manager.stopAll()
      expect(owned.exitCode !== null || owned.signalCode !== null).toBe(true)
      expect(foreign.exitCode).toBeNull()
      expect(foreign.signalCode).toBeNull()
      expect(() => process.kill(foreign.pid!, 0)).not.toThrow()
    } finally {
      // These are the two harmless workers created by this test, not engines.
      await Promise.all([owned, foreign].map(async child => {
        if (child.exitCode !== null || child.signalCode !== null) return
        const closed = once(child, 'exit')
        child.kill('SIGKILL')
        await closed
      }))
    }
  })

  it('never signals an unowned discovered engine or rewrites its session status', async () => {
    vi.useFakeTimers()
    const manager = new SessionManager()
    const detect = vi.spyOn(manager, 'detect').mockResolvedValue([
      { pid: 902, port: 8002, modelPath: '/foreign/image', healthy: true },
    ])
    const kill = vi.spyOn(process, 'kill').mockReturnValue(true)
    const ownedKill = vi.fn()
    ;(manager as any).processes.set('owned', { process: { pid: 901, kill: ownedKill }, adoptedPid: null })
    state.sessions = [
      { id: 'owned', status: 'running', type: 'local', pid: 901 },
      { id: 'foreign', status: 'standby', type: 'local', pid: 902 },
      { id: 'remote', status: 'running', type: 'remote', pid: undefined },
    ]
    const stopping = manager.stopAll()
    await vi.runAllTimersAsync()
    await stopping
    expect(detect).not.toHaveBeenCalled()
    expect(kill.mock.calls.some(([pid]) => Math.abs(Number(pid)) === 902)).toBe(false)
    expect(ownedKill).toHaveBeenCalledWith('SIGTERM')
    expect(ownedKill).toHaveBeenCalledWith('SIGKILL')
    expect(state.sessions[0].status).toBe('stopped')
    expect(state.sessions[1].status).toBe('standby')
    expect(state.sessions[2].status).toBe('running')
  })

  it('still stops explicitly adopted engines, including deep standby', async () => {
    vi.useFakeTimers()
    const manager = new SessionManager()
    vi.spyOn(manager, 'detect').mockResolvedValue([])
    const kill = vi.spyOn(process, 'kill').mockReturnValue(true)
    ;(manager as any).processes.set('adopted', { process: null, adoptedPid: 903 })
    state.sessions = [{ id: 'adopted', status: 'standby', type: 'local', pid: 903 }]
    const stopping = manager.stopAll()
    await vi.runAllTimersAsync()
    await stopping
    expect(kill).toHaveBeenCalledWith(-903, 'SIGTERM')
    expect(kill.mock.calls.some(([pid, sig]) => Math.abs(Number(pid)) === 903 && sig === 'SIGKILL')).toBe(true)
    expect(state.sessions[0]).toMatchObject({ status: 'stopped', pid: undefined, standbyDepth: null })
    expect((manager as any).processes.size).toBe(0)
  })

  it('does not force-kill a child that exited during graceful shutdown', async () => {
    vi.useFakeTimers()
    const manager = new SessionManager()
    vi.spyOn(manager, 'detect').mockResolvedValue([])
    vi.spyOn(process, 'kill').mockReturnValue(true)
    const child = { pid: 904, exitCode: null as number | null, signalCode: null, kill: vi.fn() }
    child.kill.mockImplementation((signal: string) => { if (signal === 'SIGTERM') child.exitCode = 0 })
    ;(manager as any).processes.set('owned', { process: child, adoptedPid: null })
    state.sessions = [{ id: 'owned', status: 'running', type: 'local', pid: 904 }]
    const stopping = manager.stopAll()
    await vi.runAllTimersAsync()
    await stopping
    expect(child.kill.mock.calls.map(([signal]) => signal)).toEqual(['SIGTERM'])
  })
})
