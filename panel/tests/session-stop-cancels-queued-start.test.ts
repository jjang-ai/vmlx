import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { db, state } = vi.hoisted(() => {
  const state = {
    sessions: [] as any[],
    settings: new Map<string, string>(),
  }
  const db = {
    getSessions: vi.fn(() => state.sessions),
    getSession: vi.fn((id: string) => state.sessions.find(session => session.id === id)),
    getSessionByModelPath: vi.fn((modelPath: string) =>
      state.sessions.find(session => session.modelPath === modelPath),
    ),
    getSetting: vi.fn((key: string) => state.settings.get(key)),
    createSession: vi.fn((session: any) => {
      state.sessions.push(session)
    }),
    updateSession: vi.fn((id: string, patch: Record<string, unknown>) => {
      const session = state.sessions.find(candidate => candidate.id === id)
      if (session) Object.assign(session, patch)
    }),
    deleteSession: vi.fn(),
  }
  return { db, state }
})

vi.mock('../src/main/database', () => ({ db }))
vi.mock('electron', () => ({
  app: {
    getAppPath: () => process.cwd(),
    getPath: () => '/tmp',
    isPackaged: false,
  },
  powerSaveBlocker: {
    isStarted: () => false,
    start: () => 1,
    stop: () => undefined,
  },
}))

import { SessionManager } from '../src/main/sessions'
import { beginImageGeneration, bindImageGenerationRequest, getImageGenerationStatus, resetImageGenerationStateForTests, wasImageGenerationCancelled } from '../src/main/ipc/imageGenerationState'

const temporaryBundles: string[] = []

function modelBundle(): string {
  const directory = mkdtempSync(join(tmpdir(), 'vmlx-stop-cancels-start-'))
  temporaryBundles.push(directory)
  writeFileSync(join(directory, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }))
  return directory
}

afterEach(() => {
  for (const directory of temporaryBundles.splice(0)) {
    rmSync(directory, { recursive: true, force: true })
  }
  state.sessions.length = 0
  state.settings.clear()
  resetImageGenerationStateForTests()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('explicit Stop cancels a queued start', () => {
  it('marks the owning image request before manual stop without releasing it prematurely', async () => {
    const manager = new SessionManager()
    state.sessions = [{ id: 'image-owner', type: 'local', modelPath: modelBundle(), status: 'stopped', config: '{}' }]
    const controller = beginImageGeneration('history')
    bindImageGenerationRequest(controller, 'image-owner', 'request')
    await manager.stopSession('image-owner')
    expect(wasImageGenerationCancelled(controller)).toBe(true)
    expect(getImageGenerationStatus()).toMatchObject({ generating: true, cancelling: true })
    expect(controller.signal.aborted).toBe(false)
  })

  it('marks the owning image request before detected-process replacement sends its first signal', async () => {
    vi.useFakeTimers()
    const manager = new SessionManager()
    const modelPath = modelBundle()
    state.sessions = [{ id: 'image-owner', type: 'local', modelPath, port: 8013, pid: 987654321, status: 'running', config: '{}' }]
    const controller = beginImageGeneration('history')
    bindImageGenerationRequest(controller, 'image-owner', 'request')
    const kill = vi.spyOn(manager as any, 'killPid').mockImplementation(() => {
      expect(wasImageGenerationCancelled(controller)).toBe(true)
    })
    vi.spyOn(process, 'kill').mockImplementation(() => true)
    const stop = (manager as any).terminateDetectedLocalEngine({ modelPath, port: 8013, pid: 987654321 })
    expect(kill).toHaveBeenCalledWith(987654321)
    await vi.advanceTimersByTimeAsync(1500)
    await stop
    expect(getImageGenerationStatus()).toMatchObject({ generating: true, cancelling: true })
  })
  /**
   * Save & Restart is renderer-orchestrated update -> stop -> start over
   * separate IPC calls. The session lock serializes those operations but
   * serialization is not cancellation: a user Stop that lands between the
   * restart's stop and its queued start used to execute FIRST and set the
   * session Stopped, after which the queued start spawned an engine the UI
   * no longer tracked. Observed live on the installed 1.6.44 smoke: UI and
   * DB said Stopped, port down, while a fresh ~98GB DSV4 engine (PID 98861)
   * stayed resident. Stop must defeat every queued start.
   */
  it('a start queued before an explicit stop aborts instead of spawning', async () => {
    const manager = new SessionManager()
    const sessionId = 'stop-beats-queued-start'
    state.sessions = [
      {
        id: sessionId,
        type: 'local',
        modelPath: modelBundle(),
        status: 'stopped',
        config: JSON.stringify({}),
      },
    ]

    const inner = vi
      .spyOn(manager as any, '_startSessionInner')
      .mockResolvedValue(undefined)

    // Hold the session lock so both operations queue in a known order.
    let releaseLock: () => void = () => {}
    const gate = new Promise<void>(resolve => {
      releaseLock = resolve
    })
    const lockHolder = (manager as any).withSessionLock(sessionId, () => gate)

    // 1. Start queues first (captures the pre-stop epoch)...
    const start = manager.startSession(sessionId)
    const startOutcome = start.then(
      () => ({ rejected: false as const }),
      (error: Error) => ({ rejected: true as const, message: error.message }),
    )
    // 2. ...then the explicit Stop arrives while the start is still queued.
    const stop = manager.stopSession(sessionId)

    releaseLock()
    await lockHolder
    await stop

    const outcome = await startOutcome
    expect(outcome.rejected).toBe(true)
    expect(outcome).toMatchObject({
      message: expect.stringContaining('stopped after this start was requested'),
    })
    expect(inner).not.toHaveBeenCalled()
    expect(state.sessions[0].status).toBe('stopped')
  })

  /**
   * The entry-epoch capture alone only cancels starts REQUESTED before the
   * Stop. Save & Restart dispatches its start after its own stop resolves, so
   * a user Stop landing between the pair advanced the epoch BEFORE the
   * restart's start was requested — the start captured the post-Stop epoch
   * and passed. restartSession carries the generation captured when the
   * restart began; after its own stop, it may start only if no later
   * explicit Stop advanced the generation.
   *
   * Exact ordering: restart begins -> restart-owned stop completes ->
   * explicit user Stop completes -> stale restart attempts start -> zero
   * spawn, and zero pre-lock side effects (single-model detection/adoption).
   */
  it('an explicit Stop after the restart-owned stop cancels the restart start', async () => {
    const manager = new SessionManager()
    const sessionId = 'stop-beats-stale-restart-start'
    state.sessions = [
      {
        id: sessionId,
        type: 'local',
        modelPath: modelBundle(),
        status: 'stopped',
        config: JSON.stringify({}),
      },
    ]
    // Single-model mode ON so the pre-lock side effects exist to be skipped.
    state.settings.set('gateway_single_model_mode', 'true')

    const inner = vi
      .spyOn(manager as any, '_startSessionInner')
      .mockResolvedValue(undefined)
    const detection = vi
      .spyOn(manager as any, 'stopDetectedLocalEnginesForSingleModel')
      .mockResolvedValue(undefined)
    const adoption = vi
      .spyOn(manager as any, 'adoptDetectedTargetProcessForStart')
      .mockResolvedValue(false)

    // Pause the restart BETWEEN its own stop and its start attempt so the
    // explicit user Stop can complete in that exact window.
    let restartStopDone: () => void = () => {}
    const restartStopCompleted = new Promise<void>(resolve => {
      restartStopDone = resolve
    })
    let releaseRestart: () => void = () => {}
    const restartGate = new Promise<void>(resolve => {
      releaseRestart = resolve
    })
    const realStop = SessionManager.prototype.stopSession.bind(manager)
    const stopSpy = vi
      .spyOn(manager, 'stopSession')
      .mockImplementation(async (id: string) => {
        stopSpy.mockRestore()
        await realStop(id)
        restartStopDone()
        await restartGate
      })

    // 1. Restart begins; its restart-owned stop completes.
    const restart = manager.restartSession(sessionId)
    const restartOutcome = restart.then(
      () => ({ rejected: false as const }),
      (error: Error) => ({ rejected: true as const, message: error.message }),
    )
    await restartStopCompleted
    // 2. The explicit user Stop completes (spy restored: this is a real stop).
    await manager.stopSession(sessionId)
    // 3. The stale restart attempts its start.
    releaseRestart()

    const outcome = await restartOutcome
    expect(outcome.rejected).toBe(true)
    expect(outcome).toMatchObject({
      message: expect.stringContaining('stopped after this start was requested'),
    })
    // 4. Zero spawn AND zero pre-lock side effects.
    expect(inner).not.toHaveBeenCalled()
    expect(detection).not.toHaveBeenCalled()
    expect(adoption).not.toHaveBeenCalled()
    expect(state.sessions[0].status).toBe('stopped')
  })

  it('an uninterrupted restartSession still starts', async () => {
    const manager = new SessionManager()
    const sessionId = 'restart-uninterrupted-starts'
    state.sessions = [
      {
        id: sessionId,
        type: 'local',
        modelPath: modelBundle(),
        status: 'stopped',
        config: JSON.stringify({}),
      },
    ]

    const inner = vi
      .spyOn(manager as any, '_startSessionInner')
      .mockResolvedValue(undefined)

    await expect(manager.restartSession(sessionId)).resolves.toBeUndefined()
    expect(inner).toHaveBeenCalledTimes(1)
  })

  it('the renderer restart ordering (stop, then start) still starts', async () => {
    const manager = new SessionManager()
    const sessionId = 'restart-ordering-still-works'
    state.sessions = [
      {
        id: sessionId,
        type: 'local',
        modelPath: modelBundle(),
        status: 'stopped',
        config: JSON.stringify({}),
      },
    ]

    const inner = vi
      .spyOn(manager as any, '_startSessionInner')
      .mockResolvedValue(undefined)

    // Save & Restart awaits the stop BEFORE requesting the start, so the
    // start captures the post-stop epoch and must proceed.
    await manager.stopSession(sessionId)
    await expect(manager.startSession(sessionId)).resolves.toBeUndefined()
    expect(inner).toHaveBeenCalledTimes(1)
  })
})
