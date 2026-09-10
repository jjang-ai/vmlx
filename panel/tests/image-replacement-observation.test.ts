import { readFileSync } from 'fs'
import { join } from 'path'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'

// Execute the actual callback with deferred IPC, rather than a second model of
// its state transitions. React setters are recorded; no engine is started.
function harness() {
  const source = readFileSync(join(process.cwd(), 'src/renderer/src/components/image/ImageTab.tsx'), 'utf8')
  const start = source.indexOf('useCallback(async (modelId:') + 'useCallback('.length
  const end = source.indexOf('}, [serverStatus,', start) + 1
  const expression = source.slice(start, end)
  let resolve!: (value: any) => void
  const pending = new Promise(r => { resolve = r })
  const ref = { current: 'old-session' as string | null }
  const caps = { mode: 'generate' }
  const ctx: Record<string, any> = {
    selectedModel: 'old-folder', selectedModelDisplayName: 'old name',
    serverStatus: 'running', quantize: 6, sessionMode: 'generate',
    settings: { steps: 4 }, settingsOwner: 'old-session',
    serverSessionIdRef: ref, serverPort: 8018, capabilities: caps,
    settingsRevision: { current: 1 }, settingsEdited: { current: false },
    healthRevision: { current: 9 }, serverSettingsRef: { current: undefined },
    hydrateSettings: vi.fn().mockResolvedValue(undefined),
    describeStartError: (r: any) => r.error, t: (s: string) => s,
    getImageModel: () => ({ id: 'z-image-turbo', quantizeOptions: [6] }),
    resolveImageModelFromDirectoryName: () => null,
    getDefaultSteps: () => 4, getDefaultGuidance: () => 1,
    window: { api: { image: {
      startServer: vi.fn(() => pending),
      saveRuntimeSettings: vi.fn().mockResolvedValue(undefined),
      stopServer: vi.fn(),
    } } },
  }
  for (const name of expression.match(/\bset[A-Z]\w+/g) || []) ctx[name] = vi.fn()
  ctx.setServerSessionId = vi.fn((id: string | null) => { ref.current = id })
  const js = ts.transpile('return (' + expression + ')', { target: ts.ScriptTarget.ES2022 })
  const launch = Function(...Object.keys(ctx), js)(...Object.values(ctx))
  return { ctx, resolve, launch, caps }
}

describe('image replacement observation ownership', () => {
  it('detaches old polls/events throughout deferred preflight and restores a kept owner', async () => {
    const { ctx, resolve, launch, caps } = harness()
    const work = launch('replacement', 6, 'generate')
    expect(ctx.serverSessionIdRef.current).toBeNull()
    expect(ctx.healthRevision.current).toBe(10)
    expect(ctx.setServerPort).toHaveBeenLastCalledWith(null)
    expect(ctx.setCapabilities).toHaveBeenLastCalledWith(null)
    expect(ctx.setServerStatus).toHaveBeenLastCalledWith('starting')
    expect(ctx.window.api.image.stopServer).not.toHaveBeenCalled()
    resolve({ success: false, serverKept: true, error: 'Incomplete folder' })
    await work
    expect(ctx.serverSessionIdRef.current).toBe('old-session')
    expect(ctx.setSelectedModel).toHaveBeenLastCalledWith('old-folder')
    expect(ctx.setServerPort).toHaveBeenLastCalledWith(8018)
    expect(ctx.setCapabilities).toHaveBeenLastCalledWith(caps)
    expect(ctx.setSettingsOwner).toHaveBeenLastCalledWith('old-session')
  })

  it('binds only the accepted replacement and waits for its health', async () => {
    const { ctx, resolve, launch } = harness()
    const work = launch('replacement', 6, 'generate')
    resolve({ success: true, sessionId: 'new-session', port: 8020, quantize: 6 })
    await work
    expect(ctx.serverSessionIdRef.current).toBe('new-session')
    expect(ctx.setServerPort).toHaveBeenLastCalledWith(8020)
    expect(ctx.setServerStatus).toHaveBeenLastCalledWith('starting')
    expect(ctx.hydrateSettings).toHaveBeenCalledWith('new-session', false, 2)
  })

  it('does not restore an old owner after a destructive start failure', async () => {
    const { ctx, resolve, launch } = harness()
    const work = launch('replacement', 6, 'generate')
    resolve({ success: false, serverKept: false, error: 'Start failed' })
    await work
    expect(ctx.serverSessionIdRef.current).toBeNull()
    expect(ctx.setServerPort).toHaveBeenLastCalledWith(null)
    expect(ctx.setServerStatus).toHaveBeenLastCalledWith('error')
  })

  it('leaves the old observation intact if saving its settings fails first', async () => {
    const { ctx, launch } = harness()
    ctx.settingsEdited.current = true
    ctx.window.api.image.saveRuntimeSettings.mockRejectedValue(new Error('write failed'))
    await launch('replacement', 6, 'generate')
    expect(ctx.serverSessionIdRef.current).toBe('old-session')
    expect(ctx.healthRevision.current).toBe(9)
    expect(ctx.window.api.image.startServer).not.toHaveBeenCalled()
  })
})
