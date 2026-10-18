import { describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'
import ts from 'typescript'

// Execute the actual callback with controlled IPC completion, rather than
// duplicating the selection algorithm in the test.
function selectionCallback(get: any, update: any, dispatch: any) {
  const src = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'App.tsx'), 'utf8')
  const body = src.split('const handleSessionChange = useCallback(async (sessionId: string) => {')[1].split('\n  }, [')[0]
  const js = ts.transpile(`async function select(sessionId: string) {${body}\n}`, { target: ts.ScriptTarget.ES2020 })
  return new Function('state', 'sessions', 'window', 'dispatch', 't', `${js}; return select`)(
    { activeChatId: 'chat' }, [], { api: { sessions: { get }, chat: { update } } }, dispatch, (key: string) => key,
  )
}

describe('new remote session binding', () => {
  it('waits for the authoritative session and saved identity before navigation', async () => {
    const get = vi.fn().mockResolvedValue({ id: 'remote', modelPath: 'remote://model@endpoint' })
    let saved!: () => void
    const update = vi.fn(() => new Promise<void>(resolve => { saved = resolve }))
    const dispatch = vi.fn()
    const pending = selectionCallback(get, update, dispatch)('remote')
    await vi.waitFor(() => expect(update).toHaveBeenCalledWith('chat', {
      modelId: 'remote://model@endpoint', modelPath: 'remote://model@endpoint',
    }))
    expect(dispatch).not.toHaveBeenCalled()
    saved()
    await pending
    expect(dispatch).toHaveBeenCalledWith({ type: 'OPEN_CHAT', chatId: 'chat', sessionId: 'remote' })
  })

  it('does not switch the endpoint after a failed save', async () => {
    const dispatch = vi.fn()
    const select = selectionCallback(vi.fn().mockResolvedValue({ modelPath: 'remote://model@endpoint' }),
      vi.fn().mockRejectedValue(new Error('save failed')), dispatch)
    await expect(select('remote')).rejects.toThrow('save failed')
    expect(dispatch).not.toHaveBeenCalled()
  })

  it('does not navigate when the selected session no longer exists', async () => {
    const update = vi.fn(), dispatch = vi.fn()
    await expect(selectionCallback(vi.fn().mockResolvedValue(null), update, dispatch)('remote')).rejects.toThrow()
    expect(update).not.toHaveBeenCalled()
    expect(dispatch).not.toHaveBeenCalled()
  })
})

describe('new chat targets a running session over a stopped pinned one', () => {
  it('App.tsx prefers the running session when the pinned one is not running', () => {
    const src = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'App.tsx'), 'utf8')
    expect(src).toContain("const explicitUsable = explicit && (explicit.status === 'running' || explicit.status === 'loading' || !running)")
    expect(src).toContain("const target = (explicitUsable ? explicit : null) || running || explicit || sessions[0]")
  })
})
