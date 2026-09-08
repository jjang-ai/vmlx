import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'

// Execute the owning production method; filesystem/model default helpers are
// outside this lifecycle test. Assert no mutation before an active-session error.
const source = readFileSync('src/main/sessions.ts', 'utf8')
const ast = ts.createSourceFile('sessions.ts', source, ts.ScriptTarget.Latest, true)
let method: string | undefined
function visit(node: ts.Node) {
  if (ts.isMethodDeclaration(node) && node.name.getText(ast) === '_createSessionInner') {
    method = node.getText(ast)
  }
  ts.forEachChild(node, visit)
}
visit(ast)
if (!method) throw new Error('Production creation method missing')
const js = ts.transpileModule(`class Subject { ${method} }`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022 },
}).outputText

function fixture(status: string, managed?: Record<string, unknown>) {
  let row = {
    id: 'existing', modelPath: '/bundle', host: '127.0.0.1', port: 8121,
    status, type: 'local', config: JSON.stringify({ port: 8121, customSaved: 42 }),
  }
  const before = structuredClone(row)
  const update = vi.fn((_id: string, changes: Partial<typeof row>) => { row = { ...row, ...changes } })
  const env: Record<string, unknown> = {
    db: { getSessionByModelPath: () => row, getSessions: () => [row],
      updateSession: update, getSession: () => row },
    normalizePath: (p: string) => p,
  }
  for (const name of method!.matchAll(/\b(apply\w+|liftStaleFlatCacheIndex|normalizeCacheStackMutualExclusion|markCacheStackStartupDefaultsCurrent)\(/g)) {
    env[name[1]] = () => {}
  }
  const Subject = new Function(...Object.keys(env), `${js}; return Subject`)(...Object.values(env))
  const subject = new Subject()
  subject.processes = new Map(managed ? [['existing', managed]] : [])
  return { before, update, row: () => row,
    create: () => subject._createSessionInner('/bundle', { port: 8122, servedModelName: 'changed' }) }
}

describe('creating an already active local bundle preserves its runtime identity', () => {
  it.each(['running', 'loading', 'standby'])('rejects %s before changing any saved field', async status => {
    const f = fixture(status)
    await expect(f.create()).rejects.toThrow(/active session/i)
    expect(f.update).not.toHaveBeenCalled()
    expect(f.row()).toEqual(f.before)
  })

  it.each([{ process: {} }, { adoptedPid: 123 }])('protects a managed process even if the database says stopped: %j', async managed => {
    const f = fixture('stopped', managed)
    await expect(f.create()).rejects.toThrow(/active session/i)
    expect(f.update).not.toHaveBeenCalled()
    expect(f.row()).toEqual(f.before)
  })

  it.each([undefined, { process: null, adoptedPid: null, exitCode: 1 }])('still merges a stopped session, including after an exited process: %j', async managed => {
    const f = fixture('stopped', managed)
    const result = await f.create()
    expect(result.port).toBe(8122)
    expect(JSON.parse(result.config)).toMatchObject({ port: 8122, customSaved: 42, servedModelName: 'changed' })
    expect(f.update).toHaveBeenCalledTimes(1)
  })
})
