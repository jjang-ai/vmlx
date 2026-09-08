import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'

// Execute the production handler, not a rewritten approximation of its state
// machine. Delayed IPC promises expose the interval before detection completes.
const source = readFileSync('src/renderer/src/components/sessions/CreateSession.tsx', 'utf8')
const ast = ts.createSourceFile('create.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let expression: string | undefined
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name)
    && node.name.text === 'handleReset') expression = node.initializer?.getText(ast)
  ts.forEachChild(node, visit)
}
visit(ast)
if (!expression) throw new Error('Production handleReset missing')
const js = ts.transpileModule(`const handler = ${expression}`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText

function fixture(initialPending = false) {
  let resolve!: (value: unknown) => void
  let reject!: (error: Error) => void
  const detection = new Promise((yes, no) => { resolve = yes; reject = no })
  const state = { pending: initialPending, config: undefined as unknown, writes: [] as boolean[] }
  const epoch = { current: 0 }
  const mounted = { current: true }
  const environment: Record<string, unknown> = {
    modelDefaultsRequestRef: epoch, mountedRef: mounted,
    RESET_CONFIG: { timeout: 300, blockDiskCacheMaxGb: undefined },
    config: { port: 8110 }, selectedModel: '/test/model',
    window: { api: { models: {
      detectConfig: () => detection,
      getGenerationDefaults: () => Promise.resolve(null),
    } } },
    setDefaultsPending: (value: boolean) => { state.pending = value; state.writes.push(value) },
    setConfig: (value: unknown) => { state.config = value },
    applyBundleGenerationDefaultsToSessionConfig: (value: unknown) => value,
    usesExactTypedPromptDiskCache: () => false,
    DSV4_PAGED_CACHE_BLOCK_SIZE: 64, DSV4_MAX_CACHE_BLOCKS: 4097,
  }
  for (const name of expression!.matchAll(/\b(setDetected\w+)\(/g)) environment[name[1]] = () => {}
  const reset = new Function(...Object.keys(environment), `${js}\nreturn handler`)(...Object.values(environment)) as () => Promise<void>
  return { state, epoch, mounted, reset, resolve, reject }
}

describe('creation Reset owns the model-defaults pending interval', () => {
  it('blocks Launch while Reset is resolving and releases it after applying defaults', async () => {
    const f = fixture()
    const operation = f.reset()
    expect(f.state.pending).toBe(true)
    f.resolve({ family: 'llama3' })
    await operation
    expect(f.state.pending).toBe(false)
    expect(f.state.config).toMatchObject({ port: 8110, timeout: 300 })
  })

  it('releases a pending selection superseded by Reset rather than stranding Launch', async () => {
    const f = fixture(true)
    const operation = f.reset()
    f.resolve({ family: 'llama3' })
    await operation
    expect(f.state.pending).toBe(false)
  })

  it('does not clear a newer detection owner or overwrite its config', async () => {
    const f = fixture(true)
    const operation = f.reset()
    f.epoch.current += 1
    f.resolve({ family: 'llama3' })
    await operation
    expect(f.state.pending).toBe(true)
    expect(f.state.config).toBeUndefined()
    expect(f.state.writes).not.toContain(false)
  })

  it('releases its pending state when detector failure falls back to default config', async () => {
    const f = fixture(true)
    const operation = f.reset()
    f.reject(new Error('detector unavailable'))
    await operation
    expect(f.state.pending).toBe(false)
    expect(f.state.config).toMatchObject({ port: 8110 })
  })

  it('does not apply or release state after unmount', async () => {
    const f = fixture(true)
    const operation = f.reset()
    f.mounted.current = false
    f.resolve({ family: 'llama3' })
    await operation
    expect(f.state.config).toBeUndefined()
    expect(f.state.writes).not.toContain(false)
  })
})
