import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'
import { DEFAULT_BLOCK_DISK_CACHE_PERCENT } from '../src/shared/cacheDefaults'

const directory = 'src/renderer/src/components/sessions/'
const source = readFileSync(`${directory}SessionConfigForm.tsx`, 'utf8')
const ast = ts.createSourceFile('form.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
function config(name: string, defaults?: object): Record<string, unknown> {
  const declaration = ast.statements.filter(ts.isVariableStatement)
    .flatMap(s => [...s.declarationList.declarations])
    .find(d => ts.isIdentifier(d.name) && d.name.text === name)
  if (!declaration?.initializer) throw new Error(`Missing production ${name}`)
  const js = ts.transpileModule(`return (${declaration.initializer.getText(ast)})`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  }).outputText
  return new Function('DEFAULT_CONFIG', 'DEFAULT_BLOCK_DISK_CACHE_PERCENT', js)(defaults, DEFAULT_BLOCK_DISK_CACHE_PERCENT)
}

describe('explicit session reset payload', () => {
  it('explicitly clears optional overrides, including a hidden absolute SSD ceiling', () => {
    const reset = structuredClone(config('RESET_CONFIG', config('DEFAULT_CONFIG')))
    for (const key of ['blockDiskCacheMaxGb', 'modelFamily', 'enableAutoToolChoice',
      'chatTemplate', 'imageTokenBudget', 'videoTokenBudget', 'videoMaxPixels',
      'idleTimeoutSoftMin', 'idleTimeoutHardMin', 'autoSleepEnabled',
      'distributedEnabled', 'distributedMode', 'distributedSecret', 'distributedNodes']) {
      expect(Object.hasOwn(reset, key), key).toBe(true)
      expect(reset[key], key).toBeUndefined()
    }
    expect(reset.blockDiskCacheMaxPercent).toBe(DEFAULT_BLOCK_DISK_CACHE_PERCENT)
    expect(reset.blockDiskCacheDir).toBe('')
  })

  it('covers every typed setting without changing any fresh default value', () => {
    const defaults = config('DEFAULT_CONFIG')
    const reset = config('RESET_CONFIG', defaults)
    const schema = ast.statements.find(s => ts.isInterfaceDeclaration(s) && s.name.text === 'SessionConfig') as ts.InterfaceDeclaration
    for (const field of schema.members) {
      if (field.name && ts.isIdentifier(field.name)) expect(Object.hasOwn(reset, field.name.text), field.name.text).toBe(true)
    }
    for (const [key, value] of Object.entries(defaults)) expect(reset[key], key).toEqual(value)
    expect(Object.hasOwn(defaults, 'blockDiskCacheMaxGb')).toBe(false)
  })

  it.each(['ServerSettingsDrawer.tsx', 'SessionSettings.tsx', 'CreateSession.tsx'])(
    '%s uses reset semantics only for explicit Reset', file => {
      const text = readFileSync(directory + file, 'utf8')
      const reset = text.slice(text.indexOf('const handleReset = async () =>'))
      expect(reset.split('const base = ')[1].split('\n')[0]).toContain('...RESET_CONFIG')
      expect(text).toContain('DEFAULT_CONFIG')
    },
  )
})
