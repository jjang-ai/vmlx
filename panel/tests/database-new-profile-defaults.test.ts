import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'
import { GATEWAY_SINGLE_MODEL_MODE_KEY, isGatewaySettingEnabled } from '../src/shared/gatewaySettingsKeys'

// Execute the production constructor with an observable settings store. The
// live Electron rows exercise the same initialization against actual SQLite.
const source = readFileSync('src/main/database.ts', 'utf8')
const ast = ts.createSourceFile('database.ts', source, ts.ScriptTarget.Latest, true)
const declaration = ast.statements.find(s => ts.isClassDeclaration(s) && s.name?.text === 'DatabaseManager') as ts.ClassDeclaration
const constructor = declaration.members.find(ts.isConstructorDeclaration)
if (!constructor) throw new Error('Production database constructor missing')
const js = ts.transpileModule(`class Subject {
  ${constructor.getText(ast)}
  initialize() { events.push('initialize') }
}`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText

function open(existingFiles: string[], stored?: string) {
  const settings = new Map<string, string>()
  if (stored !== undefined) settings.set(GATEWAY_SINGLE_MODEL_MODE_KEY, stored)
  const events: string[] = []
  const files = new Set(existingFiles)
  class FakeDatabase {
    constructor(path: string) { files.add(path); events.push('open') }
    prepare(sql: string) {
      expect(sql).toMatch(/INSERT OR IGNORE INTO settings/i)
      return { run(key: string, value: string) {
        events.push('seed')
        if (!settings.has(key)) settings.set(key, value)
      } }
    }
  }
  const environment = {
    app: { getPath: () => '/profile' }, join: (...parts: string[]) => parts.join('/'),
    existsSync: (path: string) => files.has(path), Database: FakeDatabase,
    GATEWAY_SINGLE_MODEL_MODE_KEY, events,
    renameSync: () => { throw new Error('Unexpected recovery') },
    unlinkSync: () => { throw new Error('Unexpected recovery') },
  }
  const Subject = new Function(...Object.keys(environment), `${js}; return Subject`)(...Object.values(environment))
  new Subject()
  return { settings, events, reopen: () => new Subject() }
}

describe('new-profile single-model default', () => {
  it('enables only a genuinely new database, after schema initialization', () => {
    const result = open([])
    expect(result.settings.get(GATEWAY_SINGLE_MODEL_MODE_KEY)).toBe('true')
    expect(result.events).toEqual(['open', 'initialize', 'seed'])
  })

  it.each([undefined, 'false', 'true', ''])('preserves existing database choice %s', stored => {
    const result = open(['/profile/chats.db'], stored)
    expect(result.settings.get(GATEWAY_SINGLE_MODEL_MODE_KEY)).toBe(stored)
    expect(result.events).not.toContain('seed')
    expect(isGatewaySettingEnabled(result.settings.get(GATEWAY_SINGLE_MODEL_MODE_KEY))).toBe(stored === 'true')
  })

  it.each(['-wal', '-shm'])('does not treat a surviving %s file as a fresh profile', suffix => {
    const result = open([`/profile/chats.db${suffix}`])
    expect(result.settings.has(GATEWAY_SINGLE_MODEL_MODE_KEY)).toBe(false)
  })

  it('does not re-enable a user choice on subsequent database opens', () => {
    const result = open([])
    result.settings.set(GATEWAY_SINGLE_MODEL_MODE_KEY, 'false')
    result.reopen()
    expect(result.settings.get(GATEWAY_SINGLE_MODEL_MODE_KEY)).toBe('false')
    expect(result.events.filter(e => e === 'seed')).toHaveLength(1)
  })

  it('does not replace a setting populated during initial schema setup', () => {
    const result = open([], 'false')
    expect(result.settings.get(GATEWAY_SINGLE_MODEL_MODE_KEY)).toBe('false')
  })
})
