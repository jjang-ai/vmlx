import { describe, expect, it } from 'vitest'
import { dsv4EnvFromConfig } from '../src/shared/dsv4Env'

describe('dsv4EnvFromConfig', () => {
  it('returns empty object for null/undefined config', () => {
    expect(dsv4EnvFromConfig(null)).toEqual({})
    expect(dsv4EnvFromConfig(undefined)).toEqual({})
  })

  it('returns empty object when no DSV4 fields set', () => {
    expect(dsv4EnvFromConfig({})).toEqual({})
    expect(dsv4EnvFromConfig({ host: 'x', port: 1 })).toEqual({})
  })

  it('pins production DSV4 runtime env when the detected family is DSV4', () => {
    expect(dsv4EnvFromConfig({}, { dsv4Active: true })).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_POOL_QUANT: '0',
    })
  })

  it('keeps DSV4 pool quant explicit and opt-in when enabled', () => {
    expect(dsv4EnvFromConfig({ dsv4PoolQuant: true }, { dsv4Active: true })).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_POOL_QUANT: '1',
    })
  })

  it('does not leak DSV4 pool quant env into non-DSV4 launches', () => {
    expect(dsv4EnvFromConfig({ dsv4PoolQuant: true }, { dsv4Active: false })).toEqual({})
  })

  it('combines supported DSV4 fields when set together', () => {
    const env = dsv4EnvFromConfig({
      dsv4PoolQuant: true,
    }, { dsv4Active: true })
    expect(env).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_POOL_QUANT: '1',
    })
  })

  it('ignores unrelated config fields', () => {
    const env = dsv4EnvFromConfig({
      modelPath: '/some/path',
      port: 8080,
      host: '0.0.0.0',
    } as any)
    expect(env).toEqual({})
  })
})


describe('dsv4EnvFromConfig wired into sessions.ts spawnEnv', () => {
  it('main process imports the helper and merges into spawnEnv', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const sessionsPath = path.resolve(__dirname, '../src/main/sessions.ts')
    const source = fs.readFileSync(sessionsPath, 'utf8')

    // Import statement present
    expect(source).toContain("import { dsv4EnvFromConfig } from '../shared/dsv4Env'")

    // Called and merged into spawnEnv (each emitted env var assigned)
    expect(source).toContain('const dsv4Env = dsv4EnvFromConfig(config as any, {')
    expect(source).toContain("dsv4Active: freshDetectedFamily === 'deepseek-v4'")
    expect(source).toContain('spawnEnv[key] = value')
  })

  it('marks DSV4 pool quant as restart-required session config', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const sessionsPath = path.resolve(__dirname, '../src/main/sessions.ts')
    const source = fs.readFileSync(sessionsPath, 'utf8')

    expect(source).toContain("'dsv4PoolQuant'")
    expect(source).not.toContain("'dsv4RawMax'")
    expect(source).not.toContain("'dsv4ForceDirect'")
  })
})

describe('DSV4 runtime controls in SessionConfigForm', () => {
  it('declares typed session config fields with safe defaults', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const formPath = path.resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx')
    const source = fs.readFileSync(formPath, 'utf8')

    expect(source).toContain('dsv4PoolQuant?: boolean')
    expect(source).toContain('dsv4PoolQuant: false')
    expect(source).not.toContain('dsv4FinalizerTokens')
  })

  it('renders only DSV4 pool quant control in performance settings', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const formPath = path.resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx')
    const source = fs.readFileSync(formPath, 'utf8')

    expect(source).toContain('{dsv4Active && (')
    expect(source).toContain('DSV4 Pool Quantization')
    expect(source).toContain("onChange={v => onChange('dsv4PoolQuant', v)}")
    expect(source).not.toContain('DSV4 Raw Max Thinking')
    expect(source).not.toContain('DSV4 Force Direct Rail')
    expect(source).not.toContain('DSV4 Finalizer Tokens')
  })
})
