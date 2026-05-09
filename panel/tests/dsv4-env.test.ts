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

  it('maps dsv4RawMax=true to VMLX_DSV4_RAW_MAX=1', () => {
    expect(dsv4EnvFromConfig({ dsv4RawMax: true })).toEqual({
      VMLX_DSV4_RAW_MAX: '1',
    })
  })

  it('does NOT set VMLX_DSV4_RAW_MAX when dsv4RawMax is false/missing', () => {
    expect(dsv4EnvFromConfig({ dsv4RawMax: false })).toEqual({})
    expect(dsv4EnvFromConfig({ dsv4RawMax: undefined })).toEqual({})
  })

  it('maps dsv4FinalizerTokens to VMLX_DSV4_FINALIZER_TOKENS', () => {
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: 4096 })).toEqual({
      VMLX_DSV4_FINALIZER_TOKENS: '4096',
    })
  })

  it('does NOT set finalizer when value is 0/negative/non-finite', () => {
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: 0 })).toEqual({})
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: -1 })).toEqual({})
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: NaN })).toEqual({})
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: Infinity })).toEqual({})
  })

  it('floors and clamps finalizer tokens to a sane integer', () => {
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: 4096.7 })).toEqual({
      VMLX_DSV4_FINALIZER_TOKENS: '4096',
    })
  })

  it('maps dsv4ForceDirect=true to VMLX_DSV4_FORCE_DIRECT_RAIL=1', () => {
    expect(dsv4EnvFromConfig({ dsv4ForceDirect: true })).toEqual({
      VMLX_DSV4_FORCE_DIRECT_RAIL: '1',
    })
  })

  it('combines all DSV4 fields when set together', () => {
    const env = dsv4EnvFromConfig({
      dsv4RawMax: true,
      dsv4FinalizerTokens: 4096,
      dsv4ForceDirect: false,
    })
    expect(env).toEqual({
      VMLX_DSV4_RAW_MAX: '1',
      VMLX_DSV4_FINALIZER_TOKENS: '4096',
    })
  })

  it('rawMax + forceDirect together is allowed (engine reconciles)', () => {
    // Engine: forceDirect short-circuits before rawMax matters. Helper just
    // emits both vars — engine logic decides precedence.
    const env = dsv4EnvFromConfig({
      dsv4RawMax: true,
      dsv4ForceDirect: true,
    })
    expect(env).toEqual({
      VMLX_DSV4_RAW_MAX: '1',
      VMLX_DSV4_FORCE_DIRECT_RAIL: '1',
    })
  })

  it('ignores unrelated config fields', () => {
    const env = dsv4EnvFromConfig({
      dsv4RawMax: true,
      modelPath: '/some/path',
      port: 8080,
      host: '0.0.0.0',
    } as any)
    expect(env).toEqual({ VMLX_DSV4_RAW_MAX: '1' })
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
    expect(source).toContain('const dsv4Env = dsv4EnvFromConfig(config as any)')
    expect(source).toContain('spawnEnv[key] = value')
  })
})
