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

  it('does not enable the rejected DSV4 pool quant codec even if old config asks for it', () => {
    expect(dsv4EnvFromConfig({ dsv4PoolQuant: true }, { dsv4Active: true })).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_POOL_QUANT: '0',
    })
  })

  it('keeps DSV4 composite prefix cache explicit and opt-in when enabled', () => {
    expect(dsv4EnvFromConfig({ dsv4PrefixCache: true }, { dsv4Active: true })).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_POOL_QUANT: '0',
      VMLX_DSV4_ENABLE_PREFIX_CACHE: '1',
    })
  })

  it('does not leak DSV4 pool quant env into non-DSV4 launches', () => {
    expect(dsv4EnvFromConfig({ dsv4PoolQuant: true }, { dsv4Active: false })).toEqual({})
  })

  it('does not gate raw max through an env opt-in anymore', () => {
    expect(dsv4EnvFromConfig({ dsv4RawMax: true })).toEqual({})
  })

  it('does NOT set VMLX_DSV4_RAW_MAX when dsv4RawMax is false/missing', () => {
    expect(dsv4EnvFromConfig({ dsv4RawMax: false })).toEqual({})
    expect(dsv4EnvFromConfig({ dsv4RawMax: undefined })).toEqual({})
  })

  it('does not map legacy dsv4FinalizerTokens into decode-forcing env', () => {
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: 4096 })).toEqual({})
  })

  it('does NOT set finalizer when value is 0/negative/non-finite', () => {
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: 0 })).toEqual({})
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: -1 })).toEqual({})
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: NaN })).toEqual({})
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: Infinity })).toEqual({})
  })

  it('ignores legacy fractional finalizer tokens', () => {
    expect(dsv4EnvFromConfig({ dsv4FinalizerTokens: 4096.7 })).toEqual({})
  })

  it('does not map legacy dsv4ForceDirect into decode-forcing env', () => {
    expect(dsv4EnvFromConfig({ dsv4ForceDirect: true })).toEqual({})
  })

  it('combines legacy no-op fields without emitting decode-forcing env', () => {
    const env = dsv4EnvFromConfig({
      dsv4RawMax: true,
      dsv4ForceDirect: false,
    })
    expect(env).toEqual({})
  })

  it('rawMax and forceDirect legacy config do not emit decode-forcing env', () => {
    const env = dsv4EnvFromConfig({
      dsv4RawMax: true,
      dsv4ForceDirect: true,
    })
    expect(env).toEqual({})
  })

  it('combines supported DSV4 fields when set together', () => {
    const env = dsv4EnvFromConfig({
      dsv4PrefixCache: true,
      dsv4PoolQuant: true,
      dsv4RawMax: true,
      dsv4ForceDirect: true,
    }, { dsv4Active: true })
    expect(env).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_POOL_QUANT: '0',
      VMLX_DSV4_ENABLE_PREFIX_CACHE: '1',
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

  it('marks DSV4 cache controls as restart-required session config', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const sessionsPath = path.resolve(__dirname, '../src/main/sessions.ts')
    const source = fs.readFileSync(sessionsPath, 'utf8')

    expect(source).toContain("'dsv4PrefixCache'")
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
    expect(source).toContain('dsv4PrefixCache?: boolean')
    expect(source).toContain('dsv4PoolQuant: false')
    expect(source).toContain('dsv4PrefixCache: false')
    expect(source).not.toContain('dsv4FinalizerTokens')
  })

  it('renders only supported DSV4 runtime controls in performance settings', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const formPath = path.resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx')
    const source = fs.readFileSync(formPath, 'utf8')

    expect(source).toContain('{dsv4Active && (')
    expect(source).not.toContain('DSV4 Raw Max Thinking')
    expect(source).not.toContain("onChange={v => onChange('dsv4RawMax', v)}")
    expect(source).not.toContain('DSV4 Finalizer Tokens')
    expect(source).not.toContain("onChange={v => onChange('dsv4FinalizerTokens', v)}")
    expect(source).not.toContain('DSV4 Force Direct Rail')
    expect(source).not.toContain("onChange={v => onChange('dsv4ForceDirect', v)}")
    expect(source).toContain('DSV4 Composite Prefix Cache')
    expect(source).toContain("onChange('dsv4PrefixCache', v)")
    expect(source).toContain('DSV4 Pool Quantization')
    expect(source).toContain('dequantizes and concatenates the full CSA/HCA pool during decode')
    expect(source).toContain("onChange={() => onChange('dsv4PoolQuant', false)}")
    expect(source).toContain('disabled')
    expect(source).not.toContain('DSV4 Raw Max Thinking')
    expect(source).not.toContain('DSV4 Force Direct Rail')
    expect(source).not.toContain('DSV4 Finalizer Tokens')
  })

  it('does not document hidden DSV4 finalizer behavior in the settings UI', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const formPath = path.resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx')
    const source = fs.readFileSync(formPath, 'utf8')

    expect(source).not.toContain('0 uses the app/API default of 2048 extra visible tokens')
    expect(source).not.toContain('Default 0 keeps the request max_tokens contract exact')
  })
})

describe('DSV4 runtime controls in chat request wiring', () => {
  it('does not forward legacy DSV4 finalizer token settings per request', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const chatPath = path.resolve(__dirname, '../src/main/ipc/chat.ts')
    const source = fs.readFileSync(chatPath, 'utf8')

    expect(source).not.toContain('let sessionDsv4FinalizerTokens')
    expect(source).not.toContain('sessionDsv4FinalizerTokens = sessionConfig.dsv4FinalizerTokens')
    expect(source).not.toContain('dsv4_finalizer_tokens')
  })
})
