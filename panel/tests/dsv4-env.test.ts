import { describe, expect, it } from 'vitest'
import { dsv4EnvFromConfig, resolveEffectiveModelFamily } from '../src/shared/dsv4Env'

describe('resolveEffectiveModelFamily', () => {
  it('gives an explicit non-Auto family precedence over autodetection', () => {
    expect(resolveEffectiveModelFamily('deepseek_v4', 'qwen3_5')).toBe('deepseek_v4')
    expect(resolveEffectiveModelFamily('qwen3_5', 'deepseek-v4')).toBe('qwen3_5')
  })

  it('retains autodetection for Auto, empty, and missing overrides', () => {
    expect(resolveEffectiveModelFamily('auto', 'deepseek-v4')).toBe('deepseek-v4')
    expect(resolveEffectiveModelFamily('', 'deepseek-v4')).toBe('deepseek-v4')
    expect(resolveEffectiveModelFamily(undefined, 'deepseek-v4')).toBe('deepseek-v4')
  })
})

describe('dsv4EnvFromConfig', () => {
  it('returns empty object for null/undefined config', () => {
    expect(dsv4EnvFromConfig(null)).toEqual({})
    expect(dsv4EnvFromConfig(undefined)).toEqual({})
  })

  it('returns empty object when no DSV4 fields set', () => {
    expect(dsv4EnvFromConfig({})).toEqual({})
    expect(dsv4EnvFromConfig({ host: 'x', port: 1 })).toEqual({})
  })

  it('keeps product cache policy out of the DSV4 env helper', () => {
    expect(dsv4EnvFromConfig({}, { dsv4Active: true })).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_ACTIVATION_QAT: '0',
    })
  })

  it('leaves standard cache controls CLI-owned and an unstamped pool codec model-owned', () => {
    expect(dsv4EnvFromConfig({ dsv4PrefixCache: true, dsv4PoolQuant: true } as any, { dsv4Active: true })).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_ACTIVATION_QAT: '0',
    })
  })

  it('emits activation QAT off by default and on only for an explicit DSV4 setting', () => {
    expect(dsv4EnvFromConfig({}, { dsv4Active: true })).toMatchObject({
      DSV4_ACTIVATION_QAT: '0',
    })
    expect(dsv4EnvFromConfig({ dsv4ActivationQat: true }, { dsv4Active: true })).toMatchObject({
      DSV4_ACTIVATION_QAT: '1',
    })
  })

  it('emits the explicit bundle pool codec default when true', () => {
    expect(dsv4EnvFromConfig({}, {
      dsv4Active: true,
      dsv4PoolQuantDefault: true,
    })).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_ACTIVATION_QAT: '0',
      DSV4_POOL_QUANT: '1',
    })
  })

  it('emits the explicit bundle pool codec default when false', () => {
    expect(dsv4EnvFromConfig({}, {
      dsv4Active: true,
      dsv4PoolQuantDefault: false,
    })).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_ACTIVATION_QAT: '0',
      DSV4_POOL_QUANT: '0',
    })
  })

  it('does not leak stale DSV4 cache fields into non-DSV4 launches', () => {
    expect(dsv4EnvFromConfig({ dsv4PrefixCache: true, dsv4PoolQuant: true, dsv4ActivationQat: true } as any, { dsv4Active: false })).toEqual({})
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

  it('keeps legacy behavior fields inert under the fixed product envelope', () => {
    const env = dsv4EnvFromConfig({
      dsv4RawMax: true,
      dsv4ForceDirect: true,
    }, { dsv4Active: true })
    expect(env).toEqual({
      DSV4_LONG_CTX: '1',
      DSV4_ACTIVATION_QAT: '0',
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
    expect(source).toContain("import { dsv4EnvFromConfig, resolveEffectiveModelFamily } from '../shared/dsv4Env'")

    // Called and merged into spawnEnv (each emitted env var assigned)
    expect(source).toContain('const dsv4Env = dsv4EnvFromConfig(config as any, {')
    expect(source).toContain('resolveEffectiveModelFamily(config.modelFamily, freshDetectedFamily)')
    expect(source).toContain('dsv4Active: freshDsv4Active')
    expect(source).toContain('dsv4PoolQuantDefault: freshDetectedConfig?.dsv4PoolQuantDefault')
    expect(source).toContain('delete spawnEnv.DSV4_LONG_CTX')
    expect(source).toContain('delete spawnEnv.DSV4_POOL_QUANT')
    expect(source).toContain('delete spawnEnv.DSV4_ACTIVATION_QAT')
    expect(source).toContain('spawnEnv[key] = value')
  })

  it('uses the effective family for DSV4 defaults and launch gating', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const sessionsPath = path.resolve(__dirname, '../src/main/sessions.ts')
    const source = fs.readFileSync(sessionsPath, 'utf8')

    const familyDefaultsStart = source.indexOf('function applyFamilyStartupDefaults')
    const familyDefaultsEnd = source.indexOf('const ADDITIONAL_ARG_VALUE_FLAGS', familyDefaultsStart)
    const familyDefaultsBlock = source.slice(familyDefaultsStart, familyDefaultsEnd)
    expect(familyDefaultsBlock).toContain('resolveEffectiveModelFamily(config.modelFamily, detectedFamily)')
    expect(familyDefaultsBlock).toContain("effectiveFamily === 'deepseek-v4'")

    const cacheDefaultsStart = source.indexOf('function applyMissingCacheStackStartupDefaults')
    const cacheDefaultsEnd = source.indexOf('function isZayaCacheStackMigrationTarget', cacheDefaultsStart)
    const cacheDefaultsBlock = source.slice(cacheDefaultsStart, cacheDefaultsEnd)
    expect(cacheDefaultsBlock).toContain('resolveEffectiveModelFamily(config.modelFamily, detectedFamily)')
    expect(cacheDefaultsBlock).toContain("const dsv4Active = effectiveFamily === 'deepseek-v4'")

    const buildArgsStart = source.indexOf('buildArgs(config: ServerConfig)')
    const buildArgsBlock = source.slice(buildArgsStart)
    expect(buildArgsBlock).toContain('resolveEffectiveModelFamily(config.modelFamily, detectedFamily)')
    expect(buildArgsBlock).toContain("const dsv4Active = effectiveFamily === 'deepseek-v4'")
  })

  it('marks DSV4 cache controls as restart-required session config', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const sessionsPath = path.resolve(__dirname, '../src/main/sessions.ts')
    const source = fs.readFileSync(sessionsPath, 'utf8')

    expect(source).toContain("'dsv4PrefixCache'")
    expect(source).toContain("'dsv4PoolQuant'")
    expect(source).toContain("'dsv4ActivationQat'")
    expect(source).not.toContain("'dsv4RawMax'")
    expect(source).not.toContain("'dsv4ForceDirect'")
  })

  it('logs the effective DSV4 native cache env knobs once in the engine child probe', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const sessionsPath = path.resolve(__dirname, '../src/main/sessions.ts')
    const source = fs.readFileSync(sessionsPath, 'utf8')
    const probeStart = source.indexOf('const scrubbedEnvProbeKeys = [')
    const probeEnd = source.indexOf(']', probeStart)
    const probeBlock = source.slice(probeStart, probeEnd)

    expect(probeBlock).toContain("'DSV4_LONG_CTX'")
    expect(probeBlock).toContain("'DSV4_POOL_QUANT'")
    expect(probeBlock).toContain("'DSV4_ACTIVATION_QAT'")
    expect(probeBlock).toContain("'VMLX_DSV4_ENABLE_PREFIX_CACHE'")
    expect((probeBlock.match(/DSV4_POOL_QUANT/g) ?? [])).toHaveLength(1)
  })

  it('scrubs inherited MCP policy env exactly once per key before applying session policy', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const sessionsPath = path.resolve(__dirname, '../src/main/sessions.ts')
    const source = fs.readFileSync(sessionsPath, 'utf8')

    for (const key of [
      'VLLM_MLX_MCP_CONFIG',
      'VLLM_MLX_MCP_ENABLED_SERVERS',
      'VLLM_MLX_MCP_DISABLED_SERVERS',
      'VLLM_MLX_MCP_ENABLED_TOOLS',
      'VLLM_MLX_MCP_DISABLED_TOOLS',
    ]) {
      expect((source.match(new RegExp(`delete spawnEnv\\.${key}`, 'g')) ?? [])).toHaveLength(1)
    }
  })
})

describe('DSV4 runtime controls in SessionConfigForm', () => {
  it('keeps the legacy cache mirror inert without inventing a pool-codec default', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const formPath = path.resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx')
    const source = fs.readFileSync(formPath, 'utf8')

    expect(source).toContain('dsv4PoolQuant?: boolean')
    expect(source).toContain('dsv4PrefixCache?: boolean')
    expect(source).toContain('dsv4ActivationQat?: boolean')
    expect(source).toContain('dsv4ActivationQat: false')
    expect(source).not.toContain('dsv4PoolQuant: false')
    expect(source).toContain('dsv4PrefixCache: false')
    expect(source).not.toContain('dsv4FinalizerTokens')
  })

  it('shows a DSV4-only restart-required activation-QAT toggle with honest scope', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const formPath = path.resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx')
    const source = fs.readFileSync(formPath, 'utf8')

    expect(source).toContain('resolveEffectiveModelFamily(config.modelFamily, normalizedDetectedFamily)')
    expect(source).toContain("const dsv4Active = normalizedEffectiveFamily === 'deepseek-v4'")
    expect(source).toContain('{dsv4Active && (')
    expect(source).toContain("label={t('sessions.config.dsv4ActivationQat')}")
    expect(source).toContain("checked={config.dsv4ActivationQat === true}")
    expect(source).toContain("onChange={v => onChange('dsv4ActivationQat', v)}")
    const enLocale = fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8')
    expect(enLocale).toContain('"dsv4ActivationQat": "DSV4 Activation QAT"')
    expect(enLocale).toContain('E4M3 round-trips for attention KV and compressed pools')
    expect(enLocale).toContain('Hadamard-128 + FP4 E2M1 indexer round-trips')
    expect(enLocale).toContain('FP32 compressor staging remains enabled')
    expect(enLocale).toContain('Default Off')
  })

  it('renders standard DSV4 cache controls without legacy behavior toggles', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const formPath = path.resolve(__dirname, '../src/renderer/src/components/sessions/SessionConfigForm.tsx')
    const source = fs.readFileSync(formPath, 'utf8')

    expect(source).toContain('const effectivePrefixCacheEnabled = config.enablePrefixCache')
    expect(source).not.toContain('DSV4 Raw Max Thinking')
    expect(source).not.toContain("onChange={v => onChange('dsv4RawMax', v)}")
    expect(source).not.toContain('DSV4 Finalizer Tokens')
    expect(source).not.toContain("onChange={v => onChange('dsv4FinalizerTokens', v)}")
    expect(source).not.toContain('DSV4 Force Direct Rail')
    expect(source).not.toContain("onChange={v => onChange('dsv4ForceDirect', v)}")
    expect(source).not.toContain('DSV4 Native Composite Prefix Cache')
    expect(source).not.toContain('cacheControlUpdatesForDsv4CompositeToggle')
    expect(source).not.toContain('applyDsv4CompositeCacheToggle')
    expect(source).not.toContain('DSV4 CSA/HCA Pool Codec')
    expect(source).not.toContain('cacheControlUpdatesForDsv4PoolQuantToggle')
    expect(source).not.toContain('applyDsv4PoolQuantToggle')
    expect(source).not.toContain('DSV4 Native Cache')
    expect(source).not.toContain('DSV4 Composite Prefix Cache')
    expect(source).not.toContain('DSV4 Pool Quantization')
    expect(source).not.toContain('DSV4 Flash composite prefix cache is disabled')
    const enLocale = fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8')
    expect(source).toContain("text={t('sessions.config.dsv4BatchPathNote')}")
    expect(enLocale).toContain('Prefix reuse defaults On')
    expect(enLocale).toContain('Block Disk Cache (SSD / L2) defaults On')
    expect(enLocale).toContain('CSA/HCA pool codec remains bundle-derived')
    expect(source).toContain("<CheckField label={t('sessions.config.enablePrefixCache')}")
    expect(source).not.toContain("label={t('sessions.config.pagedKVCache')}")
    expect(source).toContain('cacheControlUpdatesForBlockDiskToggle(v, cacheControlState)')
    expect(source).toContain("label={t('sessions.cache.blockDiskCache')}")
    expect(enLocale).toContain('require fixed 256-token blocks')
    expect(source).toContain('disabled={dsv4Active}')
    expect(source).toContain("t('sessions.config.storedQuantNativeTyped')")
    expect(enLocale).toContain('Native typed codec (bundle-derived)')
    // copy lives in the locale catalog now that the form is translated
    expect(
      fs.readFileSync('src/renderer/src/i18n/locales/en.json', 'utf8'),
    ).toContain('Native CSA/HCA Pool Codec')
    expect(source).toContain("config.dsv4PoolQuant === true")
    expect(source).toContain("t('sessions.config.poolQuantOnBundle')")
    expect(source).toContain("t('sessions.config.poolQuantOffBundle')")
    expect(enLocale).toContain('ON (BUNDLE)')
    expect(enLocale).toContain('OFF (BUNDLE)')
    expect(source).toContain("{!dsv4Active && <PerformanceHint text={t('sessions.config.continuousBatchingHint')} />}")
    expect(source).toContain('{showCachingHelp && <Modal')
    expect(source).not.toContain("disabled={!dsv4CompositeCacheOptIn}")
    expect(source).not.toContain("onChange={v => onChange('dsv4PoolQuant', dsv4CompositeCacheOptIn && v)}")
    expect(source).not.toContain("onChange={() => onChange('dsv4PoolQuant', false)}")
    expect(source).not.toContain('DSV4 Raw Max Thinking')
    expect(source).not.toContain('DSV4 Force Direct Rail')
    expect(source).not.toContain('DSV4 Finalizer Tokens')
  })

  it('renders the live DSV4 pool codec from health separately from generic TurboQuant', () => {
    const fs = require('node:fs')
    const path = require('node:path')
    const performancePath = path.resolve(__dirname, '../src/renderer/src/components/sessions/PerformancePanel.tsx')
    const source = fs.readFileSync(performancePath, 'utf8')

    expect(source).toContain('pool_quant?: {')
    expect(source).toContain('health.native_cache?.pool_quant')
    // Label and status copy render through i18n now; the English lives in the catalog.
    expect(source).toContain("label={t('sessions.performance.dsv4PoolQuant')}")
    expect(source).toContain("health.native_cache.pool_quant.enabled ? t('sessions.cache.statusEnabled') : t('sessions.cache.statusDisabled')")
    const catalog = fs.readFileSync(
      path.resolve(__dirname, '../src/renderer/src/i18n/locales/en.json'),
      'utf8',
    )
    expect(catalog).toContain('DSV4 Pool Quant')
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
