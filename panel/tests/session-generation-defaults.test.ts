import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import {
  applyBundleDsv4PoolQuantToSessionConfig,
  applyBundleGenerationDefaultsToSessionConfig,
  resolveBundleGenerationDefaults,
} from '../src/shared/sessionGenerationDefaults'

describe('session generation-default hydration', () => {
  it('uses one resolver for standard and JANG sampling precedence', () => {
    expect(resolveBundleGenerationDefaults(
      {
        do_sample: false,
        temperature: 1,
        top_p: 0.95,
        top_k: 40,
        repetition_penalty: 1.1,
        max_new_tokens: 2048,
      },
      {
        chat: {
          reasoning: { default_mode: 'chat' },
          sampling_defaults: {
            temperature: 0.6,
            top_p: 0.9,
            repetition_penalty_chat: 1.05,
          },
        },
      },
      { model_type: 'qwen3' },
    )).toEqual({
      temperature: 0.6,
      topP: 0.9,
      topK: 0,
      repeatPenalty: 1.05,
      maxNewTokens: 2048,
      source: 'jang_config',
    })
  })

  it('rejects invalid output-token defaults and normalizes disabled top-k', () => {
    expect(resolveBundleGenerationDefaults(
      { max_new_tokens: -1, top_k: -1 },
      null,
      null,
    )).toEqual({
      topK: 0,
      source: 'generation_config',
    })
  })

  it('keeps DSV4 direct-chat scalar repetition precedence', () => {
    expect(resolveBundleGenerationDefaults(
      null,
      {
        chat: {
          reasoning: { default_mode: 'chat' },
          sampling_defaults: {
            repetition_penalty: 1,
            repetition_penalty_chat: 1.05,
            repetition_penalty_thinking: 1.1,
          },
        },
      },
      { model_type: 'deepseek_v4' },
    )).toMatchObject({
      repeatPenalty: 1,
      source: 'jang_config',
    })
  })

  it('keeps Laguna top-k metadata-owned', () => {
    expect(resolveBundleGenerationDefaults(
      {
        do_sample: true,
        temperature: 1,
        top_p: 1,
      },
      {
        chat: {
          sampling_defaults: {
            temperature: 1,
            top_p: 1,
          },
        },
      },
      { model_type: 'laguna' },
    )).toEqual({
      temperature: 1,
      topP: 1,
      source: 'jang_config',
    })

    expect(resolveBundleGenerationDefaults(
      { do_sample: true, top_k: 32 },
      null,
      { model_type: 'laguna' },
    )).toMatchObject({ topK: 32 })
    expect(resolveBundleGenerationDefaults(
      { do_sample: false },
      null,
      { model_type: 'laguna' },
    )).toMatchObject({ doSample: false })
  })

  it('preserves an explicit do_sample=false declaration', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { doSample: false })).toMatchObject({
      defaultDoSample: false,
      defaultSamplingDefaultsDeclared: true,
    })
  })

  it('preserves an unusually large model-derived top-k', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { topK: 151552 }).defaultTopK)
      .toBe(151552)
  })

  it('hydrates the model-derived maximum output-token default', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({}, { maxNewTokens: 32768 }).defaultMaxNewTokens)
      .toBe(32768)
  })

  it.each([4096, 3072, 16113])('does not present effective engine cap %d as a bundle declaration', cap => {
    const result = applyBundleGenerationDefaultsToSessionConfig(
      { maxTokens: 4096, defaultMaxNewTokens: 12000, unrelated: true },
      { maxNewTokens: cap, maxNewTokensFromEngine: true, temperature: 0.6 },
    )
    expect(result).toMatchObject({
      maxTokens: 4096, defaultMaxNewTokens: 0, defaultTemperature: 60, unrelated: true,
    })
  })

  it('keeps default inheritance rather than promoting an observed engine limit', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig(
      { maxTokens: 0 }, { maxNewTokens: 4096, maxNewTokensFromEngine: true },
    )).toMatchObject({ maxTokens: 0, defaultMaxNewTokens: 0 })
  })

  it('resets absent bundle defaults to neutral inheritance sentinels', () => {
    expect(applyBundleGenerationDefaultsToSessionConfig({ unrelated: true }, null)).toEqual({
      unrelated: true,
      defaultTemperature: 0,
      defaultTopP: 0,
      defaultTopK: 0,
      defaultMinP: 0,
      defaultRepetitionPenalty: 0,
      defaultMaxNewTokens: 0,
      defaultDoSample: undefined,
      defaultSamplingDefaultsDeclared: false,
    })
  })

  it('replaces stale saved DSV4 pool quant with the current bundle value only', () => {
    const saved = {
      dsv4PoolQuant: true,
      enablePrefixCache: false,
      maxCacheBlocks: 2048,
      additionalArgs: '--user-owned value',
    }

    const hydrated = applyBundleDsv4PoolQuantToSessionConfig(saved, {
      family: 'deepseek-v4',
      dsv4PoolQuantDefault: false,
    })

    expect(hydrated).toEqual({
      ...saved,
      dsv4PoolQuant: false,
    })
    expect(saved.dsv4PoolQuant).toBe(true)
  })

  it('clears a stale DSV4-only value for a known non-DSV4 bundle', () => {
    const hydrated = applyBundleDsv4PoolQuantToSessionConfig({
      dsv4PoolQuant: true,
      enablePrefixCache: false,
      maxCacheBlocks: 2048,
    }, {
      family: 'qwen3.5',
      dsv4PoolQuantDefault: false,
    })

    expect(hydrated).toEqual({
      enablePrefixCache: false,
      maxCacheBlocks: 2048,
    })
  })

  it('keeps saved state intact when current bundle detection is unavailable', () => {
    const saved = { dsv4PoolQuant: true, maxCacheBlocks: 2048 }
    expect(applyBundleDsv4PoolQuantToSessionConfig(saved, null)).toBe(saved)
    expect(applyBundleDsv4PoolQuantToSessionConfig(saved, { family: 'unknown' })).toBe(saved)
  })

  it('wires both settings surfaces on initial load and Reset', () => {
    for (const sourcePath of [
      'src/renderer/src/components/sessions/SessionSettings.tsx',
      'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
    ]) {
      const source = readFileSync(sourcePath, 'utf8')
      expect(source.match(/getGenerationDefaults\(/g)).toHaveLength(2)
      expect(source.match(/applyBundleGenerationDefaultsToSessionConfig\(/g)).toHaveLength(2)
      expect(source).toContain('return () => { active = false }')
      expect(source).toContain('resetStillCurrent')
      expect(source).toContain('setConfig(current => applyBundleGenerationDefaultsToSessionConfig(current, generationDefaults))')
    }
  })

  it('uses one shared mapper for fresh and previously launched sessions', () => {
    const source = readFileSync(
      'src/renderer/src/components/sessions/CreateSession.tsx',
      'utf8',
    )
    expect(source.match(/applyBundleGenerationDefaultsToSessionConfig\(/g)).toHaveLength(3)
    expect(source).not.toContain('function applyGenerationDefaultsToConfig')
    expect(source).not.toContain('function applyGenerationDefaultsToStoredConfig')
    expect(source).toContain('Promise.all([')
  })

  it('reconciles current DSV4 bundle state in every existing-session settings surface', () => {
    for (const sourcePath of [
      'src/renderer/src/components/sessions/SessionSettings.tsx',
      'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
      'src/renderer/src/components/sessions/CreateSession.tsx',
    ]) {
      const source = readFileSync(sourcePath, 'utf8')
      expect(source).toContain('applyBundleDsv4PoolQuantToSessionConfig')
      expect(source).toContain(
        'setConfig(current => applyBundleDsv4PoolQuantToSessionConfig(current, det))',
      )
    }
  })
})

describe('generic prefix-cache index capacity', () => {
  const source = readFileSync('src/main/sessions.ts', 'utf8')

  it('sizes the generic default by target tokens, not a flat block count', () => {
    // --max-cache-blocks counts BLOCKS. DSV4 was given an explicit 1M-token
    // index (4097 x 256) while every other family was left at a flat 1000,
    // which at the generic 64-token block indexes only 63,936 tokens. Measured
    // on the box: a 77k Gemma prompt reported 0 cached tokens on an EXACT
    // repeat and ran slower than a cold prefill (82.5s vs 55.7s); the same
    // probe at 28k reused 28,199 tokens and cut TTFT 9.10s -> 0.98s.
    expect(source).toContain('GENERIC_INDEX_TARGET_TOKENS')
    expect(source).toContain('function indexBlocksForCapacity(')
    expect(source).toContain('indexBlocksForCapacity(mutable.pagedCacheBlockSize)')
  })

  it('keeps DSV4 on its own explicit 1M-token sizing', () => {
    expect(source).toContain('const DSV4_MAX_CACHE_BLOCKS = 4097')
    expect(source).toContain('dsv4Active\n          ? DSV4_MAX_CACHE_BLOCKS')
  })

  it('bounds the index only — RAM stays governed by the byte ceiling', () => {
    expect(source).toContain('resident RAM stays governed by')
  })
})

describe('v14 lifts existing sessions off the stale cache index', () => {
  const source = readFileSync('src/main/sessions.ts', 'utf8')

  it('bumped the cache-defaults version so stored sessions re-migrate', () => {
    const version = Number(
      /const CACHE_STACK_STARTUP_DEFAULTS_VERSION = (\d+)/.exec(source)?.[1],
    )
    expect(version).toBeGreaterThanOrEqual(14)
  })

  it('lifts only the exact stale 1000, never a value the user chose', () => {
    // A flat 1000 at the generic 64-token block indexes just 63,936 tokens.
    // Measured on Gemma 4: a 77k prompt reported 0 cached tokens on an EXACT
    // repeat and ran slower than a cold prefill.
    expect(source).toContain('Number(config.maxCacheBlocks) === 1000')
    expect(source).toContain('indexBlocksForCapacity(config.pagedCacheBlockSize)')
  })

  it('does not blanket-overwrite maxCacheBlocks in the generic branch', () => {
    // The old unconditional `?? 1000` would have stranded every session.
    expect(source).not.toContain('config.maxCacheBlocks = config.maxCacheBlocks ?? 1000')
  })
})
