import { afterEach, describe, expect, it, vi } from 'vitest'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { detectModelConfigFromDir } from '../src/main/model-config-registry'
import { glmNativeSsdRuntimeEnabled, resolveGlmDiskCacheControls, usesExactTypedPromptDiskCache, usesGlmNativeSsdPool } from '../src/shared/detectedFamilyNames'
import { buildCacheLaunchArgs } from '../src/shared/cacheLaunchArgs'

const dirs: string[] = []
afterEach(() => {
  vi.unstubAllEnvs()
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true })
})

describe('GLM native SSD defaults preserve runtime route and saved choices', () => {
  it.each([undefined, '1', '0', 'true', ''])('propagates normal startup and explicit disable %s', (flag) => {
    vi.stubEnv('VMLX_GLM5_NATIVE_SSD', flag)
    const dir = mkdtempSync(join(tmpdir(), 'vmlx-glm-native-policy-'))
    dirs.push(dir)
    writeFileSync(join(dir, 'config.json'), JSON.stringify({
      model_type: 'glm5_next', text_config: { model_type: 'glm5_next_text' },
      vision_config: { model_type: 'glm5_next_vision' },
    }))
    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('glm5-next')
    const enabled = flag === undefined || flag === '1'
    expect(glmNativeSsdRuntimeEnabled(flag)).toBe(enabled)
    expect(detected.nativeGlmSsd).toBe(enabled ? true : undefined)
    expect(usesGlmNativeSsdPool(detected)).toBe(enabled)
    expect(usesExactTypedPromptDiskCache(detected.family, usesGlmNativeSsdPool(detected))).toBe(!enabled)
    // The setting is main-process runtime metadata, never stamped into a bundle.
    expect(detected.cacheSubtype).toBe('glm5_next_native_v2')
  })

  it.each([
    [{}, {}, true],
    [{}, { isMultimodal: false }, false],
    [{}, { smelt: true }, false],
    [{ forceTextOnly: true }, { isMultimodal: true }, false],
    [{ isMultimodal: false }, {}, false],
    [{ isMultimodal: false }, { isMultimodal: true }, true],
    [{ nativeGlmSsd: false }, {}, false],
    [{ family: 'qwen4_exp' }, {}, false],
    [{ family: 'openpangu_v2' }, {}, false],
  ] as const)('uses the effective route without changing it: %j %j', (overrides, config, expected) => {
    const detected = { family: 'glm5-next', nativeGlmSsd: true, isMultimodal: true, ...overrides }
    expect(usesGlmNativeSsdPool(detected, config)).toBe(expected)
    expect(usesExactTypedPromptDiskCache('glm5-next', usesGlmNativeSsdPool(detected, config))).toBe(!expected)
  })

  it('preserves saved SSD/prefix opt-outs, old directories and ceilings', () => {
    const detected = { family: 'glm5-next', nativeGlmSsd: true, isMultimodal: true }
    for (const enablePrefixCache of [false, true]) {
      const saved = {
        isMultimodal: true, continuousBatching: true, enablePrefixCache, enableBlockDiskCache: false,
        enableDiskCache: true, blockDiskCacheDir: '/saved/pool', diskCacheDir: '/saved/legacy',
        blockDiskCacheMaxGb: 3, blockDiskCacheMaxPercent: 7, maxTokens: 4096,
      }
      const before = JSON.stringify(saved)
      const native = usesGlmNativeSsdPool(detected, saved)
      const result = buildCacheLaunchArgs({ ...saved, ...resolveGlmDiskCacheControls(detected.family, native, saved) })
      expect(JSON.stringify(saved)).toBe(before)
      expect(result.args).not.toContain('--enable-disk-cache')
      expect(result.args).not.toContain('--enable-block-disk-cache')
      expect(result.args).toContain(enablePrefixCache ? '--disable-block-disk-cache' : '--disable-prefix-cache')
    }
    const source = readFileSync('src/main/sessions.ts', 'utf8')
    const start = source.indexOf('function applyCacheStackStartupDefaultMigration(')
    const block = source.slice(start, source.indexOf('function applyLegacyCacheStackMigrations(', start))
    expect(block.indexOf('if (usesGlmNativeSsdPool(detected, config))')).toBeLessThan(block.indexOf('const legacyChanged'))
    expect(block).toContain('return markCacheStackStartupDefaultsCurrent(config, modelPath || config.modelPath) || retiredRam')
  })

  it.each([false, true])('keeps the text route preview and explicit off aligned (SSD=%s)', (enabled) => {
    const saved = { enableDiskCache: false, enableBlockDiskCache: enabled }
    const before = JSON.stringify(saved)
    const text = resolveGlmDiskCacheControls('glm5-next', false, saved)
    expect(text).toEqual({ enableDiskCache: enabled, enableBlockDiskCache: false })
    const args = buildCacheLaunchArgs({ continuousBatching: true, enablePrefixCache: true, ...text }).args
    expect(args.includes('--enable-disk-cache')).toBe(enabled)
    expect(args).toContain('--disable-block-disk-cache')
    expect(resolveGlmDiskCacheControls('glm5-next', true, saved)).toEqual(saved)
    expect(resolveGlmDiskCacheControls('qwen4_exp', false, saved)).toEqual(saved)
    expect(JSON.stringify(saved)).toBe(before)
    const form = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
    expect(form).toContain("if (normalizedDetectedFamily === 'glm5-next' && !nativeGlmSsdActive) onChange('enableBlockDiskCache', false)")
  })

  it.each(['glm5_next', 'glm5_next_text', 'glm5-next'])('preserves the explicit SSD switch and cap for %s', (family) => {
    for (const enabled of [true, false]) {
      const legacy = usesExactTypedPromptDiskCache(family, true)
      const result = buildCacheLaunchArgs({
        continuousBatching: true, enablePrefixCache: true, usePagedCache: false,
        enableDiskCache: false, enableBlockDiskCache: legacy ? false : enabled,
        forceMemoryAwareCache: legacy, blockDiskCacheDir: '/test/owned-native-pool',
        blockDiskCacheMaxGb: 3, blockDiskCacheMaxPercent: 15,
      })
      expect(result.args).toContain('--no-paged-cache')
      expect(result.args).not.toContain('--use-paged-cache')
      expect(result.policy.enableBlockDiskCache).toBe(enabled)
      expect(result.args.includes('--disable-block-disk-cache')).toBe(!enabled)
      if (enabled) {
        expect(result.args).toContain('/test/owned-native-pool')
        const cap = result.args.indexOf('--block-disk-cache-max-gb')
        expect(cap).toBeGreaterThanOrEqual(0)
        expect(result.args[cap + 1]).toBe('3')
        // The engine resolves explicit GB ahead of percent. Preserve both
        // user inputs; the native scheduler receives that resolved GB cap.
        const percent = result.args.indexOf('--block-disk-cache-max-percent')
        expect(percent).toBeGreaterThanOrEqual(0)
        expect(result.args[percent + 1]).toBe('15')
        expect(result.args).not.toContain('--enable-disk-cache')
      }
    }
  })

  it('keeps GLM text-only and other family prompt-cache policies distinct', () => {
    expect(usesExactTypedPromptDiskCache('glm5-next')).toBe(true)
    expect(usesExactTypedPromptDiskCache('openpangu_v2', true)).toBe(true)
    expect(usesExactTypedPromptDiskCache('qwen4_exp', true)).toBe(false)
  })

  it('hides non-applicable RAM and generic block controls only for native GLM', () => {
    const form = readFileSync('src/renderer/src/components/sessions/SessionConfigForm.tsx', 'utf8')
    expect(form).toContain('const nativeGlmSsdActive = usesGlmNativeSsdPool({')
    expect(form).toContain('isMultimodal: detectedIsMultimodal, forceTextOnly: detectedForceTextOnly')
    expect(form).toContain('!nativeGlmSsdActive && !dsv4Active && !blockDiskOnly')
    expect(form).toContain('!nativeGlmSsdActive && !nativeSpeculativeCache && (effectiveUsePagedCache || cachePolicy.blockDiskCacheChecked)')
    expect(form).toContain('!nativeGlmSsdActive && (exactTypedPromptDiskCache || cachePolicy.legacyDiskCacheChecked)')
    expect(form).toContain('data-vmlx-section="nativeGlmSsd"')
    for (const language of ['en', 'zh', 'ko', 'ja', 'es']) {
      const locale = JSON.parse(readFileSync(`src/renderer/src/i18n/locales/${language}.json`, 'utf8'))
      expect(locale.sessions.config.glmNativeSsdNote.length).toBeGreaterThan(40)
    }
    for (const path of ['src/main/sessions.ts', 'src/renderer/src/components/sessions/SessionSettings.tsx']) {
      const source = readFileSync(path, 'utf8')
      expect(source).toContain('const nativeGlmSsd = usesGlmNativeSsdPool(detected, config)')
      expect(source).toContain('const diskControls = resolveGlmDiskCacheControls(detectedFamily, nativeGlmSsd, config)')
      // ffc5bcc6: the prompt L2 stays off only for a DFlash2 speculative session (native SSD tier).
      expect(source).toContain('enableDiskCache: !dflash2Speculative && !dsparkSingleRequest && !!diskControls.enableDiskCache')
    }
  })
})
