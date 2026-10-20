import { execFileSync } from 'node:child_process'
import { describe, expect, it } from 'vitest'
import { buildCommandPreview } from '../src/renderer/src/components/sessions/SessionSettings'
import { DEFAULT_CONFIG } from '../src/renderer/src/components/sessions/SessionConfigForm'

describe('actual DFlash2 command preview', () => {
  it('mirrors LiLiCorr opt-in and preserves old native-MTP sessions as AR', () => {
    const bolt = { family: 'nemotron-h', isMultimodal: true,
      architectureHints: { nemotronImageOnly: true },
      nativeMtp: { supported: true, adapter: 'nemotron_lilicorr' as const, defaultMode: 'off' as const } }
    const config = { ...DEFAULT_CONFIG, nativeMtpMode: 'adaptive' as const }
    const off = buildCommandPreview('/models/bolt', config, bolt)
    expect(off).toContain('VMLX_NEMOTRON_LILICORR=0 ')
    expect(off).toContain('--disable-native-mtp')
    const adaptive = buildCommandPreview('/models/bolt', { ...config, lilicorrMode: 'adaptive' }, bolt)
    expect(adaptive).toContain('VMLX_NEMOTRON_LILICORR=1 ')
    expect(adaptive).toContain('--native-mtp-depth-policy')
    expect(adaptive).not.toContain('--disable-native-mtp')
    expect(adaptive).not.toMatch(/--native-mtp-depth\s/)
  })
  const detected = { family: 'glm5-next', isMultimodal: true, nativeGlmSsd: true }
  it('shows the bundled drafter, non-batched lane, and saved SSD settings', () => {
    const preview = buildCommandPreview('/models/target', { ...DEFAULT_CONFIG,
      blockDiskCacheDir: '/private/ssd', blockDiskCacheMaxPercent: 7 }, detected,
      '/models/target/dflash2', true)
    expect(preview).toContain('--no-continuous-batching')
    expect(preview).toContain('--speculative-model')
    expect(preview).toContain('/models/target/dflash2')
    expect(preview).toContain('--block-disk-cache-max-percent')
    expect(preview).toContain('/private/ssd')
    expect(preview).not.toContain('--use-paged-cache')
  })
  it('shows the bundled-drafter opt-out rather than silently relying on the CLI default', () => {
    const preview = buildCommandPreview('/models/target', { ...DEFAULT_CONFIG,
      useBundledDflash2: false }, detected, '', true)
    expect(preview).toContain('--no-bundled-dflash2')
    expect(preview).not.toContain('--speculative-model')
    expect(preview).toContain('--continuous-batching')
  })
  it('uses explicit metadata for renamed drafters and authoritative non-drafter configs', () => {
    const renamed = buildCommandPreview('/models/target', { ...DEFAULT_CONFIG,
      speculativeModel: '/models/Custom Drafter', blockDiskCacheDir: '/private/ssd' }, detected, '', false, true)
    expect(renamed).toContain('--no-continuous-batching')
    expect(renamed).toContain('--enable-block-disk-cache')
    const notDrafter = buildCommandPreview('/models/target', { ...DEFAULT_CONFIG,
      speculativeModel: '/models/dflash2-not-a-drafter' }, detected, '', false, false)
    expect(notDrafter).toContain('--continuous-batching')
    expect(notDrafter).not.toContain('--no-continuous-batching')
  })

  it('round-trips spaces and shell metacharacters without exposing an API key', () => {
    const model = "/models/Eric's Model $(touch nope)"
    const drafter = '/models/Custom Drafter; echo nope'
    const command = buildCommandPreview(model, { ...DEFAULT_CONFIG,
      speculativeModel: drafter, apiKey: 'private-test-key',
      blockDiskCacheDir: '/cache/space dir', additionalArgs: '--custom-flag "friendly model"' }, detected, '', false, true)
    const parsed = JSON.parse(execFileSync('/usr/bin/python3', ['-c',
      'import json,shlex,sys; print(json.dumps(shlex.split(sys.stdin.read().replace(chr(92)+chr(10),""),comments=True)))'], { input: command, encoding: 'utf8' }))
    expect(parsed.slice(0, 3)).toEqual(['vmlx-engine', 'serve', model])
    expect(parsed[parsed.indexOf('--speculative-model') + 1]).toBe(drafter)
    expect(parsed[parsed.indexOf('--block-disk-cache-dir') + 1]).toBe('/cache/space dir')
    expect(parsed[parsed.indexOf('--custom-flag') + 1]).toBe('friendly model')
    expect(command).not.toContain('private-test-key')
    expect(parsed).not.toContain('#')
  })

  it('keeps the settings renderer usable while a quote is unfinished', () => {
    const preview = buildCommandPreview('/models/target', { ...DEFAULT_CONFIG, additionalArgs: '--custom "unfinished' }, detected)
    expect(preview).toBe('# Additional Arguments has an unclosed quote')
    expect(preview).not.toContain('vmlx-engine serve')
  })

})
