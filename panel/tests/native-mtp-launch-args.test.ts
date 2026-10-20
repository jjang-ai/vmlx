import { describe, expect, it } from 'vitest'
import {
  buildNativeMtpLaunchArgs,
  buildNativeMtpLaunchEnvironment,
  nativeMtpSessionChoice,
  normalizeNativeMtpMode,
  resolveNativeMtpMode,
  resolveNativeMtpStartupMode,
} from '../src/shared/nativeMtpLaunchArgs'

const ADAPTIVE = ['--native-mtp-depth-policy', 'adaptive', '--native-mtp-sampling-policy', 'compatible-only']

describe('native MTP launch args — two product modes', () => {
  it('attaches LiLiCorr only for an explicit adaptive session, without a depth override', () => {
    const input = { supported: true, adapter: 'nemotron_lilicorr', modelDefaultMode: 'off' as const }
    expect(buildNativeMtpLaunchEnvironment(input)).toEqual({ VMLX_NEMOTRON_LILICORR: '0' })
    expect(buildNativeMtpLaunchEnvironment({ ...input, mode: 'adaptive' }))
      .toEqual({ VMLX_NEMOTRON_LILICORR: '1' })
    expect(buildNativeMtpLaunchEnvironment({ ...input, mode: 'off' }))
      .toEqual({ VMLX_NEMOTRON_LILICORR: '0' })
    expect(buildNativeMtpLaunchEnvironment({ ...input, mode: 'adaptive', externalSpeculativeActive: true }))
      .toEqual({ VMLX_NEMOTRON_LILICORR: '0' })
    expect(buildNativeMtpLaunchEnvironment({ ...input, adapter: undefined })).toEqual({})
    expect(buildNativeMtpLaunchArgs({ ...input, mode: 'adaptive' })).toEqual(ADAPTIVE)
    const oldSession = { nativeMtpMode: 'adaptive' as const }
    const mode = nativeMtpSessionChoice(oldSession, input.adapter)
    expect(buildNativeMtpLaunchEnvironment({ ...input, mode }))
      .toEqual({ VMLX_NEMOTRON_LILICORR: '0' })
    expect(nativeMtpSessionChoice(oldSession)).toBe('adaptive')
    expect(nativeMtpSessionChoice({ ...oldSession, lilicorrMode: 'adaptive' }, input.adapter))
      .toBe('adaptive')
  })
  it('adaptive emits the adaptive policy, compatible-only sampling and NO depth', () => {
    expect(buildNativeMtpLaunchArgs({ supported: true, mode: 'adaptive' })).toEqual(ADAPTIVE)
    // --native-mtp-depth is the engine's explicit VMLINUX_NATIVE_MTP_DEPTH
    // override; emitting it pins the start depth and bypasses tuning
    // sidecars/bundle stamps. The engine resolves its own ceiling.
    expect(buildNativeMtpLaunchArgs({ supported: true, mode: 'adaptive' })).not.toContain('--native-mtp-depth')
  })

  it('off emits only --disable-native-mtp', () => {
    expect(buildNativeMtpLaunchArgs({ supported: true, mode: 'off' })).toEqual(['--disable-native-mtp'])
  })

  it('never emits greedy-only or deterministic-defaults any more', () => {
    for (const mode of ['adaptive', 'auto', 'deterministic', undefined]) {
      const args = buildNativeMtpLaunchArgs({ supported: true, mode })
      expect(args).not.toContain('greedy-only')
      expect(args).not.toContain('deterministic-defaults')
      expect(args).not.toContain('fixed')
    }
  })

  it('normalizes legacy persisted spellings to adaptive and keeps off', () => {
    expect(normalizeNativeMtpMode('auto')).toBe('adaptive')
    expect(normalizeNativeMtpMode('deterministic')).toBe('adaptive')
    expect(normalizeNativeMtpMode('adaptive')).toBe('adaptive')
    expect(normalizeNativeMtpMode('off')).toBe('off')
    for (const bad of [undefined, null, '', 'fixed', 3, {}]) {
      expect(normalizeNativeMtpMode(bad)).toBeUndefined()
    }
    expect(buildNativeMtpLaunchArgs({ supported: true, mode: 'deterministic' })).toEqual(ADAPTIVE)
    expect(buildNativeMtpLaunchArgs({ supported: true, mode: 'auto' })).toEqual(ADAPTIVE)
  })

  it('disables native MTP when an external drafter owns the decode step', () => {
    expect(buildNativeMtpLaunchArgs({
      supported: true, mode: 'adaptive', externalSpeculativeActive: true,
    })).toEqual(['--disable-native-mtp'])
  })

  it('a bundle whose verifier measured slower (defaultMode off) starts AR unless adaptive is explicit', () => {
    // undefined / legacy 'auto' never recorded an explicit opt-in.
    expect(resolveNativeMtpMode({ mode: undefined, modelDefaultMode: 'off' })).toBe('off')
    expect(resolveNativeMtpMode({ mode: 'auto', modelDefaultMode: 'off' })).toBe('off')
    expect(buildNativeMtpLaunchArgs({ supported: true, modelDefaultMode: 'off' })).toEqual(['--disable-native-mtp'])
    // 'adaptive' and legacy 'deterministic' are explicit opt-ins.
    expect(resolveNativeMtpMode({ mode: 'adaptive', modelDefaultMode: 'off' })).toBe('adaptive')
    expect(resolveNativeMtpMode({ mode: 'deterministic', modelDefaultMode: 'off' })).toBe('adaptive')
    expect(buildNativeMtpLaunchArgs({ supported: true, mode: 'adaptive', modelDefaultMode: 'off' })).toEqual(ADAPTIVE)
  })

  it('startup default is adaptive for gated bundles, AR only for a measured-off bundle, explicit always wins', () => {
    for (const family of ['qwen4-exp', 'qwen4_exp', 'qwen4_exp_text', 'qwen3.5', 'glm5-next', 'hy3', undefined]) {
      expect(resolveNativeMtpStartupMode(family)).toBe('adaptive')
      expect(resolveNativeMtpStartupMode(family, undefined, 'off')).toBe('off')
      expect(resolveNativeMtpStartupMode(family, 'off')).toBe('off')
      expect(resolveNativeMtpStartupMode(family, 'adaptive', 'off')).toBe('adaptive')
      expect(resolveNativeMtpStartupMode(family, 'auto', 'off')).toBe('adaptive')
      expect(resolveNativeMtpStartupMode(family, 'deterministic')).toBe('adaptive')
    }
  })

  it('emits nothing for unsupported bundles', () => {
    expect(buildNativeMtpLaunchArgs({ supported: false, mode: 'adaptive' })).toEqual([])
  })
})
