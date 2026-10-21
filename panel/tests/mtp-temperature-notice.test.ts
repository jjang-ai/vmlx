import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import {
  parseSessionNativeMtpMode,
  resolveMtpTemperatureNotice,
} from '../src/shared/mtpTemperatureNotice'

describe('MTP temperature disclosure (adaptive never pins the sampler)', () => {
  it('says nothing for remote servers or bundles without MTP heads', () => {
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'adaptive', temperature: 1, isRemote: true })).toBeNull()
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: false, mode: 'adaptive', temperature: 1 })).toBeNull()
  })

  it('says nothing at temperature 0: greedy requests verify by identity, nothing was pinned', () => {
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'adaptive', temperature: 0 })).toBeNull()
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: undefined, temperature: 0 })).toBeNull()
  })

  it('does not claim MTP is active when the loaded runtime disabled it for an external drafter', () => {
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, runtimeAvailable: false, mode: 'adaptive', temperature: 0.6 })).toBeNull()
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, runtimeAvailable: true, mode: 'adaptive', temperature: 0.6 }))
      .toEqual({ kind: 'active', temperature: 0.6 })
  })

  it('discloses a nonzero temperature as honored with rejection-sampling verification', () => {
    for (const mode of ['adaptive', 'auto', 'deterministic', undefined]) {
      expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode, temperature: 0.7 }))
        .toEqual({ kind: 'active', temperature: 0.7 })
    }
  })

  it('stays silent when MTP is off, explicitly or by the bundle measured default', () => {
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'off', temperature: 1 })).toBeNull()
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: undefined, modelDefaultMode: 'off', temperature: 1 })).toBeNull()
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'auto', modelDefaultMode: 'off', temperature: 1 })).toBeNull()
    expect(resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'adaptive', modelDefaultMode: 'off', temperature: 1 }))
      .toEqual({ kind: 'active', temperature: 1 })
  })

  it('parses and normalizes the persisted mode', () => {
    expect(parseSessionNativeMtpMode(undefined)).toBeUndefined()
    expect(parseSessionNativeMtpMode('not json')).toBeUndefined()
    expect(parseSessionNativeMtpMode('{}')).toBeUndefined()
    expect(parseSessionNativeMtpMode(JSON.stringify({ nativeMtpMode: 'deterministic' }))).toBe('adaptive')
    expect(parseSessionNativeMtpMode(JSON.stringify({ nativeMtpMode: 'auto' }))).toBe('adaptive')
    expect(parseSessionNativeMtpMode({ nativeMtpMode: 'adaptive' })).toBe('adaptive')
    expect(parseSessionNativeMtpMode({ nativeMtpMode: 'off' })).toBe('off')
  })

  it('uses the separate LiLiCorr choice rather than an old native-MTP default', () => {
    const old = { nativeMtpMode: 'adaptive' }
    expect(parseSessionNativeMtpMode(old, 'nemotron_lilicorr')).toBeUndefined()
    expect(parseSessionNativeMtpMode({ ...old, lilicorrMode: 'off' }, 'nemotron_lilicorr')).toBe('off')
    expect(parseSessionNativeMtpMode(JSON.stringify({ ...old, lilicorrMode: 'adaptive' }), 'nemotron_lilicorr')).toBe('adaptive')
  })

  it('is rendered with the temperature control and no sampler control is disabled by MTP', () => {
    const source = readFileSync(resolve(__dirname, '../src/renderer/src/components/chat/ChatSettings.tsx'), 'utf8')
    expect(source).toContain('resolveMtpTemperatureNotice(')
    expect(source).toContain('data-testid="mtp-temperature-notice"')
    expect(source).toMatch(/resolveMtpTemperatureNotice\(\{\s*isRemote,/)
    expect(source).not.toContain('mtpGreedyEnforced')
    expect(source).not.toContain('chat.settings.mtpTempPinned')
    expect(source).not.toContain('chat.settings.mtpTempDefault')
    expect(source).not.toContain('chat.settings.mtpTempInactive')
    const tempAt = source.indexOf("t('chat.settings.temperature')")
    const noticeAt = source.indexOf('data-testid="mtp-temperature-notice"')
    const topPAt = source.indexOf("t('chat.settings.topP')")
    expect(tempAt).toBeGreaterThan(-1)
    expect(noticeAt).toBeGreaterThan(tempAt)
    expect(noticeAt).toBeLessThan(topPAt)
  })

  it('ships copy in every locale that names Adaptive MTP and rejection sampling, with no retired keys', () => {
    for (const locale of ['en', 'es', 'ja', 'ko', 'zh']) {
      const json = JSON.parse(readFileSync(resolve(__dirname, `../src/renderer/src/i18n/locales/${locale}.json`), 'utf8'))
      expect(json.chat.settings.mtpTempActive).toContain('{{temperature}}')
      for (const retired of ['mtpTempPinned', 'mtpTempDefault', 'mtpTempInactive']) {
        expect(json.chat.settings).not.toHaveProperty(retired)
      }
      const cfg = json.sessions.config
      for (const key of ['nativeMtpModeAdaptive', 'nativeMtpModeAr', 'nativeMtpAdaptiveNote', 'nativeMtpModeTooltip', 'nativeMtpHint', 'nativeMtpDefaultOffNote']) {
        expect(typeof cfg[key]).toBe('string')
      }
      for (const retired of ['nativeMtpDepth', 'nativeMtpDepthPolicy', 'nativeMtpDepthFixedNote', 'nativeMtpDepthAdaptiveNote', 'mtpAutoBundleDefaults', 'mtpDeterministicOverride', 'nativeMtpDeterministicNote', 'nativeMtpAutoNote', 'nativeMtpCompatibleNote']) {
        expect(cfg).not.toHaveProperty(retired)
      }
      for (const key of ['mtpMode', 'mtpModeAdaptive', 'mtpModeAr', 'mtpModeFixed']) {
        expect(typeof json.sessions.performance[key]).toBe('string')
      }
    }
    const en = JSON.parse(readFileSync(resolve(__dirname, '../src/renderer/src/i18n/locales/en.json'), 'utf8'))
    expect(en.chat.settings.mtpTempActive).toMatch(/rejection/i)
    expect(en.sessions.config.nativeMtpAdaptiveNote).toMatch(/never pins/i)
    expect(en.sessions.config.nativeMtpModeAr).toBe('AR (MTP off)')
    expect(en.sessions.config.nativeMtpModeAdaptive).toBe('Adaptive MTP')
  })
})
