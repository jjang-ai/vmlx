import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import {
  parseSessionNativeMtpMode,
  resolveMtpTemperatureNotice,
} from '../src/shared/mtpTemperatureNotice'

describe('MTP temperature disclosure', () => {
  it('says nothing when the bundle has no MTP heads', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: false, mode: 'deterministic', temperature: 0 }),
    ).toBeNull()
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: false, mode: 'auto', temperature: 1 }),
    ).toBeNull()
  })

  it('explains the pinned 0 in deterministic mode', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'deterministic', temperature: 0 }),
    ).toEqual({ kind: 'pinned' })
  })

  it('describes an explicit sampled temperature in Auto as honored (kwargs win)', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'auto', temperature: 1 }),
    ).toEqual({ kind: 'active', temperature: 1 })
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'auto', temperature: 0.01 }),
    ).toEqual({ kind: 'active', temperature: 0.01 })
  })

  it('distinguishes the overridable Auto default from forced Deterministic mode', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'auto', temperature: 0 }),
    ).toEqual({ kind: 'default' })
  })

  it('flags a stale nonzero temperature in deterministic mode', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'deterministic', temperature: 1 }),
    ).toEqual({ kind: 'inactive', temperature: 1 })
  })

  it('stays silent when the user explicitly turned MTP off', () => {
    expect(
      resolveMtpTemperatureNotice({ nativeMtpSupported: true, mode: 'off', temperature: 1 }),
    ).toBeNull()
  })

  it('treats a missing/!unreadable mode as auto, never as deterministic', () => {
    expect(parseSessionNativeMtpMode(undefined)).toBe('auto')
    expect(parseSessionNativeMtpMode('not json')).toBe('auto')
    expect(parseSessionNativeMtpMode('{}')).toBe('auto')
    expect(parseSessionNativeMtpMode(JSON.stringify({ nativeMtpMode: 'deterministic' }))).toBe('deterministic')
    expect(parseSessionNativeMtpMode({ nativeMtpMode: 'off' })).toBe('off')
  })

  it('is actually rendered in the temperature area of Chat Settings', () => {
    const source = readFileSync(
      resolve(__dirname, '../src/renderer/src/components/chat/ChatSettings.tsx'),
      'utf8',
    )
    expect(source).toContain('resolveMtpTemperatureNotice(')
    expect(source).toContain('data-testid="mtp-temperature-notice"')
    expect(source).toContain("const mtpGreedyEnforced = detectedNativeMtpSupported === true && nativeMtpMode === 'deterministic'")
    expect(source).toContain('disabled={mtpGreedyEnforced}')
    expect(source).toContain('const displayedTemperature = mtpGreedyEnforced')
    expect(source).toContain('const displayedTopP = mtpGreedyEnforced')
    expect(source).toContain('const displayedTopKValue = mtpGreedyEnforced')
    expect(source).toContain('const displayedMinP = mtpGreedyEnforced')
    // all three states have copy wired
    expect(source).toContain('chat.settings.mtpTempPinned')
    expect(source).toContain('chat.settings.mtpTempDefault')
        expect(source).toContain('chat.settings.mtpTempInactive')
    // it must sit with the temperature control, not in some unrelated section
    const tempAt = source.indexOf("t('chat.settings.temperature')")
    const noticeAt = source.indexOf('data-testid="mtp-temperature-notice"')
    const topPAt = source.indexOf("t('chat.settings.topP')")
    expect(tempAt).toBeGreaterThan(-1)
    expect(noticeAt).toBeGreaterThan(tempAt)
    expect(noticeAt).toBeLessThan(topPAt)
  })

  it('ships user-facing copy that names MTP as the reason', () => {
    const en = JSON.parse(
      readFileSync(resolve(__dirname, '../src/renderer/src/i18n/locales/en.json'), 'utf8'),
    )
    const settings = en.chat.settings
    expect(settings.mtpTempPinned).toMatch(/MTP/)
    expect(settings.mtpTempPinned).toMatch(/greedy/i)
    expect(settings.mtpTempPinned).toMatch(/Auto/)
    expect(settings.mtpTempActive).toMatch(/stochastic/i)
  })

  it('discloses Auto startup defaults without promising to preserve bundle sampling', () => {
    const en = JSON.parse(readFileSync(
      resolve(__dirname, '../src/renderer/src/i18n/locales/en.json'), 'utf8',
    ))
    for (const text of [en.sessions.config.nativeMtpAutoNote, en.sessions.config.nativeMtpModeTooltip]) {
      expect(text).toMatch(/greedy.*default/i)
      expect(text).toMatch(/explicit/i)
      expect(text).not.toMatch(/preserves.*bundle sampling/i)
    }
    expect(en.sessions.config.nativeMtpHint).not.toContain('falling back to depth 1')
  })
})
