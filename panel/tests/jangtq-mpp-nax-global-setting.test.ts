import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

import {
  JANGTQ_MPP_NAX_SETTING_KEY,
  normalizeJangtqMppNaxMode,
} from '../src/shared/jangtqMppNax'

describe('JANGTQ MPP/NAX app-wide setting', () => {
  it('normalizes persisted UI and CLI aliases to the three supported modes', () => {
    expect(JANGTQ_MPP_NAX_SETTING_KEY).toBe('jangtq_mpp_nax_mode')
    expect(normalizeJangtqMppNaxMode('on')).toBe('on')
    expect(normalizeJangtqMppNaxMode('1')).toBe('on')
    expect(normalizeJangtqMppNaxMode('true')).toBe('on')
    expect(normalizeJangtqMppNaxMode('off')).toBe('off')
    expect(normalizeJangtqMppNaxMode('0')).toBe('off')
    expect(normalizeJangtqMppNaxMode('false')).toBe('off')
    expect(normalizeJangtqMppNaxMode('bogus')).toBe('auto')
    expect(normalizeJangtqMppNaxMode(undefined)).toBe('auto')
  })

  it('is exposed in tray, server settings, preload events, and launch args from one setting key', () => {
    const tray = readFileSync('src/main/tray.ts', 'utf8')
    const main = readFileSync('src/main/index.ts', 'utf8')
    const preload = readFileSync('src/preload/index.ts', 'utf8')
    const env = readFileSync('src/env.d.ts', 'utf8')
    const sessions = readFileSync('src/main/sessions.ts', 'utf8')
    const drawer = readFileSync('src/renderer/src/components/sessions/ServerSettingsDrawer.tsx', 'utf8')

    expect(tray).toContain('JANGTQ_MPP_NAX_SETTING_KEY')
    expect(tray).toContain('main.tray.jangtqMppNax')
    expect(tray).toContain('runtime:jangtqMppNaxChanged')
    expect(main).toContain('JANGTQ_MPP_NAX_SETTING_KEY')
    expect(main).toContain('normalizeJangtqMppNaxMode(value)')
    expect(main).toContain('runtime:jangtqMppNaxChanged')
    expect(preload).toContain('onJangtqMppNaxChanged')
    expect(env).toContain('onJangtqMppNaxChanged')
    expect(sessions).toContain('getGlobalJangtqMppNaxMode')
    expect(sessions).toContain("--jangtq-mpp-nax', getGlobalJangtqMppNaxMode()")
    expect(drawer).toContain('handleJangtqMppNaxModeChange')
    expect(drawer).toContain('window.api.settings.set(JANGTQ_MPP_NAX_SETTING_KEY')
  })
})
