import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'

describe('JANGTQ acceleration user surface', () => {
  it('does not expose the internal acceleration lane as a persisted app setting', () => {
    const files = [
      'src/main/tray.ts',
      'src/main/index.ts',
      'src/preload/index.ts',
      'src/env.d.ts',
      'src/main/sessions.ts',
      'src/renderer/src/components/sessions/SessionConfigForm.tsx',
      'src/renderer/src/components/sessions/ServerSettingsDrawer.tsx',
      'src/renderer/src/components/sessions/SessionSettings.tsx',
      'src/renderer/src/components/sessions/PerformancePanel.tsx',
    ]

    const combined = files.map(file => readFileSync(file, 'utf8')).join('\n')
    expect(combined).not.toContain('jangtqMppNax')
    expect(combined).not.toContain('JANGTQ_MPP_NAX_SETTING_KEY')
    expect(combined).not.toContain('runtime:jangtqMppNaxChanged')
    expect(combined).not.toContain('--jangtq-mpp-nax')
    expect(combined).not.toContain('JANGTQ MPP/NAX')
  })
})
