import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'

/**
 * Stable automation/theming contract: controls are addressed by
 * data-vmlx-control / data-vmlx-setting / data-vmlx-section values that never
 * depend on translated text, Tailwind classes or layout. A restyle may change
 * any class or copy, but must keep every value listed here.
 */
const R = (p: string) => readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'components', p), 'utf8')

const CONTRACT: Record<string, string[]> = {
  'chat/InputBox.tsx': ['chat-composer', 'chat-attach', 'chat-stop', 'chat-send'],
  'image/ImagePromptBar.tsx': ['image-generate', 'image-cancel'],
  'image/ImageTopBar.tsx': ['image-switch-model', 'image-browse-custom', 'image-logs', 'image-settings', 'image-stop', 'image-retry', 'image-toggle-sidebar'],
  'image/ImageModelPicker.tsx': ['image-keep-current-model'],
  'sessions/SessionCard.tsx': ['session-card-open', 'session-card-start', 'session-card-stop', 'session-card-configure', 'session-card-sleep', 'session-card-wake', 'session-card-delete', 'session-card-repoint'],
  'layout/ChatModeToolbar.tsx': ['chat-settings', 'server-settings'],
  'sessions/SessionView.tsx': ['session-start', 'session-stop'],
}

describe('stable control contract (data-vmlx-*)', () => {
  for (const [file, values] of Object.entries(CONTRACT)) {
    it(`${file} keeps its data-vmlx-control values`, () => {
      const src = R(file)
      for (const v of values) expect(src, `${file} lacks data-vmlx-control="${v}"`).toContain(`data-vmlx-control="${v}"`)
    })
  }
  it('mode tabs carry mode-<id> and an active state, independent of their translated labels', () => {
    const src = R('layout/TitleBar.tsx')
    expect(src).toContain('data-vmlx-control={`mode-${mode}`}')
    expect(src).toContain('data-vmlx-state={active ? "active" : "inactive"}')
    for (const m of ['code', 'chat', 'server', 'tools', 'image', 'api']) expect(src).toContain(`mode="${m}"`)
  })
  it('session card actions carry the session id', () => {
    const src = R('sessions/SessionCard.tsx')
    expect((src.match(/data-vmlx-session-id=\{session\.id\}/g) || []).length).toBeGreaterThanOrEqual(8)
  })
  it('form primitives expose setting and section keys; the video fields use them', () => {
    const src = R('sessions/SessionConfigForm.tsx')
    expect(src).toContain('data-vmlx-setting={settingKey}')
    expect(src).toContain('data-vmlx-section={sectionKey}')
    expect(src).toContain("data-vmlx-state={expanded ? 'open' : 'closed'}")
    for (const k of ['videoFps', 'videoMaxFrames', 'videoMaxPixels', 'videoTokenBudget']) expect(src).toContain(`settingKey="${k}"`)
  })
})
