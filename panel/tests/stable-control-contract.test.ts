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
  'chat/ChatSettings.tsx': ['chat-thinking-auto', 'chat-thinking-on', 'chat-thinking-off', 'chat-effort-default'],
  'sessions/ServerSettingsDrawer.tsx': ['server-settings-save', 'server-settings-save-restart', 'server-settings-reset'],
  'sessions/SessionDashboard.tsx': ['session-create'],
  'layout/SidebarHeader.tsx': ['chat-new'],
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
  it('selected state is an attribute, not a class, on the thinking and effort groups', () => {
    const src = R('chat/ChatSettings.tsx')
    expect(src).toContain("data-vmlx-control={`chat-effort-${effort}`}")
    expect((src.match(/data-vmlx-state=\{[^}]*\? 'selected' : 'unselected'\}/g) || []).length).toBeGreaterThanOrEqual(5)
  })
  it('alerts carry a tone attribute the harness reads instead of a class', () => {
    expect(R('Toast.tsx')).toContain('data-vmlx-tone={toast.type}')
    expect(R('image/ImageTab.tsx')).toContain('role="alert" data-vmlx-tone="error"')
    expect(R('chat/ChatSettings.tsx')).toContain('data-vmlx-tone="warning"')
    expect(R('tools/ModelConverter.tsx')).toContain('data-vmlx-tone="warning"')
    const harness = readFileSync(join(__dirname, '..', 'scripts', 'live-real-ui-model-proof.mjs'), 'utf8')
    expect(harness).toContain("const tone = alert.getAttribute('data-vmlx-tone')")
    // selected thinking/effort buttons are read from the state attribute, the class is only a fallback for older builds
    expect((harness.match(/getAttribute\('data-vmlx-state'\) === 'selected'/g) || []).length).toBeGreaterThanOrEqual(2)
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
    for (const k of ['videoFps', 'videoMaxFrames', 'videoMaxPixels', 'videoTokenBudget', 'isMultimodal']) expect(src).toContain(`settingKey="${k}"`)
    // the multimodal/video controls live in their own, default-open section, not under Tool Integration & MCP
    expect(src).toContain('sectionKey="multimodal"')
    expect(src).toContain('multimodal: true,')
    expect(src.indexOf('sectionKey="tools"')).toBeLessThan(src.indexOf('sectionKey="multimodal"'))
    expect(src.indexOf("t('sessions.config.multimodalSupport')")).toBeGreaterThan(src.indexOf('sectionKey="multimodal"'))
  })
})

describe('keyboard interaction on the server settings drawer', () => {
  it('Escape closes the drawer from anywhere inside it (parity with the modal)', () => {
    const src = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'components', 'sessions', 'ServerSettingsDrawer.tsx'), 'utf8')
    // document-level while mounted: after a keyboard save the focused Save button disables itself and focus
    // falls to the body, so a drawer-scoped key handler never saw Escape (live contract run at c740cac8)
    expect(src).toContain("const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape' && !e.defaultPrevented) onClose() }")
    expect(src).toContain("document.addEventListener('keydown', onKey)")
  })
})

describe('every collapsible settings section carries its contract key', () => {
  it('each Section bound to expandedSections.<key> declares sectionKey="<key>"', () => {
    const src = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'components', 'sessions', 'SessionConfigForm.tsx'), 'utf8')
    // one Section opening per line; arrow functions inside the tag contain '>' so match by line, not by tag
    const sections = src.split('\n').filter(l => l.includes('<Section ') && l.includes('expandedSections.'))
    expect(sections.length).toBeGreaterThan(10)
    for (const sec of sections) {
      const key = sec.match(/expandedSections\.([a-zA-Z]+)/)![1]
      expect(sec, `section ${key}`).toContain(`sectionKey="${key}"`)
      expect(sec.split(`sectionKey="${key}"`).length, `section ${key} declared once`).toBe(2)
    }
  })
})

describe('the full-page session settings carry the same contract as the drawer', () => {
  it('surface, session id and the back / save / save-and-restart controls are addressable', () => {
    const src = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'components', 'sessions', 'SessionSettings.tsx'), 'utf8')
    expect(src).toContain('data-vmlx-surface="session-settings" data-vmlx-session-id={sessionId}')
    for (const c of ['session-settings-back', 'session-settings-save', 'session-settings-save-restart']) expect(src).toContain(`data-vmlx-control="${c}"`)
  })
})

describe('every config control bound to a config key carries that key as its setting attribute', () => {
  it("each SliderField / SelectField / Field whose onChange names a key declares settingKey=<key>", () => {
    const src = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'components', 'sessions', 'SessionConfigForm.tsx'), 'utf8')
    const re = /<(SliderField|SelectField|Field)\b/g; let m; let checked = 0; const missing: string[] = []
    while ((m = re.exec(src))) {
      let j = m.index + m[0].length, depth = 0
      for (; j < src.length; j++) { const c = src[j]; if (c === '{') depth++; else if (c === '}') depth--; else if (c === '>' && depth === 0) break }
      const tag = src.slice(m.index, j + 1); const k = tag.match(/onChange\('([a-zA-Z]+)'/)
      if (!k) continue
      checked++
      if (!tag.includes(`settingKey="${k[1]}"`)) missing.push(k[1])
    }
    expect(checked).toBeGreaterThan(20)
    expect(missing).toEqual([])
  })
})
