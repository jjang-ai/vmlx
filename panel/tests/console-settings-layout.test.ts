import { describe, expect, it } from 'vitest'
import { readFileSync, existsSync } from 'node:fs'
import { resolve } from 'node:path'

const read = (path: string) => readFileSync(resolve(__dirname, '../src', path), 'utf8')
describe('Console settings and responsive layout contract', () => {
  it('uses dark on-accent Markdown without recoloring assistant prose or code surfaces', () => {
    expect(read('renderer/src/components/chat/MessageBubble.tsx')).toContain('prose-on-accent bg-primary text-primary-foreground')
    const css = read('renderer/src/index.css')
    const rule = css.split('.dark .prose-on-accent .prose {')[1].split('}')[0]
    for (const token of ['body', 'headings', 'links', 'bold', 'counters', 'bullets', 'quotes', 'code']) {
      expect(rule).toContain(`--tw-prose-${token}: var(--text-on-accent)`)
    }
    expect(rule).not.toContain('--tw-prose-pre-')
    expect(css).toContain('.dark .prose-on-accent .prose blockquote')
  })
  it('keeps connection fields plus six supported server groups', () => {
    const form = read('renderer/src/components/sessions/SessionConfigForm.tsx')
    const groups = [...form.matchAll(/<Section [^\n]*sectionKey="([^"]+)"/g)].map(m => m[1])
    expect(groups).toEqual(['server', 'concurrent', 'prefixCache', 'power', 'performance', 'tools', 'specDecode'])
    expect(form).not.toContain('cacheControlUpdatesForPagedToggle(v')
    expect(form).not.toContain('DistributedNodeList')
    expect(form).toContain('data-vmlx-section="retired-distributed"')
    expect(form).toContain("onChange('distributedEnabled', false)")
    for (const field of ['maxNumSeqs', 'prefillBatchSize', 'blockDiskCacheMaxPercent', 'videoFps', 'videoMaxFrames', 'videoMaxPixels', 'videoTokenBudget', 'nativeMtpMode', 'nativeMtpDepth', 'streamInterval', 'maxTokens', 'maxContextLength']) {
      expect(form).toContain(`settingKey="${field}"`)
    }
    expect(form).toContain('nativeMtpDetected &&')
    expect(form).toContain('exactTypedPromptDiskCache || cachePolicy.legacyDiskCacheChecked')
  })
  it('uses only Console, even for legacy light/system preferences', () => {
    const provider = read('renderer/src/providers/ThemeProvider.tsx')
    expect(provider).toContain("root.classList.remove('light')")
    expect(provider).toContain("localStorage.setItem('vmlx-theme', 'console-amber')")
    expect(provider).not.toContain('matchMedia')
    expect(read('renderer/src/components/layout/TitleBar.tsx')).not.toContain('ThemeToggle')
    expect(existsSync(resolve(__dirname, '../src/renderer/src/components/ui/theme-toggle.tsx'))).toBe(false)
    expect(read('renderer/index.html')).toContain('name="color-scheme" content="dark"')
  })
  it('has no experience-mode gate or alternate default preset', () => {
    expect(read('renderer/src/components/layout/Sidebar.tsx')).not.toContain('InferenceMode')
    expect(read('renderer/src/components/sessions/ServerSettingsDrawer.tsx')).not.toContain('InferenceMode')
    expect(read('renderer/src/components/sessions/ServerSettingsDrawer.tsx')).toContain('useState<SessionConfig>(DEFAULT_CONFIG)')
    expect(read('main/index.ts')).not.toContain("getSetting('inference_mode')")
    expect(existsSync(resolve(__dirname, '../src/renderer/src/components/layout/InferenceMode.tsx'))).toBe(false)
  })
  it('keeps tall image pickers top-aligned and card actions wrappable', () => {
    const picker = read('renderer/src/components/image/ImageModelPicker.tsx')
    expect(picker).not.toContain('h-full flex items-center justify-center')
    expect(picker).toContain('data-vmlx-section="image-model-picker"')
    expect(read('renderer/src/components/sessions/SessionCard.tsx')).toContain('data-vmlx-section="session-card-actions" className="flex flex-wrap')
    expect(read('renderer/src/App.tsx')).not.toContain("t('chat.quickStart.smeltTitle')")
  })
})
