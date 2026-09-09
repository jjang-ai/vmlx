import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
vi.mock('../src/renderer/src/i18n', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
import { ImagePromptBar } from '../src/renderer/src/components/image/ImagePromptBar'
import { ImageGallery } from '../src/renderer/src/components/image/ImageGallery'
import type { ImageGenerationInfo } from '../src/renderer/src/components/image/ImageTab'
import type { ImageCapabilities } from '../src/shared/imageCapabilities'
import { imageGuidanceFromInput } from '../src/shared/imageCapabilities'
const noop = () => {}
function prompt(modelName: string, capabilities: ImageCapabilities, maskBase64?: string) {
  return renderToStaticMarkup(React.createElement(ImagePromptBar, {
    prompt: 'Make the marked area blue', onPromptChange: noop, onGenerate: noop,
    disabled: false, generating: false, onSettingsChange: noop,
    mode: 'edit', modelName, sourceImage: { name: 'input.png', dataUrl: 'data:image/png;base64,AAAA' },
    onSourceImageChange: noop, capabilities, maskBase64, onMaskChange: noop,
    settings: { steps: 20, width: 512, height: 512, guidance: 4, negativePrompt: '', count: 1, quantize: 8, strength: 0.8 },
  }))
}
function submit(html: string) {
  return html.match(/<button[^>]*data-vmlx-control="image-generate"[^>]*>/)![0]
}
describe('image presentation follows task capabilities', () => {
  it('preserves the Fill guidance preset instead of silently capping at 20', () => {
    for (const value of [0, 3.5, 20, 30, 45]) expect(imageGuidanceFromInput(String(value))).toBe(value)
    expect(imageGuidanceFromInput('')).toBe(0)
    expect(imageGuidanceFromInput('Infinity')).toBe(0)
    const html = prompt('dev-fill', { loaded: true, mode: 'edit', mask: 'required' })
    const input = html.match(/<input[^>]*data-vmlx-control="image-guidance-quick"[^>]*>/)![0]
    expect(input).not.toContain('max=')
  })
  it('a renamed Fill model requires a mask before submit', () => {
    const caps: ImageCapabilities = { loaded: true, mode: 'edit', mflux_class: 'Flux1Fill', mask: 'required', edit_strength: false }
    const missing = prompt('my-local-q8', caps)
    expect(missing).toContain('image.prompt.paintMask')
    expect(submit(missing)).toContain('disabled=""')
    expect(submit(prompt('my-local-q8', caps, 'AAAA'))).not.toContain('disabled=""')
  })
  it('a non-Fill model with fill in its name does not invent a mask requirement', () => {
    const html = prompt('refill-photo-editor', { loaded: true, mode: 'edit', mflux_class: 'QwenImageEdit', mask: 'none', edit_strength: false })
    expect(html).not.toContain('image.prompt.paintMask')
    expect(submit(html)).not.toContain('disabled=""')
  })
  it('unloaded capability data is not a live mask requirement', () => {
    expect(prompt('fill', { loaded: false, mask: 'required' })).not.toContain('image.prompt.paintMask')
  })
  for (const mode of ['edit', 'generate', undefined] as const) {
    for (const hasSource of [false, true]) {
      it(`shows stored strength only for an actual variation: mode=${mode}, source=${hasSource}`, () => {
        const row: ImageGenerationInfo = {
          id: 'a', sessionId: 's', prompt: 'test', modelName: 'local-folder', width: 512, height: 512,
          steps: 8, guidance: 4, strength: 0.85, imagePath: '/output.png',
          sourceImagePath: hasSource ? '/input.png' : undefined, createdAt: 0,
        }
        const html = renderToStaticMarkup(React.createElement(ImageGallery, { generations: [row], generating: false, mode }))
        expect(html.includes('image.gallery.strengthLabel')).toBe(mode === 'generate' && hasSource)
        expect(row.strength).toBe(0.85) // history is preserved, not rewritten
      })
    }
  }
})
