import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it, vi } from 'vitest'
vi.mock('../src/renderer/src/i18n', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
import { ImagePromptBar } from '../src/renderer/src/components/image/ImagePromptBar'
import { ImageGallery } from '../src/renderer/src/components/image/ImageGallery'
import { ImageSettings } from '../src/renderer/src/components/image/ImageSettings'
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
  it('keeps Qwen 2.1 negative conditioning inactive until CFG is enabled without losing the saved prompt', () => {
    for (const [supported, guidance, disabled, conditionalNotice] of [
      [true, 1, true, true], [true, 2, false, false], [false, 1, true, false],
    ] as const) {
      const settings = { steps: 8, width: 512, height: 512, guidance,
        negativePrompt: 'blur', count: 1, quantize: 4, strength: 1 }
      const html = renderToStaticMarkup(React.createElement(ImageSettings, {
        settings, onChange: noop, model: 'renamed-local', mode: 'generate',
        capabilities: { loaded: true, mflux_class: 'QwenImage21Edit', negative_prompt: supported },
      }))
      const input = html.match(/<input[^>]*data-vmlx-image-cap="negative(?:-unused)?"[^>]*>/)![0]
      expect(input.includes('disabled=""')).toBe(disabled)
      expect(html.includes('image.settings.negativeNeedsGuidance')).toBe(conditionalNotice)
      expect(input).toContain('value="blur"')
      expect(settings.guidance).toBe(guidance)
    }
  })
  it('disables guidance only when the loaded adapter declares it unused', () => {
    for (const supported of [false, true, undefined]) {
      const html = prompt('renamed-schnell', { loaded: true, guidance: supported })
      const input = html.match(/<input[^>]*data-vmlx-control="image-guidance-quick"[^>]*>/)![0]
      expect(input.includes('disabled=""')).toBe(supported === false)
    }
  })
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

it('shows native Ideogram presets and disables constant guidance in both control surfaces', () => {
  const caps: ImageCapabilities = {loaded: true, mflux_class:'Ideogram4', mode:'generate', variation_strength:false, negative_prompt:false}
  const quick = prompt('renamed-q4', caps)
  const full = renderToStaticMarkup(React.createElement(ImageSettings, {model:'renamed-q4',mode:'generate', capabilities:caps, onChange:noop,
    settings:{steps:20,guidance:7,width:512,height:512,negativePrompt:'',count:1,quantize:4,strength:.8}}))
  for (const html of [quick, full]) {
    expect(html).toContain('data-vmlx-control="image-native-preset"')
    expect(html).toContain('image.settings.presetGuidance')
    expect(html).toContain('image.settings.ideogramCaption')
    const input = html.match(/<input[^>]*data-vmlx-control="image-guidance-[^"]+"[^>]*>/)![0]
    expect(input).toContain('disabled=""')
  }
})

describe('native SeedVR and FIBO composer controls', () => {
  const settings = {steps:20,width:512,height:768,guidance:4,negativePrompt:'blur',count:2,quantize:0,strength:.8,resolution:'2x',softness:.2}
  const source = {name:'source.png',dataUrl:'data:image/png;base64,AAAA'}
  const caps: ImageCapabilities = {loaded:true,mflux_class:'SeedVR2',mode:'edit',upscale:true,prompt:false,steps:false,guidance:false,negative_prompt:false,mask:'none',edit_strength:false,count:false}
  function composer(promptText: string, capabilities: ImageCapabilities, sourceImage: typeof source | null = source) {
    return renderToStaticMarkup(React.createElement(ImagePromptBar, {
      prompt:promptText,onPromptChange:noop,onGenerate:noop,disabled:false,generating:false,
      settings,onSettingsChange:noop,mode:'edit',sourceImage,onSourceImageChange:noop,capabilities,
    }))
  }
  it('requires only source plus valid native settings, without a fake text prompt', () => {
    const html = composer('',caps)
    expect(submit(html)).not.toContain('disabled=""')
    expect(submit(composer('',caps,null))).toContain('disabled=""')
    expect(html).toContain('image.prompt.upscale')
    expect(html).toContain('image-upscale-resolution')
    expect(html).toContain('image-upscale-softness')
    expect(html).not.toContain('<textarea')
    for (const unused of ['image-guidance-quick','image.prompt.steps','image.prompt.size','image.prompt.strength','image.prompt.paintMask']) expect(html).not.toContain(unused)
  })
  it('keeps the full settings panel limited to native resolution/softness/seed', () => {
    const html = renderToStaticMarkup(React.createElement(ImageSettings,{settings,onChange:noop,model:'native-upscale',mode:'edit',capabilities:caps}))
    expect(html).toContain('image-upscale-resolution')
    expect(html).toContain('image-upscale-softness')
    expect(html).toContain('image.settings.seed')
    for (const unused of ['image-guidance-settings','image.settings.steps','image.settings.size','image.settings.negativePrompt','image.settings.strength','image.settings.widthCustom']) expect(html).not.toContain(unused)
  })
  it('explains JSON and prevents invalid FIBO captions without rewriting them', () => {
    const fibo: ImageCapabilities = {loaded:true,mflux_class:'FIBO',mode:'generate',variation_strength:true}
    expect(submit(composer('plain prose',fibo))).toContain('disabled=""')
    const json = ' {"short_description":"Red mug"} '
    const html = composer(json,fibo)
    expect(submit(html)).not.toContain('disabled=""')
    expect(html).toContain('image-json-prompt-help')
    expect(html).toContain('Red mug')
  })
})

 it('uses prompt-free empty-gallery help only for a loaded native upscaler', () => {
  const html = (mode: 'generate' | 'edit', upscale?: boolean) => renderToStaticMarkup(React.createElement(ImageGallery, {generations:[],generating:false,mode,upscale}))
  expect(html('edit',true)).toContain('image.gallery.emptyUpscaleBody')
  expect(html('edit',true)).not.toContain('image.gallery.emptyEditBody')
  expect(html('edit')).toContain('image.gallery.emptyEditBody')
  expect(html('generate')).toContain('image.gallery.emptyGenBody')
})
