import { describe, expect, it, vi } from 'vitest'
vi.mock('../src/renderer/src/i18n', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
import { buildImageCurl, buildImagePython, buildImageJavaScript } from '../src/renderer/src/components/api/CodeSnippets'

describe('image generation snippets preserve engine defaults', () => {
  for (const model of ['dev', 'schnell', 'z-image-turbo', 'my-local-image-alias']) {
    for (const builder of [buildImageCurl, buildImagePython, buildImageJavaScript]) {
      it(`${builder.name} does not force Schnell settings on ${model}`, () => {
        const text = builder('http://localhost:8115', null, model, false)
        expect(text).toContain('/v1/images/generations')
        expect(text).toContain(model)
        expect(text).toContain('1024x1024')
        expect(text).not.toMatch(/\bsteps\b|\bguidance\b/)
      })
    }
  }
  it('curl emits valid image request JSON', () => {
    const text = buildImageCurl('http://localhost:8115', null, 'dev', false)
    const body = JSON.parse(text.match(/-d '([\s\S]+)'$/)![1])
    expect(body).toMatchObject({model: 'dev', size: '1024x1024', response_format: 'b64_json'})
    expect(body).not.toHaveProperty('steps')
    expect(body).not.toHaveProperty('guidance')
  })
})
