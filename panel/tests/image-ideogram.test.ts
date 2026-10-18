import { describe, it, expect } from 'vitest'
import { imageNativePreset } from '../src/shared/imageCapabilities'
import { resolveImageModelFromDirectoryName } from '../src/shared/imageModels'
import { defaultImageRuntimeSettings } from '../src/shared/imageRuntimeSettings'
describe('Ideogram native settings', () => {
  it('resolves both saved affine bundles', () => {
    for (const q of [4,8]) expect(resolveImageModelFromDirectoryName(`ideogram-4-mflux-q${q}`)?.mfluxClass).toBe('Ideogram4')
  })
  it('keeps native presets separate from constant API guidance', () => {
    const caps = {loaded: true, mflux_class: 'Ideogram4'}
    expect(imageNativePreset(caps,12)).toBe('V4_TURBO_12')
    expect(imageNativePreset(caps,20)).toBe('V4_DEFAULT_20')
    expect(imageNativePreset(caps,48)).toBe('V4_QUALITY_48')
    expect(() => imageNativePreset(caps,19)).toThrow()
    expect(imageNativePreset({loaded: true,mflux_class:'QwenImage21Edit'},20)).toBeUndefined()
    expect(defaultImageRuntimeSettings('ideogram-4',4)).toMatchObject({steps:20,guidance:7})
  })
})
