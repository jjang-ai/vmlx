import { describe, expect, it } from 'vitest'
import { readFileSync } from 'fs'
import { join } from 'path'
import { imageVariationPreservesSource } from '../src/shared/imageCapabilities'

const R = (p: string) => readFileSync(join(__dirname, '..', 'src', p), 'utf8')

describe('image controls follow the loaded model\'s real capabilities (/health.image)', () => {
  it('describes native source preservation only for qualified loaded adapters', () => {
    for (const mflux_class of ['Flux1', 'ZImage']) {
      expect(imageVariationPreservesSource({ loaded: true, mflux_class, variation_strength: true })).toBe(true)
      expect(imageVariationPreservesSource({ loaded: false, mflux_class, variation_strength: true })).toBe(false)
      expect(imageVariationPreservesSource({ loaded: true, mflux_class, variation_strength: false })).toBe(false)
    }
    expect(imageVariationPreservesSource(null)).toBe(false)
    expect(imageVariationPreservesSource({ loaded: true, mflux_class: 'Unknown', variation_strength: true })).toBe(false)
    const bar = R('renderer/src/components/image/ImagePromptBar.tsx')
    expect(bar).toContain('imageVariationPreservesSource(capabilities)')
    expect(bar).toContain("'image.prompt.sourcePreservationTip'")
    for (const loc of ['en', 'es', 'ja', 'ko', 'zh']) {
      const strings = JSON.parse(R(`renderer/src/i18n/locales/${loc}.json`)).image.prompt
      expect(strings.sourcePreservation).toBeTruthy()
      expect(strings.sourcePreservationTip).toBeTruthy()
    }
  })
  it('the tab reads capabilities when the server runs and forgets them when it stops', () => {
    const tab = R('renderer/src/components/image/ImageTab.tsx')
    expect(tab).toContain("imageRuntimeSnapshot(await resp.json())")
    expect(tab).toContain("setCapabilities(snapshot.capabilities)")
    expect(tab).toContain("setCapabilities(null)")
    expect(tab).toContain('capabilities={capabilities}')
  })
  it('settings hide edit strength and disable the negative prompt only when the model says so', () => {
    const s = R('renderer/src/components/image/ImageSettings.tsx')
    expect(s).toContain("const strengthUnused = isEdit && capabilities?.edit_strength === false")
    expect(s).toContain("const negativeUnused = capabilities?.negative_prompt === false")
    expect(s).toContain('data-vmlx-image-cap="strength-unused"')
    expect(s).toContain('disabled={negativeUnused}')
    expect(s).toContain('data-vmlx-image-cap="summary"')
  })
  it('the prompt bar shows strength only where it is effective', () => {
    const p = R('renderer/src/components/image/ImagePromptBar.tsx')
    expect(p).toContain("(isEdit && capabilities?.edit_strength !== false) || (isVariation && capabilities?.variation_strength !== false)")
  })
  it('the custom class list includes every edit class the engine accepts', () => {
    const picker = R('renderer/src/components/image/ImageModelPicker.tsx')
    for (const cls of ['QwenImageEdit', 'Flux1Kontext', 'Flux1Fill', 'Flux2KleinEdit']) expect(picker).toContain(`<option value="${cls}">`)
  })
  it('every locale carries the capability strings with their placeholders', () => {
    const dir = join(__dirname, '..', 'src', 'renderer', 'src', 'i18n', 'locales')
    const en = JSON.parse(readFileSync(join(dir, 'en.json'), 'utf8'))
    for (const loc of ['es', 'ja', 'ko', 'zh']) {
      const other = JSON.parse(readFileSync(join(dir, `${loc}.json`), 'utf8'))
      for (const key of ['strengthNotUsed', 'negativeNotUsed', 'capClass']) {
        const want = (en.image.settings[key].match(/\{\w+\}/g) || []).sort()
        const got = (other.image.settings[key].match(/\{\w+\}/g) || []).sort()
        expect(got, `${loc}.${key}`).toEqual(want)
      }
      expect(other.image.picker.classFlux2KleinEdit).toBeTruthy()
    }
  })
})
