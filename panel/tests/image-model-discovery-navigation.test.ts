import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { IMAGE_MODEL_DISCOVERY_NAVIGATION, resolveDownloadModelType } from '../src/shared/modelDiscoveryNavigation'

describe('Image folder picker to model discovery', () => {
  it('requests image discovery without a model, precision or launch instruction', () => {
    expect(IMAGE_MODEL_DISCOVERY_NAVIGATION).toEqual({ mode: 'models', downloadModelType: 'image' })
    expect(resolveDownloadModelType(IMAGE_MODEL_DISCOVERY_NAVIGATION.downloadModelType)).toBe('image')
  })
  it.each([undefined, null, 'text', 'Image', {}, 8])('keeps unsupported navigation payloads on the text default: %s', value => {
    expect(resolveDownloadModelType(value)).toBe('text')
  })
  it('wires the route into the first downloader render, without mount-time text results', () => {
    const root = join(__dirname, '../src/renderer/src')
    const app = readFileSync(join(root, 'App.tsx'), 'utf8')
    const download = readFileSync(join(root, 'components/sessions/DownloadTab.tsx'), 'utf8')
    expect(app).toContain('setDownloadModelType(resolveDownloadModelType(detail.downloadModelType))')
    expect(app).toContain('initialModelType={downloadModelType}')
    expect(download).toContain('useState<DownloadModelType>(initialModelType)')
  })
  it('removes preset downloads and preserves local inspection plus explicit load', () => {
    const picker = readFileSync(join(__dirname, '../src/renderer/src/components/image/ImageModelPicker.tsx'), 'utf8')
    expect(picker).toContain('detail: IMAGE_MODEL_DISCOVERY_NAVIGATION')
    expect(picker).toContain('window.api.image.inspectLocalModel(customPath.trim())')
    expect(picker).toContain('data-vmlx-control="image-load-folder"')
    for (const obsolete of ['NAMED_MODELS', 'checkImageModel(', 'downloadImageModel(', 'onDownloadProgress(']) {
      expect(picker).not.toContain(obsolete)
    }
  })
})
