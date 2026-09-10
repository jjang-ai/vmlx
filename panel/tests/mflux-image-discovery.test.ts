import { describe, expect, it } from 'vitest'
import { isMfluxImageCandidate, mfluxImageSearchParams, verifyMfluxComponentExport } from '../src/shared/mfluxImageDiscovery'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const bundle = (id = 'OtherCreator/Qwen-Image-Edit-mflux-q8') => ({
  id, library_name: 'mflux', tags: ['mflux', 'safetensors'], pipeline_tag: 'image-to-image',
  siblings: [{ rfilename: 'transformer/0.safetensors' }, { rfilename: 'text_encoder/0.safetensors' }],
})
describe('mflux-only image discovery', () => {
  const untagged = () => ({ ...bundle('mflux-community/flux-1-schnell-mflux-q6'),
    library_name: undefined, tags: ['safetensors'], pipeline_tag: undefined, sha: 'a'.repeat(40),
    siblings: [...bundle().siblings, ...['transformer', 'text_encoder'].map(c => ({ rfilename: c + '/model.safetensors.index.json' }))] })
  const index = () => ({ metadata: { mflux_version: '0.18.0', quantization_level: '6' }, weight_map: { 'linear.weight': '0.safetensors' } })
  it('discovers untagged exporters only from pinned component metadata', async () => {
    const paths: string[] = []
    expect(isMfluxImageCandidate(untagged())).toBe(false)
    expect(await verifyMfluxComponentExport(untagged(), async path => { paths.push(path); return index() })).toBe(true)
    expect(paths).toHaveLength(2)
    expect(paths.every(p => p.includes('/resolve/' + 'a'.repeat(40) + '/'))).toBe(true)
  })
  it('rejects missing metadata, missing shards, unsafe revisions and incompatible formats', async () => {
    for (const bad of [{}, { ...index(), metadata: {} }, { ...index(), weight_map: { 'linear.weight': '../0.safetensors' } }, { ...index(), weight_map: { 'linear.weight': 'missing.safetensors' } }]) {
      expect(await verifyMfluxComponentExport(untagged(), async () => bad)).toBe(false)
    }
    expect(await verifyMfluxComponentExport({ ...untagged(), sha: 'main' }, async () => index())).toBe(false)
    expect(await verifyMfluxComponentExport({ ...untagged(), tags: ['gguf'] }, async () => index())).toBe(false)
    expect(await verifyMfluxComponentExport(untagged(), async () => { throw Error('network') })).toBe(false)
  })
  it.each(['OtherCreator/Qwen-Image-Edit-mflux-q8', 'OtherCreator/Z-Image-Turbo-mflux-4bit', 'dhairyashil/FLUX.1-schnell-mflux-8bit'])('admits declared supported mflux weights from any creator: %s', id => {
    expect(isMfluxImageCandidate(bundle(id))).toBe(true)
  })
  it('accepts renamed models from supported base metadata, not an author allow-list', () => {
    expect(isMfluxImageCandidate({ ...bundle('thirdparty/artwork'), cardData: { base_model: 'Qwen/Qwen-Image' } })).toBe(true)
    expect(isMfluxImageCandidate({ ...bundle('thirdparty/artwork'), tags: ['mflux', 'base_model:finetune:Qwen/Qwen-Image'] })).toBe(true)
  })
  it.each(['GGUF', 'EXL3', 'NVFP4', 'LoRA', 'ControlNet'])('excludes %s even if named or tagged mflux', format => {
    expect(isMfluxImageCandidate({ ...bundle(), tags: ['mflux', format.toLowerCase()] })).toBe(false)
    expect(isMfluxImageCandidate(bundle('OtherCreator/Qwen-Image-mflux-' + format))).toBe(false)
  })
  it('does not equate a filename, MLX library or image pipeline with mflux compatibility', () => {
    expect(isMfluxImageCandidate({ ...bundle(), library_name: 'mlx', tags: ['safetensors'] })).toBe(false)
    expect(isMfluxImageCandidate({ ...bundle(), pipeline_tag: 'text-generation' })).toBe(false)
    expect(isMfluxImageCandidate({ ...bundle(), siblings: [] })).toBe(false)
    expect(isMfluxImageCandidate({ ...bundle(), siblings: [{ rfilename: 'adapter.safetensors' }] })).toBe(false)
  })
  it('excludes mflux families unsupported by this image server', () => {
    expect(isMfluxImageCandidate(bundle('other/seedvr2-7b-8bit'))).toBe(false)
  })
  it('recognizes nested quant folders and tag-only mflux exports', () => {
    const m = { ...bundle(), library_name: 'mlx', siblings: [{ rfilename: 'q8/transformer/0.safetensors' }, { rfilename: 'q8/text_encoder/0.safetensors' }] }
    expect(isMfluxImageCandidate(m)).toBe(true)
    expect(isMfluxImageCandidate({ ...m, siblings: [{ rfilename: 'q4/transformer/0.safetensors' }, { rfilename: 'q8/text_encoder/0.safetensors' }] })).toBe(false)
  })
  it('builds the same constrained query for empty discovery and typed search', () => {
    for (const search of ['', 'qwen']) {
      const params = new URLSearchParams({ search, limit: '30', 'expand[]': 'safetensors' })
      const image = mfluxImageSearchParams(params)
      expect(image.get('filter')).toBe('mflux')
      expect(image.get('search')).toBe(search)
      expect(image.getAll('expand[]')).toEqual(expect.arrayContaining(['siblings', 'tags', 'cardData', 'safetensors']))
      expect(params.has('filter')).toBe(false)
    }
  })
  it('puts the same compatibility contract on picker and downloader in all locales', () => {
    const root = join(__dirname, '../src/renderer/src')
    for (const file of ['components/image/ImageModelPicker.tsx', 'components/sessions/DownloadTab.tsx']) {
      expect(readFileSync(join(root, file), 'utf8')).toContain("t('image.picker.mfluxCompatibility')")
    }
    for (const lang of ['en', 'es', 'ja', 'ko', 'zh']) {
      const strings = JSON.parse(readFileSync(join(root, 'i18n/locales', lang + '.json'), 'utf8')).image.picker
      expect(strings.mfluxCompatibility).toMatch(/mflux/)
      expect(strings.mfluxCompatibility).toMatch(/GGUF/)
      expect(strings.mfluxDiscovery).toMatch(/mflux/)
    }
  })
  it('keeps Image discovery out of text collections and rejects obsolete search responses', () => {
    const source = readFileSync(join(__dirname, '../src/renderer/src/components/sessions/DownloadTab.tsx'), 'utf8')
    expect(source).not.toContain('getCollectionModels(')
    expect(source).toContain("window.api.models.searchHF('', 'lastModified', 'desc', 'image')")
    expect(source).toContain("modelType === 'image' ? 'image:mflux'")
    expect(source).toContain('if (revision !== searchRevision.current) return')
    expect(source).toContain('collectionModels[collectionKey]')
  })
})
