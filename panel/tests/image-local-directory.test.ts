import { describe, expect, it } from 'vitest'
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import {
  inferImageQuantizeFromName,
  inspectLocalImageModel,
  looksLikeLocalPath,
  readBundleQuantization,
  readSafetensorsHeaderMetadata,
  resolveLocalImageModelDirectory,
  resolveImageModelForLocalDirectory,
  isVariantFolderName,
  describeVariants,
  localImageModelError,
  unmountedVolume,
  type LocalImageModelFs,
} from '../src/shared/imageLocalModel'

/** Write a safetensors file the way mflux does: LE u64 header length, JSON header with `__metadata__`, then bytes. */
function writeSafetensors(p: string, metadata: Record<string, string> | null): void {
  const header: Record<string, unknown> = { 'x.weight': { dtype: 'BF16', shape: [2, 2], data_offsets: [0, 8] } }
  if (metadata) header.__metadata__ = metadata
  const json = Buffer.from(JSON.stringify(header), 'utf8')
  const len = Buffer.alloc(8)
  len.writeBigUInt64LE(BigInt(json.length))
  writeFileSync(p, Buffer.concat([len, json, Buffer.alloc(8)]))
}

/** An mflux bundle: transformer/ + text_encoder/, shards stamped with (or without) a quantization level. */
function mfluxBundle(root: string, name: string, stored: number | null, mfluxVersion = '0.19.0'): string {
  const dir = join(root, name)
  mkdirSync(join(dir, 'transformer'), { recursive: true })
  mkdirSync(join(dir, 'text_encoder'), { recursive: true })
  writeFileSync(join(dir, 'README.md'), 'x')
  const meta: Record<string, string> = { mflux_version: mfluxVersion }
  if (stored !== null) meta.quantization_level = String(stored)
  writeSafetensors(join(dir, 'transformer', '0.safetensors'), meta)
  writeSafetensors(join(dir, 'transformer', '1.safetensors'), meta)
  return dir
}

const tmp = () => mkdtempSync(join(tmpdir(), 'vmlx-img-'))

describe('local image model directories (external drive bundles)', () => {
  it('previews a q8 edit subfolder with its own adapter, task and defaults', () => {
    const parent = join(tmp(), 'Qwen-Image-Edit-mflux')
    mkdirSync(parent)
    const folder = mfluxBundle(parent, 'q8', 8, '0.17.4')
    expect(inspectLocalImageModel(folder)).toMatchObject({
      success: true, path: folder, quantize: 8, quantizeSource: 'metadata',
      model: { id: 'qwen-image-edit', category: 'edit', mfluxClass: 'QwenImageEdit', steps: 28, guidance: 4 },
    })
  })

  it('does not silently select q4 from a variant root in automatic mode', () => {
    const parent = join(tmp(), 'Qwen-Image-Edit-mflux')
    mkdirSync(parent)
    mfluxBundle(parent, 'q4', 4)
    mfluxBundle(parent, 'q8', 8)
    expect(inspectLocalImageModel(parent)).toMatchObject({ success: false, errorCode: 'variants' })
  })

  it('uses declared pipeline metadata ahead of a misleading folder name', () => {
    const folder = mfluxBundle(tmp(), 'FLUX.1-schnell', 8)
    writeFileSync(join(folder, 'model_index.json'), JSON.stringify({ _class_name: 'QwenImageEditPipeline' }))
    expect(inspectLocalImageModel(folder)).toMatchObject({ success: true, model: { mfluxClass: 'QwenImageEdit', category: 'edit' } })
  })

  it('leaves unknown architecture unresolved rather than defaulting to Flux1', () => {
    const folder = mfluxBundle(tmp(), 'unknown-export', 6)
    expect(inspectLocalImageModel(folder)).toMatchObject({ success: true, quantize: 6, model: undefined })
  })
  it('reads the safetensors header metadata the way mflux writes it', () => {
    const root = tmp()
    const f = join(root, 'a.safetensors')
    writeSafetensors(f, { mflux_version: '0.6.2', quantization_level: '8' })
    expect(readSafetensorsHeaderMetadata(f)).toEqual({ mflux_version: '0.6.2', quantization_level: '8' })
    const g = join(root, 'b.safetensors')
    writeSafetensors(g, null)
    expect(readSafetensorsHeaderMetadata(g)).toBeNull()
    expect(readSafetensorsHeaderMetadata(join(root, 'missing.safetensors'))).toBeNull()
  })

  it('reads the precision from the folder name only as a hint', () => {
    expect(inferImageQuantizeFromName('FLUX.1-schnell-mflux-8bit')).toBe(8)
    expect(inferImageQuantizeFromName('/Volumes/EricsLLMDrive/image/Qwen-Image-mflux-6bit')).toBe(6)
    expect(inferImageQuantizeFromName('q8')).toBe(8)
    expect(inferImageQuantizeFromName('FLUX.2-klein-9B')).toBeNull()
  })

  it('treats only filesystem-looking strings as local paths', () => {
    expect(looksLikeLocalPath('/Volumes/EricsLLMDrive/image/FLUX.1-schnell-mflux-8bit')).toBe(true)
    expect(looksLikeLocalPath('~/models/x')).toBe(true)
    expect(looksLikeLocalPath('black-forest-labs/FLUX.1-schnell')).toBe(false)
    expect(looksLikeLocalPath('schnell')).toBe(false)
    expect(resolveLocalImageModelDirectory('black-forest-labs/FLUX.1-schnell', 4)).toBeNull()
  })

  it('the bundle metadata decides the precision, not the picker (8-bit bundle, picker said 4)', () => {
    const dir = mfluxBundle(tmp(), 'FLUX.1-schnell-mflux-8bit', 8, '0.6.2')
    expect(resolveLocalImageModelDirectory(dir, 4)).toEqual({ kind: 'model', path: dir, quantize: 8, quantizeSource: 'metadata', mfluxVersion: '0.6.2' })
  })

  it('the bundle metadata also beats a misleading folder name', () => {
    // folder says 4bit, shards say 8 — the shards are what loads (mflux ignores a conflicting request)
    const dir = mfluxBundle(tmp(), 'model-4bit', 8)
    expect(readBundleQuantization(dir)).toEqual({ bits: 8, source: 'metadata', mfluxVersion: '0.19.0' })
    expect(resolveLocalImageModelDirectory(dir, 0)?.kind === 'model' && resolveLocalImageModelDirectory(dir, 0)).toMatchObject({ quantize: 8, quantizeSource: 'metadata' })
  })

  it('a full-precision bundle keeps the requested precision as an on-the-fly quantization, or none', () => {
    const dir = mfluxBundle(tmp(), 'FLUX.2-klein-9B', null)
    expect(resolveLocalImageModelDirectory(dir, 4)).toEqual({ kind: 'model', path: dir, quantize: 4, quantizeSource: 'requested', mfluxVersion: '0.19.0' })
    expect(resolveLocalImageModelDirectory(dir, 0)).toEqual({ kind: 'model', path: dir, quantize: null, quantizeSource: null, mfluxVersion: '0.19.0' })
  })

  it('falls back to config.json quantization_config.bits, then to the folder name', () => {
    const root = tmp()
    const cfgDir = join(root, 'some-diffusers-model'); mkdirSync(cfgDir)
    writeFileSync(join(cfgDir, 'model_index.json'), '{"_class_name": "FluxPipeline"}')
    writeFileSync(join(cfgDir, 'config.json'), '{"quantization_config": {"bits": 6}}')
    expect(readBundleQuantization(cfgDir)).toEqual({ bits: 6, source: 'config', mfluxVersion: null })
    const nameDir = join(root, 'plain-model-4bit'); mkdirSync(join(nameDir, 'transformer'), { recursive: true }); mkdirSync(join(nameDir, 'vae'))
    expect(readBundleQuantization(nameDir)).toEqual({ bits: 4, source: 'name', mfluxVersion: null })
    expect(resolveLocalImageModelDirectory(nameDir, 8)).toMatchObject({ kind: 'model', quantize: 4, quantizeSource: 'name' })
  })

  it('picks the variant whose stored precision matches, or lists the variants', () => {
    const parent = join(tmp(), 'Qwen-Image-Edit-mflux'); mkdirSync(parent)
    writeFileSync(join(parent, 'README.md'), 'x')
    for (const [name, bits] of [['q3', 3], ['q4', 4], ['q8', 8]] as const) mfluxBundle(parent, name, bits, '0.17.4')
    expect(resolveLocalImageModelDirectory(parent, 8)).toMatchObject({ kind: 'model', path: join(parent, 'q8'), quantize: 8, quantizeSource: 'metadata', mfluxVersion: '0.17.4' })
    const none = resolveLocalImageModelDirectory(parent, 6)
    expect(none?.kind).toBe('variants')
    if (none?.kind === 'variants') {
      expect(none.variants.map((v) => [v.name, v.quantize])).toEqual([['q3', 3], ['q4', 4], ['q8', 8]])
      expect(describeVariants(none)).toContain('q8 (8-bit)')
      const err = localImageModelError(none)
      expect(err.code).toBe('variants')
      expect(err.params).toEqual({ name: 'Qwen-Image-Edit-mflux', variants: 'q3 (3-bit), q4 (4-bit), q8 (8-bit)' })
    }
  })

  it('reports a directory that holds no model instead of pretending it needs a download', () => {
    const dir = join(tmp(), 'empty'); mkdirSync(dir)
    const res = resolveLocalImageModelDirectory(dir, 4)
    expect(res).toEqual({ kind: 'not-a-model-directory', path: dir })
    expect(localImageModelError(res as any).code).toBe('notAModelDirectory')
  })

  it('tells an unmounted external volume apart from a deleted folder', () => {
    const gone = join(tmp(), 'deleted-model')
    const res = resolveLocalImageModelDirectory(gone, 4)
    expect(res).toEqual({ kind: 'path-missing', path: gone, volume: null })
    expect(localImageModelError(res as any).code).toBe('pathMissing')

    const fakeFs: LocalImageModelFs = {
      existsSync: () => false,
      isDirectory: (p) => p === '/Volumes/Mounted',
      readdirSync: () => [],
      readSafetensorsMetadata: () => null,
      readJson: () => null,
    }
    expect(unmountedVolume('/Volumes/EricsLLMDrive/image/x', fakeFs)).toBe('EricsLLMDrive')
    expect(unmountedVolume('/Volumes/Mounted/image/x', fakeFs)).toBeNull()
    expect(unmountedVolume('/Users/eric/models/x', fakeFs)).toBeNull()
    const unplugged = resolveLocalImageModelDirectory('/Volumes/EricsLLMDrive/image/FLUX.1-schnell-mflux-8bit', 4, fakeFs)
    expect(unplugged).toEqual({ kind: 'path-missing', path: '/Volumes/EricsLLMDrive/image/FLUX.1-schnell-mflux-8bit', volume: 'EricsLLMDrive' })
    const err = localImageModelError(unplugged as any)
    expect(err.code).toBe('volumeUnavailable')
    expect(err.params.volume).toBe('EricsLLMDrive')
  })

  it('the image server start handler resolves the target BEFORE stopping the running server, and keeps registrations', () => {
    const src = readFileSync(join(__dirname, '..', 'src', 'main', 'ipc', 'image.ts'), 'utf8')
    const handler = src.slice(src.indexOf("ipcMain.handle('image:startServer'"), src.indexOf("ipcMain.handle('image:stopServer'"))
    const resolveAt = handler.indexOf('resolveLocalImageModelDirectory(modelName')
    expect(resolveAt).toBeGreaterThan(-1)
    // local directory first, registry second
    expect(resolveAt).toBeLessThan(handler.indexOf('db.getImageModelPath('))
    // validate first, stop second: a rejected folder must leave the running server untouched
    expect(handler.indexOf('db.getImageModelPath(')).toBeLessThan(handler.indexOf('sessionManager.stopSession(activeImageSessionId)'))
    expect(handler).toContain('serverKept: true')
    // a registration whose folder is temporarily missing (drive unplugged) is kept, not deleted
    expect(handler).not.toContain('deleteImageModelPath(')
    expect(handler).toContain('Using local model directory')
    // the registry fallback and its literals stay (pinned elsewhere)
    expect(handler).toContain('findDownloadedImageModelPath(modelName')
    expect(handler).toContain('not downloaded. Use the Download button first.')
    // every failure carries a stable code the renderer translates
    for (const code of ['variants', 'notAModelDirectory', 'volumeUnavailable', 'pathMissing', 'storedVolumeUnavailable', 'notDownloaded']) {
      expect(handler.includes(`'${code}'`) || handler.includes(`localImageModelError(`)).toBe(true)
    }
  })

  it('the renderer no longer stops the running server before the main process has validated the new folder', () => {
    const tab = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'components', 'image', 'ImageTab.tsx'), 'utf8')
    const select = tab.slice(tab.indexOf('const handleModelSelect'), tab.indexOf('window.api.image.startServer('))
    expect(select).not.toContain('window.api.image.stopServer()')
    expect(tab).toContain('result.serverKept')
    expect(tab).toContain('image.server.errors.')
  })

  it('every locale carries the image server error strings with their placeholders', () => {
    const locales = join(__dirname, '..', 'src', 'renderer', 'src', 'i18n', 'locales')
    const en = JSON.parse(readFileSync(join(locales, 'en.json'), 'utf8'))
    const errors = en.image.server.errors
    expect(Object.keys(errors).sort()).toEqual(['notAModelDirectory', 'notDownloaded', 'pathMissing', 'storedVolumeUnavailable', 'variants', 'volumeUnavailable'])
    for (const loc of ['es', 'ja', 'ko', 'zh']) {
      const other = JSON.parse(readFileSync(join(locales, `${loc}.json`), 'utf8')).image.server.errors
      for (const key of Object.keys(errors)) {
        const want = (errors[key].match(/\{\w+\}/g) || []).sort()
        const got = (other[key].match(/\{\w+\}/g) || []).sort()
        expect(got, `${loc}.${key} placeholders`).toEqual(want)
      }
    }
  })
})

describe('low-precision edit variants: warn and offer the better sibling', () => {
  it('reports sibling variants and picks a >= 8-bit alternative for a 4-bit edit model', async () => {
    const { editPrecisionAlternative } = await import('../src/shared/imageLocalModel')
    const parent = join(mkdtempSync(join(tmpdir(), 'vmlx-img-')), 'Qwen-Image-Edit-mflux'); mkdirSync(parent)
    for (const [name, bits] of [['q3', 3], ['q4', 4], ['q8', 8]] as const) mfluxBundle(parent, name, bits, '0.17.4')
    const chosen = resolveLocalImageModelDirectory(join(parent, 'q4'), 4)
    expect(chosen?.kind).toBe('model')
    if (chosen?.kind === 'model') {
      expect((chosen.siblings || []).map((v) => v.name)).toEqual(['q3', 'q8'])
      expect(editPrecisionAlternative(chosen)?.name).toBe('q8')
      const viaFolder = resolveLocalImageModelDirectory(parent, 4)
      expect(viaFolder?.kind === 'model' && editPrecisionAlternative(viaFolder)?.name).toBe('q8')
      const eight = resolveLocalImageModelDirectory(join(parent, 'q8'), 8)
      expect(eight?.kind === 'model' && editPrecisionAlternative(eight)).toBeNull()
    }
  })
  it('the start handler warns on a <= 4-bit edit class and the tab offers the alternative', () => {
    const src = readFileSync(join(__dirname, '..', 'src', 'main', 'ipc', 'image.ts'), 'utf8')
    // the measured claim is scoped to the class it was measured on; other edit classes get the neutral note
    expect(src).toContain("warningCode = mfluxClass === 'QwenImageEdit' ? 'editLowPrecision' : 'editLowPrecisionUntested'")
    expect(src).toContain('localDir.quantize <= 4')
    const tab = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'components', 'image', 'ImageTab.tsx'), 'utf8')
    expect(tab).toContain('data-vmlx-control="image-use-alternative"')
    expect(tab).toContain('image.server.warnings.')
    const locales = join(__dirname, '..', 'src', 'renderer', 'src', 'i18n', 'locales')
    for (const loc of ['en', 'es', 'ja', 'ko', 'zh']) {
      const w = JSON.parse(readFileSync(join(locales, `${loc}.json`), 'utf8')).image.server.warnings
      expect(w.editLowPrecision).toContain('{bits}')
      expect(w.editLowPrecision).toContain('Qwen')
      expect(w.editLowPrecisionUntested).toContain('{bits}')
      expect(w.useAlternative).toContain('{name}')
    }
  })
})

describe('a precision variant folder resolves its model through the bundle root', () => {
  it('names like q8 / 8bit / int4 are precision names, not model names', () => {
    for (const n of ['q8', 'Q4', '8bit', '8-bit', '4_bit', 'int4', 'fp8']) expect(isVariantFolderName(n)).toBe(true)
    for (const n of ['Qwen-Image-Edit-mflux', 'FLUX.1-schnell-mflux-4bit', 'transformer']) expect(isVariantFolderName(n)).toBe(false)
  })

  it('the q8 sibling of Qwen-Image-Edit-mflux carries the QwenImageEdit class and canonical name', () => {
    const def = resolveImageModelForLocalDirectory('/Volumes/EricsLLMDrive/image/Qwen-Image-Edit-mflux/q8')
    expect(def?.mfluxClass).toBe('QwenImageEdit')
    expect(def?.id).toBe('qwen-image-edit')
    // a folder that names the model itself resolves directly
    expect(resolveImageModelForLocalDirectory('/models/FLUX.1-schnell-mflux-4bit')?.id).toBe('schnell')
    // an unknown bundle root stays unresolved rather than guessing
    expect(resolveImageModelForLocalDirectory('/models/Some-Unknown-Bundle/q8')).toBeUndefined()
  })

  it('the start handler resolves the class from the local folder and the warning action reuses the server settings', () => {
    const main = readFileSync(join(__dirname, '..', 'src', 'main', 'ipc', 'image.ts'), 'utf8')
    expect(main).toContain("resolveImageModelForLocalDirectory(modelPath)")
    const tab = readFileSync(join(__dirname, '..', 'src', 'renderer', 'src', 'components', 'image', 'ImageTab.tsx'), 'utf8')
    expect(tab).toContain("handleModelSelect(alt.alternativePath!, alt.alternativeBits, 'edit', serverSettingsRef.current)")
    expect(tab).toContain("serverSettingsRef.current = serverSettings")
    // a switch over a live server re-arms readiness on the new session (the old session's stopped event had left the tab at Stopped)
    expect(tab.split("serverSettingsRef.current = serverSettings")[1]).toContain("setServerStatus('starting')")
  })
})
