import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, unlinkSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { describe, expect, it } from 'vitest'
import { bundledDsparkDefault, configDeclaresDspark, dsparkUsesTextLane, findBundledDsparkDrafter } from '../src/main/dsparkBundle'
import { buildCommandPreview } from '../src/renderer/src/components/sessions/SessionSettings'
import { DEFAULT_CONFIG } from '../src/renderer/src/components/sessions/SessionConfigForm'
import { addDraftAdmissionWeights, writeDsv41Bundle, writeMinimaxDsparkBundle, type Dsv41Layout } from './helpers/draftAdmissionFixture'

function bundle(drafter: Record<string, unknown> | null, dir = 'dspark', stamp?: Record<string, unknown>): string {
  const root = mkdtempSync(join(tmpdir(), 'dspark-bundle-'))
  writeFileSync(join(root, 'config.json'), JSON.stringify({ model_type: 'minicpmv4_7' }))
  if (stamp) writeFileSync(join(root, 'jang_config.json'), JSON.stringify({ speculative: stamp }))
  if (drafter) {
    mkdirSync(join(root, dir))
    writeFileSync(join(root, dir, 'config.json'), JSON.stringify(drafter))
    addDraftAdmissionWeights(root, dir)
  }
  return root
}

describe('bundled DSpark drafter detection', () => {
  it('rejects config-only, truncated and target-incompatible drafts', () => {
    const root = bundle({ architectures: ['Qwen3DSparkModel'] })
    const file = join(root, 'dspark', 'model.safetensors')
    const valid = readFileSync(file)
    unlinkSync(file)
    expect(findBundledDsparkDrafter(root)).toBeNull()
    writeFileSync(file, valid.subarray(0, 64))
    expect(findBundledDsparkDrafter(root)).toBeNull()
    writeFileSync(file, valid)
    const cfg = JSON.parse(readFileSync(join(root, 'config.json'), 'utf8'))
    cfg.text_config.hidden_size = 64
    writeFileSync(join(root, 'config.json'), JSON.stringify(cfg))
    expect(findBundledDsparkDrafter(root)).toBeNull()
  })
  it('recognizes the openbmb (MiniCPM-V-4.7) and SpecForge (Naive N0.5) drafter configs', () => {
    expect(configDeclaresDspark({ architectures: ['Qwen3DSparkModel'] })).toBe(true)
    expect(configDeclaresDspark({ architectures: ['DSparkDraftModel'] })).toBe(true)
    expect(configDeclaresDspark({ dflash_config: { projector_type: 'dspark' } })).toBe(true)
    expect(configDeclaresDspark({ architectures: ['DFlash2DraftModel'] })).toBe(false)
    expect(configDeclaresDspark(null)).toBe(false)
  })

  it('finds <bundle>/dspark and ignores bundles without one', () => {
    const root = bundle({ architectures: ['Qwen3DSparkModel'] })
    expect(findBundledDsparkDrafter(root)).toBe(join(root, 'dspark'))
    expect(findBundledDsparkDrafter(bundle(null))).toBeNull()
    expect(findBundledDsparkDrafter(bundle({ architectures: ['DFlash2DraftModel'] }, 'dflash2'))).toBeNull()
    expect(findBundledDsparkDrafter('')).toBeNull()
  })

  it('defaults ON whenever the bundle ships a drafter, whatever the stamp says', () => {
    const drafter = { architectures: ['Qwen3DSparkModel'] }
    expect(bundledDsparkDefault(bundle(drafter, 'dspark', { method: 'dspark', default: false }))).toBe(true)
    expect(bundledDsparkDefault(bundle(drafter, 'dspark', { method: 'dspark', default: true }))).toBe(true)
    expect(bundledDsparkDefault(bundle(drafter))).toBe(true)
    expect(bundledDsparkDefault(bundle(null, 'dspark', { method: 'dspark', default: true }))).toBe(false)   // shipped without
  })

  it('uses the engine-declared path and does not select unrelated sidecars', () => {
    const drafter = { architectures: ['Qwen3DSparkModel'] }
    const declared = bundle(drafter, 'custom-draft', { method: 'dspark', drafter_path: 'custom-draft' })
    expect(findBundledDsparkDrafter(declared)).toBe(join(declared, 'custom-draft'))
    expect(findBundledDsparkDrafter(bundle(drafter, 'custom-draft'))).toBeNull()
    expect(findBundledDsparkDrafter(bundle(drafter, 'dspark', { method: 'dspark', drafter_path: 'missing' }))).toBeNull()
    expect(findBundledDsparkDrafter(bundle(drafter, 'dspark', { method: 'dflash2' }))).toBeNull()
  })

  it('does not advertise DSpark for targets the engine does not support', () => {
    const root = bundle({ architectures: ['DSparkDraftModel'] })
    writeFileSync(join(root, 'config.json'), JSON.stringify({ model_type: 'mimo' }))
    expect(findBundledDsparkDrafter(root)).toBeNull()
    expect(bundledDsparkDefault(root)).toBe(false)
  })
})

function minimaxBundle(modelType = 'minimax_m3_vl'): string {
  const root = mkdtempSync(join(tmpdir(), 'dspark-m31-'))
  writeMinimaxDsparkBundle(root, 'dspark', modelType)
  return root
}

describe('MiniMax-M3 / M3.1 bundled DSpark drafter (text lane)', () => {
  it('finds the M3.1 drafter (DSparkMiniMaxDraftModel, release tensor names, MXFP8 linears) for both model types', () => {
    for (const modelType of ['minimax_m3_vl', 'minimax_m3']) {
      const root = minimaxBundle(modelType)
      expect(configDeclaresDspark({ architectures: ['DSparkMiniMaxDraftModel'] })).toBe(true)
      expect(findBundledDsparkDrafter(root)).toBe(join(root, 'dspark'))
      expect(bundledDsparkDefault(root)).toBe(true)
      expect(dsparkUsesTextLane(root)).toBe(true)
    }
  })

  it('MiniCPM-V-4.7 stays on the single-request lane', () => {
    expect(dsparkUsesTextLane(bundle({ architectures: ['Qwen3DSparkModel'] }))).toBe(false)
    expect(dsparkUsesTextLane('')).toBe(false)
  })

  it('rejects an M3.1 drafter whose taps, geometry, weights or shard index do not fit the target', () => {
    const edit = (root: string, file: string, fn: (cfg: any) => void) => {
      const cfg = JSON.parse(readFileSync(join(root, file), 'utf8'))
      fn(cfg)
      writeFileSync(join(root, file), JSON.stringify(cfg))
    }
    let root = minimaxBundle()
    edit(root, 'dspark/config.json', cfg => { cfg.text_config.dspark_target_layer_ids = [1, 4] })
    expect(findBundledDsparkDrafter(root)).toBeNull()
    root = minimaxBundle()
    edit(root, 'dspark/config.json', cfg => { cfg.text_config.hidden_size = 64 })
    expect(findBundledDsparkDrafter(root)).toBeNull()
    root = minimaxBundle()
    edit(root, 'dspark/config.json', cfg => { cfg.text_config.num_hidden_layers = 2 })   // layer 1 has no weights
    expect(findBundledDsparkDrafter(root)).toBeNull()
    root = minimaxBundle()
    edit(root, 'dspark/model.safetensors.index.json', idx => { idx.weight_map['language_model.model.dspark.fc.weight'] = 'other.safetensors' })
    expect(findBundledDsparkDrafter(root)).toBeNull()
    root = minimaxBundle()
    unlinkSync(join(root, 'dspark', 'model-dspark-00001.safetensors'))
    expect(findBundledDsparkDrafter(root)).toBeNull()
  })

  it('a declared non-DSpark method or a missing drafter keeps M3 on plain decoding', () => {
    const root = minimaxBundle()
    writeFileSync(join(root, 'jang_config.json'), JSON.stringify({ speculative: { method: 'mtp' } }))
    expect(findBundledDsparkDrafter(root)).toBeNull()
    const plain = mkdtempSync(join(tmpdir(), 'm3-plain-'))
    writeFileSync(join(plain, 'config.json'), JSON.stringify({ model_type: 'minimax_m3_vl' }))
    expect(findBundledDsparkDrafter(plain)).toBeNull()
  })
})

function dsv41Bundle(layout: Dsv41Layout): string {
  const root = mkdtempSync(join(tmpdir(), 'dspark-dsv41-'))
  writeDsv41Bundle(root, layout)
  return root
}

describe('DeepSeek-V4.1 bundled dSpark draft (engine loader admission)', () => {
  it('finds the draft of each release layout: JANGHT1.4 and JANG_2K ship dspark/dspark.safetensors, no dspark/config.json', () => {
    for (const layout of ['janght', 'jang2k'] as const) {
      const root = dsv41Bundle(layout)
      expect(findBundledDsparkDrafter(root)).toBe(join(root, 'dspark'))
      expect(bundledDsparkDefault(root)).toBe(true)
      expect(dsparkUsesTextLane(root)).toBe(true)
    }
    const rootFile = dsv41Bundle('root-file')
    expect(findBundledDsparkDrafter(rootFile)).toBe(rootFile)
    // A non-JANGHT bundle reads the release mtp.* tensors from its own shards (load_dspark's source path).
    const shards = dsv41Bundle('source-shards')
    expect(findBundledDsparkDrafter(shards)).toBe(shards)
  })

  it('serves AR (no control) when the bundle declares no draft or ships none', () => {
    expect(findBundledDsparkDrafter(dsv41Bundle('none'))).toBeNull()
    expect(bundledDsparkDefault(dsv41Bundle('none'))).toBe(false)
    // Declared but the draft file is gone: JANGHT has no shard fallback (load_dspark raises, the engine serves AR).
    const janght = dsv41Bundle('janght')
    unlinkSync(join(janght, 'dspark', 'dspark.safetensors'))
    expect(findBundledDsparkDrafter(janght)).toBeNull()
    // A JANG_2K-style (affine) bundle without the file and without mtp.* in its shards has nothing to load either.
    const jang2k = dsv41Bundle('jang2k')
    unlinkSync(join(jang2k, 'dspark', 'dspark.safetensors'))
    expect(findBundledDsparkDrafter(jang2k)).toBeNull()
    // The file alone does not declare the draft: the loader gates on runtime.bundle_has_mtp first.
    const undeclared = dsv41Bundle('janght')
    const cfg = JSON.parse(readFileSync(join(undeclared, 'config.json'), 'utf8'))
    cfg.runtime.bundle_has_mtp = false
    writeFileSync(join(undeclared, 'config.json'), JSON.stringify(cfg))
    expect(findBundledDsparkDrafter(undeclared)).toBeNull()
  })

  it('a DeepSeek-V4 (V4.0) bundle with the same files gets no DSpark control: V4.0 has no dSpark runtime', () => {
    const root = dsv41Bundle('janght')
    const cfg = JSON.parse(readFileSync(join(root, 'config.json'), 'utf8'))
    cfg.model_type = 'deepseek_v4'
    writeFileSync(join(root, 'config.json'), JSON.stringify(cfg))
    expect(findBundledDsparkDrafter(root)).toBeNull()
    expect(dsparkUsesTextLane(root)).toBe(false)
  })
})

const flat = (cmd: string) => cmd.replace(/\s*\\\n\s*/g, ' ')

describe('command preview with a bundled DSpark drafter', () => {
  const base: any = { ...DEFAULT_CONFIG, continuousBatching: true }
  const detected = { family: 'minicpm-v4.7', isMultimodal: true }

  const on = { default: true }
  const off = { default: false }
  const preview = (cfg: any, dspark: any) => flat(buildCommandPreview('/m', { ...base, ...cfg }, detected, '', false, undefined, dspark))

  it('unset follows the stamp: stamped on = single-request lane with --dspark auto', () => {
    const cmd = preview({}, on)
    expect(cmd).toContain('--dspark auto')
    expect(cmd).toContain('--no-continuous-batching')
    expect(cmd).not.toContain('--continuous-batching ')
  })

  it('unset follows the stamp: stamped off = the configured batching with --dspark auto', () => {
    const cmd = preview({}, off)
    expect(cmd).toContain('--dspark auto')
    expect(cmd).toContain('--continuous-batching')
  })

  it('an explicit choice wins over the stamp', () => {
    const forcedOn = preview({ useBundledDspark: true }, off)
    expect(forcedOn).toContain('--dspark on')
    expect(forcedOn).toContain('--no-continuous-batching')
    const forcedOff = preview({ useBundledDspark: false }, on)
    expect(forcedOff).toContain('--dspark off')
    expect(forcedOff).toContain('--continuous-batching')
  })

  it('bundles without a drafter get no --dspark flag', () => {
    expect(preview({}, false)).not.toContain('--dspark')
  })

  it('DeepSeek-V4.1: --dspark auto / on / off on the DSV4 lane (--max-num-seqs 1, batching on), nothing else changes', () => {
    const v41 = { family: 'deepseek-v4', isMultimodal: false }
    const run = (cfg: any, dspark: any) => flat(buildCommandPreview('/m', { ...base, ...cfg }, v41, '', false, undefined, dspark))
    const strip = (cmd: string) => cmd.replace(/ --dspark (auto|on|off)/, '')
    const lane = { default: true, textLane: true }
    expect(run({}, false)).not.toContain('--dspark')
    for (const [cfg, mode] of [[{}, 'auto'], [{ useBundledDspark: true }, 'on'], [{ useBundledDspark: false }, 'off'],
      // A stale draft model does not turn V4.1's own draft off: the DSV4 engine drops --speculative-model.
      [{ speculativeModel: '/models/other-draft' }, 'auto']] as const) {
      const cmd = run(cfg, lane)
      expect(cmd).toContain(`--dspark ${mode}`)
      expect(cmd).toContain('--max-num-seqs 1')
      expect(cmd).toContain('--continuous-batching')
      expect(cmd).not.toContain('--no-continuous-batching')
      expect(strip(cmd)).toBe(strip(run(cfg, false)))
    }
  })

  it('a text-lane (M3.1) drafter only selects --dspark: batching and every cache flag stay as without a drafter', () => {
    const m3 = { family: 'minimax-m3', isMultimodal: false }
    const run = (cfg: any, dspark: any) => flat(buildCommandPreview('/m', { ...base, ...cfg }, m3, '', false, undefined, dspark))
    const strip = (cmd: string) => cmd.replace(/ --dspark (auto|on|off)/, '')
    const lane = { default: true, textLane: true }
    const none = run({}, false)
    expect(none).toContain('--continuous-batching')
    expect(none).not.toContain('--dspark')
    for (const [cfg, mode] of [[{}, 'auto'], [{ useBundledDspark: true }, 'on'], [{ useBundledDspark: false }, 'off']] as const) {
      const cmd = run(cfg, lane)
      expect(cmd).toContain(`--dspark ${mode}`)
      expect(cmd).not.toContain('--no-continuous-batching')
      expect(strip(cmd)).toBe(strip(run(cfg, false)))
    }
  })
})
