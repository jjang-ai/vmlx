import { writeFileSync, readFileSync } from 'fs'
import { join } from 'path'

function addWeights(target: string, child: string): void {
  const root = JSON.parse(readFileSync(join(target, 'config.json'), 'utf8'))
  writeFileSync(join(target, 'config.json'), JSON.stringify({ ...root, hidden_size: 32, vocab_size: 64, num_hidden_layers: 2 }))
  const cfg = JSON.parse(readFileSync(join(target, child, 'config.json'), 'utf8'))
  writeFileSync(join(target, child, 'config.json'), JSON.stringify({ ...cfg, hidden_size: 32, vocab_size: 64, num_hidden_layers: 1, dflash_config: { target_layer_ids: [0] } }))
  const header = Buffer.from(JSON.stringify({
    'fc.weight': { dtype: 'F16', shape: [32, 32], data_offsets: [0, 2048] },
    'layers.0.self_attn.q_proj.weight': { dtype: 'F16', shape: [32, 32], data_offsets: [2048, 4096] }
  }))
  const length = Buffer.alloc(8)
  length.writeBigUInt64LE(BigInt(header.length))
  writeFileSync(join(target, child, 'model.safetensors'), Buffer.concat([length, header, Buffer.alloc(4096)]))
}

import { describe, expect, it } from 'vitest'
import {
  QWEN38_27B_DFLASH2_REPO,
  configDeclaresDflash2,
  isQwen38Dense27b,
  looksLikeDflash2Drafter,
} from '../src/shared/dflash2Drafter'

describe('DFlash2 drafter helpers', () => {
  it('detects Qwen3.8 27B bundles by name or path', () => {
    for (const id of [
      'Qwen3.8-27B-JANG_4D /Volumes/EricsLLMDrive/jangq-ai/Qwen3.8-27B-JANG_4D',
      'qwen3_8-27b-jang_2d',
      '/models/Qwen3.8-27B-JANG_6D',
      'Qwen3-8-27B-MLX-4bit',
    ]) {
      expect(isQwen38Dense27b(id)).toBe(true)
    }
  })

  it('keeps other families out (no published Qwen3.8 drafter for them)', () => {
    for (const id of [
      'Qwen3.6-27B-JANG_4M-CRACK',
      'Qwen3.8-Flash-Next-JANG_4M',
      '/Users/eric/q38fn-jangh/run_v5/build/bundle',
      'Allosaurus-v0.1-125B-A6B-JANGH2',
      '',
      undefined,
    ]) {
      expect(isQwen38Dense27b(id as string)).toBe(false)
    }
  })

  it('recognizes drafters by config architecture and by name', () => {
    expect(configDeclaresDflash2({ architectures: ['DFlash2DraftModel'] })).toBe(true)
    expect(configDeclaresDflash2({ architectures: ['Qwen3ForCausalLM'] })).toBe(false)
    expect(configDeclaresDflash2(null)).toBe(false)
    expect(looksLikeDflash2Drafter('/Volumes/EricsLLMDrive/z-lab/Qwen3.8-27B-DFlash2')).toBe(true)
    expect(looksLikeDflash2Drafter('mlx-community/small-draft-model')).toBe(false)
    expect(QWEN38_27B_DFLASH2_REPO).toBe('z-lab/Qwen3.8-27B-DFlash2')
  })
})

describe('bundled DFlash2 drafter detection', () => {
  it('finds <bundle>/dflash2 by config architecture and ignores bundles without one', async () => {
    const { mkdtempSync, mkdirSync, writeFileSync } = await import('fs')
    const { join } = await import('path')
    const { tmpdir } = await import('os')
    const { findBundledDflash2Drafter } = await import('../src/main/dflash2Bundle')
    const root = mkdtempSync(join(tmpdir(), 'vmlx-dflash2-'))
    const withDrafter = join(root, 'Qwen3.8-27B-JANG_4D')
    mkdirSync(join(withDrafter, 'dflash2'), { recursive: true })
    writeFileSync(join(withDrafter, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }))
    writeFileSync(join(withDrafter, 'dflash2', 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }))
    addWeights(withDrafter, 'dflash2')
    expect(findBundledDflash2Drafter(withDrafter)).toBe(join(withDrafter, 'dflash2'))

    const without = join(root, 'Qwen3.8-27B-JANG_2D')
    mkdirSync(join(without, 'mtp_draft'), { recursive: true })
    writeFileSync(join(without, 'mtp_draft', 'config.json'), JSON.stringify({ architectures: ['Qwen3ForCausalLM'] }))
    expect(findBundledDflash2Drafter(without)).toBeNull()
    expect(findBundledDflash2Drafter(join(root, 'missing'))).toBeNull()
    expect(findBundledDflash2Drafter('')).toBeNull()
  })

  it("finds MiMo-V2.6's native DFlash drafter (<bundle>/dflash) only inside a mimo_v2 bundle", async () => {
    const { mkdtempSync, mkdirSync, writeFileSync } = await import('fs')
    const { join } = await import('path')
    const { tmpdir } = await import('os')
    const { findBundledDflash2Drafter, isDflash2DrafterSelection } = await import('../src/main/dflash2Bundle')
    const root = mkdtempSync(join(tmpdir(), 'vmlx-mimo-dflash-'))
    const mimo = join(root, 'MiMo-V2.6-Flash-MOPD-JANGHT2.5')
    mkdirSync(join(mimo, 'dflash'), { recursive: true })
    mkdirSync(join(mimo, 'audio_tokenizer'), { recursive: true })
    writeFileSync(join(mimo, 'config.json'), JSON.stringify({ model_type: 'mimo_v2' }))
    writeFileSync(join(mimo, 'dflash', 'config.json'), JSON.stringify({ architectures: ['DFlashDraftModel'], model_type: 'qwen3' }))
    writeFileSync(join(mimo, 'audio_tokenizer', 'config.json'), JSON.stringify({ architectures: ['MiMoAudioTokenizer'] }))
    addWeights(mimo, 'dflash')
    expect(findBundledDflash2Drafter(mimo)).toBe(join(mimo, 'dflash'))
    expect(isDflash2DrafterSelection(join(mimo, 'dflash'))).toBe(true)
    // the same drafter architecture in another family's bundle is not run by the MiMo adapter
    const other = join(root, 'Other-Model')
    mkdirSync(join(other, 'dflash'), { recursive: true })
    writeFileSync(join(other, 'config.json'), JSON.stringify({ model_type: 'qwen3' }))
    writeFileSync(join(other, 'dflash', 'config.json'), JSON.stringify({ architectures: ['DFlashDraftModel'] }))
    expect(findBundledDflash2Drafter(other)).toBeNull()
  })
})


describe('explicit config-first selection shared by launch and settings', () => {
  it('recognizes renamed local drafters and rejects misleading folder names', async () => {
    const { mkdtempSync, mkdirSync, writeFileSync, rmSync } = await import('fs')
    const { join } = await import('path')
    const { tmpdir } = await import('os')
    const { isDflash2DrafterSelection } = await import('../src/main/dflash2Bundle')
    const root = mkdtempSync(join(tmpdir(), 'vmlx-selection-'))
    try {
      const renamed = join(root, 'Custom Drafter')
      const misleading = join(root, 'dflash2-not-a-drafter')
      for (const dir of [renamed, misleading]) mkdirSync(dir)
      writeFileSync(join(renamed, 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }))
      writeFileSync(join(misleading, 'config.json'), JSON.stringify({ architectures: ['Qwen3ForCausalLM'] }))
      expect(isDflash2DrafterSelection(renamed)).toBe(true)
      expect(isDflash2DrafterSelection(misleading)).toBe(false)
      expect(isDflash2DrafterSelection('z-lab/Qwen3.8-27B-DFlash2')).toBe(true)
      expect(isDflash2DrafterSelection('')).toBe(false)
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})


describe('bundled draft weight admission', () => {
  it('rejects absent, truncated and target-incompatible weights', async () => {
    const { mkdtempSync, mkdirSync, rmSync } = await import('fs')
    const { tmpdir } = await import('os')
    const { findBundledDflash2Drafter } = await import('../src/main/dflash2Bundle')
    const root = mkdtempSync(join(tmpdir(), 'draft-admission-'))
    try {
      mkdirSync(join(root, 'dflash2'))
      writeFileSync(join(root, 'config.json'), JSON.stringify({ model_type: 'qwen3_5' }))
      writeFileSync(join(root, 'dflash2', 'config.json'), JSON.stringify({ architectures: ['DFlash2DraftModel'] }))
      expect(findBundledDflash2Drafter(root)).toBeNull()
      addWeights(root, 'dflash2')
      const path = join(root, 'dflash2', 'model.safetensors')
      const complete = readFileSync(path)
      writeFileSync(path, complete.subarray(0, 64))
      expect(findBundledDflash2Drafter(root)).toBeNull()
      writeFileSync(path, complete)
      const cfg = JSON.parse(readFileSync(join(root, 'config.json'), 'utf8'))
      writeFileSync(join(root, 'config.json'), JSON.stringify({ ...cfg, hidden_size: 64 }))
      expect(findBundledDflash2Drafter(root)).toBeNull()
    } finally { rmSync(root, { recursive: true, force: true }) }
  })
})
