import { afterEach, describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { detectModelConfigFromDir } from '../src/main/model-config-registry'

const createdDirs: string[] = []

function makeModelDir(config: Record<string, unknown>, jangConfig?: Record<string, unknown>): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-model-config-'))
  createdDirs.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify(config, null, 2))
  if (jangConfig !== undefined) {
    writeFileSync(join(dir, 'jang_config.json'), JSON.stringify(jangConfig, null, 2))
  }
  return dir
}

afterEach(() => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir) rmSync(dir, { recursive: true, force: true })
  }
})

describe('detectModelConfigFromDir JANG multimodal detection', () => {
  it('detects ZAYA as tools-only CCA hybrid without reasoning parser', () => {
    const dir = makeModelDir(
      { model_type: 'zaya' },
      {
        cache_subtype: 'zaya_cca',
        capabilities: {
          family: 'zaya',
          tool_parser: 'zaya_xml',
          reasoning_parser: 'qwen3',
          supports_thinking: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('zaya')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(false)
    expect(detected.toolParser).toBe('zaya_xml')
    expect(detected.reasoningParser).toBeUndefined()
  })

  it('detects Ling/Bailing hybrid with parser and hybrid cache defaults', () => {
    const dir = makeModelDir(
      {
        model_type: 'bailing_hybrid',
        num_hidden_layers: 32,
        layer_group_size: 8,
      },
      {
        capabilities: {
          family: 'bailing_hybrid',
          tool_parser: 'deepseek',
          reasoning_parser: 'deepseek_r1',
          cache_type: 'hybrid',
          modality: 'text',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('ling')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('deepseek')
    expect(detected.reasoningParser).toBe('deepseek_r1')
    expect(detected.isMultimodal).toBe(false)
  })

  it('keeps JANG VLM enabled from capabilities.modality=vision when architecture.has_vision is absent', () => {
    const dir = makeModelDir(
      { model_type: 'qwen3_5', vision_config: { hidden_size: 1024 } },
      { capabilities: { modality: 'vision' } },
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(true)
  })

  it('falls back to config.json vision_config when jang_config has no vision stamp', () => {
    const dir = makeModelDir(
      { model_type: 'qwen3_5', vision_config: { hidden_size: 1024 } },
      {},
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(true)
  })

  it('detects top-level JANG has_vision without relying on registry family defaults', () => {
    const dir = makeModelDir(
      { model_type: 'qwen3_5' },
      { has_vision: true },
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(true)
  })

  it('keeps affine-JANG Qwen hybrid VLM text-only so the panel does not pass --is-mllm', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        text_config: {
          model_type: 'qwen3_5_text',
          layer_types: ['linear_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1024 },
      },
      { format: 'jang', architecture: { has_vision: true } },
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(false)
  })

  it('keeps MXTQ/JANGTQ Qwen hybrid VLM multimodal', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5_moe',
        text_config: {
          model_type: 'qwen3_5_moe',
          layer_types: ['linear_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1024 },
      },
      { weight_format: 'mxtq', architecture: { has_vision: true } },
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(true)
  })

  it('does not classify text_config-only MoE models as VLMs', () => {
    const dir = makeModelDir(
      { model_type: 'qwen3_5_moe', text_config: { hidden_size: 3072 } },
      {},
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(false)
  })
})
