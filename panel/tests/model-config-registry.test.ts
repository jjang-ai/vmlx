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
  it('marks Qwen3.6 VL JANG bundles with indexed MTP tensors as native MTP capable', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        vision_config: { model_type: 'qwen3_5_vl' },
        text_config: {
          model_type: 'qwen3_5_text',
          mtp_num_hidden_layers: 1,
        },
      },
      {
        format: 'jang',
        runtime: {
          bundle_has_mtp: true,
          mtp_layers: 1,
          mtp_mode: 'preserved_enabled',
        },
        mtp: { kept: true, enabled: true, num_layers: 1 },
        capabilities: {
          family: 'qwen3_5',
          modality: 'vision',
          cache_type: 'hybrid',
        },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'language_model.model.embed_tokens.weight': 'model.safetensors',
        'vision_tower.patch_embed.proj.weight': 'model.safetensors',
        'mtp.fc.weight': 'model.safetensors',
        'mtp.layers.0.self_attn.q_proj.weight': 'model.safetensors',
        'mtp.norm.weight': 'model.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('qwen3.5')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.nativeMtp).toMatchObject({
      supported: true,
      depth: 3,
      runtimeScope: 'text+vl',
      requiresDeterministicSampling: true,
    })
  })

  it('detects text ZAYA as CCA hybrid with opt-in qwen3 reasoning parser', () => {
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
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('zaya_xml')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.thinkInTemplate).toBe(false)
  })

  it('detects ZAYA1-VL as multimodal CCA hybrid with qwen3 reasoning parser', () => {
    const dir = makeModelDir(
      {
        model_type: 'zaya1_vl',
        vision_config: { model_type: 'qwen2_5_vl' },
      },
      {
        cache_subtype: 'zaya_cca',
        capabilities: {
          family: 'zaya1_vl',
          tool_parser: 'zaya_xml',
          reasoning_parser: 'qwen3',
          think_in_template: false,
          supports_thinking: true,
          cache_type: 'hybrid',
          modality: 'vision',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('zaya1-vl')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('zaya_xml')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.isMultimodal).toBe(true)
  })

  it('keeps ZAYA1-VL JANGTQ_K reasoning parser enabled while preserving VL and typed CCA detection', () => {
    const dir = makeModelDir(
      {
        model_type: 'zaya1_vl',
        vision_config: { model_type: 'qwen2_5_vl' },
        weight_format: 'mxtq',
        mxtq_bits: {
          routed_expert: { gate_proj: 2, up_proj: 2, down_proj: 4 },
        },
      },
      {
        profile: 'JANGTQ_K',
        weight_format: 'mxtq',
        cache_subtype: 'zaya_cca',
        mxtq_bits: {
          routed_expert: { gate_proj: 2, up_proj: 2, down_proj: 4 },
        },
        capabilities: {
          family: 'zaya1_vl',
          tool_parser: 'zaya_xml',
          reasoning_parser: 'qwen3',
          think_in_template: false,
          supports_thinking: true,
          cache_type: 'hybrid',
          modality: 'vision',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('zaya1-vl')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('zaya_xml')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.isTurboQuant).toBe(true)
  })

  it('keeps ZAYA1-VL JANGTQ2 reasoning parser enabled because bit profiles are not runtime-clamped', () => {
    const dir = makeModelDir(
      {
        model_type: 'zaya1_vl',
        vision_config: { model_type: 'qwen2_5_vl' },
        weight_format: 'mxtq',
        mxtq_bits: { routed_expert: 2 },
      },
      {
        profile: 'JANGTQ2',
        weight_format: 'mxtq',
        cache_subtype: 'zaya_cca',
        mxtq_bits: { routed_expert: 2 },
        capabilities: {
          family: 'zaya1_vl',
          tool_parser: 'zaya_xml',
          reasoning_parser: 'qwen3',
          think_in_template: false,
          supports_thinking: true,
          cache_type: 'hybrid',
          modality: 'vision',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('zaya1-vl')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.isTurboQuant).toBe(true)
  })

  it('keeps ZAYA1-VL JANGTQ4 reasoning parser enabled', () => {
    const dir = makeModelDir(
      {
        model_type: 'zaya1_vl',
        vision_config: { model_type: 'qwen2_5_vl' },
        weight_format: 'mxtq',
        mxtq_bits: { routed_expert: 4 },
      },
      {
        profile: 'JANGTQ4',
        weight_format: 'mxtq',
        cache_subtype: 'zaya_cca',
        mxtq_bits: { routed_expert: 4 },
        capabilities: {
          family: 'zaya1_vl',
          tool_parser: 'zaya_xml',
          reasoning_parser: 'qwen3',
          think_in_template: false,
          supports_thinking: true,
          cache_type: 'hybrid',
          modality: 'vision',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('zaya1-vl')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.isTurboQuant).toBe(true)
  })

  it('keeps ZAYA1-VL multimodal when a stale stamp says text', () => {
    const dir = makeModelDir(
      {
        model_type: 'zaya1_vl',
        vision_config: { model_type: 'qwen2_5_vl' },
      },
      {
        cache_subtype: 'zaya_cca',
        capabilities: {
          family: 'zaya1_vl',
          tool_parser: 'zaya_xml',
          reasoning_parser: 'qwen3',
          think_in_template: true,
          supports_thinking: true,
          cache_type: 'hybrid',
          modality: 'text',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('zaya1-vl')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('zaya_xml')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.isMultimodal).toBe(true)
  })

  it('detects Ling/Bailing hybrid with tools and no reasoning parser', () => {
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
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.supportsThinking).toBe(false)
    expect(detected.isMultimodal).toBe(false)
  })

  it('keeps Ling/Bailing non-reasoning even without a JANG capability stamp', () => {
    const dir = makeModelDir(
      {
        model_type: 'bailing_hybrid',
        num_hidden_layers: 32,
        layer_group_size: 8,
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('ling')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('deepseek')
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.supportsThinking).toBe(false)
    expect(detected.isMultimodal).toBe(false)
  })

  it('detects Hy3 as text-only KV with Hunyuan tools and qwen3 reasoning', () => {
    const dir = makeModelDir(
      {
        model_type: 'hy_v3',
        num_hidden_layers: 80,
        num_nextn_predict_layers: 1,
      },
      {
        weight_format: 'mxtq',
        capabilities: {
          family: 'hy_v3',
          tool_parser: 'hunyuan',
          reasoning_parser: 'qwen3',
          think_in_template: true,
          supports_thinking: true,
          cache_type: 'kv',
          modality: 'text',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('hy3')
    expect(detected.cacheType).toBe('kv')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('hunyuan')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.enableAutoToolChoice).toBe(true)
    expect(detected.isMultimodal).toBe(false)
    expect(detected.isTurboQuant).toBe(true)
  })

  it('keeps Hy3 JANGTQ2 Low/High reasoning contract despite 2-bit routed experts', () => {
    const dir = makeModelDir(
      {
        model_type: 'hy_v3',
        num_hidden_layers: 80,
        weight_format: 'mxtq',
        mxtq_bits: { routed_expert: 2 },
      },
      {
        profile: 'JANGTQ2',
        weight_format: 'mxtq',
        mxtq_bits: { routed_expert: 2 },
        capabilities: {
          family: 'hy_v3',
          tool_parser: 'hunyuan',
          reasoning_parser: 'qwen3',
          think_in_template: false,
          supports_thinking: true,
          cache_type: 'kv',
          modality: 'text',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('hy3')
    expect(detected.toolParser).toBe('hunyuan')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.isTurboQuant).toBe(true)
  })

  it('keeps Hy3 JANGTQ_K Low/High reasoning contract', () => {
    const dir = makeModelDir(
      {
        model_type: 'hy_v3',
        num_hidden_layers: 80,
        weight_format: 'mxtq',
        mxtq_bits: {
          routed_expert: { gate_proj: 2, up_proj: 2, down_proj: 4 },
        },
      },
      {
        profile: 'JANGTQ_K',
        weight_format: 'mxtq',
        mxtq_bits: {
          routed_expert: { gate_proj: 2, up_proj: 2, down_proj: 4 },
        },
        capabilities: {
          family: 'hy_v3',
          tool_parser: 'hunyuan',
          reasoning_parser: 'qwen3',
          think_in_template: false,
          supports_thinking: true,
          cache_type: 'kv',
          modality: 'text',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('hy3')
    expect(detected.toolParser).toBe('hunyuan')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.isTurboQuant).toBe(true)
  })

  it('keeps Gemma 4 VLM wrapper multimodal instead of demoting to gemma4-text', () => {
    const dir = makeModelDir(
      {
        model_type: 'gemma4',
        text_config: { model_type: 'gemma4_text' },
        vision_config: { hidden_size: 1152 },
      },
      {},
    )

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('gemma4')
    expect(detected.reasoningParser).toBe('gemma4')
    expect(detected.toolParser).toBe('gemma4')
    expect(detected.isMultimodal).toBe(true)
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

  it.each([
    ['qwen3_5', 'qwen3_5_text'],
    ['qwen3_vl', 'qwen3_vl'],
    ['qwen3_vl_moe', 'qwen3_vl_moe'],
  ])('routes affine-JANG %s text-only until the mlx-vlm M-RoPE path is fixed', (modelType, textModelType) => {
    const dir = makeModelDir(
      {
        model_type: modelType,
        text_config: {
          model_type: textModelType,
          layer_types: ['linear_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1024 },
        video_token_id: 151666,
        video_token_index: 151666,
      },
      { format: 'jang', architecture: { has_vision: true } },
    )

    const detected = detectModelConfigFromDir(dir)
    expect(detected.isMultimodal).toBe(false)
    expect(detected.forceTextOnly).toBe(true)
  })

  it('keeps affine-JANG Qwen native-MTP VL artifacts multimodal when indexed MTP and vision tensors exist', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        text_config: {
          model_type: 'qwen3_5_text',
          mtp_num_hidden_layers: 1,
          layer_types: ['linear_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1024 },
        image_token_id: 151665,
        video_token_id: 151666,
      },
      {
        format: 'jang',
        architecture: { has_vision: true },
        runtime: { mtp_layers: 1 },
        mtp: { kept: true, enabled: true, num_layers: 1 },
        capabilities: { family: 'qwen3_5', modality: 'vision', cache_type: 'hybrid' },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model-00001-of-00001.safetensors',
        'vision_tower.patch_embed.proj.weight': 'model-00001-of-00001.safetensors',
        'mtp.layers.0.self_attn.q_proj.weight': 'model-00001-of-00001.safetensors',
        'mtp.norm.weight': 'model-00001-of-00001.safetensors',
      },
    }, null, 2))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('qwen3.5')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.isMultimodal).toBe(true)
    expect(detected.forceTextOnly).toBeUndefined()
  })

  it('keeps MXTQ/JANGTQ Qwen hybrid VLM multimodal', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        text_config: {
          model_type: 'qwen3_5_text',
          layer_types: ['linear_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1024 },
        video_token_id: 151666,
        video_token_index: 151666,
      },
      { format: 'mxtq', weight_format: 'mxtq', architecture: { has_vision: true } },
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(true)
  })

  it('marks non-JANG Qwen 3.6 MoE bundles with vision/video metadata as multimodal', () => {
    const dir = makeModelDir({
      model_type: 'qwen3_5_moe',
      text_config: {
        model_type: 'qwen3_5_moe',
        layer_types: ['linear_attention', 'full_attention'],
      },
      vision_config: { hidden_size: 1024 },
      video_token_id: 151666,
      video_token_index: 151666,
    })

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('qwen3.5-moe')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.isMultimodal).toBe(true)
  })

  it('does not route Nemotron-H text extracts through MLLM from stale sidecars', () => {
    const dir = makeModelDir(
      {
        model_type: 'nemotron_h',
        architectures: ['NemotronHForCausalLM'],
        text_config: {
          layer_types: ['mamba', 'full_attention'],
        },
      },
      {
        capabilities: {
          family: 'nemotron_h',
          modality: 'omni',
          cache_type: 'hybrid',
        },
      },
    )
    writeFileSync(join(dir, 'preprocessor_config.json'), '{}')

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('nemotron-h')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.isMultimodal).toBe(false)
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

    const detected = detectModelConfigFromDir(dir)
    expect(detected.isMultimodal).toBe(true)
    expect(detected.isTurboQuant).toBe(true)
  })

  it('detects TurboQuant from config.json weight_format when jang_config is absent', () => {
    const dir = makeModelDir({
      model_type: 'minimax_m2',
      weight_format: 'mxtq',
    })

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('minimax')
    expect(detected.isTurboQuant).toBe(true)
  })

  it('detects TurboQuant from config.json quantization when jang_config is malformed', () => {
    const dir = makeModelDir({
      model_type: 'qwen3_5_moe',
      quantization: { weight_format: 'mxtq' },
    })
    writeFileSync(join(dir, 'jang_config.json'), '{not-json')

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('qwen3.5-moe')
    expect(detected.isTurboQuant).toBe(true)
  })

  it('uses JANG capabilities cache and parser stamps for Qwen3.6 hybrid bundles', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5_moe',
        text_config: {
          model_type: 'qwen3_5_moe_text',
          layer_types: ['linear_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1024 },
      },
      {
        weight_format: 'mxtq',
        capabilities: {
          family: 'qwen3_5_moe',
          cache_type: 'hybrid',
          modality: 'vision',
          tool_parser: 'qwen',
          reasoning_parser: 'qwen3',
          supports_tools: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('qwen3.5-moe')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('qwen')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.supportsThinking).toBeUndefined()
    expect(detected.enableAutoToolChoice).toBe(true)
    expect(detected.isMultimodal).toBe(true)
    expect(detected.isTurboQuant).toBe(true)
  })

  it('does not classify text_config-only MoE models as VLMs', () => {
    const dir = makeModelDir(
      { model_type: 'qwen3_5_moe', text_config: { hidden_size: 3072 } },
      {},
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(false)
  })
})
