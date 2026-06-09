import { afterEach, describe, expect, it } from 'vitest'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'fs'
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
      depthSource: 'default',
      runtimeScope: 'text+vl',
      requiresDeterministicSampling: true,
    })
  })

  it('uses validated model-local MTP tuning depth when present', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        text_config: {
          model_type: 'qwen3_5_text',
          mtp_num_hidden_layers: 1,
        },
      },
      {
        format: 'mxfp4',
        mtp: { kept: true, enabled: true, num_layers: 1 },
        capabilities: {
          family: 'qwen3_5',
          cache_type: 'hybrid',
        },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'mtp.fc.weight': 'model.safetensors',
        'mtp.layers.0.self_attn.q_proj.weight': 'model.safetensors',
      },
    }))
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      native_mtp: {
        best_depth: 2,
        validated: true,
        output_equivalent: true,
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.nativeMtp).toMatchObject({
      supported: true,
      depth: 2,
      depthSource: 'vmlx_mtp_tuning.json:native_mtp.best_depth',
    })
  })

  it('does not expose Native MTP when model-local tuning blocks the runtime', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5_moe',
        text_config: {
          model_type: 'qwen3_5_moe_text',
          mtp_num_hidden_layers: 1,
        },
      },
      {
        format: 'jang',
        mtp: { kept: true, enabled: true, num_layers: 1 },
        capabilities: {
          family: 'qwen3_5_moe',
          cache_type: 'hybrid',
          modality: 'vision',
        },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'model.visual.patch_embed.proj.weight': 'model.safetensors',
        'mtp.fc.weight': 'model.safetensors',
        'mtp.layers.0.self_attn.q_proj.weight': 'model.safetensors',
      },
    }))
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      native_mtp: {
        blocked: true,
        validated: false,
        output_equivalent: false,
        reason: 'failed runtime validation',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.nativeMtp).toBeUndefined()
  })

  it('does not expose Native MTP for config-only bundles without indexed mtp tensors', () => {
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
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('qwen3.5')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.isMultimodal).toBe(false)
    expect(detected.forceTextOnly).toBe(true)
    expect(detected.nativeMtp).toBeUndefined()
  })

  it('does not expose Native MTP for Ling/Bailing config-only bundles without indexed mtp tensors', () => {
    const dir = makeModelDir(
      {
        model_type: 'bailing_hybrid',
        num_nextn_predict_layers: 1,
      },
      {
        format: 'mxtq',
        capabilities: {
          family: 'ling',
          cache_type: 'hybrid_ssm',
        },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'model.layers.0.self_attn.q_proj.weight': 'model.safetensors',
        'model.layers.0.mlp.gate_proj.weight': 'model.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('ling')
    // The panel collapses Ling/Bailing's engine-level hybrid_ssm_typed cache
    // contract into the existing hybrid settings category.
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.nativeMtp).toBeUndefined()
  })

  it('keeps JANG_2K Native MTP blocked by default to match Python runtime policy', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5_moe',
        text_config: {
          model_type: 'qwen3_5_moe_text',
          mtp_num_hidden_layers: 1,
        },
      },
      {
        format: 'jang',
        mtp: { kept: true, enabled: true, num_layers: 1 },
        quantization: { profile: 'JANG_2K' },
        capabilities: {
          family: 'qwen3_5_moe',
          cache_type: 'hybrid',
          modality: 'vision',
        },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'model.visual.patch_embed.proj.weight': 'model.safetensors',
        'mtp.fc.weight': 'model.safetensors',
        'mtp.layers.0.self_attn.q_proj.weight': 'model.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.nativeMtp).toBeUndefined()
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
    expect(detected.defaultEnableThinking).toBe(false)
  })

  it('detects ZAYA1-VL as multimodal CCA hybrid without a reasoning claim', () => {
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
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.supportsThinking).toBe(false)
    expect(detected.thinkInTemplate).toBe(false)
    expect(detected.defaultEnableThinking).toBe(false)
    expect(detected.isMultimodal).toBe(true)
  })

  it('keeps ZAYA1-VL JANGTQ_K multimodal while suppressing the stale reasoning rail', () => {
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
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.supportsThinking).toBe(false)
    expect(detected.thinkInTemplate).toBe(false)
    expect(detected.isMultimodal).toBe(true)
    expect(detected.isTurboQuant).toBe(true)
  })

  it('keeps ZAYA1-VL JANGTQ2 multimodal without a reasoning rail', () => {
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
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.supportsThinking).toBe(false)
    expect(detected.thinkInTemplate).toBe(false)
    expect(detected.isMultimodal).toBe(true)
    expect(detected.isTurboQuant).toBe(true)
  })

  it('keeps ZAYA1-VL JANGTQ4 multimodal without a reasoning rail', () => {
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
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.supportsThinking).toBe(false)
    expect(detected.thinkInTemplate).toBe(false)
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
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.supportsThinking).toBe(false)
    expect(detected.thinkInTemplate).toBe(false)
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

  it('does not expose Native MTP for Hy3 config-only bundles without indexed mtp tensors', () => {
    const dir = makeModelDir(
      {
        model_type: 'hy_v3',
        num_hidden_layers: 80,
        num_nextn_predict_layers: 1,
      },
      {
        weight_format: 'mxtq',
        runtime: {
          bundle_has_mtp: true,
          mtp_layers: 1,
          mtp_mode: 'preserved_disabled',
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
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'model.layers.0.self_attn.q_proj.weight': 'model.safetensors',
        'model.layers.0.mlp.gate_proj.weight': 'model.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('hy3')
    expect(detected.toolParser).toBe('hunyuan')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.nativeMtp).toBeUndefined()
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

  it('detects Gemma 3 tool_code parser without reasoning extraction', () => {
    const dir = makeModelDir({
      model_type: 'gemma3',
      vision_config: { hidden_size: 1024 },
    })

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('gemma3')
    expect(detected.toolParser).toBe('gemma3')
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.enableAutoToolChoice).toBe(true)
    expect(detected.isMultimodal).toBe(true)
  })

  it('detects Gemma 3n as Gemma tool_code parser without reasoning extraction', () => {
    const dir = makeModelDir({
      model_type: 'gemma3n',
      vision_config: { hidden_size: 1024 },
      audio_config: { hidden_size: 512 },
    })

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('gemma3n')
    expect(detected.toolParser).toBe('gemma3')
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.enableAutoToolChoice).toBe(true)
    expect(detected.isMultimodal).toBe(true)
  })

  it('keeps Gemma 3 text bundles text-only with Gemma tool_code parser', () => {
    const dir = makeModelDir({
      model_type: 'gemma3_text',
    })

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('gemma3-text')
    expect(detected.toolParser).toBe('gemma3')
    expect(detected.reasoningParser).toBeUndefined()
    expect(detected.enableAutoToolChoice).toBe(true)
    expect(detected.isMultimodal).toBe(false)
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

  it('marks Gemma 4 mixed-SWA wrappers as rotating KV so cache UI cannot treat them as plain KV', () => {
    const dir = makeModelDir(
      {
        model_type: 'gemma4',
        text_config: {
          model_type: 'gemma4_text',
          layer_types: [
            'sliding_attention',
            'sliding_attention',
            'full_attention',
          ],
        },
        vision_config: { hidden_size: 1152 },
      },
      {},
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('gemma4')
    expect(detected.cacheType).toBe('rotating_kv')
    expect(detected.usePagedCache).toBe(true)
  })

  it('keeps JANG VLM enabled from capabilities.modality=vision when architecture.has_vision is absent', () => {
    const dir = makeModelDir(
      { model_type: 'qwen3_5', vision_config: { hidden_size: 1024 } },
      { capabilities: { modality: 'vision' } },
    )

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(true)
  })

  it('routes Step3.7 JANG bridge through the source VLM runtime when available', () => {
    const dir = makeModelDir(
      {
        model_type: 'step3p7',
        model_file: 'step3p7_mlx.py',
        text_config: { model_type: 'step3p5' },
        vision_config: { hidden_size: 1152 },
        image_token_id: 151655,
      },
      {
        format: 'jang',
        architecture: { has_vision: true, text_model_type: 'step3p5' },
        capabilities: {
          family: 'step3p7',
          modality: 'vision',
          tool_parser: 'step3p5',
          reasoning_parser: 'qwen3',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('step-3.7-flash')
    expect(detected.toolParser).toBe('step3p5')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.cacheSubtype).toBe('step3p7_full_sliding_kv')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.isMultimodal).toBe(true)
    expect(detected.forceTextOnly).toBeUndefined()
    expect(detected.architectureHints).toMatchObject({
      runtimeScope: 'source_vlm_needs_live_proof',
      vlRuntimeAvailable: true,
      textBridgeRuntimeScope: 'text_bridge_ignored_for_source_vlm',
      slidingWindow: 512,
    })
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

  it('keeps MiMo V2 JANG text-only when capabilities say media sidecars are unwired', () => {
    const dir = makeModelDir(
      {
        model_type: 'mimo_v2',
        vision_config: { model_type: 'mimo_v2_vision' },
        audio_config: { model_type: 'mimo_v2_audio' },
        image_token_id: 151655,
        video_token_id: 151656,
      },
      {
        format: 'jang',
        capabilities: {
          family: 'mimo_v2',
          modalities: ['text'],
          preserved_modalities: ['vision', 'audio', 'video'],
          unwired_modalities: ['vision', 'audio', 'video'],
          supports_tools: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('mimo_v2')
    expect(detected.toolParser).toBe('xml_function')
    expect(detected.isMultimodal).toBe(false)
    expect(detected.forceTextOnly).toBe(true)
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

  it('routes N2 Pro affine-JANG Qwen-MoE metadata text-only until VL is live-proven', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5_moe',
        architectures: ['Qwen3_5MoeForConditionalGeneration'],
        text_config: {
          model_type: 'qwen3_5_moe_text',
          layer_types: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
          mtp_num_hidden_layers: 1,
        },
        vision_config: { model_type: 'qwen3_5_moe' },
      },
      {
        format: 'jang',
        architecture: {
          type: 'hybrid_moe_ssm',
          has_vision: true,
          has_ssm: true,
          has_moe: true,
        },
        runtime: {
          bundle_has_mtp: false,
          mtp_layers: 1,
          mtp_mode: 'metadata_only_missing_weights',
        },
        mtp: { kept: false, enabled: false, num_layers: 1 },
        capabilities: {
          family: 'qwen3_5_moe',
          modality: 'vision',
          cache_type: 'hybrid',
          tool_parser: 'qwen',
          reasoning_parser: 'qwen3',
          think_in_template: true,
          supports_tools: true,
          supports_thinking: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('qwen3.5-moe')
    expect(detected.cacheType).toBe('hybrid')
    expect(detected.usePagedCache).toBe(true)
    expect(detected.toolParser).toBe('qwen')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.supportsThinking).toBe(true)
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

  it.each(['mxfp4', 'mxfp8'])('keeps %s Qwen hybrid VLM multimodal', (weightFormat) => {
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
      {
        format: 'jang',
        weight_format: weightFormat,
        quantization: { method: weightFormat },
        architecture: { has_vision: true },
      },
    )

    const detected = detectModelConfigFromDir(dir)
    expect(detected.isMultimodal).toBe(true)
    expect(detected.forceTextOnly).toBeUndefined()
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

  it('documents Qwen 3.6 release rows intentionally use qwen3.5 family aliases', () => {
    const dense = makeModelDir({
      model_type: 'qwen3_5',
      text_config: {
        model_type: 'qwen3_5_text',
        layer_types: ['linear_attention', 'full_attention'],
      },
      vision_config: { hidden_size: 1024 },
      video_token_id: 151666,
      video_token_index: 151666,
    })
    const moe = makeModelDir({
      model_type: 'qwen3_5_moe',
      text_config: {
        model_type: 'qwen3_5_moe_text',
        layer_types: ['linear_attention', 'full_attention'],
      },
      vision_config: { hidden_size: 1024 },
      video_token_id: 151666,
      video_token_index: 151666,
    })

    const denseDetected = detectModelConfigFromDir(dense)
    const moeDetected = detectModelConfigFromDir(moe)

    expect(denseDetected.family).toBe('qwen3.5')
    expect(denseDetected.cacheType).toBe('hybrid')
    expect(denseDetected.toolParser).toBe('qwen')
    expect(denseDetected.reasoningParser).toBe('qwen3')
    expect(denseDetected.isMultimodal).toBe(true)
    expect(moeDetected.family).toBe('qwen3.5-moe')
    expect(moeDetected.cacheType).toBe('hybrid')
    expect(moeDetected.toolParser).toBe('qwen')
    expect(moeDetected.reasoningParser).toBe('qwen3')
    expect(moeDetected.isMultimodal).toBe(true)
  })

  it('does not route Nemotron-H text extracts through MLLM from stale sidecars', () => {
    const hybridPattern = 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME'
    const dir = makeModelDir(
      {
        model_type: 'nemotron_h',
        architectures: ['NemotronHForCausalLM'],
        hybrid_override_pattern: hybridPattern,
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
    expect(detected.cacheSubtype).toBe('nemotron_h_ssm_attention')
    expect(detected.architectureHints?.attentionArch).toBe('hybrid_ssm_attention')
    expect(detected.architectureHints?.hybridOverridePattern).toBe(hybridPattern)
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
    expect(detected.reasoningParser).toBe('minimax_m2')
  })

  it('uses the registered MiniMax reasoning parser even when bundle sidecars say qwen3', () => {
    const dir = makeModelDir(
      { model_type: 'minimax_m2' },
      {
        weight_format: 'mxtq',
        capabilities: {
          family: 'minimax',
          cache_type: 'kv',
          tool_parser: 'minimax',
          reasoning_parser: 'qwen3',
          think_in_template: true,
          supports_thinking: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('minimax')
    expect(detected.toolParser).toBe('minimax')
    expect(detected.reasoningParser).toBe('minimax_m2')
    expect(detected.enableAutoToolChoice).toBe(true)
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

  it('keeps Mistral Small 4 VLM multimodal while inheriting Mistral 4 reasoning defaults', () => {
    const dir = makeModelDir(
      {
        model_type: 'mistral3',
        architectures: ['Mistral3ForConditionalGeneration'],
        text_config: { model_type: 'mistral4' },
        vision_config: { model_type: 'pixtral' },
      },
      {
        format: 'jang',
        architecture: { has_vision: true },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('mistral4')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.toolParser).toBe('mistral')
    expect(detected.reasoningParser).toBe('mistral')
  })
})

describe('detectModelConfigFromDir backend parity coverage', () => {
  const cases: Array<{
    modelType: string
    family: string
    cacheType: string
    cacheSubtype?: string
    toolParser?: string
    reasoningParser?: string
    isMultimodal?: boolean
  }> = [
    { modelType: 'deepseek_v32', family: 'deepseek-v3', cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1' },
    { modelType: 'falcon_h1', family: 'falcon-h1', cacheType: 'hybrid' },
    { modelType: 'glm_moe_dsa', family: 'glm5', cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1' },
    { modelType: 'got_ocr2', family: 'got-ocr', cacheType: 'kv', isMultimodal: true },
    { modelType: 'granitemoehybrid', family: 'granitemoehybrid', cacheType: 'hybrid', toolParser: 'granite' },
    { modelType: 'kimi_k25', family: 'kimi-k25', cacheType: 'kv', toolParser: 'kimi', reasoningParser: 'deepseek_r1', isMultimodal: true },
    { modelType: 'laguna', family: 'laguna', cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3' },
    { modelType: 'lfm2', family: 'lfm2', cacheType: 'hybrid', cacheSubtype: 'lfm2_moe_hybrid_ssm', toolParser: 'lfm2', reasoningParser: 'qwen3' },
    { modelType: 'lfm2_moe', family: 'lfm2', cacheType: 'hybrid', cacheSubtype: 'lfm2_moe_hybrid_ssm', toolParser: 'lfm2', reasoningParser: 'qwen3' },
    { modelType: 'ministral3', family: 'ministral3', cacheType: 'kv', toolParser: 'mistral' },
    { modelType: 'mistral3', family: 'mistral3', cacheType: 'kv', toolParser: 'mistral', isMultimodal: true },
    { modelType: 'mistral4', family: 'mistral4', cacheType: 'kv', toolParser: 'mistral', reasoningParser: 'mistral' },
    { modelType: 'mimo_v2', family: 'mimo_v2', cacheType: 'kv', toolParser: 'xml_function', reasoningParser: 'think_xml', isMultimodal: true },
    { modelType: 'nemotron_h_v2', family: 'nemotron-h', cacheType: 'hybrid', toolParser: 'nemotron', reasoningParser: 'deepseek_r1', cacheSubtype: 'nemotron_h_ssm_attention' },
    { modelType: 'rwkv7', family: 'rwkv', cacheType: 'mamba' },
    { modelType: 'step3p7', family: 'step-3.7-flash', cacheType: 'kv', cacheSubtype: 'step3p7_full_sliding_kv', toolParser: 'step3p5', reasoningParser: 'qwen3', isMultimodal: true },
  ]

  for (const row of cases) {
    it(`detects backend-covered model_type=${row.modelType}`, () => {
      const dir = makeModelDir({ model_type: row.modelType })

      const detected = detectModelConfigFromDir(dir)

      expect(detected.family).toBe(row.family)
      expect(detected.cacheType).toBe(row.cacheType)
      if ('cacheSubtype' in row) expect(detected.cacheSubtype).toBe(row.cacheSubtype)
      if (row.toolParser !== undefined) expect(detected.toolParser).toBe(row.toolParser)
      if (row.reasoningParser !== undefined) expect(detected.reasoningParser).toBe(row.reasoningParser)
      if (row.isMultimodal !== undefined) expect(detected.isMultimodal).toBe(row.isMultimodal)
      if (row.modelType === 'lfm2_moe') {
        expect(detected.architectureHints).toMatchObject({
          attentionArch: 'hybrid_ssm_attention',
          cacheSchema: 'hybrid_ssm_v1',
          ssmCompanionCache: true,
          attentionKvStorageQuantization: true,
        })
      }
    })
  }

  it('enables MiMo-V2 JANG_2L xml_function tools from verified capability stamps', () => {
    const dir = makeModelDir(
      {
        model_type: 'mimo_v2',
        vision_config: { hidden_size: 1280 },
        audio_config: { hidden_size: 1024 },
        max_position_embeddings: 1048576,
      },
      {
        weight_format: 'jang',
        runtime: { bundle_has_mtp: false, mtp_mode: 'absent' },
        capabilities: {
          family: 'mimo_v2',
          modality: 'multimodal',
          cache_type: 'kv',
          reasoning_parser: 'think_xml',
          tool_parser: 'xml_function',
          supports_tools: true,
          supports_thinking: true,
          think_in_template: false,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('mimo_v2')
    expect(detected.cacheType).toBe('kv')
    expect(detected.reasoningParser).toBe('think_xml')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.thinkInTemplate).toBe(false)
    expect(detected.toolParser).toBe('xml_function')
    expect(detected.enableAutoToolChoice).toBe(true)
    expect(detected.isMultimodal).toBe(true)
  })

  it('rejects stale MiMo-V2 non-xml tool parser claims', () => {
    const dir = makeModelDir(
      {
        model_type: 'mimo_v2',
        vision_config: { hidden_size: 1280 },
      },
      {
        weight_format: 'jang',
        capabilities: {
          family: 'mimo_v2',
          cache_type: 'kv',
          reasoning_parser: 'qwen3',
          tool_parser: 'qwen',
          supports_tools: true,
          supports_thinking: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('mimo_v2')
    expect(detected.reasoningParser).toBe('think_xml')
    expect(detected.toolParser).toBe('xml_function')
    expect(detected.enableAutoToolChoice).toBe(true)
  })
})

describe('detectModelConfigFromDir local high-risk artifact parity', () => {
  it('matches current local high-risk model paths to panel parser cache and modality policy', () => {
    const rows: Array<{
      name: string
      path: string
      family: string
      cacheType: string
      toolParser?: string
      reasoningParser?: string
      isMultimodal: boolean
    }> = [
      {
        name: 'dsv4_k',
        path: '/Users/example/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K',
        family: 'deepseek-v4',
        cacheType: 'kv',
        toolParser: 'dsml',
        reasoningParser: 'deepseek_r1',
        isMultimodal: false,
      },
      {
        name: 'qwen27_jang4m',
        path: '/Users/example/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK',
        family: 'qwen3.5',
        cacheType: 'hybrid',
        toolParser: 'qwen',
        reasoningParser: 'qwen3',
        isMultimodal: false,
      },
      {
        name: 'qwen27_jang4m_mtp',
        path: '/Users/example/models/JANGQ/Qwen3.6-27B-JANG_4M-MTP',
        family: 'qwen3.5',
        cacheType: 'hybrid',
        toolParser: 'qwen',
        reasoningParser: 'qwen3',
        isMultimodal: true,
      },
      {
        name: 'qwen27_mxfp4',
        path: '/Users/example/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK',
        family: 'qwen3.5',
        cacheType: 'hybrid',
        toolParser: 'qwen',
        reasoningParser: 'qwen3',
        isMultimodal: true,
      },
      {
        name: 'qwen27_mxfp8_mtp',
        path: '/Users/example/models/JANGQ/Qwen3.6-27B-MXFP8-MTP',
        family: 'qwen3.5',
        cacheType: 'hybrid',
        toolParser: 'qwen',
        reasoningParser: 'qwen3',
        isMultimodal: true,
      },
      {
        name: 'qwen35_jangtq',
        path: '/Users/example/models/dealign.ai/Qwen3.6-35B-A3B-JANGTQ-CRACK',
        family: 'qwen3.5-moe',
        cacheType: 'hybrid',
        toolParser: 'qwen',
        reasoningParser: 'qwen3',
        isMultimodal: true,
      },
      {
        name: 'qwen35_4bit',
        path: '/Users/example/models/Qwen3.6-35B-A3B-4bit',
        family: 'qwen3.5-moe',
        cacheType: 'hybrid',
        toolParser: 'qwen',
        reasoningParser: 'qwen3',
        isMultimodal: true,
      },
      {
        name: 'qwen35_mxfp8_mtp',
        path: '/Users/example/models/JANGQ/Qwen3.6-35B-A3B-MXFP8-MTP',
        family: 'qwen3.5-moe',
        cacheType: 'hybrid',
        toolParser: 'qwen',
        reasoningParser: 'qwen3',
        isMultimodal: true,
      },
      {
        name: 'hy3',
        path: '/Users/example/models/JANGQ/Hy3-preview-JANGTQ2',
        family: 'hy3',
        cacheType: 'kv',
        toolParser: 'hunyuan',
        reasoningParser: 'qwen3',
        isMultimodal: false,
      },
      {
        name: 'nemotron_jangtq',
        path: '/Users/example/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ-CRACK',
        family: 'nemotron-h',
        cacheType: 'hybrid',
        toolParser: 'nemotron',
        reasoningParser: 'deepseek_r1',
        isMultimodal: false,
      },
      {
        name: 'nemotron_omni_nano_jangtq4',
        path: '/Users/example/models/dealign.ai/Nemotron-Omni-Nano-JANGTQ4-CRACK',
        family: 'nemotron-h',
        cacheType: 'hybrid',
        toolParser: 'nemotron',
        reasoningParser: 'deepseek_r1',
        isMultimodal: false,
      },
      {
        name: 'nemotron_mxfp4',
        path: '/Users/example/models/dealign.ai/Nemotron-Omni-Nano-MXFP4-CRACK',
        family: 'nemotron-h',
        cacheType: 'hybrid',
        toolParser: 'nemotron',
        reasoningParser: 'deepseek_r1',
        isMultimodal: false,
      },
    ]

    const missing = rows.filter(row => !existsSync(row.path)).map(row => row.path)
    if (missing.length > 0) {
      return
    }

    for (const row of rows) {
      const detected = detectModelConfigFromDir(row.path)
      expect(detected.family, row.name).toBe(row.family)
      expect(detected.cacheType, row.name).toBe(row.cacheType)
      expect(detected.toolParser, row.name).toBe(row.toolParser)
      expect(detected.reasoningParser, row.name).toBe(row.reasoningParser)
      expect(detected.isMultimodal, row.name).toBe(row.isMultimodal)
    }
  })
})
