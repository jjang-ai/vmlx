import { afterEach, describe, expect, it } from 'vitest'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import { detectModelConfigFromDir } from '../src/main/model-config-registry'
import { resolveEffectiveToolParser } from '../src/shared/toolParserAliases'
import { buildToolLaunchArgs } from '../src/shared/toolLaunchArgs'

const createdDirs: string[] = []

describe('Spark native defaults', () => {
  it('recognizes the native no_think mode in the bundle stamp', () => {
    const dir = makeModelDir({ model_type: 'spark2_5' }, {
      capabilities: { family: 'spark2_5', supports_thinking: true, tool_parser: 'spark25' },
      chat: { reasoning: { supported: true, parser: 'qwen3', modes: ['think', 'no_think'],
        template_flag: 'enable_thinking', default_enabled: true } },
    })
    writeFileSync(join(dir, 'chat_template.jinja'),
      "{% if enable_thinking %}<think>{% else %}</think>{% endif %}")
    const config = detectModelConfigFromDir(dir)
    expect(config.supportsInstructMode).toBe(true)
    expect(config.honorsEnableThinking).toBe(true)
    expect(config.defaultEnableThinking).toBe(true)
  })
  it('derives parser, reasoning and text-only mode from config identity', () => {
    const dir = makeModelDir({ model_type: 'spark2_5', num_hidden_layers: 36,
      layer_types: ['sliding_attention', 'full_attention'], sliding_window: 512 })
    const config = detectModelConfigFromDir(dir)
    expect(config.toolParser).toBe('spark25')
    expect(config.reasoningParser).toBe('qwen3')
    expect(config.thinkInTemplate).toBe(true)
    expect(config.defaultEnableThinking).toBe(true)
    expect(config.isMultimodal).toBe(false)
    expect(buildToolLaunchArgs({ toolParser: config.toolParser!, enableAutoToolChoice: true }))
      .toEqual(['--tool-call-parser', 'spark25', '--enable-auto-tool-choice'])
  })
})

describe('MiniCPM5 declared XML dialect', () => {
  it.each(['minicpm5_xml_function', 'unrelated'])('scopes the capability correction to %s', (dialect) => {
    const dir = makeModelDir({ model_type: 'llama' }, {
      capabilities: { family: 'llama', tool_parser: 'llama', supports_thinking: false, think_in_template: false },
      tool_calling: { dialect },
    })
    const config = detectModelConfigFromDir(dir)
    expect(config.toolParser).toBe(dialect === 'minicpm5_xml_function' ? 'minicpm5' : 'llama')
    expect(config.supportsThinking).toBe(dialect === 'minicpm5_xml_function')
    expect(config.defaultEnableThinking).toBeUndefined()
    expect(config.isMultimodal).toBe(false)
    if (dialect === 'minicpm5_xml_function') {
      for (const configuredParser of ['auto', 'minicpm5', 'minicpm5_xml_function']) {
        const toolParser = resolveEffectiveToolParser({ configuredParser, detectedParser: config.toolParser })
        expect(buildToolLaunchArgs({ toolParser, enableAutoToolChoice: true }))
          .toEqual(['--tool-call-parser', 'minicpm5', '--enable-auto-tool-choice'])
      }
    }
  })
})

function makeModelDir(config: Record<string, unknown>, jangConfig?: Record<string, unknown>): string {
  const dir = mkdtempSync(join(tmpdir(), 'vmlx-model-config-'))
  createdDirs.push(dir)
  writeFileSync(join(dir, 'config.json'), JSON.stringify(config, null, 2))
  if (jangConfig !== undefined) {
    writeFileSync(join(dir, 'jang_config.json'), JSON.stringify(jangConfig, null, 2))
  }
  return dir
}

function writeSafetensorsHeader(path: string, tensors: Record<string, { dtype: string; shape: number[]; data_offsets: number[] }>): void {
  const rawHeader = Buffer.from(JSON.stringify(tensors), 'utf8')
  const headerBytes = Math.ceil(rawHeader.length / 8) * 8
  const endOffset = Math.max(0, ...Object.values(tensors).map(tensor => tensor.data_offsets[1] ?? 0))
  const contents = Buffer.alloc(8 + headerBytes + endOffset, 0)
  contents.writeBigUInt64LE(BigInt(headerBytes), 0)
  rawHeader.copy(contents, 8)
  contents.fill(0x20, 8 + rawHeader.length, 8 + headerBytes)
  writeFileSync(path, contents)
}

afterEach(() => {
  while (createdDirs.length > 0) {
    const dir = createdDirs.pop()
    if (dir) rmSync(dir, { recursive: true, force: true })
  }
})

describe('detectModelConfigFromDir quantization label', () => {
  it('returns the bundle-grounded JANGTQ label instead of a folder-name guess', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5_moe',
        text_config: {
          model_type: 'qwen3_5_moe_text',
          layer_types: ['linear_attention', 'full_attention'],
        },
        weight_format: 'mxtq',
      },
      {
        weight_format: 'mxtq',
        profile: 'JANGTQ2',
        quantization: { method: 'affine+mxtq', bits_default: 2 },
        capabilities: {
          family: 'qwen3_5_moe',
          cache_type: 'hybrid',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.isTurboQuant).toBe(true)
    expect(detected.quantizationLabel).toBe('JANGTQ2 (2b)')
  })
})

describe('detectModelConfigFromDir DSV4 native cache defaults', () => {
  it('reads the native pool codec default from the bundle cache stamp', () => {
    const disabledDir = makeModelDir(
      { model_type: 'deepseek_v4' },
      {
        capabilities: { family: 'deepseek_v4' },
        cache: { pool_quant_default: false },
      },
    )
    const enabledDir = makeModelDir(
      { model_type: 'deepseek_v4' },
      {
        capabilities: { family: 'deepseek_v4' },
        cache: { pool_quant_default: true },
      },
    )

    expect(detectModelConfigFromDir(disabledDir).dsv4PoolQuantDefault).toBe(false)
    expect(detectModelConfigFromDir(enabledDir).dsv4PoolQuantDefault).toBe(true)
  })
})

describe('detectModelConfigFromDir JANG multimodal detection', () => {
  it('detects base Qwen4Exp QSA/GDN reasoning, VL, context, and native MTP without a JANG sidecar', () => {
    const dir = makeModelDir({
      model_type: 'qwen4_exp',
      text_config: {
        model_type: 'qwen4_exp_text',
        max_position_embeddings: 262144,
        mtp_num_hidden_layers: 1,
        layer_types: ['linear_attention', 'linear_attention', 'linear_attention', 'full_attention'],
      },
      vision_config: { model_type: 'qwen4_exp' },
    })
    writeFileSync(join(dir, 'chat_template.jinja'), [
      "{% set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}",
      "{% if enable_thinking is undefined or enable_thinking is true %}",
      "{{ resolved_reasoning_effort }}",
      "{% endif %}",
    ].join('\n'))
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model-00001-of-00131.safetensors',
        'vision_tower.patch_embed.proj.weight': 'model-00130-of-00131.safetensors',
        'mtp.fc_embedding.weight': 'model-00131-of-00131.safetensors',
        'mtp.layers.0.self_attn.q_proj.weight': 'model-00131-of-00131.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected).toMatchObject({
      family: 'qwen4-exp',
      cacheType: 'hybrid',
      cacheSubtype: 'qsa_gdn_ple_v1',
      toolParser: 'qwen',
      reasoningParser: 'qwen3',
      supportsThinking: true,
      supportsInstructMode: true,
      honorsEnableThinking: true,
      supportedReasoningEfforts: ['low', 'medium', 'xhigh'],
      defaultReasoningEffort: 'xhigh',
      thinkInTemplate: true,
      defaultEnableThinking: true,
      usePagedCache: false,
      enableAutoToolChoice: true,
      isMultimodal: true,
      maxContextLength: 262144,
      architectureHints: {
        attentionArch: 'qsa_gdn_ple',
        cacheSchema: 'qsa_gdn_ple_v1',
        cachePrecision: 'full',
        pleStorage: 'ssd_row_addressed',
      },
    })
    expect(detected.nativeMtp).toMatchObject({
      supported: true,
      depth: 1,
      depthSource: 'config.text_config.mtp_num_hidden_layers',
      runtimeScope: 'text+vl',
      requiresDeterministicSampling: false,
    })

    const textDir = makeModelDir({
      model_type: 'qwen4_exp_text',
      max_position_embeddings: 262144,
    })
    expect(detectModelConfigFromDir(textDir)).toMatchObject({
      family: 'qwen4-exp',
      cacheSubtype: 'qsa_gdn_ple_v1',
      supportedReasoningEfforts: ['low', 'medium', 'xhigh'],
    })
  })

  it('uses embedded Qwen4Exp JANG metadata and hides MTP when the head is absent', () => {
    const dir = makeModelDir({
      model_type: 'qwen4_exp',
      text_config: {
        model_type: 'qwen4_exp_text',
        max_position_embeddings: 262144,
        mtp_num_hidden_layers: 1,
      },
      vision_config: { model_type: 'qwen4_exp' },
      jang_config: {
        format: 'jang_v2',
        mtp: { mtp_mode: 'none' },
        capabilities: {
          tool_parser: 'hermes',
          reasoning_parser: 'qwen3',
          supports_thinking: true,
          has_vision: true,
          has_video: true,
          has_audio: false,
        },
        reasoning: {
          default: 'on',
          default_reasoning_effort: 'xhigh',
        },
      },
    })
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model-00001-of-00019.safetensors',
        'vision_tower.patch_embed.proj.weight': 'model-00019-of-00019.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected).toMatchObject({
      family: 'qwen4-exp',
      // Stale converter stamps say hermes (JSON-only), but the bundle
      // template teaches <function=...><parameter=...> XML; the qwen
      // parser accepts both dialects, so the stale stamp is neutralized —
      // mirror of the engine-side registry exception. Live proof:
      // sampled required-mode 400s on Flash-Next JANG_1L under hermes.
      toolParser: 'qwen',
      reasoningParser: 'qwen3',
      supportsThinking: true,
      isMultimodal: true,
    })
    expect(detected.nativeMtp).toBeUndefined()
  })

  it('detects GLM-5.3-Flash (glm5_next) with its serving contract', () => {
    const dir = makeModelDir({
      model_type: 'glm5_next',
      text_config: {
        model_type: 'glm5_next_text',
        max_position_embeddings: 1048576,
        num_hidden_layers: 45,
      },
      vision_config: { model_type: 'glm5_next' },
    })

    expect(detectModelConfigFromDir(dir)).toMatchObject({
      family: 'glm5-next',
      cacheType: 'hybrid',
      cacheSubtype: 'glm5_next_native_v2',
      toolParser: 'glm_xml_args',
      reasoningParser: 'glm_think_block',
      supportsThinking: true,
      thinkInTemplate: true,
      supportedReasoningEfforts: ['low', 'high', 'max'],
      defaultReasoningEffort: 'max',
      usePagedCache: false,
      isMultimodal: true,
    })
    expect(detectModelConfigFromDir(dir).forceTextOnly).toBeUndefined()
  })

  it('keeps an explicit non-hermes Qwen4Exp tool_parser stamp authoritative', () => {
    const dir = makeModelDir({
      model_type: 'qwen4_exp',
      text_config: { model_type: 'qwen4_exp_text', max_position_embeddings: 262144 },
      vision_config: { model_type: 'qwen4_exp' },
      jang_config: {
        format: 'jang_v2',
        mtp: { mtp_mode: 'none' },
        capabilities: {
          tool_parser: 'xml_function',
          reasoning_parser: 'qwen3',
          supports_thinking: true,
          has_vision: true,
          has_video: true,
          has_audio: false,
        },
      },
    })

    expect(detectModelConfigFromDir(dir)).toMatchObject({
      family: 'qwen4-exp',
      toolParser: 'xml_function',
    })
  })

  it('detects Nemotron Omni media from its sidecar and matching component tensors', () => {
    const dir = makeModelDir(
      {
        model_type: 'nemotron_h',
        weight_format: 'mxtq',
        _jang_modality: 'text',
      },
      {
        weight_format: 'mxtq',
        modality: 'omni',
        capabilities: {
          family: 'nemotron_h',
          modality: 'omni',
          cache_type: 'hybrid',
        },
      },
    )
    writeFileSync(join(dir, 'config_omni.json'), JSON.stringify({
      model_type: 'NemotronH_Nano_Omni_Reasoning_V3',
      sound_config: { model_type: 'parakeet', sampling_rate: 16000 },
      vision_config: { model_type: 'radio' },
    }))
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'sound_encoder.layers.0.weight': 'model.safetensors',
        'sound_projection.0.weight': 'model.safetensors',
        'vision_model.encoder.weight': 'model.safetensors',
        'mlp1.0.weight': 'model.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('nemotron-h')
    expect(detected.isTurboQuant).toBe(true)
    expect(detected.isMultimodal).toBe(true)
  })

  it('does not trust a Nemotron Omni metadata stamp without matching media tensors', () => {
    const dir = makeModelDir(
      { model_type: 'nemotron_h', weight_format: 'mxtq', _jang_modality: 'text' },
      {
        weight_format: 'mxtq',
        modality: 'omni',
        capabilities: { family: 'nemotron_h', modality: 'omni', cache_type: 'hybrid' },
      },
    )
    writeFileSync(join(dir, 'config_omni.json'), JSON.stringify({
      sound_config: { model_type: 'parakeet', sampling_rate: 16000 },
      vision_config: { model_type: 'radio' },
    }))
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: { 'model.layers.0.mixer.weight': 'model.safetensors' },
    }))

    expect(detectModelConfigFromDir(dir).isMultimodal).toBe(false)
  })

  it('renders native MTP for dots3-note, whose MTP the engine actually runs', () => {
    // The app ran MTP for this bundle and showed the user NOTHING: engine health
    // reported index_has_mtp_tensors true, 3 MTP tensors, runtime_active true,
    // effective_depth 1, "native MTP runtime is active for text+vl" — while the
    // Server Settings drawer rendered no label, no mode selector and no mention
    // of MTP. Cause was the supportedFamilies allowlist, not the bundle.
    const dir = makeModelDir(
      {
        model_type: 'dots3_note',
        vision_config: { model_type: 'dots3_vision' },
      },
      {
        format: 'jang',
        runtime: { bundle_has_mtp: true, mtp_layers: 1 },
        capabilities: { family: 'dots3_note', modality: 'vision', cache_type: 'hybrid' },
      },
    )
    // Verbatim key shape from the shipped bundle's index.
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'vision_tower.patch_embed.proj.weight': 'model.safetensors',
        'model.mtp.embed_tokens.weight': 'model.safetensors',
        'model.mtp.embed_tokens.scales': 'model.safetensors',
        'model.mtp.embed_tokens.biases': 'model.safetensors',
      },
    }))
    // Its sidecar carries best_depth at TOP level with no nested native_mtp
    // block and no validated/output_equivalent keys ("UNMEASURED default").
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      best_depth: 1,
      blocked: false,
      model_types: ['dots3_note'],
      reason: 'stamped by stamp_dots3; recommendation, not measurement',
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.nativeMtp).toMatchObject({
      supported: true,
      depth: 1,
      runtimeScope: 'text+vl',
      // Matches the engine's reported native_cache schema for this bundle.
      nativeCacheType: 'hybrid_ssm_v1',
      requiresDeterministicSampling: false,
    })
  })

  it('renders native MTP for ERNIE-4.5, whose head is stored as mtp_block (no bare mtp segment)', () => {
    // Verbatim key shape from the shipped JANG_6M bundle index: model.mtp_block.0.*,
    // model.mtp_emb_norm.0.weight, ... The generic /(^|\.)mtp(\.|$)/ test never
    // matches these, so without the mtp_* pattern the family allowlist alone
    // would still hide the control while the engine runs MTP.
    const dir = makeModelDir(
      { model_type: 'ernie4_5_moe', num_hidden_layers: 28, num_nextn_predict_layers: 1 },
      { format: 'jang', mtp: { kept: true, enabled: true, num_layers: 1, tensor_count: 12 } },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'model.mtp_block.0.self_attn.q_proj.weight': 'model.safetensors',
        'model.mtp_emb_norm.0.weight': 'model.safetensors',
        'model.mtp_hidden_norm.0.weight': 'model.safetensors',
        'model.mtp_linear_proj.0.weight': 'model.safetensors',
      },
    }))
    // Sidecar shape written 2026-09-05 from the measured depth sweep.
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      native_mtp: { best_depth: 1, validated: true },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.nativeMtp).toMatchObject({
      supported: true,
      depth: 1,
      depthSource: 'vmlx_mtp_tuning.json:native_mtp.best_depth',
      runtimeScope: 'text',
      // Plain GQA attention: the engine reports native_cache schema plain_kv_v1.
      nativeCacheType: 'plain_kv_v1',
      requiresDeterministicSampling: false,
      defaultMode: 'auto',
    })
  })

  it('hides native MTP for an ERNIE-4.5 bundle whose MTP tensors were dropped', () => {
    const dir = makeModelDir(
      { model_type: 'ernie4_5_moe', num_hidden_layers: 28, num_nextn_predict_layers: 1 },
      { format: 'jang' },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: { 'model.embed_tokens.weight': 'model.safetensors' },
    }))
    expect(detectModelConfigFromDir(dir).nativeMtp).toBeUndefined()
  })

  it('detects the appended layer-N GLM MTP block for the native VL runtime', () => {
    const dir = makeModelDir({
      model_type: 'glm5_next',
      text_config: {
        model_type: 'glm5_next_text',
        num_hidden_layers: 45,
        num_nextn_predict_layers: 1,
        layer_types: ['linear_attention', 'deepseek_sparse_attention'],
      },
      vision_config: { model_type: 'glm5_next_vision' },
    })
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model-00001.safetensors',
        'visual.patch_embed.proj.weight': 'model-00027.safetensors',
        'model.layers.45.enorm.weight': 'model-00021.safetensors',
        'model.layers.45.self_attn.q_a_proj.weight': 'model-00021.safetensors',
        'model.layers.45.shared_head.norm.weight': 'model-00021.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.nativeMtp).toMatchObject({
      supported: true,
      depth: 1,
      depthSource: 'config.text_config.num_nextn_predict_layers',
      runtimeScope: 'text+vl',
      nativeCacheType: 'glm5_next_native_v2',
      requiresDeterministicSampling: false,
      defaultMode: 'off',
    })
    expect(detected).toMatchObject({ isMultimodal: true })
    expect(detected.forceTextOnly).toBeUndefined()
  })

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
      depth: 1,
      depthSource: 'jang_config.runtime.mtp_layers',
      runtimeScope: 'text+vl',
      requiresDeterministicSampling: false,
    })
  })

  it('detects Qwen3.8-27B MTP tensors in a supplemental shard omitted from the base index', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        text_config: {
          model_type: 'qwen3_5_text',
          num_hidden_layers: 64,
          mtp_num_hidden_layers: 1,
        },
      },
      {
        format: 'jang',
        runtime: { bundle_has_mtp: true, mtp_layers: 1 },
        mtp: { kept: true, enabled: true, num_layers: 1, tensor_count: 31 },
        capabilities: { family: 'qwen3_5', cache_type: 'hybrid' },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model-00001-of-00004.safetensors',
      },
    }))
    writeSafetensorsHeader(join(dir, 'model-mtp-of-00005.safetensors'), {
      'mtp.fc.weight': { dtype: 'F16', shape: [1], data_offsets: [0, 2] },
      'mtp.layers.0.self_attn.q_proj.weight': { dtype: 'F16', shape: [1], data_offsets: [2, 4] },
      'mtp.norm.weight': { dtype: 'F16', shape: [1], data_offsets: [4, 6] },
    })
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      native_mtp: { best_depth: 2, validated: true, output_equivalent: true },
    }))

    expect(detectModelConfigFromDir(dir).nativeMtp).toMatchObject({
      supported: true,
      depth: 2,
      depthSource: 'vmlx_mtp_tuning.json:native_mtp.best_depth',
      nativeCacheType: 'hybrid_ssm_v1',
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

  it('uses validated flat Qwen tuning only when wall-speed evidence agrees', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        text_config: { model_type: 'qwen3_5_text', mtp_num_hidden_layers: 1 },
      },
      {
        format: 'jang',
        mtp: { kept: true, enabled: true, num_layers: 1, recommended_num_drafts: 1 },
        capabilities: { family: 'qwen3_5', cache_type: 'hybrid' },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: { 'mtp.fc.weight': 'model.safetensors' },
    }))
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      best_depth: 2,
      blocked: false,
      validated: true,
      baseline_tok_s: 46.24,
      best_tok_s: 56.26,
      speedup_vs_baseline: 1.216,
      measured_tok_s_by_depth: { 1: 46.24, 2: 56.26, 3: 30.14 },
    }))

    expect(detectModelConfigFromDir(dir).nativeMtp).toMatchObject({
      depth: 2,
      depthSource: 'vmlx_mtp_tuning.json:best_depth',
    })
  })

  it('rejects a flat D3 stamp whose own measured wall speed selects D2', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        text_config: { model_type: 'qwen3_5_text', mtp_num_hidden_layers: 1 },
      },
      {
        format: 'jang',
        mtp: { kept: true, enabled: true, num_layers: 1, recommended_num_drafts: 2 },
        capabilities: { family: 'qwen3_5', cache_type: 'hybrid' },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: { 'mtp.fc.weight': 'model.safetensors' },
    }))
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      best_depth: 3,
      blocked: false,
      validated: true,
      baseline_tok_s: 46.24,
      best_tok_s: 30.14,
      speedup_vs_baseline: 0.652,
      measured_tok_s_by_depth: { 1: 46.24, 2: 56.26, 3: 30.14 },
    }))

    expect(detectModelConfigFromDir(dir).nativeMtp).toMatchObject({
      depth: 2,
      depthSource: 'jang_config.mtp.recommended_num_drafts',
    })
  })

  it('keeps HY3 MTP visible but blocks runtime without token-identity proof', () => {
    const dir = makeModelDir(
      {
        model_type: 'hy_v3',
        num_hidden_layers: 80,
        num_nextn_predict_layers: 1,
      },
      {
        format: 'jang',
        profile: 'JANG_2K',
        runtime: {
          bundle_has_mtp: true,
          mtp_layers: 1,
          mtp_mode: 'preserved_native_candidate',
        },
        mtp: { kept: true, enabled: true, num_layers: 1 },
        capabilities: {
          family: 'hy_v3',
          modality: 'text',
          cache_type: 'kv',
        },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'mtp.0.eh_proj.weight': 'model.safetensors',
        'mtp.0.block.self_attn.q_proj.weight': 'model.safetensors',
      },
    }))
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      native_mtp: {
        best_depth: 1,
        validated: true,
        measured: 'coherent but byte-divergent multi-row verifier',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.nativeMtp).toMatchObject({
      supported: false,
      depth: 1,
      depthSource: 'validation-blocked',
      runtimeScope: 'text',
      nativeCacheType: 'plain_kv_v1',
      requiresDeterministicSampling: true,
    })
    expect(detected.nativeMtp?.blockedReason).toContain('token-identical greedy output')
  })

  it('enables HY3 MTP only with an explicit token-identity attestation', () => {
    const dir = makeModelDir(
      {
        model_type: 'hy_v3',
        num_hidden_layers: 80,
        num_nextn_predict_layers: 1,
      },
      {
        format: 'jang',
        profile: 'JANG_2K',
        runtime: {
          bundle_has_mtp: true,
          mtp_layers: 1,
          mtp_mode: 'preserved_native_candidate',
        },
        mtp: { kept: true, enabled: true, num_layers: 1 },
        capabilities: {
          family: 'hy_v3',
          modality: 'text',
          cache_type: 'kv',
        },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'model.embed_tokens.weight': 'model.safetensors',
        'mtp.0.eh_proj.weight': 'model.safetensors',
        'mtp.0.block.self_attn.q_proj.weight': 'model.safetensors',
      },
    }))
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({
      native_mtp: {
        best_depth: 1,
        validated: true,
        output_equivalent: true,
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.nativeMtp).toMatchObject({
      supported: true,
      depth: 1,
      depthSource: 'vmlx_mtp_tuning.json:native_mtp.best_depth',
      runtimeScope: 'text',
      nativeCacheType: 'plain_kv_v1',
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
    // Missing indexed MTP disables speculation only. The independently
    // indexed vision tower remains a runnable multimodal artifact.
    expect(detected.isMultimodal).toBe(true)
    expect(detected.forceTextOnly).toBeUndefined()
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
    expect(detected.usePagedCache).toBe(false)
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

  it('honors the Nemotron stamp spellings of the reasoning default', () => {
    // 2026-08 Nemotron stamps declare "on"/"off" strings in a TOP-LEVEL
    // reasoning block and under capabilities.default_reasoning — neither is
    // the canonical chat.reasoning.default_enabled spelling. "off" is the
    // load-bearing case: reasoning-capable families default ON otherwise, so
    // an ignored reader is silently wrong exactly for "off" bundles.
    const dir = makeModelDir(
      { model_type: 'nemotron_h' },
      {
        capabilities: {
          family: 'nemotron_h',
          reasoning_parser: 'deepseek_r1',
          supports_thinking: true,
          default_reasoning: 'on',
        },
        reasoning: { supported: true, default: 'off' },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('nemotron-h')
    // The dedicated reasoning block outranks the capabilities summary.
    expect(detected.defaultEnableThinking).toBe(false)
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
    expect(detected.usePagedCache).toBe(false)
    expect(detected.toolParser).toBe('zaya_xml')
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.thinkInTemplate).toBe(false)
    expect(detected.defaultEnableThinking).toBe(true)
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
    expect(detected.usePagedCache).toBe(false)
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
    expect(detected.usePagedCache).toBe(false)
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
    expect(detected.usePagedCache).toBe(false)
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
    expect(detected.usePagedCache).toBe(false)
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
    expect(detected.usePagedCache).toBe(false)
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
    // In-RAM paged cache is OFF for EVERY family; SSD block-disk L2 is the only
    // cache tier. Hy3 declares no paged capability, and an undeclared family
    // must NOT acquire a RAM tier by omission -- that fallback used to default
    // every plain-KV text model paged-ON.
    expect(detected.usePagedCache).toBe(false)
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

  it('autodetects MiniMax-M3 (minimax_m3_vl) with paged OFF (typed serializer is prefix/block-L2 only)', () => {
    const dir = makeModelDir(
      { model_type: 'minimax_m3_vl', text_config: { model_type: 'minimax_m3' }, vision_config: { hidden_size: 1024 } },
      {
        has_vision: true,
        capabilities: {
          family: 'minimax_m3',
          reasoning_parser: 'minimax_m3',
          tool_parser: 'minimax_m3',
          supports_tools: true,
          supports_thinking: true,
          think_in_template: false,
          cache_type: 'kv',
          modality: 'multimodal',
        },
      },
    )
    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('minimax_m3')
    expect(detected.reasoningParser).toBe('minimax_m3')
    expect(detected.toolParser).toBe('minimax_m3')
    expect(detected.isMultimodal).toBe(true)
    // The typed M3 serializer (keys/values/idx_keys) lives only in the
    // prefix/block-disk path; the generic paged tier has no M3 sparse-MSA
    // handling and corrupts partial-prefix reuse (live-proven 2026-08-10), so
    // M3 must detect with paged OFF. The engine also hard-forces this off.
    expect(detected.usePagedCache).toBe(false)
  })

  it('autodetects openPangu-2.0-Flash (openpangu_v2) with openpangu tool parser and kv/composite cache despite the stamped hybrid/qwen sidecar', () => {
    // Mirrors the live JANG_2L jang_config stamp: the converter writes the
    // coarse cache_type="hybrid" and tool_parser="qwen"; both are stale for
    // this family (engine d1a588487 + openpangu parser) and must be
    // neutralized so paged cache is NOT forced and the panel passes
    // --tool-call-parser openpangu. Regression: without the registerFamily
    // entry, detection fell through to generic — openpangu startup defaults
    // (timeout 900, JIT off) never fired and the chat thinking toggle stayed
    // disabled (2026-07-02 live UI matrix).
    const dir = makeModelDir(
      { model_type: 'openpangu_v2' },
      {
        capabilities: {
          family: 'openpangu_v2',
          reasoning_parser: 'deepseek_r1',
          tool_parser: 'qwen',
          supports_tools: true,
          supports_thinking: true,
          think_in_template: true,
          cache_type: 'hybrid',
          modality: 'text',
        },
      },
    )
    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('openpangu_v2')
    expect(detected.toolParser).toBe('openpangu')
    expect(detected.reasoningParser).toBe('deepseek_r1')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.thinkInTemplate).toBe(true)
    expect(detected.cacheType).toBe('kv')
    expect(detected.cacheSubtype).toBe('openpangu_v2_composite')
    expect(detected.usePagedCache).toBe(false)
    expect(detected.isMultimodal).toBe(false)
  })

  it('resolves minimax_m3 family even when only the inner text model_type is minimax_m3', () => {
    const dir = makeModelDir(
      { model_type: 'minimax_m3' },
      { capabilities: { family: 'minimax_m3', reasoning_parser: 'minimax_m3', tool_parser: 'minimax_m3' } },
    )
    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('minimax_m3')
    expect(detected.toolParser).toBe('minimax_m3')
  })

  it('autodetects gemma4_unified (Gemma 4 unified VL+audio) as gemma4 multimodal, not unknown', () => {
    const dir = makeModelDir(
      {
        model_type: 'gemma4_unified',
        text_config: {
          model_type: 'gemma4_unified_text',
          layer_types: ['sliding_attention', 'sliding_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1152 },
      },
      {
        has_vision: true,
        has_audio: true,
        capabilities: {
          family: 'gemma4',
          reasoning_parser: 'gemma4',
          tool_parser: 'gemma4',
          supports_tools: true,
          supports_thinking: true,
          think_in_template: false,
          cache_type: 'kv',
          modality: 'multimodal',
          modalities: { text: true, vision: true, audio: true, video: false },
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('gemma4')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.reasoningParser).toBe('gemma4')
    expect(detected.toolParser).toBe('gemma4')
    expect(detected.enableAutoToolChoice).toBe(true)
    expect(detected.supportsThinking).toBe(true)
    expect(detected.thinkInTemplate).toBe(false)
    expect(detected.defaultEnableThinking).toBe(false)
    // Gemma4 mixed-SWA is typed block-disk L2 only: in-RAM paged cache is OFF
    // for every family, mixed-SWA included.
    expect(detected.cacheType).toBe('rotating_kv')
    expect(detected.usePagedCache).toBe(false)
  })

  it('resolves family from JANG capabilities.family when config.json model_type is unrecognized', () => {
    const dir = makeModelDir(
      { model_type: 'some_brand_new_unmapped_type' },
      { capabilities: { family: 'gemma4', reasoning_parser: 'gemma4', tool_parser: 'gemma4' } },
    )
    const detected = detectModelConfigFromDir(dir)
    expect(detected.family).toBe('gemma4')
    expect(detected.reasoningParser).toBe('gemma4')
    expect(detected.toolParser).toBe('gemma4')
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
    expect(detected.supportsThinking).toBe(true)
    expect(detected.thinkInTemplate).toBe(false)
    expect(detected.defaultEnableThinking).toBe(false)
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
    // The typed mixed-SWA runtime restores from block-disk L2 only; it must not
    // claim the retired paged RAM tier.
    expect(detected.usePagedCache).toBe(false)
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
    expect(detected.supportsThinking).toBe(true)
    expect(detected.supportsInstructMode).toBe(false)
    expect(detected.supportedReasoningEfforts).toEqual(['low', 'medium', 'high'])
    expect(detected.thinkInTemplate).toBe(true)
    expect(detected.cacheSubtype).toBe('step3p7_full_sliding_kv')
    expect(detected.usePagedCache).toBe(false)
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
  ])('routes affine-JANG %s metadata-only artifacts text-only without indexed vision tensors', (modelType, textModelType) => {
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

  it('keeps converted affine-JANG Qwen bundles multimodal when the index contains a real vision tower', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        architectures: ['Qwen3_5ForConditionalGeneration'],
        text_config: {
          model_type: 'qwen3_5_text',
          layer_types: ['linear_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1152 },
        image_token_id: 248056,
        video_token_id: 248057,
      },
      {
        format: 'jang',
        architecture: { has_vision: true, has_ssm: true },
        capabilities: {
          family: 'qwen3_5',
          modality: 'vision',
          has_vision: true,
        },
      },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({
      weight_map: {
        'language_model.model.embed_tokens.weight': 'model-00001-of-00002.safetensors',
        'vision_tower.patch_embed.proj.weight': 'model-00002-of-00002.safetensors',
      },
    }))

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('qwen3.5')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.forceTextOnly).toBeUndefined()
  })

  it('keeps runtime-verified affine-JANG Qwen image/video bundles multimodal', () => {
    const dir = makeModelDir(
      {
        model_type: 'qwen3_5',
        architectures: ['Qwen3_5ForConditionalGeneration'],
        text_config: {
          model_type: 'qwen3_5_text',
          layer_types: ['linear_attention', 'full_attention'],
        },
        vision_config: { hidden_size: 1152 },
        image_token_id: 248056,
        video_token_id: 248057,
      },
      {
        format: 'jang',
        architecture: { has_vision: true, has_ssm: true },
        capabilities: {
          family: 'qwen3_5',
          modality: 'vision',
          supports_vision: true,
          supports_video: true,
        },
        runtime: {
          status: 'runtime_verified',
          vision_verified: true,
          video_verified: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('qwen3.5')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.forceTextOnly).toBeUndefined()
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
    expect(detected.usePagedCache).toBe(false)
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
    expect(detected.usePagedCache).toBe(false)
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
    expect(detected.usePagedCache).toBe(false)
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

  it('routes Mistral Medium 3.5 through its implemented text runtime', () => {
    const dir = makeModelDir(
      {
        model_type: 'mistral3',
        architectures: ['Mistral3ForConditionalGeneration'],
        text_config: { model_type: 'ministral3' },
        vision_config: { model_type: 'pixtral' },
      },
      {
        format: 'mxfp4',
        architecture: { has_vision: true },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('mistral3')
    expect(detected.isMultimodal).toBe(false)
    expect(detected.forceTextOnly).toBe(true)
    expect(detected.usePagedCache).toBe(false)
    expect(detected.architectureHints).toMatchObject({
      runtimeScope: 'text_only_until_pixtral_processor_is_wired',
      vlRuntimeAvailable: false,
    })
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
    { modelType: 'laguna', family: 'laguna', cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'qwen3' },
    { modelType: 'lfm2', family: 'lfm2', cacheType: 'hybrid', cacheSubtype: 'lfm2_moe_hybrid_ssm', toolParser: 'lfm2', reasoningParser: 'qwen3' },
    { modelType: 'lfm2_moe', family: 'lfm2', cacheType: 'hybrid', cacheSubtype: 'lfm2_moe_hybrid_ssm', toolParser: 'lfm2', reasoningParser: 'qwen3' },
    { modelType: 'ministral3', family: 'ministral3', cacheType: 'kv', toolParser: 'mistral' },
    { modelType: 'mistral3', family: 'mistral3', cacheType: 'kv', toolParser: 'mistral', isMultimodal: true },
    { modelType: 'mistral4', family: 'mistral4', cacheType: 'kv', toolParser: 'mistral', reasoningParser: 'mistral' },
    { modelType: 'mimo_v2', family: 'mimo_v2', cacheType: 'kv', toolParser: 'xml_function', reasoningParser: 'think_xml', isMultimodal: true },
    { modelType: 'nanbeige', family: 'nanbeige', cacheType: 'kv', toolParser: 'xml_function', reasoningParser: 'qwen3', isMultimodal: false },
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
      if (row.cacheType === 'hybrid' || row.cacheType === 'mamba') {
        // In-RAM paged cache is retired for every family: no hybrid/SSM/Mamba
        // registry entry may claim the paged capability any more.
        expect(detected.usePagedCache).toBe(false)
      }
      if (row.modelType === 'lfm2_moe') {
        expect(detected.supportsThinking).toBe(true)
        expect(detected.supportsInstructMode).toBe(false)
        expect(detected.architectureHints).toMatchObject({
          attentionArch: 'hybrid_ssm_attention',
          cacheSchema: 'hybrid_ssm_v1',
          ssmCompanionCache: true,
          attentionKvStorageQuantization: true,
        })
      }
    })
  }

  it('reports Nanbeige text/protocol/thinking/EOS/cache-slot truth without native MTP', () => {
    const dir = makeModelDir(
      {
        model_type: 'nanbeige',
        num_hidden_layers: 22,
        num_loops: 2,
        jang_runtime: {
          cache_layout: 'looped_kv_v1',
          cache_slots: 44,
        },
      },
      {
        weight_format: 'affine',
        capabilities: {
          family: 'nanbeige',
          cache_type: 'kv',
          reasoning_parser: 'qwen3',
          tool_parser: 'xml_function',
          supports_thinking: true,
          think_in_template: true,
          modality: 'text',
        },
        runtime: {
          num_loops: 2,
          cache_slots: 44,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected).toMatchObject({
      family: 'nanbeige',
      cacheType: 'kv',
      toolParser: 'xml_function',
      reasoningParser: 'qwen3',
      supportsThinking: true,
      thinkInTemplate: true,
      defaultEnableThinking: true,
      isMultimodal: false,
      architectureHints: {
        cacheSchema: 'looped_kv_v1',
        numLoops: 2,
        cacheSlots: 44,
      },
    })
    expect(detected.nativeMtp).toBeUndefined()
  })

  it('keeps unstamped Laguna reasoning available while Auto follows the family fallback', () => {
    const dir = makeModelDir({ model_type: 'laguna' })
    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('laguna')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.reasoningParser).toBe('qwen3')
    expect(detected.defaultEnableThinking).toBe(false)
  })

  it('identifies Laguna XS 2.1 from authoritative JANG source metadata', () => {
    const dir = makeModelDir(
      {
        model_type: 'laguna',
        _name_or_path: 'Laguna-S-2.1-JANG_2L',
      },
      {
        source_model: { name: 'Laguna-XS-2.1' },
      },
    )

    expect(detectModelConfigFromDir(dir).architectureHints).toMatchObject({
      lagunaVariant: 'xs-2.1',
    })
  })

  it('lets authoritative JANG source metadata suppress a stale XS config name', () => {
    const dir = makeModelDir(
      {
        model_type: 'laguna',
        _name_or_path: 'Laguna-XS-2.1-JANG_4M',
      },
      {
        source_model: { name: 'Laguna-S-2.1' },
      },
    )

    expect(detectModelConfigFromDir(dir).architectureHints?.lagunaVariant).toBeUndefined()
  })

  it('uses config _name_or_path as the Laguna XS 2.1 compatibility fallback', () => {
    const dir = makeModelDir({
      model_type: 'laguna',
      _name_or_path: 'poolside/Laguna-XS-2.1-JANG_6M',
    })

    expect(detectModelConfigFromDir(dir).architectureHints).toMatchObject({
      lagunaVariant: 'xs-2.1',
    })
  })

  it('does not infer a Laguna variant for a different model family', () => {
    const dir = makeModelDir({
      model_type: 'qwen3',
      _name_or_path: 'poolside/Laguna-XS-2.1-JANG_4M',
    })

    expect(detectModelConfigFromDir(dir).architectureHints?.lagunaVariant).toBeUndefined()
  })

  it('classifies Laguna full/sliding attention without changing its KV cache policy', () => {
    const dir = makeModelDir({
      model_type: 'laguna',
      num_hidden_layers: 3,
      layer_types: [
        'full_attention',
        'sliding_attention',
        'sliding_attention',
      ],
      sliding_window: 512,
    })

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('laguna')
    expect(detected.cacheType).toBe('kv')
    expect(detected.cacheSubtype).toBeUndefined()
    expect(detected.architectureHints).toMatchObject({
      attentionArch: 'full_and_sliding_kv',
      cacheSchema: 'mixed_swa_kv_v1',
      selectiveTurboQuantKv: false,
      nativeCacheDefault: true,
    })
  })

  it('does not classify Laguna as mixed-SWA without both full and sliding layers', () => {
    const dir = makeModelDir({
      model_type: 'laguna',
      num_hidden_layers: 2,
      layer_types: ['full_attention', 'full_attention'],
    })

    expect(detectModelConfigFromDir(dir).architectureHints).toBeUndefined()
  })

  it('does not classify Laguna selective TQ from a truncated per-layer layout', () => {
    const dir = makeModelDir({
      model_type: 'laguna',
      num_hidden_layers: 48,
      layer_types: ['full_attention', 'sliding_attention'],
    })

    expect(detectModelConfigFromDir(dir).architectureHints).toBeUndefined()
  })

  it('preserves a Laguna bundle-owned TurboQuant disable alongside mixed topology', () => {
    const dir = makeModelDir(
      {
        model_type: 'laguna',
        num_hidden_layers: 2,
        layer_types: ['full_attention', 'sliding_attention'],
      },
      {
        turboquant: { enabled: false },
      },
    )

    expect(detectModelConfigFromDir(dir).architectureHints).toMatchObject({
      attentionArch: 'full_and_sliding_kv',
      cacheSchema: 'mixed_swa_kv_v1',
      selectiveTurboQuantKv: false,
      nativeCacheDefault: true,
      loaderTurboQuantEnabled: false,
    })
  })

  it('uses Laguna JANG chat metadata to default Auto reasoning on for S-2.1 bundles', () => {
    const dir = makeModelDir(
      { model_type: 'laguna' },
      {
        chat: {
          reasoning: {
            supported: true,
            parser: 'deepseek_r1',
            default_enabled: true,
            default_mode: 'think',
          },
          template_kwargs_defaults: {
            enable_thinking: true,
          },
          tool_calling: {
            supported: true,
            parser: 'glm47',
          },
          sampling_defaults: {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 20,
          },
        },
        capabilities: {
          family: 'laguna',
          cache_type: 'kv',
          reasoning_parser: 'deepseek_r1',
          tool_parser: 'glm47',
          supports_tools: true,
          supports_thinking: true,
          think_in_template: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('laguna')
    expect(detected.toolParser).toBe('glm47')
    expect(detected.reasoningParser).toBe('deepseek_r1')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.thinkInTemplate).toBe(true)
    expect(detected.defaultEnableThinking).toBe(true)
  })

  it('uses Laguna top-level JANG chat metadata even when capabilities are absent', () => {
    const dir = makeModelDir(
      { model_type: 'laguna' },
      {
        chat: {
          reasoning: {
            supported: true,
            parser: 'deepseek_r1',
            default_enabled: true,
            default_mode: 'think',
          },
          template_kwargs_defaults: {
            enable_thinking: true,
          },
          tool_calling: {
            supported: true,
            parser: 'glm47',
          },
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('laguna')
    expect(detected.toolParser).toBe('glm47')
    expect(detected.enableAutoToolChoice).toBe(true)
    expect(detected.reasoningParser).toBe('deepseek_r1')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.thinkInTemplate).toBe(true)
    expect(detected.defaultEnableThinking).toBe(true)
  })

  it('lets Laguna JANG template defaults override reasoning.default_enabled when present', () => {
    const dir = makeModelDir(
      { model_type: 'laguna' },
      {
        chat: {
          reasoning: { supported: true, default_enabled: true },
          template_kwargs_defaults: { enable_thinking: false },
        },
        capabilities: {
          family: 'laguna',
          supports_thinking: true,
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.defaultEnableThinking).toBe(false)
  })

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

describe('detectModelConfigFromDir supportsThinkingBudget capability', () => {
  // Ground truth: ONLY families whose engine honors a top-level
  // max_thinking_tokens reasoning-phase cap are flagged. Derived from the engine
  // _REASONING_ANSWER_PASS_FAMILIES set plus the minimax_m3 thinking-cap branch.
  const budgetFamilies: Array<{ modelType: string; family: string }> = [
    { modelType: 'qwen3_5', family: 'qwen3.5' },
    { modelType: 'qwen3_5_moe', family: 'qwen3.5-moe' },
    { modelType: 'gemma4', family: 'gemma4' },
    { modelType: 'gemma4_text', family: 'gemma4-text' },
    { modelType: 'minimax_m2', family: 'minimax' },
    { modelType: 'minimax_m3', family: 'minimax_m3' },
    { modelType: 'openpangu_v2', family: 'openpangu_v2' },
    { modelType: 'hy_v3', family: 'hy3' },
  ]

  for (const { modelType, family } of budgetFamilies) {
    it(`flags supportsThinkingBudget for ${family} (model_type=${modelType})`, () => {
      const dir = makeModelDir({ model_type: modelType })
      const detected = detectModelConfigFromDir(dir)
      expect(detected.family).toBe(family)
      expect(detected.supportsThinkingBudget).toBe(true)
    })
  }

  // Reasoning-parser families with NO engine-side thinking-budget behavior must
  // NOT be flagged (dsv4 uses reasoning_effort; step-3.7-flash has no cap).
  const nonBudgetFamilies: Array<{ modelType: string; family: string }> = [
    { modelType: 'deepseek_v4', family: 'deepseek-v4' },
    { modelType: 'step3p7', family: 'step-3.7-flash' },
  ]

  for (const { modelType, family } of nonBudgetFamilies) {
    it(`does NOT flag supportsThinkingBudget for ${family} (model_type=${modelType})`, () => {
      const dir = makeModelDir({ model_type: modelType })
      const detected = detectModelConfigFromDir(dir)
      expect(detected.family).toBe(family)
      // Was `undefined` while each row declared the flag by hand and these two
      // simply omitted it. It is now an explicit `false` derived from the one
      // shared THINKING_BUDGET_FAMILIES list — a stronger statement, not a
      // weaker one, and what the ChatSettings `=== true` gate has always read.
      expect(detected.supportsThinkingBudget).toBe(false)
    })
  }

  it('reads a TOP-LEVEL reasoning stamp, as the Qwen bundles ship it', () => {
    // Qwen stamps put `reasoning` at the top level and leave `chat` holding
    // only sampling defaults, so an overlay that looked solely at
    // `chat.reasoning` contributed nothing for that whole vintage. Qwen3.8
    // accepts exactly low/medium/xhigh and its template RAISES on anything
    // else, so the panel must not keep offering high/max.
    const dir = makeModelDir(
      { model_type: 'qwen3_5_moe' },
      {
        chat: { sampling_defaults: { temperature: 1.0 } },
        reasoning: {
          supported: true,
          parser: 'qwen3',
          supported_reasoning_efforts: ['low', 'medium', 'xhigh'],
          default_reasoning_effort: 'xhigh',
        },
      },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.supportedReasoningEfforts).toEqual(['low', 'medium', 'xhigh'])
    expect(detected.supportedReasoningEfforts).not.toContain('high')
    expect(detected.supportedReasoningEfforts).not.toContain('max')
    expect(detected.defaultReasoningEffort).toBe('xhigh')
  })

  it('leaves an unstamped bundle with no effort control at all', () => {
    // Qwen3.6 genuinely has no tiers; absence must not read as "unconstrained".
    const dir = makeModelDir(
      { model_type: 'qwen3_5_moe' },
      { reasoning: { supported: true, parser: 'qwen3', default: 'on' } },
    )

    const detected = detectModelConfigFromDir(dir)

    expect(detected.supportedReasoningEfforts).toBeUndefined()
  })

  it('derives the DSV4-0731 reasoning contract from the selected bundle sidecar', () => {
    const dir = makeModelDir(
      { model_type: 'deepseek_v4' },
      {
        chat: {
          reasoning: {
            supported: true,
            modes: ['chat', 'thinking'],
            default_mode: 'thinking',
            default_effort: 'low',
            reasoning_effort_levels: ['low', 'high', 'max'],
          },
          tool_calling: {
            supported: true,
            parser: 'dsml',
          },
        },
      },
    )
    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('deepseek-v4')
    expect(detected.reasoningParser).toBe('deepseek_r1')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.supportsInstructMode).toBe(true)
    expect(detected.defaultEnableThinking).toBe(true)
    expect(detected.defaultReasoningEffort).toBe('low')
    expect(detected.supportedReasoningEfforts).toEqual(['low', 'high', 'max'])
    // Explicit false now, derived from the shared THINKING_BUDGET_FAMILIES
    // list rather than omitted per-row — DSV4 keys on reasoning_effort, not a
    // token budget, so a budget control would send an inert value.
    expect(detected.supportsThinkingBudget).toBe(false)
  })

  it('does not invent DSV4 effort levels or a default for an unstamped bundle', () => {
    const dir = makeModelDir({ model_type: 'deepseek_v4' })
    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('deepseek-v4')
    expect(detected.supportedReasoningEfforts).toBeUndefined()
    expect(detected.defaultReasoningEffort).toBeUndefined()
    expect(detected.defaultEnableThinking).toBeUndefined()
    expect(detected.supportsInstructMode).toBeUndefined()
  })

  it('keeps Muse paged-OFF like every family — SSD block-disk L2 is the only cache tier', () => {
    // This test used to pin Muse paged-ON to match the engine's
    // _PAGED_MLLM_EXEMPT_FAMILIES set. That exemption is moot now: in-RAM paged
    // cache is OFF for EVERY family and the app always spawns --no-paged-cache,
    // so the registry must not single Muse out with a paged capability.
    const dir = makeModelDir({ model_type: 'muse_glimmer' })
    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('muse-glimmer')
    expect(detected.isMultimodal).toBe(true)
    expect(detected.usePagedCache).toBe(false)
  })

  it('surfaces Muse reasoning_strength modes as effort buttons (no reasoning_effort_levels)', () => {
    // Muse Glimmer declares control: reasoning_strength with modes but NO
    // reasoning_effort_levels; the panel must still show the four strength
    // buttons (previously only Auto/On/Off rendered).
    const dir = makeModelDir(
      { model_type: 'muse_glimmer' },
      {
        chat: {
          reasoning: {
            supported: true,
            parser: 'muse_glimmer',
            control: 'reasoning_strength',
            default_mode: 'high',
            modes: ['low', 'medium', 'high', 'xhigh'],
          },
        },
      },
    )
    const detected = detectModelConfigFromDir(dir)

    expect(detected.family).toBe('muse-glimmer')
    expect(detected.reasoningParser).toBe('muse_glimmer')
    expect(detected.supportsThinking).toBe(true)
    expect(detected.supportedReasoningEfforts).toEqual(['low', 'medium', 'high', 'xhigh'])
    expect(detected.defaultReasoningEffort).toBe('high')
    // Muse has no chat/instruct mode — the modes list is purely strength levels.
    expect(detected.supportsInstructMode).toBe(false)
  })
})

describe('sidecar evidence rule is shared across formats', () => {
  it('a nested native_mtp block without validated: true may recommend depth 1 but not deeper', () => {
    const dir = makeModelDir(
      { model_type: 'qwen3_5', text_config: { model_type: 'qwen3_5_text', mtp_num_hidden_layers: 1 } },
      { format: 'jang', mtp: { kept: true, enabled: true, num_layers: 1 }, capabilities: { family: 'qwen3_5', cache_type: 'hybrid' } },
    )
    writeFileSync(join(dir, 'model.safetensors.index.json'), JSON.stringify({ weight_map: { 'mtp.fc.weight': 'model.safetensors' } }))
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({ native_mtp: { best_depth: 3 } }))
    expect(detectModelConfigFromDir(dir).nativeMtp?.depthSource).not.toBe('vmlx_mtp_tuning.json:native_mtp.best_depth')
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({ native_mtp: { best_depth: 1 } }))
    expect(detectModelConfigFromDir(dir).nativeMtp).toMatchObject({ depth: 1, depthSource: 'vmlx_mtp_tuning.json:native_mtp.best_depth' })
    writeFileSync(join(dir, 'vmlx_mtp_tuning.json'), JSON.stringify({ best_native_mtp_depth: { best_depth: 2, validated: true } }))
    expect(detectModelConfigFromDir(dir).nativeMtp).toMatchObject({ depth: 2, depthSource: 'vmlx_mtp_tuning.json:best_native_mtp_depth.best_depth' })
  })
})
