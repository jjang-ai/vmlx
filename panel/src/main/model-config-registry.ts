/**
 * Model configuration registry for auto-detecting tool/reasoning parsers.
 * Mirrors the Python model_configs.py patterns for client-side detection.
 *
 * Detection: reads model's config.json model_type field and maps to a registered family.
 * No name-based regex detection — config.json is authoritative.
 * Users can always override auto-detected values via Server Settings UI.
 */

import { readFileSync, existsSync, readdirSync, statSync } from 'fs'
import { join } from 'path'
import { homedir } from 'os'

/**
 * Resolve an HF repo id (e.g. `mlx-community/gemma-4-e2b-it-4bit`) to the
 * newest local snapshot directory inside the HuggingFace hub cache.
 *
 * The panel stores `chat.modelPath` as whatever the user downloaded/loaded
 * with, which for HF hub downloads is the bare repo id — NOT a filesystem
 * path. `detectModelConfigFromDir()` used to try to read
 * `<repo id>/config.json` directly, which always fell through to
 * `DEFAULT_CONFIG` (including `isMultimodal: false`). The panel then stripped
 * every attached image on its way to the server, so vision requests on HF-
 * downloaded Gemma 4 E2B / E4B / 26B arrived as text-only.
 *
 * This helper mirrors `huggingface_hub`'s cache layout:
 *   ~/.cache/huggingface/hub/models--{owner}--{repo}/snapshots/{sha}/
 * Returns null if the repo isn't in the cache or the cache layout is unusable.
 */
function resolveHuggingFaceRepoToLocalPath(repoId: string): string | null {
  if (!repoId || !repoId.includes('/') || repoId.startsWith('/')) return null
  // An HF repo id is "owner/name"; HF transforms slashes in the folder name
  // to `--`, so "mlx-community/foo" becomes "models--mlx-community--foo".
  const folder = 'models--' + repoId.replace(/\//g, '--')
  const hubDir = join(homedir(), '.cache', 'huggingface', 'hub', folder, 'snapshots')
  if (!existsSync(hubDir)) return null
  try {
    const entries = readdirSync(hubDir)
      .map(name => {
        const full = join(hubDir, name)
        try { return { full, mtime: statSync(full).mtimeMs } } catch { return null }
      })
      .filter((x): x is { full: string; mtime: number } => x !== null)
    if (entries.length === 0) return null
    entries.sort((a, b) => b.mtime - a.mtime)
    return entries[0].full
  } catch {
    return null
  }
}

interface ModelConfig {
  familyName: string
  cacheType: 'kv' | 'mamba' | 'hybrid' | 'rotating_kv'
  toolParser?: string
  reasoningParser?: string
  supportsThinking?: boolean
  thinkInTemplate?: boolean
  defaultEnableThinking?: boolean
  usePagedCache?: boolean
  enableAutoToolChoice?: boolean
  isMultimodal?: boolean
  description: string
  priority: number
}

export interface DetectedConfig {
  family: string
  toolParser?: string
  reasoningParser?: string
  supportsThinking?: boolean
  thinkInTemplate?: boolean
  defaultEnableThinking?: boolean
  cacheType: string
  usePagedCache: boolean
  enableAutoToolChoice: boolean
  isMultimodal: boolean
  forceTextOnly?: boolean
  isTurboQuant?: boolean
  nativeMtp?: {
    supported: boolean
    depth: number
    depthSource?: string
    runtimeScope: 'text' | 'text+vl'
    requiresDeterministicSampling: boolean
  }
  description: string
  maxContextLength?: number
}

const CONFIG_BY_FAMILY = new Map<string, Omit<ModelConfig, 'pattern' | 'familyName'>>()

function registerFamily(familyName: string, config: Omit<ModelConfig, 'familyName'>) {
  CONFIG_BY_FAMILY.set(familyName, config)
}

// ZAYA / Zyphra: CCA attention + top-1 MoE. Live 2026-05-18 gates showed
// thinking-on can stay in an open <think> rail without a visible answer, so
// auto-detection keeps reasoning disabled while preserving zaya_xml tools and
// the typed CCA cache contract.
registerFamily('zaya', { cacheType: 'hybrid', toolParser: 'zaya_xml', reasoningParser: 'qwen3', supportsThinking: true, thinkInTemplate: false, defaultEnableThinking: false, usePagedCache: true, enableAutoToolChoice: true, description: 'ZAYA CCA hybrid MoE', priority: 3 })
// ZAYA1-VL is detected separately so the UI does not fall through to generic
// VLM defaults. The Python runtime uses a typed CCA cache contract, so panel
// sessions must start on paged cache instead of the legacy prefix-cache backend.
registerFamily('zaya1-vl', { cacheType: 'hybrid', toolParser: 'zaya_xml', reasoningParser: 'qwen3', supportsThinking: true, thinkInTemplate: false, defaultEnableThinking: false, usePagedCache: true, enableAutoToolChoice: true, isMultimodal: true, description: 'ZAYA1-VL CCA hybrid vision-language', priority: 3 })

// Qwen
// Qwen 3.5 dense and MoE share model_types with VL variants — VL detection
// relies on config.json vision_config, not the family's isMultimodal flag.
registerFamily('qwen3.5', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: false, description: 'Qwen 3.5 (dense)', priority: 4 })
registerFamily('qwen3.5-moe', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: false, description: 'Qwen 3.5 MoE', priority: 4 })
registerFamily('qwen3-next', { cacheType: 'mamba', toolParser: 'qwen', reasoningParser: 'qwen3', usePagedCache: true, enableAutoToolChoice: true, description: 'Qwen 3 Next (hybrid Mamba)', priority: 1 })
registerFamily('qwen3-vl', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: true, description: 'Qwen 3 Vision-Language', priority: 5 })
registerFamily('qwen3-moe', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'Qwen 3 MoE', priority: 5 })
registerFamily('qwen3', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'Qwen 3 / QwQ', priority: 10 })
registerFamily('qwen2-vl', { cacheType: 'kv', toolParser: 'qwen', enableAutoToolChoice: true, isMultimodal: true, description: 'Qwen 2 Vision-Language', priority: 10 })
registerFamily('qwen2', { cacheType: 'kv', toolParser: 'qwen', enableAutoToolChoice: true, description: 'Qwen 2', priority: 20 })
registerFamily('qwen-mamba', { cacheType: 'mamba', toolParser: 'qwen', usePagedCache: true, description: 'Qwen Mamba', priority: 5 })

// Llama
registerFamily('llama4', { cacheType: 'kv', toolParser: 'llama', enableAutoToolChoice: true, description: 'Llama 4', priority: 5 })
registerFamily('llama3', { cacheType: 'kv', toolParser: 'llama', enableAutoToolChoice: true, description: 'Llama 3', priority: 10 })
registerFamily('llama', { cacheType: 'kv', toolParser: 'llama', description: 'Llama', priority: 50 })

// Mistral/Mixtral/Devstral/Codestral
registerFamily('mistral4', { cacheType: 'kv', toolParser: 'mistral', reasoningParser: 'mistral', enableAutoToolChoice: true, description: 'Mistral 4 (MLA/MoE reasoning)', priority: 4 })
registerFamily('mistral3', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, isMultimodal: true, description: 'Mistral 3 / Pixtral-style VLM wrapper', priority: 5 })
registerFamily('ministral3', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Ministral 3 text decoder', priority: 5 })
registerFamily('devstral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Devstral (Mistral coding)', priority: 5 })
registerFamily('codestral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Codestral (Mistral coding)', priority: 5 })
registerFamily('pixtral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, isMultimodal: true, description: 'Pixtral Vision', priority: 5 })
registerFamily('mixtral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Mixtral MoE', priority: 10 })
registerFamily('mistral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Mistral', priority: 20 })

// DeepSeek
registerFamily('deepseek-v4', { cacheType: 'kv', toolParser: 'dsml', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'DeepSeek V4 Flash', priority: 4 })
registerFamily('deepseek-vl', { cacheType: 'kv', toolParser: 'deepseek', isMultimodal: true, description: 'DeepSeek-VL vision-language', priority: 5 })
registerFamily('deepseek-r1', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', description: 'DeepSeek R1', priority: 5 })
registerFamily('deepseek-v3', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'DeepSeek V3', priority: 5 })
registerFamily('deepseek-v2', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', description: 'DeepSeek V2', priority: 10 })
registerFamily('deepseek', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', description: 'DeepSeek', priority: 50 })

// GLM
registerFamily('gpt-oss', { cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'openai_gptoss', enableAutoToolChoice: true, description: 'GPT-OSS (Harmony reasoning)', priority: 3 })
registerFamily('glm47-flash', { cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'openai_gptoss', enableAutoToolChoice: true, description: 'GLM-4.7 Flash (reasoning)', priority: 3 })
registerFamily('glm5', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'GLM-5.1 / GLM MoE DSA', priority: 5 })
registerFamily('glm47', { cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'GLM-Z1 (deepseek_r1 reasoning)', priority: 5 })
registerFamily('glm4', { cacheType: 'kv', toolParser: 'glm47', enableAutoToolChoice: true, description: 'GLM-4 (tools only)', priority: 20 })

// Gemma
registerFamily('medgemma', { cacheType: 'kv', isMultimodal: true, description: 'Google MedGemma (medical multimodal)', priority: 3 })
registerFamily('paligemma', { cacheType: 'kv', isMultimodal: true, description: 'Google PaliGemma', priority: 5 })
registerFamily('gemma4', { cacheType: 'kv', toolParser: 'gemma4', reasoningParser: 'gemma4', enableAutoToolChoice: true, isMultimodal: true, description: 'Gemma 4 (multimodal)', priority: 5 })
registerFamily('gemma4-text', { cacheType: 'kv', toolParser: 'gemma4', reasoningParser: 'gemma4', enableAutoToolChoice: true, description: 'Gemma 4 (text-only)', priority: 4 })
registerFamily('gemma3', { cacheType: 'kv', toolParser: 'gemma3', enableAutoToolChoice: true, isMultimodal: true, description: 'Gemma 3 (multimodal)', priority: 10 })
registerFamily('gemma3-text', { cacheType: 'kv', toolParser: 'gemma3', enableAutoToolChoice: true, description: 'Gemma 3 (text-only)', priority: 8 })
registerFamily('gemma3n', { cacheType: 'kv', toolParser: 'gemma3', enableAutoToolChoice: true, isMultimodal: true, description: 'Gemma 3n (multimodal)', priority: 10 })
registerFamily('gemma3n-text', { cacheType: 'kv', toolParser: 'gemma3', enableAutoToolChoice: true, description: 'Gemma 3n (text-only)', priority: 8 })
registerFamily('gemma2', { cacheType: 'kv', description: 'Gemma 2', priority: 15 })
registerFamily('gemma', { cacheType: 'kv', description: 'Gemma', priority: 30 })

// Phi
registerFamily('phi4-reasoning', { cacheType: 'kv', toolParser: 'hermes', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'Phi 4 Reasoning', priority: 2 })
registerFamily('phi4-multimodal', { cacheType: 'kv', isMultimodal: true, description: 'Phi 4 Multimodal', priority: 2 })
registerFamily('phi4', { cacheType: 'kv', toolParser: 'hermes', enableAutoToolChoice: true, description: 'Phi 4', priority: 10 })
registerFamily('phi3-vision', { cacheType: 'kv', isMultimodal: true, description: 'Phi 3 Vision', priority: 8 })
registerFamily('phi3', { cacheType: 'kv', description: 'Phi 3', priority: 20 })

// Hermes
registerFamily('hermes', { cacheType: 'kv', toolParser: 'hermes', enableAutoToolChoice: true, description: 'Hermes', priority: 30 })

// Nemotron
registerFamily('nemotron', { cacheType: 'kv', toolParser: 'nemotron', reasoningParser: 'deepseek_r1', description: 'Nemotron', priority: 10 })
registerFamily('nemotron-h', { cacheType: 'hybrid', toolParser: 'nemotron', reasoningParser: 'deepseek_r1', usePagedCache: true, description: 'Nemotron Hybrid', priority: 10 })

// Poolside / Laguna
registerFamily('laguna', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', supportsThinking: true, thinkInTemplate: true, enableAutoToolChoice: true, description: 'Laguna / Poolside coding model', priority: 10 })

// Jamba
registerFamily('jamba', { cacheType: 'hybrid', usePagedCache: true, description: 'Jamba (Hybrid)', priority: 10 })

// Cohere
registerFamily('command-r-plus', { cacheType: 'kv', description: 'Command R+', priority: 10 })
registerFamily('command-r', { cacheType: 'kv', description: 'Command R', priority: 20 })

// Granite
registerFamily('granite', { cacheType: 'kv', toolParser: 'granite', enableAutoToolChoice: true, description: 'Granite', priority: 20 })
registerFamily('granitemoehybrid', { cacheType: 'hybrid', toolParser: 'granite', enableAutoToolChoice: true, usePagedCache: true, description: 'Granite MoE Hybrid', priority: 10 })

// Functionary
registerFamily('functionary', { cacheType: 'kv', toolParser: 'functionary', enableAutoToolChoice: true, description: 'Functionary', priority: 20 })

// MiniMax
registerFamily('minimax', { cacheType: 'kv', toolParser: 'minimax', reasoningParser: 'minimax_m2', enableAutoToolChoice: true, description: 'MiniMax', priority: 20 })

// Ling / Bailing hybrid: MLA softmax layers plus linear-attention/SSM-style
// companion state. Eric directive 2026-05-11: treat Ling chat output as plain
// content. Keep DeepSeek tool parsing, but do not advertise a reasoning parser
// or thinking capability even when stale JANG sidecars claim deepseek_r1.
registerFamily('ling', { cacheType: 'hybrid', toolParser: 'deepseek', supportsThinking: false, thinkInTemplate: false, usePagedCache: true, enableAutoToolChoice: true, description: 'Ling / Bailing hybrid', priority: 20 })

// Tencent Hy3-preview: text-only dense GQA KV + MoE. The chat template uses
// reasoning_effort=no_think|low|high, so Python normalizes the UI thinking
// toggle into Hy3's effort field before render.
registerFamily('hy3', { cacheType: 'kv', toolParser: 'hunyuan', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'Tencent Hy3-preview', priority: 4 })

// StepFun
registerFamily('step-vl', { cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: true, description: 'StepFun Step-1V Vision-Language', priority: 3 })
registerFamily('step-3.5-flash', { cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'StepFun Step-3.5-Flash (MoE)', priority: 5 })
registerFamily('step', { cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'StepFun Step models', priority: 30 })

// xLAM (Salesforce)
registerFamily('xlam', { cacheType: 'kv', toolParser: 'xlam', enableAutoToolChoice: true, description: 'xLAM', priority: 20 })

// Kimi/Moonshot
registerFamily('kimi-k25', { cacheType: 'kv', toolParser: 'kimi', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, isMultimodal: true, description: 'Kimi K2.5/K2.6 Vision-Language', priority: 5 })
registerFamily('kimi-k2', { cacheType: 'kv', toolParser: 'kimi', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'Kimi K2 (MoE)', priority: 5 })
registerFamily('kimi', { cacheType: 'kv', toolParser: 'kimi', enableAutoToolChoice: true, description: 'Kimi/Moonshot', priority: 20 })

// InternLM
registerFamily('internlm3', { cacheType: 'kv', description: 'InternLM 3', priority: 10 })
registerFamily('internlm', { cacheType: 'kv', description: 'InternLM', priority: 30 })

// EXAONE
registerFamily('exaone', { cacheType: 'kv', description: 'EXAONE', priority: 20 })

// OLMo
registerFamily('olmo', { cacheType: 'kv', description: 'OLMo', priority: 20 })

// Liquid / hybrid SSM
registerFamily('lfm2', { cacheType: 'hybrid', usePagedCache: true, description: 'Liquid LFM2 / LFM2-MoE hybrid', priority: 10 })

// StarCoder / StableLM / Baichuan
registerFamily('starcoder', { cacheType: 'kv', description: 'StarCoder', priority: 30 })
registerFamily('stablelm', { cacheType: 'kv', description: 'StableLM', priority: 30 })
registerFamily('baichuan', { cacheType: 'kv', description: 'Baichuan', priority: 30 })

// VLM / MLLM models
registerFamily('yi-vl', { cacheType: 'kv', isMultimodal: true, description: 'Yi Vision-Language', priority: 15 })
registerFamily('llava', { cacheType: 'kv', isMultimodal: true, description: 'LLaVA vision-language', priority: 20 })
registerFamily('idefics', { cacheType: 'kv', isMultimodal: true, description: 'Idefics vision-language', priority: 5 })
registerFamily('molmo', { cacheType: 'kv', isMultimodal: true, description: 'Molmo multimodal', priority: 20 })
registerFamily('cogvlm', { cacheType: 'kv', isMultimodal: true, description: 'CogVLM vision-language', priority: 20 })
registerFamily('internvl', { cacheType: 'kv', isMultimodal: true, description: 'InternVL vision-language', priority: 15 })
registerFamily('minicpm-v', { cacheType: 'kv', isMultimodal: true, description: 'MiniCPM-V vision', priority: 20 })
registerFamily('florence', { cacheType: 'kv', isMultimodal: true, description: 'Florence vision', priority: 20 })
registerFamily('got-ocr', { cacheType: 'kv', isMultimodal: true, description: 'GOT-OCR2 document/scene OCR', priority: 15 })
registerFamily('smolvlm', { cacheType: 'kv', isMultimodal: true, description: 'SmolVLM', priority: 20 })
registerFamily('internlm-xcomposer', { cacheType: 'kv', isMultimodal: true, description: 'InternLM-XComposer', priority: 8 })

// Pure SSM
registerFamily('falcon-h1', { cacheType: 'hybrid', usePagedCache: true, description: 'Falcon H1 hybrid SSM/attention', priority: 5 })
registerFamily('falcon-mamba', { cacheType: 'mamba', usePagedCache: true, description: 'Falcon Mamba (SSM)', priority: 5 })
registerFamily('mamba', { cacheType: 'mamba', usePagedCache: true, description: 'Mamba SSM', priority: 30 })
registerFamily('rwkv', { cacheType: 'mamba', usePagedCache: true, description: 'RWKV', priority: 30 })

/**
 * Map model_type values from config.json to registry family names.
 * This is the authoritative detection method — model_type reflects the actual
 * architecture regardless of what the model is named (e.g., a Qwen3 fine-tune
 * named "Nemotron-Orchestrator" has model_type="qwen3").
 */
/**
 * Exhaustive map of config.json model_type → registry family.
 * Includes all known variants, MoE suffixes, VL suffixes, etc.
 * If a model_type isn't here, falls back to name regex (line 210+).
 * Users can always override via manual parser selection in Server Settings.
 */
const MODEL_TYPE_TO_FAMILY: Record<string, string> = {
  // ── Qwen family ──
  'zaya': 'zaya',
  'zaya1_vl': 'zaya1-vl',
  'qwen3_5': 'qwen3.5',
  'qwen3_5_moe': 'qwen3.5-moe',
  'qwen3_5_moe_text': 'qwen3.5-moe', // Qwen3.6-35B-A3B inner text_config model_type
  'qwen3': 'qwen3',
  'qwen3_next': 'qwen3-next',
  'qwen3_moe': 'qwen3-moe',
  'qwen3_vl': 'qwen3-vl',
  'qwen3_vl_moe': 'qwen3-vl',
  'qwen2': 'qwen2',
  'qwen2_moe': 'qwen2',
  'qwen2_vl': 'qwen2-vl',
  'qwen2_5_vl': 'qwen2-vl',
  'qwen': 'qwen2',
  'qwen_mamba': 'qwen-mamba',
  // ── Llama family ──
  'llama': 'llama3',
  'llama4': 'llama4',
  // ── Mistral family ──
  'mistral': 'mistral',
  'mixtral': 'mixtral',
  'pixtral': 'pixtral',
  'mistral3': 'mistral3',
  'mistral4': 'mistral4',
  'ministral3': 'ministral3',
  'codestral': 'codestral',
  'devstral': 'devstral',
  'codestral_mamba': 'mamba',
  // ── DeepSeek family ──
  'deepseek_v4': 'deepseek-v4',
  'deepseek_v3': 'deepseek-v3',
  'deepseek_v32': 'deepseek-v3',
  'deepseek_v2': 'deepseek-v2',
  'deepseek_vl': 'deepseek-vl',
  'deepseek_vl2': 'deepseek-vl',
  'deepseek_vl_v2': 'deepseek-vl',
  'deepseek2': 'deepseek',
  'deepseek': 'deepseek',
  // ── Ling / Bailing family (inclusionAI / Ant Group) ──
  // Hybrid MLA + Lightning-Attn-2 (linear attention). Engine-side
  // model_configs.py registers `bailing_hybrid`/`bailing_moe_v2_5`
  // under canonical family `ling` with cache_type=hybrid, deepseek_r1
  // opt-in reasoning parser, and deepseek tool parser.
  'bailing_hybrid': 'ling',
  'bailing_moe_v2_5': 'ling',
  'bailing_moe_linear': 'ling',
  'bailing_moe': 'ling',
  // ── Tencent Hy3 ──
  'hy_v3': 'hy3',
  // ── GLM family ──
  'chatglm': 'glm4',
  'glm_moe_dsa': 'glm5',
  'glm4': 'glm4',
  'glm4_moe': 'glm47-flash',
  'glm4_moe_lite': 'glm47-flash',
  'glm': 'glm4',
  // ── GPT-OSS (Harmony protocol) — needs openai_gptoss reasoning, not deepseek_r1
  'gpt_oss': 'gpt-oss',
  // ── StepFun ──
  'step1v': 'step-vl',
  'step3p5': 'step-3.5-flash',
  'step': 'step',
  // ── Gemma family ──
  'gemma': 'gemma',
  'gemma2': 'gemma2',
  'gemma3': 'gemma3',
  'gemma3_text': 'gemma3-text',
  'gemma3n': 'gemma3n',
  'gemma3n_text': 'gemma3n-text',
  'gemma4': 'gemma4',
  'gemma4_text': 'gemma4-text',
  // ── Phi family ──
  'phi3': 'phi3',
  'phi3v': 'phi3-vision',
  'phi3small': 'phi3',
  'phi4': 'phi4',
  'phi4mm': 'phi4-multimodal',
  'phi4flash': 'phi4',
  'phi4_reasoning': 'phi4-reasoning',
  'phi': 'phi3',
  // ── MiniMax family ──
  'minimax': 'minimax',
  'minimax_m2': 'minimax',
  'minimax_m2_5': 'minimax',
  // ── Jamba / Mamba / SSM ──
  'jamba': 'jamba',
  'mamba': 'mamba',
  'mamba2': 'mamba',
  'falcon_h1': 'falcon-h1',
  'falcon_mamba': 'falcon-mamba',
  'rwkv': 'rwkv',
  'rwkv5': 'rwkv',
  'rwkv6': 'rwkv',
  'rwkv7': 'rwkv',
  // ── NVIDIA ──
  'nemotron': 'nemotron',
  'nemotron_h': 'nemotron-h',
  'nemotron_h_v2': 'nemotron-h',
  // ── IBM ──
  'granite': 'granite',
  'granite_moe': 'granite',
  'granitemoehybrid': 'granitemoehybrid',
  // ── Cohere ──
  'cohere': 'command-r',
  'cohere2': 'command-r',
  // ── Hermes (NousResearch) ──
  'hermes': 'hermes',
  // ── Kimi/Moonshot ──
  'kimi_k2': 'kimi-k2',
  'kimi_k25': 'kimi-k25',
  // ── EXAONE ──
  'exaone': 'exaone',
  'exaone3': 'exaone',
  // ── OLMo ──
  'olmo': 'olmo',
  'olmo2': 'olmo',
  // ── Liquid AI ──
  'lfm2': 'lfm2',
  'lfm2_moe': 'lfm2',
  // ── Laguna / Poolside ──
  'laguna': 'laguna',
  // ── Gemma extras ──
  'paligemma': 'paligemma',
  'paligemma2': 'paligemma',
  // ── MLLM / Vision-Language ──
  'llava': 'llava',
  'llava_next': 'llava',
  'idefics2': 'idefics',
  'idefics3': 'idefics',
  'cogvlm': 'cogvlm',
  'cogvlm2': 'cogvlm',
  'florence2': 'florence',
  'got_ocr2': 'got-ocr',
  'molmo': 'molmo',
  'minicpmv': 'minicpm-v',
  'smolvlm': 'smolvlm',
  'internvl_chat': 'internvl',
  // ── Others (architecture-compatible mappings) ──
  'starcoder2': 'starcoder',
  'stablelm': 'stablelm',
  'baichuan': 'baichuan',
  'internlm': 'internlm',
  'internlm2': 'internlm',
  'internlm3': 'internlm3',
  'internlm_xcomposer2': 'internlm-xcomposer',
  'yi': 'llama3',
  'orion': 'llama3',
}

const DEFAULT_CONFIG: DetectedConfig = {
  family: 'unknown',
  cacheType: 'kv',
  usePagedCache: true,
  enableAutoToolChoice: false,
  isMultimodal: false,
  description: 'Unknown model'
}

function configMarksTurboQuant(config: any): boolean {
  const candidates = [
    config?.weight_format,
    config?.format,
    config?.quantization?.weight_format,
    config?.quantization?.format,
  ]
  return candidates.some(value =>
    typeof value === 'string' && value.toLowerCase() === 'mxtq'
  )
}

function configDeclaresMedia(config: any): boolean {
  if (!config || typeof config !== 'object') return false
  for (const key of ['vision_config', 'audio_config', 'video_config']) {
    if (key in config && config[key] != null) return true
  }
  for (const key of [
    'image_token_id',
    'image_token_index',
    'video_token_id',
    'video_token_index',
    'audio_token_id',
    'audio_token_index',
  ]) {
    if (key in config && config[key] != null) return true
  }
  return false
}

function isMxtqJangConfig(jangCfg: any): boolean {
  if (!jangCfg || typeof jangCfg !== 'object') return false
  const quant = jangCfg.quantization && typeof jangCfg.quantization === 'object'
    ? jangCfg.quantization
    : {}
  const candidates = [
    jangCfg.weight_format,
    jangCfg.format,
    quant.weight_format,
    quant.format,
    quant.method,
    quant.profile,
  ]
  if (candidates.some(value => {
    const s = String(value || '').toLowerCase()
    return s.includes('mxtq') || s.includes('jangtq')
  })) {
    return true
  }
  return 'mxtq_bits' in jangCfg || 'mxtq_bits' in quant
}

function isExplicitAffineJangConfig(jangCfg: any): boolean {
  if (!jangCfg || typeof jangCfg !== 'object') return false
  const quant = jangCfg.quantization && typeof jangCfg.quantization === 'object'
    ? jangCfg.quantization
    : {}
  const values = [
    jangCfg.weight_format,
    jangCfg.format,
    quant.weight_format,
    quant.format,
    quant.method,
    quant.profile,
  ].map(value => String(value || '').toLowerCase())
  if (values.some(value => value.includes('mxfp') || value.includes('mxtq') || value.includes('jangtq'))) {
    return false
  }
  return values.some(value =>
    value === 'jang' ||
    value === 'jang_v2' ||
    value === 'affine' ||
    value === 'jang-importance' ||
    value.startsWith('jang_')
  )
}

function isAffineJangQwenHybridVlm(parsedConfig: any, jangCfg: any): boolean {
  if (!parsedConfig || typeof parsedConfig !== 'object') return false
  if (!jangCfg || typeof jangCfg !== 'object') return false
  const qwenTypes = new Set([
    'qwen3_5',
    'qwen3_5_text',
    'qwen3_5_moe',
    'qwen3_vl',
    'qwen3_vl_moe',
  ])
  const modelTypes = [
    parsedConfig.model_type,
    parsedConfig.text_config?.model_type,
  ].map(value => String(value || '').toLowerCase())
  if (!modelTypes.some(value => qwenTypes.has(value))) return false
  if (!configDeclaresMedia(parsedConfig)) return false
  if (!isExplicitAffineJangConfig(jangCfg)) return false
  return !isMxtqJangConfig(jangCfg)
}

function qwenNativeMtpVlArtifactReady(
  parsedConfig: any,
  jangCfg: any,
  modelPath: string,
): boolean {
  if (!parsedConfig || typeof parsedConfig !== 'object') return false
  if (!jangCfg || typeof jangCfg !== 'object') return false
  if (!configDeclaresMedia(parsedConfig)) return false

  const qwenFamilies = new Set(['qwen3_5', 'qwen3_5_text', 'qwen3_5_moe', 'qwen3_5_moe_text'])
  const modelTypes = [
    parsedConfig.model_type,
    parsedConfig.text_config?.model_type,
    jangCfg.capabilities?.family,
  ].map(value => String(value || '').toLowerCase())
  if (!modelTypes.some(value => qwenFamilies.has(value))) return false

  const configMtpLayers = [
    parsedConfig.num_nextn_predict_layers,
    parsedConfig.mtp_num_hidden_layers,
    parsedConfig.text_config?.num_nextn_predict_layers,
    parsedConfig.text_config?.mtp_num_hidden_layers,
    jangCfg.runtime?.mtp_layers,
    jangCfg.mtp?.num_layers,
  ].some(value => Number.isFinite(Number(value)) && Number(value) > 0)
  if (!configMtpLayers) return false
  if (jangCfg.drop_mtp === true || jangCfg.mtp?.enabled === false || jangCfg.mtp?.kept === false) {
    return false
  }

  try {
    const raw = readFileSync(join(modelPath, 'model.safetensors.index.json'), 'utf-8')
    const index = JSON.parse(raw)
    const weightMap = index?.weight_map
    if (!weightMap || typeof weightMap !== 'object') return false
    const keys = Object.keys(weightMap)
    const hasMtp = keys.some(key => /(^|\.)mtp(\.|$)/.test(key))
    const hasVision = keys.some(key =>
      /(^|\.)(vision_tower|vision_model|visual|patch_embed|multi_modal_projector|mm_projector|image_newline)(\.|$)/.test(key),
    )
    return hasMtp && hasVision
  } catch {
    return false
  }
}

function configuredNativeMtpLayers(parsedConfig: any, jangCfg: any): number {
  const candidates = [
    parsedConfig?.num_nextn_predict_layers,
    parsedConfig?.mtp_num_hidden_layers,
    parsedConfig?.text_config?.num_nextn_predict_layers,
    parsedConfig?.text_config?.mtp_num_hidden_layers,
    jangCfg?.runtime?.mtp_layers,
    jangCfg?.mtp?.num_layers,
  ]
  for (const value of candidates) {
    const n = Number(value)
    if (Number.isFinite(n) && n > 0) return Math.floor(n)
  }
  return 0
}

function coerceNativeMtpDepth(raw: unknown): number | undefined {
  const value = Number(raw)
  if (!Number.isFinite(value)) return undefined
  return Math.max(1, Math.min(3, Math.round(value)))
}

function readNativeMtpTuningDepth(modelPath: string): { depth: number; source: string } | undefined {
  try {
    const tuningPath = join(modelPath, 'vmlx_mtp_tuning.json')
    if (!existsSync(tuningPath)) return undefined
    const tuning = JSON.parse(readFileSync(tuningPath, 'utf-8'))
    const nativeMtp = tuning?.native_mtp
    if (nativeMtp && typeof nativeMtp === 'object') {
      const allowed =
        nativeMtp.blocked !== true &&
        nativeMtp.validated !== false &&
        nativeMtp.output_equivalent !== false
      const depth = allowed ? coerceNativeMtpDepth(nativeMtp.best_depth) : undefined
      if (depth) {
        return { depth, source: 'vmlx_mtp_tuning.json:native_mtp.best_depth' }
      }
    }
    const sweep = tuning?.best_native_mtp_depth
    if (sweep && typeof sweep === 'object') {
      const depth = coerceNativeMtpDepth(sweep.best_depth)
      if (depth) {
        return { depth, source: 'vmlx_mtp_tuning.json:best_native_mtp_depth.best_depth' }
      }
    }
    const depth = coerceNativeMtpDepth(tuning?.best_depth)
    if (depth) return { depth, source: 'vmlx_mtp_tuning.json:best_depth' }
  } catch {
    return undefined
  }
  return undefined
}

function nativeMtpBlockedByTuning(modelPath: string): boolean {
  try {
    const tuningPath = join(modelPath, 'vmlx_mtp_tuning.json')
    if (!existsSync(tuningPath)) return false
    const tuning = JSON.parse(readFileSync(tuningPath, 'utf-8'))
    const nativeMtp = tuning?.native_mtp
    if (!nativeMtp || typeof nativeMtp !== 'object') return false
    return (
      nativeMtp.blocked === true ||
      nativeMtp.validated === false ||
      nativeMtp.output_equivalent === false
    )
  } catch {
    return false
  }
}

function nativeMtpBlockedByProfile(jangCfg: any): boolean {
  const profile = String(
    jangCfg?.quantization?.profile ??
    jangCfg?.profile ??
    '',
  ).trim().toUpperCase()
  if (profile !== 'JANG_2K') return false
  return !['1', 'true', 'yes', 'on'].includes(
    String(process.env.VMLINUX_NATIVE_MTP_ALLOW_JANG2K ?? process.env.VMLX_NATIVE_MTP_ALLOW_JANG2K ?? '').toLowerCase(),
  )
}

function detectNativeMtpCapability(
  parsedConfig: any,
  jangCfg: any,
  modelPath: string,
): DetectedConfig['nativeMtp'] | undefined {
  if (!parsedConfig || typeof parsedConfig !== 'object') return undefined
  if (jangCfg?.drop_mtp === true || jangCfg?.mtp?.enabled === false || jangCfg?.mtp?.kept === false) {
    return undefined
  }
  if (nativeMtpBlockedByTuning(modelPath) || nativeMtpBlockedByProfile(jangCfg)) return undefined

  const qwenFamilies = new Set([
    'qwen3_5',
    'qwen3_5_text',
    'qwen3_5_moe',
    'qwen3_5_moe_text',
  ])
  const modelTypes = [
    parsedConfig.model_type,
    parsedConfig.text_config?.model_type,
    jangCfg?.capabilities?.family,
  ].map(value => String(value || '').toLowerCase())
  if (!modelTypes.some(value => qwenFamilies.has(value))) return undefined

  if (configuredNativeMtpLayers(parsedConfig, jangCfg) <= 0) return undefined

  try {
    const raw = readFileSync(join(modelPath, 'model.safetensors.index.json'), 'utf-8')
    const index = JSON.parse(raw)
    const weightMap = index?.weight_map
    if (!weightMap || typeof weightMap !== 'object') return undefined
    const keys = Object.keys(weightMap)
    const hasMtp = keys.some(key => /(^|\.)mtp(\.|$)/.test(key))
    if (!hasMtp) return undefined
    const hasVisionWeights = keys.some(key =>
      /(^|\.)(vision_tower|vision_model|visual|patch_embed|multi_modal_projector|mm_projector|image_newline)(\.|$)/.test(key),
    )
    const tuningDepth = readNativeMtpTuningDepth(modelPath)
    return {
      supported: true,
      depth: tuningDepth?.depth ?? 3,
      depthSource: tuningDepth?.source ?? 'default',
      runtimeScope: configDeclaresMedia(parsedConfig) && hasVisionWeights ? 'text+vl' : 'text',
      requiresDeterministicSampling: true,
    }
  } catch {
    return undefined
  }
}

function configDeclaresLinearAttention(config: any): boolean {
  if (!config || typeof config !== 'object') return false
  const containers = [config]
  if (config.text_config && typeof config.text_config === 'object') {
    containers.push(config.text_config)
  }
  for (const container of containers) {
    for (const key of ['layer_types', 'layer_type']) {
      const value = container[key]
      if (typeof value === 'string' && value.toLowerCase() === 'linear_attention') {
        return true
      }
      if (Array.isArray(value) && value.some(v => String(v).toLowerCase() === 'linear_attention')) {
        return true
      }
    }
  }
  return false
}

function applyConfigMetadataOverrides(
  detected: DetectedConfig,
  parsedConfig: any,
): DetectedConfig {
  const next = { ...detected }
  const isQwen36 = next.family === 'qwen3.5' || next.family === 'qwen3.5-moe'
  if (isQwen36 && configDeclaresLinearAttention(parsedConfig)) {
    next.cacheType = 'hybrid'
    next.usePagedCache = true
  }
  if (isQwen36 && configDeclaresMedia(parsedConfig) && !next.forceTextOnly) {
    next.isMultimodal = true
  }
  if (next.family === 'nemotron-h' && !configDeclaresMedia(parsedConfig)) {
    next.isMultimodal = false
  }
  return next
}

function configToDetected(family: string, config: Omit<ModelConfig, 'pattern' | 'familyName'>): DetectedConfig {
  return {
    family: family,
    toolParser: config.toolParser,
    reasoningParser: config.reasoningParser,
    supportsThinking: config.supportsThinking,
    thinkInTemplate: config.thinkInTemplate,
    defaultEnableThinking: config.defaultEnableThinking,
    cacheType: config.cacheType,
    usePagedCache: config.usePagedCache ?? true,
    enableAutoToolChoice: config.enableAutoToolChoice ?? false,
    isMultimodal: config.isMultimodal ?? false,
    description: config.description
  }
}

function applyJangCapabilities(
  detected: DetectedConfig,
  jangCfg: any,
): DetectedConfig {
  const caps = jangCfg?.capabilities
  const next = { ...detected }
  const zayaTypedCca = next.family === 'zaya' || next.family === 'zaya1-vl'
  if (jangCfg?.weight_format === 'mxtq' || jangCfg?.format === 'mxtq') {
    next.isTurboQuant = true
  }
  if (!caps || typeof caps !== 'object') return next

  if (typeof caps.tool_parser === 'string') {
    next.toolParser = caps.tool_parser === 'none' ? undefined : caps.tool_parser
    if (next.toolParser && caps.supports_tools !== false) {
      next.enableAutoToolChoice = true
    }
  }
  if (next.family === 'zaya' || next.family === 'zaya1-vl') {
    next.reasoningParser = 'qwen3'
    next.supportsThinking = true
    next.thinkInTemplate = false
    next.defaultEnableThinking = false
  } else if (next.family === 'hy3') {
    next.reasoningParser = 'qwen3'
    next.supportsThinking = true
    next.thinkInTemplate = false
  } else if (next.family === 'minimax') {
    next.reasoningParser = 'minimax_m2'
  } else if (next.family === 'ling') {
    next.reasoningParser = undefined
    next.supportsThinking = false
    next.thinkInTemplate = false
  } else if (caps.supports_thinking === false) {
    next.reasoningParser = undefined
    next.supportsThinking = false
    next.thinkInTemplate = false
  } else if (typeof caps.reasoning_parser === 'string') {
    next.reasoningParser =
      caps.reasoning_parser === 'none' ? undefined : caps.reasoning_parser
  }
  if (
    next.family !== 'zaya' &&
    next.family !== 'zaya1-vl' &&
    next.family !== 'hy3' &&
    next.family !== 'ling'
  ) {
    if (typeof caps.supports_thinking === 'boolean') {
      next.supportsThinking = caps.supports_thinking
    }
    if (typeof caps.think_in_template === 'boolean' && next.supportsThinking !== false) {
      next.thinkInTemplate = caps.think_in_template
    }
  }
  if (typeof caps.cache_type === 'string') {
    const cacheType = caps.cache_type
    if (cacheType === 'kv' || cacheType === 'mamba' || cacheType === 'hybrid' || cacheType === 'rotating_kv') {
      next.cacheType = cacheType
      if (cacheType === 'mamba' || cacheType === 'hybrid') {
        next.usePagedCache = true
      }
    }
  }
  if (zayaTypedCca) {
    next.usePagedCache = true
  }
  return next
}

function resolveJangMultimodal(jangCfg: any, parsedConfig: any): boolean {
  const hasMediaConfig = configDeclaresMedia(parsedConfig)
  const modality =
    jangCfg?.capabilities?.modality ??
    jangCfg?.modality ??
    parsedConfig?._jang_modality

  if (parsedConfig?.model_type === 'zaya1_vl' && hasMediaConfig) {
    return true
  }

  if (isAffineJangQwenHybridVlm(parsedConfig, jangCfg)) {
    return false
  }

  // Explicit converter stamps are authoritative. A JANG bundle may keep a
  // vision_config in config.json even when the emitted artifact is text-only.
  if (typeof jangCfg?.has_vision === 'boolean') {
    return jangCfg.has_vision
  }
  if (typeof jangCfg?.architecture?.has_vision === 'boolean') {
    return jangCfg.architecture.has_vision
  }
  if (typeof modality === 'string') {
    if (modality === 'omni') return hasMediaConfig
    return modality !== 'text' && modality !== 'embedding' && modality !== 'rerank'
  }
  return hasMediaConfig
}

/**
 * Detect model configuration ONLY by reading the model's config.json.
 * This is the authoritative way. We no longer guess based on folder name/regex.
 * Also reads max_position_embeddings for context length detection.
 */
export function detectModelConfigFromDir(modelPath: string): DetectedConfig {
  try {
    // HF repo id fallback: if `modelPath` isn't a local directory, try
    // resolving it to the HuggingFace cache snapshot. Without this, every
    // model loaded via "Download from HuggingFace" ends up with
    // `isMultimodal: false` and the panel strips attached images.
    if (!existsSync(join(modelPath, 'config.json'))) {
      const resolved = resolveHuggingFaceRepoToLocalPath(modelPath)
      if (resolved) {
        modelPath = resolved
      }
    }
    const configPath = join(modelPath, 'config.json')
    if (existsSync(configPath)) {
      const raw = readFileSync(configPath, 'utf-8')
      const parsed = JSON.parse(raw)
      const modelType = parsed.model_type?.toLowerCase()

      // Read max context length from config.json (check multiple field names)
      const maxContextLength: number | undefined =
        (typeof parsed.max_position_embeddings === 'number' ? parsed.max_position_embeddings : undefined) ??
        (typeof parsed.max_sequence_length === 'number' ? parsed.max_sequence_length : undefined) ??
        (typeof parsed.seq_length === 'number' ? parsed.seq_length : undefined) ??
        // Some models nest it in text_config (VL models)
        (typeof parsed.text_config?.max_position_embeddings === 'number' ? parsed.text_config.max_position_embeddings : undefined)

      if (modelType && MODEL_TYPE_TO_FAMILY[modelType]) {
        let familyName = MODEL_TYPE_TO_FAMILY[modelType]

        // Name-based disambiguation for models sharing model_type:
        // GLM-Z1 uses model_type "glm4" but needs deepseek_r1 reasoning (not plain glm4)
        if (modelType === 'glm4' && /glm.?z1/i.test(modelPath)) {
          familyName = 'glm47'
        }
        // MedGemma uses gemma2 model_type but is multimodal
        if (modelType === 'gemma2' && /medgemma/i.test(modelPath)) {
          familyName = 'medgemma'
        }
        // Mistral Small 4 VLM uses a Pixtral-style `mistral3` wrapper around
        // an inner `mistral4` MLA language model. Preserve the wrapper's media
        // route while inheriting Mistral 4 parser defaults for UI/CLI parity.
        if (
          modelType === 'mistral3' &&
          parsed.text_config?.model_type === 'mistral4' &&
          configDeclaresMedia(parsed)
        ) {
          familyName = 'mistral4'
        }

        const config = CONFIG_BY_FAMILY.get(familyName)
          if (config) {
            let detected = configToDetected(familyName, config)
            detected.maxContextLength = maxContextLength
            if (configMarksTurboQuant(parsed)) {
              detected.isTurboQuant = true
            }
            // JANG model detection: read jang_config.json for VLM
            const jangConfigPath = join(modelPath, 'jang_config.json')
            if (existsSync(jangConfigPath)) {
            try {
              const jangCfg = JSON.parse(readFileSync(jangConfigPath, 'utf-8'))
              detected = applyJangCapabilities(detected, jangCfg)
              const nativeMtp = detectNativeMtpCapability(parsed, jangCfg, modelPath)
              if (nativeMtp) {
                detected.nativeMtp = nativeMtp
              }
              const nativeMtpVlReady = qwenNativeMtpVlArtifactReady(parsed, jangCfg, modelPath)
              if (isAffineJangQwenHybridVlm(parsed, jangCfg) && !nativeMtpVlReady) {
                detected.forceTextOnly = true
              }
              detected.isMultimodal = resolveJangMultimodal(jangCfg, parsed)
            } catch {
              if ('vision_config' in parsed) {
                detected.isMultimodal = true
              }
            }
          } else if (configDeclaresMedia(parsed)) {
            detected.isMultimodal = true
          }
          detected = applyConfigMetadataOverrides(detected, parsed)
          return detected
        }
      }

      // Even if model_type isn't recognized, still return context length + VLM detection
      const fallback = { ...DEFAULT_CONFIG }
      if (maxContextLength) fallback.maxContextLength = maxContextLength
      if (configMarksTurboQuant(parsed)) fallback.isTurboQuant = true
      if (configDeclaresMedia(parsed)) {
        fallback.isMultimodal = true
      }
      return fallback
    }
  } catch (_) {
    console.log(`[MODEL-CONFIG] Error reading or parsing config.json at ${modelPath}`)
  }

  // Fallback if no matching config.json or model_type is found
  return DEFAULT_CONFIG
}
