/**
 * Model configuration registry for auto-detecting tool/reasoning parsers.
 * Mirrors the Python model_configs.py patterns for client-side detection.
 *
 * Detection strategy:
 * 1. Read model's config.json → use model_type/architectures for authoritative detection
 * 2. Fall back to name/path regex matching (for GGUF models without config.json, etc.)
 *
 * This prevents misdetection of fine-tunes (e.g., a Qwen3 model named "Nemotron-Orchestrator"
 * would be incorrectly flagged as hybrid Nemotron by name alone, but config.json reveals Qwen3).
 */

import { readFileSync, existsSync } from 'fs'
import { join } from 'path'

interface ModelConfig {
  familyName: string
  pattern: RegExp
  cacheType: 'kv' | 'mamba' | 'hybrid' | 'rotating_kv'
  toolParser?: string
  reasoningParser?: string
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
  cacheType: string
  usePagedCache: boolean
  enableAutoToolChoice: boolean
  isMultimodal: boolean
  description: string
}

const MODEL_CONFIGS = [
  // Qwen family
  { familyName: 'qwen3-next', pattern: /qwen[\-_.]?3[\-_.]?(?:coder[\-_.]?)?next/i, cacheType: 'mamba', toolParser: 'qwen', reasoningParser: 'qwen3', usePagedCache: true, enableAutoToolChoice: true, description: 'Qwen 3 Next (hybrid Mamba)', priority: 1 },
  { familyName: 'qwen3-vl', pattern: /qwen[\-_.]?3.*VL/i, cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: true, description: 'Qwen 3 Vision-Language', priority: 5 },
  { familyName: 'qwen3-moe', pattern: /qwen[\-_.]?3.*(?:moe|MoE)/i, cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'Qwen 3 MoE', priority: 5 },
  { familyName: 'qwen3', pattern: /qwen[\-_.]?3/i, cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'Qwen 3', priority: 10 },
  { familyName: 'qwen2-vl', pattern: /qwen[\-_.]?2.*VL/i, cacheType: 'kv', toolParser: 'qwen', enableAutoToolChoice: true, isMultimodal: true, description: 'Qwen 2 Vision-Language', priority: 10 },
  { familyName: 'qwen2', pattern: /qwen[\-_.]?2/i, cacheType: 'kv', toolParser: 'qwen', enableAutoToolChoice: true, description: 'Qwen 2', priority: 20 },
  { familyName: 'qwen-mamba', pattern: /qwen.*mamba/i, cacheType: 'mamba', toolParser: 'qwen', usePagedCache: true, description: 'Qwen Mamba', priority: 5 },

  // Llama family
  { familyName: 'llama4', pattern: /llama[\-_.]?4/i, cacheType: 'kv', toolParser: 'llama', enableAutoToolChoice: true, description: 'Llama 4', priority: 5 },
  { familyName: 'llama3', pattern: /llama[\-_.]?3/i, cacheType: 'kv', toolParser: 'llama', enableAutoToolChoice: true, description: 'Llama 3', priority: 10 },
  { familyName: 'llama', pattern: /llama/i, cacheType: 'kv', toolParser: 'llama', description: 'Llama', priority: 50 },

  // Mistral/Mixtral/Devstral/Codestral
  { familyName: 'devstral', pattern: /devstral/i, cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Devstral (Mistral coding)', priority: 5 },
  { familyName: 'codestral', pattern: /codestral/i, cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Codestral (Mistral coding)', priority: 5 },
  { familyName: 'pixtral', pattern: /pixtral/i, cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, isMultimodal: true, description: 'Pixtral Vision', priority: 5 },
  { familyName: 'mixtral', pattern: /mixtral/i, cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Mixtral MoE', priority: 10 },
  { familyName: 'mistral', pattern: /mistral/i, cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Mistral', priority: 20 },

  // DeepSeek
  { familyName: 'deepseek-vl', pattern: /deepseek[\-_.]?vl/i, cacheType: 'kv', toolParser: 'deepseek', isMultimodal: true, description: 'DeepSeek-VL vision-language', priority: 5 },
  { familyName: 'deepseek-r1', pattern: /deepseek[\-_.]?r1/i, cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', description: 'DeepSeek R1', priority: 5 },
  { familyName: 'deepseek-v3', pattern: /deepseek[\-_.]?v3/i, cacheType: 'kv', toolParser: 'deepseek', enableAutoToolChoice: true, description: 'DeepSeek V3', priority: 5 },
  { familyName: 'deepseek-v2', pattern: /deepseek[\-_.]?v2/i, cacheType: 'kv', toolParser: 'deepseek', description: 'DeepSeek V2', priority: 10 },
  { familyName: 'deepseek', pattern: /deepseek/i, cacheType: 'kv', toolParser: 'deepseek', description: 'DeepSeek', priority: 50 },

  // GPT-OSS models (20B, 120B) — Harmony protocol reasoning, same as GLM-4.7 Flash
  { familyName: 'gpt-oss', pattern: /gpt[\-_.]?oss/i, cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'openai_gptoss', enableAutoToolChoice: true, description: 'GPT-OSS (Harmony reasoning)', priority: 3 },
  // GLM-4.7 Flash (MoE, reasoning-capable — must match before glm47)
  // Uses Harmony/GPT-OSS protocol: <|channel|>analysis/final, NOT <think> tags
  { familyName: 'glm47-flash', pattern: /glm[\-_.]?4[\-_.]?7[\-_.]?flash/i, cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'openai_gptoss', enableAutoToolChoice: true, description: 'GLM-4.7 Flash (reasoning)', priority: 3 },
  // GLM-4.7 / GLM-Z1 uses <think> tags (like DeepSeek R1), NOT Harmony protocol
  { familyName: 'glm47', pattern: /glm[\-_.]?(?:4\.7|4[\-_]7|z1)/i, cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'GLM-4.7 / GLM-Z1 (reasoning)', priority: 5 },
  // GLM-4 base (tools only, no reasoning)
  { familyName: 'glm4', pattern: /glm[\-_.]?4/i, cacheType: 'kv', toolParser: 'glm47', enableAutoToolChoice: true, description: 'GLM-4 (tools only)', priority: 20 },

  // Gemma
  { familyName: 'medgemma', pattern: /medgemma/i, cacheType: 'kv', isMultimodal: true, description: 'Google MedGemma (medical multimodal)', priority: 3 },
  { familyName: 'paligemma', pattern: /paligemma/i, cacheType: 'kv', isMultimodal: true, description: 'Google PaliGemma', priority: 5 },
  { familyName: 'gemma3', pattern: /gemma[\-_.]?3/i, cacheType: 'kv', toolParser: 'hermes', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, isMultimodal: true, description: 'Gemma 3 (multimodal)', priority: 10 },
  // gemma3-text: text-only Gemma 3 (detected via config.json model_type="gemma3_text")
  // Shares parsers with gemma3 but is NOT multimodal — supports batching, paged cache, KV quant
  { familyName: 'gemma3-text', pattern: /gemma[\-_.]?3[\-_.]?text/i, cacheType: 'kv', toolParser: 'hermes', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'Gemma 3 (text-only)', priority: 8 },
  { familyName: 'gemma2', pattern: /gemma[\-_.]?2/i, cacheType: 'kv', description: 'Gemma 2', priority: 15 },
  { familyName: 'gemma', pattern: /gemma/i, cacheType: 'kv', description: 'Gemma', priority: 30 },

  // Phi
  { familyName: 'phi4-reasoning', pattern: /phi[\-_.]?4.*(?:reason|think)/i, cacheType: 'kv', toolParser: 'hermes', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'Phi 4 Reasoning', priority: 2 },
  { familyName: 'phi4-multimodal', pattern: /phi[\-_.]?4.*(?:vision|multimodal|vlm)/i, cacheType: 'kv', isMultimodal: true, description: 'Phi 4 Multimodal', priority: 2 },
  { familyName: 'phi4', pattern: /phi[\-_.]?4/i, cacheType: 'kv', toolParser: 'hermes', enableAutoToolChoice: true, description: 'Phi 4', priority: 10 },
  { familyName: 'phi3-vision', pattern: /phi[\-_.]?3[\-_.]?(?:vision|v)/i, cacheType: 'kv', isMultimodal: true, description: 'Phi 3 Vision', priority: 8 },
  { familyName: 'phi3', pattern: /phi[\-_.]?3/i, cacheType: 'kv', description: 'Phi 3', priority: 20 },

  // Hermes
  { familyName: 'hermes', pattern: /hermes/i, cacheType: 'kv', toolParser: 'hermes', enableAutoToolChoice: true, description: 'Hermes', priority: 30 },

  // Nemotron
  { familyName: 'nemotron', pattern: /nemotron/i, cacheType: 'hybrid', toolParser: 'nemotron', usePagedCache: true, description: 'Nemotron (Hybrid)', priority: 10 },

  // Jamba
  { familyName: 'jamba', pattern: /jamba/i, cacheType: 'hybrid', usePagedCache: true, description: 'Jamba (Hybrid)', priority: 10 },

  // Cohere
  { familyName: 'command-r-plus', pattern: /command[\-_.]?r[\-_.]?\+|command[\-_.]?r[\-_.]?plus/i, cacheType: 'kv', description: 'Command R+', priority: 10 },
  { familyName: 'command-r', pattern: /command[\-_.]?r/i, cacheType: 'kv', description: 'Command R', priority: 20 },

  // Granite
  { familyName: 'granite', pattern: /granite/i, cacheType: 'kv', toolParser: 'granite', enableAutoToolChoice: true, description: 'Granite', priority: 20 },

  // Functionary
  { familyName: 'functionary', pattern: /functionary/i, cacheType: 'kv', toolParser: 'functionary', enableAutoToolChoice: true, description: 'Functionary', priority: 20 },

  // MiniMax
  { familyName: 'minimax', pattern: /minimax/i, cacheType: 'kv', toolParser: 'minimax', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'MiniMax', priority: 20 },

  // StepFun
  { familyName: 'step-3.5-flash', pattern: /step[\-_.]?3[\-_.]?5[\-_.]?flash/i, cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'StepFun Step-3.5-Flash (MoE)', priority: 5 },
  { familyName: 'step', pattern: /(?:^|\/)step[\-_.]/i, cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'StepFun Step models', priority: 30 },

  // xLAM (Salesforce)
  { familyName: 'xlam', pattern: /xlam/i, cacheType: 'kv', toolParser: 'xlam', enableAutoToolChoice: true, description: 'Salesforce xLAM', priority: 20 },

  // Kimi/Moonshot
  { familyName: 'kimi-k2', pattern: /kimi[\-_.]?k2/i, cacheType: 'kv', toolParser: 'kimi', enableAutoToolChoice: true, description: 'Kimi K2 (MoE)', priority: 5 },
  { familyName: 'kimi', pattern: /kimi|moonshot/i, cacheType: 'kv', toolParser: 'kimi', enableAutoToolChoice: true, description: 'Kimi/Moonshot', priority: 20 },

  // InternLM
  { familyName: 'internlm3', pattern: /internlm[\-_.]?3/i, cacheType: 'kv', description: 'InternLM 3', priority: 10 },
  { familyName: 'internlm', pattern: /internlm/i, cacheType: 'kv', description: 'InternLM', priority: 30 },

  // EXAONE (LG AI Research)
  { familyName: 'exaone', pattern: /exaone/i, cacheType: 'kv', description: 'EXAONE', priority: 20 },

  // OLMo (AI2)
  { familyName: 'olmo', pattern: /olmo/i, cacheType: 'kv', description: 'OLMo', priority: 20 },

  // StarCoder (no tool support — do NOT map to qwen2)
  { familyName: 'starcoder', pattern: /starcoder/i, cacheType: 'kv', description: 'StarCoder code models', priority: 30 },

  // StableLM (no tool support — do NOT map to qwen2)
  { familyName: 'stablelm', pattern: /stable[\-_.]?lm/i, cacheType: 'kv', description: 'StableLM', priority: 30 },

  // Baichuan (no tool support — do NOT map to qwen2)
  { familyName: 'baichuan', pattern: /baichuan/i, cacheType: 'kv', description: 'Baichuan', priority: 30 },

  // Vision-Language / Multimodal (MLLM) models
  { familyName: 'yi-vl', pattern: /yi[\-_.].*(?:vl|vision)/i, cacheType: 'kv', isMultimodal: true, description: 'Yi Vision-Language', priority: 15 },
  { familyName: 'llava', pattern: /llava/i, cacheType: 'kv', isMultimodal: true, description: 'LLaVA vision-language', priority: 20 },
  { familyName: 'idefics', pattern: /idefics/i, cacheType: 'kv', isMultimodal: true, description: 'Idefics vision-language', priority: 5 },
  { familyName: 'molmo', pattern: /molmo/i, cacheType: 'kv', isMultimodal: true, description: 'Molmo multimodal', priority: 20 },
  { familyName: 'cogvlm', pattern: /cogvlm/i, cacheType: 'kv', isMultimodal: true, description: 'CogVLM vision-language', priority: 20 },
  { familyName: 'internvl', pattern: /internvl/i, cacheType: 'kv', isMultimodal: true, description: 'InternVL vision-language', priority: 15 },
  { familyName: 'minicpm-v', pattern: /minicpm[\-_.]?v/i, cacheType: 'kv', isMultimodal: true, description: 'MiniCPM-V vision', priority: 20 },
  { familyName: 'florence', pattern: /florence/i, cacheType: 'kv', isMultimodal: true, description: 'Florence vision', priority: 20 },
  { familyName: 'smolvlm', pattern: /smol[\-_.]?vlm/i, cacheType: 'kv', isMultimodal: true, description: 'SmolVLM', priority: 20 },
  { familyName: 'internlm-xcomposer', pattern: /internlm[\-_.]?xcomposer/i, cacheType: 'kv', isMultimodal: true, description: 'InternLM-XComposer', priority: 8 },

  // Pure SSM / Mamba-architecture models
  { familyName: 'falcon-mamba', pattern: /falcon[\-_.]?mamba/i, cacheType: 'mamba', usePagedCache: true, description: 'Falcon Mamba (SSM)', priority: 5 },
  { familyName: 'mamba', pattern: /(?:^|\/)mamba[\-_.]/i, cacheType: 'mamba', usePagedCache: true, description: 'Mamba SSM', priority: 30 },
  { familyName: 'rwkv', pattern: /rwkv/i, cacheType: 'mamba', usePagedCache: true, description: 'RWKV', priority: 30 },
] satisfies ModelConfig[] as ModelConfig[]
MODEL_CONFIGS.sort((a, b) => a.priority - b.priority)

// Build lookup by familyName for O(1) access from architecture detection
const CONFIG_BY_FAMILY = new Map<string, ModelConfig>()
for (const c of MODEL_CONFIGS) {
  if (!CONFIG_BY_FAMILY.has(c.familyName)) CONFIG_BY_FAMILY.set(c.familyName, c)
}

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
  'codestral': 'codestral',
  'devstral': 'devstral',
  'codestral_mamba': 'mamba',
  // ── DeepSeek family ──
  'deepseek_v3': 'deepseek-v3',
  'deepseek_v2': 'deepseek-v2',
  'deepseek_vl': 'deepseek-vl',
  'deepseek_vl2': 'deepseek-vl',
  'deepseek_vl_v2': 'deepseek-vl',
  'deepseek2': 'deepseek',
  'deepseek': 'deepseek',
  // ── GLM family ──
  'chatglm': 'glm4',
  'glm4': 'glm4',
  'glm4_moe': 'glm47-flash',
  'glm4_moe_lite': 'glm47-flash',
  'glm': 'glm4',
  // ── GPT-OSS (Harmony protocol) — needs openai_gptoss reasoning, not deepseek_r1
  'gpt_oss': 'gpt-oss',
  // ── StepFun ──
  'step3p5': 'step-3.5-flash',
  'step': 'step',
  // ── Gemma family ──
  'gemma': 'gemma',
  'gemma2': 'gemma2',
  'gemma3': 'gemma3',
  'gemma3_text': 'gemma3-text',
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
  'falcon_mamba': 'falcon-mamba',
  'rwkv': 'rwkv',
  'rwkv5': 'rwkv',
  'rwkv6': 'rwkv',
  // ── NVIDIA ──
  'nemotron': 'nemotron',
  'nemotron_h': 'nemotron',
  // ── IBM ──
  'granite': 'granite',
  'granite_moe': 'granite',
  // ── Cohere ──
  'cohere': 'command-r',
  'cohere2': 'command-r',
  // ── Hermes (NousResearch) ──
  'hermes': 'hermes',
  // ── Kimi/Moonshot ──
  'kimi_k2': 'kimi-k2',
  // ── EXAONE ──
  'exaone': 'exaone',
  'exaone3': 'exaone',
  // ── OLMo ──
  'olmo': 'olmo',
  'olmo2': 'olmo',
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
  'molmo': 'molmo',
  'minicpmv': 'minicpm-v',
  'smolvlm': 'smolvlm',
  'internvl_chat': 'internvl',
  // ── Others (architecture-compatible mappings) ──
  'starcoder2': 'starcoder',
  'stablelm': 'stablelm',
  'baichuan': 'baichuan',
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

function configToDetected(config: ModelConfig): DetectedConfig {
  return {
    family: config.familyName,
    toolParser: config.toolParser,
    reasoningParser: config.reasoningParser,
    cacheType: config.cacheType,
    usePagedCache: config.usePagedCache ?? true,
    enableAutoToolChoice: config.enableAutoToolChoice ?? false,
    isMultimodal: config.isMultimodal ?? false,
    description: config.description
  }
}

/**
 * Detect model configuration from model path or name.
 * Fast path (sync, no I/O) — uses regex pattern matching only.
 * For authoritative detection that reads config.json, use detectModelConfigFromDir().
 */
function detectModelConfig(modelPath: string): DetectedConfig {
  const modelName = modelPath.split('/').pop() || modelPath

  for (const config of MODEL_CONFIGS) {
    if (config.pattern.test(modelName) || config.pattern.test(modelPath)) {
      return configToDetected(config)
    }
  }

  return DEFAULT_CONFIG
}

/**
 * Detect model configuration by reading the model's config.json first (authoritative),
 * then falling back to name-based regex matching.
 *
 * This prevents misdetection of fine-tunes where the model name doesn't match
 * the actual architecture (e.g., "Nemotron-Orchestrator-8B" is actually Qwen3).
 *
 * Synchronous I/O — suitable for buildArgs() which runs once at session start.
 */
export function detectModelConfigFromDir(modelPath: string): DetectedConfig {
  // 1. Try reading config.json for authoritative model_type
  try {
    const configPath = join(modelPath, 'config.json')
    if (existsSync(configPath)) {
      const raw = readFileSync(configPath, 'utf-8')
      const parsed = JSON.parse(raw)
      const modelType = parsed.model_type?.toLowerCase()

      if (modelType && MODEL_TYPE_TO_FAMILY[modelType]) {
        const familyName = MODEL_TYPE_TO_FAMILY[modelType]
        const config = CONFIG_BY_FAMILY.get(familyName)
        if (config) {
          const nameDetected = detectModelConfig(modelPath)
          const result = configToDetected(config)

          // If name-based detection found a more specific family, check if it should win
          if (nameDetected.family !== 'unknown' && nameDetected.family !== result.family) {
            const nameConfig = CONFIG_BY_FAMILY.get(nameDetected.family)
            if (nameConfig) {
              // Allow name-based refinement in two cases:
              // 1. Name-detected family maps to the same model_type (exact match)
              const nameModelType = Object.entries(MODEL_TYPE_TO_FAMILY).find(([, f]) => f === nameDetected.family)?.[0]
              if (nameModelType === modelType) {
                console.log(`[MODEL-CONFIG] Architecture: ${modelType}, using specific name match: ${nameDetected.family} (from path)`)
                return nameDetected
              }
              // 2. Name-detected family is more specific (lower priority) within the same
              //    family tree — e.g., "glm47-flash" (priority 3) refines "glm4" (priority 20)
              //    This handles GLM-4.7 Flash having model_type "chatglm" but needing openai_gptoss
              // Strip version suffixes to get family root: 'glm4' → 'glm', 'deepseek-r1' → 'deepseek'
              // (?:[-_][a-z])? handles separator+letter prefixes like -r1, -v2, -k2
              const archBase = familyName.replace(/(?:[-_][a-z])?\d.*$/, '').replace(/[-_]+$/, '')
              const nameBase = nameDetected.family.replace(/(?:[-_][a-z])?\d.*$/, '').replace(/[-_]+$/, '')
              if (archBase === nameBase && nameConfig.priority < config.priority) {
                console.log(`[MODEL-CONFIG] Name-based refinement: ${familyName} → ${nameDetected.family} (more specific, same family tree, model_type=${modelType})`)
                return nameDetected
              }
            }
            console.log(`[MODEL-CONFIG] Architecture override: path suggests "${nameDetected.family}" but config.json model_type="${modelType}" → using "${familyName}"`)
            if (nameDetected.family === 'nemotron' && familyName !== 'nemotron') {
              console.log(`[MODEL-CONFIG] Note: Model name contains "Nemotron" but architecture is ${familyName}. This is likely a fine-tune. Using ${familyName} parsers (tool=${config?.toolParser}, reasoning=${config?.reasoningParser})`)
            }
          }
          return result
        }
      }
    }
  } catch (_) { /* config.json not readable, fall through to name matching */ }

  // 2. Fall back to name-based detection
  return detectModelConfig(modelPath)
}
