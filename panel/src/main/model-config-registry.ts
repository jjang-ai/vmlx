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
  maxContextLength?: number
}

const CONFIG_BY_FAMILY = new Map<string, Omit<ModelConfig, 'pattern' | 'familyName'>>()

function registerFamily(familyName: string, config: Omit<ModelConfig, 'familyName'>) {
  CONFIG_BY_FAMILY.set(familyName, config)
}

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
registerFamily('devstral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Devstral (Mistral coding)', priority: 5 })
registerFamily('codestral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Codestral (Mistral coding)', priority: 5 })
registerFamily('pixtral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, isMultimodal: true, description: 'Pixtral Vision', priority: 5 })
registerFamily('mixtral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Mixtral MoE', priority: 10 })
registerFamily('mistral', { cacheType: 'kv', toolParser: 'mistral', enableAutoToolChoice: true, description: 'Mistral', priority: 20 })

// DeepSeek
registerFamily('deepseek-vl', { cacheType: 'kv', toolParser: 'deepseek', isMultimodal: true, description: 'DeepSeek-VL vision-language', priority: 5 })
registerFamily('deepseek-r1', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', description: 'DeepSeek R1', priority: 5 })
registerFamily('deepseek-v3', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'DeepSeek V3', priority: 5 })
registerFamily('deepseek-v2', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', description: 'DeepSeek V2', priority: 10 })
registerFamily('deepseek', { cacheType: 'kv', toolParser: 'deepseek', reasoningParser: 'deepseek_r1', description: 'DeepSeek', priority: 50 })

// GLM
registerFamily('gpt-oss', { cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'openai_gptoss', enableAutoToolChoice: true, description: 'GPT-OSS (Harmony reasoning)', priority: 3 })
registerFamily('glm47-flash', { cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'openai_gptoss', enableAutoToolChoice: true, description: 'GLM-4.7 Flash (reasoning)', priority: 3 })
registerFamily('glm47', { cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'GLM-Z1 (deepseek_r1 reasoning)', priority: 5 })
registerFamily('glm4', { cacheType: 'kv', toolParser: 'glm47', enableAutoToolChoice: true, description: 'GLM-4 (tools only)', priority: 20 })

// Gemma
registerFamily('medgemma', { cacheType: 'kv', isMultimodal: true, description: 'Google MedGemma (medical multimodal)', priority: 3 })
registerFamily('paligemma', { cacheType: 'kv', isMultimodal: true, description: 'Google PaliGemma', priority: 5 })
registerFamily('gemma4', { cacheType: 'kv', toolParser: 'gemma4', reasoningParser: 'gemma4', enableAutoToolChoice: true, isMultimodal: true, description: 'Gemma 4 (multimodal)', priority: 5 })
registerFamily('gemma4-text', { cacheType: 'kv', toolParser: 'gemma4', reasoningParser: 'gemma4', enableAutoToolChoice: true, description: 'Gemma 4 (text-only)', priority: 4 })
registerFamily('gemma3', { cacheType: 'kv', toolParser: 'hermes', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, isMultimodal: true, description: 'Gemma 3 (multimodal)', priority: 10 })
registerFamily('gemma3-text', { cacheType: 'kv', toolParser: 'hermes', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'Gemma 3 (text-only)', priority: 8 })
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

// Jamba
registerFamily('jamba', { cacheType: 'hybrid', usePagedCache: true, description: 'Jamba (Hybrid)', priority: 10 })

// Cohere
registerFamily('command-r-plus', { cacheType: 'kv', description: 'Command R+', priority: 10 })
registerFamily('command-r', { cacheType: 'kv', description: 'Command R', priority: 20 })

// Granite
registerFamily('granite', { cacheType: 'kv', toolParser: 'granite', enableAutoToolChoice: true, description: 'Granite', priority: 20 })

// Functionary
registerFamily('functionary', { cacheType: 'kv', toolParser: 'functionary', enableAutoToolChoice: true, description: 'Functionary', priority: 20 })

// MiniMax
registerFamily('minimax', { cacheType: 'kv', toolParser: 'minimax', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'MiniMax', priority: 20 })

// StepFun
registerFamily('step-vl', { cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: true, description: 'StepFun Step-1V Vision-Language', priority: 3 })
registerFamily('step-3.5-flash', { cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'StepFun Step-3.5-Flash (MoE)', priority: 5 })
registerFamily('step', { cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'StepFun Step models', priority: 30 })

// xLAM (Salesforce)
registerFamily('xlam', { cacheType: 'kv', toolParser: 'xlam', enableAutoToolChoice: true, description: 'xLAM', priority: 20 })

// Kimi/Moonshot
registerFamily('kimi-k2', { cacheType: 'kv', toolParser: 'kimi', reasoningParser: 'deepseek_r1', enableAutoToolChoice: true, description: 'Kimi K2 (MoE)', priority: 5 })
registerFamily('kimi', { cacheType: 'kv', toolParser: 'kimi', enableAutoToolChoice: true, description: 'Kimi/Moonshot', priority: 20 })

// InternLM
registerFamily('internlm3', { cacheType: 'kv', description: 'InternLM 3', priority: 10 })
registerFamily('internlm', { cacheType: 'kv', description: 'InternLM', priority: 30 })

// EXAONE
registerFamily('exaone', { cacheType: 'kv', description: 'EXAONE', priority: 20 })

// OLMo
registerFamily('olmo', { cacheType: 'kv', description: 'OLMo', priority: 20 })

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
registerFamily('smolvlm', { cacheType: 'kv', isMultimodal: true, description: 'SmolVLM', priority: 20 })
registerFamily('internlm-xcomposer', { cacheType: 'kv', isMultimodal: true, description: 'InternLM-XComposer', priority: 8 })

// Pure SSM
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
  'step1v': 'step-vl',
  'step3p5': 'step-3.5-flash',
  'step': 'step',
  // ── Gemma family ──
  'gemma': 'gemma',
  'gemma2': 'gemma2',
  'gemma3': 'gemma3',
  'gemma3_text': 'gemma3-text',
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
  'falcon_mamba': 'falcon-mamba',
  'rwkv': 'rwkv',
  'rwkv5': 'rwkv',
  'rwkv6': 'rwkv',
  // ── NVIDIA ──
  'nemotron': 'nemotron',
  'nemotron_h': 'nemotron-h',
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

function configToDetected(family: string, config: Omit<ModelConfig, 'pattern' | 'familyName'>): DetectedConfig {
  return {
    family: family,
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

        const config = CONFIG_BY_FAMILY.get(familyName)
        if (config) {
          const detected = configToDetected(familyName, config)
          detected.maxContextLength = maxContextLength
          // JANG model detection: read jang_config.json for VLM
          const jangConfigPath = join(modelPath, 'jang_config.json')
          if (existsSync(jangConfigPath)) {
            try {
              const jangCfg = JSON.parse(readFileSync(jangConfigPath, 'utf-8'))
              // VLM: check has_vision (not config.json vision_config)
              if ('vision_config' in parsed) {
                detected.isMultimodal = jangCfg?.architecture?.has_vision === true
              }
            } catch { /* ignore parse errors */ }
          } else if ('vision_config' in parsed) {
            detected.isMultimodal = true
          }
          return detected
        }
      }

      // Even if model_type isn't recognized, still return context length + VLM detection
      const fallback = { ...DEFAULT_CONFIG }
      if (maxContextLength) fallback.maxContextLength = maxContextLength
      if ('vision_config' in parsed) {
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
