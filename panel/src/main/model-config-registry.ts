/**
 * Model configuration registry for auto-detecting tool/reasoning parsers.
 * Mirrors the Python model_configs.py patterns for client-side detection.
 *
 * Detection: reads model's config.json model_type field and maps to a registered family.
 * No name-based regex detection — config.json is authoritative.
 * Users can always override auto-detected values via Server Settings UI.
 */

import { closeSync, existsSync, fstatSync, openSync, readFileSync, readdirSync, readSync, statSync } from 'fs'
import { basename, join } from 'path'
import { homedir } from 'os'
import { formatJangQuantizationLabel } from '../shared/jangQuantization'
import {
  normalizeReasoningEffort,
  normalizeReasoningEffortLevels,
  type ReasoningEffort,
} from '../shared/reasoningEffortPolicy'
import { familySupportsThinkingBudget } from '../shared/thinkingBudgetFamilies'

const MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024

/** Read tensor names without materializing a safetensors payload. */
function readSafetensorsHeaderKeys(filePath: string): string[] | undefined {
  let fd: number | undefined
  try {
    fd = openSync(filePath, 'r')
    const prefix = Buffer.alloc(8)
    if (readSync(fd, prefix, 0, prefix.length, 0) !== prefix.length) return undefined
    const rawHeaderBytes = prefix.readBigUInt64LE(0)
    if (
      rawHeaderBytes <= 0n ||
      rawHeaderBytes > BigInt(MAX_SAFETENSORS_HEADER_BYTES) ||
      rawHeaderBytes > BigInt(Number.MAX_SAFE_INTEGER)
    ) return undefined
    const headerBytes = Number(rawHeaderBytes)
    if (8 + headerBytes > fstatSync(fd).size) return undefined
    const buffer = Buffer.allocUnsafe(headerBytes)
    let offset = 0
    while (offset < headerBytes) {
      const count = readSync(fd, buffer, offset, headerBytes - offset, 8 + offset)
      if (count <= 0) return undefined
      offset += count
    }
    const header = JSON.parse(buffer.toString('utf8').trim())
    if (!header || typeof header !== 'object' || Array.isArray(header)) return undefined
    return Object.keys(header).filter(key => key !== '__metadata__')
  } catch {
    return undefined
  } finally {
    if (fd !== undefined) closeSync(fd)
  }
}

/**
 * Return artifact tensor names, including supplemental shards omitted from
 * the Hugging Face weight index. Qwen3.8-27B ships its MTP head this way.
 */
function bundleArtifactWeightKeys(modelPath: string): string[] | undefined {
  const keys = new Set<string>()
  const indexedFiles = new Set<string>()
  const indexPath = join(modelPath, 'model.safetensors.index.json')
  if (existsSync(indexPath)) {
    try {
      const index = JSON.parse(readFileSync(indexPath, 'utf-8'))
      const weightMap = index?.weight_map
      if (!weightMap || typeof weightMap !== 'object' || Array.isArray(weightMap)) {
        return undefined
      }
      for (const [key, rawFile] of Object.entries(weightMap)) {
        keys.add(String(key))
        if (typeof rawFile === 'string') indexedFiles.add(basename(rawFile))
      }
    } catch {
      return undefined
    }
  }

  try {
    for (const name of readdirSync(modelPath).filter(name => name.endsWith('.safetensors')).sort()) {
      if (indexedFiles.has(name)) continue
      const supplementalKeys = readSafetensorsHeaderKeys(join(modelPath, name))
      if (supplementalKeys === undefined) {
        // Fail closed for an unreadable file that explicitly advertises MTP.
        // Unrelated custom sidecars do not erase a sound base-index result.
        if (name.toLowerCase().includes('mtp')) return undefined
        continue
      }
      for (const key of supplementalKeys) keys.add(key)
    }
  } catch {
    return undefined
  }
  return [...keys]
}

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
  cacheSubtype?: string
  toolParser?: string
  reasoningParser?: string
  supportsThinking?: boolean
  supportsInstructMode?: boolean
  /** Does this family's chat template actually READ `enable_thinking`?
   *
   * `supportsThinking` conflates two different questions: does the model reason
   * at all, and does its template honour the enable_thinking kwarg. Muse reasons
   * (supportsThinking: true) but its template reads ONLY `reasoning_strength` —
   * so the Auto/On toggle rendered off supportsThinking sent a value nothing
   * consumed. MEASURED live: Auto and On produced byte-identical output
   * (148-char reasoning, same 64-token answer). Default true; set false for a
   * family whose template ignores the kwarg, and the Thinking toggle is replaced
   * by an honest notice instead of a dead control.
   */
  honorsEnableThinking?: boolean
  supportedReasoningEfforts?: ReasoningEffort[]
  defaultReasoningEffort?: ReasoningEffort
  supportsThinkingBudget?: boolean
  thinkInTemplate?: boolean
  defaultEnableThinking?: boolean
  usePagedCache?: boolean
  enableAutoToolChoice?: boolean
  isMultimodal?: boolean
  architectureHints?: Record<string, string | number | boolean>
  description: string
  priority: number
}

export interface DetectedConfig {
  family: string
  toolParser?: string
  reasoningParser?: string
  supportsThinking?: boolean
  supportsInstructMode?: boolean
  /** Does this family's chat template actually READ `enable_thinking`?
   *
   * `supportsThinking` conflates two different questions: does the model reason
   * at all, and does its template honour the enable_thinking kwarg. Muse reasons
   * (supportsThinking: true) but its template reads ONLY `reasoning_strength` —
   * so the Auto/On toggle rendered off supportsThinking sent a value nothing
   * consumed. MEASURED live: Auto and On produced byte-identical output
   * (148-char reasoning, same 64-token answer). Default true; set false for a
   * family whose template ignores the kwarg, and the Thinking toggle is replaced
   * by an honest notice instead of a dead control.
   */
  honorsEnableThinking?: boolean
  supportedReasoningEfforts?: ReasoningEffort[]
  defaultReasoningEffort?: ReasoningEffort
  supportsThinkingBudget?: boolean
  thinkInTemplate?: boolean
  defaultEnableThinking?: boolean
  /** Bundle-owned DSV4 native CSA/HCA pool codec startup default. */
  dsv4PoolQuantDefault?: boolean
  cacheType: string
  cacheSubtype?: string
  architectureHints?: Record<string, string | number | boolean>
  usePagedCache: boolean
  enableAutoToolChoice: boolean
  isMultimodal: boolean
  forceTextOnly?: boolean
  /** Bundle-declared `capabilities.modalities` (lower-cased) when the sidecar carries one. */
  runtimeModalities?: string[]
  // MiniMax-M3 VL routing: M3 vision is handled in-engine by SingleBatchGenerator
  // gated behind env VMLX_M3_VL=1 (NOT via mlx_vlm --is-mllm). When set, the panel
  // emits NEITHER --is-mllm NOR --text-only and threads VMLX_M3_VL=1 into the engine
  // child env so images are preprocessed on the text-routed path.
  m3VlRoute?: boolean
  isTurboQuant?: boolean
  quantizationLabel?: string
  nativeMtp?: {
    supported: boolean
    depth: number
    depthSource?: string
    runtimeScope: 'text' | 'text+vl'
    nativeCacheType: string
    requiresDeterministicSampling: boolean
    defaultMode?: 'auto' | 'off'
    blockedReason?: string
  }
  description: string
  maxContextLength?: number
}

const CONFIG_BY_FAMILY = new Map<string, Omit<ModelConfig, 'pattern' | 'familyName'>>()

function registerFamily(familyName: string, config: Omit<ModelConfig, 'familyName'>) {
  // supportsThinkingBudget is NOT a per-row property any more. It gates the only
  // control that lets a user cap the reasoning pass, and that cap is what arms
  // the engine's never-empty answer pass — so a row that forgets it is a row
  // whose empty-answer fix the user cannot reach. It was forgotten for
  // nemotron, nemotron-h and nanbeige (measured 2026-08-12: the Chat Settings
  // drawer for a live Nemotron session offered Enable Thinking and no budget
  // field, while the engine had capped that family since 2026-08-11).
  // One list owns it now, kept in step with the engine's own set by
  // tests/test_cli_panel_settings_parity.py.
  CONFIG_BY_FAMILY.set(familyName, {
    ...config,
    supportsThinkingBudget: familySupportsThinkingBudget(familyName),
  })
}

// ZAYA / Zyphra: CCA attention + top-1 MoE. Text ZAYA is reasoning-capable and
// current engine registry policy defaults Auto reasoning ON while preserving
// zaya_xml tools and the typed CCA cache contract.
registerFamily('zaya', { cacheType: 'hybrid', toolParser: 'zaya_xml', reasoningParser: 'qwen3', supportsThinking: true, thinkInTemplate: false, defaultEnableThinking: true, usePagedCache: false, enableAutoToolChoice: true, description: 'ZAYA CCA hybrid MoE', priority: 3 })
// ZAYA1-VL is detected separately so the UI does not fall through to generic
// VLM defaults. Current plain-template ZAYA1-VL bundles are vision/tool/cache
// capable, but live proof shows the synthetic qwen3 thinking rail produces
// hidden-only output, so the panel must not expose a reasoning mode until the
// artifact ships a real VLM thinking contract.
registerFamily('zaya1-vl', { cacheType: 'hybrid', toolParser: 'zaya_xml', supportsThinking: false, thinkInTemplate: false, defaultEnableThinking: false, usePagedCache: false, enableAutoToolChoice: true, isMultimodal: true, description: 'ZAYA1-VL CCA hybrid vision-language', priority: 3 })

// Qwen
// Qwen 3.5 dense and MoE share model_types with VL variants — VL detection
// relies on config.json vision_config, not the family's isMultimodal flag.
registerFamily('qwen3.5', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: false, description: 'Qwen 3.5 (dense)', priority: 4 })
registerFamily('qwen3.5-moe', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: false, description: 'Qwen 3.5 MoE', priority: 4 })
// Qwen3.8 Flash Next is a Qwen4Exp VL wrapper around the hybrid QSA/Gated
// DeltaNet decoder.  Base MLX bundles do not require a jang_config sidecar, so
// every user-visible capability that is invariant for this model_type must be
// present in the registry just as it is in Python model_configs.py.  Falling
// through to DEFAULT_CONFIG hid low/medium/xhigh, the Qwen tool parser, and the
// native-MTP control even while the loaded engine advertised all three.
registerFamily('glm5-next', {
  cacheType: 'hybrid',
  cacheSubtype: 'glm5_next_native_v2',
  // Bundle-stamped ids are registered engine aliases: glm_xml_args -> the
  // glm47 arg_key/arg_value XML parser, glm_think_block -> the R1 <think>
  // rail parser. Template force-opens <think> in the assistant prefix.
  toolParser: 'glm_xml_args',
  reasoningParser: 'glm_think_block',
  supportsThinking: true,
  thinkInTemplate: true,
  defaultEnableThinking: true,
  supportedReasoningEfforts: ['low', 'high', 'max'],
  defaultReasoningEffort: 'max',
  usePagedCache: false,
  enableAutoToolChoice: true,
  // Keep the family default text-only: artifact inspection below promotes an
  // individual bundle only when its config actually declares media. The
  // native GLM VLM runtime handles those image/video artifacts.
  isMultimodal: false,
  architectureHints: {
    attentionArch: 'kda_mla_dsa',
    cacheSchema: 'glm5_next_native_v2',
    cachePrecision: 'full',
  },
  description: 'GLM-5.3-Flash (KDA + sparse-MLA/DSA + mHC MoE)',
  priority: 3,
})

registerFamily('qwen4-exp', {
  cacheType: 'hybrid',
  cacheSubtype: 'qsa_gdn_ple_v1',
  toolParser: 'qwen',
  reasoningParser: 'qwen3',
  supportsThinking: true,
  supportsInstructMode: true,
  supportedReasoningEfforts: ['low', 'medium', 'xhigh'],
  defaultReasoningEffort: 'xhigh',
  thinkInTemplate: true,
  defaultEnableThinking: true,
  usePagedCache: false,
  enableAutoToolChoice: true,
  isMultimodal: true,
  architectureHints: {
    attentionArch: 'qsa_gdn_ple',
    cacheSchema: 'qsa_gdn_ple_v1',
    cachePrecision: 'full',
    pleStorage: 'ssd_row_addressed',
  },
  description: 'Qwen3.8 Flash Next (QSA + Gated DeltaNet VL)',
  priority: 3,
})
registerFamily('qwen3-next', { cacheType: 'mamba', toolParser: 'qwen', reasoningParser: 'qwen3', usePagedCache: false, enableAutoToolChoice: true, description: 'Qwen 3 Next (hybrid Mamba)', priority: 1 })
registerFamily('qwen3-vl', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: true, description: 'Qwen 3 Vision-Language', priority: 5 })
registerFamily('qwen3-moe', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'Qwen 3 MoE', priority: 5 })
registerFamily('qwen3', { cacheType: 'kv', toolParser: 'qwen', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'Qwen 3 / QwQ', priority: 10 })
registerFamily('qwen2-vl', { cacheType: 'kv', toolParser: 'qwen', enableAutoToolChoice: true, isMultimodal: true, description: 'Qwen 2 Vision-Language', priority: 10 })
registerFamily('qwen2', { cacheType: 'kv', toolParser: 'qwen', enableAutoToolChoice: true, description: 'Qwen 2', priority: 20 })
registerFamily('qwen-mamba', { cacheType: 'mamba', toolParser: 'qwen', usePagedCache: false, description: 'Qwen Mamba', priority: 5 })
// MiMo-V2.5 JANG_2L keeps multimodal assets. Its template emits generic XML
// function calls and <think> reasoning, not Qwen tool JSON.
registerFamily('mimo_v2', { cacheType: 'kv', toolParser: 'xml_function', reasoningParser: 'think_xml', supportsThinking: true, thinkInTemplate: false, enableAutoToolChoice: true, isMultimodal: true, description: 'MiMo V2.5 multimodal MoE', priority: 4 })
// Nanbeige 4.2 reuses 22 module layers for two forward loops. The Python
// loader owns the fail-closed 44-slot cache invariant; the panel mirrors the
// text/protocol/default truth and blocks external draft decoding below.
registerFamily('nanbeige', { cacheType: 'kv', toolParser: 'xml_function', reasoningParser: 'qwen3', supportsThinking: true, thinkInTemplate: true, defaultEnableThinking: true, architectureHints: { cacheSchema: 'looped_kv_v1', numLoops: 2, cacheSlots: 44 }, enableAutoToolChoice: true, isMultimodal: false, description: 'Nanbeige 4.2 looped transformer (22 layers x 2 loops)', priority: 3 })

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
registerFamily('deepseek-v4', {
  cacheType: 'kv',
  usePagedCache: false,
  toolParser: 'dsml',
  reasoningParser: 'deepseek_r1',
  supportsThinking: true,
  enableAutoToolChoice: true,
  description: 'DeepSeek V4 Flash',
  priority: 4,
})
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
// Gemma 4's native template defaults enable_thinking=false and emits a closed
// empty thought channel for Auto/Off. Keep the UI's Auto label and the raw API
// on that same native default; explicit On remains available and creates its
// own prompt/cache chain.
registerFamily('gemma4', { cacheType: 'kv', toolParser: 'gemma4', reasoningParser: 'gemma4', supportsThinking: true, thinkInTemplate: false, defaultEnableThinking: false, enableAutoToolChoice: true, isMultimodal: true, usePagedCache: false, description: 'Gemma 4 (multimodal)', priority: 5 })
registerFamily('gemma4-text', { cacheType: 'kv', toolParser: 'gemma4', reasoningParser: 'gemma4', supportsThinking: true, thinkInTemplate: false, defaultEnableThinking: false, enableAutoToolChoice: true, usePagedCache: false, description: 'Gemma 4 (text-only)', priority: 4 })
registerFamily('spark2_5', { cacheType: 'kv', toolParser: 'spark25', reasoningParser: 'qwen3', supportsThinking: true, thinkInTemplate: true, defaultEnableThinking: true, enableAutoToolChoice: true, isMultimodal: false, usePagedCache: false, description: 'Spark-X2.5 (mixed full/sliding attention)', priority: 4 })
// Muse Glimmer: Gemma-shaped text backbone (sliding/full 3:1) + windowed ViT,
// vision AND video. Reasoning is routed by recipient (to=self / to=user), not an
// inline think pair, and its only live control is the reasoning_strength template
// kwarg — enable_thinking and reasoning_effort are both ignored by its template.
// Same mixed-SWA cache shape as gemma4, so it takes paged like gemma4 does.
// Vision is live: the vendored MuseGlimmerImageProcessor/VideoProcessor supply
// the pixel_values and grid_thw the bundle's processor_config.json asks for, and
// <|patch|> expands to one id per merged cell. Verified live on JANG_4M — a red
// circle and a blue square are each described correctly and distinctly, and
// projected features land at per-token RMS 0.961 against the text stream's ~1.0
// (they were 21x that, and unreadable, until the adapter's trailing activation
// was restored).
// Muse is deliberately absent from THINKING_BUDGET_FAMILIES (see
// src/shared/thinkingBudgetFamilies.ts): its template reads neither a token
// budget nor enable_thinking/reasoning_effort. Listing it rendered a Max
// Thinking Tokens control and sent max_thinking_tokens + thinking_budget to a
// template that ignores both — a control that looks like it works and does
// nothing. Depth is set only by the `reasoning_strength` template kwarg.
// Muse's ONLY live reasoning control is the `reasoning_strength` template
// kwarg (low/medium/high/xhigh, default high). Its template reads neither a
// token budget nor enable_thinking, so:
//   supportsInstructMode: false — there is no truthful thinking-off rail;
//     without this the Off toggle sends enable_thinking:false and nothing
//     happens (same reason Step-3.7 sets it).
// The effort levels below are surfaced through the existing reasoning control
// and translated to `reasoning_strength` in the request builder, so the knob
// the user sees is the knob the model actually has.
registerFamily('muse-glimmer', { cacheType: 'kv', toolParser: 'atem', reasoningParser: 'muse_glimmer', supportsThinking: true, supportsInstructMode: false, honorsEnableThinking: false, supportedReasoningEfforts: ['low', 'medium', 'high', 'xhigh'], enableAutoToolChoice: true, isMultimodal: true, usePagedCache: false, description: 'Muse Glimmer (vision + video)', priority: 5 })
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
registerFamily('nemotron-h', { cacheType: 'hybrid', cacheSubtype: 'nemotron_h_ssm_attention', architectureHints: { attentionArch: 'hybrid_ssm_attention' }, toolParser: 'nemotron', reasoningParser: 'deepseek_r1', usePagedCache: false, description: 'Nemotron Hybrid', priority: 10 })

// Poolside / Laguna
registerFamily('laguna', { cacheType: 'kv', toolParser: 'glm47', reasoningParser: 'qwen3', supportsThinking: true, thinkInTemplate: false, defaultEnableThinking: false, enableAutoToolChoice: true, description: 'Laguna / Poolside coding model', priority: 10 })

// Jamba
registerFamily('jamba', { cacheType: 'hybrid', usePagedCache: false, description: 'Jamba (Hybrid)', priority: 10 })

// Cohere
registerFamily('command-r-plus', { cacheType: 'kv', description: 'Command R+', priority: 10 })
registerFamily('command-r', { cacheType: 'kv', description: 'Command R', priority: 20 })

// Granite
registerFamily('granite', { cacheType: 'kv', toolParser: 'granite', enableAutoToolChoice: true, description: 'Granite', priority: 20 })
registerFamily('granitemoehybrid', { cacheType: 'hybrid', toolParser: 'granite', enableAutoToolChoice: true, usePagedCache: false, description: 'Granite MoE Hybrid', priority: 10 })

// Functionary
registerFamily('functionary', { cacheType: 'kv', toolParser: 'functionary', enableAutoToolChoice: true, description: 'Functionary', priority: 20 })

// MiniMax uses its own parser name so panel-emitted CLI args match the engine
// registry and diagnostics instead of relying on a generic qwen3 alias.
registerFamily('minimax', { cacheType: 'kv', toolParser: 'minimax', reasoningParser: 'minimax_m2', enableAutoToolChoice: true, description: 'MiniMax', priority: 20 })
// MiniMax-M3 sparse MSA (MiniMaxM3SparseCache: GQA K/V + append-only idx_keys).
// The engine's typed paged serializer preserves K/V/idx_keys and absolute offsets,
// so M3 uses the paged L1 + block-disk L2 path by default. Generic KV q4/q8 stays
// disabled because only the native M3 codec understands the full sparse state.
registerFamily('minimax_m3', { cacheType: 'kv', toolParser: 'minimax_m3', reasoningParser: 'minimax_m3', enableAutoToolChoice: true, isMultimodal: true, usePagedCache: false, description: 'MiniMax-M3 (sparse MSA + Lightning-Indexer, VL)', priority: 5 })

// openPangu-2.0-Flash: 92B MoE (6B active) MLA + DSA/SWA hybrid + 3 stateful
// causal convs + mHC hyper-connections. Mirrors the engine registry entry
// (vmlx_engine/model_configs.py): cache_type kv + openpangu_v2_composite
// subtype ON PURPOSE (converter's coarse "hybrid" stamp would misroute into
// SSM hybrid handling), tool parser "openpangu" (converter stamps "qwen"
// which never matches the <|tool_call_start|> JSON-list format — see
// applyJangCapabilities neutralization below), deepseek_r1 reasoning with
// thinking in-template. Paged OFF: conv state is path-dependent, the typed
// prefix/paged lane is Phase-2. Without this entry family detection fell
// through to generic — openpangu startup defaults (timeout 900, JIT off) in
// sessions.ts never fired and the chat thinking toggle stayed disabled
// (live UI matrix finding, 2026-07-02).
registerFamily('openpangu_v2', { cacheType: 'kv', cacheSubtype: 'openpangu_v2_composite', toolParser: 'openpangu', reasoningParser: 'deepseek_r1', supportsThinking: true, honorsEnableThinking: false, thinkInTemplate: true, usePagedCache: false, enableAutoToolChoice: true, isMultimodal: false, description: 'openPangu-2.0-Flash (MLA + DSA/SWA + mHC MoE)', priority: 20 })
// dots3-note: 280B/16B-active omni MoE — hybrid dual-geometry MLA (13 full/DSA
// + 33 sliding), DSA indexer past 2048, MTP layer 46, vision + audio towers
// (video rides the IMAGE token path). Thinking ON by default and the template
// honors enable_thinking (off = <no_think> user marker + prefilled closed
// think block). Tool dialect: dots XML — parser 'dots'.
registerFamily('dots3_note', { cacheType: 'kv', cacheSubtype: 'dots3_hybrid_full_swa', toolParser: 'dots', reasoningParser: 'dots3', supportsThinking: true, honorsEnableThinking: true, thinkInTemplate: true, usePagedCache: false, enableAutoToolChoice: true, isMultimodal: true, description: 'dots3-note omni MoE (hybrid MLA full+SWA, DSA, MTP)', priority: 20 })

// Ling / Bailing hybrid: MLA softmax layers plus linear-attention/SSM-style
// companion state. Eric directive 2026-05-11: treat Ling chat output as plain
// content. Keep DeepSeek tool parsing, but do not advertise a reasoning parser
// or thinking capability even when stale JANG sidecars claim deepseek_r1.
registerFamily('ling', { cacheType: 'hybrid', toolParser: 'deepseek', supportsThinking: false, thinkInTemplate: false, usePagedCache: false, enableAutoToolChoice: true, description: 'Ling / Bailing hybrid', priority: 20 })

// Tencent Hy3-preview: text-only dense GQA KV + MoE. The chat template uses
// reasoning_effort=no_think|low|high, so Python normalizes the UI thinking
// toggle into Hy3's effort field before render.
registerFamily('hy3', { cacheType: 'kv', toolParser: 'hunyuan', reasoningParser: 'qwen3', enableAutoToolChoice: true, description: 'Tencent Hy3-preview', priority: 4 })

// StepFun
registerFamily('step-vl', { cacheType: 'kv', toolParser: 'step3p5', reasoningParser: 'qwen3', enableAutoToolChoice: true, isMultimodal: true, description: 'StepFun Step-1V Vision-Language', priority: 3 })
registerFamily('step-3.7-flash', { cacheType: 'kv', cacheSubtype: 'step3p7_full_sliding_kv', architectureHints: { textModelType: 'step3p5', attentionArch: 'full_and_sliding_kv', slidingWindow: 512 }, toolParser: 'step3p5', reasoningParser: 'qwen3', supportsThinking: true, honorsEnableThinking: false, supportsInstructMode: false, supportedReasoningEfforts: ['low', 'medium', 'high'], thinkInTemplate: true, enableAutoToolChoice: true, isMultimodal: true, usePagedCache: false, description: 'StepFun Step-3.7-Flash JANG/VL', priority: 4 })
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
registerFamily('lfm2', { cacheType: 'hybrid', cacheSubtype: 'lfm2_moe_hybrid_ssm', architectureHints: { attentionArch: 'hybrid_ssm_attention', cacheSchema: 'hybrid_ssm_v1', ssmCompanionCache: true, attentionKvStorageQuantization: true }, toolParser: 'lfm2', reasoningParser: 'qwen3', supportsThinking: true, honorsEnableThinking: false, supportsInstructMode: false, usePagedCache: false, enableAutoToolChoice: true, description: 'Liquid LFM2 / LFM2-MoE hybrid', priority: 10 })

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
registerFamily('falcon-h1', { cacheType: 'hybrid', usePagedCache: false, description: 'Falcon H1 hybrid SSM/attention', priority: 5 })
registerFamily('falcon-mamba', { cacheType: 'mamba', usePagedCache: false, description: 'Falcon Mamba (SSM)', priority: 5 })
registerFamily('mamba', { cacheType: 'mamba', usePagedCache: false, description: 'Mamba SSM', priority: 30 })
registerFamily('rwkv', { cacheType: 'mamba', usePagedCache: false, description: 'RWKV', priority: 30 })

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
  'qwen4_exp': 'qwen4-exp',
  'glm5_next': 'glm5-next',
  'glm5_next_text': 'glm5-next',
  'qwen4_exp_text': 'qwen4-exp',
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
  'mimo_v2': 'mimo_v2',
  'nanbeige': 'nanbeige',
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
  // ── dots studio ──
  'dots3_note': 'dots3_note',
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
  'step3p7': 'step-3.7-flash',
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
  'spark2_5': 'spark2_5',
  'gemma4_unified': 'gemma4',
  'gemma4_unified_text': 'gemma4-text',
  'muse_glimmer': 'muse-glimmer',
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
  'minimax_m3': 'minimax_m3',
  'openpangu_v2': 'openpangu_v2',
  'minimax_m3_vl': 'minimax_m3',
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
  // Phase-1 cache policy (2026-06-13): paged RAM block cache OFF by default for ALL families;
  // SSD prefix cache (disk_cache L2 + memory-aware L1) is the default. Path-dependent families
  // (hybrid/mamba/zaya/nemotron-h/ling/lfm2/step-3.7/qwen3-next/...) opt INTO paged explicitly
  // via registerFamily usePagedCache:true (Phase-2 ports their typed lanes to non-paged SSD).
  usePagedCache: false,
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

function readJangDefaultEnableThinking(jangCfg: any): boolean | undefined {
  if (!jangCfg || typeof jangCfg !== 'object') return undefined
  const chat = jangCfg.chat
  if (chat && typeof chat === 'object') {
    const templateDefaults = chat.template_kwargs_defaults
    if (templateDefaults && typeof templateDefaults === 'object') {
      const value = templateDefaults.enable_thinking
      if (typeof value === 'boolean') return value
    }
    const reasoning = chat.reasoning
    if (reasoning && typeof reasoning === 'object') {
      const value = reasoning.default_enabled
      if (typeof value === 'boolean') return value
      const defaultMode = String(reasoning.default_mode || '').trim().toLowerCase()
      if (['thinking', 'reasoning', 'think', 'on', 'true'].includes(defaultMode)) return true
      if (['chat', 'direct', 'instruct', 'off', 'false'].includes(defaultMode)) return false
    }
  }
  // 2026-08 Nemotron stamps declare "on"/"off" strings in a TOP-LEVEL
  // reasoning block and under capabilities (mirrors the engine reader in
  // model_config_registry._jang_stamp_default_enable_thinking). Already
  // shipped, so honor the spelling; an "off" sibling must not be ignored.
  const topReasoning = jangCfg.reasoning
  if (topReasoning && typeof topReasoning === 'object') {
    const value = String(topReasoning.default || '').trim().toLowerCase()
    if (value === 'on') return true
    if (value === 'off') return false
  }
  const capabilities = jangCfg.capabilities
  if (capabilities && typeof capabilities === 'object') {
    const value = String(capabilities.default_reasoning || '').trim().toLowerCase()
    if (value === 'on') return true
    if (value === 'off') return false
  }
  return undefined
}

function readJangChatMetadata(
  detected: DetectedConfig,
  jangCfg: any,
): DetectedConfig {
  if (!jangCfg || typeof jangCfg !== 'object') return detected
  const chat = jangCfg.chat && typeof jangCfg.chat === 'object' ? jangCfg.chat : {}
  // DSV4/Muse nest these under `chat`; the Qwen stamps put them at the top
  // level, where `chat` carries only sampling defaults. Reading just one place
  // meant a whole vintage of bundle silently contributed nothing here.
  const reasoning = chat.reasoning ?? jangCfg.reasoning
  const toolCalling = chat.tool_calling ?? jangCfg.tool_calling
  if (!reasoning && !toolCalling && !jangCfg.chat) return detected

  const next = { ...detected }

  if (reasoning && typeof reasoning === 'object') {
    if (reasoning.supported === false) {
      next.reasoningParser = undefined
      next.supportsThinking = false
      next.thinkInTemplate = false
      next.defaultEnableThinking = false
      delete next.supportedReasoningEfforts
      delete next.defaultReasoningEffort
    } else if (reasoning.supported === true) {
      next.supportsThinking = true
      if (typeof reasoning.parser === 'string') {
        next.reasoningParser = reasoning.parser === 'none' ? undefined : reasoning.parser
      }
      if (typeof reasoning.think_in_template === 'boolean') {
        next.thinkInTemplate = reasoning.think_in_template
      } else if (
        next.family === 'laguna' &&
        (
          reasoning.default_mode === 'think' ||
          (Array.isArray(reasoning.modes) && reasoning.modes.includes('think'))
        )
      ) {
        // Laguna/Poolside templates own the <think> rail. Real S-2.1 JANG_2L
        // sidecars carry this fact in the top-level chat block, not in the
        // older capabilities stamp. Preserve Auto reasoning in the UI by
        // deriving the same template ownership before the capabilities guard.
        next.thinkInTemplate = true
      }

      const modes = Array.isArray(reasoning.modes)
        ? reasoning.modes.map((mode: unknown) => String(mode || '').trim().toLowerCase())
        : undefined
      if (modes) {
        next.supportsInstructMode = modes.some((mode: string) =>
          ['chat', 'direct', 'instruct', 'off'].includes(mode),
        )
      }

      // Effort levels: prefer an explicit `reasoning_effort_levels`, but fall
      // back to the model's own `modes` list when those modes ARE effort
      // levels. Muse Glimmer declares `control: reasoning_strength` with
      // `modes: [low, medium, high, xhigh]` and no `reasoning_effort_levels`,
      // so without this the Chat Settings panel showed only the Auto/On/Off
      // thinking toggle and never surfaced the strength buttons the model
      // actually has. A `modes` list that carries no recognizable effort level
      // (e.g. only chat/think) yields undefined and leaves the registry value.
      // `supported_reasoning_efforts` mirrors the engine identifier and is what
      // Qwen3.8-era stamps ship; `reasoning_effort_levels` is the older DSV4 and
      // Muse spelling. Both name the same fact, so read both here rather than
      // letting one vintage of bundle silently lose its effort list.
      let effortLevels = normalizeReasoningEffortLevels(
        reasoning.supported_reasoning_efforts ?? reasoning.reasoning_effort_levels,
      )
      if (effortLevels === undefined && modes) {
        const modeEffortLevels = normalizeReasoningEffortLevels(modes)
        if (modeEffortLevels && modeEffortLevels.length > 0) {
          effortLevels = modeEffortLevels
        }
      }
      if (effortLevels !== undefined) {
        next.supportedReasoningEfforts = effortLevels
      }
      // Default effort: an explicit default (either spelling), else the
      // `default_mode` when it names one of the effort levels (Muse ships
      // default_mode: high).
      const defaultEffort =
        normalizeReasoningEffort(reasoning.default_reasoning_effort) ??
        normalizeReasoningEffort(reasoning.default_effort) ??
        normalizeReasoningEffort(reasoning.default_mode)
      if (
        defaultEffort &&
        (effortLevels === undefined || effortLevels.includes(defaultEffort))
      ) {
        next.defaultReasoningEffort = defaultEffort
      } else if (reasoning.default_effort != null) {
        delete next.defaultReasoningEffort
      }
    }
  }

  const stampedDefaultEnableThinking = readJangDefaultEnableThinking(jangCfg)
  if (typeof stampedDefaultEnableThinking === 'boolean' && next.supportsThinking !== false) {
    next.defaultEnableThinking = stampedDefaultEnableThinking
  }

  if (toolCalling && typeof toolCalling === 'object') {
    if (toolCalling.supported === false) {
      next.toolParser = undefined
      next.enableAutoToolChoice = false
    } else if (toolCalling.supported === true) {
      if (typeof toolCalling.parser === 'string') {
        next.toolParser = toolCalling.parser === 'none' ? undefined : toolCalling.parser
      }
      if (next.toolParser) {
        next.enableAutoToolChoice = true
      }
    }
  }

  return next
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

function affineJangRuntimeHasVerifiedVision(jangCfg: any): boolean {
  if (!jangCfg || typeof jangCfg !== 'object') return false
  const runtime = jangCfg.runtime
  const capabilities = jangCfg.capabilities
  if (!runtime || typeof runtime !== 'object') return false
  if (!capabilities || typeof capabilities !== 'object') return false
  return runtime.status === 'runtime_verified' &&
    runtime.vision_verified === true &&
    capabilities.supports_vision === true
}

function modelIndexHasVisionWeights(modelPath: string): boolean {
  const keys = bundleArtifactWeightKeys(modelPath)
  return keys?.some(key =>
    /(^|\.)(vision_tower|vision_model|visual|patch_embed|multi_modal_projector|mm_projector|image_newline)(\.|$)/.test(key),
  ) === true
}

/**
 * Nemotron Omni keeps its media contract outside the text decoder's
 * config.json.  Treat it as media-capable only when the sidecar declaration
 * and the matching encoder/projector tensors are both present.  This mirrors
 * the engine's artifact-first Omni dispatch without trusting a folder name or
 * a JANG modality stamp by itself.
 */
function nemotronOmniArtifactHasMedia(
  parsedConfig: any,
  jangCfg: any,
  modelPath: string,
): boolean {
  const modelTypes = [
    parsedConfig?.model_type,
    parsedConfig?.text_config?.model_type,
    jangCfg?.capabilities?.family,
  ].map(value => String(value || '').toLowerCase())
  if (!modelTypes.some(value => value === 'nemotron_h' || value === 'nemotron-h')) {
    return false
  }

  try {
    const omni = JSON.parse(readFileSync(join(modelPath, 'config_omni.json'), 'utf-8'))
    const index = JSON.parse(readFileSync(join(modelPath, 'model.safetensors.index.json'), 'utf-8'))
    const weightMap = index?.weight_map
    if (!omni || typeof omni !== 'object' || !weightMap || typeof weightMap !== 'object') {
      return false
    }
    const keys = Object.keys(weightMap)
    const hasPrefix = (prefix: string): boolean => keys.some(key => key.startsWith(prefix))
    const audioReady = omni.sound_config != null &&
      hasPrefix('sound_encoder.') &&
      hasPrefix('sound_projection.')
    const visionReady = omni.vision_config != null &&
      hasPrefix('vision_model.') &&
      hasPrefix('mlp1.')
    return audioReady || visionReady
  } catch {
    return false
  }
}

function affineJangArtifactHasVision(jangCfg: any, modelPath: string): boolean {
  if (affineJangRuntimeHasVerifiedVision(jangCfg)) return true
  if (
    jangCfg?.has_vision === false ||
    jangCfg?.architecture?.has_vision === false ||
    jangCfg?.capabilities?.has_vision === false
  ) {
    return false
  }
  // Current affine JANG conversion preserves the real Qwen vision tower but
  // does not stamp the historical runtime_verified fields. The engine owns
  // runtime availability and already auto-routes through qwen3_5_family; the
  // panel only needs to establish that this artifact actually carries vision
  // tensors. Metadata-only/text extracts still stay forceTextOnly.
  return modelIndexHasVisionWeights(modelPath)
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

  const keys = bundleArtifactWeightKeys(modelPath)
  if (!keys) return false
  const hasMtp = keys.some(key => /(^|\.)mtp(\.|$)/.test(key))
  const hasVision = keys.some(key =>
    /(^|\.)(vision_tower|vision_model|visual|patch_embed|multi_modal_projector|mm_projector|image_newline)(\.|$)/.test(key),
  )
  return hasMtp && hasVision
}

function configuredNativeMtpLayers(parsedConfig: any, jangCfg: any): { layers: number; source: string } {
  const candidates = [
    [jangCfg?.runtime?.mtp_layers, 'jang_config.runtime.mtp_layers'],
    [jangCfg?.mtp?.num_layers, 'jang_config.mtp.num_layers'],
    [jangCfg?.mtp_layers, 'jang_config.mtp_layers'],
    [parsedConfig?.num_nextn_predict_layers, 'config.num_nextn_predict_layers'],
    [parsedConfig?.mtp_num_hidden_layers, 'config.mtp_num_hidden_layers'],
    [parsedConfig?.text_config?.num_nextn_predict_layers, 'config.text_config.num_nextn_predict_layers'],
    [parsedConfig?.text_config?.mtp_num_hidden_layers, 'config.text_config.mtp_num_hidden_layers'],
  ] as const
  for (const [value, source] of candidates) {
    const n = Number(value)
    if (Number.isFinite(n) && n > 0) return { layers: Math.floor(n), source }
  }
  return { layers: 0, source: 'missing' }
}

function coerceNativeMtpDepth(raw: unknown): number | undefined {
  const value = Number(raw)
  if (!Number.isFinite(value)) return undefined
  return Math.max(1, Math.min(3, Math.round(value)))
}

/**
 * Draft depth the BUNDLE itself declares, before any measured tuning.
 *
 * 2026-08-17. Qwen3.8-27B-JANG_4D-CRACK states `recommended_num_drafts: 1` and
 * `upstream_num_speculative_tokens: 2` in its jang_config, yet nothing read
 * either field — depth fell through to a hardcoded 3. Asking a head trained
 * for 1-2 tokens to draft 3 ahead tanks acceptance (measured 58.6%, under the
 * ~0.68 breakeven), so MTP burns draft+verify work and then falls back to AR.
 *
 * ⚠️ ONLY `recommended_num_drafts` may be read here. NEVER
 * `upstream_num_speculative_tokens` — that is a terminology collision the
 * engine already documents (vmlx_engine/native_mtp.py): vLLM's field counts
 * DRAFTS while the kit's D-notation counts drafts+verified, so upstream's "2"
 * means 2 drafts = the kit's forbidden D3, the exact config the Nemotron sweep
 * measured at 0.48x. The stamp keeps that value for provenance only, flagged
 * do-not-copy. This mirrors the engine's resolution order so panel and engine
 * cannot disagree about depth.
 */
function declaredNativeMtpDrafts(jangCfg: any): { depth: number; source: string } | undefined {
  const depth = coerceNativeMtpDepth(jangCfg?.mtp?.recommended_num_drafts)
  if (depth === undefined) return undefined
  return { depth, source: 'jang_config.mtp.recommended_num_drafts' }
}

function positiveFiniteNumber(raw: unknown): number | undefined {
  if (typeof raw !== 'number' || !Number.isFinite(raw) || raw <= 0) return undefined
  return raw
}

function validatedFlatNativeMtpDepth(tuning: any): number | undefined {
  if (!tuning || typeof tuning !== 'object') return undefined
  if (tuning.blocked === true || tuning.output_equivalent === false) return undefined
  const depth = coerceNativeMtpDepth(tuning.best_depth)
  if (depth === undefined) return undefined
  // Flat D1 is the conservative legacy seed. D2/D3 must be selected by
  // matched-output wall throughput; expected tokens/cycle alone is not a
  // launch recommendation and a hand-edited best_depth cannot override its
  // own speed table.
  if (depth === 1) return 1
  if (tuning.validated !== true) return undefined
  if (
    positiveFiniteNumber(tuning.baseline_tok_s) === undefined ||
    positiveFiniteNumber(tuning.best_tok_s) === undefined ||
    positiveFiniteNumber(tuning.speedup_vs_baseline) === undefined ||
    tuning.speedup_vs_baseline <= 1
  ) return undefined

  const rawSpeeds = tuning.measured_tok_s_by_depth
  if (rawSpeeds !== undefined) {
    if (!rawSpeeds || typeof rawSpeeds !== 'object' || Array.isArray(rawSpeeds)) return undefined
    const speeds = Object.entries(rawSpeeds)
      .map(([rawDepth, rawSpeed]) => ({
        depth: Number(rawDepth),
        speed: positiveFiniteNumber(rawSpeed),
      }))
      .filter((row): row is { depth: number; speed: number } =>
        Number.isInteger(row.depth) && row.depth >= 1 && row.depth <= 3 && row.speed !== undefined,
      )
    const selected = speeds.find(row => row.depth === depth)
    if (!selected || speeds.length === 0) return undefined
    const fastestSpeed = Math.max(...speeds.map(row => row.speed))
    const fastestDepth = Math.min(...speeds.filter(row => row.speed === fastestSpeed).map(row => row.depth))
    if (fastestDepth !== depth) return undefined
  }
  return depth
}

// Nested native_mtp / best_native_mtp_depth blocks carry an attestation, not
// a speed table. Same evidence rule as the engine (vmlx_engine/native_mtp.py
// _attested_block_depth): depth 1 is the conservative seed; deeper needs
// validated === true. A block that merely omitted `validated` used to pass.
function attestedBlockNativeMtpDepth(block: any): number | undefined {
  if (!block || typeof block !== 'object') return undefined
  if (block.blocked === true || block.output_equivalent === false) return undefined
  const depth = coerceNativeMtpDepth(block.best_depth)
  if (depth === undefined) return undefined
  if (depth === 1) return 1
  return block.validated === true ? depth : undefined
}

function readNativeMtpTuningDepth(modelPath: string): { depth: number; source: string } | undefined {
  try {
    const tuningPath = join(modelPath, 'vmlx_mtp_tuning.json')
    if (!existsSync(tuningPath)) return undefined
    const tuning = JSON.parse(readFileSync(tuningPath, 'utf-8'))
    const nativeMtp = tuning?.native_mtp
    if (nativeMtp && typeof nativeMtp === 'object') {
      if (nativeMtp.blocked === true || nativeMtp.validated === false || nativeMtp.output_equivalent === false) {
        return undefined
      }
      const depth = attestedBlockNativeMtpDepth(nativeMtp)
      if (depth) {
        return { depth, source: 'vmlx_mtp_tuning.json:native_mtp.best_depth' }
      }
    }
    const sweep = tuning?.best_native_mtp_depth
    if (sweep && typeof sweep === 'object') {
      const depth = attestedBlockNativeMtpDepth(sweep)
      if (depth) {
        return { depth, source: 'vmlx_mtp_tuning.json:best_native_mtp_depth.best_depth' }
      }
    }
    const depth = validatedFlatNativeMtpDepth(tuning)
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
    if (tuning?.blocked === true) return true
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

function nativeMtpOutputEquivalent(modelPath: string): boolean | undefined {
  try {
    const tuningPath = join(modelPath, 'vmlx_mtp_tuning.json')
    if (!existsSync(tuningPath)) return undefined
    const tuning = JSON.parse(readFileSync(tuningPath, 'utf-8'))
    const nativeMtp = tuning?.native_mtp
    if (!nativeMtp || typeof nativeMtp !== 'object') return undefined
    return typeof nativeMtp.output_equivalent === 'boolean'
      ? nativeMtp.output_equivalent
      : undefined
  } catch {
    return undefined
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
  const stampedMtpMode = String(jangCfg?.mtp?.mtp_mode ?? '').trim().toLowerCase()
  if (
    jangCfg?.drop_mtp === true ||
    jangCfg?.mtp?.enabled === false ||
    jangCfg?.mtp?.kept === false ||
    ['none', 'absent', 'disabled', 'off'].includes(stampedMtpMode)
  ) {
    return undefined
  }
  // Adding a family here is what makes the Native MTP control EXIST for it.
  // dots3-note was missing, so the app ran MTP for that bundle and showed the
  // user nothing: the engine reported index_has_mtp_tensors true, 3 MTP
  // tensors, runtime_active true and effective_depth 1 ("native MTP runtime is
  // active for text+vl") while the drawer rendered no label, no mode selector
  // and no mention of MTP at all. Every other value the branch below computes
  // already matches what the engine reports for this bundle — native_cache
  // schema hybrid_ssm_v1, runtime scope text+vl, and depth 1 from its
  // vmlx_mtp_tuning.json best_depth — so the allowlist was the only thing
  // holding the control back.
  const supportedFamilies = new Set([
    'qwen3_5',
    'qwen3_5_text',
    'qwen3_5_moe',
    'qwen3_5_moe_text',
    'qwen4_exp',
    'qwen4_exp_text',
    'hy_v3',
    'dots3_note',
    'glm5_next',
    'glm5_next_text',
  ])
  const modelTypes = [
    parsedConfig.model_type,
    parsedConfig.text_config?.model_type,
    jangCfg?.capabilities?.family,
  ].map(value => String(value || '').toLowerCase())
  if (!modelTypes.some(value => supportedFamilies.has(value))) return undefined
  const hy3 = modelTypes.includes('hy_v3')
  const glm5Next = modelTypes.includes('glm5_next') || modelTypes.includes('glm5_next_text')
  const tuningDepth = readNativeMtpTuningDepth(modelPath)
  const hy3OutputEquivalent = hy3
    ? nativeMtpOutputEquivalent(modelPath)
    : undefined
  const hy3IdentityBlocked = hy3 && hy3OutputEquivalent !== true
  if (
    (!hy3 && nativeMtpBlockedByTuning(modelPath)) ||
    (!hy3 && nativeMtpBlockedByProfile(jangCfg) && !tuningDepth)
  ) return undefined

  const declaredDrafts = declaredNativeMtpDrafts(jangCfg)
  const configuredMtp = configuredNativeMtpLayers(parsedConfig, jangCfg)
  if (configuredMtp.layers <= 0) return undefined

  try {
    const keys = bundleArtifactWeightKeys(modelPath)
    if (!keys) return undefined
    const baseLayerCount = Number(
      parsedConfig?.text_config?.num_hidden_layers
      ?? parsedConfig?.num_hidden_layers,
    )
    const hasAppendedGlmMtp = glm5Next
      && Number.isInteger(baseLayerCount)
      && baseLayerCount > 0
      && keys.some(key => key.startsWith(`model.layers.${baseLayerCount}.`))
    const hasMtp = keys.some(key => /(^|\.)mtp(\.|$)/.test(key)) || hasAppendedGlmMtp
    if (!hasMtp) return undefined
    const hasVisionWeights = keys.some(key =>
      /(^|\.)(vision_tower|vision_model|visual|patch_embed|multi_modal_projector|mm_projector|image_newline)(\.|$)/.test(key),
    )
    if (hy3IdentityBlocked) {
      return {
        supported: false,
        depth: 1,
        depthSource: 'validation-blocked',
        runtimeScope: 'text',
        nativeCacheType: 'plain_kv_v1',
        requiresDeterministicSampling: true,
        blockedReason:
          'Native MTP weights were detected, but this HY3 affine bundle has not proven token-identical greedy output for its two-token verifier. Autoregressive decode remains active.',
      }
    }
    return {
      supported: true,
      // Measured tuning wins; then what the bundle declares for itself; then
      // the layer count. The final fallback is 1, NOT 3: an uncharacterised
      // head must draft conservatively, because an over-deep draft is pure
      // loss (verify + replay) once acceptance falls under breakeven.
      depth:
        tuningDepth?.depth
        ?? declaredDrafts?.depth
        ?? coerceNativeMtpDepth(configuredMtp.layers)
        ?? 1,
      depthSource:
        tuningDepth?.source
        ?? declaredDrafts?.source
        ?? configuredMtp.source,
      runtimeScope:
        configDeclaresMedia(parsedConfig) && hasVisionWeights
          ? 'text+vl'
          : 'text',
      // Match the runtime schema string reported by /v1/capabilities
      // (cache.native.schema): hy3 is plain attention (plain_kv_v1); the
      // qwen3.6 hybrid SSM+attention bundle reports hybrid_ssm_v1 (NOT
      // hybrid_ssm_attention_kv_v1, which matched nothing the engine emits).
      nativeCacheType: hy3
        ? 'plain_kv_v1'
        : glm5Next
          ? 'glm5_next_native_v2'
          : 'hybrid_ssm_v1',
      requiresDeterministicSampling: false,
      // Keep the GLM MTP capability visible while making its unmodified Auto
      // path AR. Live measurements show the current GLM verifier loses to AR;
      // a fixed D1-D3 selection remains an explicit opt-in.
      defaultMode: glm5Next ? 'off' : 'auto',
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

function configDeclaresMixedSwaAttention(config: any): boolean {
  if (!config || typeof config !== 'object') return false
  const containers = [config]
  if (config.text_config && typeof config.text_config === 'object') {
    containers.push(config.text_config)
  }
  for (const container of containers) {
    const value = container.layer_types ?? container.layer_type
    const values = Array.isArray(value) ? value : [value]
    const normalized = values.map(v => String(v || '').toLowerCase())
    if (
      normalized.some(v => v === 'sliding_attention') &&
      normalized.some(v => v === 'full_attention')
    ) {
      return true
    }
  }
  return false
}

/**
 * Match the loader's minimum pre-launch proof for Laguna selective TQ.
 *
 * The Python loader applies the mixed-SWA wrapper only when `layer_types`
 * maps one-to-one onto native cache slots. The panel cannot instantiate those
 * slots before launch, but it can reject incomplete metadata by requiring the
 * bundle's declared hidden-layer count to match the complete per-layer list.
 */
function configDeclaresCompleteLagunaMixedSwaAttention(config: any): boolean {
  if (!config || typeof config !== 'object') return false
  const containers = [config]
  if (config.text_config && typeof config.text_config === 'object') {
    containers.push(config.text_config)
  }
  for (const container of containers) {
    const layerTypes = container.layer_types
    const declaredLayers = Number(
      container.num_hidden_layers ?? config.num_hidden_layers,
    )
    if (
      !Array.isArray(layerTypes) ||
      !Number.isInteger(declaredLayers) ||
      declaredLayers <= 0 ||
      layerTypes.length !== declaredLayers
    ) {
      continue
    }
    const normalized = layerTypes.map(v => String(v || '').toLowerCase())
    if (
      normalized.some(v => v.includes('sliding')) &&
      normalized.some(v => !v.includes('sliding'))
    ) {
      return true
    }
  }
  return false
}

function applyLagunaVariantHint(
  detected: DetectedConfig,
  parsedConfig: any,
  jangCfg: any,
): DetectedConfig {
  if (detected.family !== 'laguna') return detected

  const sourceModelName = typeof jangCfg?.source_model?.name === 'string'
    ? jangCfg.source_model.name.trim()
    : ''
  const configuredModelName = typeof parsedConfig?._name_or_path === 'string'
    ? parsedConfig._name_or_path.trim()
    : ''
  const authoritativeName = sourceModelName || configuredModelName
  if (!authoritativeName) return detected

  const modelLeaf = authoritativeName.split('/').pop() ?? authoritativeName
  if (!/^laguna[-_ ]xs[-_ ]2[._-]1(?:[-_ ]|$)/i.test(modelLeaf)) {
    return detected
  }

  return {
    ...detected,
    architectureHints: {
      ...(detected.architectureHints ?? {}),
      lagunaVariant: 'xs-2.1',
    },
  }
}

function applyConfigMetadataOverrides(
  detected: DetectedConfig,
  parsedConfig: any,
): DetectedConfig {
  const next = { ...detected }
  const isQwen36 = next.family === 'qwen3.5' || next.family === 'qwen3.5-moe'
  if (isQwen36 && configDeclaresLinearAttention(parsedConfig)) {
    next.cacheType = 'hybrid'
    next.usePagedCache = false
  }
  if (isQwen36 && configDeclaresMedia(parsedConfig) && !next.forceTextOnly) {
    next.isMultimodal = true
  }
  // The current Mistral-Medium-3.5 loader instantiates the outer
  // Mistral3/inner Ministral3 text decoder with a tokenizer only. It does not
  // construct a Pixtral processor or expose a live VL prefill route. Treating
  // the preserved vision tower as runnable made Electron launch --is-mllm,
  // disable paged/L2 defaults, and advertise attachments even though the
  // loaded BatchedEngine reported mllm=False and vl_runtime_available=false.
  // Mistral Small 4 uses inner model_type=mistral4 and is intentionally not
  // covered by this guard.
  if (
    next.family === 'mistral3' &&
    String(parsedConfig?.model_type ?? '').toLowerCase() === 'mistral3' &&
    String(parsedConfig?.text_config?.model_type ?? '').toLowerCase() === 'ministral3'
  ) {
    next.isMultimodal = false
    next.forceTextOnly = true
    next.usePagedCache = false
    next.architectureHints = {
      ...(next.architectureHints ?? {}),
      runtimeScope: 'text_only_until_pixtral_processor_is_wired',
      vlRuntimeAvailable: false,
    }
  }
  if (
    next.family === 'nemotron-h' &&
    typeof parsedConfig?.hybrid_override_pattern === 'string' &&
    parsedConfig.hybrid_override_pattern.length > 0
  ) {
    next.architectureHints = {
      ...(next.architectureHints ?? {}),
      hybridOverridePattern: parsedConfig.hybrid_override_pattern,
    }
  }
  if (
    (next.family === 'gemma4' || next.family === 'gemma4-text') &&
    configDeclaresMixedSwaAttention(parsedConfig)
  ) {
    next.cacheType = 'rotating_kv'
    // Gemma 4's typed mixed-SWA cache supports paged prefix reuse and block-disk
    // restore. Keep the effective UI default aligned with that runtime path.
    next.usePagedCache = false
  }
  if (
    next.family === 'laguna' &&
    configDeclaresCompleteLagunaMixedSwaAttention(parsedConfig)
  ) {
    // Laguna is still a KV family: every layer is attention-backed. Record the
    // per-layer full/sliding topology separately so the panel can preserve the
    // native rotating/full cache contract without routing Laguna through
    // SSM-hybrid controls or inferring a generic TQ wrapper.
    next.architectureHints = {
      ...(next.architectureHints ?? {}),
      attentionArch: 'full_and_sliding_kv',
      cacheSchema: 'mixed_swa_kv_v1',
      selectiveTurboQuantKv: false,
      nativeCacheDefault: true,
    }
  }
  // 2026-07-12 (paged default ON, MLLM/#98 guard): a family can be marked
  // multimodal AFTER its registry paged default was computed (e.g. Qwen3.5 media,
  // Mistral-4 remapped mistral3->mistral4 text then media-detected). Dense/rotating
  // KV VL/MLLM loads must NOT default to paged (the engine excludes MLLM from the
  // generic paged default until the #98 byte-ceiling lands). Hybrid/mamba VL that
  // REQUIRE paged (Qwen3.5 linear-attn, zaya1-vl) keep their cacheType-driven paged;
  // step-3.7's typed full+sliding KV keeps its explicit paged. Only clear the
  // default-on for non-paged-required KV cache types.
  // MiniMax-M3 is NOT excluded here: the paged tier (paged_cache.py) has no M3
  // sparse-MSA handling and corrupts partial-prefix reuse (live-proven 2026-08-10),
  // so M3 flows through the normal multimodal paged-off path and relies on its
  // native MSA cache + the M3-aware prefix/block-disk L2 tier. gemma4 stays
  // excluded (its typed mixed-SWA paged lane is proven).
  // muse-glimmer is excluded for the same reason as gemma4: it rides that same
  // typed mixed-SWA paged lane, and the engine already exempts muse_glimmer in
  // _PAGED_MLLM_EXEMPT_FAMILIES (cli.py) so a bare CLI launch runs PAGED. Without
  // the exclusion here the app cleared usePagedCache and spawned the engine with
  // --no-paged-cache, so the UI ran unpaged while the CLI ran paged — observed
  // live on JANG_4M (app engine argv --no-paged-cache vs bare CLI reporting
  // paged+mixed_swa+disk+tq-native). That is the exact divergence both lists
  // exist to prevent.
  if (
    next.isMultimodal === true &&
    !next.forceTextOnly &&
    next.family !== 'gemma4' &&
    next.family !== 'muse-glimmer' &&
    (next.cacheType === 'kv' || next.cacheType === 'rotating_kv') &&
    next.cacheSubtype !== 'step3p7_full_sliding_kv'
  ) {
    next.usePagedCache = false
  }
  return next
}

function isStep37TextBridge(parsedConfig: any): boolean {
  const modelType = String(parsedConfig?.model_type ?? '').toLowerCase()
  const modelFile = String(parsedConfig?.model_file ?? '').split('/').pop()?.toLowerCase()
  const textModelType = String(parsedConfig?.text_config?.model_type ?? '').toLowerCase()
  return modelType === 'step3p7' && modelFile === 'step3p7_mlx.py' && textModelType === 'step3p5'
}

function configToDetected(family: string, config: Omit<ModelConfig, 'pattern' | 'familyName'>): DetectedConfig {
  return {
    family: family,
    toolParser: config.toolParser,
    reasoningParser: config.reasoningParser,
    supportsThinking: config.supportsThinking,
    supportsInstructMode: config.supportsInstructMode,
    honorsEnableThinking: config.honorsEnableThinking,
    supportedReasoningEfforts: config.supportedReasoningEfforts,
    defaultReasoningEffort: config.defaultReasoningEffort,
    supportsThinkingBudget: config.supportsThinkingBudget,
    thinkInTemplate: config.thinkInTemplate,
    defaultEnableThinking: config.defaultEnableThinking,
    cacheType: config.cacheType,
    cacheSubtype: config.cacheSubtype,
    architectureHints: config.architectureHints,
    // In-RAM paged cache is OFF for EVERY family; SSD block-disk L2 is the only
    // cache tier. This used to default undeclared TEXT families to ON, which
    // meant a family could acquire a RAM tier merely by not declaring one.
    // There is no per-family paged capability any more: the answer is false.
    usePagedCache: config.usePagedCache ?? false,
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
  const next = readJangChatMetadata(detected, jangCfg)
  if (
    next.family === 'laguna' &&
    jangCfg?.turboquant &&
    typeof jangCfg.turboquant === 'object' &&
    jangCfg.turboquant.enabled === false
  ) {
    next.architectureHints = {
      ...(next.architectureHints ?? {}),
      loaderTurboQuantEnabled: false,
    }
  }
  if (
    next.family === 'deepseek-v4' &&
    typeof jangCfg?.cache?.pool_quant_default === 'boolean'
  ) {
    next.dsv4PoolQuantDefault = jangCfg.cache.pool_quant_default
  }
  const zayaTypedCca = next.family === 'zaya' || next.family === 'zaya1-vl'
  const quantizationLabel = formatJangQuantizationLabel(jangCfg ?? {})
  if (quantizationLabel) {
    next.quantizationLabel = quantizationLabel
  }
  if (jangCfg?.weight_format === 'mxtq' || jangCfg?.format === 'mxtq') {
    next.isTurboQuant = true
  }
  if (!caps || typeof caps !== 'object') return next

  const runtimeModalities = Array.isArray(caps.modalities)
    ? caps.modalities.map((item: any) => String(item || '').toLowerCase()).filter(Boolean)
    : []
  const unwiredModalities = Array.isArray(caps.unwired_modalities)
    ? caps.unwired_modalities.map((item: any) => String(item || '').toLowerCase()).filter(Boolean)
    : []
  const capsRuntimeHasMedia = runtimeModalities.some((item: string) =>
    item === 'vision' || item === 'image' || item === 'video' || item === 'audio' || item === 'omni',
  )
  const capsRuntimeTextOnly = runtimeModalities.length > 0 && !capsRuntimeHasMedia
  if (runtimeModalities.length > 0) next.runtimeModalities = runtimeModalities
  const capsHasUnwiredMedia = unwiredModalities.some((item: string) =>
    item === 'vision' || item === 'image' || item === 'video' || item === 'audio' || item === 'omni',
  )

  if (next.family === 'mimo_v2') {
    next.toolParser = 'xml_function'
    next.enableAutoToolChoice = caps.supports_tools !== false
    if (capsRuntimeTextOnly || capsHasUnwiredMedia) {
      next.isMultimodal = false
      next.forceTextOnly = true
    }
  } else if (next.family === 'openpangu_v2') {
    // The converter stamps tool_parser="qwen", but openPangu emits a JSON
    // LIST inside <|tool_call_start|>/<|tool_call_end|> (token ids
    // 148903/148904) which the qwen parser never matches (live-proven
    // tool_calls=None on JANG_2L). The panel passes --tool-call-parser
    // explicitly, so the stale stamp must be neutralized here too — mirror
    // of the engine-side model_config_registry exception.
    next.toolParser = 'openpangu'
    next.enableAutoToolChoice = caps.supports_tools !== false
  } else if (next.family === 'qwen4-exp' && caps.tool_parser === 'hermes') {
    // Qwen4Exp ships the Qwen function/parameter template
    // (<function=name><parameter=arg>), but older converter stamps wrote
    // tool_parser="hermes", which only parses JSON bodies. Live
    // required-mode proof on Flash-Next JANG_1L: sampled turns following
    // the template's XML dialect failed closed with tool_calls_required
    // (3/14 at temp 0.7). The engine "qwen" parser accepts both the hermes
    // JSON body and every XML variant. The panel passes --tool-call-parser
    // explicitly, so the stale stamp must be neutralized here too — mirror
    // of the engine-side model_config_registry exception.
    next.toolParser = 'qwen'
    if (caps.supports_tools !== false) {
      next.enableAutoToolChoice = true
    }
  } else if (typeof caps.tool_parser === 'string') {
    next.toolParser = caps.tool_parser === 'none' ? undefined : caps.tool_parser
    if (next.toolParser && caps.supports_tools !== false) {
      next.enableAutoToolChoice = true
    }
  }

  // MiniMax-M3 (config.json model_type=minimax_m3_vl) is registered multimodal. The
  // mlx_vlm VL wrapper (mlx_vlm.models.minimax_m3_vl) is still unpublished, so loading
  // it as a VLM (--is-mllm) would crash at startup (ModuleNotFoundError). BUT the engine
  // now wires M3 vision end-to-end through the TEXT runtime: with env VMLX_M3_VL=1,
  // is_mllm_model() returns False and M3 runs text-routed through SingleBatchGenerator,
  // which preprocesses pixel_values when the server is NOT --text-only. So for M3 we emit
  // NEITHER --is-mllm NOR --text-only and set VMLX_M3_VL=1 in the engine child env. The
  // m3VlRoute flag drives that wiring in sessions.ts buildArgs + spawnEnv. isMultimodal
  // stays true so detection still knows it's a VL bundle.
  if (next.family === 'minimax_m3') {
    next.m3VlRoute = true
  }
  if (next.family === 'zaya') {
    next.reasoningParser = 'qwen3'
    next.supportsThinking = true
    next.thinkInTemplate = false
    next.defaultEnableThinking = true
  } else if (next.family === 'zaya1-vl') {
    next.reasoningParser = undefined
    next.supportsThinking = false
    next.thinkInTemplate = false
    next.defaultEnableThinking = false
  } else if (next.family === 'hy3') {
    next.reasoningParser = 'qwen3'
    next.supportsThinking = true
    next.thinkInTemplate = false
  } else if (next.family === 'minimax') {
    next.reasoningParser = 'minimax_m2'
  } else if (next.family === 'mimo_v2') {
    next.reasoningParser = 'think_xml'
    next.supportsThinking = true
    next.thinkInTemplate = false
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
    next.family !== 'mimo_v2' &&
    next.family !== 'ling'
  ) {
    if (typeof caps.supports_thinking === 'boolean') {
      next.supportsThinking = caps.supports_thinking
    }
    if (typeof caps.think_in_template === 'boolean' && next.supportsThinking !== false) {
      next.thinkInTemplate = caps.think_in_template
    }
  }
  const stampedDefaultEnableThinking = readJangDefaultEnableThinking(jangCfg)
  if (
    typeof stampedDefaultEnableThinking === 'boolean' &&
    next.family !== 'zaya' &&
    next.family !== 'zaya1-vl' &&
    next.family !== 'hy3' &&
    next.family !== 'mimo_v2' &&
    next.family !== 'ling' &&
    next.supportsThinking !== false
  ) {
    // Mirror vmlx_engine.model_config_registry._jang_stamp_default_enable_thinking:
    // artifact-scoped JANG chat metadata overrides coarse family fallbacks. This
    // is required for mixed artifacts in the same family, e.g. Laguna XS.2
    // default-off versus Laguna S-2.1 default-on.
    next.defaultEnableThinking = stampedDefaultEnableThinking
  }
  // openPangu-2.0-Flash: the converter stamps the coarse cache_type="hybrid",
  // which would misroute the conv-state + mixed DSA/SWA composite cache into
  // the SSM hybrid handling and force paged ON. The registry contract
  // (kv + openpangu_v2_composite, paged OFF) wins — mirror of the engine-side
  // stamp neutralization in vmlx_engine/model_config_registry (d1a588487).
  const stampCacheNeutralized = next.family === 'openpangu_v2'
  if (!stampCacheNeutralized && typeof caps.cache_type === 'string') {
    const cacheType = caps.cache_type
    if (cacheType === 'kv' || cacheType === 'mamba' || cacheType === 'hybrid' || cacheType === 'rotating_kv') {
      next.cacheType = cacheType
      if (cacheType === 'mamba' || cacheType === 'hybrid') {
        next.usePagedCache = false
      }
    }
  }
  if (!stampCacheNeutralized && typeof caps.cache_subtype === 'string' && caps.cache_subtype.length > 0) {
    next.cacheSubtype = caps.cache_subtype
  }
  if (zayaTypedCca) {
    next.usePagedCache = false
  }
  // Explicit native dialect supersedes this artifact's coarse Llama stamp.
  // Keep Auto unspecified: the template distinguishes omitted/true/false.
  if (['llama', 'llama3'].includes(next.family) && jangCfg?.tool_calling?.dialect === 'minicpm5_xml_function') {
    next.toolParser = 'minicpm5'
    next.reasoningParser = 'qwen3'
    next.supportsThinking = true
    next.honorsEnableThinking = true
    next.thinkInTemplate = false
    next.defaultEnableThinking = undefined
    next.enableAutoToolChoice = true
  }
  return next
}

function resolveJangMultimodal(jangCfg: any, parsedConfig: any, modelPath: string): boolean {
  const hasMediaConfig = configDeclaresMedia(parsedConfig)
  const capsModalities = Array.isArray(jangCfg?.capabilities?.modalities)
    ? jangCfg.capabilities.modalities.map((item: any) => String(item || '').toLowerCase()).filter(Boolean)
    : []
  if (
    capsModalities.length > 0 &&
    !capsModalities.some((item: string) =>
      item === 'vision' || item === 'image' || item === 'video' || item === 'audio' || item === 'omni',
    )
  ) {
    return false
  }
  const modality =
    jangCfg?.capabilities?.modality ??
    jangCfg?.modality ??
    parsedConfig?._jang_modality

  if (modality === 'omni') {
    return nemotronOmniArtifactHasMedia(parsedConfig, jangCfg, modelPath)
  }

  if (parsedConfig?.model_type === 'zaya1_vl' && hasMediaConfig) {
    return true
  }

  if (isAffineJangQwenHybridVlm(parsedConfig, jangCfg)) {
    return affineJangArtifactHasVision(jangCfg, modelPath)
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
    return modality !== 'text' && modality !== 'embedding' && modality !== 'rerank'
  }
  return hasMediaConfig
}

/**
 * Detect model configuration ONLY by reading the model's config.json.
 * This is the authoritative way. We no longer guess based on folder name/regex.
 * Also reads max_position_embeddings for context length detection.
 */

/** Does THIS BUNDLE's chat template read `enable_thinking`?
 *
 * Per-bundle, because the answer is not uniform within a family: measured on
 * disk, MiniMax-M2.7-JANG_K-CRACK READS the kwarg while
 * MiniMax-M2.7-Small-JANGTQ does not. A family-level flag is wrong for one of
 * them either way, so the bundle's own template is the oracle and it overrides
 * the registry default.
 *
 * Returns undefined when there is no template to read — unknown must not be
 * treated as "does not honour it", or a control would be hidden on missing
 * data rather than on evidence.
 */
function bundleReadsEnableThinking(modelPath: string): boolean | undefined {
  let sawTemplate = false
  for (const name of ['chat_template.jinja', 'tokenizer_config.json']) {
    const file = join(modelPath, name)
    if (!existsSync(file)) continue
    try {
      const text = readFileSync(file, 'utf-8')
      sawTemplate = true
      if (text.includes('enable_thinking')) return true
    } catch {
      // unreadable file tells us nothing; fall through to "unknown"
    }
  }
  return sawTemplate ? false : undefined
}

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

      let familyName: string | undefined = modelType ? MODEL_TYPE_TO_FAMILY[modelType] : undefined
      // JANG Tier-1 fallback: when config.json model_type is not a recognized
      // panel family, the JANG capabilities stamp carries the engine's resolved
      // family (same oracle the engine uses). Engine family names overlap the
      // model_type map keys (e.g. deepseek_v4, nemotron_h, step3p7), so map it the
      // same way; otherwise accept it directly if it is a registered panel family.
      if (!familyName) {
        const jangCapPath = join(modelPath, 'jang_config.json')
        if (existsSync(jangCapPath)) {
          try {
            const capFamily = String(
              JSON.parse(readFileSync(jangCapPath, 'utf-8'))?.capabilities?.family ?? '',
            ).toLowerCase()
            if (capFamily) {
              familyName = MODEL_TYPE_TO_FAMILY[capFamily] ?? (CONFIG_BY_FAMILY.has(capFamily) ? capFamily : undefined)
            }
          } catch {}
        }
      }
      if (familyName) {

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
            // The BUNDLE's own template outranks the registry default for the
            // thinking-toggle capability: the answer is not uniform within a
            // family. Measured on disk, MiniMax-M2.7-JANG_K-CRACK READS
            // enable_thinking while MiniMax-M2.7-Small-JANGTQ does not, so a
            // family-level flag is wrong for one of them either way. Unknown
            // (no template on disk) keeps the registry value — a control must
            // be hidden on evidence, never on missing data.
            {
              const readsIt = bundleReadsEnableThinking(modelPath)
              if (readsIt !== undefined) detected.honorsEnableThinking = readsIt
            }
            if (configMarksTurboQuant(parsed)) {
              detected.isTurboQuant = true
            }
            // JANG model detection: prefer the sidecar, then the exact
            // embedded stamp. Qwen4Exp JANG bundles intentionally embed their
            // full contract in config.json so copying the 51B PLE table never
            // depends on a separately drifting metadata file.
            const jangConfigPath = join(modelPath, 'jang_config.json')
            let parsedJangConfig: any
            if (existsSync(jangConfigPath)) {
            try {
              const jangCfg = JSON.parse(readFileSync(jangConfigPath, 'utf-8'))
              parsedJangConfig = jangCfg
              detected = applyJangCapabilities(detected, jangCfg)
              const nativeMtpVlReady = qwenNativeMtpVlArtifactReady(parsed, jangCfg, modelPath)
              if (
                isAffineJangQwenHybridVlm(parsed, jangCfg) &&
                !nativeMtpVlReady &&
                !affineJangArtifactHasVision(jangCfg, modelPath)
              ) {
                detected.forceTextOnly = true
              }
              if (isStep37TextBridge(parsed)) {
                delete detected.forceTextOnly
                detected.cacheSubtype = 'step3p7_full_sliding_kv'
                detected.usePagedCache = false
                detected.architectureHints = {
                  ...(detected.architectureHints ?? {}),
                  runtimeScope: 'source_vlm_needs_live_proof',
                  vlRuntimeAvailable: true,
                  textBridgeRuntimeScope: 'text_bridge_ignored_for_source_vlm',
                  slidingWindow: 512,
                }
              }
              detected.isMultimodal = resolveJangMultimodal(jangCfg, parsed, modelPath)
            } catch {
              if ('vision_config' in parsed) {
                detected.isMultimodal = true
              }
            }
          } else if (parsed?.jang_config && typeof parsed.jang_config === 'object') {
            parsedJangConfig = parsed.jang_config
            detected = applyJangCapabilities(detected, parsedJangConfig)
            detected.isMultimodal = resolveJangMultimodal(parsedJangConfig, parsed, modelPath)
          } else if (parsed?.jang && typeof parsed.jang === 'object') {
            parsedJangConfig = parsed.jang
            detected = applyJangCapabilities(detected, parsedJangConfig)
            detected.isMultimodal = resolveJangMultimodal(parsedJangConfig, parsed, modelPath)
          } else if (configDeclaresMedia(parsed)) {
            detected.isMultimodal = true
          }
          // Native MTP is an indexed artifact capability, not a JANG-sidecar
          // capability. Base MLX Qwen4Exp bundles carry their MTP declaration
          // in config.text_config and their tensors in the safetensors index.
          // Keeping this call inside the jang_config branch made the engine run
          // MTP while Electron rendered no control for the exact same bundle.
          const nativeMtp = detectNativeMtpCapability(parsed, parsedJangConfig, modelPath)
          if (nativeMtp) {
            detected.nativeMtp = nativeMtp
          }
          detected = applyLagunaVariantHint(detected, parsed, parsedJangConfig)
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
