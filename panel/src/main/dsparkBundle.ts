import { existsSync, readFileSync } from 'fs'
import { dirname, join, resolve } from 'path'
import { bundledDraftFits } from './bundledDraft'

/**
 * True when a drafter config.json declares DSpark: openbmb's MiniCPM-V-4.7 drafters (`Qwen3DSparkModel`,
 * `dflash_config.projector_type: "dspark"`), SpecForge's Naive N0.5 drafter (`DSparkDraftModel`) and MiniMax's
 * M3.1 drafter (`DSparkMiniMaxDraftModel`).
 */
export function configDeclaresDspark(cfg: any): boolean {
  if (!cfg || typeof cfg !== 'object') return false
  const archs: string[] = Array.isArray(cfg.architectures) ? cfg.architectures.map(String) : []
  if (archs.some(a => /dspark/i.test(a))) return true
  const projector = cfg.dflash_config?.projector_type ?? cfg.projector_type
  return String(projector || '').toLowerCase() === 'dspark'
}

/**
 * DSpark default for a bundle: ON whenever the bundle ships a drafter (Eric 2026-10-13: "check model files if it has
 * or not ... enabled if it has it and if not don't use it"). The jang_config `speculative.default` stamp is
 * informational only; the engine applies the same rule (`--dspark auto`, models/mllm._load_dspark_runtime).
 */
export function bundledDsparkDefault(modelPath: string | undefined | null): boolean {
  return findBundledDsparkDrafter(modelPath) !== null
}

// Target families with a DSpark runtime (vmlx_engine/dspark/runtime.py TARGETS). MiniMax-M3 / M3.1 draft on the text
// lane (TEXT_LANE_MODEL_TYPES): continuous batching and the normal cache stack stay as configured. MiniCPM-V-4.7
// drafts on the multimodal single-request lane. DeepSeek-V4.1 attaches its bundled dSpark draft in its own loader
// (loaders/load_jangtq_dsv41._maybe_load_dsv41_dspark) and drafts on the DSV4 single-sequence lane, which already runs
// --max-num-seqs 1 with continuous batching and its own cache stack: like the text lane, the choice only selects
// --dspark.
const DSPARK_SINGLE_REQUEST_MODEL_TYPES = new Set(['minicpmv4_7'])
const DSPARK_TEXT_LANE_MODEL_TYPES = new Set(['minimax_m3', 'minimax_m3_vl'])
const DSPARK_DSV41_MODEL_TYPES = new Set(['deepseek_v41'])

function bundleConfig(root: string): any | null {
  try {
    const cfg = JSON.parse(readFileSync(join(root, 'config.json'), 'utf8'))
    return cfg && typeof cfg === 'object' ? cfg : null
  } catch {
    return null
  }
}

function bundleModelType(root: string): string | null {
  const cfg = bundleConfig(root)
  return cfg ? String(cfg.model_type || '').toLowerCase() : null
}

/**
 * Follow the engine's DeepSeek-V4.1 draft admission (loaders/load_jangtq_dsv41._maybe_load_dsv41_dspark, then
 * jang_tools.dsv41.dspark.load_dspark): config.json `runtime.bundle_has_mtp` declares the draft, which is the
 * prestacked `dspark/dspark.safetensors` (or `dspark.safetensors` at the bundle root); without that file, a bundle
 * that is not JANGHT (no mxfp8 / jangt quantization entries, model.BundleSpecs.wants) reads the release `mtp.*`
 * tensors from its own shards. Returns the directory holding the draft, or null (the engine serves AR).
 */
function findDsv41Draft(root: string, cfg: any): string | null {
  if (!cfg?.runtime || typeof cfg.runtime !== 'object' || !cfg.runtime.bundle_has_mtp) return null
  for (const file of [join(root, 'dspark', 'dspark.safetensors'), join(root, 'dspark.safetensors')]) {
    if (existsSync(file)) return dirname(file)
  }
  const quant = cfg.quantization
  const janght = !!quant && typeof quant === 'object' &&
    Object.values(quant).some((v: any) => v && typeof v === 'object' && ['mxfp8', 'jangt'].includes(v.mode))
  if (janght) return null
  try {
    const map = JSON.parse(readFileSync(join(root, 'model.safetensors.index.json'), 'utf8')).weight_map || {}
    return Object.keys(map).some(name => name.startsWith('mtp.')) ? resolve(root) : null
  } catch {
    return null
  }
}

/**
 * True when the bundle's DSpark drafter runs on the engine's text lane (MiniMax-M3 / M3.1) or DeepSeek-V4.1's
 * single-sequence lane: the session keeps its batching and cache settings and only `--dspark on|off|auto` follows
 * the DSpark choice.
 */
export function dsparkUsesTextLane(modelPath: string | undefined | null): boolean {
  const root = String(modelPath || '').trim()
  const modelType = root ? bundleModelType(root) || '' : ''
  return DSPARK_TEXT_LANE_MODEL_TYPES.has(modelType) || DSPARK_DSV41_MODEL_TYPES.has(modelType)
}

/**
 * Follow vmlx_engine/dspark/runtime.load_dspark_runtime: a target family with a DSpark runtime (MiniCPM-V-4.7,
 * MiniMax-M3 / M3.1), using the declared drafter_path or <bundle>/dspark; DeepSeek-V4.1 follows its own loader
 * (findDsv41Draft). Finding a different sidecar must not select a serving lane the engine will not actually use.
 */
export function findBundledDsparkDrafter(modelPath: string | undefined | null): string | null {
  const root = String(modelPath || '').trim()
  if (!root) return null
  const cfg = bundleConfig(root)
  const modelType = cfg ? String(cfg.model_type || '').toLowerCase() : null
  if (modelType && DSPARK_DSV41_MODEL_TYPES.has(modelType)) return findDsv41Draft(root, cfg)
  if (!modelType || !(DSPARK_SINGLE_REQUEST_MODEL_TYPES.has(modelType) || DSPARK_TEXT_LANE_MODEL_TYPES.has(modelType))) return null
  let spec: any = {}
  try {
    spec = JSON.parse(readFileSync(join(root, 'jang_config.json'), 'utf8')).speculative || {}
  } catch {
    // Like the engine, an absent or unreadable optional stamp uses dspark/.
  }
  const method = String(spec.method || '').toLowerCase()
  if (method && method !== 'dspark') return null
  const drafter = resolve(root, String(spec.drafter_path || 'dspark'))
  try {
    if (configDeclaresDspark(JSON.parse(readFileSync(join(drafter, 'config.json'), 'utf8'))) &&
        bundledDraftFits(drafter, root)) return drafter
  } catch {
    // A bundle without its declared drafter stays on plain decoding.
  }
  return null
}
