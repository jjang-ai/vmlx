import { readFileSync } from 'fs'
import { join, resolve } from 'path'
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
// drafts on the multimodal single-request lane.
const DSPARK_SINGLE_REQUEST_MODEL_TYPES = new Set(['minicpmv4_7'])
const DSPARK_TEXT_LANE_MODEL_TYPES = new Set(['minimax_m3', 'minimax_m3_vl'])

function bundleModelType(root: string): string | null {
  try {
    return String(JSON.parse(readFileSync(join(root, 'config.json'), 'utf8')).model_type || '').toLowerCase()
  } catch {
    return null
  }
}

/**
 * True when the bundle's DSpark drafter runs on the engine's text lane (MiniMax-M3 / M3.1): the session keeps its
 * batching and cache settings and only `--dspark on|off|auto` follows the DSpark choice.
 */
export function dsparkUsesTextLane(modelPath: string | undefined | null): boolean {
  const root = String(modelPath || '').trim()
  return !!root && DSPARK_TEXT_LANE_MODEL_TYPES.has(bundleModelType(root) || '')
}

/**
 * Follow vmlx_engine/dspark/runtime.load_dspark_runtime: a target family with a DSpark runtime (MiniCPM-V-4.7,
 * MiniMax-M3 / M3.1), using the declared drafter_path or <bundle>/dspark. Finding a different sidecar must not
 * select a serving lane the engine will not actually use.
 */
export function findBundledDsparkDrafter(modelPath: string | undefined | null): string | null {
  const root = String(modelPath || '').trim()
  if (!root) return null
  const modelType = bundleModelType(root)
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
