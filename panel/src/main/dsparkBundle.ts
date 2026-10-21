import { readFileSync } from 'fs'
import { join, resolve } from 'path'
import { bundledDraftFits } from './bundledDraft'

/**
 * True when a drafter config.json declares DSpark: openbmb's MiniCPM-V-4.7 drafters (`Qwen3DSparkModel`,
 * `dflash_config.projector_type: "dspark"`) and SpecForge's Naive N0.5 drafter (`DSparkDraftModel`).
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

/**
 * Follow models/mllm.py::_load_dspark_runtime: MiniCPM-V-4.7 only, using the
 * declared drafter_path or <bundle>/dspark. Finding a different sidecar must
 * not select a serving lane the engine will not actually use.
 */
export function findBundledDsparkDrafter(modelPath: string | undefined | null): string | null {
  const root = String(modelPath || '').trim()
  if (!root) return null
  try {
    const target = JSON.parse(readFileSync(join(root, 'config.json'), 'utf8'))
    if (String(target.model_type || '').toLowerCase() !== 'minicpmv4_7') return null
  } catch {
    return null
  }
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
