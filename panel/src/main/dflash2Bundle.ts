import { existsSync, readdirSync, readFileSync } from 'fs'
import { dirname, join } from 'path'
import { configDeclaresDflash2, configDeclaresMimoDflash, looksLikeDflash2Drafter } from '../shared/dflash2Drafter'

function modelTypeOf(dir: string): string {
  try {
    return String(JSON.parse(readFileSync(join(dir, 'config.json'), 'utf8')).model_type || '')
  } catch {
    return ''
  }
}

/** A drafter dir the engine runs on the DFlash2 lane: a DFlash2 drafter, or MiMo's DFlash inside a mimo_v2 bundle. */
function drafterConfigOnDflash2Lane(cfg: unknown, targetDir: string): boolean {
  return configDeclaresDflash2(cfg) || configDeclaresMimoDflash(cfg, modelTypeOf(targetDir))
}

/**
 * A DFlash2 drafter shipped INSIDE the model bundle: any immediate subfolder
 * whose config.json declares DFlash2DraftModel (convention: `<bundle>/dflash2`).
 * Bundles without one return null and launch exactly as before.
 */
export function findBundledDflash2Drafter(modelPath: string | undefined | null): string | null {
  const root = String(modelPath || '').trim()
  if (!root) return null
  let entries: string[] = []
  try {
    entries = readdirSync(root)
  } catch {
    return null
  }
  const rank = (name: string) => (name === 'dflash2' ? 0 : name === 'dflash' ? 1 : 2)
  const ordered = [...entries].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
  for (const entry of ordered) {
    if (entry.startsWith('.')) continue
    const cfg = join(root, entry, 'config.json')
    try {
      if (existsSync(cfg) && drafterConfigOnDflash2Lane(JSON.parse(readFileSync(cfg, 'utf8')), root)) {
        return join(root, entry)
      }
    } catch {
      /* not a drafter */
    }
  }
  return null
}

/** DFlash2 drafter by its config.json architecture (local path), else by name. */
export function isDflash2DrafterSelection(path: string | undefined | null): boolean {
  const value = String(path || '').trim()
  if (!value) return false
  try {
    const cfg = join(value, 'config.json')
    if (existsSync(cfg)) return drafterConfigOnDflash2Lane(JSON.parse(readFileSync(cfg, 'utf8')), dirname(value))
  } catch {
    /* fall back to the name test */
  }
  return looksLikeDflash2Drafter(value)
}
