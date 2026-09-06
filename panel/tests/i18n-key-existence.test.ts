import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'fs'
import { join } from 'path'

/**
 * Every literal translation key used in source must exist in the English
 * catalog. translate.ts falls back to the raw key at runtime, so a typo or a
 * key added to one place but not the catalog silently ships as "sessions.x.y".
 */
function walk(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name)
    if (statSync(p).isDirectory()) walk(p, out)
    else if (/\.(tsx?|cjs|mjs)$/.test(name) && !/\.test\./.test(name)) out.push(p)
  }
  return out
}
function flatten(obj: any, prefix = '', out: Set<string> = new Set()): Set<string> {
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === 'object') flatten(v, key, out)
    else out.add(key)
  }
  return out
}

const root = join(__dirname, '..', 'src')
const en = flatten(JSON.parse(readFileSync(join(root, 'renderer', 'src', 'i18n', 'locales', 'en.json'), 'utf8')))
// t('a.b.c') / t("a.b.c") / t(`a.b.c`) with a literal (no interpolation) first argument
const CALL = /\bt\(\s*(['"`])([A-Za-z0-9_.-]+)\1/g

describe('i18n keys used in source exist in en.json', () => {
  const files = walk(join(root, 'renderer', 'src')).concat(walk(join(root, 'main')))
  const missing: string[] = []
  let used = 0
  for (const f of files) {
    const src = readFileSync(f, 'utf8')
    for (const m of src.matchAll(CALL)) {
      used++
      if (!en.has(m[2])) missing.push(`${f.replace(root, 'src')}: ${m[2]}`)
    }
  }
  it('finds a meaningful number of literal keys', () => {
    expect(used).toBeGreaterThan(500)
  })
  it('has no missing keys', () => {
    expect(missing, missing.slice(0, 40).join('\n')).toEqual([])
  })
})
