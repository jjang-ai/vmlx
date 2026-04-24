/**
 * i18n consistency regression guard.
 *
 * Locks invariants across every locale file:
 *   1. Valid JSON.
 *   2. Identical key tree — every key present in en.json exists in every
 *      other locale (missing keys fall back to en at runtime, but the
 *      contract is that translations are complete).
 *   3. Interpolation placeholders ({n}, {{count}}, {name}, etc.) are
 *      preserved in translations — missing a placeholder would produce a
 *      silently-wrong UI string.
 *   4. No empty / whitespace-only string values.
 *
 * This test must pass before any release. If you add a new key to en.json,
 * add the translation to every locale file or this test fails.
 */

import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import { resolve } from 'path'

const LOCALES = ['en', 'zh', 'ko', 'ja', 'es'] as const
type LocaleKey = (typeof LOCALES)[number]
const LOCALE_DIR = resolve(
  __dirname,
  '..',
  'src',
  'renderer',
  'src',
  'i18n',
  'locales',
)

function loadLocale(locale: LocaleKey): Record<string, any> {
  const path = resolve(LOCALE_DIR, `${locale}.json`)
  const raw = readFileSync(path, 'utf-8')
  return JSON.parse(raw)
}

function flatten(obj: Record<string, any>, prefix = ''): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${k}` : k
    if (v && typeof v === 'object' && !Array.isArray(v)) {
      Object.assign(out, flatten(v, path))
    } else if (typeof v === 'string') {
      out[path] = v
    }
  }
  return out
}

function extractPlaceholders(value: string): Set<string> {
  // Match both {var} and {{var}} interpolation forms. Returns bare names.
  const out = new Set<string>()
  const matches = value.matchAll(/\{\{?(\w+)\}?\}/g)
  for (const m of matches) {
    out.add(m[1])
  }
  return out
}

describe('i18n locale consistency', () => {
  const locales = Object.fromEntries(
    LOCALES.map((l) => [l, loadLocale(l)]),
  ) as Record<LocaleKey, Record<string, any>>
  const flat = Object.fromEntries(
    LOCALES.map((l) => [l, flatten(locales[l])]),
  ) as Record<LocaleKey, Record<string, string>>

  it('every locale parses as valid JSON', () => {
    for (const l of LOCALES) {
      expect(typeof locales[l]).toBe('object')
      expect(locales[l]).not.toBeNull()
    }
  })

  it('every locale has the same key set as en', () => {
    const enKeys = Object.keys(flat.en).sort()
    for (const l of LOCALES) {
      if (l === 'en') continue
      const lKeys = Object.keys(flat[l]).sort()
      const missing = enKeys.filter((k) => !(k in flat[l]))
      const extra = lKeys.filter((k) => !(k in flat.en))
      expect(
        missing,
        `Locale '${l}' is MISSING these keys present in en.json:\n  ${missing.join('\n  ')}`,
      ).toEqual([])
      expect(
        extra,
        `Locale '${l}' has EXTRA keys not present in en.json (add to en first):\n  ${extra.join('\n  ')}`,
      ).toEqual([])
    }
  })

  it('all interpolation placeholders are preserved in every translation', () => {
    for (const key of Object.keys(flat.en)) {
      const enPlaceholders = extractPlaceholders(flat.en[key])
      if (enPlaceholders.size === 0) continue
      for (const l of LOCALES) {
        if (l === 'en') continue
        const translated = flat[l][key]
        if (!translated) continue
        const translatedPlaceholders = extractPlaceholders(translated)
        const missing = [...enPlaceholders].filter(
          (p) => !translatedPlaceholders.has(p),
        )
        expect(
          missing,
          `Locale '${l}' lost placeholder(s) {${missing.join(',')}} on key '${key}'. en: "${flat.en[key]}" | ${l}: "${translated}"`,
        ).toEqual([])
      }
    }
  })

  it('no string value is empty or whitespace-only', () => {
    for (const l of LOCALES) {
      for (const [key, val] of Object.entries(flat[l])) {
        expect(
          val.trim().length,
          `Locale '${l}' has empty / whitespace-only value at key '${key}'`,
        ).toBeGreaterThan(0)
      }
    }
  })

  it('LOCALES array in i18n/index.tsx stays in sync with locale files', () => {
    // If you add a new JSON locale file, also add it to the Locale type
    // and LOCALES / LOCALE_NAMES / LOCALE_FLAGS in i18n/index.tsx.
    const indexSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'i18n', 'index.tsx'),
      'utf-8',
    )
    for (const l of LOCALES) {
      expect(
        indexSrc.includes(`'${l}'`) || indexSrc.includes(`"${l}"`),
        `Locale '${l}' JSON exists but is not registered in i18n/index.tsx`,
      ).toBe(true)
    }
  })

  it('language switcher persists to localStorage key vmlx-locale', () => {
    // Verifies the persistence contract both language pickers depend on.
    const indexSrc = readFileSync(
      resolve(__dirname, '..', 'src', 'renderer', 'src', 'i18n', 'index.tsx'),
      'utf-8',
    )
    expect(indexSrc).toContain("'vmlx-locale'")
    expect(indexSrc).toContain('localStorage.setItem')
    expect(indexSrc).toContain('localStorage.getItem')
  })
})
