/**
 * Main-process i18n bridge.
 *
 * The renderer owns its own i18n via React Context + localStorage
 * (`src/renderer/src/i18n/index.tsx`). This module mirrors that system
 * for the Electron main process so the tray menu, native dialogs,
 * session load-progress labels, and IPC error toasts all honor the
 * user's picked locale.
 *
 * Persistence: the renderer writes `vmlx-locale` into its localStorage
 * on every picker click. On startup the renderer tells the main process
 * its current locale via `i18n:set-locale`. The main process caches it
 * in-memory and in the SQLite settings table for the next restart (so
 * tray labels match the picked locale even before the renderer attaches).
 *
 * The locale JSON files are imported statically — vite inlines them
 * into the main-process bundle at build time. No filesystem reads needed,
 * which means this works identically in dev and inside the packaged asar.
 *
 * Fallback chain: t(key) = locales[locale][key] ?? locales.en[key] ?? key
 */

// Static imports — vite bundles each JSON into the main chunk at build time.
// Both main and renderer import from the same canonical source-of-truth.
import en from '../renderer/src/i18n/locales/en.json'
import zh from '../renderer/src/i18n/locales/zh.json'
import ko from '../renderer/src/i18n/locales/ko.json'
import ja from '../renderer/src/i18n/locales/ja.json'
import es from '../renderer/src/i18n/locales/es.json'

export type Locale = 'en' | 'zh' | 'ko' | 'ja' | 'es'
export const SUPPORTED_LOCALES: Locale[] = ['en', 'zh', 'ko', 'ja', 'es']

type LocaleTree = Record<string, any>

const locales: Record<Locale, LocaleTree> = { en, zh, ko, ja, es }
let currentLocale: Locale = 'en'

/**
 * No-op kept for API compatibility — the renderer imports the same files
 * so there is nothing to load at runtime. Left as a named export so the
 * bootstrap site in `main/index.ts` does not need a guard.
 */
export function loadLocales(): void {
  // Intentional no-op: statically imported above.
}

/** Override the active locale (called from renderer → main via IPC). */
export function setLocale(loc: string): void {
  if ((SUPPORTED_LOCALES as string[]).includes(loc)) {
    currentLocale = loc as Locale
  }
}

export function getLocale(): Locale {
  return currentLocale
}

function lookup(tree: LocaleTree | undefined, path: string): string | undefined {
  if (!tree) return undefined
  const parts = path.split('.')
  let cur: any = tree
  for (const p of parts) {
    if (cur == null || typeof cur !== 'object') return undefined
    cur = cur[p]
  }
  return typeof cur === 'string' ? cur : undefined
}

function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) return template
  return template.replace(/\{(\w+)\}|\{\{(\w+)\}\}/g, (_, k1, k2) => {
    const key = k1 || k2
    return String(params[key] ?? `{${key}}`)
  })
}

/**
 * Translate a dot-path key. Mirrors the renderer's `t()` fallback chain.
 */
export function t(key: string, params?: Record<string, string | number>): string {
  const val =
    lookup(locales[currentLocale], key) ?? lookup(locales.en, key) ?? key
  return interpolate(val, params)
}
