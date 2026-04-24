import React, { createContext, useContext, useState, useCallback, useEffect } from 'react'
import en from './locales/en.json'
import zh from './locales/zh.json'
import ko from './locales/ko.json'
import ja from './locales/ja.json'
import es from './locales/es.json'

export type Locale = 'en' | 'zh' | 'ko' | 'ja' | 'es'

export const LOCALE_NAMES: Record<Locale, string> = {
  en: 'English',
  zh: '中文',
  ko: '한국어',
  ja: '日本語',
  es: 'Español',
}

export const LOCALE_FLAGS: Record<Locale, string> = {
  en: '🇺🇸',
  zh: '🇨🇳',
  ko: '🇰🇷',
  ja: '🇯🇵',
  es: '🇪🇸',
}

const locales: Record<Locale, Record<string, any>> = { en, zh, ko, ja, es }

interface I18nContextType {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: (key: string, params?: Record<string, string | number>) => string
}

const I18nContext = createContext<I18nContextType>({
  locale: 'en',
  setLocale: () => {},
  t: (key) => key,
})

function get(obj: Record<string, any> | undefined, path: string): string | undefined {
  if (!obj || typeof path !== 'string') return undefined
  try {
    const keys = path.split('.')
    let current: any = obj
    for (const k of keys) {
      if (current == null || typeof current !== 'object') return undefined
      current = current[k]
    }
    return typeof current === 'string' ? current : undefined
  } catch {
    return undefined
  }
}

function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) return template
  try {
    return template.replace(/\{(\w+)\}|\{\{(\w+)\}\}/g, (_, k1, k2) => String(params[k1 || k2] ?? `{${k1 || k2}}`))
  } catch {
    return template
  }
}

// localStorage key — namespaced under the vMLX prefix to not collide with
// any other stored setting. If future code ever changes this, also update
// main/i18n.ts (which hard-codes the same name in the persistence DB row)
// and panel/tests/i18n-consistency.test.ts (which asserts the contract).
const LOCALE_STORAGE_KEY = 'vmlx-locale'

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    try {
      const saved = localStorage.getItem(LOCALE_STORAGE_KEY)
      return (saved && saved in locales) ? saved as Locale : 'en'
    } catch {
      return 'en'
    }
  })

  // Mirror the initial locale to the main process once on mount so the
  // tray menu, system dialogs, and load-progress labels match what the
  // renderer shows — even before the user clicks the picker.
  useEffect(() => {
    try {
      const api = (window as any).api
      if (api?.i18n?.setLocale) api.i18n.setLocale(locale)
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l)
    try { localStorage.setItem(LOCALE_STORAGE_KEY, l) } catch {}
    // Push to main so tray/menu/dialogs update live alongside the React tree.
    try {
      const api = (window as any).api
      if (api?.i18n?.setLocale) api.i18n.setLocale(l)
    } catch {}
  }, [])

  // `t` must NEVER throw — a thrown error inside a render function would
  // unmount the whole React tree (white screen). All paths wrap in try/catch
  // and fall back to the raw key as a last resort.
  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    try {
      if (typeof key !== 'string') return String(key ?? '')
      const val = get(locales[locale], key) ?? get(locales.en, key) ?? key
      return interpolate(val, params)
    } catch {
      return key
    }
  }, [locale])

  return (
    <I18nContext.Provider value={{ locale, setLocale, t }}>
      {children}
    </I18nContext.Provider>
  )
}

export function useTranslation() {
  return useContext(I18nContext)
}
