import React, { createContext, useContext, useState, useCallback } from 'react'
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

function get(obj: Record<string, any>, path: string): string | undefined {
  const keys = path.split('.')
  let current: any = obj
  for (const k of keys) {
    if (current == null || typeof current !== 'object') return undefined
    current = current[k]
  }
  return typeof current === 'string' ? current : undefined
}

function interpolate(template: string, params?: Record<string, string | number>): string {
  if (!params) return template
  return template.replace(/\{(\w+)\}|\{\{(\w+)\}\}/g, (_, k1, k2) => String(params[k1 || k2] ?? `{${k1 || k2}}`))
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    const saved = localStorage.getItem('vmlx-locale')
    return (saved && saved in locales) ? saved as Locale : 'en'
  })

  const setLocale = useCallback((l: Locale) => {
    setLocaleState(l)
    localStorage.setItem('vmlx-locale', l)
  }, [])

  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    const val = get(locales[locale], key) ?? get(locales.en, key) ?? key
    return interpolate(val, params)
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
